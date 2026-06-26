from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .alpaca_rest import AlpacaDataClient
from .alpaca_trading import AlpacaTradingAPIError, AlpacaTradingClient
from .backtest import _apply_v25_candlestick_filter, _v27_apply_preselection_filters, _v27_select_top_n_with_caps, _v28_calculate_risk_budget
from .config import PROJECT_ROOT, AlpacaSettings, StrategyParams
from .indicators import add_daily_features, add_intraday_features, build_qqq_context, merge_market_context
from .live_store import LiveStore, utc_now_iso
from .strategy import compute_signals
from .q_learning_policy import apply_q_policy, load_q_model
from .ml_ranker_policy import load_ranker_model, score_candidates as score_ml_ranker_candidates
from .symbols import WATCHLISTS, parse_symbols

NY = ZoneInfo("America/New_York")
LIVE_DIR = PROJECT_ROOT / "data" / "live_trading"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
ORDER_PREFIXES = ("rmv33-", "rmv32-", "rm33-")



@dataclass
class LiveRiskConfig:
    account_value_fallback: float = 10_000.0
    position_sizing_mode: str = "fixed_dollar_risk"
    fixed_risk_dollars: float = 100.0
    base_risk_pct: float = 1.0
    min_risk_dollars: float = 10.0
    max_risk_dollars: float = 100.0
    dd1_risk_pct: float = 0.75
    dd2_risk_pct: float = 0.50
    pause_dd_pct: float = 15.0
    allow_fractional: bool = False


@dataclass
class LiveSettings:
    enabled: bool = False
    dry_run: bool = True
    allow_live_trading: bool = False
    feed: str = "iex"
    symbols: list[str] | None = None
    lookback_days: int = 75
    incremental_fetch_days: int = 3
    poll_seconds: int = 60
    max_daily_trades: int = 2
    max_open_positions: int = 2
    max_orders_per_symbol_per_day: int = 1
    max_daily_loss_dollars: float = 500.0
    force_market_open: bool = True
    enable_max_hold_exit: bool = True
    entry_start_time_et: str = "09:35"
    entry_end_time_et: str = "15:55"
    allow_extended_hours_entries: bool = False
    use_news_proxy: bool = True
    q_learning_filter_enabled: bool = False
    q_learning_policy_path: str = ""
    q_learning_min_edge: float = 0.0
    q_learning_min_state_count: int = 8
    ml_ranker_filter_enabled: bool = False
    ml_ranker_model_path: str = ""
    ml_ranker_min_pred_r: float = 0.05
    ml_ranker_min_win_prob: float = 0.0
    selection_mode: str = "seen_so_far_top_n"
    strategy_variant: str = "best_report_153601"
    strategy_run_mode: str = "single"
    active_strategy_count: int = 1
    active_strategy_variants: list[str] | None = None
    live_config_source: str = "env"
    log_path: Path = LIVE_DIR / "paper_trading_log.csv"
    state_path: Path = LIVE_DIR / "paper_trading_state.json"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def load_live_settings_from_env() -> tuple[LiveSettings, LiveRiskConfig]:
    preset = os.getenv("LIVE_WATCHLIST_PRESET", "v25_playbook")
    custom_symbols = os.getenv("LIVE_SYMBOLS", "").strip()
    symbols = parse_symbols(custom_symbols, preset=preset)
    if not symbols:
        symbols = WATCHLISTS.get("v25_playbook", [])
    live = LiveSettings(
        enabled=_env_bool("PAPER_TRADING_ENABLED", False),
        dry_run=_env_bool("PAPER_TRADING_DRY_RUN", True),
        allow_live_trading=_env_bool("ALLOW_LIVE_TRADING", False),
        feed=os.getenv("ALPACA_FEED", "iex"),
        symbols=symbols,
        lookback_days=max(35, _env_int("LIVE_LOOKBACK_DAYS", 75)),
        incremental_fetch_days=max(1, _env_int("LIVE_INCREMENTAL_FETCH_DAYS", 3)),
        poll_seconds=max(30, _env_int("LIVE_POLL_SECONDS", 60)),
        max_daily_trades=max(1, _env_int("LIVE_MAX_DAILY_TRADES", 2)),
        max_open_positions=max(1, _env_int("LIVE_MAX_OPEN_POSITIONS", 2)),
        max_orders_per_symbol_per_day=max(1, _env_int("LIVE_MAX_ORDERS_PER_SYMBOL_PER_DAY", 1)),
        max_daily_loss_dollars=max(0.0, _env_float("LIVE_MAX_DAILY_LOSS_DOLLARS", 500.0)),
        force_market_open=_env_bool("LIVE_REQUIRE_MARKET_OPEN", True),
        enable_max_hold_exit=_env_bool("LIVE_ENABLE_MAX_HOLD_EXIT", True),
        entry_start_time_et=os.getenv("LIVE_ENTRY_START_TIME_ET", "09:35"),
        entry_end_time_et=os.getenv("LIVE_ENTRY_END_TIME_ET", "15:55"),
        allow_extended_hours_entries=_env_bool("LIVE_ALLOW_EXTENDED_HOURS_ENTRIES", False),
        use_news_proxy=_env_bool("LIVE_USE_NEWS_PROXY", True),
        q_learning_filter_enabled=_env_bool("LIVE_Q_POLICY_ENABLED", False),
        q_learning_policy_path=os.getenv("LIVE_Q_POLICY_PATH", ""),
        q_learning_min_edge=_env_float("LIVE_Q_POLICY_MIN_EDGE", 0.0),
        q_learning_min_state_count=max(1, _env_int("LIVE_Q_POLICY_MIN_STATE_COUNT", 8)),
        ml_ranker_filter_enabled=_env_bool("LIVE_ML_RANKER_ENABLED", False),
        ml_ranker_model_path=os.getenv("LIVE_ML_RANKER_MODEL_PATH", ""),
        ml_ranker_min_pred_r=_env_float("LIVE_ML_RANKER_MIN_PRED_R", 0.05),
        ml_ranker_min_win_prob=_env_float("LIVE_ML_RANKER_MIN_WIN_PROB", 0.0),
        selection_mode=os.getenv("LIVE_SELECTION_MODE", "seen_so_far_top_n"),
        strategy_variant=os.getenv("LIVE_STRATEGY_VARIANT", "best_report_153601").strip().lower(),
        strategy_run_mode=os.getenv("LIVE_STRATEGY_RUN_MODE", "single").strip().lower(),
        active_strategy_count=1,
        active_strategy_variants=None,
        live_config_source="env",
    )
    risk = LiveRiskConfig(
        account_value_fallback=_env_float("LIVE_ACCOUNT_VALUE_FALLBACK", 10000.0),
        position_sizing_mode=os.getenv("LIVE_RISK_MODE", "fixed_dollar_risk"),
        fixed_risk_dollars=_env_float("LIVE_FIXED_RISK_DOLLARS", 100.0),
        base_risk_pct=_env_float("LIVE_BASE_RISK_PCT", 1.0),
        min_risk_dollars=_env_float("LIVE_MIN_RISK_DOLLARS", 10.0),
        max_risk_dollars=_env_float("LIVE_MAX_RISK_DOLLARS", 100.0),
        dd1_risk_pct=_env_float("LIVE_DD1_RISK_PCT", 0.75),
        dd2_risk_pct=_env_float("LIVE_DD2_RISK_PCT", 0.50),
        pause_dd_pct=_env_float("LIVE_PAUSE_DD_PCT", 15.0),
        allow_fractional=_env_bool("LIVE_ALLOW_FRACTIONAL_SHARES", False),
    )
    return live, risk


def make_best_report_153601_params(risk: LiveRiskConfig, variant: str | None = None) -> StrategyParams:
    """One paper-trading strategy preset; risk sizing remains separate.

    Set LIVE_STRATEGY_VARIANT=v358_live_raw_optimized or v359_live_hunter to use
    live-safe raw-bar quality gates in paper trading.
    """
    variant = (variant or os.getenv("LIVE_STRATEGY_VARIANT", "best_report_153601")).strip().lower()
    use_v358 = variant in {"v358_live_raw_optimized", "raw_live_optimized", "live_raw_optimized_v358"}
    use_v359 = variant in {"v359_live_hunter", "v359_professional_live_hunter", "live_hunter_v359", "pro_live_hunter"}
    use_v363 = variant in {"v363_grid_robust", "live_grid_robust_v363", "v363_robust_raw_live"}
    use_v364 = variant in {"v364_professional_momentum", "live_professional_momentum_v364", "professional_momentum_hybrid", "v364_momentum_hybrid"}
    use_v377 = variant in {"v377_positive_context", "positive_context_v377", "live_positive_context_v377", "report_positive_context"}
    use_v379 = variant in {"v379_indicator_pattern", "v379_decision_pattern", "live_indicator_pattern_v379", "decision_time_pattern_v379", "v38_active_pattern", "v38_stable_pattern", "v382_active_plus", "v382_more_trades", "v38_active_plus", "v383_adaptive", "v383_regime_adaptive", "v38_3_adaptive", "v385_adaptive_plus", "v38_5_adaptive_plus", "adaptive_plus_v385", "v384_failure_reversal", "v38_4_failure_aware"}
    use_v38_active = variant in {"v38_active_pattern", "v38_active"}
    use_v38_stable = variant in {"v38_stable_pattern", "v38_stable"}
    use_v382_active_plus = variant in {"v382_active_plus", "v38_active_plus", "active_plus_v382"}
    use_v382_more_trades = variant in {"v382_more_trades", "more_trades_v382", "v382_high_activity"}
    use_v383_regime_adaptive = variant in {"v383_regime_adaptive", "v38_3_regime_adaptive", "regime_adaptive_v383", "live_regime_adaptive_v383"}
    use_v383_adaptive = variant in {"v383_adaptive", "v383_regime_adaptive", "v38_3_adaptive", "adaptive_composite_v383"}
    use_v385_adaptive_plus = variant in {"v385_adaptive_plus", "v38_5_adaptive_plus", "live_adaptive_plus_v385", "adaptive_composite_plus_v385", "adaptive_plus_v385"}
    use_v384_failure_reversal = variant in {"v384_failure_reversal", "v38_4_failure_aware", "live_failure_reversal_v384", "failure_aware_reversal"}
    return StrategyParams(
        strategy_profile="symbol_playbook_v25",
        direction_mode="long_short",
        initial_account_value=float(risk.account_value_fallback),
        risk_per_trade_dollars=float(risk.fixed_risk_dollars),
        requested_risk_percent=float(risk.base_risk_pct),
        risk_per_trade_pct=float(risk.base_risk_pct) / 100.0,
        position_sizing_mode=str(risk.position_sizing_mode),
        compounding_base_risk_pct=float(risk.base_risk_pct),
        compounding_min_risk_dollars=float(risk.min_risk_dollars),
        compounding_max_risk_dollars=float(risk.max_risk_dollars),
        compounding_dd1_risk_pct=float(risk.dd1_risk_pct),
        compounding_dd2_risk_pct=float(risk.dd2_risk_pct),
        compounding_pause_dd_pct=float(risk.pause_dd_pct),
        max_position_notional_pct=9999.0,
        min_candidate_score=10.0 if use_v364 else (0.0 if (use_v363 or use_v377 or use_v379) else 2.0),
        max_trades_per_day=(10 if use_v385_adaptive_plus else (5 if use_v384_failure_reversal else (3 if use_v383_adaptive else (15 if use_v382_active_plus else (3 if use_v382_more_trades else (2 if (use_v377 or use_v379) else (1 if (use_v359 or use_v363 or use_v364) else 2))))))),
        max_open_positions=(10 if use_v385_adaptive_plus else (5 if use_v384_failure_reversal else (3 if use_v383_adaptive else (15 if use_v382_active_plus else (3 if use_v382_more_trades else (2 if (use_v377 or use_v379) else (1 if (use_v359 or use_v363 or use_v364) else 2))))))),
        max_alerts_per_symbol_per_day=1,
        daily_loss_limit_pct=100.0,
        max_consecutive_losses=99,
        slippage_bps=3.0,
        candle_pattern_mode="off" if (use_v358 or use_v359 or use_v363 or use_v364 or use_v377 or use_v379) else "selective",
        enable_mean_reversion=False,
        enable_or_retest=True if use_v364 else False,
        v27_macro_filter_mode="off",
        v27_market_stress_mode="off" if (use_v358 or use_v359 or use_v363 or use_v364 or use_v377 or use_v379) else "skip",
        v27_news_filter_mode="off" if (use_v358 or use_v359 or use_v363 or use_v364 or use_v377 or use_v379) else "skip",
        v27_symbol_kill_switch_mode="off",
        v27_qqq_stress_abs_change_pct=4.2,
        v25_target_r=0.75,
        v25_max_hold_bars=12,
        enable_v358_live_quality_filter=use_v358,
        enable_v359_live_hunter_filter=(use_v359 or use_v363),
        enable_v364_professional_momentum_filter=use_v364,
        enable_v377_positive_context_filter=use_v377,
        enable_v379_decision_pattern_filter=use_v379,
        v379_pattern_mode=("v385_adaptive_plus" if use_v385_adaptive_plus else ("v384_failure_reversal" if use_v384_failure_reversal else ("v383_adaptive" if use_v383_adaptive else ("v382_active_plus" if use_v382_active_plus else ("v382_more_trades" if use_v382_more_trades else ("v38_active" if use_v38_active else ("v38_stable" if use_v38_stable else "balanced_vwap_prevhigh"))))))),
        q_learning_filter_enabled=_env_bool("LIVE_Q_POLICY_ENABLED", False),
        q_learning_policy_path=os.getenv("LIVE_Q_POLICY_PATH", ""),
        q_learning_min_edge=_env_float("LIVE_Q_POLICY_MIN_EDGE", 0.0),
        q_learning_min_state_count=max(1, _env_int("LIVE_Q_POLICY_MIN_STATE_COUNT", 8)),
        ml_ranker_filter_enabled=_env_bool("LIVE_ML_RANKER_ENABLED", False),
        ml_ranker_model_path=os.getenv("LIVE_ML_RANKER_MODEL_PATH", ""),
        ml_ranker_min_pred_r=_env_float("LIVE_ML_RANKER_MIN_PRED_R", 0.05),
        ml_ranker_min_win_prob=_env_float("LIVE_ML_RANKER_MIN_WIN_PROB", 0.0),
        v359_quality_start_time="10:00" if use_v363 else "10:00",
        v359_quality_end_time="11:00" if use_v363 else "11:00",
        v359_min_rvol=1.20 if use_v363 else 1.00,
        v359_min_daily_atr_pct=4.00 if use_v363 else 0.0,
        v359_min_directional_rs=0.00 if use_v363 else 0.00,
        v359_max_directional_rs=3.00 if use_v363 else 999.00,
        v359_min_directional_open_rs=0.00 if use_v363 else -999.00,
        v359_max_directional_open_rs=5.00 if use_v363 else 999.00,
        v359_min_directional_vwap_extension_atr=0.50 if use_v363 else 0.50,
        v359_max_directional_vwap_extension_atr=2.00 if use_v363 else 2.00,
        v359_max_abs_vwap_extension_atr=1.50 if use_v363 else 1.50,
        v364_quality_start_time="10:00",
        v364_quality_end_time="12:00",
        v364_min_rvol=1.00,
        v364_min_daily_atr_pct=4.00,
        v364_min_directional_rs=-1.00,
        v364_max_directional_rs=3.00,
        v364_min_directional_open_rs=-999.00,
        v364_max_directional_open_rs=999.00,
        v364_min_directional_vwap_extension_atr=0.50,
        v364_max_directional_vwap_extension_atr=2.00,
        v364_max_abs_vwap_extension_atr=2.00,
        v364_min_signal_risk_pct=0.15,
        v364_max_signal_risk_pct=1.50,
    )



PRESET_TO_LIVE_VARIANT = {
    "manual": "manual",
    "best_qqq_news": "best_report_153601",
    "live_raw_optimized_v358": "live_raw_optimized_v358",
    "live_hunter_v359": "live_hunter_v359",
    "live_longrun_robust_v362": "v363_grid_robust",
    "live_grid_robust_v363": "v363_grid_robust",
    "live_professional_momentum_v364": "live_professional_momentum_v364",
    "live_positive_context_v377": "live_positive_context_v377",
    "live_indicator_pattern_v379": "live_indicator_pattern_v379",
    "live_active_pattern_v38": "v38_active_pattern",
    "live_stable_pattern_v38": "v38_stable_pattern",
    "live_active_plus_v382": "v382_active_plus",
    "live_more_trades_v382": "v382_more_trades",
    "live_adaptive_composite_v383": "v383_adaptive",
    "live_failure_reversal_v384": "v384_failure_reversal",
    "live_adaptive_plus_v385": "v385_adaptive_plus",
}

GATE_TO_LIVE_VARIANT = {
    "off": "best_report_153601",
    "v358": "live_raw_optimized_v358",
    "v359": "live_hunter_v359",
    "v364": "live_professional_momentum_v364",
    "v377": "live_positive_context_v377",
    "v379": "live_indicator_pattern_v379",
    "v38_active": "v38_active_pattern",
    "v38_stable": "v38_stable_pattern",
    "v382_active_plus": "v382_active_plus",
    "v382_more_trades": "v382_more_trades",
    "v383_adaptive": "v383_adaptive",
    "v384_failure_reversal": "v384_failure_reversal",
    "v385_adaptive_plus": "v385_adaptive_plus",
}


LIVE_STRATEGY_EXPERIMENT_SPECS: tuple[dict[str, str], ...] = (
    {"preset": "best_qqq_news", "label": "Best Report 153601 baseline", "variant": "best_report_153601", "quality_gate": "off", "code": "base"},
    {"preset": "live_raw_optimized_v358", "label": "V35.8 Raw quality gate", "variant": "live_raw_optimized_v358", "quality_gate": "v358", "code": "v358"},
    {"preset": "live_hunter_v359", "label": "V35.9 Live Hunter", "variant": "live_hunter_v359", "quality_gate": "v359", "code": "v359"},
    {"preset": "live_grid_robust_v363", "label": "V36.3 Grid-tested robust", "variant": "v363_grid_robust", "quality_gate": "v359", "code": "v363"},
    {"preset": "live_professional_momentum_v364", "label": "V36.4 Pro momentum hybrid", "variant": "live_professional_momentum_v364", "quality_gate": "v364", "code": "v364"},
    {"preset": "live_positive_context_v377", "label": "V37.8 Mined pattern matcher", "variant": "live_positive_context_v377", "quality_gate": "v377", "code": "v377"},
    {"preset": "live_indicator_pattern_v379", "label": "V37.9 Indicator pattern scorer", "variant": "live_indicator_pattern_v379", "quality_gate": "v379", "code": "v379"},
    {"preset": "live_active_pattern_v38", "label": "V38 Active pattern scorer", "variant": "v38_active_pattern", "quality_gate": "v38_active", "code": "v38a"},
    {"preset": "live_stable_pattern_v38", "label": "V38 Stable pattern scorer", "variant": "v38_stable_pattern", "quality_gate": "v38_stable", "code": "v38s"},
    {"preset": "live_active_plus_v382", "label": "V38.2 Active Plus", "variant": "v382_active_plus", "quality_gate": "v382_active_plus", "code": "v382a"},
    {"preset": "live_more_trades_v382", "label": "V38.2 More Trades Research", "variant": "v382_more_trades", "quality_gate": "v382_more_trades", "code": "v382m"},
    {"preset": "live_adaptive_composite_v383", "label": "V38.3 Adaptive Composite", "variant": "v383_adaptive", "quality_gate": "v383_adaptive", "code": "v383"},
    {"preset": "live_failure_reversal_v384", "label": "V38.4 Failure-aware reversal router", "variant": "v384_failure_reversal", "quality_gate": "v384_failure_reversal", "code": "v384"},
    {"preset": "live_adaptive_plus_v385", "label": "V38.5 Adaptive Plus", "variant": "v385_adaptive_plus", "quality_gate": "v385_adaptive_plus", "code": "v385"},
)


def all_live_strategy_specs() -> list[dict[str, str]]:
    """Every deterministic live strategy preset used by all-strategies paper mode."""
    return [dict(spec) for spec in LIVE_STRATEGY_EXPERIMENT_SPECS]


def _spec_for_preset_or_variant(preset: str | None = None, variant: str | None = None, quality_gate: str | None = None) -> dict[str, str]:
    preset_l = str(preset or "").strip().lower()
    variant_l = str(variant or "").strip().lower()
    for spec in LIVE_STRATEGY_EXPERIMENT_SPECS:
        if preset_l and preset_l == spec["preset"]:
            return dict(spec)
        if variant_l and variant_l == spec["variant"]:
            return dict(spec)
    v = variant_l or live_variant_from_dashboard(preset_l or "manual", quality_gate or "off")
    return {"preset": preset_l or "manual", "label": preset_l or v or "Manual", "variant": v, "quality_gate": str(quality_gate or "off").strip().lower(), "code": _strategy_code(v)}


def _strategy_code(value: str | None) -> str:
    raw = str(value or "strat").strip().lower()
    mapping = {spec["variant"]: spec["code"] for spec in LIVE_STRATEGY_EXPERIMENT_SPECS}
    if raw in mapping:
        return mapping[raw]
    raw = raw.replace("live_", "").replace("strategy_", "").replace("adaptive_plus", "v385")
    raw = re.sub(r"[^a-z0-9]", "", raw)
    return (raw or "strat")[:7]


def _strategy_variant_from_code(code: str | None) -> str:
    code_l = str(code or "").strip().lower()
    for spec in LIVE_STRATEGY_EXPERIMENT_SPECS:
        if code_l and code_l == str(spec.get("code", "")).lower():
            return str(spec.get("variant") or "")
    return ""


def _strategy_spec_from_signal_values(variant: str | None = None, preset: str | None = None, gate: str | None = None, code: str | None = None) -> dict[str, str]:
    code_variant = _strategy_variant_from_code(code) if code else ""
    return _spec_for_preset_or_variant(preset or None, variant or code_variant or None, gate or None)


def _client_order_strategy_code(client_order_id: str) -> str:
    parts = str(client_order_id or "").split("-")
    # New V8 order id shape: rmv33-v385-AAPL-YYYYMMDDHHMM-l
    if len(parts) >= 5 and re.fullmatch(r"[a-z0-9]{2,8}", parts[1] or ""):
        return parts[1]
    return ""


def live_variant_from_dashboard(settings_preset: str | None, quality_gate: str | None) -> str:
    preset = str(settings_preset or "manual").strip().lower()
    gate = str(quality_gate or "off").strip().lower()
    variant = PRESET_TO_LIVE_VARIANT.get(preset, preset)
    if variant == "manual":
        variant = GATE_TO_LIVE_VARIANT.get(gate, "best_report_153601")
    return str(variant or "best_report_153601").strip().lower()


def _apply_quality_gate_to_params(params: StrategyParams, cfg: dict[str, Any]) -> None:
    gate_mode = str(cfg.get("live_quality_gate") or cfg.get("quality_gate") or "off").lower()
    # Reset all mutually-exclusive live quality gates first. This makes the live worker follow the dashboard exactly.
    params.enable_v358_live_quality_filter = False
    params.enable_v359_live_hunter_filter = False
    params.enable_v364_professional_momentum_filter = False
    params.enable_v377_positive_context_filter = False
    params.enable_v379_decision_pattern_filter = False
    if gate_mode == "v358":
        params.enable_v358_live_quality_filter = True
        prefix = "v358"
    elif gate_mode == "v364":
        params.enable_v364_professional_momentum_filter = True
        prefix = "v364"
    elif gate_mode == "v377":
        params.enable_v377_positive_context_filter = True
        return
    elif gate_mode in {"v379", "v38_active", "v38_stable", "v382_active_plus", "v383_adaptive", "v384_failure_reversal", "v385_adaptive_plus", "v382_more_trades"}:
        params.enable_v379_decision_pattern_filter = True
        mode_map = {
            "v379": "balanced_vwap_prevhigh",
            "v38_active": "v38_active",
            "v38_stable": "v38_stable",
            "v382_active_plus": "v382_active_plus",
            "v382_more_trades": "v382_more_trades",
            "v383_adaptive": "v383_adaptive",
            "v384_failure_reversal": "v384_failure_reversal",
            "v385_adaptive_plus": "v385_adaptive_plus",
        }
        params.v379_pattern_mode = mode_map.get(gate_mode, "balanced_vwap_prevhigh")
        return
    elif gate_mode in {"v359", "custom"}:
        params.enable_v359_live_hunter_filter = True
        prefix = "v359"
    else:
        return
    # For the scalar quality-gate settings, use the dashboard values for live too.
    setattr(params, f"{prefix}_quality_start_time", str(cfg.get("quality_start_time") or "10:00"))
    setattr(params, f"{prefix}_quality_end_time", str(cfg.get("quality_end_time") or "11:00"))
    setattr(params, f"{prefix}_min_rvol", _safe_float(cfg.get("quality_min_rvol"), 0.0))
    setattr(params, f"{prefix}_min_daily_atr_pct", _safe_float(cfg.get("quality_min_daily_atr"), 0.0))
    setattr(params, f"{prefix}_min_directional_rs", _safe_float(cfg.get("quality_min_dir_rs"), -999.0))
    setattr(params, f"{prefix}_max_directional_rs", _safe_float(cfg.get("quality_max_dir_rs"), 999.0))
    setattr(params, f"{prefix}_min_directional_open_rs", _safe_float(cfg.get("quality_min_dir_open_rs"), -999.0))
    setattr(params, f"{prefix}_max_directional_open_rs", _safe_float(cfg.get("quality_max_dir_open_rs"), 999.0))
    setattr(params, f"{prefix}_min_directional_vwap_extension_atr", _safe_float(cfg.get("quality_min_dir_vwap"), -999.0))
    setattr(params, f"{prefix}_max_directional_vwap_extension_atr", _safe_float(cfg.get("quality_max_dir_vwap"), 999.0))
    setattr(params, f"{prefix}_max_abs_vwap_extension_atr", _safe_float(cfg.get("quality_max_abs_vwap"), 999.0))

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _floor_5min(ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return ts - timedelta(minutes=ts.minute % 5)


def latest_closed_5m_start(now: datetime | None = None) -> pd.Timestamp:
    now = now or _now_utc()
    return pd.Timestamp(_floor_5min(now) - timedelta(minutes=5)).tz_convert("UTC")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _parse_hhmm(value: Any, default: str) -> tuple[int, int]:
    raw = str(value or default).strip()
    try:
        parts = raw.split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m
    except Exception:
        pass
    h, m = default.split(":")
    return int(h), int(m)




def _session_date_series(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    return ts.dt.tz_convert(NY).dt.date


def _daily_feature_table_for_next_session(daily: pd.DataFrame) -> pd.DataFrame:
    """Build previous-completed-day features for live intraday rows.

    Alpaca's 1Day endpoint may not include the current trading day before the
    regular close.  The old live path merged intraday rows on the same
    session_date and therefore produced NaN prev_close/daily ATR in premarket.
    For live decisions we need the most recent completed daily bar strictly
    before the intraday session.
    """
    if daily is None or daily.empty:
        return pd.DataFrame()
    d = daily.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"])
    if d.empty:
        return pd.DataFrame()
    d["symbol"] = d.get("symbol", "").astype(str).str.upper()
    d["daily_session_date"] = d["timestamp"].dt.tz_convert(NY).dt.date
    d = d.sort_values(["symbol", "daily_session_date", "timestamp"])
    d = d.drop_duplicates(["symbol", "daily_session_date"], keep="last")
    for col in ["open", "high", "low", "close", "volume"]:
        d[col] = pd.to_numeric(d.get(col), errors="coerce")
    d["daily_dollar_volume"] = d["volume"] * d["close"]
    prev_close_for_tr = d.groupby("symbol")["close"].shift(1)
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close_for_tr).abs(),
            (d["low"] - prev_close_for_tr).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["daily_atr14"] = tr.groupby(d["symbol"]).transform(lambda x: x.rolling(14, min_periods=14).mean())
    d["daily_atr14_percent"] = d["daily_atr14"] / d["close"].replace(0, pd.NA) * 100.0
    d["avg_20d_dollar_volume"] = d.groupby("symbol")["daily_dollar_volume"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    features = d[["symbol", "daily_session_date", "high", "low", "close", "avg_20d_dollar_volume", "daily_atr14_percent"]].copy()
    features = features.rename(columns={"high": "prev_day_high", "low": "prev_day_low", "close": "prev_close"})
    features["feature_date_key"] = pd.to_datetime(features["daily_session_date"].astype(str), errors="coerce")
    return features.sort_values(["symbol", "feature_date_key"])


def _add_daily_features_live_safe(intraday: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Attach previous-completed-daily features to live intraday bars.

    This is intentionally used only by the live worker.  It fixes the premarket
    / early-session case where the current 1Day Alpaca row is not available yet,
    which otherwise caused no same-day indicator row and blank 0/0 monitor rows.
    """
    if intraday is None or intraday.empty:
        return pd.DataFrame()
    left = intraday.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
    left = left.dropna(subset=["timestamp"])
    if left.empty:
        return left
    left["symbol"] = left.get("symbol", "").astype(str).str.upper()
    if "session_date" not in left.columns:
        left["session_date"] = left["timestamp"].dt.tz_convert(NY).dt.date
    left["feature_date_key"] = pd.to_datetime(left["session_date"].astype(str), errors="coerce")
    feats = _daily_feature_table_for_next_session(daily)
    if feats.empty:
        for col in ["prev_day_high", "prev_day_low", "prev_close", "avg_20d_dollar_volume", "daily_atr14_percent"]:
            if col not in left.columns:
                left[col] = pd.NA
        return left.drop(columns=["feature_date_key"], errors="ignore")
    frames = []
    for sym, lgrp in left.sort_values(["symbol", "feature_date_key", "timestamp"]).groupby("symbol", sort=False):
        fgrp = feats[feats["symbol"] == sym].sort_values("feature_date_key")
        if fgrp.empty:
            tmp = lgrp.copy()
            for col in ["prev_day_high", "prev_day_low", "prev_close", "avg_20d_dollar_volume", "daily_atr14_percent"]:
                tmp[col] = pd.NA
            frames.append(tmp)
            continue
        merged = pd.merge_asof(
            lgrp.sort_values("feature_date_key"),
            fgrp[["feature_date_key", "prev_day_high", "prev_day_low", "prev_close", "avg_20d_dollar_volume", "daily_atr14_percent"]].sort_values("feature_date_key"),
            on="feature_date_key",
            direction="backward",
            allow_exact_matches=False,
        )
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True) if frames else left
    return out.drop(columns=["feature_date_key"], errors="ignore")


def _build_qqq_context_live_safe(qqq_5m: pd.DataFrame, qqq_daily: pd.DataFrame, session_mode: str = "regular_only") -> pd.DataFrame:
    if qqq_5m is None or qqq_5m.empty:
        return pd.DataFrame()
    q = add_intraday_features(qqq_5m, session_mode=session_mode)
    q = _add_daily_features_live_safe(q, qqq_daily)
    if q.empty:
        return q
    q_resample = q.set_index("timestamp_ny").copy() if "timestamp_ny" in q.columns else pd.DataFrame()
    fifteen_frames = []
    if not q_resample.empty:
        for session_date, group in q_resample.groupby("session_date"):
            ohlcv = group.resample("15min", label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
            if not ohlcv.empty:
                ohlcv["session_date"] = session_date
                fifteen_frames.append(ohlcv)
    if fifteen_frames:
        q15 = pd.concat(fifteen_frames).reset_index().rename(columns={"timestamp_ny": "timestamp_ny_15m"})
        q15["timestamp"] = q15["timestamp_ny_15m"].dt.tz_convert("UTC")
        q15 = q15.sort_values("timestamp")
        q15["qqq_15m_close"] = q15["close"]
        q15["qqq_15m_ema50"] = q15["close"].ewm(span=50, adjust=False, min_periods=50).mean()
        q15 = q15[["timestamp", "qqq_15m_close", "qqq_15m_ema50"]]
    else:
        q15 = pd.DataFrame(columns=["timestamp", "qqq_15m_close", "qqq_15m_ema50"])
    needed = ["timestamp", "session_date", "close", "session_vwap", "prev_close", "session_open", "ema9", "ema20", "rsi2", "atr5m14", "daily_atr14_percent"]
    for col in needed:
        if col not in q.columns:
            q[col] = pd.NA
    q5 = q[needed].copy()
    q5 = q5.rename(columns={"close": "qqq_close", "session_vwap": "qqq_session_vwap", "prev_close": "qqq_prev_close", "session_open": "qqq_session_open", "ema9": "qqq_ema9", "ema20": "qqq_ema20", "rsi2": "qqq_rsi2", "atr5m14": "qqq_atr5m14", "daily_atr14_percent": "qqq_daily_atr14_percent"})
    q5["qqq_day_change_percent"] = (pd.to_numeric(q5["qqq_close"], errors="coerce") - pd.to_numeric(q5["qqq_prev_close"], errors="coerce")) / pd.to_numeric(q5["qqq_prev_close"], errors="coerce").replace(0, pd.NA) * 100
    q5["qqq_change_from_open"] = (pd.to_numeric(q5["qqq_close"], errors="coerce") - pd.to_numeric(q5["qqq_session_open"], errors="coerce")) / pd.to_numeric(q5["qqq_session_open"], errors="coerce").replace(0, pd.NA) * 100
    q5["qqq_15min_change_percent"] = q5.groupby("session_date")["qqq_close"].transform(lambda x: (x - x.shift(3)) / x.shift(3) * 100)
    q5 = q5.sort_values("timestamp")
    if not q15.empty:
        q5 = pd.merge_asof(q5.sort_values("timestamp"), q15.sort_values("timestamp"), on="timestamp", direction="backward")
    else:
        q5["qqq_15m_close"] = pd.NA
        q5["qqq_15m_ema50"] = pd.NA
    q5["market_filter_pass"] = (
        (pd.to_numeric(q5["qqq_15m_close"], errors="coerce") > pd.to_numeric(q5["qqq_15m_ema50"], errors="coerce"))
        & (pd.to_numeric(q5["qqq_close"], errors="coerce") > pd.to_numeric(q5["qqq_session_vwap"], errors="coerce"))
        & (pd.to_numeric(q5["qqq_daily_atr14_percent"], errors="coerce") <= 3.2)
    )
    return q5.sort_values("timestamp")

def _time_in_window(ts_et: pd.Timestamp | datetime, start_hhmm: Any, end_hhmm: Any) -> bool:
    ts = pd.Timestamp(ts_et)
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY)
    else:
        ts = ts.tz_convert(NY)
    start_h, start_m = _parse_hhmm(start_hhmm, "09:35")
    end_h, end_m = _parse_hhmm(end_hhmm, "15:55")
    minutes = ts.hour * 60 + ts.minute
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    if start <= end:
        return start <= minutes <= end
    return minutes >= start or minutes <= end


def _order_side(strategy_side: str) -> str:
    return "buy" if str(strategy_side).lower() == "long" else "sell"


def _round_price(price: float) -> float:
    return round(float(price), 2) if price >= 1 else round(float(price), 4)


def _load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _append_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def _merge_bar_cache(existing: pd.DataFrame | None, new: pd.DataFrame, start_keep: datetime) -> pd.DataFrame:
    frames = []
    if existing is not None and not existing.empty:
        frames.append(existing)
    if new is not None and not new.empty:
        frames.append(new)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out = out[out["timestamp"] >= pd.Timestamp(start_keep)].copy()
    out = out.drop_duplicates(["symbol", "timestamp"], keep="last")
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


class LivePaperTradingEngine:
    def __init__(self, live: LiveSettings, risk: LiveRiskConfig):
        self.live = live
        self.risk = risk
        self.params = make_best_report_153601_params(risk, variant=getattr(live, "strategy_variant", None))
        self.settings = AlpacaSettings()
        self.data_client = AlpacaDataClient(self.settings)
        self.trading_client = AlpacaTradingClient(self.settings)
        self.store = LiveStore()
        self.state = _load_state(self.live.state_path)
        db_state = self.store.get_state("engine_state", {})
        if isinstance(db_state, dict):
            self.state.update({k: v for k, v in db_state.items() if k not in self.state})
        self._apply_runtime_config_override()
        self._bars_5m_cache: pd.DataFrame = pd.DataFrame()
        self._daily_cache: pd.DataFrame = pd.DataFrame()

    def validate_environment(self) -> None:
        self.store.set_state("settings", {"live": self._serializable_live(), "risk": asdict(self.risk), "alpaca": self.settings.redacted()})
        if not self.settings.is_configured:
            self.store.insert_event("worker_start_blocked", {"message": "Missing Alpaca API keys."}, status="blocked")
            raise RuntimeError("Missing Alpaca API keys. Add ALPACA_API_KEY and ALPACA_SECRET_KEY as Render environment variables.")
        base = str(self.settings.trading_base_url or "").lower()
        is_paper = "paper-api" in base
        if not is_paper and not self.live.allow_live_trading:
            self.store.insert_event("worker_start_blocked", {"message": "Trading URL is not Alpaca paper endpoint."}, status="blocked")
            raise RuntimeError("Refusing to start because ALPACA_TRADING_BASE_URL is not the paper endpoint and ALLOW_LIVE_TRADING is false.")
        if not self.live.enabled:
            self.store.insert_event("worker_start_blocked", {"message": "PAPER_TRADING_ENABLED is false."}, status="blocked")
            raise RuntimeError("PAPER_TRADING_ENABLED is false. Set it to true only when you want the worker to submit paper orders.")
        self.store.insert_event("worker_started", {"message": "Paper worker validated and started.", "dry_run": self.live.dry_run}, status="started")

    def _serializable_live(self) -> dict[str, Any]:
        data = asdict(self.live)
        data["log_path"] = str(self.live.log_path)
        data["state_path"] = str(self.live.state_path)
        return data

    def _make_params_for_live_config(self, cfg: dict[str, Any], preset: str | None = None, variant: str | None = None, quality_gate: str | None = None) -> StrategyParams:
        """Build one live StrategyParams object from DB config plus a strategy spec.

        Global risk/schedule/user controls remain shared; the strategy-specific
        preset/variant/gate is injected per strategy for all-strategies mode.
        """
        cfg = cfg if isinstance(cfg, dict) else {}
        spec = _spec_for_preset_or_variant(preset or cfg.get("settings_preset"), variant or cfg.get("strategy_variant"), quality_gate or cfg.get("live_quality_gate"))
        params = make_best_report_153601_params(self.risk, variant=spec.get("variant"))
        params.direction_mode = str(cfg.get("direction_mode") or params.direction_mode)
        params.backtest_session_mode = str(cfg.get("backtest_session_mode") or params.backtest_session_mode)
        params.min_candidate_score = _safe_float(cfg.get("min_score"), params.min_candidate_score)
        params.max_trades_per_day = max(1, int(_safe_float(cfg.get("max_trades"), params.max_trades_per_day)))
        params.max_open_positions = max(1, int(_safe_float(cfg.get("max_open_positions"), self.live.max_open_positions)))
        params.slippage_bps = _safe_float(cfg.get("slippage_bps"), params.slippage_bps)
        params.candle_pattern_mode = str(cfg.get("candle_mode") or params.candle_pattern_mode)
        params.enable_mean_reversion = str(cfg.get("enable_mr", params.enable_mean_reversion)).lower() in {"1", "true", "yes", "on"}
        params.enable_or_retest = str(cfg.get("enable_or", params.enable_or_retest)).lower() in {"1", "true", "yes", "on"}
        params.v27_macro_filter_mode = str(cfg.get("macro_filter") or params.v27_macro_filter_mode)
        params.v27_market_stress_mode = str(cfg.get("stress_filter") or params.v27_market_stress_mode)
        params.v27_news_filter_mode = str(cfg.get("news_filter") or params.v27_news_filter_mode)
        params.v27_symbol_kill_switch_mode = str(cfg.get("kill_switch") or params.v27_symbol_kill_switch_mode)
        params.v27_qqq_stress_abs_change_pct = _safe_float(cfg.get("qqq_stress_threshold"), params.v27_qqq_stress_abs_change_pct)
        gate_cfg = dict(cfg)
        gate_cfg["live_quality_gate"] = spec.get("quality_gate") or cfg.get("live_quality_gate") or "off"
        _apply_quality_gate_to_params(params, gate_cfg)
        if bool(cfg.get("custom_symbols_active", False)):
            params.v25_allow_generic_symbols = True
            params.min_price = 0.01
            params.min_avg_20d_dollar_volume = 0.0
            params.min_current_5m_dollar_volume = 0.0
            params.min_daily_atr_pct = 0.0
            params.max_daily_atr_pct = 999.0
            params.v25_min_rvol = 0.0
        setattr(params, "live_config_source", getattr(self.live, "live_config_source", "dashboard_db"))
        setattr(params, "live_strategy_variant", spec.get("variant") or "best_report_153601")
        setattr(params, "live_strategy_preset", spec.get("preset") or "manual")
        setattr(params, "live_strategy_label", spec.get("label") or spec.get("preset") or spec.get("variant"))
        setattr(params, "live_strategy_code", spec.get("code") or _strategy_code(spec.get("variant")))
        setattr(params, "live_quality_gate", spec.get("quality_gate") or gate_cfg.get("live_quality_gate") or "off")
        return params



    def _apply_runtime_config_override(self) -> None:
        """Apply dashboard-persisted live settings from the shared DB.

        On Render, the Dash web service and the worker share DATABASE_URL.  This
        lets the selected dashboard strategy/preset and visible settings control
        the already-running paper/live worker without changing environment vars
        or redeploying the worker.
        """
        try:
            cfg = self.store.get_state("live_config_override", {})
        except Exception:
            cfg = {}
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            self.live.live_config_source = "env"
            self.live.strategy_variant = os.getenv("LIVE_STRATEGY_VARIANT", self.live.strategy_variant or "best_report_153601").strip().lower()
            self.params = make_best_report_153601_params(self.risk, variant=self.live.strategy_variant)
            setattr(self.params, "live_config_source", "env")
            setattr(self.params, "live_strategy_variant", self.live.strategy_variant)
            setattr(self.params, "live_strategy_preset", "env")
            setattr(self.params, "live_quality_gate", "env")
            return

        self._runtime_config = dict(cfg)
        self.live.live_config_source = "dashboard_db"
        self.live.strategy_variant = str(cfg.get("strategy_variant") or live_variant_from_dashboard(cfg.get("settings_preset"), cfg.get("live_quality_gate"))).strip().lower()
        self.live.strategy_run_mode = str(cfg.get("live_strategy_run_mode") or "single").strip().lower()
        if self.live.strategy_run_mode not in {"single", "all_strategies"}:
            self.live.strategy_run_mode = "single"
        self.live.feed = str(cfg.get("feed") or self.live.feed or "iex")
        if isinstance(cfg.get("symbols"), list) and cfg.get("symbols"):
            self.live.symbols = [str(s).upper() for s in cfg.get("symbols") if str(s).strip()]
        self.live.max_daily_trades = max(1, int(_safe_float(cfg.get("max_daily_trades"), self.live.max_daily_trades)))
        self.live.max_open_positions = max(1, int(_safe_float(cfg.get("max_open_positions"), self.live.max_open_positions)))
        self.live.max_orders_per_symbol_per_day = max(1, int(_safe_float(cfg.get("max_orders_per_symbol_per_day"), self.live.max_orders_per_symbol_per_day)))
        self.live.use_news_proxy = str(cfg.get("use_news", self.live.use_news_proxy)).lower() in {"1", "true", "yes", "on"}
        self.live.selection_mode = str(cfg.get("selection_mode") or self.live.selection_mode or "seen_so_far_top_n")
        self.live.entry_start_time_et = str(cfg.get("live_entry_start_time_et") or cfg.get("quality_start_time") or self.live.entry_start_time_et or "09:35")
        self.live.entry_end_time_et = str(cfg.get("live_entry_end_time_et") or cfg.get("quality_end_time") or self.live.entry_end_time_et or "15:55")
        self.live.allow_extended_hours_entries = str(cfg.get("live_allow_extended_hours_entries", self.live.allow_extended_hours_entries)).lower() in {"1", "true", "yes", "on"}
        self.live.force_market_open = str(cfg.get("live_require_market_open", self.live.force_market_open)).lower() in {"1", "true", "yes", "on"}
        self.live.enable_max_hold_exit = str(cfg.get("live_enable_max_hold_exit", self.live.enable_max_hold_exit)).lower() in {"1", "true", "yes", "on"}

        self.risk.account_value_fallback = _safe_float(cfg.get("account_value"), self.risk.account_value_fallback)
        self.risk.fixed_risk_dollars = _safe_float(cfg.get("risk_dollars"), self.risk.fixed_risk_dollars)
        self.risk.position_sizing_mode = str(cfg.get("risk_mode") or self.risk.position_sizing_mode)
        self.risk.base_risk_pct = _safe_float(cfg.get("base_risk_pct"), self.risk.base_risk_pct)
        self.risk.min_risk_dollars = _safe_float(cfg.get("min_risk_dollars"), self.risk.min_risk_dollars)
        self.risk.max_risk_dollars = _safe_float(cfg.get("max_risk_dollars"), self.risk.max_risk_dollars)
        self.risk.dd1_risk_pct = _safe_float(cfg.get("dd1_risk_pct"), self.risk.dd1_risk_pct)
        self.risk.dd2_risk_pct = _safe_float(cfg.get("dd2_risk_pct"), self.risk.dd2_risk_pct)
        self.risk.pause_dd_pct = _safe_float(cfg.get("pause_dd_pct"), self.risk.pause_dd_pct)

        self.params = self._make_params_for_live_config(cfg, cfg.get("settings_preset"), self.live.strategy_variant, cfg.get("live_quality_gate"))
        if self.live.strategy_run_mode == "all_strategies":
            specs = all_live_strategy_specs()
            self.live.active_strategy_count = len(specs)
            self.live.active_strategy_variants = [spec["variant"] for spec in specs]
        else:
            self.live.active_strategy_count = 1
            self.live.active_strategy_variants = [self.live.strategy_variant]

    def _heartbeat(self, status: str, message: str = "", extra: dict[str, Any] | None = None) -> None:
        payload = {
            "updated_at_utc": utc_now_iso(),
            "status": status,
            "message": message,
            "dry_run": self.live.dry_run,
            "enabled": self.live.enabled,
            "symbols": len(self.live.symbols or []),
            "feed": self.live.feed,
            "realtime_feed_for_orders": self._live_realtime_feed(),
            "live_data_policy": "real_time_only_for_order_decisions",
            "delayed_data_used_for_orders": False,
            "strategy_preset": getattr(self.params, "live_strategy_preset", getattr(self.live, "strategy_variant", "best_report_153601")),
            "strategy_variant": getattr(self.params, "live_strategy_variant", getattr(self.live, "strategy_variant", "best_report_153601")),
            "strategy_run_mode": getattr(self.live, "strategy_run_mode", "single"),
            "active_strategy_count": getattr(self.live, "active_strategy_count", 1),
            "active_strategy_variants": getattr(self.live, "active_strategy_variants", None) or [getattr(self.live, "strategy_variant", "best_report_153601")],
            "quality_gate": getattr(self.params, "live_quality_gate", "env"),
            "config_source": getattr(self.live, "live_config_source", "env"),
            "entry_start_time_et": getattr(self.live, "entry_start_time_et", "09:35"),
            "entry_end_time_et": getattr(self.live, "entry_end_time_et", "15:55"),
            "allow_extended_hours_entries": getattr(self.live, "allow_extended_hours_entries", False),
            "force_market_open": getattr(self.live, "force_market_open", True),
            "bar_session_mode": self._live_session_mode(),
        }
        if extra:
            payload.update(extra)
        self.store.set_state("heartbeat", payload)

    def _account_snapshot(self) -> dict[str, Any]:
        try:
            account = self.trading_client.get_account()
            self.store.insert_account_snapshot(account)
            return account
        except Exception as exc:
            self.store.insert_event("account_sync_error", {"message": str(exc)}, status="error")
            return {}

    def _account_equity(self, account: dict[str, Any] | None = None) -> tuple[float, float, float]:
        account = account or self._account_snapshot()
        equity = _safe_float(account.get("equity"), self.risk.account_value_fallback) if account else self.risk.account_value_fallback
        last_equity = _safe_float(account.get("last_equity"), equity) if account else equity
        high_watermark = max(equity, last_equity, self.risk.account_value_fallback)
        daily_pl = equity - last_equity
        return equity, high_watermark, daily_pl

    def _account_buying_power(self, account: dict[str, Any] | None = None) -> float:
        """Conservative buying-power value for order sizing.

        Alpaca can reject an order even when `buying_power` looks high if the
        Reg-T/day-trading buckets are smaller or if previous orders already
        reserve capital.  Use the smallest positive bucket we receive instead
        of sizing from equity/risk alone.
        """
        account = account or {}
        values: list[float] = []
        for key in ("regt_buying_power", "buying_power", "daytrading_buying_power", "non_marginable_buying_power"):
            val = _safe_float(account.get(key), 0.0)
            if val > 0:
                values.append(val)
        return min(values) if values else 0.0

    def _buying_power_safety_fraction(self) -> float:
        """Configured reserve applied to Alpaca buying power before submitting orders."""
        cfg = getattr(self, "_runtime_config", {}) if isinstance(getattr(self, "_runtime_config", {}), dict) else {}
        return max(0.05, min(0.95, _safe_float(cfg.get("live_buying_power_safety_pct", 80.0), 80.0) / 100.0))

    def _configured_notional_caps(self, equity: float, buying_power: float) -> list[tuple[str, float]]:
        """Optional user-configured exposure caps.

        These are safety rails, not a replacement for the existing risk/compounding
        model. By default there is no hard-coded percent cap. If the dashboard/DB
        explicitly sets live_max_order_notional_dollars or
        live_max_position_notional_pct/max_position_notional_pct, those caps are
        honored.
        """
        cfg = getattr(self, "_runtime_config", {}) if isinstance(getattr(self, "_runtime_config", {}), dict) else {}
        caps: list[tuple[str, float]] = []
        dollar_cap = _safe_float(cfg.get("live_max_order_notional_dollars", cfg.get("max_order_notional_dollars", 0.0)), 0.0)
        if dollar_cap > 0:
            caps.append(("configured_dollar_cap", dollar_cap))
        pct_raw = cfg.get("live_max_position_notional_pct", cfg.get("max_position_notional_pct", 0.0))
        pct_cap = _safe_float(pct_raw, 0.0)
        # Backtest StrategyParams historically defaulted max_position_notional_pct to
        # 9999 to effectively disable it. Treat <=0 and >=1000 as disabled in live.
        if 0.0 < pct_cap < 1000.0:
            base = max(float(equity or 0.0), 0.0)
            if base > 0:
                caps.append(("configured_equity_pct_cap", base * pct_cap / 100.0))
        bp_pct_cap = _safe_float(cfg.get("live_max_order_buying_power_pct", 0.0), 0.0)
        if 0.0 < bp_pct_cap <= 100.0 and buying_power > 0:
            caps.append(("configured_buying_power_pct_cap", buying_power * bp_pct_cap / 100.0))
        return [(name, val) for name, val in caps if val > 0]

    def _order_notional_budget(self, buying_power: float, reserved_notional: float, remaining_capacity: int, strategy_multiplier: int) -> tuple[float, str]:
        """Return the maximum notional allowed for the next order.

        The existing strategy risk model still decides the desired dollar risk
        using fixed risk / percent equity / controlled compounding. This function
        only prevents Alpaca rejects by translating the shared paper account's
        current buying power into an execution budget.

        In all-strategies mode, many strategies can signal in the same scan. The
        default allocation is therefore per remaining available slot, derived from
        max_daily_trades/max_open_positions and active_strategy_count. This is not
        a hard-coded 10% cap; it adapts to the user's existing capacity settings
        and current Alpaca buying power.
        """
        usable_bp = max(0.0, float(buying_power or 0.0) * self._buying_power_safety_fraction() - float(reserved_notional or 0.0))
        if usable_bp <= 0:
            return 0.0, "no_usable_buying_power"
        cfg = getattr(self, "_runtime_config", {}) if isinstance(getattr(self, "_runtime_config", {}), dict) else {}
        mode = str(cfg.get("live_budget_allocation_mode", "per_available_slot") or "per_available_slot").strip().lower()
        if mode in {"full_available", "shared_full"}:
            cap = usable_bp
            reason = "available_buying_power"
        elif mode in {"per_configured_slot", "per_configured_capacity"}:
            configured_slots = max(1, int(getattr(self.live, "max_open_positions", 1) or 1) * max(1, int(strategy_multiplier or 1)))
            cap = usable_bp / configured_slots
            reason = f"buying_power_per_configured_slot_{configured_slots}"
        else:
            slots = max(1, int(remaining_capacity or 1))
            cap = usable_bp / slots
            reason = f"buying_power_per_remaining_slot_{slots}"
        for cap_name, cap_val in self._configured_notional_caps(self._last_equity_for_budget if hasattr(self, "_last_equity_for_budget") else 0.0, buying_power):
            if cap_val > 0 and cap_val < cap:
                cap = cap_val
                reason = cap_name
        return max(0.0, cap), reason

    def _market_is_open(self) -> bool:
        if not self.live.force_market_open:
            return True
        try:
            clock = self.trading_client.get_clock()
            self.store.set_state("market_clock", clock)
            return bool(clock.get("is_open"))
        except Exception:
            ny = datetime.now(NY)
            if ny.weekday() >= 5:
                return False
            if bool(getattr(self.live, "allow_extended_hours_entries", False)):
                return _time_in_window(pd.Timestamp(ny), getattr(self.live, "entry_start_time_et", "04:00"), getattr(self.live, "entry_end_time_et", "20:00"))
            return _time_in_window(pd.Timestamp(ny), max(str(getattr(self.live, "entry_start_time_et", "09:35")), "09:30"), min(str(getattr(self.live, "entry_end_time_et", "15:55")), "16:00"))

    def _live_session_mode(self) -> str:
        """Which bar session the live indicator pipeline should evaluate.

        The original live strategies were regular-session backtests.  When the
        dashboard explicitly enables extended-hours entries we must feed
        pre/post-market bars into add_intraday_features()/QQQ context; otherwise
        the indicator pipeline filters them out as regular_only and every
        premarket symbol appears as 0/0 checks.
        """
        return "extended_hours" if bool(getattr(self.live, "allow_extended_hours_entries", False)) else "regular_only"


    def _live_realtime_feed(self) -> str:
        """Feed used by the LIVE trading decision path.

        Live trading must never use delayed or historical fallback data to create
        orders.  The free paper-account live feed is IEX.  If an old DB setting
        still contains delayed_sip/sip_delayed from an earlier diagnostic build,
        force the live decision path back to IEX and expose the conversion in
        diagnostics instead of silently trading from delayed bars.
        """
        feed = str(getattr(self.live, "feed", "iex") or "iex").lower().strip()
        if feed in {"delayed_sip", "free_delayed_sip", "sip_delayed"}:
            return "iex"
        if feed not in {"iex", "sip"}:
            return "iex"
        return feed

    def _max_bar_age_minutes(self) -> int:
        """Maximum age of a bar that may still be used for live decisions.

        Alpaca Basic/free accounts usually run on IEX for equities.  IEX is a
        single-exchange feed and can be sparse in pre/after-hours, while the
        historical endpoint may lag the wall clock.  The worker should evaluate
        the most recent usable bar, not require every symbol to have a print on
        the exact same wall-clock 5-minute slot.
        """
        cfg = getattr(self, "_runtime_config", {}) if isinstance(getattr(self, "_runtime_config", {}), dict) else {}
        configured = cfg.get("live_max_bar_age_minutes", cfg.get("max_bar_age_minutes"))
        default = 45 if (str(getattr(self.live, "feed", "iex")).lower() == "iex" and bool(getattr(self.live, "allow_extended_hours_entries", False))) else 15
        try:
            value = int(float(configured if configured not in (None, "") else default))
        except Exception:
            value = default
        return max(5, min(180, value))

    def _bar_age_minutes(self, ts: Any, now: datetime | None = None) -> float | None:
        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            else:
                t = t.tz_convert("UTC")
            ref = pd.Timestamp(now or _now_utc())
            if ref.tzinfo is None:
                ref = ref.tz_localize("UTC")
            else:
                ref = ref.tz_convert("UTC")
            return max(0.0, float((ref - t).total_seconds() / 60.0))
        except Exception:
            return None

    def _is_bar_recent_enough(self, ts: Any) -> bool:
        age = self._bar_age_minutes(ts)
        return age is not None and age <= float(self._max_bar_age_minutes())

    def _is_regular_session_time(self, ts: pd.Timestamp | datetime | None = None) -> bool:
        ts = pd.Timestamp(ts or datetime.now(NY))
        if ts.tzinfo is None:
            ts = ts.tz_localize(NY)
        else:
            ts = ts.tz_convert(NY)
        return ts.weekday() < 5 and _time_in_window(ts, "09:30", "16:00")

    def _is_extended_session_order_context(self, signal_time: Any | None = None) -> bool:
        if not bool(getattr(self.live, "allow_extended_hours_entries", False)):
            return False
        try:
            sig_ts = pd.Timestamp(signal_time) if signal_time is not None else pd.Timestamp.now(tz=NY)
            if sig_ts.tzinfo is None:
                sig_ts = sig_ts.tz_localize("UTC")
            return not self._is_regular_session_time(sig_ts)
        except Exception:
            return not self._is_regular_session_time()

    def _extended_limit_price(self, plan: dict[str, Any]) -> float:
        """Marketable protective limit for paper extended-hours entries.

        Alpaca only accepts extended-hours eligible equity orders as limit orders
        with day/gtc TIF.  For paper testing we use a small configurable buffer
        around the quote/reference price so the order behaves similarly to a
        marketable entry while still satisfying Alpaca's extended-hours rules.
        """
        ref = _safe_float(plan.get("entry_reference_price"), _safe_float(plan.get("signal_close"), 0.0))
        if ref <= 0:
            return 0.0
        cfg = getattr(self, "_runtime_config", {}) if isinstance(getattr(self, "_runtime_config", {}), dict) else {}
        bps = max(0.0, _safe_float(cfg.get("extended_limit_buffer_bps", cfg.get("slippage_bps", 5.0)), 5.0))
        side = str(plan.get("alpaca_side", "buy")).lower()
        price = ref * (1.0 + bps / 10000.0) if side == "buy" else ref * (1.0 - bps / 10000.0)
        return _round_price(price)

    def _regular_bracket_price_buffer(self, base_price: float, buffer_multiplier: float = 1.0) -> float:
        """Minimum separation used for Alpaca regular-session bracket legs.

        Alpaca validates bracket child orders against a moving internal base price.
        If a fast market moves a few cents between planning and submission, a
        short stop can be rejected with messages such as
        "stop_loss.stop_price must be >= base_price + 0.01".  This buffer is
        an execution guard only; position size is reduced below if widening the
        stop would otherwise increase dollar risk beyond the configured risk
        budget.
        """
        cfg = getattr(self, "_runtime_config", {}) if isinstance(getattr(self, "_runtime_config", {}), dict) else {}
        # Use the user-configured slippage bps as the primary live-order safety
        # buffer.  This is intentionally automatic: the user should not need a
        # second setting just to avoid Alpaca bracket validation rejects.
        bps = max(0.0, _safe_float(cfg.get("slippage_bps", 3.0), 3.0))
        multiplier = max(1.0, _safe_float(buffer_multiplier, 1.0))
        return max(0.02, float(base_price or 0.0) * bps * multiplier / 10000.0)

    def _prepare_regular_bracket_plan_for_submit(self, plan: dict[str, Any], current_reference_price: float | None = None, buffer_multiplier: float = 1.0) -> dict[str, Any] | None:
        """Return a submit-ready regular-market bracket plan.

        The strategy/risk model creates a target stop and quantity from the signal
        bar or quote reference.  Immediately before submitting a market bracket,
        refresh the base/reference price and make the child legs safely valid
        relative to that base.  If this widens risk/share, reduce quantity so the
        actual submitted risk remains inside the original risk budget and current
        affordability guard.
        """
        out = dict(plan or {})
        alpaca_side = str(out.get("alpaca_side", "")).lower().strip()
        strategy_side = str(out.get("strategy_side", "")).lower().strip()
        if alpaca_side not in {"buy", "sell"}:
            return None
        base = _safe_float(current_reference_price, 0.0)
        if base <= 0:
            base = _safe_float(out.get("entry_reference_price"), _safe_float(out.get("signal_close"), 0.0))
        if base <= 0:
            return None
        old_stop = _safe_float(out.get("stop_price"), 0.0)
        old_target = _safe_float(out.get("target_price"), 0.0)
        if old_stop <= 0 or old_target <= 0:
            return None
        buffer = self._regular_bracket_price_buffer(base, buffer_multiplier=buffer_multiplier)
        # Long entry = buy market bracket.  Stop must be below base, target above.
        # Short entry = sell market bracket. Stop must be above base, target below.
        if alpaca_side == "buy" or strategy_side == "long":
            min_stop = base - buffer
            min_target = base + buffer
            new_stop = min(old_stop, min_stop)
            new_target = max(old_target, min_target)
            actual_risk_per_share = max(0.0, base - new_stop)
        else:
            min_stop = base + buffer
            min_target = base - buffer
            new_stop = max(old_stop, min_stop)
            new_target = min(old_target, min_target)
            actual_risk_per_share = max(0.0, new_stop - base)
        if actual_risk_per_share <= 0:
            return None
        risk_budget = _safe_float(out.get("risk_budget"), 0.0)
        qty = _safe_float(out.get("qty"), 0.0)
        if risk_budget > 0:
            max_qty_by_risk = risk_budget / actual_risk_per_share
            if not self.risk.allow_fractional:
                max_qty_by_risk = math.floor(max_qty_by_risk)
            if max_qty_by_risk < qty:
                qty = max_qty_by_risk
        if qty <= 0:
            return None
        if not self.risk.allow_fractional:
            qty = math.floor(qty)
        if qty <= 0:
            return None
        old_qty = _safe_float(out.get("qty"), 0.0)
        changed = (abs(new_stop - old_stop) >= 0.005) or (abs(new_target - old_target) >= 0.005) or (abs(qty - old_qty) >= 1e-9)
        out.update({
            "qty": qty,
            "target_price": _round_price(new_target),
            "stop_price": _round_price(new_stop),
            "regular_bracket_base_reference_price": _round_price(base),
            "regular_bracket_price_buffer": _round_price(buffer),
            "regular_bracket_guard_adjusted": bool(changed),
            "regular_bracket_guard_note": "regular_session_bracket_validated_against_latest_reference",
            "actual_risk_dollars": float(qty) * float(actual_risk_per_share),
            "risk_budget_shortfall": max(0.0, risk_budget - float(qty) * float(actual_risk_per_share)) if risk_budget > 0 else _safe_float(out.get("risk_budget_shortfall"), 0.0),
            "estimated_notional": float(qty) * float(base),
        })
        return out


    @staticmethod
    def _retry_client_order_id(client_order_id: str, attempt: int) -> str:
        base = str(client_order_id or "rmv33-retry")
        suffix = f"-r{int(attempt)}"
        return (base[: max(1, 48 - len(suffix))] + suffix)[:48]

    @staticmethod
    def _parse_alpaca_error_payload(exc: Exception) -> dict[str, Any]:
        """Best-effort parse of Alpaca's JSON error body from our REST wrapper."""
        text = str(exc or "")
        # Typical wrapper text: Alpaca trading request failed 422: {"base_price":"521.2",...}
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
                return payload if isinstance(payload, dict) else {}
            except Exception:
                pass
        return {}

    @classmethod
    def _is_repairable_bracket_rejection(cls, exc: Exception) -> bool:
        payload = cls._parse_alpaca_error_payload(exc)
        message = str(payload.get("message") or exc or "").lower()
        code = str(payload.get("code") or "")
        if code == "42210000":
            return True
        return any(token in message for token in ("stop_loss", "take_profit", "stop_price", "limit_price", "base_price"))

    def _base_price_from_alpaca_error_or_quote(self, exc: Exception, symbol: str, fallback: float | None = None) -> float:
        payload = self._parse_alpaca_error_payload(exc)
        base = _safe_float(payload.get("base_price"), 0.0)
        if base > 0:
            return base
        try:
            latest = self._latest_reference_prices([symbol]).get(symbol)
            if latest and latest > 0:
                return float(latest)
        except Exception:
            pass
        return _safe_float(fallback, 0.0)

    def _submit_regular_market_bracket_with_auto_repair(self, plan: dict[str, Any], initial_reference_price: float | None = None) -> tuple[Any, dict[str, Any]]:
        """Submit a regular-session bracket, automatically repairing Alpaca price rejects.

        This keeps the strategy/risk model as the source of truth, but treats
        Alpaca 422 bracket-leg validation as an execution-time race condition:
        the order was planned from a signal/reference price, while Alpaca validates
        children against its current base price at submit time.  On repairable
        rejects we refresh/parse the base price, widen the child legs using the
        configured slippage bps, reduce qty if needed so dollar risk does not
        exceed the original budget, and retry with a fresh client_order_id.
        """
        symbol = str(plan.get("symbol") or "").upper()
        if not symbol:
            raise ValueError("Cannot submit bracket order without symbol.")
        attempts: list[dict[str, Any]] = []
        base_reference = _safe_float(initial_reference_price, _safe_float(plan.get("entry_reference_price"), 0.0))
        working_plan = dict(plan)
        base_client_order_id = str(plan.get("client_order_id") or "")
        last_exc: Exception | None = None
        # Attempt 1 uses the configured slippage buffer.  Repairs expand it
        # automatically (2x, then 4x) only if Alpaca rejects the child prices.
        for attempt in range(1, 4):
            submit_plan = self._prepare_regular_bracket_plan_for_submit(
                working_plan,
                base_reference,
                buffer_multiplier=(2 ** (attempt - 1)),
            )
            if not submit_plan:
                raise ValueError("Regular-session bracket order became invalid after latest-price validation; order was not submitted.")
            if attempt > 1:
                submit_plan["client_order_id"] = self._retry_client_order_id(base_client_order_id, attempt - 1)
            submit_plan["regular_bracket_submit_attempt"] = attempt
            submit_plan["regular_bracket_auto_repair_attempts"] = attempts
            try:
                result = self.trading_client.submit_market_bracket_order(
                    symbol=submit_plan["symbol"],
                    side=submit_plan["alpaca_side"],
                    qty=float(submit_plan["qty"]),
                    take_profit_price=float(submit_plan["target_price"]),
                    stop_price=float(submit_plan["stop_price"]),
                    client_order_id=submit_plan["client_order_id"],
                    fractional=self.risk.allow_fractional,
                )
                submit_plan["regular_bracket_auto_repaired"] = attempt > 1
                submit_plan["regular_bracket_auto_repair_attempts"] = attempts
                return result, submit_plan
            except Exception as exc:
                last_exc = exc
                attempts.append({
                    "attempt": attempt,
                    "client_order_id": submit_plan.get("client_order_id"),
                    "qty": submit_plan.get("qty"),
                    "stop_price": submit_plan.get("stop_price"),
                    "target_price": submit_plan.get("target_price"),
                    "base_reference": submit_plan.get("regular_bracket_base_reference_price"),
                    "buffer": submit_plan.get("regular_bracket_price_buffer"),
                    "error": str(exc)[:700],
                })
                if attempt >= 3 or not self._is_repairable_bracket_rejection(exc):
                    break
                base_reference = self._base_price_from_alpaca_error_or_quote(exc, symbol, fallback=base_reference)
                working_plan = submit_plan
                continue
        self._last_bracket_repair_attempts = attempts
        raise last_exc or AlpacaTradingAPIError("Regular-session bracket order failed after automatic repair attempts.")

    def _parse_strategy_client_order_id(self, client_order_id: str) -> dict[str, Any] | None:
        cid = str(client_order_id or "").strip()
        if not cid.startswith(ORDER_PREFIXES):
            return None
        # Expected V33 shape: rmv33-SYMBOL-YYYYMMDDHHMM-l/s. Keep the parser
        # tolerant so older V32/V33 test IDs do not break dashboard/state recovery.
        match = re.match(r"^(rmv\d+|rm\d+)-(?:(?P<strategy_code>[a-z0-9]{2,8})-)?(?P<symbol>[A-Z0-9.]+)-(?P<stamp>\d{12})-(?P<side>[ls])", cid, flags=re.IGNORECASE)
        if not match:
            return {"client_order_id": cid}
        symbol = match.group("symbol")
        stamp = match.group("stamp")
        side_code = match.group("side")
        strategy_code = match.group("strategy_code") or ""
        try:
            ts_et = datetime.strptime(stamp, "%Y%m%d%H%M").replace(tzinfo=NY)
            session_date = ts_et.date().isoformat()
            signal_time_et = ts_et.strftime("%Y-%m-%d %H:%M")
        except Exception:
            session_date = ""
            signal_time_et = ""
        return {
            "client_order_id": cid,
            "strategy_code": strategy_code,
            "symbol": symbol.upper(),
            "session_date": session_date,
            "signal_time_et": signal_time_et,
            "strategy_side": "long" if side_code.lower() == "l" else "short",
        }

    def _strategy_order_records_from_orders(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}

        def visit(order: dict[str, Any]) -> None:
            parsed = self._parse_strategy_client_order_id(str(order.get("client_order_id", "")))
            if parsed:
                cid = parsed.get("client_order_id", "")
                if cid:
                    rec = dict(parsed)
                    rec.update({
                        "order_id": order.get("id", ""),
                        "status": order.get("status", ""),
                        "submitted_at": order.get("submitted_at") or order.get("created_at") or "",
                        "filled_at": order.get("filled_at") or "",
                    })
                    if not rec.get("symbol"):
                        rec["symbol"] = str(order.get("symbol", "")).upper()
                    if rec.get("strategy_code"):
                        rec.setdefault("strategy_code", rec.get("strategy_code"))
                        variant = _strategy_variant_from_code(rec.get("strategy_code"))
                        if variant:
                            rec.setdefault("strategy_variant", variant)
                            spec = _spec_for_preset_or_variant(variant=variant)
                            rec.setdefault("strategy_preset", spec.get("preset"))
                            rec.setdefault("quality_gate", spec.get("quality_gate"))
                    records[str(cid)] = rec
            for leg in order.get("legs", []) or []:
                if isinstance(leg, dict):
                    visit(leg)

        for order in orders or []:
            if isinstance(order, dict):
                visit(order)
        return list(records.values())

    def _merge_strategy_order_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for rec in self.state.get("submitted_strategy_order_records", []) or []:
            cid = str(rec.get("client_order_id", "")) if isinstance(rec, dict) else ""
            if cid:
                merged[cid] = dict(rec)
        for rec in records or []:
            cid = str(rec.get("client_order_id", "")) if isinstance(rec, dict) else ""
            if cid:
                old = merged.get(cid, {})
                old.update({k: v for k, v in rec.items() if v not in (None, "")})
                merged[cid] = old
        out = list(merged.values())
        out.sort(key=lambda r: str(r.get("submitted_at") or r.get("signal_time_et") or ""))
        return out[-1000:]

    def _sync_account_state(self) -> list[dict[str, Any]]:
        positions: list[dict[str, Any]] = []
        try:
            positions = self.trading_client.list_positions()
            current_symbols = {str(p.get("symbol", "")).upper() for p in positions if p.get("symbol")}
            for pos in positions:
                self.store.upsert_position(pos)
            self.store.mark_missing_positions_closed(current_symbols)
        except Exception as exc:
            self.store.insert_event("position_sync_error", {"message": str(exc)}, status="error")
        try:
            after = (_now_utc() - timedelta(days=5)).isoformat()
            orders = self.trading_client.list_orders(status="all", symbols=None, limit=500, nested=True, after=after, direction="desc")
            self.store.upsert_orders_recursive(orders)
            records = self._strategy_order_records_from_orders(orders)
            if records:
                self.state["submitted_strategy_order_records"] = self._merge_strategy_order_records(records)
        except Exception as exc:
            self.store.insert_event("order_sync_error", {"message": str(exc)}, status="error")
        return positions

    def _existing_symbols(self) -> set[str]:
        symbols: set[str] = set()
        try:
            for pos in self.trading_client.list_positions():
                sym = str(pos.get("symbol", "")).upper()
                if sym:
                    symbols.add(sym)
        except Exception:
            pass
        try:
            for order in self.trading_client.list_open_orders(self.live.symbols or []):
                sym = str(order.get("symbol", "")).upper()
                if sym:
                    symbols.add(sym)
        except Exception:
            pass
        return symbols

    def _daily_order_count(self, session_date: str) -> int:
        key_count = len({str(k) for k in self.state.get("submitted_signal_keys", []) if f"|{session_date}|" in str(k)})
        order_count = len({
            str(r.get("client_order_id", ""))
            for r in self.state.get("submitted_strategy_order_records", []) or []
            if isinstance(r, dict) and str(r.get("session_date", "")) == str(session_date) and str(r.get("client_order_id", ""))
        })
        return max(key_count, order_count)

    def _symbol_daily_count(self, symbol: str, session_date: str) -> int:
        key_count = len({
            str(k) for k in self.state.get("submitted_signal_keys", [])
            if f"|{symbol.upper()}|{session_date}|" in str(k) or str(k).startswith(f"{symbol.upper()}|{session_date}|")
        })
        order_count = len({
            str(r.get("client_order_id", ""))
            for r in self.state.get("submitted_strategy_order_records", []) or []
            if isinstance(r, dict)
            and str(r.get("symbol", "")).upper() == symbol.upper()
            and str(r.get("session_date", "")) == str(session_date)
            and str(r.get("client_order_id", ""))
        })
        return max(key_count, order_count)

    def _update_bar_cache(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        now = _now_utc()
        configured_feed = str(getattr(self.live, "feed", "iex") or "iex").lower().strip()
        session_mode = self._live_session_mode()
        primary_api_feed = self._live_realtime_feed()
        delayed_sip_mode = configured_feed in {"delayed_sip", "free_delayed_sip", "sip_delayed"}
        end_dt = now + timedelta(minutes=5)
        keep_start = now - timedelta(days=max(35, int(self.live.lookback_days)))
        symbols = list(dict.fromkeys([s.upper() for s in (self.live.symbols or []) if s]))
        all_symbols = list(dict.fromkeys(["QQQ"] + symbols))
        if self._bars_5m_cache.empty:
            fetch_start_5m = keep_start
        else:
            latest = pd.to_datetime(self._bars_5m_cache["timestamp"], utc=True).max().to_pydatetime()
            incremental_start = now - timedelta(days=max(1, int(self.live.incremental_fetch_days)))
            fetch_start_5m = max(incremental_start, latest - timedelta(minutes=30))
        fetch_start_daily = now - timedelta(days=max(45, int(self.live.lookback_days) + 20))

        try:
            self.data_client.last_request_errors = []
        except Exception:
            pass
        bars_5m_new = self.data_client.get_stock_bars(all_symbols, "5Min", fetch_start_5m, end_dt, feed=primary_api_feed, adjustment="split", use_cache=False, session_mode=session_mode)
        daily_new = self.data_client.get_stock_bars(all_symbols, "1Day", fetch_start_daily, end_dt, feed=primary_api_feed, adjustment="split", use_cache=False, session_mode=session_mode)
        effective_feed = primary_api_feed
        fallback_rows = 0
        fallback_daily_rows = 0
        fallback_reason = ""
        if delayed_sip_mode:
            fallback_reason = "Saved feed was delayed_sip, but live order decisions are real-time only; using IEX for live scans. Use debug endpoints/reports for delayed research, not live entries."

        try:
            latest_by_symbol = {}
            stale_symbols = []
            missing_symbols = []
            if bars_5m_new is not None and not bars_5m_new.empty:
                tmp = bars_5m_new.copy()
                tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True, errors="coerce")
                tmp = tmp.dropna(subset=["timestamp"])
                latest_series = tmp.groupby("symbol")["timestamp"].max()
                latest_by_symbol = {str(sym): ts.isoformat() for sym, ts in latest_series.items()}
                max_age = float(self._max_bar_age_minutes())
                for sym in all_symbols:
                    ts = latest_series.get(sym)
                    if ts is None or pd.isna(ts):
                        missing_symbols.append(sym)
                    else:
                        age = self._bar_age_minutes(ts, now=now)
                        if age is None or age > max_age:
                            stale_symbols.append({"symbol": sym, "age_minutes": age})
            else:
                missing_symbols = all_symbols
            self.store.set_state("last_bar_fetch", {
                "updated_at_utc": utc_now_iso(),
                "session_mode": session_mode,
                "configured_feed": configured_feed,
                "primary_api_feed": primary_api_feed,
                "effective_feed": effective_feed,
                "feed": effective_feed,
                "live_data_policy": "real_time_only_for_order_decisions",
                "delayed_data_used_for_orders": False,
                "symbols_requested": len(all_symbols),
                "bars_5m_rows": int(0 if bars_5m_new is None else len(bars_5m_new)),
                "daily_rows": int(0 if daily_new is None else len(daily_new)),
                "fallback_sip_16m_rows": fallback_rows,
                "fallback_sip_16m_daily_rows": fallback_daily_rows,
                "fallback_note": fallback_reason,
                "latest_5m_by_symbol": latest_by_symbol,
                "missing_symbols": missing_symbols[:50],
                "stale_symbols": stale_symbols[:50],
                "max_bar_age_minutes": self._max_bar_age_minutes(),
                "api_errors": getattr(self.data_client, "last_request_errors", [])[-20:],
                "free_plan_note": "Alpaca Basic/free historical bars can use IEX without subscription; SIP is broad-market and may require paid real-time entitlement unless queried with enough delay. IEX is single-exchange and can be sparse in extended hours.",
            })
        except Exception:
            pass
        self._bars_5m_cache = _merge_bar_cache(self._bars_5m_cache, bars_5m_new, keep_start)
        self._daily_cache = _merge_bar_cache(self._daily_cache, daily_new, fetch_start_daily)
        return self._bars_5m_cache.copy(), self._daily_cache.copy()

    def _add_live_v25_filter_aliases(self, alerts: pd.DataFrame) -> pd.DataFrame:
        out = alerts.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out["date"] = out["timestamp"].dt.tz_convert("America/New_York").dt.date.astype(str)
        out["score"] = pd.to_numeric(out.get("candidate_score", 0.0), errors="coerce").fillna(-9999.0)
        out["qqq_chg_open"] = pd.to_numeric(out.get("qqq_change_from_open", out.get("qqq_day_change_percent", 0.0)), errors="coerce").fillna(0.0)
        out["gap_pct"] = pd.to_numeric(out.get("gap_percent", 0.0), errors="coerce").fillna(0.0)
        out["rvol_tod"] = pd.to_numeric(out.get("rvol_time_of_day", 0.0), errors="coerce").fillna(0.0)
        for col in [
            "bullish_continuation_candle",
            "bullish_rejection_candle",
            "bullish_engulfing_candle",
            "bearish_continuation_candle",
            "bearish_rejection_candle",
            "bearish_engulfing_candle",
        ]:
            if col not in out.columns:
                out[col] = False
            out[col] = out[col].fillna(False).astype(bool)
        long_mask = out.get("side", "").astype(str).str.lower().eq("long")
        short_mask = out.get("side", "").astype(str).str.lower().eq("short")
        out["side_continuation_candle"] = False
        out["side_rejection_candle"] = False
        out["side_engulfing_candle"] = False
        out["opposing_candle_warning"] = False
        out.loc[long_mask, "side_continuation_candle"] = out.loc[long_mask, "bullish_continuation_candle"]
        out.loc[long_mask, "side_rejection_candle"] = out.loc[long_mask, "bullish_rejection_candle"]
        out.loc[long_mask, "side_engulfing_candle"] = out.loc[long_mask, "bullish_engulfing_candle"]
        out.loc[long_mask, "opposing_candle_warning"] = (
            out.loc[long_mask, "bearish_continuation_candle"] | out.loc[long_mask, "bearish_rejection_candle"] | out.loc[long_mask, "bearish_engulfing_candle"]
        )
        out.loc[short_mask, "side_continuation_candle"] = out.loc[short_mask, "bearish_continuation_candle"]
        out.loc[short_mask, "side_rejection_candle"] = out.loc[short_mask, "bearish_rejection_candle"]
        out.loc[short_mask, "side_engulfing_candle"] = out.loc[short_mask, "bearish_engulfing_candle"]
        out.loc[short_mask, "opposing_candle_warning"] = (
            out.loc[short_mask, "bullish_continuation_candle"] | out.loc[short_mask, "bullish_rejection_candle"] | out.loc[short_mask, "bullish_engulfing_candle"]
        )
        out["entry_candle_ok"] = out["side_continuation_candle"] | out["side_rejection_candle"] | out["side_engulfing_candle"]
        out["candle_pattern_score"] = (
            out["side_continuation_candle"].astype(int) + out["side_rejection_candle"].astype(int) + out["side_engulfing_candle"].astype(int) - out["opposing_candle_warning"].astype(int)
        )
        return out

    def _candidate_audit_key(self, row: pd.Series | dict[str, Any]) -> str:
        get = row.get if isinstance(row, dict) else row.get
        ts = pd.Timestamp(get("timestamp", get("candidate_time_utc", pd.Timestamp.now(tz="UTC"))))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        variant = str(get("strategy_variant", getattr(self.params, "live_strategy_variant", getattr(self.live, "strategy_variant", ""))) or "").lower()
        return "|".join([
            str(ts.isoformat()),
            str(get("symbol", "")).upper(),
            str(get("side", get("strategy_side", ""))).lower(),
            str(get("trigger_type", "")),
            variant,
        ])

    def _audit_record_from_signal(self, signal: pd.Series | dict[str, Any], run_id: str, status: str, reason: str = "", stage: str = "candidate", rank_before: int | None = None, rank_after: int | None = None, plan: dict[str, Any] | None = None) -> dict[str, Any]:
        data = signal.to_dict() if hasattr(signal, "to_dict") else dict(signal or {})
        ts = pd.Timestamp(data.get("timestamp", data.get("candidate_time_utc", pd.Timestamp.now(tz="UTC"))))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        ts_et = ts.tz_convert(NY)
        plan = plan or {}
        raw_code = data.get("strategy_code") or plan.get("strategy_code") or ""
        raw_variant = data.get("strategy_variant") or plan.get("strategy_variant") or _strategy_variant_from_code(raw_code) or getattr(self.params, "live_strategy_variant", getattr(self.live, "strategy_variant", "best_report_153601"))
        raw_preset = data.get("strategy_preset") or plan.get("strategy_preset") or getattr(self.params, "live_strategy_preset", "env")
        raw_gate = data.get("quality_gate") or plan.get("quality_gate") or getattr(self.params, "live_quality_gate", "env")
        spec = _strategy_spec_from_signal_values(raw_variant, raw_preset, raw_gate, raw_code)
        strategy_variant = str(raw_variant or spec.get("variant") or "best_report_153601").strip().lower()
        strategy_code = str(raw_code or spec.get("code") or _strategy_code(strategy_variant)).strip().lower()
        strategy_preset = str(raw_preset or spec.get("preset") or "manual")
        quality_gate = str(raw_gate or spec.get("quality_gate") or "off")
        ctx_payload = self._signal_context_payload(pd.Series(data)) if not isinstance(signal, pd.Series) else self._signal_context_payload(signal)
        record = {
            "audit_key": self._candidate_audit_key(data),
            "run_id": run_id,
            "candidate_time_utc": ts.isoformat(),
            "candidate_time_et": ts_et.strftime("%Y-%m-%d %H:%M"),
            "session_date": ts_et.date().isoformat(),
            "symbol": str(data.get("symbol", "")).upper(),
            "strategy_side": str(data.get("side", data.get("strategy_side", ""))).lower(),
            "trigger_type": str(data.get("trigger_type", "")),
            "strategy_variant": strategy_variant,
            "strategy_code": strategy_code,
            "strategy_preset": strategy_preset,
            "strategy_profile": str(data.get("strategy_profile") or plan.get("strategy_profile") or getattr(self.params, "strategy_profile", "symbol_playbook_v25")),
            "quality_gate": quality_gate,
            "pattern_mode": str(data.get("pattern_mode") or plan.get("pattern_mode") or getattr(self.params, "v379_pattern_mode", "")),
            "selection_mode": str(getattr(self.live, "selection_mode", "seen_so_far_top_n")),
            "audit_stage": stage,
            "decision_status": status,
            "reject_reason": reason,
            "rank_before_filter": rank_before,
            "rank_after_filter": rank_after,
            "candidate_score": _safe_float(data.get("candidate_score", data.get("score", 0.0)), 0.0),
            "final_rank_score": _safe_float(data.get("score", data.get("candidate_score", 0.0)), 0.0),
            "entry_reference_price": (plan or {}).get("entry_reference_price"),
            "stop_price": (plan or {}).get("stop_price"),
            "target_price": (plan or {}).get("target_price"),
            "risk_budget": (plan or {}).get("risk_budget"),
            "rvol_time_of_day": _safe_float(data.get("rvol_time_of_day", data.get("rvol_tod", 0.0)), 0.0),
            "daily_atr14_percent": _safe_float(data.get("daily_atr14_percent", 0.0), 0.0),
            "gap_percent": _safe_float(data.get("gap_percent", data.get("gap_pct", 0.0)), 0.0),
            "day_relative_strength": _safe_float(data.get("day_relative_strength", 0.0), 0.0),
            "open_relative_strength": _safe_float(data.get("open_relative_strength", 0.0), 0.0),
            "vwap_extension_atr": _safe_float(data.get("vwap_extension_atr", 0.0), 0.0),
            "qqq_change_from_open": _safe_float(data.get("qqq_change_from_open", data.get("qqq_chg_open", 0.0)), 0.0),
            "qqq_day_change_percent": _safe_float(data.get("qqq_day_change_percent", 0.0), 0.0),
            "atr5m14": _safe_float(data.get("atr5m14", 0.0), 0.0),
            "entry_candle_pattern": data.get("entry_candle_pattern") or data.get("candle_pattern_primary"),
            "candle_pattern_score": _safe_float(data.get("candle_pattern_score", 0.0), 0.0),
            "payload": {"signal_context": ctx_payload, "plan": plan or {}},
        }
        return record

    def _audit_candidates(self, df: pd.DataFrame, run_id: str, status: str, reason: str = "", stage: str = "candidate", rank_col: str | None = None) -> None:
        if df is None or df.empty:
            return
        records = []
        ranked = df.copy()
        if rank_col and rank_col in ranked.columns:
            ranked = ranked.sort_values(rank_col, ascending=False).reset_index(drop=True)
        for idx, row in ranked.iterrows():
            records.append(self._audit_record_from_signal(row, run_id, status, reason=reason, stage=stage, rank_before=int(idx) + 1))
        self.store.upsert_candidate_audits(records)


    def _monitor_gate_prefix(self) -> str:
        if bool(getattr(self.params, "enable_v358_live_quality_filter", False)):
            return "v358"
        if bool(getattr(self.params, "enable_v359_live_hunter_filter", False)):
            return "v359"
        if bool(getattr(self.params, "enable_v364_professional_momentum_filter", False)):
            return "v364"
        return ""

    def _monitor_quality_gate_label(self) -> str:
        return str(getattr(self.params, "live_quality_gate", "off") or "off")

    def _monitor_bool(self, value: Any, default: bool = False) -> bool:
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return default
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)
        except Exception:
            return default

    def _monitor_num(self, row: pd.Series | dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            if isinstance(row, pd.Series):
                return _safe_float(row.get(key), default)
            return _safe_float(row.get(key), default)
        except Exception:
            return default

    def _monitor_text(self, row: pd.Series | dict[str, Any], key: str, default: str = "") -> str:
        try:
            value = row.get(key) if isinstance(row, (pd.Series, dict)) else ""
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return default
            return str(value)
        except Exception:
            return default

    def _monitor_check(self, name: str, value: Any, threshold: str, passed: bool | None, weight: str = "") -> dict[str, Any]:
        return {"name": name, "value": value, "threshold": threshold, "passed": passed, "weight": weight}

    def _symbol_monitor_base_config(self, symbol: str, run_id: str, status: str, reason: str = "") -> dict[str, Any]:
        now = utc_now_iso()
        return {
            "symbol": str(symbol or "").upper(),
            "run_id": run_id,
            "updated_at_utc": now,
            "strategy_variant": getattr(self.params, "live_strategy_variant", getattr(self.live, "strategy_variant", "best_report_153601")),
            "strategy_code": getattr(self.params, "live_strategy_code", _strategy_code(getattr(self.params, "live_strategy_variant", getattr(self.live, "strategy_variant", "best_report_153601")))),
            "strategy_label": getattr(self.params, "live_strategy_label", getattr(self.params, "live_strategy_preset", "env")),
            "strategy_preset": getattr(self.params, "live_strategy_preset", "env"),
            "strategy_profile": str(getattr(self.params, "strategy_profile", "symbol_playbook_v25")),
            "quality_gate": self._monitor_quality_gate_label(),
            "pattern_mode": str(getattr(self.params, "v379_pattern_mode", "")),
            "selection_mode": str(getattr(self.live, "selection_mode", "seen_so_far_top_n")),
            "feed": str(getattr(self.live, "feed", "iex")),
            "monitor_status": status,
            "decision_status": "idle" if status.lower().startswith("paused") else "waiting",
            "reject_reason": reason,
            "setup_signal": 0,
            "selected_signal": 0,
            "in_entry_window": 0,
            "checks_passed": 0,
            "checks_total": 0,
            "check_summary": reason or status,
            "payload": {"reason": reason, "symbols_configured": len(self.live.symbols or []), "bar_session_mode": self._live_session_mode()},
        }

    def _idle_symbol_monitor_records(self, run_id: str, status: str, reason: str) -> list[dict[str, Any]]:
        symbols = list(dict.fromkeys([s.upper() for s in (self.live.symbols or []) if s]))
        records: list[dict[str, Any]] = []
        original_params = self.params
        try:
            for spec in self._strategy_specs_for_current_run():
                self.params = self._make_params_for_live_config(getattr(self, "_runtime_config", {}), spec.get("preset"), spec.get("variant"), spec.get("quality_gate"))
                records.extend([self._symbol_monitor_base_config(sym, run_id, status, reason) for sym in symbols])
        finally:
            self.params = original_params
        return records

    def _seed_full_strategy_symbol_monitor(self, run_id: str, closed_ts: pd.Timestamp, specs: list[dict[str, str]], symbols: list[str], reason: str = "Queued for this worker scan.") -> None:
        """Publish a complete strategy x symbol scaffold before expensive scans run.

        Without this, all-strategies mode can look broken in the dashboard while
        a scan is still running: the worker writes strategy monitor rows one
        strategy at a time, and the reader naturally sees only the first 1-2
        strategies until the scan finishes.  This scaffold makes the monitor
        table immediately contain the expected active_strategy_count * symbols
        rows for the current run_id; each strategy then replaces its own rows
        with real indicator/check values as it completes.
        """
        symbols = list(dict.fromkeys([str(s or "").upper() for s in (symbols or []) if s]))
        if not specs or not symbols:
            return
        original_params = self.params
        rows: list[dict[str, Any]] = []
        try:
            for spec in specs:
                self.params = self._make_params_for_live_config(getattr(self, "_runtime_config", {}), spec.get("preset"), spec.get("variant"), spec.get("quality_gate"))
                for sym in symbols:
                    rec = self._symbol_monitor_base_config(sym, run_id, "Scanning - queued", reason)
                    rec["latest_bar_time_utc"] = pd.Timestamp(closed_ts).isoformat()
                    rec["latest_bar_time_et"] = pd.Timestamp(closed_ts).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")
                    rec["decision_status"] = "waiting"
                    rec["check_summary"] = reason
                    rec["payload"] = {
                        "reason": reason,
                        "scaffold": True,
                        "strategy_run_mode": getattr(self.live, "strategy_run_mode", "single"),
                        "expected_strategy_count": len(specs),
                        "expected_symbol_count": len(symbols),
                        "bar_session_mode": self._live_session_mode(),
                    }
                    rows.append(rec)
        finally:
            self.params = original_params
        self._write_symbol_monitor_records(rows)

    def _monitor_record_from_latest_row(self, row: pd.Series, run_id: str, closed_ts: pd.Timestamp) -> dict[str, Any]:
        symbol = self._monitor_text(row, "symbol").upper()
        ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp(closed_ts)
        ts_et = ts.tz_convert("America/New_York")
        side = self._monitor_text(row, "side").lower()
        side_short = side == "short"
        score = self._monitor_num(row, "score", self._monitor_num(row, "candidate_score", 0.0))
        candidate_score = self._monitor_num(row, "candidate_score", score)
        close_px = self._monitor_num(row, "close", 0.0)
        rvol = self._monitor_num(row, "rvol_time_of_day", self._monitor_num(row, "rvol_tod", 0.0))
        daily_atr = self._monitor_num(row, "daily_atr14_percent", 0.0)
        gap_pct = self._monitor_num(row, "gap_percent", self._monitor_num(row, "gap_pct", 0.0))
        rs = self._monitor_num(row, "day_relative_strength", 0.0)
        ors = self._monitor_num(row, "open_relative_strength", 0.0)
        vwap = self._monitor_num(row, "vwap_extension_atr", 0.0)
        dir_rs = -rs if side_short else rs
        dir_ors = -ors if side_short else ors
        dir_vwap = -vwap if side_short else vwap
        abs_vwap = abs(vwap)
        trigger = self._monitor_text(row, "trigger_type")
        setup_signal = bool(trigger) or self._monitor_bool(row.get("buy_alert"), False)
        raw_alert = self._monitor_bool(row.get("buy_alert"), False)
        in_window = _time_in_window(ts_et, getattr(self.live, "entry_start_time_et", "09:35"), getattr(self.live, "entry_end_time_et", "15:55"))
        if not bool(getattr(self.live, "allow_extended_hours_entries", False)):
            in_window = in_window and _time_in_window(ts_et, "09:30", "16:00")

        liquidity_ok = self._monitor_bool(row.get("liquidity_filter"), True)
        if "liquidity_filter" not in row.index:
            avg20 = self._monitor_num(row, "avg_20d_dollar_volume", 0.0)
            cur5 = self._monitor_num(row, "current_5m_dollar_volume", 0.0)
            atr5 = self._monitor_num(row, "atr5m14", 0.0)
            liquidity_ok = (
                close_px >= float(getattr(self.params, "min_price", 0.0) or 0.0)
                and avg20 >= float(getattr(self.params, "min_avg_20d_dollar_volume", 0.0) or 0.0)
                and cur5 >= float(getattr(self.params, "min_current_5m_dollar_volume", 0.0) or 0.0)
                and daily_atr >= float(getattr(self.params, "min_daily_atr_pct", 0.0) or 0.0)
                and daily_atr <= float(getattr(self.params, "max_daily_atr_pct", 999.0) or 999.0)
                and atr5 > 0
            )
        min_score = float(getattr(self.params, "min_candidate_score", 2.0) or 2.0)
        score_ok = score >= min_score if setup_signal else False
        candle_mode = str(getattr(self.params, "candle_pattern_mode", "selective") or "selective").lower()
        if candle_mode == "off" or not setup_signal:
            candle_ok = True
        else:
            candle_ok = self._monitor_bool(row.get("entry_candle_ok"), False)

        gate = self._monitor_quality_gate_label()
        prefix = self._monitor_gate_prefix()
        quality_gate_ok = True
        if prefix:
            output_col = {"v358": "v358_live_quality_ok", "v359": "v359_live_hunter_ok", "v364": "v364_professional_momentum_ok"}.get(prefix, "")
            quality_gate_ok = self._monitor_bool(row.get(output_col), False) if output_col else True
        elif gate == "v377":
            quality_gate_ok = self._monitor_bool(row.get("positive_context_profile_match"), False) if setup_signal else False
        elif gate.startswith("v379") or gate.startswith("v38"):
            quality_gate_ok = self._monitor_bool(row.get("v379_pattern_match"), False) if setup_signal else False

        def rng(prefix_name: str, suffix: str, default: float) -> float:
            return float(getattr(self.params, f"{prefix_name}_{suffix}", default) or default)

        age_min = self._bar_age_minutes(ts)
        max_age_min = self._max_bar_age_minutes()
        data_recent = bool(age_min is not None and age_min <= max_age_min)
        checks: list[dict[str, Any]] = []
        checks.append(self._monitor_check("Data recency", "unknown" if age_min is None else round(age_min, 1), f"<= {max_age_min} min", data_recent, "data"))
        checks.append(self._monitor_check("Setup", trigger or "none", "latest bar has strategy setup", setup_signal, "required"))
        checks.append(self._monitor_check("Entry window", ts_et.strftime("%H:%M"), f"{self.live.entry_start_time_et}-{self.live.entry_end_time_et} ET", in_window, "required"))
        checks.append(self._monitor_check("Liquidity", "ok" if liquidity_ok else "blocked", "price/volume/ATR floor", liquidity_ok, "required"))
        checks.append(self._monitor_check("Candidate score", round(score, 3), f">= {min_score:g}", score_ok if setup_signal else None, "ranking"))
        checks.append(self._monitor_check("RVOL", round(rvol, 3), f">= {float(getattr(self.params, 'v25_min_rvol', 0.75) or 0.75):g}", rvol >= float(getattr(self.params, "v25_min_rvol", 0.75) or 0.75), "setup"))
        checks.append(self._monitor_check("Daily ATR %", round(daily_atr, 3), f"{float(getattr(self.params, 'min_daily_atr_pct', 0.0) or 0.0):g}..{float(getattr(self.params, 'max_daily_atr_pct', 999.0) or 999.0):g}", daily_atr >= float(getattr(self.params, "min_daily_atr_pct", 0.0) or 0.0) and daily_atr <= float(getattr(self.params, "max_daily_atr_pct", 999.0) or 999.0), "setup"))
        if prefix:
            checks.append(self._monitor_check("Gate RVOL", round(rvol, 3), f">= {rng(prefix, 'min_rvol', 1.0):g}", rvol >= rng(prefix, "min_rvol", 1.0), "gate"))
            checks.append(self._monitor_check("Gate daily ATR %", round(daily_atr, 3), f">= {rng(prefix, 'min_daily_atr_pct', 0.0):g}", daily_atr >= rng(prefix, "min_daily_atr_pct", 0.0), "gate"))
            checks.append(self._monitor_check("Directional RS", round(dir_rs, 3), f"{rng(prefix, 'min_directional_rs', -999.0):g}..{rng(prefix, 'max_directional_rs', 999.0):g}", dir_rs >= rng(prefix, "min_directional_rs", -999.0) and dir_rs <= rng(prefix, "max_directional_rs", 999.0), "gate"))
            checks.append(self._monitor_check("Directional open RS", round(dir_ors, 3), f"{rng(prefix, 'min_directional_open_rs', -999.0):g}..{rng(prefix, 'max_directional_open_rs', 999.0):g}", dir_ors >= rng(prefix, "min_directional_open_rs", -999.0) and dir_ors <= rng(prefix, "max_directional_open_rs", 999.0), "gate"))
            checks.append(self._monitor_check("Directional VWAP ATR", round(dir_vwap, 3), f"{rng(prefix, 'min_directional_vwap_extension_atr', -999.0):g}..{rng(prefix, 'max_directional_vwap_extension_atr', 999.0):g}", dir_vwap >= rng(prefix, "min_directional_vwap_extension_atr", -999.0) and dir_vwap <= rng(prefix, "max_directional_vwap_extension_atr", 999.0), "gate"))
            checks.append(self._monitor_check("Abs VWAP ATR", round(abs_vwap, 3), f"<= {rng(prefix, 'max_abs_vwap_extension_atr', 999.0):g}", abs_vwap <= rng(prefix, "max_abs_vwap_extension_atr", 999.0), "gate"))
        if gate == "v377":
            checks.append(self._monitor_check("Positive profile", self._monitor_text(row, "positive_context_profile_name", "none"), "must match mined profile", quality_gate_ok, "gate"))
        elif gate.startswith("v379") or gate.startswith("v38"):
            checks.append(self._monitor_check("Pattern match", self._monitor_text(row, "v379_pattern_mode", gate), "must match decision pattern", quality_gate_ok, "gate"))
        checks.append(self._monitor_check("Quality gate", gate, "selected gate passes", quality_gate_ok if setup_signal else None, "gate"))
        checks.append(self._monitor_check("Candle", candle_mode, "candle filter passes", candle_ok if setup_signal else None, "gate"))

        counted = [c for c in checks if c.get("passed") is not None]
        passed = sum(1 for c in counted if bool(c.get("passed")))
        total = len(counted)
        if not in_window:
            status = "Paused - outside entry window"
            decision = "idle"
            reason = f"Dashboard entry window is {self.live.entry_start_time_et}-{self.live.entry_end_time_et} ET."
        elif not setup_signal:
            status = "Watching - no setup"
            decision = "watching"
            reason = "Latest closed bar has no strategy setup."
        elif not liquidity_ok:
            status = "Blocked - liquidity"
            decision = "rejected"
            reason = "Price, volume, ATR, or dollar-volume floor failed."
        elif not score_ok:
            status = "Blocked - score"
            decision = "rejected"
            reason = f"Score {score:.2f} below minimum {min_score:g}."
        elif not quality_gate_ok:
            status = "Blocked - quality gate"
            decision = "rejected"
            reason = f"{gate} gate failed."
        elif not candle_ok:
            status = "Blocked - candle"
            decision = "rejected"
            reason = "Candlestick filter failed."
        elif raw_alert:
            status = "Candidate - passed latest-bar checks"
            decision = "candidate"
            reason = "Latest closed bar is eligible before top-N/capacity selection."
        else:
            status = "Watching - no setup"
            decision = "watching"
            reason = "Latest closed bar did not remain an active setup after strategy filters."

        bias = "long-bias" if rs > 0.15 else ("short-bias" if rs < -0.15 else "neutral")
        summary = f"{passed}/{total} checks passed"
        if reason:
            summary = f"{summary}; {reason}"
        return {
            "symbol": symbol,
            "run_id": run_id,
            "updated_at_utc": utc_now_iso(),
            "latest_bar_time_utc": ts.isoformat(),
            "latest_bar_time_et": ts_et.strftime("%Y-%m-%d %H:%M"),
            "session_date": ts_et.date().isoformat(),
            "strategy_variant": getattr(self.params, "live_strategy_variant", getattr(self.live, "strategy_variant", "best_report_153601")),
            "strategy_code": getattr(self.params, "live_strategy_code", _strategy_code(getattr(self.params, "live_strategy_variant", getattr(self.live, "strategy_variant", "best_report_153601")))),
            "strategy_label": getattr(self.params, "live_strategy_label", getattr(self.params, "live_strategy_preset", "env")),
            "strategy_preset": getattr(self.params, "live_strategy_preset", "env"),
            "strategy_profile": str(getattr(self.params, "strategy_profile", "symbol_playbook_v25")),
            "quality_gate": gate,
            "pattern_mode": str(getattr(self.params, "v379_pattern_mode", "")),
            "selection_mode": str(getattr(self.live, "selection_mode", "seen_so_far_top_n")),
            "feed": str(getattr(self.live, "feed", "iex")),
            "monitor_status": status,
            "decision_status": decision,
            "reject_reason": reason,
            "setup_signal": setup_signal,
            "selected_signal": False,
            "in_entry_window": in_window,
            "liquidity_ok": liquidity_ok,
            "score_ok": score_ok if setup_signal else False,
            "quality_gate_ok": quality_gate_ok if setup_signal else False,
            "candle_ok": candle_ok if setup_signal else False,
            "checks_passed": passed,
            "checks_total": total,
            "check_summary": summary,
            "symbol_side_bias": bias,
            "strategy_side": side,
            "trigger_type": trigger,
            "candidate_score": candidate_score,
            "final_rank_score": score,
            "close_price": close_px,
            "volume": self._monitor_num(row, "volume", 0.0),
            "rvol_time_of_day": rvol,
            "daily_atr14_percent": daily_atr,
            "gap_percent": gap_pct,
            "day_relative_strength": rs,
            "open_relative_strength": ors,
            "vwap_extension_atr": vwap,
            "qqq_change_from_open": self._monitor_num(row, "qqq_change_from_open", self._monitor_num(row, "qqq_chg_open", 0.0)),
            "qqq_day_change_percent": self._monitor_num(row, "qqq_day_change_percent", 0.0),
            "atr5m14": self._monitor_num(row, "atr5m14", 0.0),
            "ema9": self._monitor_num(row, "ema9", 0.0),
            "ema20": self._monitor_num(row, "ema20", 0.0),
            "session_vwap": self._monitor_num(row, "session_vwap", 0.0),
            "payload": {
                "checks": checks,
                "raw_buy_alert": raw_alert,
                "gate_prefix": prefix,
                "directional_rs": dir_rs,
                "directional_open_rs": dir_ors,
                "directional_vwap_atr": dir_vwap,
                "abs_vwap_atr": abs_vwap,
                "strategy_fields": {k: self._monitor_text(row, k) for k in ["quality", "v379_reason", "positive_context_profile_reason"] if k in row.index},
                "bar_session_mode": self._live_session_mode(),
                "bar_age_minutes": age_min,
                "max_bar_age_minutes": max_age_min,
                "data_recent_enough": data_recent,
            },
        }

    def _monitor_record_from_feature_row(self, row: pd.Series | dict[str, Any], run_id: str, closed_ts: pd.Timestamp, status_hint: str = "Watching - no setup", reason_hint: str = "") -> dict[str, Any]:
        """Build a symbol-monitor row from the latest usable indicator row even when no trade signal exists.

        The previous implementation wrote a blank Waiting row whenever the final
        signal frame did not line up with the wall-clock closed bar.  On the free
        IEX feed, especially pre/after-hours, bars are sparse and can lag.  This
        fallback keeps the professional panel useful by showing the latest close,
        RVOL, ATR, relative-strength and VWAP readings, plus a clear reason that
        no setup was active.
        """
        rec = self._monitor_record_from_latest_row(pd.Series(row), run_id, closed_ts)
        if not bool(rec.get("setup_signal")):
            rec["monitor_status"] = status_hint
            rec["decision_status"] = "watching" if status_hint.lower().startswith("watching") else "waiting"
            if reason_hint:
                rec["reject_reason"] = reason_hint
                base = str(rec.get("check_summary") or "")
                rec["check_summary"] = (base + "; " if base else "") + reason_hint
        payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
        payload.update({"fallback_feature_row": True, "reason_hint": reason_hint})
        rec["payload"] = payload
        return rec

    def _latest_feature_monitor_record(self, symbol: str, run_id: str, closed_ts: pd.Timestamp, frame: pd.DataFrame, status_hint: str, reason_hint: str) -> dict[str, Any] | None:
        if frame is None or frame.empty or "timestamp" not in frame.columns:
            return None
        work = frame.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
        work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
        if work.empty:
            return None
        # Prefer same-session rows up to the worker's wall-clock closed bar, but
        # keep the newest available row as a diagnostic fallback when IEX is sparse.
        try:
            today_ny = pd.Timestamp(closed_ts).tz_convert(NY).date()
            same_day = work["timestamp"].dt.tz_convert(NY).dt.date.eq(today_ny)
            candidates = work[same_day & (work["timestamp"] <= pd.Timestamp(closed_ts))].copy()
            if candidates.empty:
                candidates = work[work["timestamp"] <= pd.Timestamp(closed_ts)].copy()
        except Exception:
            candidates = work.copy()
        if candidates.empty:
            return None
        row = candidates.iloc[-1].copy()
        if "symbol" not in row.index or not str(row.get("symbol") or "").strip():
            row["symbol"] = str(symbol).upper()
        return self._monitor_record_from_feature_row(row, run_id, closed_ts, status_hint=status_hint, reason_hint=reason_hint)

    def _monitor_record_for_missing_symbol(self, symbol: str, run_id: str, closed_ts: pd.Timestamp, status: str, reason: str) -> dict[str, Any]:
        rec = self._symbol_monitor_base_config(symbol, run_id, status, reason)
        try:
            ts_et = pd.Timestamp(closed_ts).tz_convert("America/New_York")
            rec.update({"latest_bar_time_utc": pd.Timestamp(closed_ts).isoformat(), "latest_bar_time_et": ts_et.strftime("%Y-%m-%d %H:%M"), "session_date": ts_et.date().isoformat()})
        except Exception:
            pass
        return rec

    def _mark_selected_symbol_monitors(self, records: list[dict[str, Any]], selected: pd.DataFrame) -> list[dict[str, Any]]:
        if selected is None or selected.empty or not records:
            return records
        keys: set[tuple[str, str, str]] = set()
        for _, row in selected.iterrows():
            variant = str(row.get("strategy_variant") or getattr(self.params, "live_strategy_variant", "")).lower()
            try:
                ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
                keys.add((str(row.get("symbol", "")).upper(), ts.isoformat(), variant))
            except Exception:
                keys.add((str(row.get("symbol", "")).upper(), "", variant))
        out = []
        for rec in records:
            rec_variant = str(rec.get("strategy_variant") or "").lower()
            key = (str(rec.get("symbol", "")).upper(), str(rec.get("latest_bar_time_utc") or ""), rec_variant)
            if key in keys or (str(rec.get("symbol", "")).upper(), "", rec_variant) in keys:
                updated = dict(rec)
                updated.update({
                    "selected_signal": True,
                    "monitor_status": "Selected - order planning",
                    "decision_status": "accepted",
                    "reject_reason": "Selected by latest-bar/top-N/capacity logic.",
                    "check_summary": f"{updated.get('checks_passed', 0)}/{updated.get('checks_total', 0)} checks passed; selected for order planning.",
                })
                out.append(updated)
            else:
                out.append(rec)
        return out

    def _write_symbol_monitor_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        try:
            self.store.upsert_symbol_monitor_snapshots(records)
        except Exception as exc:
            self.store.insert_event("symbol_monitor_write_error", {"message": str(exc)}, status="warning")

    def _within_live_entry_window(self, ts: pd.Timestamp | datetime | None = None) -> bool:
        ts = pd.Timestamp(ts or datetime.now(NY))
        if ts.tzinfo is None:
            ts = ts.tz_localize(NY)
        else:
            ts = ts.tz_convert(NY)
        if ts.weekday() >= 5:
            return False
        return _time_in_window(ts, getattr(self.live, "entry_start_time_et", "09:35"), getattr(self.live, "entry_end_time_et", "15:55"))

    def _strategy_specs_for_current_run(self) -> list[dict[str, str]]:
        if str(getattr(self.live, "strategy_run_mode", "single")).lower() == "all_strategies":
            return all_live_strategy_specs()
        return [_spec_for_preset_or_variant(getattr(self.params, "live_strategy_preset", "manual"), getattr(self.params, "live_strategy_variant", getattr(self.live, "strategy_variant", "best_report_153601")), getattr(self.params, "live_quality_gate", "off"))]

    def _fetch_recent_signals_for_params(self, run_id: str, closed_ts: pd.Timestamp, today_ny: Any, symbols: list[str], bars_5m: pd.DataFrame, daily: pd.DataFrame, qqq_context: pd.DataFrame, params: StrategyParams, session_mode: str = "regular_only") -> pd.DataFrame:
        monitor_records: list[dict[str, Any]] = []
        frames: list[pd.DataFrame] = []
        original_params = self.params
        self.params = params
        strategy_variant = str(getattr(params, "live_strategy_variant", getattr(self.live, "strategy_variant", "best_report_153601")))
        strategy_preset = str(getattr(params, "live_strategy_preset", "manual"))
        try:
            for symbol in symbols:
                sym_5m = bars_5m[bars_5m["symbol"] == symbol].copy()
                sym_daily = daily[daily["symbol"] == symbol].copy()
                if sym_5m.empty or sym_daily.empty:
                    monitor_records.append(self._monitor_record_for_missing_symbol(symbol, run_id, closed_ts, "Waiting - symbol data", "No recent 5-minute or daily bars for this symbol."))
                    continue
                intraday = pd.DataFrame()
                merged = pd.DataFrame()
                try:
                    intraday = add_intraday_features(sym_5m, session_mode=session_mode)
                    intraday = _add_daily_features_live_safe(intraday, sym_daily)
                    merged = merge_market_context(intraday, qqq_context)
                    signals = compute_signals(merged, params)
                except Exception as exc:
                    fallback = self._latest_feature_monitor_record(symbol, run_id, closed_ts, merged if not merged.empty else (intraday if not intraday.empty else sym_5m), "Error - indicator calculation", str(exc)[:220])
                    monitor_records.append(fallback or self._monitor_record_for_missing_symbol(symbol, run_id, closed_ts, "Error - indicator calculation", str(exc)[:220]))
                    continue
                if signals.empty:
                    fallback = self._latest_feature_monitor_record(symbol, run_id, closed_ts, merged if not merged.empty else intraday, "Watching - no setup", "Indicator pipeline produced rows but no strategy output for this strategy.")
                    monitor_records.append(fallback or self._monitor_record_for_missing_symbol(symbol, run_id, closed_ts, "Waiting - no indicator frame", "Indicator pipeline returned no rows."))
                    continue
                signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True, errors="coerce")
                signals["strategy_variant"] = strategy_variant
                signals["strategy_preset"] = strategy_preset
                signals["quality_gate"] = str(getattr(params, "live_quality_gate", "off"))
                signals["pattern_mode"] = str(getattr(params, "v379_pattern_mode", ""))
                same_day = signals["timestamp"].dt.tz_convert("America/New_York").dt.date.eq(today_ny)
                latest_frame = signals[same_day & (signals["timestamp"] <= closed_ts)].copy()
                if not latest_frame.empty:
                    latest_row = latest_frame.sort_values("timestamp").iloc[-1]
                    monitor_records.append(self._monitor_record_from_latest_row(latest_row, run_id, closed_ts))
                else:
                    latest_raw = pd.to_datetime(sym_5m.get("timestamp"), utc=True, errors="coerce").max() if "timestamp" in sym_5m.columns else None
                    extra = ""
                    try:
                        if pd.notna(latest_raw):
                            age = self._bar_age_minutes(latest_raw)
                            age_txt = "" if age is None else f" age {age:.1f} min"
                            extra = f" Latest raw {self.live.feed} bar was {pd.Timestamp(latest_raw).tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M')} ET{age_txt}."
                    except Exception:
                        extra = ""
                    fallback = self._latest_feature_monitor_record(symbol, run_id, closed_ts, merged if not merged.empty else signals, "Watching - latest usable bar", f"No same-day signal row aligned to the wall-clock bar after {session_mode} filtering; using latest available feature row for diagnostics.{extra}")
                    monitor_records.append(fallback or self._monitor_record_for_missing_symbol(symbol, run_id, closed_ts, "Waiting - no latest bar", f"No same-day indicator row after {session_mode} filtering.{extra}"))
                if "buy_alert" not in signals.columns:
                    continue
                alert_mask = signals["buy_alert"].fillna(False).astype(bool)
                alerts = signals[same_day & (signals["timestamp"] <= closed_ts) & alert_mask].copy()
                if alerts.empty:
                    continue
                frames.append(alerts)
            if not frames:
                self._write_symbol_monitor_records(monitor_records)
                return pd.DataFrame()
            alerts_all = pd.concat(frames, ignore_index=True)
            alerts_all = self._add_live_v25_filter_aliases(alerts_all)
            alerts_all["strategy_variant"] = strategy_variant
            alerts_all["strategy_preset"] = strategy_preset
            alerts_all["quality_gate"] = str(getattr(params, "live_quality_gate", "off"))
            alerts_all["pattern_mode"] = str(getattr(params, "v379_pattern_mode", ""))
            self._audit_candidates(alerts_all, run_id, "seen", "raw_signal_seen", stage="raw_signal", rank_col="score")

            ts_et = pd.to_datetime(alerts_all["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
            in_window = ts_et.map(lambda x: _time_in_window(x, getattr(self.live, "entry_start_time_et", "09:35"), getattr(self.live, "entry_end_time_et", "15:55")))
            if not bool(getattr(self.live, "allow_extended_hours_entries", False)):
                regular = ts_et.map(lambda x: _time_in_window(x, "09:30", "16:00"))
                in_window = in_window & regular
            rejected_schedule = alerts_all[~in_window].copy()
            self._audit_candidates(rejected_schedule, run_id, "rejected", "outside_dashboard_live_entry_window", stage="global_schedule", rank_col="score")
            alerts_all = alerts_all[in_window].copy()
            if alerts_all.empty:
                self._write_symbol_monitor_records(monitor_records)
                return alerts_all

            min_score = float(getattr(params, "min_candidate_score", 2.0) or 2.0)
            score_ok = pd.to_numeric(alerts_all["score"], errors="coerce").fillna(-9999.0) >= min_score
            self._audit_candidates(alerts_all[~score_ok].copy(), run_id, "rejected", f"score_below_min_{min_score:g}", stage="min_score", rank_col="score")
            alerts_all = alerts_all[score_ok].copy()
            if alerts_all.empty:
                self._write_symbol_monitor_records(monitor_records)
                return alerts_all

            before_candle = alerts_all.copy()
            alerts_all = _apply_v25_candlestick_filter(alerts_all, str(getattr(params, "candle_pattern_mode", "selective")))
            removed_keys = set(before_candle.apply(lambda r: self._candidate_audit_key(r), axis=1)) - set(alerts_all.apply(lambda r: self._candidate_audit_key(r), axis=1)) if not before_candle.empty else set()
            if removed_keys:
                removed = before_candle[before_candle.apply(lambda r: self._candidate_audit_key(r) in removed_keys, axis=1)].copy()
                self._audit_candidates(removed, run_id, "rejected", "candlestick_filter", stage="candlestick", rank_col="score")
            if alerts_all.empty:
                self._write_symbol_monitor_records(monitor_records)
                return alerts_all

            before_pre = alerts_all.copy()
            alerts_all, stats = _v27_apply_preselection_filters(alerts_all, params, use_news=bool(self.live.use_news_proxy))
            removed_keys = set(before_pre.apply(lambda r: self._candidate_audit_key(r), axis=1)) - set(alerts_all.apply(lambda r: self._candidate_audit_key(r), axis=1)) if not before_pre.empty else set()
            if removed_keys:
                removed = before_pre[before_pre.apply(lambda r: self._candidate_audit_key(r) in removed_keys, axis=1)].copy()
                self._audit_candidates(removed, run_id, "rejected", "preselection_or_quality_gate_filter", stage="preselection", rank_col="score")

            # Q/ML filters remain global experimental gates. They are applied to
            # each strategy's candidate set independently if enabled.
            if bool(getattr(self.live, "q_learning_filter_enabled", False)) and not alerts_all.empty:
                try:
                    policy_path = str(getattr(self.live, "q_learning_policy_path", "") or getattr(params, "q_learning_policy_path", "") or "").strip()
                    if policy_path:
                        q_model = load_q_model(policy_path)
                        q_reviewed = apply_q_policy(alerts_all, q_model, min_edge=float(getattr(self.live, "q_learning_min_edge", 0.0) or 0.0), min_state_count=int(getattr(self.live, "q_learning_min_state_count", 8) or 8))
                        approved = q_reviewed["q_policy_approved"].fillna(False).astype(bool)
                        self._audit_candidates(q_reviewed[~approved].copy(), run_id, "rejected", "q_policy_rejected", stage="q_policy", rank_col="score")
                        alerts_all = q_reviewed[approved].copy()
                    else:
                        self.store.set_state("last_q_policy_stats", {"enabled": True, "error": "LIVE_Q_POLICY_PATH is blank"})
                        alerts_all = pd.DataFrame()
                except Exception as exc:
                    self.store.insert_event("q_policy_filter_error", {"message": str(exc), "strategy_variant": strategy_variant}, status="error")
                    alerts_all = pd.DataFrame()
            if bool(getattr(self.live, "ml_ranker_filter_enabled", False)) and not alerts_all.empty:
                try:
                    ml_path = str(getattr(self.live, "ml_ranker_model_path", "") or getattr(params, "ml_ranker_model_path", "") or "").strip()
                    if ml_path:
                        ml_model = load_ranker_model(ml_path)
                        ml_reviewed = score_ml_ranker_candidates(alerts_all, ml_model)
                        min_pred = float(getattr(self.live, "ml_ranker_min_pred_r", getattr(params, "ml_ranker_min_pred_r", 0.05)) or 0.05)
                        min_win = float(getattr(self.live, "ml_ranker_min_win_prob", getattr(params, "ml_ranker_min_win_prob", 0.0)) or 0.0)
                        approved = pd.to_numeric(ml_reviewed.get("ml_pred_r", -9999.0), errors="coerce").fillna(-9999.0) >= min_pred
                        if min_win > 0 and "ml_pred_win_prob" in ml_reviewed.columns:
                            approved = approved & (pd.to_numeric(ml_reviewed["ml_pred_win_prob"], errors="coerce").fillna(0.0) >= min_win)
                        self._audit_candidates(ml_reviewed[~approved].copy(), run_id, "rejected", "ml_ranker_rejected", stage="ml_ranker", rank_col="score")
                        alerts_all = ml_reviewed[approved].copy()
                    else:
                        self.store.set_state("last_ml_ranker_stats", {"enabled": True, "error": "LIVE_ML_RANKER_MODEL_PATH is blank"})
                        alerts_all = pd.DataFrame()
                except Exception as exc:
                    self.store.insert_event("ml_ranker_filter_error", {"message": str(exc), "strategy_variant": strategy_variant}, status="error")
                    alerts_all = pd.DataFrame()

            self.store.set_state(f"last_filter_stats_{strategy_variant}", stats if 'stats' in locals() else {})
            if alerts_all.empty:
                self._write_symbol_monitor_records(monitor_records)
                return alerts_all
            selected_so_far = _v27_select_top_n_with_caps(alerts_all, int(getattr(params, "max_trades_per_day", 2) or 2), params)
            if selected_so_far.empty:
                self._audit_candidates(alerts_all, run_id, "rejected", "topn_caps_selected_none", stage="topn", rank_col="score")
                self._write_symbol_monitor_records(monitor_records)
                return selected_so_far
            selected_so_far["timestamp"] = pd.to_datetime(selected_so_far["timestamp"], utc=True, errors="coerce")
            alerts_all["timestamp"] = pd.to_datetime(alerts_all["timestamp"], utc=True, errors="coerce")
            # On Alpaca Basic/IEX the latest available historical bar can lag the
            # wall-clock 5-minute slot or skip slots in pre/after-hours.  Use the
            # latest available selected bar within a bounded age window instead of
            # requiring timestamp == closed_ts.  Duplicate order protection still
            # keys by strategy+symbol+signal time, so the same stale signal is not
            # resubmitted every poll.
            max_age_min = self._max_bar_age_minutes()
            recent_cutoff = pd.Timestamp(closed_ts) - pd.Timedelta(minutes=max_age_min)
            if str(self.live.selection_mode).lower() == "latest_bar_only":
                recent_alerts = alerts_all[(alerts_all["timestamp"] <= closed_ts) & (alerts_all["timestamp"] >= recent_cutoff)].copy()
                latest_signal_ts = recent_alerts["timestamp"].max() if not recent_alerts.empty else pd.NaT
                out = recent_alerts[recent_alerts["timestamp"] == latest_signal_ts].copy() if pd.notna(latest_signal_ts) else pd.DataFrame()
                selected_keys = set(out.apply(lambda r: self._candidate_audit_key(r), axis=1)) if not out.empty else set()
            else:
                selected_recent = selected_so_far[(selected_so_far["timestamp"] <= closed_ts) & (selected_so_far["timestamp"] >= recent_cutoff)].copy()
                latest_signal_ts = selected_recent["timestamp"].max() if not selected_recent.empty else pd.NaT
                out = selected_recent[selected_recent["timestamp"] == latest_signal_ts].copy() if pd.notna(latest_signal_ts) else pd.DataFrame()
                selected_keys = set(out.apply(lambda r: self._candidate_audit_key(r), axis=1)) if not out.empty else set()
            if out.empty and not selected_so_far.empty:
                self._audit_candidates(selected_so_far.copy(), run_id, "rejected", f"no_recent_selected_bar_within_{max_age_min}_minutes_of_worker_clock", stage="topn_latest", rank_col="score")
            not_selected = alerts_all[~alerts_all.apply(lambda r: self._candidate_audit_key(r) in selected_keys, axis=1)].copy()
            self._audit_candidates(not_selected, run_id, "rejected", "not_latest_selected_candidate_or_topn_cap", stage="topn_latest", rank_col="score")
            if out.empty:
                self._write_symbol_monitor_records(monitor_records)
                return out
            out["session_date"] = out["timestamp"].dt.tz_convert("America/New_York").dt.date.astype(str)
            out = out.sort_values(["score", "timestamp"], ascending=[False, True]).reset_index(drop=True)
            out["strategy_variant"] = strategy_variant
            out["strategy_code"] = getattr(params, "live_strategy_code", _strategy_code(strategy_variant))
            out["strategy_label"] = getattr(params, "live_strategy_label", strategy_preset)
            out["strategy_preset"] = strategy_preset
            out["quality_gate"] = str(getattr(params, "live_quality_gate", "off"))
            out["pattern_mode"] = str(getattr(params, "v379_pattern_mode", ""))
            monitor_records = self._mark_selected_symbol_monitors(monitor_records, out)
            self._write_symbol_monitor_records(monitor_records)
            records = []
            for idx, row in out.iterrows():
                records.append(self._audit_record_from_signal(row, run_id, "accepted", "selected_for_order_planning", stage="selected", rank_after=int(idx) + 1))
            self.store.upsert_candidate_audits(records)
            return out
        finally:
            self.params = original_params

    def fetch_recent_signals(self) -> pd.DataFrame:
        now = _now_utc()
        run_id = pd.Timestamp(now).strftime("%Y%m%dT%H%M%SZ")
        closed_ts = latest_closed_5m_start(now)
        today_ny = closed_ts.tz_convert("America/New_York").date()
        symbols = list(dict.fromkeys([s.upper() for s in (self.live.symbols or []) if s]))
        session_mode = self._live_session_mode()
        specs = self._strategy_specs_for_current_run()
        # Make all-strategies mode visible immediately.  The actual indicator
        # rows overwrite this scaffold as each strategy finishes scanning.
        if str(getattr(self.live, "strategy_run_mode", "single")).lower() == "all_strategies":
            try:
                self._seed_full_strategy_symbol_monitor(run_id, closed_ts, specs, symbols, "Queued; worker is scanning all strategies for this symbol.")
            except Exception as exc:
                self.store.insert_event("symbol_monitor_seed_error", {"message": str(exc)}, status="warning")
        bars_5m, daily = self._update_bar_cache()
        if bars_5m.empty or daily.empty:
            records = []
            for spec in specs:
                params = self._make_params_for_live_config(getattr(self, "_runtime_config", {}), spec.get("preset"), spec.get("variant"), spec.get("quality_gate"))
                original_params = self.params
                self.params = params
                records.extend([self._monitor_record_for_missing_symbol(s, run_id, closed_ts, "Waiting - market data", "No 5-minute or daily bars were returned.") for s in symbols])
                self.params = original_params
            self._write_symbol_monitor_records(records)
            return pd.DataFrame()
        bars_5m = bars_5m[pd.to_datetime(bars_5m["timestamp"], utc=True) <= closed_ts].copy()
        qqq_5m = bars_5m[bars_5m["symbol"] == "QQQ"].copy()
        qqq_daily = daily[daily["symbol"] == "QQQ"].copy()
        if qqq_5m.empty or qqq_daily.empty:
            records = []
            for spec in specs:
                params = self._make_params_for_live_config(getattr(self, "_runtime_config", {}), spec.get("preset"), spec.get("variant"), spec.get("quality_gate"))
                original_params = self.params
                self.params = params
                records.extend([self._monitor_record_for_missing_symbol(s, run_id, closed_ts, "Waiting - QQQ context", "QQQ context is required for relative-strength indicators.") for s in symbols])
                self.params = original_params
            self._write_symbol_monitor_records(records)
            return pd.DataFrame()
        qqq_context = _build_qqq_context_live_safe(qqq_5m, qqq_daily, session_mode=session_mode)
        selected_frames: list[pd.DataFrame] = []
        scan_summary = {
            "updated_at_utc": utc_now_iso(),
            "run_id": run_id,
            "closed_bar_utc": pd.Timestamp(closed_ts).isoformat(),
            "session_mode": session_mode,
            "feed": self.live.feed,
            "realtime_feed_for_orders": self._live_realtime_feed(),
            "max_bar_age_minutes": self._max_bar_age_minutes(),
            "symbols": symbols,
            "strategy_run_mode": getattr(self.live, "strategy_run_mode", "single"),
            "strategies": [],
        }
        for spec in specs:
            params = self._make_params_for_live_config(getattr(self, "_runtime_config", {}), spec.get("preset"), spec.get("variant"), spec.get("quality_gate"))
            try:
                selected = self._fetch_recent_signals_for_params(run_id, closed_ts, today_ny, symbols, bars_5m, daily, qqq_context, params, session_mode=session_mode)
                selected_count = int(0 if selected is None else len(selected))
                scan_summary["strategies"].append({
                    "variant": spec.get("variant"),
                    "preset": spec.get("preset"),
                    "quality_gate": spec.get("quality_gate"),
                    "code": spec.get("code"),
                    "selected_signals": selected_count,
                    "status": "ok",
                })
                if selected is not None and not selected.empty:
                    selected_frames.append(selected)
            except Exception as exc:
                scan_summary["strategies"].append({
                    "variant": spec.get("variant"),
                    "preset": spec.get("preset"),
                    "quality_gate": spec.get("quality_gate"),
                    "code": spec.get("code"),
                    "selected_signals": 0,
                    "status": "error",
                    "error": str(exc)[:300],
                })
                self.store.insert_event("strategy_scan_error", {"strategy_variant": spec.get("variant"), "message": str(exc)}, status="error")
                # Keep the Live Symbol Intelligence panel complete even if one
                # experimental strategy errors.  Without this, all-strategies mode
                # can show fewer strategy views than active_strategy_count and make
                # the dashboard look like data is missing.
                original_params = self.params
                try:
                    self.params = params
                    err_records = [
                        self._monitor_record_for_missing_symbol(
                            s,
                            run_id,
                            closed_ts,
                            "Error - strategy scan",
                            f"Strategy scan failed before symbol checks: {str(exc)[:180]}",
                        )
                        for s in symbols
                    ]
                    self._write_symbol_monitor_records(err_records)
                finally:
                    self.params = original_params
        try:
            scan_summary["selected_total"] = int(sum(int(x.get("selected_signals", 0) or 0) for x in scan_summary["strategies"]))
            self.store.set_state("last_strategy_scan_summary", scan_summary)
        except Exception:
            pass
        if not selected_frames:
            return pd.DataFrame()
        out = pd.concat(selected_frames, ignore_index=True)
        if "score" in out.columns:
            out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
            out = out.sort_values(["score", "timestamp", "strategy_variant"], ascending=[False, True, True])
        return out.reset_index(drop=True)

    def _latest_reference_prices(self, symbols: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            quotes = self.data_client.latest_quotes(symbols, feed=self._live_realtime_feed())
            if not quotes.empty:
                for _, r in quotes.iterrows():
                    sym = str(r.get("symbol", "")).upper()
                    mid = _safe_float(r.get("mid"), 0.0)
                    if sym and mid > 0:
                        out[sym] = mid
        except Exception as exc:
            self.store.insert_event("quote_sync_warning", {"message": str(exc)}, status="warning")
        return out



    def _signal_context_payload(self, signal: pd.Series) -> dict[str, Any]:
        keep_prefixes = ("v379_", "v383_", "v384_", "v385_", "positive_context_", "q_", "ml_")
        keep_names = {
            "symbol", "timestamp", "side", "trigger_type", "candidate_score", "score", "quality",
            "strategy_variant", "strategy_preset", "quality_gate", "pattern_mode",
            "rvol_time_of_day", "daily_atr14_percent", "gap_percent", "day_relative_strength",
            "open_relative_strength", "vwap_extension_atr", "qqq_change_from_open",
            "qqq_day_change_percent", "atr5m14", "close", "open", "high", "low", "volume",
            "entry_candle_pattern", "candle_pattern_primary", "candle_pattern_score",
            "bullish_continuation_candle", "bearish_continuation_candle",
            "bullish_rejection_candle", "bearish_rejection_candle",
            "bullish_engulfing_candle", "bearish_engulfing_candle",
        }
        out: dict[str, Any] = {}
        for key, value in signal.to_dict().items():
            if key in keep_names or str(key).startswith(keep_prefixes):
                if isinstance(value, (pd.Timestamp, datetime)):
                    out[str(key)] = value.isoformat()
                else:
                    try:
                        if hasattr(value, "item"):
                            value = value.item()
                    except Exception:
                        pass
                    out[str(key)] = value
        return out

    def _params_for_signal(self, signal: pd.Series | dict[str, Any]) -> StrategyParams:
        data = signal.to_dict() if hasattr(signal, "to_dict") else dict(signal or {})
        spec = _strategy_spec_from_signal_values(
            data.get("strategy_variant"),
            data.get("strategy_preset"),
            data.get("quality_gate"),
            data.get("strategy_code"),
        )
        return self._make_params_for_live_config(getattr(self, "_runtime_config", {}), spec.get("preset"), spec.get("variant"), spec.get("quality_gate"))

    def build_order_plan(self, signal: pd.Series, equity: float, high_watermark: float, reference_price: float | None = None, params: StrategyParams | None = None, max_notional: float | None = None, max_notional_reason: str = "") -> dict[str, Any] | None:
        params = params or self._params_for_signal(signal)
        side = str(signal.get("side", "")).lower()
        if side not in {"long", "short"}:
            return None
        signal_close = _safe_float(signal.get("close"), 0.0)
        entry_price = _safe_float(reference_price, 0.0) if reference_price else signal_close
        atr_value = _safe_float(signal.get("atr5m14"), 0.0)
        if entry_price <= 0 or atr_value <= 0:
            return None
        risk_per_share = max(entry_price * float(getattr(params, "v25_min_stop_pct", 0.0015)), float(getattr(params, "v25_stop_atr_mult", 0.60)) * atr_value)
        # Alpaca validates bracket stop/target prices against its own base price,
        # which can move a few cents from our quote/reference price before the
        # order reaches /orders.  A tiny ATR-based stop can round to the same
        # cent as the base price and be rejected (for example: stop_loss.stop_price
        # must be >= base_price + 0.01 on a short bracket).  Force a practical
        # minimum distance before sizing so the stop widening is included in qty.
        min_order_distance = max(0.05, entry_price * 0.0005)
        risk_per_share = max(risk_per_share, min_order_distance)
        if risk_per_share <= 0:
            return None
        target_r = float(getattr(params, "v25_target_r", 0.75))
        if side == "long":
            stop_price = min(entry_price - risk_per_share, entry_price - min_order_distance)
            target_price = max(entry_price + target_r * risk_per_share, entry_price + min_order_distance)
        else:
            stop_price = max(entry_price + risk_per_share, entry_price + min_order_distance)
            target_price = min(entry_price - target_r * risk_per_share, entry_price - min_order_distance)
        risk_budget, effective_pct, dd_pct, paused = _v28_calculate_risk_budget(equity, high_watermark, params)
        if paused or risk_budget <= 0:
            return None
        uncapped_qty = risk_budget / risk_per_share
        raw_qty = uncapped_qty
        notional_cap = _safe_float(max_notional, 0.0)
        notional_capped = False
        if notional_cap > 0 and entry_price > 0:
            cap_qty = notional_cap / entry_price
            if cap_qty < raw_qty:
                raw_qty = cap_qty
                notional_capped = True
        qty = raw_qty if self.risk.allow_fractional else math.floor(raw_qty)
        if qty <= 0:
            return None
        estimated_notional = float(qty) * float(entry_price)
        actual_risk_dollars = float(qty) * float(risk_per_share)
        risk_budget_shortfall = max(0.0, float(risk_budget) - actual_risk_dollars)
        min_actual_risk = max(0.0, _safe_float(getattr(params, "compounding_min_risk_dollars", self.risk.min_risk_dollars), self.risk.min_risk_dollars))
        if notional_capped and min_actual_risk > 0 and actual_risk_dollars < min_actual_risk:
            return None
        sig_ts = pd.Timestamp(signal.get("timestamp")).tz_convert("UTC")
        ts_et = sig_ts.tz_convert("America/New_York")
        symbol = str(signal.get("symbol", "")).upper()
        strategy_variant = str(signal.get("strategy_variant") or getattr(params, "live_strategy_variant", getattr(self.live, "strategy_variant", "best_report_153601"))).strip().lower()
        strategy_code = str(signal.get("strategy_code") or getattr(params, "live_strategy_code", _strategy_code(strategy_variant))).strip().lower()
        client_order_id = f"rmv33-{strategy_code}-{symbol}-{ts_et.strftime('%Y%m%d%H%M')}-{side[0]}"[:48]
        max_hold_until = sig_ts + pd.Timedelta(minutes=5 * int(getattr(params, "v25_max_hold_bars", 12)))
        return {
            "symbol": symbol,
            "strategy_side": side,
            "alpaca_side": _order_side(side),
            "signal_time_utc": sig_ts.isoformat(),
            "signal_time_et": ts_et.strftime("%Y-%m-%d %H:%M"),
            "session_date": ts_et.date().isoformat(),
            "trigger_type": str(signal.get("trigger_type", "")),
            "candidate_score": _safe_float(signal.get("candidate_score", signal.get("score", 0.0)), 0.0),
            "entry_reference_price": _round_price(entry_price),
            "signal_close": _round_price(signal_close),
            "risk_per_share": risk_per_share,
            "risk_budget": risk_budget,
            "effective_risk_pct": effective_pct,
            "drawdown_before_trade_pct": dd_pct,
            "qty": qty,
            "uncapped_qty": uncapped_qty,
            "estimated_notional": estimated_notional,
            "actual_risk_dollars": actual_risk_dollars,
            "risk_budget_shortfall": risk_budget_shortfall,
            "risk_capped_by_budget": bool(risk_budget_shortfall > 0.01),
            "notional_cap": notional_cap,
            "notional_cap_reason": str(max_notional_reason or ""),
            "notional_capped": notional_capped,
            "target_price": _round_price(target_price),
            "stop_price": _round_price(stop_price),
            "max_hold_until_utc": max_hold_until.isoformat(),
            "client_order_id": client_order_id,
            "submitted_at_utc": utc_now_iso(),
            "strategy_variant": strategy_variant,
            "strategy_code": strategy_code,
            "strategy_preset": str(signal.get("strategy_preset") or getattr(params, "live_strategy_preset", "env")),
            "strategy_profile": str(getattr(params, "strategy_profile", "symbol_playbook_v25")),
            "quality_gate": str(signal.get("quality_gate") or getattr(params, "live_quality_gate", "env")),
            "pattern_mode": str(signal.get("pattern_mode") or getattr(params, "v379_pattern_mode", "")),
            "selection_mode": str(getattr(self.live, "selection_mode", "seen_so_far_top_n")),
            "signal_context": self._signal_context_payload(signal),
        }

    def _daily_loss_reached(self, daily_pl: float) -> bool:
        limit = float(self.live.max_daily_loss_dollars or 0.0)
        return limit > 0 and daily_pl <= -abs(limit)

    def _save_engine_state(self) -> None:
        self.state["last_run_utc"] = utc_now_iso()
        self.state["settings"] = {"live": self._serializable_live(), "risk": asdict(self.risk)}
        self.store.set_state("engine_state", self.state)
        self.store.set_state("settings", {"live": self._serializable_live(), "risk": asdict(self.risk), "alpaca": self.settings.redacted()})
        _save_state(self.live.state_path, self.state)

    def enforce_max_hold_exits(self) -> list[dict[str, Any]]:
        if not self.live.enable_max_hold_exit:
            return []
        rows: list[dict[str, Any]] = []
        open_positions = self.store.open_positions()
        if open_positions.empty or "max_hold_until_utc" not in open_positions.columns:
            return rows
        now_ts = pd.Timestamp(_now_utc())
        submitted = set(self.state.get("max_hold_exit_keys", []))
        for _, pos in open_positions.iterrows():
            symbol = str(pos.get("symbol", "")).upper()
            deadline_raw = pos.get("max_hold_until_utc")
            if not symbol or not deadline_raw:
                continue
            deadline = pd.Timestamp(deadline_raw)
            if deadline.tzinfo is None:
                deadline = deadline.tz_localize("UTC")
            else:
                deadline = deadline.tz_convert("UTC")
            key = f"{symbol}|{deadline.isoformat()}"
            if now_ts < deadline or key in submitted:
                continue
            row = {"timestamp": utc_now_iso(), "event": "max_hold_exit_due", "symbol": symbol, "max_hold_until_utc": deadline.isoformat(), "dry_run": self.live.dry_run}
            if self.live.dry_run:
                row["alpaca_status"] = "dry_run_close_not_submitted"
            else:
                cancelled = self.trading_client.cancel_open_orders_for_symbol(symbol)
                response = self.trading_client.close_position(symbol)
                self.store.upsert_order(response)
                row["alpaca_status"] = str(response.get("status", "close_submitted"))
                row["alpaca_order_id"] = response.get("id", "")
                row["cancelled_open_orders"] = cancelled
            self.store.insert_event("max_hold_exit_due", row, symbol=symbol, status=row.get("alpaca_status"))
            submitted.add(key)
            rows.append(row)
        self.state["max_hold_exit_keys"] = list(submitted)[-1000:]
        return rows

    def run_once(self) -> list[dict[str, Any]]:
        self._apply_runtime_config_override()
        rows: list[dict[str, Any]] = []
        timestamp = utc_now_iso()
        self._heartbeat("running", "Worker scan started.")
        account = self._account_snapshot()
        positions = self._sync_account_state()
        if not self._market_is_open():
            row = {"timestamp": timestamp, "event": "market_closed", "message": "No scan/order because market is closed."}
            rows.append(row)
            self.store.insert_event("market_closed", row, status="idle")
            run_id = pd.Timestamp(_now_utc()).strftime("%Y%m%dT%H%M%SZ")
            self._write_symbol_monitor_records(self._idle_symbol_monitor_records(run_id, "Paused - market closed", "Market is closed; indicator snapshots update on market-open scans."))
            _append_log(self.live.log_path, rows)
            self._heartbeat("idle", "Market is closed.", {"open_positions": len(positions)})
            self._save_engine_state()
            return rows
        if not self._within_live_entry_window(pd.Timestamp.now(tz=NY)):
            row = {
                "timestamp": timestamp,
                "event": "entry_window_closed",
                "message": f"No new entries because dashboard live entry window is {self.live.entry_start_time_et}-{self.live.entry_end_time_et} ET.",
                "entry_start_time_et": self.live.entry_start_time_et,
                "entry_end_time_et": self.live.entry_end_time_et,
            }
            rows.append(row)
            self.store.insert_event("entry_window_closed", row, status="idle")
            _append_log(self.live.log_path, rows)
            self._heartbeat("idle", "Entry window closed.", {"open_positions": len(positions)})
            self._save_engine_state()
            return rows
        rows.extend(self.enforce_max_hold_exits())
        existing = self._existing_symbols()
        equity, high_watermark, daily_pl = self._account_equity(account)
        account_buying_power = self._account_buying_power(account)
        self._last_equity_for_budget = equity
        if self._daily_loss_reached(daily_pl):
            row = {"timestamp": timestamp, "event": "daily_loss_limit", "message": "Daily loss limit reached; no new entries.", "daily_pl": daily_pl}
            rows.append(row)
            self.store.insert_event("daily_loss_limit", row, status="blocked")
            _append_log(self.live.log_path, rows)
            self._heartbeat("blocked", "Daily loss limit reached.", {"daily_pl": daily_pl})
            self._save_engine_state()
            return rows
        signals = self.fetch_recent_signals()
        if signals.empty:
            row = {"timestamp": timestamp, "event": "no_signal", "message": f"No signal on latest eligible closed 5-minute bar for strategy mode {getattr(self.live, 'strategy_run_mode', 'single')} ({getattr(self.live, 'active_strategy_count', 1)} active strategy/strategies)."}
            rows.append(row)
            self.store.insert_event("no_signal", row, status="idle")
            _append_log(self.live.log_path, rows)
            self._heartbeat("idle", "No latest-bar signal.", {"open_positions": len(existing), "equity": equity})
            self._save_engine_state()
            return rows
        submitted = list(self.state.get("submitted_signal_keys", []))
        today = datetime.now(NY).date().isoformat()
        strategy_multiplier = max(1, int(getattr(self.live, "active_strategy_count", 1) or 1)) if str(getattr(self.live, "strategy_run_mode", "single")).lower() == "all_strategies" else 1
        effective_daily_limit = max(1, int(self.live.max_daily_trades)) * strategy_multiplier
        effective_open_limit = max(1, int(self.live.max_open_positions)) * strategy_multiplier
        remaining_daily = max(0, effective_daily_limit - self._daily_order_count(today))
        available_slots = max(0, effective_open_limit - len(existing))
        capacity = min(remaining_daily, available_slots)
        if capacity <= 0:
            row = {"timestamp": timestamp, "event": "capacity_full", "message": "Daily/open-position limits reached.", "signals_seen": len(signals), "strategy_run_mode": getattr(self.live, "strategy_run_mode", "single"), "effective_daily_limit": effective_daily_limit, "effective_open_limit": effective_open_limit}
            rows.append(row)
            self.store.insert_event("capacity_full", row, status="blocked")
            _append_log(self.live.log_path, rows)
            self._heartbeat("blocked", "Capacity full.", {"signals_seen": len(signals), "open_positions": len(existing), "effective_daily_limit": effective_daily_limit, "effective_open_limit": effective_open_limit})
            self._save_engine_state()
            return rows
        quote_lookup = self._latest_reference_prices([str(s).upper() for s in signals["symbol"].dropna().unique().tolist()])
        taken = 0
        reserved_notional = 0.0
        attempted_symbols: set[str] = set()
        buying_power_safety = self._buying_power_safety_fraction()
        for _, signal in signals.iterrows():
            sym = str(signal.get("symbol", "")).upper()
            if sym in attempted_symbols:
                self.store.upsert_candidate_audit(self._audit_record_from_signal(signal, timestamp, "rejected", "symbol_already_attempted_this_scan", stage="submit_checks"))
                continue
            signal_params = self._params_for_signal(signal)
            order_notional_cap, order_notional_reason = self._order_notional_budget(account_buying_power, reserved_notional, capacity - taken, strategy_multiplier)
            plan = self.build_order_plan(signal, equity, high_watermark, reference_price=quote_lookup.get(sym), params=signal_params, max_notional=order_notional_cap, max_notional_reason=order_notional_reason)
            if not plan:
                self.store.upsert_candidate_audit(self._audit_record_from_signal(signal, timestamp, "rejected", "sizing_or_invalid_order_plan", stage="order_plan"))
                continue
            key = f"{plan.get('strategy_variant','')}|{plan['symbol']}|{plan['session_date']}|{plan['signal_time_et']}|{plan['strategy_side']}|{plan['trigger_type']}"
            if key in submitted:
                self.store.upsert_candidate_audit(self._audit_record_from_signal(signal, timestamp, "rejected", "duplicate_signal_already_submitted", stage="submit_checks", plan=plan))
                continue
            if plan["symbol"] in existing:
                self.store.upsert_candidate_audit(self._audit_record_from_signal(signal, timestamp, "rejected", "position_already_open", stage="submit_checks", plan=plan))
                continue
            if self._symbol_daily_count(plan["symbol"], plan["session_date"]) >= self.live.max_orders_per_symbol_per_day:
                self.store.upsert_candidate_audit(self._audit_record_from_signal(signal, timestamp, "rejected", "symbol_daily_order_limit", stage="submit_checks", plan=plan))
                continue
            if account_buying_power > 0 and (reserved_notional + _safe_float(plan.get("estimated_notional"), 0.0)) > account_buying_power * buying_power_safety:
                self.store.upsert_candidate_audit(self._audit_record_from_signal(signal, timestamp, "rejected", "buying_power_reserve_limit", stage="submit_checks", plan=plan))
                continue
            self.store.upsert_candidate_audit(self._audit_record_from_signal(signal, timestamp, "accepted", "order_plan_created", stage="order_plan", plan=plan))
            attempted_symbols.add(plan["symbol"])
            log_row = {"timestamp": timestamp, "event": "paper_order_plan", **plan, "dry_run": self.live.dry_run, "account_buying_power": account_buying_power, "buying_power_safety_pct": buying_power_safety * 100.0, "reserved_notional_before_order": reserved_notional}
            if self.live.dry_run:
                log_row["alpaca_status"] = "dry_run_not_submitted"
                self.store.upsert_signal_plan(log_row, status="dry_run_not_submitted")
            else:
                try:
                    extended_order = self._is_extended_session_order_context(plan.get("signal_time_utc"))
                    if extended_order:
                        limit_price = self._extended_limit_price(plan)
                        if limit_price <= 0:
                            raise ValueError("Cannot submit extended-hours order without a valid limit price.")
                        result = self.trading_client.submit_limit_order(
                            symbol=plan["symbol"],
                            side=plan["alpaca_side"],
                            qty=float(plan["qty"]),
                            limit_price=limit_price,
                            client_order_id=plan["client_order_id"],
                            fractional=self.risk.allow_fractional,
                            extended_hours=True,
                            time_in_force="day",
                        )
                        log_row["order_mode"] = "extended_hours_simple_limit"
                        log_row["limit_price"] = limit_price
                        log_row["extended_hours"] = True
                        log_row["message"] = "Extended-hours entry submitted as simple limit order; bracket target/stop are recorded for reporting but Alpaca extended-hours API does not accept market bracket entries."
                    else:
                        latest_ref = self._latest_reference_prices([plan["symbol"]]).get(plan["symbol"], quote_lookup.get(plan["symbol"], plan.get("entry_reference_price")))
                        submit_plan = self._prepare_regular_bracket_plan_for_submit(plan, latest_ref)
                        if not submit_plan:
                            raise ValueError("Regular-session bracket order became invalid after latest-price validation; order was not submitted.")
                        if submit_plan is not plan:
                            plan.update(submit_plan)
                            log_row.update(submit_plan)
                        result, submit_plan = self._submit_regular_market_bracket_with_auto_repair(plan, latest_ref)
                        if submit_plan:
                            plan.update(submit_plan)
                            log_row.update(submit_plan)
                        log_row["order_mode"] = "regular_market_bracket"
                        log_row["extended_hours"] = False
                    log_row["alpaca_status"] = result.status
                    log_row["alpaca_order_id"] = result.response.get("id", "")
                    self.store.upsert_order(result.response)
                    self.store.upsert_signal_plan(log_row, status=result.status)
                    existing.add(plan["symbol"])
                    reserved_notional += _safe_float(plan.get("estimated_notional"), 0.0)
                except Exception as exc:
                    if hasattr(self, "_last_bracket_repair_attempts"):
                        log_row["regular_bracket_auto_repair_attempts"] = getattr(self, "_last_bracket_repair_attempts", [])
                    log_row["alpaca_status"] = "submit_error"
                    log_row["message"] = str(exc)
                    self.store.upsert_signal_plan(log_row, status="submit_error")
                    self.store.insert_event("paper_order_submit_error", log_row, symbol=plan["symbol"], status="submit_error")
                    rows.append(log_row)
                    continue
            self.store.insert_event("paper_order_plan", log_row, symbol=plan["symbol"], status=log_row.get("alpaca_status"))
            submitted.append(key)
            self.state["submitted_strategy_order_records"] = self._merge_strategy_order_records([
                {
                    "client_order_id": plan["client_order_id"],
                    "strategy_variant": plan.get("strategy_variant"),
                    "strategy_preset": plan.get("strategy_preset"),
                    "strategy_code": plan.get("strategy_code"),
                    "quality_gate": plan.get("quality_gate"),
                    "symbol": plan["symbol"],
                    "session_date": plan["session_date"],
                    "signal_time_et": plan["signal_time_et"],
                    "strategy_side": plan["strategy_side"],
                    "status": log_row.get("alpaca_status", ""),
                    "submitted_at": log_row.get("submitted_at_utc", ""),
                }
            ])
            rows.append(log_row)
            taken += 1
            if taken >= capacity:
                break
        self.state["submitted_signal_keys"] = submitted[-1000:]
        if not rows:
            rows.append({"timestamp": timestamp, "event": "signals_filtered", "message": "Signals existed but were duplicate, already open, or failed sizing/capacity checks.", "signals_seen": len(signals)})
            self.store.insert_event("signals_filtered", rows[-1], status="filtered")
        try:
            self._sync_account_state()
        except Exception:
            pass
        _append_log(self.live.log_path, rows)
        self._heartbeat("running", f"Scan complete; submitted/planned {taken} order(s).", {"signals_seen": len(signals), "orders_planned": taken, "equity": equity})
        self._save_engine_state()
        return rows
