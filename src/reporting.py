from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import REPORTS_DIR

REPORT_DIR = REPORTS_DIR


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_")[:120] or "report"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _write_df(path: Path, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].astype(str)
    clean.to_csv(path, index=False)


def _selected_trade_market_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Compact audit table: what the market looked like at each selected entry.

    Keeps every original selected-trade column, but moves the decision/indicator
    columns to the front so the report immediately explains why each trade was
    selected and which market conditions existed at execution.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Build a fallback explanation for older selected_trades files that do not
    # yet include the V37.4 decision_reason columns.
    if "decision_reason" not in out.columns:
        def f(row):
            return " | ".join([
                f"symbol={row.get('symbol', row.get('_symbol', 'NA'))}",
                f"side={row.get('side', row.get('_side', 'NA'))}",
                f"trigger={row.get('trigger_type', row.get('_trigger', 'NA'))}",
                f"score={row.get('candidate_score', row.get('score', 'NA'))}",
                f"rvol={row.get('rvol_time_of_day', 'NA')}",
                f"daily_atr_pct={row.get('daily_atr14_percent', 'NA')}",
                f"gap_pct={row.get('gap_percent', 'NA')}",
                f"day_rs={row.get('day_relative_strength', 'NA')}",
                f"open_rs={row.get('open_relative_strength', 'NA')}",
                f"vwap_ext_atr={row.get('vwap_extension_atr', 'NA')}",
                f"qqq_change_open={row.get('qqq_change_from_open', row.get('qqq_day_change_percent', 'NA'))}",
            ])
        out["decision_reason"] = out.apply(f, axis=1)
    preferred = [
        "decision_reason", "market_conditions_at_entry", "entry_time", "exit_time", "timestamp", "_ts",
        "session_date", "_date", "symbol", "_symbol", "side", "_side", "trigger_type", "_trigger",
        "entry_price", "exit_price", "target_price", "stop_price", "r_multiple", "pnl_dollars",
        "candidate_score", "score", "_tune_rank_score", "v379_pattern_mode", "v379_pattern_match",
        "v379_reason", "v379_pattern_score", "v379_rank_score", "v379_original_score",
        "v379_candle_component", "v379_rvol_component", "v379_rs_component", "v379_vwap_component",
        "v379_directional_rs", "v379_directional_open_rs", "v379_directional_vwap_atr",
        "v379_abs_vwap_atr", "v379_abs_gap_percent", "v379_abs_qqq_change",
        "v385_rule_name", "v385_quality_score", "v385_hard_damage_block",
        "v385_block_ma_long_vwap", "v385_block_intc_short_late", "v385_block_coin_short", "v385_block_after_noon",
        "v385_rule_amd_mu_short_vwap", "v385_rule_crm_xom_long_orb", "v385_rule_coin_long_vwap", "v385_rule_xom_late_long_exception",
        "positive_context_profile_match",
        "positive_context_profile_name", "positive_context_profile_reason",
        "positive_context_dir_rs", "positive_context_dir_open_rs", "positive_context_dir_vwap",
        "positive_context_rvol_time_of_day", "positive_context_daily_atr14_percent",
        "positive_context_abs_gap", "positive_context_abs_qqq",
        "selected_trade_number_for_day",
        "selected_symbol_trade_number_for_day", "_eligible_day_count", "_eligible_order_in_day",
        "_eligible_same_timestamp_rank", "rvol_time_of_day", "daily_atr14_percent", "gap_percent",
        "day_relative_strength", "open_relative_strength", "vwap_extension_atr", "_directional_rs",
        "_directional_open_rs", "_directional_vwap_atr", "_abs_vwap_atr", "qqq_change_from_open",
        "qqq_day_change_percent", "_abs_qqq", "risk_per_share", "_risk_per_share_pct",
        "entry_candle_ok", "opposing_candle_warning_at_entry", "_catalyst_proxy",
    ]
    first = [c for c in preferred if c in out.columns]
    rest = [c for c in out.columns if c not in first]
    return out[first + rest]



def _strategy_meta_columns(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_strategy_profile": params.get("strategy_profile", ""),
        "report_direction_mode": params.get("direction_mode", ""),
        "report_backtest_decision_mode": params.get("backtest_decision_mode", ""),
        "report_live_quality_gate": params.get("live_quality_gate", ""),
        "report_live_strategy_variant": params.get("live_strategy_variant", ""),
        "report_live_strategy_preset": params.get("live_strategy_preset", ""),
        "report_v379_pattern_mode": params.get("v379_pattern_mode", ""),
        "report_enable_v358_gate": params.get("enable_v358_live_quality_filter", ""),
        "report_enable_v359_gate": params.get("enable_v359_live_hunter_filter", ""),
        "report_enable_v364_gate": params.get("enable_v364_professional_momentum_filter", ""),
        "report_enable_v377_gate": params.get("enable_v377_positive_context_filter", ""),
        "report_enable_v379_gate": params.get("enable_v379_decision_pattern_filter", ""),
    }


def _add_strategy_meta(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for k, v in _strategy_meta_columns(params).items():
        if k not in out.columns:
            out[k] = v
    return out

def generate_report_zip(result: dict[str, Any]) -> Path:
    """Write a complete diagnostic report for upload/review.

    The ZIP intentionally includes both selected trades and raw candidates. That
    lets us decide whether entries are bad, exits are bad, or portfolio rules are
    filtering the wrong trades.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = result.get("params", {}) or {}
    metrics = result.get("metrics", {}) or {}
    mode = _safe_name(params.get("strategy_profile", "strategy"))
    start = _safe_name(result.get("start_date", "start"))
    end = _safe_name(result.get("end_date", "end"))
    risk_value = metrics.get("risk_per_trade_dollars", params.get("risk_per_trade_dollars", "risk"))
    try:
        risk_label = f"risk{float(risk_value):g}"
    except Exception:
        risk_label = _safe_name(risk_value)
    folder = REPORT_DIR / f"backtest_{mode}_{start}_{end}_{risk_label}_{ts}"
    folder.mkdir(parents=True, exist_ok=True)

    selected_trades = _add_strategy_meta(result.get("selected_trades", pd.DataFrame()), params)
    candidates = _add_strategy_meta(result.get("candidates", pd.DataFrame()), params)
    portfolio_trades = _add_strategy_meta(result.get("portfolio_trades", pd.DataFrame()), params)
    tables = {
        "selected_trades.csv": selected_trades,
        "selected_trade_market_conditions.csv": _selected_trade_market_conditions(selected_trades),
        "raw_candidates.csv": candidates,
        "portfolio_trades_all.csv": portfolio_trades,
        "symbol_summary.csv": result.get("symbol_summary", pd.DataFrame()),
        "setup_summary.csv": result.get("setup_summary", pd.DataFrame()),
        "daily_summary.csv": result.get("daily_summary", pd.DataFrame()),
        "exit_summary.csv": result.get("exit_summary", pd.DataFrame()),
        "score_band_summary.csv": result.get("score_band_summary", pd.DataFrame()),
        "hour_summary.csv": result.get("hour_summary", pd.DataFrame()),
        "direction_summary.csv": result.get("direction_summary", pd.DataFrame()),
        "module_summary.csv": result.get("module_summary", pd.DataFrame()),
        "candle_pattern_summary.csv": result.get("candle_summary", pd.DataFrame()),
        "regime_summary.csv": result.get("regime_summary", pd.DataFrame()),
        "market_context_summary.csv": result.get("market_context_summary", pd.DataFrame()),
        "sizing_scenarios.csv": result.get("sizing_scenarios", pd.DataFrame()),
        "compact_alert_signals.csv": result.get("signals", pd.DataFrame()),
        "skipped_symbols.csv": result.get("skipped_symbols", pd.DataFrame()),
        "raw_replay_alerts_after_filters.csv": result.get("raw_replay_alerts_after_filters", pd.DataFrame()),
        "openai_decisions.csv": result.get("openai_decisions", pd.DataFrame()),
    }
    for filename, df in tables.items():
        _write_df(folder / filename, df)

    manifest = {
        "created_at": ts,
        "symbols": result.get("symbols", []),
        "start_date": result.get("start_date"),
        "end_date": result.get("end_date"),
        "feed": result.get("feed"),
        "use_news": result.get("use_news"),
        "metrics": result.get("metrics", {}),
        "diagnostics": result.get("diagnostics", {}),
        "params": result.get("params", {}),
        "notes": [
            "Upload this ZIP or selected CSV files when asking for strategy debugging.",
            "selected_trades.csv includes MFE/MAE so we can see if entries worked but exits failed.",
            "selected_trade_market_conditions.csv explains each selected trade with the market conditions and indicator values at execution.",
            "raw_candidates.csv helps diagnose whether scoring/ranking is predictive.",
            "candle_pattern_summary.csv shows whether candlestick confirmation improves or hurts the strategy.",
            "regime_summary.csv shows whether low-followthrough mode is still hurting or improving the system.",
            "market_context_summary.csv shows which V8/V12 market-condition bucket produced the trades.",
            "sizing_scenarios.csv shows what the same R-multiple sequence would have produced at $25/$50/$100/$200/$500 risk per trade.",
        ],
    }
    with open(folder / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, indent=2)

    zip_path = REPORT_DIR / f"{folder.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in folder.iterdir():
            zf.write(file, arcname=file.name)
    return zip_path


def export_backtest_report(result: dict[str, Any]) -> dict[str, str]:
    zip_path = generate_report_zip(result)
    latest_zip = REPORT_DIR / "latest_backtest_report.zip"
    try:
        latest_zip.write_bytes(zip_path.read_bytes())
    except Exception:
        latest_zip = zip_path
    return {
        "zip_path": str(zip_path),
        "latest_zip": str(latest_zip),
    }
