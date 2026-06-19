from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
LOCAL_BARS_DIR = PROJECT_ROOT / "data" / "local_bars"
RESEARCH_PACKS_DIR = PROJECT_ROOT / "data" / "research_packs"
REPORTS_DIR = PROJECT_ROOT / "reports"
ML_DATASETS_DIR = PROJECT_ROOT / "data" / "ml_datasets"
ML_MODELS_DIR = PROJECT_ROOT / "data" / "ml_models"
ML_BACKTESTS_DIR = PROJECT_ROOT / "data" / "ml_backtests"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_BARS_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_PACKS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ML_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
ML_BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class AlpacaSettings:
    api_key: str | None = os.getenv("ALPACA_API_KEY")
    secret_key: str | None = os.getenv("ALPACA_SECRET_KEY")
    account_id: str | None = os.getenv("ALPACA_ACCOUNT_ID")
    data_base_url: str = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")
    trading_base_url: str = os.getenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")
    default_feed: str = os.getenv("ALPACA_FEED", "iex")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def redacted(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("api_key"):
            data["api_key"] = f"***{data['api_key'][-4:]}"
        if data.get("secret_key"):
            data["secret_key"] = "***"
        return data


@dataclass
class StrategyParams:
    """Version 19 verified-core engine with local feature cache and exact-risk sizing.

    Main goals:
    - risk dollars are applied directly to trade P&L, not just printed in reports.
    - the weak broad VWAP-reclaim module is disabled by default.
    - the default strategy is the long-period-proven trend-pullback core.
    """

    strategy_profile: str = "symbol_playbook_v25"
    direction_mode: str = "long_short"  # long_only, short_only, or long_short

    # V25 Symbol/Event Playbook. Derived from direct raw-bar research on the local
    # 2022-2026 dataset. It is intentionally different from the broad ML/event attempts:
    # only whitelisted symbol+event+side combinations are eligible, and a developing
    # volume-profile reaction is required.
    v25_target_r: float = 0.75
    v25_stop_atr_mult: float = 0.60
    v25_min_stop_pct: float = 0.0015
    v25_max_hold_bars: int = 12
    v25_profile_bins: int = 48
    v25_profile_tolerance_atr: float = 0.18
    v25_min_rvol: float = 0.75

    # V27 optional risk filters for Symbol/Event Playbook replay.
    # These are off by default so V26 baseline can be reproduced.
    v27_macro_filter_mode: str = "off"  # off, skip, top1
    v27_market_stress_mode: str = "off"  # off, skip, top1
    v27_news_filter_mode: str = "off"  # off, skip, top1
    v27_symbol_kill_switch_mode: str = "off"  # off, moderate, strict
    v27_qqq_stress_abs_change_pct: float = 1.25
    v27_news_gap_abs_pct: float = 4.0
    v27_news_rvol_min: float = 3.0

    # Backtest setup / risk controls
    initial_account_value: float = 10_000.0
    # Fixed-dollar risk is still available for comparison.
    risk_per_trade_dollars: float = 100.0
    # Percent-equity risk is used by both pure compounding and controlled compounding.
    risk_per_trade_pct: float = 0.01
    requested_risk_percent: float = 1.0
    # V28 sizing modes:
    # - fixed_dollar_risk: always use risk_per_trade_dollars.
    # - percent_equity: risk current equity * risk_per_trade_pct with no drawdown brake.
    # - controlled_compounding: risk current equity * base %, with small-account floors, caps,
    #   and drawdown-based risk reduction / pause. This is the recommended default.
    position_sizing_mode: str = "controlled_compounding"
    max_position_notional_pct: float = 999.0
    compounding_base_risk_pct: float = 1.0
    compounding_min_risk_dollars: float = 10.0
    compounding_max_risk_dollars: float = 300.0
    compounding_max_risk_pct_of_equity: float = 1.25
    compounding_dd1_pct: float = 5.0
    compounding_dd1_risk_pct: float = 0.75
    compounding_dd2_pct: float = 10.0
    compounding_dd2_risk_pct: float = 0.50
    compounding_pause_dd_pct: float = 15.0
    # Fractional shares are used in the backtest so risk scaling is exact.
    # For live trading you can round down later in the execution layer if needed.
    min_shares: float = 0.000001
    slippage_bps: float = 3.0
    max_trades_per_day: int = 2
    max_open_positions: int = 3
    daily_loss_limit_pct: float = 100.0
    max_consecutive_losses: int = 99
    max_alerts_per_symbol_per_day: int = 1

    # V18 opportunity-discovery modules. These came from the independent opportunity dataset
    # rather than the old strategy candidates. Keep them separate from the legacy trend
    # pullback/VWAP modules so we can test whether the discovered opportunity families
    # survive a real trade simulator.
    enable_opportunity_v18: bool = False
    enable_v18_m1_early_low_atr_long: bool = False
    enable_v18_m2_early_low_atr_short: bool = False
    enable_v18_m3_controlled_gap: bool = False
    enable_v18_m4_10am_continuation: bool = False
    enable_v18_m5_11am_or_rejection: bool = False
    use_legacy_core_when_v18: bool = False
    m1_risk_multiplier: float = 1.00
    m2_risk_multiplier: float = 0.75
    m3_risk_multiplier: float = 1.00
    m4_risk_multiplier: float = 0.65
    m5_risk_multiplier: float = 0.50
    m1_target1_r: float = 0.75
    m1_target2_r: float = 1.25
    m1_target1_sell_pct: float = 0.70
    m1_target2_sell_pct: float = 0.30
    m2_target1_r: float = 0.60
    m2_target2_r: float = 1.00
    m2_target1_sell_pct: float = 0.80
    m2_target2_sell_pct: float = 0.20
    m3_target1_r: float = 0.75
    m3_target2_r: float = 1.50
    m3_target1_sell_pct: float = 0.60
    m3_target2_sell_pct: float = 0.40
    m4_target1_r: float = 0.55
    m4_target2_r: float = 1.00
    m4_target1_sell_pct: float = 0.80
    m4_target2_sell_pct: float = 0.20
    m5_target1_r: float = 0.50
    m5_target2_r: float = 0.80
    m5_target1_sell_pct: float = 1.00
    m5_target2_sell_pct: float = 0.00
    m1_max_hold_bars: int = 12
    m2_max_hold_bars: int = 12
    m3_max_hold_bars: int = 12
    m4_max_hold_bars: int = 10
    m5_max_hold_bars: int = 8

    # Setup modules
    enable_trend_pullback: bool = True
    # V13 adds a stricter EMA9/VWAP micro-pullback continuation setup to increase
    # opportunity count without re-enabling the broad VWAP-reclaim module.
    enable_micro_pullback: bool = True
    micro_pullback_min_score: float = 88.0
    micro_pullback_min_rvol: float = 1.10
    micro_pullback_max_vwap_extension_atr: float = 1.05
    micro_pullback_ema9_buffer_atr: float = 0.18
    micro_pullback_min_depth_atr: float = 0.10
    micro_pullback_min_day_rs: float = 0.50
    micro_pullback_min_open_rs: float = 0.00

    # V14 long-period robustness filters derived from the 2022-2026 report.
    # They target failed high-gap momentum-chase days and weak low-followthrough
    # days without removing the core trend-pullback setup.
    enable_v14_long_period_filters: bool = True

    # V19: do not use broad opportunity modules by default. V18 modules failed
    # badly in real first-touch simulation, so V19 defaults back to the verified
    # trend-pullback / micro-pullback core while leaving V18 available only as
    # explicit experimental mode.
    enable_v19_feature_cache: bool = True
    max_momentum_gap_percent: float = 8.0
    max_momentum_stock_day_change_percent: float = 12.0
    block_lowft_if_day_rs_below: float = 1.0
    block_lowft_if_open_rs_below: float = 1.0
    min_bullish_continuation_rvol: float = 1.35

    enable_or_retest: bool = False
    enable_or_retest_only_rejection: bool = True
    avoid_low_followthrough_momentum: bool = True
    # Disabled by default. The uploaded reports showed the broad VWAP-reclaim module
    # was the main long-period drag. The core trend-pullback module remains enabled.
    enable_vwap_reclaim_reversal: bool = False
    # V12 core filter derived from long reports: keep the strong trend-pullback module,
    # and allow VWAP-reclaim reversal only when it matches the actually profitable
    # subsets: 10:00 reversal window, bullish engulfing reclaim, or controlled
    # low-followthrough rejection. This removes the generic VWAP reclaim trades
    # that were the main long-period drag.
    enable_v12_core_filter: bool = True
    v12_morning_only: bool = True
    v12_vwap_window_start: str = "10:00"
    v12_vwap_window_end: str = "10:59"
    v12_vwap_reversal_min_score: float = 96.0
    v12_vwap_engulfing_min_score: float = 94.0
    v12_vwap_min_rvol: float = 1.45
    v12_vwap_lowft_rejection_min_rvol: float = 1.10
    v12_vwap_reversal_block_neutral: bool = True

    # V8 robust-regime filters derived from long-period diagnostics.
    # These are deliberately generic market-quality rules, not symbol-specific fitting.
    enable_v8_regime_filters: bool = True
    min_momentum_daily_atr_pct: float = 2.35
    high_vol_daily_atr_pct: float = 4.50
    extreme_vol_daily_atr_pct: float = 6.50
    max_clean_vwap_extension_atr: float = 1.20
    block_all_overextended_momentum: bool = True
    min_open_rs_clean: float = 1.00
    min_open_rs_high_vol: float = 1.20
    min_day_rs_high_vol: float = 2.00
    min_rvol_high_vol: float = 1.45
    weak_open_rs_upper: float = 1.00
    block_inside_breakout_entries: bool = True
    block_neutral_confirm_entries: bool = True
    allow_engulfing_only_with_volume_rs: bool = True

    # Candidate score
    min_candidate_score: float = 80.0
    high_quality_score: float = 88.0
    weak_market_min_score: float = 90.0

    # Time filters, Eastern Time
    primary_start: str = "09:40"
    primary_end: str = "12:00"
    midday_start: str = "11:30"
    midday_end: str = "13:45"
    afternoon_start: str = "16:00"
    afternoon_end: str = "16:00"
    exit_all_time: str = "15:45"
    use_lunch_exit: bool = False
    lunch_review_time: str = "12:30"

    # Liquidity / tradability
    min_price: float = 3.0
    min_avg_20d_dollar_volume: float = 75_000_000.0
    min_current_5m_dollar_volume: float = 350_000.0
    min_daily_atr_pct: float = 0.60
    max_daily_atr_pct: float = 9.00

    # Market filter
    qqq_atr14_daily_percent_max: float = 5.50
    qqq_max_intraday_loss_pct: float = -1.60
    qqq_15min_change_min_pct: float = -0.35
    weak_market_min_relative_strength: float = 2.75

    # Reason for move
    min_gap_percent: float = 0.50
    min_rvol_reason: float = 1.15
    min_day_relative_strength: float = 0.70
    min_open_relative_strength: float = 0.35

    # No-chase / overextension
    max_vwap_extension_atr: float = 1.45
    max_candle_range_atr: float = 1.90
    max_entry_chase_atr: float = 0.18

    # Trigger buffers
    vwap_reclaim_buffer_atr: float = 0.02
    breakout_buffer_atr: float = 0.04
    entry_breakout_buffer_atr: float = 0.02
    stop_trigger_buffer_atr: float = 0.12
    ema_pullback_buffer_atr: float = 0.18
    retest_buffer_atr: float = 0.35

    # Structure / Fibonacci context
    pullback_lookback_bars: int = 6
    min_pullback_depth_atr: float = 0.25
    fib_zone_min: float = 0.236
    fib_zone_max: float = 0.786
    fib_golden_min: float = 0.382
    fib_golden_max: float = 0.618

    # Candle / volume quality
    candle_close_position_min: float = 0.58
    candle_close_position_short_max: float = 0.42
    rvol_min: float = 1.10

    # Candlestick pattern module.
    # off     = do not use patterns in entries or exits.
    # score   = add/penalize score with candle pattern and use reversal candles for exits.
    # confirm = require a favorable entry candle and use reversal candles for exits.
    candle_pattern_mode: str = "off"
    candle_entry_bonus: float = 8.0
    candle_opposing_penalty: float = 10.0
    selective_rejection_bonus: float = 12.0
    selective_continuation_bonus: float = 2.0
    selective_weak_pattern_penalty: float = 12.0
    candle_exit_min_mfe_r: float = 0.20
    candle_exit_after_target1: bool = True

    # Risk and exits
    min_risk_atr: float = 0.55
    max_risk_atr: float = 1.45
    target1_r: float = 0.60
    target2_r: float = 1.20
    target1_sell_pct: float = 0.70
    target2_sell_pct: float = 0.25
    trailing_atr_multiple: float = 0.95
    breakeven_after_r: float = 0.40
    protective_stop_after_r: float = 0.35
    protective_stop_r: float = -0.08
    skip_entry_bar_exits: bool = True
    early_breakdown_candles: int = 1
    failure_candles: int = 2
    failure_min_r: float = 0.20

    # Mean-reversion module. It is off by default for first tests; turn on only if report shows trend continuation is weak.
    enable_mean_reversion: bool = False
    mr_min_vwap_extension_atr: float = 1.85
    mr_rsi2_long_max: float = 8.0
    mr_rsi2_short_min: float = 92.0
    mr_target1_r: float = 0.45
    mr_target2_r: float = 0.90
    mr_max_hold_bars: int = 10

    # V5 adaptive low-follow-through regime handling.
    # In the reports, Jan-Mar trades often moved +0.5R but failed to reach +0.75R.
    # Below this daily ATR threshold, use faster scalp exits instead of runner-style exits.
    low_followthrough_atr_pct: float = 4.00
    lowvol_target1_r: float = 0.45
    lowvol_target2_r: float = 0.85
    lowvol_target1_sell_pct: float = 0.90
    lowvol_target2_sell_pct: float = 0.05
    lowvol_breakeven_after_r: float = 0.30
    soft_failure_stop_r: float = 0.40

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
