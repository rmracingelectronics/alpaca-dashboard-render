from __future__ import annotations

import gc
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import numpy as np

from .alpaca_rest import AlpacaDataClient, pad_start_for_indicators
from .config import AlpacaSettings, StrategyParams, PROJECT_ROOT
from .indicators import add_intraday_features, add_daily_features, build_qqq_context, merge_market_context
from .strategy import compute_signals, simulate_candidates, apply_portfolio_rules, summarize_results
from .reporting import export_backtest_report



FEATURE_CACHE_DIR = PROJECT_ROOT / "data" / "feature_cache"
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)



def _v28_calculate_risk_budget(equity: float, high_watermark: float, params: StrategyParams) -> tuple[float, float, float, bool]:
    """Return (risk_budget, effective_risk_pct, drawdown_pct, paused).

    V28 controlled compounding is designed for small accounts too:
    it compounds using a percent of current equity, but the dollar floor is low
    ($10 default) and never allowed to exceed the configured max percent of equity.
    """
    equity = float(equity or 0.0)
    high_watermark = max(float(high_watermark or equity or 0.0), equity)
    if equity <= 0:
        return 0.0, 0.0, 100.0, True
    drawdown_pct = max(0.0, (high_watermark - equity) / high_watermark * 100.0) if high_watermark > 0 else 0.0
    mode = str(getattr(params, "position_sizing_mode", "controlled_compounding") or "controlled_compounding").lower()

    if mode == "fixed_dollar_risk":
        risk = float(getattr(params, "risk_per_trade_dollars", 100.0) or 100.0)
        pct = risk / equity * 100.0 if equity > 0 else 0.0
        return max(0.0, risk), pct, drawdown_pct, False

    if mode == "percent_equity":
        pct = float(getattr(params, "risk_per_trade_pct", 0.01) or 0.01) * 100.0
        risk = equity * pct / 100.0
        return max(0.0, risk), pct, drawdown_pct, False

    # Recommended V28 controlled compounding.
    pause_dd = float(getattr(params, "compounding_pause_dd_pct", 15.0) or 15.0)
    if drawdown_pct >= pause_dd:
        return 0.0, 0.0, drawdown_pct, True

    pct = float(getattr(params, "compounding_base_risk_pct", 1.0) or 1.0)
    dd1 = float(getattr(params, "compounding_dd1_pct", 5.0) or 5.0)
    dd2 = float(getattr(params, "compounding_dd2_pct", 10.0) or 10.0)
    if drawdown_pct >= dd2:
        pct = float(getattr(params, "compounding_dd2_risk_pct", 0.50) or 0.50)
    elif drawdown_pct >= dd1:
        pct = float(getattr(params, "compounding_dd1_risk_pct", 0.75) or 0.75)

    risk = equity * pct / 100.0
    min_risk = float(getattr(params, "compounding_min_risk_dollars", 10.0) or 0.0)
    max_risk = float(getattr(params, "compounding_max_risk_dollars", 300.0) or 0.0)
    max_pct = float(getattr(params, "compounding_max_risk_pct_of_equity", 1.25) or 1.25)
    if min_risk > 0:
        risk = max(risk, min_risk)
    if max_risk > 0:
        risk = min(risk, max_risk)
    if max_pct > 0:
        risk = min(risk, equity * max_pct / 100.0)
    risk = max(0.0, risk)
    effective_pct = risk / equity * 100.0 if equity > 0 else 0.0
    return risk, effective_pct, drawdown_pct, False

def _safe_key(value: str) -> str:
    return str(value).replace(":", "").replace("/", "-").replace(" ", "_")

def _feature_cache_path(symbol: str, feed: str, fetch_start: str, fetch_end: str) -> Any:
    # Cache fully prepared per-symbol features, including QQQ context, because
    # long 4-year tests spend most of their time recomputing indicators.
    return FEATURE_CACHE_DIR / str(feed).lower() / f"{symbol.upper()}_{_safe_key(fetch_start)}_{_safe_key(fetch_end)}_features.pkl.gz"

def _read_feature_cache(path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_pickle(path, compression="gzip")
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

def _write_feature_cache(path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path, compression="gzip")
    except Exception:
        pass

_SIGNAL_EXPORT_COLUMNS = [
    "symbol", "timestamp", "timestamp_ny", "session_date", "buy_alert", "short_alert",
    "candidate_score", "trigger_type", "entry_reason", "rvol_time_of_day",
    "day_relative_strength", "open_relative_strength", "close", "session_vwap",
    "low_followthrough_context", "v8_trade_context", "v8_regime_ok", "v8_candle_quality_ok",
    "setup_family", "opportunity_module", "v12_core_setup_ok", "v13_micro_quality_ok", "v14_filter_ok", "v15_quality_ok", "candle_pattern_primary", "entry_candle_ok", "opposing_candle_warning", "candle_pattern_score",
    "bullish_continuation_candle", "bullish_rejection_candle", "bullish_engulfing_candle",
    "bearish_rejection_candle", "bearish_engulfing_candle",
]


def _reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast safe numeric columns to reduce memory pressure on long backtests."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


def _small_signal_export(signals: pd.DataFrame) -> pd.DataFrame:
    """Keep only alert rows and only diagnostic columns, not all 5-minute feature bars."""
    if signals is None or signals.empty:
        return pd.DataFrame()
    mask = pd.Series(False, index=signals.index)
    if "buy_alert" in signals.columns:
        mask = mask | signals["buy_alert"].fillna(False).astype(bool)
    if "short_alert" in signals.columns:
        mask = mask | signals["short_alert"].fillna(False).astype(bool)
    if not mask.any():
        return pd.DataFrame()
    cols = [c for c in _SIGNAL_EXPORT_COLUMNS if c in signals.columns]
    return signals.loc[mask, cols].copy()




def _select_v25_top_n(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select top N candidates per session, never repeating the same symbol in one day.

    This reproduces the raw-data V25 research selection when no optional filters
    are applied, while allowing min-score and candlestick filters to be tested
    against the full approved candidate universe rather than only against a
    preselected top-N replay file.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "date" not in work.columns:
        work["date"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date.astype(str)
    sort_cols = ["date", "score", "timestamp"]
    ascending = [True, False, True]
    work = work.sort_values(sort_cols, ascending=ascending)
    selected_rows = []
    for _, g in work.groupby("date", sort=False):
        seen_symbols: set[str] = set()
        taken = 0
        for _, r in g.iterrows():
            sym = str(r.get("symbol", "")).upper()
            if sym in seen_symbols:
                continue
            selected_rows.append(r)
            seen_symbols.add(sym)
            taken += 1
            if taken >= int(top_n):
                break
    if not selected_rows:
        return pd.DataFrame(columns=work.columns)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _apply_v25_candlestick_filter(df: pd.DataFrame, candle_mode: str) -> pd.DataFrame:
    """Apply optional V25 entry-candle filters.

    Modes:
    - off / exit_only: no V25 entry candle filter, reproduces the original V25 replay.
    - selective: require side-aligned continuation/rejection/engulfing and block opposing candles.
    - confirm: require side-aligned continuation or engulfing candle.
    - score: require positive side-aligned candle score.
    """
    if df is None or df.empty:
        return df
    mode = str(candle_mode or "off").lower()
    if mode in {"off", "exit_only", "none", "false"}:
        return df
    out = df.copy()
    for col in ["entry_candle_ok", "side_continuation_candle", "side_rejection_candle", "side_engulfing_candle", "opposing_candle_warning"]:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False).astype(bool)
    if "candle_pattern_score" not in out.columns:
        out["candle_pattern_score"] = 0.0
    score = pd.to_numeric(out["candle_pattern_score"], errors="coerce").fillna(0.0)
    if mode == "selective":
        mask = out["entry_candle_ok"] & (~out["opposing_candle_warning"])
    elif mode == "confirm":
        mask = (out["side_continuation_candle"] | out["side_engulfing_candle"]) & (~out["opposing_candle_warning"])
    elif mode == "score":
        mask = (score > 0) & (~out["opposing_candle_warning"])
    else:
        mask = pd.Series(True, index=out.index)
    return out.loc[mask].copy()





def _v27_known_macro_dates(start_year: int = 2022, end_year: int = 2026) -> set[str]:
    """Approximate recurring macro-risk dates plus known shock dates.

    This is intentionally local/offline so the playbook replay can be compared
    without web/API dependencies. It is not meant to be a perfect economic
    calendar; it gives the dashboard a togglable macro/news-risk proxy.
    """
    dates: set[str] = set()
    for year in range(int(start_year), int(end_year) + 1):
        # NFP proxy: first Friday of each month.
        for month in range(1, 13):
            d = pd.Timestamp(year=year, month=month, day=1)
            while d.weekday() != 4:
                d += pd.Timedelta(days=1)
            dates.add(d.strftime("%Y-%m-%d"))
        # CPI/PPI proxy: 10th-14th business days. Broad by design; optional.
        for month in range(1, 13):
            for day in range(10, 15):
                d = pd.Timestamp(year=year, month=month, day=day)
                if d.weekday() < 5:
                    dates.add(d.strftime("%Y-%m-%d"))
        # FOMC proxy months: commonly scheduled Fed decision months.
        for month in [1, 3, 5, 6, 7, 9, 11, 12]:
            # third Wednesday of the month.
            d = pd.Timestamp(year=year, month=month, day=1)
            wednesdays = []
            while d.month == month:
                if d.weekday() == 2:
                    wednesdays.append(d)
                d += pd.Timedelta(days=1)
            if len(wednesdays) >= 3:
                dates.add(wednesdays[2].strftime("%Y-%m-%d"))
    # Specific historical macro/shock dates seen in the weak-period review.
    for d in [
        "2022-05-04", "2022-05-05", "2022-08-26", "2022-09-13",
        "2023-03-10", "2023-03-13", "2023-05-31", "2024-12-18",
        "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07", "2025-04-08", "2025-04-09",
    ]:
        dates.add(d)
    return dates


def _v27_add_risk_flags(df: pd.DataFrame, params: StrategyParams, use_news: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date.astype(str)
    macro_dates = _v27_known_macro_dates(2022, 2026)
    out["v27_macro_event_day"] = out["date"].astype(str).isin(macro_dates)
    qqq_abs = pd.to_numeric(out.get("qqq_chg_open", 0.0), errors="coerce").fillna(0.0).abs()
    qqq_thr = float(getattr(params, "v27_qqq_stress_abs_change_pct", 1.25) or 1.25)
    # Also treat very high stock/event RVOL as market stress only when paired with a large QQQ move.
    out["v27_market_stress_day"] = qqq_abs >= qqq_thr
    gap_abs = pd.to_numeric(out.get("gap_pct", 0.0), errors="coerce").fillna(0.0).abs()
    rvol = pd.to_numeric(out.get("rvol_tod", 0.0), errors="coerce").fillna(0.0)
    news_gap = float(getattr(params, "v27_news_gap_abs_pct", 4.0) or 4.0)
    news_rvol = float(getattr(params, "v27_news_rvol_min", 3.0) or 3.0)
    # This is an offline catalyst/news proxy. If API news exists in future, merge it into this flag.
    out["v27_news_catalyst_proxy"] = (gap_abs >= news_gap) | (rvol >= news_rvol)
    if use_news and "news_count_last_3d" in out.columns:
        out["v27_news_catalyst_proxy"] = out["v27_news_catalyst_proxy"] | (pd.to_numeric(out["news_count_last_3d"], errors="coerce").fillna(0) > 0)
    return out


def _v27_apply_preselection_filters(df: pd.DataFrame, params: StrategyParams, use_news: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    if df is None or df.empty:
        return df, {"macro_filtered": 0, "stress_filtered": 0, "news_filtered": 0}
    out = _v27_add_risk_flags(df, params, use_news=use_news)
    stats = {"macro_filtered": 0, "stress_filtered": 0, "news_filtered": 0}
    macro_mode = str(getattr(params, "v27_macro_filter_mode", "off") or "off").lower()
    stress_mode = str(getattr(params, "v27_market_stress_mode", "off") or "off").lower()
    news_mode = str(getattr(params, "v27_news_filter_mode", "off") or "off").lower()
    if macro_mode == "skip" and "v27_macro_event_day" in out.columns:
        mask = out["v27_macro_event_day"].fillna(False).astype(bool)
        stats["macro_filtered"] = int(mask.sum())
        out = out.loc[~mask].copy()
    if stress_mode == "skip" and "v27_market_stress_day" in out.columns:
        mask = out["v27_market_stress_day"].fillna(False).astype(bool)
        stats["stress_filtered"] = int(mask.sum())
        out = out.loc[~mask].copy()
    if news_mode == "skip" and "v27_news_catalyst_proxy" in out.columns:
        mask = out["v27_news_catalyst_proxy"].fillna(False).astype(bool)
        stats["news_filtered"] = int(mask.sum())
        out = out.loc[~mask].copy()
    return out, stats


def _v27_select_top_n_with_caps(df: pd.DataFrame, top_n: int, params: StrategyParams) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else None)
    work = df.copy()
    macro_mode = str(getattr(params, "v27_macro_filter_mode", "off") or "off").lower()
    stress_mode = str(getattr(params, "v27_market_stress_mode", "off") or "off").lower()
    news_mode = str(getattr(params, "v27_news_filter_mode", "off") or "off").lower()
    if "date" not in work.columns:
        work["date"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date.astype(str)
    work = work.sort_values(["date", "score", "timestamp"], ascending=[True, False, True])
    selected_rows = []
    for _, g in work.groupby("date", sort=False):
        cap = int(top_n)
        if macro_mode == "top1" and g.get("v27_macro_event_day", pd.Series(False, index=g.index)).fillna(False).astype(bool).any():
            cap = min(cap, 1)
        if stress_mode == "top1" and g.get("v27_market_stress_day", pd.Series(False, index=g.index)).fillna(False).astype(bool).any():
            cap = min(cap, 1)
        if news_mode == "top1" and g.get("v27_news_catalyst_proxy", pd.Series(False, index=g.index)).fillna(False).astype(bool).any():
            cap = min(cap, 1)
        seen_symbols: set[str] = set()
        taken = 0
        for _, r in g.iterrows():
            sym = str(r.get("symbol", "")).upper()
            if sym in seen_symbols:
                continue
            selected_rows.append(r)
            seen_symbols.add(sym)
            taken += 1
            if taken >= cap:
                break
    if not selected_rows:
        return pd.DataFrame(columns=work.columns)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _v27_apply_symbol_side_kill_switch(selected: pd.DataFrame, params: StrategyParams) -> tuple[pd.DataFrame, int]:
    if selected is None or selected.empty:
        return selected, 0
    mode = str(getattr(params, "v27_symbol_kill_switch_mode", "off") or "off").lower()
    if mode in {"off", "none", "false"}:
        selected = selected.copy()
        selected["v27_kill_switch_skipped"] = False
        return selected, 0
    if mode == "strict":
        lookback, threshold = 10, -2.0
    else:
        lookback, threshold = 20, -3.0
    work = selected.sort_values("timestamp").copy().reset_index(drop=True)
    history: dict[str, list[float]] = {}
    skipped = []
    last_month: str | None = None
    kept_rows = []
    for _, r in work.iterrows():
        ts = pd.Timestamp(r["timestamp"])
        month = ts.tz_convert("America/New_York").strftime("%Y-%m") if ts.tzinfo else ts.strftime("%Y-%m")
        if last_month is None:
            last_month = month
        elif month != last_month:
            # Reset each month so an old bad patch does not permanently block a symbol.
            history = {}
            last_month = month
        key = f"{str(r.get('symbol','')).upper()}|{str(r.get('side','')).lower()}"
        recent = history.get(key, [])[-lookback:]
        paused = len(recent) >= max(3, min(lookback, 5)) and sum(recent) <= threshold
        if paused:
            skipped.append(True)
            continue
        row = r.copy()
        row["v27_kill_switch_skipped"] = False
        kept_rows.append(row)
        history.setdefault(key, []).append(float(r.get("r075", r.get("r_multiple", 0.0))))
    if not kept_rows:
        out = pd.DataFrame(columns=list(work.columns) + ["v27_kill_switch_skipped"])
    else:
        out = pd.DataFrame(kept_rows).reset_index(drop=True)
    return out, int(len(skipped))

def _run_v25_research_replay(
    symbols: list[str],
    start_date: str,
    end_date: str,
    params: StrategyParams,
    feed: str,
    export_report: bool = True,
    use_news: bool = False,
) -> dict[str, Any]:
    """Replay the V25 raw-data research candidates with optional filters.

    V26 update:
    - Uses the full approved V25 candidate universe, then applies UI filters.
    - Min score is now active for V25. Use 0 to reproduce the original V25 replay.
    - Candlestick mode is now active for V25.
    - Top 1/2/3 trades per day are selected after the optional filters.
    """
    direction = str(getattr(params, "direction_mode", "long_short") or "long_short").lower()
    if direction not in {"long_only", "short_only", "long_short"}:
        direction = "long_short"
    top_n = int(getattr(params, "max_trades_per_day", 2) or 2)
    top_n = max(1, min(3, top_n))
    all_path = PROJECT_ROOT / "data" / "v25_research" / "v25_candidates_all.csv.gz"
    fallback_path = PROJECT_ROOT / "data" / "v25_research" / f"v25_{direction}_top{top_n}.csv"
    if all_path.exists():
        df = pd.read_csv(all_path)
        candidate_source = str(all_path)
    elif fallback_path.exists():
        df = pd.read_csv(fallback_path)
        candidate_source = str(fallback_path)
    else:
        raise RuntimeError(f"V25 research replay file not found: {all_path} or {fallback_path}")
    raw_loaded_count = int(len(df))
    v27_filter_stats = {"macro_filtered": 0, "stress_filtered": 0, "news_filtered": 0}
    v27_kill_skipped = 0
    if df.empty:
        selected = pd.DataFrame()
        filtered_count = 0
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        start_ts = pd.Timestamp(start_date, tz="America/New_York").tz_convert("UTC")
        end_ts = pd.Timestamp(_plus_one_day(end_date), tz="America/New_York").tz_convert("UTC")
        requested = {s.upper() for s in symbols if s and s.upper() != "QQQ"}
        if requested:
            df = df[df["symbol"].astype(str).str.upper().isin(requested)].copy()
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
        if direction == "long_only":
            df = df[df["side"].astype(str).str.lower() == "long"].copy()
        elif direction == "short_only":
            df = df[df["side"].astype(str).str.lower() == "short"].copy()
        # V26: active min-score filter for V25. The raw V25 score normally ranges
        # from about 14 to 55. Set min score to 0 to reproduce the original replay.
        min_score = float(getattr(params, "min_candidate_score", 0.0) or 0.0)
        if min_score > 0 and "score" in df.columns:
            df = df[pd.to_numeric(df["score"], errors="coerce").fillna(-9999.0) >= min_score].copy()
        candle_mode = str(getattr(params, "candle_pattern_mode", "off") or "off")
        df = _apply_v25_candlestick_filter(df, candle_mode)
        df, v27_filter_stats = _v27_apply_preselection_filters(df, params, use_news=use_news)
        filtered_count = int(len(df))
        if all_path.exists():
            df = _v27_select_top_n_with_caps(df, top_n, params)
        else:
            df = df.sort_values("timestamp").reset_index(drop=True)
        df, v27_kill_skipped = _v27_apply_symbol_side_kill_switch(df, params)
        rows = []
        equity = float(params.initial_account_value)
        high_watermark = equity
        skipped_compounding_pause = 0
        for _, r in df.iterrows():
            r_mult = float(r.get("r075", 0.0))
            risk_budget, effective_risk_pct, drawdown_before_trade_pct, risk_paused = _v28_calculate_risk_budget(equity, high_watermark, params)
            if risk_paused or risk_budget <= 0:
                skipped_compounding_pause += 1
                continue
            pnl = r_mult * risk_budget
            entry_price = float(r.get("close", r.get("bar_close", 0.0))) if pd.notna(r.get("close", r.get("bar_close", np.nan))) else 0.0
            atr20 = float(r.get("atr20", 0.0)) if pd.notna(r.get("atr20", np.nan)) else max(entry_price * 0.0015, 0.01)
            risk_per_share = max(entry_price * 0.0015, 0.60 * atr20) if entry_price > 0 else 1.0
            side = str(r.get("side", "long")).lower()
            stop_price = entry_price - risk_per_share if side == "long" else entry_price + risk_per_share
            target1 = entry_price + 0.75 * risk_per_share if side == "long" else entry_price - 0.75 * risk_per_share
            entry_time = pd.Timestamp(r["timestamp"])
            exit_time = entry_time + pd.Timedelta(minutes=60)
            equity_at_entry = equity
            equity += pnl
            high_watermark = max(high_watermark, equity)
            shares = risk_budget / risk_per_share if risk_per_share > 0 else 0.0
            event = str(r.get("event", "v25_event"))
            candle_primary = str(r.get("candle_pattern_primary", "profile_reaction"))
            candle_ok = bool(r.get("entry_candle_ok", False)) if pd.notna(r.get("entry_candle_ok", False)) else False
            candle_warning = bool(r.get("opposing_candle_warning", False)) if pd.notna(r.get("opposing_candle_warning", False)) else False
            rows.append({
                "symbol": str(r.get("symbol", "")),
                "side": side,
                "signal_time": entry_time,
                "signal_time_et": entry_time.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
                "entry_time": entry_time,
                "entry_time_et": entry_time.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
                "entry_hour_et": entry_time.tz_convert("America/New_York").strftime("%H:00"),
                "exit_time": exit_time,
                "exit_time_et": exit_time.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
                "session_date": entry_time.tz_convert("America/New_York").date(),
                "trigger_type": "v25_" + event,
                "setup_family": "v25_symbol_playbook_replay",
                "opportunity_module": "v25_profile_playbook_replay",
                "candidate_score": float(r.get("score", r.get("robust_score", 0.0))) if pd.notna(r.get("score", np.nan)) else float(r.get("fallback_score", 0.0)),
                "quality": "positive_all_splits_volume_profile_replay",
                "entry_candle_pattern": candle_primary,
                "candle_pattern_primary": candle_primary,
                "entry_candle_ok": candle_ok,
                "opposing_candle_warning_at_entry": candle_warning,
                "opposing_candle_warning": candle_warning,
                "candle_pattern_score": float(r.get("candle_pattern_score", 0.0)) if pd.notna(r.get("candle_pattern_score", np.nan)) else 0.0,
                "candle_pattern_mode": candle_mode,
                "bullish_continuation_candle": bool(r.get("bullish_continuation_candle", False)) if pd.notna(r.get("bullish_continuation_candle", False)) else False,
                "bullish_rejection_candle": bool(r.get("bullish_rejection_candle", False)) if pd.notna(r.get("bullish_rejection_candle", False)) else False,
                "bullish_engulfing_candle": bool(r.get("bullish_engulfing_candle", False)) if pd.notna(r.get("bullish_engulfing_candle", False)) else False,
                "bearish_continuation_candle": bool(r.get("bearish_continuation_candle", False)) if pd.notna(r.get("bearish_continuation_candle", False)) else False,
                "bearish_rejection_candle": bool(r.get("bearish_rejection_candle", False)) if pd.notna(r.get("bearish_rejection_candle", False)) else False,
                "bearish_engulfing_candle": bool(r.get("bearish_engulfing_candle", False)) if pd.notna(r.get("bearish_engulfing_candle", False)) else False,
                "inside_bar": bool(r.get("inside_bar", False)) if pd.notna(r.get("inside_bar", False)) else False,
                "entry_trigger_price": entry_price,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target1": target1,
                "target2": target1,
                "risk_per_share": risk_per_share,
                "shares": shares,
                "notional": shares * entry_price,
                "notional_pct_of_equity": (shares * entry_price / equity_at_entry * 100) if equity_at_entry else 0.0,
                "risk_budget": risk_budget,
                "base_risk_per_trade_dollars": risk_budget,
                "risk_per_trade_dollars": risk_budget,
                "requested_risk_percent": float(getattr(params, "requested_risk_percent", 0.0) or 0.0),
                "risk_per_trade_percent": effective_risk_pct,
                "effective_risk_pct": effective_risk_pct,
                "drawdown_before_trade_pct": drawdown_before_trade_pct,
                "compounding_high_watermark_before_trade": high_watermark,
                "actual_dollars_at_risk": risk_budget,
                "actual_risk_pct_of_equity": effective_risk_pct,
                "requested_risk_shares": shares,
                "cash_cap_shares": 0.0,
                "sizing_cap_applied": False,
                "position_sizing_mode": str(getattr(params, "position_sizing_mode", "controlled_compounding") or "controlled_compounding"),
                "max_position_notional_pct": float(getattr(params, "max_position_notional_pct", 999.0)),
                "pnl_per_share": r_mult * risk_per_share,
                "pnl_dollars": pnl,
                "pnl_dollars_from_shares": pnl,
                "risk_application_delta": 0.0,
                "r_multiple": r_mult,
                "mfe_r": max(float(r.get("r075", 0.0)), 0.75 if r_mult > 0 else 0.0),
                "mae_r": -1.0 if r_mult < 0 else min(0.0, r_mult),
                "target1_hit": r_mult > 0,
                "target2_hit": r_mult > 0,
                "exit_reason": "v25_target_0_75r" if r_mult > 0 else "v25_stop_first",
                "duration_minutes": 60.0,
                "gap_percent": float(r.get("gap_pct", np.nan)) if pd.notna(r.get("gap_pct", np.nan)) else np.nan,
                "rvol_time_of_day": float(r.get("rvol_tod", np.nan)) if pd.notna(r.get("rvol_tod", np.nan)) else np.nan,
                "day_relative_strength": float(r.get("rs_open", np.nan)) if pd.notna(r.get("rs_open", np.nan)) else np.nan,
                "open_relative_strength": float(r.get("rs_open", np.nan)) if pd.notna(r.get("rs_open", np.nan)) else np.nan,
                "stock_day_change_percent": float(r.get("chg_open", np.nan)) if pd.notna(r.get("chg_open", np.nan)) else np.nan,
                "qqq_day_change_percent": float(r.get("qqq_chg_open", np.nan)) if pd.notna(r.get("qqq_chg_open", np.nan)) else np.nan,
                "v27_macro_event_day": bool(r.get("v27_macro_event_day", False)) if pd.notna(r.get("v27_macro_event_day", False)) else False,
                "v27_market_stress_day": bool(r.get("v27_market_stress_day", False)) if pd.notna(r.get("v27_market_stress_day", False)) else False,
                "v27_news_catalyst_proxy": bool(r.get("v27_news_catalyst_proxy", False)) if pd.notna(r.get("v27_news_catalyst_proxy", False)) else False,
                "v27_macro_filter_mode": str(getattr(params, "v27_macro_filter_mode", "off") or "off"),
                "v27_market_stress_mode": str(getattr(params, "v27_market_stress_mode", "off") or "off"),
                "v27_news_filter_mode": str(getattr(params, "v27_news_filter_mode", "off") or "off"),
                "v27_symbol_kill_switch_mode": str(getattr(params, "v27_symbol_kill_switch_mode", "off") or "off"),
                "vwap_extension_atr": float(r.get("vwap_ext_atr", np.nan)) if pd.notna(r.get("vwap_ext_atr", np.nan)) else np.nan,
                "daily_atr14_percent": float(r.get("daily_atr_pct", np.nan)) if pd.notna(r.get("daily_atr_pct", np.nan)) else np.nan,
                "low_followthrough_mode": False,
                "v8_trade_context": "v25_volume_profile_replay",
                "v8_regime_ok": True,
                "v8_candle_quality_ok": True,
                "selected": True,
                "skip_reason": "",
                "equity_at_entry": equity_at_entry,
            })
        selected = pd.DataFrame(rows)
    summary = summarize_results(selected, params)
    candidates = selected.copy()
    summary.update({
        "signals": candidates,
        "candidates": candidates,
        "portfolio_trades": selected,
        "params": params.to_dict(),
        "symbols": [s.upper() for s in symbols if s and s.upper() != "QQQ"],
        "feed": feed,
        "start_date": start_date,
        "end_date": end_date,
        "use_news": bool(use_news),
        "execution_timeframe": "5Min V27 macro/news filterable symbol playbook replay",
        "skipped_symbols": pd.DataFrame(),
        "diagnostics": {
            "symbols_scanned": len(symbols),
            "raw_candidates_loaded": raw_loaded_count,
            "candidates_after_date_symbol_direction_score_candle_filters": filtered_count,
            "selected_trades": int(len(selected)),
            "execution_timeframe": "5Min V27 macro/news filterable symbol playbook replay",
            "memory_mode": "embedded_v25_candidate_universe_with_filters_macro_news",
            "direction_mode": direction,
            "top_trades_per_day": top_n,
            "min_score_filter": float(getattr(params, "min_candidate_score", 0.0) or 0.0),
            "candle_pattern_mode": str(getattr(params, "candle_pattern_mode", "off") or "off"),
            "v27_macro_filter_mode": str(getattr(params, "v27_macro_filter_mode", "off") or "off"),
            "v27_market_stress_mode": str(getattr(params, "v27_market_stress_mode", "off") or "off"),
            "v27_news_filter_mode": str(getattr(params, "v27_news_filter_mode", "off") or "off"),
            "v27_symbol_kill_switch_mode": str(getattr(params, "v27_symbol_kill_switch_mode", "off") or "off"),
            "v27_macro_filtered_candidates": int(v27_filter_stats.get("macro_filtered", 0)),
            "v27_market_stress_filtered_candidates": int(v27_filter_stats.get("stress_filtered", 0)),
            "v27_news_filtered_candidates": int(v27_filter_stats.get("news_filtered", 0)),
            "v27_kill_switch_skipped_selected_trades": int(v27_kill_skipped),
            "v28_compounding_pause_skipped_trades": int(locals().get("skipped_compounding_pause", 0)),
            "v28_position_sizing_mode": str(getattr(params, "position_sizing_mode", "controlled_compounding") or "controlled_compounding"),
            "v28_compounding_base_risk_pct": float(getattr(params, "compounding_base_risk_pct", 0.0) or 0.0),
            "v28_compounding_min_risk_dollars": float(getattr(params, "compounding_min_risk_dollars", 0.0) or 0.0),
            "v28_compounding_max_risk_dollars": float(getattr(params, "compounding_max_risk_dollars", 0.0) or 0.0),
            "v28_compounding_pause_dd_pct": float(getattr(params, "compounding_pause_dd_pct", 0.0) or 0.0),
            "data_file": candidate_source,
            "important_note": "V27 applies score, candlestick, macro/news, QQQ stress, and symbol-side kill-switch filters before/after selecting Top 1/2/3 per day. Turn filters off to reproduce V26 baseline.",
        },
    })
    if "hour_summary" in summary and "time_bucket_summary" not in summary:
        summary["time_bucket_summary"] = summary.get("hour_summary", pd.DataFrame())
    if "mfe_mae_summary" not in summary:
        summary["mfe_mae_summary"] = pd.DataFrame()
    if export_report:
        try:
            summary["report_paths"] = export_backtest_report(summary)
        except Exception as exc:
            summary["report_paths"] = {"error": str(exc)}
    return summary

def run_backtest(
    symbols: list[str],
    start_date: str,
    end_date: str,
    params: StrategyParams | None = None,
    feed: str | None = None,
    use_cache: bool = True,
    use_news: bool = False,
    export_report: bool = True,
) -> dict[str, Any]:
    """Run the backtest using a streaming, symbol-by-symbol pipeline.

    Earlier V4 builds fetched and combined all 5-minute bars/features for every
    symbol before computing signals. That is fast for short ranges, but it can
    consume too much RAM on long tests. This version builds QQQ context once,
    then processes each tradable symbol independently and keeps only candidates
    and compact alert diagnostics in memory.
    """
    params = params or StrategyParams()
    settings = AlpacaSettings()
    feed = feed or settings.default_feed or "iex"
    if str(getattr(params, "strategy_profile", "")).lower() == "symbol_playbook_v25":
        return _run_v25_research_replay(symbols, start_date, end_date, params, feed, export_report=export_report, use_news=use_news)
    client = AlpacaDataClient(settings)

    fetch_start = pad_start_for_indicators(start_date, days=55)
    fetch_end = _plus_one_day(end_date)

    tradable_symbols = [s.upper() for s in symbols if s and s.upper() != "QQQ"]
    tradable_symbols = list(dict.fromkeys(tradable_symbols))
    if not tradable_symbols:
        raise RuntimeError("No tradable symbols supplied.")

    # V15: build/update the local bar store once, before the symbol loop.
    # This avoids calling Alpaca separately for every symbol/year during a long
    # 4-year test. After the first preload, tests over overlapping ranges run
    # almost completely from local disk.
    cache_status: list[dict[str, Any]] = []
    if use_cache:
        all_cache_symbols = list(dict.fromkeys(["QQQ"] + tradable_symbols))
        cache_status.append(client.prefetch_stock_bars(
            all_cache_symbols, "5Min", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=True
        ))
        cache_status.append(client.prefetch_stock_bars(
            all_cache_symbols, "1Day", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=True
        ))

    # Build market regime context once.
    qqq_5m = client.get_stock_bars(
        symbols=["QQQ"],
        timeframe="5Min",
        start=fetch_start,
        end=fetch_end,
        feed=feed,
        adjustment="split",
        use_cache=use_cache,
    )
    qqq_daily = client.get_stock_bars(
        symbols=["QQQ"],
        timeframe="1Day",
        start=fetch_start,
        end=fetch_end,
        feed=feed,
        adjustment="split",
        use_cache=use_cache,
    )
    if qqq_5m.empty or qqq_daily.empty:
        raise RuntimeError("QQQ data is required for the market filter but was not returned.")
    qqq_context = _reduce_memory(build_qqq_context(qqq_5m, qqq_daily))
    del qqq_5m, qqq_daily
    gc.collect()

    news = pd.DataFrame()
    if use_news:
        news = client.get_news_counts_by_day(tradable_symbols, fetch_start, fetch_end, use_cache=use_cache)
        if not news.empty:
            news["session_date"] = pd.to_datetime(news["session_date"]).dt.date

    start_ts = pd.Timestamp(start_date, tz="America/New_York")
    end_ts = pd.Timestamp(_plus_one_day(end_date), tz="America/New_York")

    candidate_frames: list[pd.DataFrame] = []
    signal_export_frames: list[pd.DataFrame] = []
    skipped_symbols: list[dict[str, Any]] = []
    scan_days_seen: set[Any] = set()
    raw_alert_count = 0

    for symbol in tradable_symbols:
        try:
            sym_5m = client.get_stock_bars(
                symbols=[symbol],
                timeframe="5Min",
                start=fetch_start,
                end=fetch_end,
                feed=feed,
                adjustment="split",
                use_cache=use_cache,
            )
            sym_daily = client.get_stock_bars(
                symbols=[symbol],
                timeframe="1Day",
                start=fetch_start,
                end=fetch_end,
                feed=feed,
                adjustment="split",
                use_cache=use_cache,
            )
            if sym_5m.empty or sym_daily.empty:
                skipped_symbols.append({"symbol": symbol, "reason": "missing_bars"})
                continue

            # V19 feature cache: after the first run, reuse prepared features
            # from disk instead of recomputing intraday, daily and QQQ merges.
            merged = pd.DataFrame()
            feature_cache_path = _feature_cache_path(symbol, feed, fetch_start, fetch_end)
            if use_cache and bool(getattr(params, "enable_v19_feature_cache", True)) and not use_news:
                merged = _read_feature_cache(feature_cache_path)

            if merged.empty:
                intraday = add_intraday_features(sym_5m)
                intraday = add_daily_features(intraday, sym_daily)
                merged = merge_market_context(intraday, qqq_context)
                merged = _reduce_memory(merged)
                if use_cache and bool(getattr(params, "enable_v19_feature_cache", True)) and not use_news:
                    _write_feature_cache(feature_cache_path, merged)

            if use_news and not news.empty:
                sym_news = news[news["symbol"] == symbol]
                if not sym_news.empty:
                    merged = merged.merge(sym_news, on=["symbol", "session_date"], how="left")
            if "news_count_last_3d" in merged.columns:
                merged["news_count_last_3d"] = merged["news_count_last_3d"].fillna(0)
            merged = _reduce_memory(merged)

            signals = compute_signals(merged, params)
            signals = signals[(signals["timestamp_ny"] >= start_ts) & (signals["timestamp_ny"] < end_ts)].copy()
            if signals.empty:
                skipped_symbols.append({"symbol": symbol, "reason": "no_rows_in_requested_window"})
                continue

            if "session_date" in signals.columns:
                scan_days_seen.update(signals["session_date"].dropna().unique().tolist())
            if "buy_alert" in signals.columns:
                raw_alert_count += int(signals["buy_alert"].fillna(False).astype(bool).sum())
            if "short_alert" in signals.columns:
                raw_alert_count += int(signals["short_alert"].fillna(False).astype(bool).sum())

            sig_export = _small_signal_export(signals)
            if not sig_export.empty:
                signal_export_frames.append(_reduce_memory(sig_export))

            candidates = simulate_candidates(signals, params)
            if not candidates.empty:
                candidate_frames.append(_reduce_memory(candidates))

        except MemoryError:
            raise RuntimeError(
                f"Memory limit hit while processing {symbol}. Try a smaller watchlist or date range, or use the memory-safe package with caching enabled."
            )
        except Exception as exc:
            skipped_symbols.append({"symbol": symbol, "reason": f"error: {exc}"})
        finally:
            for name in ["sym_5m", "sym_daily", "intraday", "merged", "signals", "candidates"]:
                if name in locals():
                    try:
                        del locals()[name]
                    except Exception:
                        pass
            gc.collect()

    candidates_all = pd.concat(candidate_frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True) if candidate_frames else pd.DataFrame()
    signals_export = pd.concat(signal_export_frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True) if signal_export_frames else pd.DataFrame()

    portfolio_trades = apply_portfolio_rules(candidates_all, params) if not candidates_all.empty else pd.DataFrame()
    summary = summarize_results(portfolio_trades, params)

    if "hour_summary" in summary and "time_bucket_summary" not in summary:
        summary["time_bucket_summary"] = summary.get("hour_summary", pd.DataFrame())
    selected_tmp = summary.get("selected_trades", pd.DataFrame())
    if "mfe_mae_summary" not in summary:
        if selected_tmp is not None and not selected_tmp.empty:
            metrics_tmp = summary.get("metrics", {})
            summary["mfe_mae_summary"] = pd.DataFrame([
                {"metric": "avg_mfe_r", "value": metrics_tmp.get("avg_mfe_r", 0), "meaning": "How far trades moved in favor before exit"},
                {"metric": "avg_mae_r", "value": metrics_tmp.get("avg_mae_r", 0), "meaning": "How far trades moved against you before exit"},
                {"metric": "target1_hit_rate", "value": metrics_tmp.get("target1_hit_rate", 0), "meaning": "If low, entries are weak or target1 is too far"},
                {"metric": "avg_win_r", "value": metrics_tmp.get("avg_win_r", 0), "meaning": "If low, exits may be too fast"},
                {"metric": "avg_loss_r", "value": metrics_tmp.get("avg_loss_r", 0), "meaning": "If near -1R, failure exits are too late"},
            ])
        else:
            summary["mfe_mae_summary"] = pd.DataFrame()

    scan_days = len(scan_days_seen)
    selected_count = int(len(summary.get("selected_trades", pd.DataFrame())))
    trade_days = int(summary.get("metrics", {}).get("trade_days", 0))

    summary.update(
        {
            "signals": signals_export,
            "candidates": candidates_all,
            "portfolio_trades": portfolio_trades,
            "params": params.to_dict(),
            "symbols": tradable_symbols,
            "feed": feed,
            "start_date": start_date,
            "end_date": end_date,
            "use_news": use_news,
            "execution_timeframe": "5Min",
            "skipped_symbols": pd.DataFrame(skipped_symbols),
            "diagnostics": {
                "symbols_scanned": len(tradable_symbols),
                "symbols_skipped": len(skipped_symbols),
                "scan_days": scan_days,
                "raw_alerts": raw_alert_count,
                "raw_candidates": int(len(candidates_all)),
                "selected_trades": selected_count,
                "trade_days": trade_days,
                "selected_trades_per_scan_day": selected_count / scan_days if scan_days else 0.0,
                "raw_alerts_per_scan_day": raw_alert_count / scan_days if scan_days else 0.0,
                "execution_timeframe": "5Min",
                "memory_mode": "local_bar_store_plus_v19_feature_cache",
                "cache_status": cache_status,
            },
        }
    )

    if export_report:
        try:
            report_paths = export_backtest_report(summary)
            summary["report_paths"] = report_paths
        except Exception as exc:
            summary["report_paths"] = {"error": str(exc)}
    return summary


def _plus_one_day(date_str: str) -> str:
    return (datetime.fromisoformat(date_str) + timedelta(days=1)).date().isoformat()
