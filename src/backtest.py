from __future__ import annotations

import gc
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import numpy as np

from .alpaca_rest import AlpacaDataClient, pad_start_for_indicators
from .config import AlpacaSettings, StrategyParams, PROJECT_ROOT
from .indicators import add_intraday_features, add_daily_features, build_qqq_context, merge_market_context, ema
from .positive_context_profiles import apply_positive_context_profile_filter
from .decision_pattern_scorer import apply_decision_time_pattern_scorer
from .strategy import compute_signals, simulate_candidates, apply_portfolio_rules, summarize_results, _simulate_trade
from .reporting import export_backtest_report
from .openai_trade_filter import review_candidates_with_openai
from .q_learning_policy import apply_q_policy, load_q_model
from .ml_ranker_policy import load_ranker_model, score_candidates as score_ml_ranker_candidates, live_style_select_by_score as ml_ranker_select



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

def _feature_cache_path(symbol: str, feed: str, fetch_start: str, fetch_end: str, session_mode: str = "regular_only") -> Any:
    # Cache fully prepared per-symbol features, including QQQ context, because
    # long 4-year tests spend most of their time recomputing indicators.  The
    # session mode is part of the key so extended-hours feature frames never
    # reuse regular-session-only features.
    mode = _safe_key(str(session_mode or "regular_only"))
    return FEATURE_CACHE_DIR / str(feed).lower() / mode / f"{symbol.upper()}_{_safe_key(fetch_start)}_{_safe_key(fetch_end)}_features.pkl.gz"

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


def _series_default(df: pd.DataFrame, col: str, default: Any = 0.0) -> pd.Series:
    """Return a full-length Series for optional columns.

    DataFrame.get(col, scalar) returns a scalar when the column is missing.
    Passing that scalar through pd.to_numeric(...).fillna(...) raises
    ``float object has no attribute fillna``. Raw-bar replay can legally miss
    some replay-only alias columns, so use this helper for robust defaults.
    """
    if df is not None and col in df.columns:
        return df[col]
    idx = df.index if df is not None else None
    return pd.Series(default, index=idx)


def _num_series_default(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(_series_default(df, col, default), errors="coerce").fillna(default)


def _bool_series_default(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    return _series_default(df, col, default).fillna(default).astype(bool)


def _v27_add_risk_flags(df: pd.DataFrame, params: StrategyParams, use_news: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date.astype(str)
    macro_dates = _v27_known_macro_dates(2022, 2026)
    out["v27_macro_event_day"] = out["date"].astype(str).isin(macro_dates)
    qqq_abs = _num_series_default(out, "qqq_chg_open", 0.0).abs()
    qqq_thr = float(getattr(params, "v27_qqq_stress_abs_change_pct", 1.25) or 1.25)
    # Also treat very high stock/event RVOL as market stress only when paired with a large QQQ move.
    out["v27_market_stress_day"] = qqq_abs >= qqq_thr
    gap_abs = _num_series_default(out, "gap_pct", 0.0).abs()
    rvol = _num_series_default(out, "rvol_tod", 0.0)
    news_gap = float(getattr(params, "v27_news_gap_abs_pct", 4.0) or 4.0)
    news_rvol = float(getattr(params, "v27_news_rvol_min", 3.0) or 3.0)
    # This is an offline catalyst/news proxy. V36.2 fix: when the UI switch
    # "Use news/catalyst proxy?" is No, the catalyst proxy must be completely
    # inactive.  Previously gap/RVOL still triggered the proxy even when use_news
    # was false, so users could change the UI and see no real effect.
    if use_news:
        out["v27_news_catalyst_proxy"] = (gap_abs >= news_gap) | (rvol >= news_rvol)
        if "news_count_last_3d" in out.columns:
            out["v27_news_catalyst_proxy"] = out["v27_news_catalyst_proxy"] | (pd.to_numeric(out["news_count_last_3d"], errors="coerce").fillna(0) > 0)
    else:
        out["v27_news_catalyst_proxy"] = pd.Series(False, index=out.index)
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
    if "score" not in work.columns and "candidate_score" in work.columns:
        work["score"] = pd.to_numeric(work["candidate_score"], errors="coerce")
    work["_selection_score"] = pd.to_numeric(work.get("score", 0.0), errors="coerce").fillna(-999999.0)
    work["_selection_ts"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.sort_values(["date", "_selection_score", "_selection_ts"], ascending=[True, False, True])
    selected_rows = []
    for day_key, g in work.groupby("date", sort=False):
        g = g.copy().reset_index(drop=False)
        cap = int(top_n)
        cap_reason = "requested_top_n"
        if macro_mode == "top1" and g.get("v27_macro_event_day", pd.Series(False, index=g.index)).fillna(False).astype(bool).any():
            cap = min(cap, 1); cap_reason = "macro_top1"
        if stress_mode == "top1" and g.get("v27_market_stress_day", pd.Series(False, index=g.index)).fillna(False).astype(bool).any():
            cap = min(cap, 1); cap_reason = "qqq_stress_top1"
        if news_mode == "top1" and g.get("v27_news_catalyst_proxy", pd.Series(False, index=g.index)).fillna(False).astype(bool).any():
            cap = min(cap, 1); cap_reason = "news_top1"
        seen_symbols: set[str] = set()
        taken = 0
        eligible_day_count = int(len(g))
        for order_in_day, (_, r) in enumerate(g.iterrows(), start=1):
            sym = str(r.get("symbol", "")).upper()
            if sym in seen_symbols:
                continue
            row = r.copy()
            row["selected_trade_number_for_day"] = int(taken + 1)
            row["selected_symbol_trade_number_for_day"] = 1
            row["eligible_day_count"] = eligible_day_count
            row["eligible_order_in_day"] = int(order_in_day)
            row["eligible_rank_score"] = float(row.get("_selection_score", 0.0))
            row["selection_top_requested"] = int(top_n)
            row["selection_top_cap_for_day"] = int(cap)
            row["selection_cap_reason"] = cap_reason
            row["selection_rule"] = "end_of_day_top_n_no_duplicate_symbol"
            selected_rows.append(row)
            seen_symbols.add(sym)
            taken += 1
            if taken >= cap:
                break
    if not selected_rows:
        return pd.DataFrame(columns=work.columns)
    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    return out.drop(columns=["_selection_score", "_selection_ts", "index"], errors="ignore")


def _v27_select_live_simulated_top_n(df: pd.DataFrame, top_n: int, params: StrategyParams) -> pd.DataFrame:
    """Walk-forward, no-lookahead Top-N selector for historical simulation.

    This is the live-style alternative to the classic V25 research replay that
    ranks the entire day after all candidates are known.  It processes candidates
    in chronological order and, at each timestamp, only ranks the candidates seen
    up to that moment.  Once a trade is selected, it cannot be replaced by a
    higher-scoring candidate that appears later in the day.

    The selector intentionally uses only candidate fields available at that
    candidate timestamp.  It does not inspect future P&L, R multiple, MFE/MAE,
    exit reason, or later candidates when deciding whether the current candidate
    is selectable.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else None)

    work = df.copy().reset_index(drop=True)
    work["_live_candidate_uid"] = range(len(work))

    time_col = "timestamp" if "timestamp" in work.columns else "entry_time"
    if time_col not in work.columns:
        return pd.DataFrame(columns=work.columns.drop("_live_candidate_uid", errors="ignore"))
    work["_live_ts"] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    work = work.dropna(subset=["_live_ts"]).copy()
    if work.empty:
        return pd.DataFrame(columns=df.columns)

    if "date" in work.columns:
        work["_live_date"] = work["date"].astype(str)
    elif "session_date" in work.columns:
        work["_live_date"] = pd.to_datetime(work["session_date"], errors="coerce").dt.date.astype(str)
    else:
        work["_live_date"] = work["_live_ts"].dt.tz_convert("America/New_York").dt.date.astype(str)

    if "score" in work.columns:
        work["_live_score"] = pd.to_numeric(work["score"], errors="coerce").fillna(-999999.0)
    elif "candidate_score" in work.columns:
        work["_live_score"] = pd.to_numeric(work["candidate_score"], errors="coerce").fillna(-999999.0)
    else:
        work["_live_score"] = 0.0

    macro_mode = str(getattr(params, "v27_macro_filter_mode", "off") or "off").lower()
    stress_mode = str(getattr(params, "v27_market_stress_mode", "off") or "off").lower()
    news_mode = str(getattr(params, "v27_news_filter_mode", "off") or "off").lower()

    selected_rows = []
    for _, day in work.sort_values(["_live_date", "_live_ts", "_live_score"], ascending=[True, True, False]).groupby("_live_date", sort=False):
        seen_parts = []
        selected_uids: set[int] = set()
        traded_symbols: set[str] = set()
        taken = 0
        for ts, at_ts in day.groupby("_live_ts", sort=True):
            seen_parts.append(at_ts)
            seen = pd.concat(seen_parts, ignore_index=False)
            cap = int(top_n)
            if macro_mode == "top1" and seen.get("v27_macro_event_day", pd.Series(False, index=seen.index)).fillna(False).astype(bool).any():
                cap = min(cap, 1)
            if stress_mode == "top1" and seen.get("v27_market_stress_day", pd.Series(False, index=seen.index)).fillna(False).astype(bool).any():
                cap = min(cap, 1)
            if news_mode == "top1" and seen.get("v27_news_catalyst_proxy", pd.Series(False, index=seen.index)).fillna(False).astype(bool).any():
                cap = min(cap, 1)
            if taken >= cap:
                continue

            # Build the live "top N seen so far" list, preserving the V25 rule of
            # no duplicate symbol inside the same session.  Already selected
            # candidates still occupy their ranking slots, exactly as a live bot
            # cannot undo a trade it took earlier.
            ranked_rows = []
            ranked_uid_order: dict[int, int] = {}
            ranked_symbols: set[str] = set()
            for order_in_seen, (_, r) in enumerate(seen.sort_values(["_live_score", "_live_ts"], ascending=[False, True]).iterrows(), start=1):
                sym = str(r.get("symbol", "")).upper()
                if sym in ranked_symbols:
                    continue
                ranked_rows.append(r)
                ranked_uid_order[int(r["_live_candidate_uid"])] = int(order_in_seen)
                ranked_symbols.add(sym)
                if len(ranked_rows) >= cap:
                    break
            eligible_uids = {int(r["_live_candidate_uid"]) for r in ranked_rows}
            current = at_ts[at_ts["_live_candidate_uid"].isin(eligible_uids)].copy()
            if current.empty:
                continue
            for _, r in current.sort_values(["_live_score", "_live_ts"], ascending=[False, True]).iterrows():
                uid = int(r["_live_candidate_uid"])
                sym = str(r.get("symbol", "")).upper()
                if uid in selected_uids or sym in traded_symbols:
                    continue
                if taken >= cap:
                    break
                row = r.copy()
                row["backtest_decision_mode"] = "live_simulated_seen_so_far_top_n"
                row["live_simulation_selected_at"] = ts
                row["live_simulation_seen_candidates"] = int(len(seen))
                row["live_simulation_rank_score"] = float(row.get("_live_score", 0.0))
                row["selected_trade_number_for_day"] = int(taken + 1)
                row["selected_symbol_trade_number_for_day"] = 1
                row["eligible_day_count"] = int(len(day))
                row["eligible_seen_count_at_entry"] = int(len(seen))
                row["eligible_order_in_day"] = int(ranked_uid_order.get(uid, 0))
                row["eligible_rank_score"] = float(row.get("_live_score", 0.0))
                row["selection_top_requested"] = int(top_n)
                row["selection_top_cap_for_day"] = int(cap)
                row["selection_cap_reason"] = "live_seen_so_far_cap"
                row["selection_rule"] = "live_seen_so_far_top_n_no_duplicate_symbol"
                selected_rows.append(row)
                selected_uids.add(uid)
                traded_symbols.add(sym)
                taken += 1

    if not selected_rows:
        cols = [c for c in work.columns if not c.startswith("_live_")]
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(selected_rows).sort_values("_live_ts").reset_index(drop=True)
    drop_cols = [c for c in out.columns if c.startswith("_live_")]
    return out.drop(columns=drop_cols, errors="ignore")

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



def _normalize_v25_like_candidate_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize candidate datasets so V25 replay can dynamically select Top N.

    Preferred input is data/v25_research/v25_candidates_all.csv.gz.  If that
    file is missing, this also supports the broad raw-replay ML candidate dataset
    produced by tools/build_q_learning_dataset.py, usually:
        data/ml_datasets/wf_ml_candidates_2022_2026/candidates.csv
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "timestamp" not in out.columns:
        for c in ["entry_time", "raw_bar_replay_signal_time", "signal_time"]:
            if c in out.columns:
                out["timestamp"] = out[c]
                break
    if "score" not in out.columns:
        for c in ["candidate_score", "robust_score", "fallback_score"]:
            if c in out.columns:
                out["score"] = out[c]
                break
    if "r075" not in out.columns:
        for c in ["r_multiple", "r", "result_r"]:
            if c in out.columns:
                out["r075"] = out[c]
                break
    if "close" not in out.columns:
        for c in ["entry_price", "bar_close", "close_price"]:
            if c in out.columns:
                out["close"] = out[c]
                break
    if "event" not in out.columns:
        for c in ["trigger_type", "setup_family", "setup_type"]:
            if c in out.columns:
                out["event"] = out[c]
                break
    alias_pairs = [
        ("gap_pct", ["gap_percent"]),
        ("rvol_tod", ["rvol_time_of_day", "rvol"]),
        ("rs_open", ["open_relative_strength", "day_relative_strength"]),
        ("chg_open", ["stock_change_from_open", "stock_day_change_percent"]),
        ("qqq_chg_open", ["qqq_change_from_open", "qqq_day_change_percent"]),
        ("vwap_ext_atr", ["vwap_extension_atr"]),
        ("daily_atr_pct", ["daily_atr14_percent"]),
    ]
    for dst, srcs in alias_pairs:
        if dst not in out.columns:
            for src in srcs:
                if src in out.columns:
                    out[dst] = out[src]
                    break
    if "side" not in out.columns and "direction" in out.columns:
        out["side"] = out["direction"]
    for c in ["score", "r075", "close", "gap_pct", "rvol_tod", "rs_open", "chg_open", "qqq_chg_open", "vwap_ext_atr", "daily_atr_pct"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _find_v25_dynamic_candidate_universe() -> tuple[pd.DataFrame | None, str, str]:
    """Return a full/dynamic candidate universe for Top N if available.

    Returns (df, path, source_kind).  source_kind is one of:
    - v25_candidates_all
    - ml_candidate_dataset
    - none
    """
    all_path = PROJECT_ROOT / "data" / "v25_research" / "v25_candidates_all.csv.gz"
    if all_path.exists():
        return _normalize_v25_like_candidate_universe(pd.read_csv(all_path)), str(all_path), "v25_candidates_all"

    candidates = [
        PROJECT_ROOT / "data" / "ml_datasets" / "wf_ml_candidates_2022_2026" / "candidates.csv",
    ]
    datasets_root = PROJECT_ROOT / "data" / "ml_datasets"
    if datasets_root.exists():
        candidates.extend(sorted(datasets_root.glob("*/candidates.csv"), key=lambda x: x.stat().st_mtime, reverse=True))
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            df = _normalize_v25_like_candidate_universe(pd.read_csv(path))
        except Exception:
            continue
        required = {"timestamp", "symbol", "side", "score", "r075"}
        if required.issubset(set(df.columns)) and not df.empty:
            return df, str(path), "ml_candidate_dataset"
    return None, "", "none"

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
    - Top 1/2/3/5/7/10/15 trades per day are selected after the optional filters when the full candidate universe is available.
    """
    direction = str(getattr(params, "direction_mode", "long_short") or "long_short").lower()
    if direction not in {"long_only", "short_only", "long_short"}:
        direction = "long_short"
    requested_top_n = int(getattr(params, "max_trades_per_day", 2) or 2)
    top_n = max(1, min(15, requested_top_n))
    # V37.5B: Top 5/7/10/15 require a dynamic candidate universe, not the old
    # preselected Top1/2/3 replay CSVs.  Prefer v25_candidates_all.csv.gz, but if
    # that is not installed use the broad ML candidate dataset created by
    # tools/build_q_learning_dataset.py.  Only fall back to old Top files when no
    # dynamic universe exists, and in that case the effective Top value is capped
    # at 3 and clearly reported.
    dynamic_df, dynamic_path, dynamic_kind = _find_v25_dynamic_candidate_universe()
    fallback_top_n = max(1, min(3, top_n))
    fallback_path = PROJECT_ROOT / "data" / "v25_research" / f"v25_{direction}_top{fallback_top_n}.csv"
    full_candidate_universe_available = dynamic_df is not None and not dynamic_df.empty
    top_n_effective = top_n
    if full_candidate_universe_available:
        df = dynamic_df.copy()
        candidate_source = f"{dynamic_path} ({dynamic_kind}; dynamic Top N enabled)"
    elif fallback_path.exists():
        if top_n > 3:
            raise RuntimeError(
                f"Top {top_n} requires a dynamic candidate universe, but only old Top1/2/3 replay files were found. "
                "Build the live-safe candidate dataset first: python tools\build_q_learning_dataset.py --start 2022-01-01 --end 2026-06-20 --preset v25_playbook --feed iex --session-mode regular_only --min-score 0 --candle-mode off --news-filter off --qqq-stress-filter off --use-news-proxy off --max-candidates-per-day 999 --max-symbol-candidates-per-day 999 --name wf_ml_candidates_2022_2026"
            )
        top_n_effective = fallback_top_n
        df = pd.read_csv(fallback_path)
        candidate_source = str(fallback_path) + f" (old fallback replay file; effective Top capped to {fallback_top_n}. Build wf_ml_candidates_2022_2026 or install v25_candidates_all.csv.gz for Top 5/7/10/15.)"
    else:
        raise RuntimeError(f"No dynamic V25 candidate universe found and fallback file not found: {fallback_path}. Build candidates with tools/build_q_learning_dataset.py --name wf_ml_candidates_2022_2026.")
    raw_loaded_count = int(len(df))
    v27_filter_stats = {"macro_filtered": 0, "stress_filtered": 0, "news_filtered": 0}
    v27_kill_skipped = 0
    openai_decisions = pd.DataFrame()
    openai_filter_diagnostics: dict[str, Any] = {"enabled": bool(getattr(params, "openai_trade_filter_enabled", False)), "path": "v25_replay", "reviewed": 0, "approved": 0, "rejected": 0}
    selected = pd.DataFrame()
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
        if bool(getattr(params, "enable_v377_positive_context_filter", False)) and not df.empty:
            profile_df = df.copy().reset_index(drop=True)
            if "trigger_type" not in profile_df.columns:
                profile_df["trigger_type"] = "v25_" + profile_df.get("event", "").astype(str)
            if "rvol_time_of_day" not in profile_df.columns and "rvol_tod" in profile_df.columns:
                profile_df["rvol_time_of_day"] = profile_df["rvol_tod"]
            if "daily_atr14_percent" not in profile_df.columns and "daily_atr_pct" in profile_df.columns:
                profile_df["daily_atr14_percent"] = profile_df["daily_atr_pct"]
            if "gap_percent" not in profile_df.columns and "gap_pct" in profile_df.columns:
                profile_df["gap_percent"] = profile_df["gap_pct"]
            if "day_relative_strength" not in profile_df.columns and "rs_open" in profile_df.columns:
                profile_df["day_relative_strength"] = profile_df["rs_open"]
            if "open_relative_strength" not in profile_df.columns and "rs_open" in profile_df.columns:
                profile_df["open_relative_strength"] = profile_df["rs_open"]
            if "vwap_extension_atr" not in profile_df.columns and "vwap_ext_atr" in profile_df.columns:
                profile_df["vwap_extension_atr"] = profile_df["vwap_ext_atr"]
            if "qqq_day_change_percent" not in profile_df.columns and "qqq_chg_open" in profile_df.columns:
                profile_df["qqq_day_change_percent"] = profile_df["qqq_chg_open"]
            if "candidate_score" not in profile_df.columns and "score" in profile_df.columns:
                profile_df["candidate_score"] = profile_df["score"]
            profile_df = apply_positive_context_profile_filter(profile_df, params)
            # keep only candidates that matched a learned positive context profile
            df = profile_df[profile_df.get("positive_context_profile_match", False).fillna(False).astype(bool)].copy()
        if bool(getattr(params, "enable_v379_decision_pattern_filter", False)) and not df.empty:
            df = apply_decision_time_pattern_scorer(df, params)
        filtered_count = int(len(df))

        # V35.2: apply the OpenAI review filter to the V25 replay path as well.
        # Earlier V35/V35.1 only reviewed raw-bar candidates.  Regular-hours
        # multi-symbol preset runs use this V25 replay path, so the dashboard could
        # show OpenAI as enabled while no OpenAI request was actually made.
        #
        # Review happens AFTER normal deterministic filters and BEFORE Top N/day
        # portfolio selection, so the model sees the candidate list produced by the
        # algorithm and returns approve/reject decisions for that list.
        if bool(getattr(params, "openai_trade_filter_enabled", False)) and not df.empty:
            review_df = df.copy().reset_index(drop=True)
            review_df["entry_time"] = pd.to_datetime(review_df["timestamp"], utc=True, errors="coerce")
            review_df["session_date"] = review_df["entry_time"].dt.tz_convert("America/New_York").dt.date
            review_df["side"] = review_df.get("side", "long")
            review_df["candidate_score"] = _num_series_default(review_df, "score", 0.0) if "score" in review_df.columns else _num_series_default(review_df, "robust_score", 0.0)
            review_df["fallback_score"] = pd.to_numeric(review_df.get("fallback_score", review_df["candidate_score"]), errors="coerce").fillna(review_df["candidate_score"])
            review_df["entry_price"] = _num_series_default(review_df, "close", 0.0) if "close" in review_df.columns else _num_series_default(review_df, "bar_close", 0.0)
            review_df["trigger_type"] = "v25_" + review_df.get("event", "event").astype(str)
            review_df["setup_family"] = "v25_symbol_playbook_replay"
            review_df["quality"] = "v25_replay_candidate_before_portfolio_selection"
            # Aliases used by the OpenAI payload.
            if "rvol_tod" in review_df.columns and "rvol_time_of_day" not in review_df.columns:
                review_df["rvol_time_of_day"] = pd.to_numeric(review_df["rvol_tod"], errors="coerce")
            if "rs_open" in review_df.columns:
                review_df["day_relative_strength"] = pd.to_numeric(review_df["rs_open"], errors="coerce")
                review_df["open_relative_strength"] = pd.to_numeric(review_df["rs_open"], errors="coerce")
            if "chg_open" in review_df.columns and "stock_change_from_open" not in review_df.columns:
                review_df["stock_change_from_open"] = pd.to_numeric(review_df["chg_open"], errors="coerce")
            if "qqq_chg_open" in review_df.columns and "qqq_change_from_open" not in review_df.columns:
                review_df["qqq_change_from_open"] = pd.to_numeric(review_df["qqq_chg_open"], errors="coerce")
            if "vwap_ext_atr" in review_df.columns and "vwap_extension_atr" not in review_df.columns:
                review_df["vwap_extension_atr"] = pd.to_numeric(review_df["vwap_ext_atr"], errors="coerce")
            if "daily_atr_pct" in review_df.columns and "daily_atr14_percent" not in review_df.columns:
                review_df["daily_atr14_percent"] = pd.to_numeric(review_df["daily_atr_pct"], errors="coerce")

            ai_result = review_candidates_with_openai(
                review_df,
                max_trades_per_day=top_n_effective,
                model=str(getattr(params, "openai_trade_filter_model", "gpt-5-mini") or "gpt-5-mini"),
                max_candidates_per_day=int(getattr(params, "openai_trade_filter_max_candidates_per_day", 200) or 200),
                min_confidence=float(getattr(params, "openai_trade_filter_min_confidence", 0.0) or 0.0),
                batch_mode=str(getattr(params, "openai_trade_filter_batch_mode", "full_run") or "full_run"),
            )
            df = ai_result.candidates.copy().reset_index(drop=True)
            openai_decisions = ai_result.decisions.copy()
            openai_filter_diagnostics = dict(ai_result.diagnostics or {})
            openai_filter_diagnostics["path"] = "v25_replay"
            openai_filter_diagnostics["input_candidates_before_openai"] = int(filtered_count)
            openai_filter_diagnostics["candidates_after_openai"] = int(len(df))

        decision_mode = str(getattr(params, "backtest_decision_mode", "end_of_day_top_n") or "end_of_day_top_n").lower()
        # V37.6: Apply Top-N selection for every dynamic/full candidate source,
        # including the raw-replay ML candidate dataset.  The previous patch
        # accidentally checked a local all_path variable that was not defined in
        # this scope; more importantly, it skipped Top-N selection when the
        # dynamic source was candidates.csv instead of v25_candidates_all.csv.gz.
        if full_candidate_universe_available:
            if decision_mode in {"live_simulated", "live", "walk_forward", "seen_so_far_top_n"}:
                df = _v27_select_live_simulated_top_n(df, top_n_effective, params) if not df.empty else df
            else:
                df = _v27_select_top_n_with_caps(df, top_n_effective, params) if not df.empty else df
        else:
            # Old top1/top2/top3 fallback files are already pre-selected and
            # cannot be expanded to Top 5/7/10/15.  Keep them explicit and
            # report the capped effective top value in diagnostics.
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
            trade_row = dict(r.to_dict())
            trade_row.update({
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
                "openai_reviewed": bool(r.get("openai_reviewed", False)) if pd.notna(r.get("openai_reviewed", False)) else False,
                "openai_trade": bool(r.get("openai_trade", False)) if pd.notna(r.get("openai_trade", False)) else False,
                "openai_confidence": float(r.get("openai_confidence", np.nan)) if pd.notna(r.get("openai_confidence", np.nan)) else np.nan,
                "openai_reason": str(r.get("openai_reason", "")) if pd.notna(r.get("openai_reason", "")) else "",
                "equity_at_entry": equity_at_entry,
            })
            # Keep a human-readable audit trail inside selected_trades.csv and
            # selected_trade_market_conditions.csv.  This preserves every raw
            # candidate indicator column from the source row and adds the
            # exact market state/reason fields used for report review.
            trade_row["selected_trade_number_for_day"] = trade_row.get("selected_trade_number_for_day", None)
            trade_row["selected_symbol_trade_number_for_day"] = trade_row.get("selected_symbol_trade_number_for_day", None)
            trade_row["decision_reason"] = " | ".join([
                f"TopN={top_n_effective}/{requested_top_n}",
                f"mode={decision_mode}",
                f"source={dynamic_kind if full_candidate_universe_available else 'fallback_top_file'}",
                f"symbol={trade_row.get('symbol','')}",
                f"side={trade_row.get('side','')}",
                f"trigger={trade_row.get('trigger_type','')}",
                f"score={trade_row.get('candidate_score','')}",
                f"rvol={trade_row.get('rvol_time_of_day','')}",
                f"daily_atr_pct={trade_row.get('daily_atr14_percent','')}",
                f"day_rs={trade_row.get('day_relative_strength','')}",
                f"open_rs={trade_row.get('open_relative_strength','')}",
                f"vwap_ext_atr={trade_row.get('vwap_extension_atr','')}",
                f"qqq_change={trade_row.get('qqq_day_change_percent','')}",
                f"candle_ok={trade_row.get('entry_candle_ok','')}",
                f"candle_warning={trade_row.get('opposing_candle_warning','')}",
                f"catalyst_proxy={trade_row.get('v27_news_catalyst_proxy','')}",
                f"positive_profile={trade_row.get('positive_context_profile_name','')}",
                f"v379_pattern={trade_row.get('v379_pattern_mode','')}",
                f"v379_score={trade_row.get('v379_pattern_score','')}",
            ])
            trade_row["market_conditions_at_entry"] = " | ".join([
                f"time_et={trade_row.get('entry_time_et','')}",
                f"gap_pct={trade_row.get('gap_percent','')}",
                f"rvol={trade_row.get('rvol_time_of_day','')}",
                f"daily_atr_pct={trade_row.get('daily_atr14_percent','')}",
                f"stock_change_open={trade_row.get('stock_day_change_percent','')}",
                f"day_rs={trade_row.get('day_relative_strength','')}",
                f"open_rs={trade_row.get('open_relative_strength','')}",
                f"vwap_ext_atr={trade_row.get('vwap_extension_atr','')}",
                f"qqq_change_open={trade_row.get('qqq_day_change_percent','')}",
                f"macro={trade_row.get('v27_macro_event_day','')}",
                f"stress={trade_row.get('v27_market_stress_day','')}",
                f"news_proxy={trade_row.get('v27_news_catalyst_proxy','')}",
                f"risk_per_share={trade_row.get('risk_per_share','')}",
                f"notional={trade_row.get('notional','')}",
                f"positive_profile_reason={trade_row.get('positive_context_profile_reason','')}",
                f"v379_reason={trade_row.get('v379_reason','')}",
            ])
            rows.append(trade_row)
        selected = pd.DataFrame(rows)
    # Final per-day audit numbering for the selected/executed trades.
    if selected is not None and not selected.empty:
        selected = selected.sort_values("entry_time").reset_index(drop=True)
        if "session_date" not in selected.columns:
            selected["session_date"] = pd.to_datetime(selected["entry_time"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date.astype(str)
        selected["selected_trade_number_for_day"] = selected.groupby(selected["session_date"].astype(str)).cumcount() + 1
        selected["selected_symbol_trade_number_for_day"] = selected.groupby([selected["session_date"].astype(str), selected["symbol"].astype(str).str.upper()]).cumcount() + 1
        selected["selected_trades_that_day"] = selected.groupby(selected["session_date"].astype(str))["symbol"].transform("count")
    summary = summarize_results(selected, params)
    if selected is not None and not selected.empty and "session_date" in selected.columns:
        _daily_selected_counts = selected.groupby(selected["session_date"].astype(str)).size()
        actual_max_selected_per_day = int(_daily_selected_counts.max())
        days_at_or_above_requested_top = int((_daily_selected_counts >= int(requested_top_n)).sum())
        days_at_or_above_effective_top = int((_daily_selected_counts >= int(top_n_effective)).sum())
    else:
        actual_max_selected_per_day = 0
        days_at_or_above_requested_top = 0
        days_at_or_above_effective_top = 0
    candidates = selected.copy()
    summary.update({
        "signals": candidates,
        "candidates": candidates,
        "portfolio_trades": selected,
        "openai_decisions": openai_decisions,
        "candidates_after_openai_filter": df.copy() if bool(getattr(params, "openai_trade_filter_enabled", False)) else pd.DataFrame(),
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
            "actual_max_selected_trades_in_single_day": int(actual_max_selected_per_day),
            "days_at_or_above_requested_top": int(days_at_or_above_requested_top),
            "days_at_or_above_effective_top": int(days_at_or_above_effective_top),
            "openai_candidates_after_filter": int(len(df)) if bool(getattr(params, "openai_trade_filter_enabled", False)) else None,
            "openai_filter": openai_filter_diagnostics,
            "execution_timeframe": "5Min V27 macro/news filterable symbol playbook replay",
            "memory_mode": "embedded_v25_candidate_universe_with_filters_macro_news",
            "direction_mode": direction,
            "requested_top_trades_per_day": requested_top_n,
            "effective_top_trades_per_day": top_n_effective,
            "fallback_top_file_limit": (fallback_top_n if not full_candidate_universe_available else None),
            "full_candidate_universe_available": bool(full_candidate_universe_available),
            "dynamic_candidate_source_kind": dynamic_kind,
            "backtest_decision_mode": str(getattr(params, "backtest_decision_mode", "end_of_day_top_n") or "end_of_day_top_n"),
            "decision_mode_note": "live_simulated processes candidates chronologically and only ranks candidates seen up to each timestamp; end_of_day_top_n preserves the original full-day research ranking.",
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
            "important_note": "V37.5B supports Top 1/2/3/5/7/10/15 when a dynamic candidate universe is available: v25_candidates_all.csv.gz or data/ml_datasets/*/candidates.csv. If old fallback Top files are used, effective Top is capped at 3. Filters, duplicate-symbol caps, and lack of enough eligible candidates can still make Top 10 and Top 15 identical.",
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




def _add_daily_features_live_safe(intraday: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Merge only prior-day daily features onto intraday bars.

    The standard research helper historically used the current daily ATR row,
    which is fine for broad research but not strict live replay.  This helper
    shifts every daily-derived value so a 10:00 candle never sees that day's
    final high/low/close/range.
    """
    daily_clean = daily.copy()
    daily_clean["timestamp"] = pd.to_datetime(daily_clean["timestamp"], utc=True, errors="coerce")
    daily_clean = daily_clean.dropna(subset=["timestamp"])
    daily_clean["session_date"] = daily_clean["timestamp"].dt.tz_convert("America/New_York").dt.date
    daily_clean = daily_clean.sort_values(["symbol", "session_date"])
    for col in ["open", "high", "low", "close", "volume"]:
        daily_clean[col] = pd.to_numeric(daily_clean[col], errors="coerce")
    daily_clean["prev_day_high"] = daily_clean.groupby("symbol")["high"].shift(1)
    daily_clean["prev_day_low"] = daily_clean.groupby("symbol")["low"].shift(1)
    daily_clean["prev_close"] = daily_clean.groupby("symbol")["close"].shift(1)
    daily_clean["daily_dollar_volume"] = daily_clean["volume"] * daily_clean["close"]
    daily_clean["avg_20d_dollar_volume"] = daily_clean.groupby("symbol")["daily_dollar_volume"].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    prev_close = daily_clean.groupby("symbol")["close"].shift(1)
    daily_tr = pd.concat([
        daily_clean["high"] - daily_clean["low"],
        (daily_clean["high"] - prev_close).abs(),
        (daily_clean["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Shift the ATR percent by one session so today's final range is never known intraday.
    daily_clean["daily_atr14"] = daily_tr.groupby(daily_clean["symbol"]).transform(lambda s: s.rolling(14, min_periods=14).mean().shift(1))
    daily_clean["daily_atr14_percent"] = daily_clean["daily_atr14"] / daily_clean["prev_close"].replace(0, np.nan) * 100
    merge_cols = ["symbol", "session_date", "prev_day_high", "prev_day_low", "prev_close", "avg_20d_dollar_volume", "daily_atr14_percent"]
    return intraday.merge(daily_clean[merge_cols], on=["symbol", "session_date"], how="left")


def _build_qqq_context_live_safe(qqq_5m: pd.DataFrame, qqq_daily: pd.DataFrame, session_mode: str = "regular_only") -> pd.DataFrame:
    """Build market context without current-day daily lookahead."""
    q = add_intraday_features(qqq_5m, session_mode=session_mode)
    q = _add_daily_features_live_safe(q, qqq_daily)
    qdaily = qqq_daily.copy()
    qdaily["timestamp"] = pd.to_datetime(qdaily["timestamp"], utc=True, errors="coerce")
    qdaily = qdaily.dropna(subset=["timestamp"])
    qdaily["session_date"] = qdaily["timestamp"].dt.tz_convert("America/New_York").dt.date
    qdaily = qdaily.sort_values("session_date")
    for col in ["open", "high", "low", "close", "volume"]:
        qdaily[col] = pd.to_numeric(qdaily[col], errors="coerce")
    qdaily_prev_close = qdaily["close"].shift(1)
    qdaily_tr = pd.concat([
        qdaily["high"] - qdaily["low"],
        (qdaily["high"] - qdaily_prev_close).abs(),
        (qdaily["low"] - qdaily_prev_close).abs(),
    ], axis=1).max(axis=1)
    qdaily["daily_atr14"] = qdaily_tr.rolling(14, min_periods=14).mean().shift(1)
    qdaily["qqq_daily_atr14_percent"] = qdaily["daily_atr14"] / qdaily_prev_close.replace(0, np.nan) * 100
    qdaily_small = qdaily[["session_date", "qqq_daily_atr14_percent"]]

    q_resample = q.set_index("timestamp_ny").copy()
    fifteen_frames = []
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
        q15["qqq_15m_ema50"] = ema(q15["close"], 50)
        q15 = q15[["timestamp", "qqq_15m_close", "qqq_15m_ema50"]]
    else:
        q15 = pd.DataFrame(columns=["timestamp", "qqq_15m_close", "qqq_15m_ema50"])

    q5 = q[["timestamp", "session_date", "close", "session_vwap", "prev_close", "session_open", "ema9", "ema20", "rsi2", "atr5m14"]].copy()
    q5 = q5.rename(columns={"close": "qqq_close", "session_vwap": "qqq_session_vwap", "prev_close": "qqq_prev_close", "session_open": "qqq_session_open", "ema9": "qqq_ema9", "ema20": "qqq_ema20", "rsi2": "qqq_rsi2", "atr5m14": "qqq_atr5m14"})
    q5["qqq_day_change_percent"] = (q5["qqq_close"] - q5["qqq_prev_close"]) / q5["qqq_prev_close"] * 100
    q5["qqq_change_from_open"] = (q5["qqq_close"] - q5["qqq_session_open"]) / q5["qqq_session_open"] * 100
    q5["qqq_15min_change_percent"] = q5.groupby("session_date")["qqq_close"].transform(lambda x: (x - x.shift(3)) / x.shift(3) * 100)
    q5 = q5.merge(qdaily_small, on="session_date", how="left")
    if not q15.empty:
        q5 = pd.merge_asof(q5.sort_values("timestamp"), q15.sort_values("timestamp"), on="timestamp", direction="backward")
        q5["qqq_15m_above_ema50"] = q5["qqq_15m_close"] > q5["qqq_15m_ema50"]
    else:
        q5["qqq_15m_close"] = np.nan
        q5["qqq_15m_ema50"] = np.nan
        q5["qqq_15m_above_ema50"] = False
    return q5

_RAW_BAR_REPLAY_MODES = {"raw_bar_replay", "full_raw_bar_replay", "historical_raw_replay", "raw_replay"}


def _signal_to_candidate_from_raw_replay(symbol_frame: pd.DataFrame, signal_index_label: Any, params: StrategyParams) -> dict[str, Any] | None:
    """Simulate one selected live signal using the normal V25 trade simulator.

    The decision to select the signal has already been made using only the signal
    bar and earlier bars.  This function is allowed to look forward only after
    that decision, exactly as a backtest must do to evaluate the trade exit.
    """
    if symbol_frame is None or symbol_frame.empty:
        return None
    g = symbol_frame.sort_values("timestamp").reset_index(drop=True)
    try:
        ts = pd.Timestamp(signal_index_label).tz_convert("UTC")
        matches = g.index[pd.to_datetime(g["timestamp"], utc=True, errors="coerce").eq(ts)].tolist()
        if not matches:
            return None
        pos = int(matches[0])
    except Exception:
        try:
            pos = int(signal_index_label)
        except Exception:
            return None
    trade = _simulate_trade(g, pos, params)
    if trade is not None:
        trade["raw_bar_replay_signal_time"] = g.iloc[pos].get("timestamp")
        trade["backtest_decision_mode"] = "full_raw_bar_replay"
    return trade


def _prepare_raw_replay_alert_aliases(alerts: pd.DataFrame) -> pd.DataFrame:
    """Match the live worker/V25 replay column names used by filters and OpenAI."""
    if alerts is None or alerts.empty:
        return pd.DataFrame()
    out = alerts.copy()
    if "score" not in out.columns:
        out["score"] = _num_series_default(out, "candidate_score", 0.0)
    if "candidate_score" not in out.columns:
        out["candidate_score"] = _num_series_default(out, "score", 0.0)
    if "date" not in out.columns:
        out["date"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date.astype(str)
    if "session_date" not in out.columns:
        out["session_date"] = out["date"]
    if "fallback_score" not in out.columns:
        out["fallback_score"] = _num_series_default(out, "supporting_score", 0.0) if "supporting_score" in out.columns else _num_series_default(out, "score", 0.0)
    if "entry_price" not in out.columns:
        out["entry_price"] = _num_series_default(out, "close", 0.0)
    if "trigger_type" not in out.columns:
        out["trigger_type"] = "raw_replay_signal"
    if "setup_family" not in out.columns:
        out["setup_family"] = "full_raw_bar_replay"
    if "quality" not in out.columns:
        out["quality"] = "raw_bar_replay_candidate_at_timestamp"

    # V36.2 raw-replay alias fix.  The UI QQQ stress and catalyst filters were
    # reading replay/research column names (qqq_chg_open, gap_pct, rvol_tod).
    # Full raw-bar replay produces live-engine names instead.  Without these
    # aliases the dropdowns appeared to run but could not change the result.
    if "qqq_chg_open" not in out.columns and "qqq_change_from_open" in out.columns:
        out["qqq_chg_open"] = _num_series_default(out, "qqq_change_from_open", 0.0)
    if "gap_pct" not in out.columns and "gap_percent" in out.columns:
        out["gap_pct"] = _num_series_default(out, "gap_percent", 0.0)
    if "rvol_tod" not in out.columns and "rvol_time_of_day" in out.columns:
        out["rvol_tod"] = _num_series_default(out, "rvol_time_of_day", 0.0)

    # Candlestick aliases for the V25 candlestick dropdown.
    if "side_continuation_candle" not in out.columns:
        side_short = out.get("side", pd.Series("", index=out.index)).astype(str).str.lower().eq("short")
        out["side_continuation_candle"] = pd.Series(np.where(side_short, _bool_series_default(out, "bearish_continuation_candle", False), _bool_series_default(out, "bullish_continuation_candle", False)), index=out.index).astype(bool)
        out["side_rejection_candle"] = pd.Series(np.where(side_short, _bool_series_default(out, "bearish_rejection_candle", False), _bool_series_default(out, "bullish_rejection_candle", False)), index=out.index).astype(bool)
        out["side_engulfing_candle"] = pd.Series(np.where(side_short, _bool_series_default(out, "bearish_engulfing_candle", False), _bool_series_default(out, "bullish_engulfing_candle", False)), index=out.index).astype(bool)
    return out


def _run_raw_bar_live_replay(
    symbols: list[str],
    start_date: str,
    end_date: str,
    params: StrategyParams,
    feed: str,
    use_cache: bool = True,
    use_news: bool = False,
    export_report: bool = True,
    session_mode: str | None = None,
) -> dict[str, Any]:
    """Full raw-bar historical replay, designed to mimic live timing.

    This mode does not use the packaged V25 candidate replay. It loads local/raw
    5-minute bars, computes causal features/signals from those bars, then walks
    forward timestamp-by-timestamp. At each historical timestamp it only allows
    the selector to see alerts with timestamp <= the current timestamp for that
    same session. Once a trade is selected, it cannot be replaced by a later
    better-looking trade.

    Notes on realism:
    - Indicator columns are precomputed once for speed, but they use rolling,
      expanding, cumulative, and shifted calculations only. No future outcome
      columns are used by the selector.
    - Trade exits are evaluated after a signal is selected, which is required for
      any backtest.
    """
    settings = AlpacaSettings()
    client = AlpacaDataClient(settings)
    active_session_mode = str(session_mode or getattr(params, "backtest_session_mode", "regular_only") or "regular_only").lower()
    fetch_start = pad_start_for_indicators(start_date, days=55)
    fetch_end = _plus_one_day(end_date)
    start_ts = pd.Timestamp(start_date, tz="America/New_York")
    end_ts = pd.Timestamp(_plus_one_day(end_date), tz="America/New_York")

    tradable_symbols = [s.upper() for s in symbols if s and s.upper() != "QQQ"]
    tradable_symbols = list(dict.fromkeys(tradable_symbols))
    if not tradable_symbols:
        raise RuntimeError("No tradable symbols supplied for raw-bar replay.")

    cache_status: list[dict[str, Any]] = []
    if use_cache:
        all_cache_symbols = list(dict.fromkeys(["QQQ"] + tradable_symbols))
        cache_status.append(client.prefetch_stock_bars(
            all_cache_symbols, "5Min", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=True, session_mode=active_session_mode
        ))
        cache_status.append(client.prefetch_stock_bars(
            all_cache_symbols, "1Day", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=True, session_mode="regular_only"
        ))

    qqq_5m = client.get_stock_bars(["QQQ"], "5Min", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=use_cache, session_mode=active_session_mode)
    qqq_daily = client.get_stock_bars(["QQQ"], "1Day", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=use_cache, session_mode="regular_only")
    if qqq_5m.empty or qqq_daily.empty:
        raise RuntimeError("Raw-bar replay requires QQQ 5Min and 1Day bars for market context.")
    qqq_context = _build_qqq_context_live_safe(qqq_5m, qqq_daily, session_mode=active_session_mode)

    news = pd.DataFrame()
    if use_news:
        news = client.get_news_counts_by_day(tradable_symbols, fetch_start, fetch_end, use_cache=use_cache)
        if not news.empty:
            news["session_date"] = pd.to_datetime(news["session_date"]).dt.date

    signals_by_symbol: dict[str, pd.DataFrame] = {}
    alert_frames: list[pd.DataFrame] = []
    signal_export_frames: list[pd.DataFrame] = []
    skipped_symbols: list[dict[str, Any]] = []
    raw_alert_count = 0
    scan_days_seen: set[Any] = set()

    for symbol in tradable_symbols:
        try:
            sym_5m = client.get_stock_bars([symbol], "5Min", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=use_cache, session_mode=active_session_mode)
            sym_daily = client.get_stock_bars([symbol], "1Day", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=use_cache, session_mode="regular_only")
            if sym_5m.empty or sym_daily.empty:
                skipped_symbols.append({"symbol": symbol, "reason": "missing_bars"})
                continue
            intraday = add_intraday_features(sym_5m, session_mode=active_session_mode)
            intraday = _add_daily_features_live_safe(intraday, sym_daily)
            merged = merge_market_context(intraday, qqq_context)
            if use_news and not news.empty:
                sym_news = news[news["symbol"] == symbol]
                if not sym_news.empty:
                    merged = merged.merge(sym_news, on=["symbol", "session_date"], how="left")
            if "news_count_last_3d" in merged.columns:
                merged["news_count_last_3d"] = merged["news_count_last_3d"].fillna(0)
            signals = compute_signals(_reduce_memory(merged), params)
            signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True, errors="coerce")
            signals = signals[(signals["timestamp_ny"] >= start_ts) & (signals["timestamp_ny"] < end_ts)].copy()
            if signals.empty:
                skipped_symbols.append({"symbol": symbol, "reason": "no_rows_in_requested_window"})
                continue
            signals_by_symbol[symbol] = signals.sort_values("timestamp").reset_index(drop=True)
            if "session_date" in signals.columns:
                scan_days_seen.update(signals["session_date"].dropna().unique().tolist())
            alert_mask = signals.get("buy_alert", pd.Series(False, index=signals.index)).fillna(False).astype(bool)
            raw_alert_count += int(alert_mask.sum())
            alerts = signals.loc[alert_mask].copy()
            if not alerts.empty:
                alerts["_raw_symbol"] = symbol
                alerts["_raw_signal_timestamp"] = pd.to_datetime(alerts["timestamp"], utc=True, errors="coerce")
                alert_frames.append(_reduce_memory(alerts))
            sig_export = _small_signal_export(signals)
            if not sig_export.empty:
                signal_export_frames.append(_reduce_memory(sig_export))
        except MemoryError:
            raise RuntimeError(f"Memory limit hit while raw-replaying {symbol}. Try a smaller watchlist/date range.")
        except Exception as exc:
            skipped_symbols.append({"symbol": symbol, "reason": f"error: {exc}"})
        finally:
            for name in ["sym_5m", "sym_daily", "intraday", "merged", "signals", "alerts"]:
                if name in locals():
                    try:
                        del locals()[name]
                    except Exception:
                        pass
            gc.collect()

    signals_export = pd.concat(signal_export_frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True) if signal_export_frames else pd.DataFrame()
    raw_alerts = pd.concat(alert_frames, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True) if alert_frames else pd.DataFrame()

    openai_decisions_all: list[pd.DataFrame] = []
    openai_diag: dict[str, Any] = {"enabled": bool(getattr(params, "openai_trade_filter_enabled", False)), "mode": "realtime_per_timestamp"}
    openai_calls = 0
    openai_reviewed = 0
    openai_approved = 0

    if raw_alerts.empty:
        portfolio_trades = pd.DataFrame()
        replay_audit = pd.DataFrame()
    else:
        alerts_all = _prepare_raw_replay_alert_aliases(raw_alerts)
        min_score = float(getattr(params, "min_candidate_score", 2.0) or 2.0)
        alerts_all = alerts_all[(_num_series_default(alerts_all, "score", -9999.0) if "score" in alerts_all.columns else _num_series_default(alerts_all, "candidate_score", -9999.0)) >= min_score].copy()
        alerts_all = _apply_v25_candlestick_filter(alerts_all, str(getattr(params, "candle_pattern_mode", "selective")))
        alerts_all, prefilter_stats = _v27_apply_preselection_filters(alerts_all, params, use_news=use_news)

        q_learning_diag = {"enabled": bool(getattr(params, "q_learning_filter_enabled", False))}
        if bool(getattr(params, "q_learning_filter_enabled", False)) and not alerts_all.empty:
            policy_path = str(getattr(params, "q_learning_policy_path", "") or "").strip()
            if not policy_path:
                raise RuntimeError("Q-learning filter is enabled but q_learning_policy_path is blank.")
            q_model = load_q_model(policy_path)
            q_reviewed = apply_q_policy(
                alerts_all,
                q_model,
                min_edge=float(getattr(params, "q_learning_min_edge", 0.0) or 0.0),
                min_state_count=int(getattr(params, "q_learning_min_state_count", 8) or 8),
            )
            before_q = int(len(q_reviewed))
            approved_q = int(q_reviewed.get("q_policy_approved", pd.Series([], dtype=bool)).fillna(False).astype(bool).sum())
            alerts_all = q_reviewed[q_reviewed["q_policy_approved"].fillna(False).astype(bool)].copy()
            q_learning_diag = {
                "enabled": True,
                "policy_path": policy_path,
                "reviewed": before_q,
                "approved": approved_q,
                "rejected": before_q - approved_q,
                "min_edge": float(getattr(params, "q_learning_min_edge", 0.0) or 0.0),
                "min_state_count": int(getattr(params, "q_learning_min_state_count", 8) or 8),
            }

        ml_ranker_diag = {"enabled": bool(getattr(params, "ml_ranker_filter_enabled", False))}
        if bool(getattr(params, "ml_ranker_filter_enabled", False)) and not alerts_all.empty:
            ml_path = str(getattr(params, "ml_ranker_model_path", "") or "").strip()
            if not ml_path:
                raise RuntimeError("ML ranker filter is enabled but ml_ranker_model_path is blank.")
            ml_model = load_ranker_model(ml_path)
            ml_reviewed = score_ml_ranker_candidates(alerts_all, ml_model)
            before_ml = int(len(ml_reviewed))
            threshold_ml = float(getattr(params, "ml_ranker_min_pred_r", 0.05) or 0.0)
            min_win_ml = float(getattr(params, "ml_ranker_min_win_prob", 0.0) or 0.0)
            ml_selected_like = ml_ranker_select(
                ml_reviewed,
                threshold=threshold_ml,
                top_trades_per_day=max(999, int(getattr(params, "max_trades_per_day", 2) or 2)),
                max_symbol_per_day=max(999, int(getattr(params, "max_alerts_per_symbol_per_day", 1) or 1)),
                min_win_prob=min_win_ml if min_win_ml > 0 else None,
            )
            # Keep all ML-approved candidates here; normal walk-forward selector still
            # enforces Top N / per-day rules after this filter.
            if not ml_selected_like.empty:
                key_cols = [c for c in ["symbol", "timestamp", "entry_time", "session_date"] if c in ml_reviewed.columns and c in ml_selected_like.columns]
                if key_cols:
                    selected_keys = set(map(tuple, ml_selected_like[key_cols].astype(str).to_numpy()))
                    ml_reviewed["ml_ranker_approved"] = [tuple(x) in selected_keys for x in ml_reviewed[key_cols].astype(str).to_numpy()]
                else:
                    ml_reviewed["ml_ranker_approved"] = ml_reviewed["ml_pred_r"] >= threshold_ml
            else:
                ml_reviewed["ml_ranker_approved"] = False
            approved_ml = int(ml_reviewed["ml_ranker_approved"].fillna(False).astype(bool).sum())
            alerts_all = ml_reviewed[ml_reviewed["ml_ranker_approved"].fillna(False).astype(bool)].copy()
            ml_ranker_diag = {
                "enabled": True,
                "model_path": ml_path,
                "reviewed": before_ml,
                "approved": approved_ml,
                "rejected": before_ml - approved_ml,
                "min_pred_r": threshold_ml,
                "min_win_prob": min_win_ml,
            }

        alerts_all = alerts_all.sort_values(["timestamp", "score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)

        # V35.7: For historical raw-bar replay, batch the OpenAI review BEFORE
        # the walk-forward selector.  This keeps the backtest live-safe because
        # each row still contains only timestamp-available fields and the prompt
        # forces independent moment-in-time decisions, but it avoids the previous
        # V35.6 behavior where raw replay called OpenAI once per timestamp /
        # sometimes once per candidate.  The user's requested experiment is one
        # prompt containing all identified candidates when they fit the configured
        # API-call limit.  If the list is larger than the limit, it chunks only
        # by that explicit limit.
        openai_input_before_filter = int(len(alerts_all))
        if bool(getattr(params, "openai_trade_filter_enabled", False)) and not alerts_all.empty:
            ai_result = review_candidates_with_openai(
                alerts_all,
                max_trades_per_day=int(getattr(params, "max_trades_per_day", 2) or 2),
                model=str(getattr(params, "openai_trade_filter_model", "gpt-5-mini") or "gpt-5-mini"),
                max_candidates_per_day=int(getattr(params, "openai_trade_filter_max_candidates_per_day", 5000) or 5000),
                min_confidence=float(getattr(params, "openai_trade_filter_min_confidence", 0.0) or 0.0),
                batch_mode=str(getattr(params, "openai_trade_filter_batch_mode", "full_run") or "full_run"),
            )
            alerts_all = ai_result.candidates.copy().sort_values(["timestamp", "score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
            if ai_result.decisions is not None and not ai_result.decisions.empty:
                d = ai_result.decisions.copy()
                d["review_stage"] = "raw_replay_batched_before_walk_forward_selection"
                openai_decisions_all.append(d)
            diag = dict(ai_result.diagnostics or {})
            openai_calls += int(diag.get("api_calls", 0) or 0)
            openai_reviewed += int(diag.get("reviewed", 0) or 0)
            openai_approved += int(diag.get("approved", 0) or 0)
            openai_diag.update(diag)
            openai_diag.update({
                "enabled": True,
                "mode": "batched_before_walk_forward_selection",
                "path": "full_raw_bar_replay_batched_before_walk_forward",
                "input_candidates_before_openai": openai_input_before_filter,
                "candidates_after_openai": int(len(alerts_all)),
                "note": "OpenAI is called before walk-forward selection in large batches. Candidate decisions remain independent real-time decisions; OpenAI is not asked to rank the day or batch.",
            })

        selected_trades: list[dict[str, Any]] = []
        skipped_candidate_rows: list[dict[str, Any]] = []
        taken_trade_keys: set[str] = set()
        max_trades_per_day = int(getattr(params, "max_trades_per_day", 2) or 2)
        max_open_positions = int(getattr(params, "max_open_positions", 2) or 2)
        max_orders_per_symbol_per_day = int(getattr(params, "max_alerts_per_symbol_per_day", 1) or 1)

        for day_key, day_alerts in alerts_all.groupby("date", sort=True):
            day_alerts = day_alerts.sort_values(["timestamp", "score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
            day_selected: list[dict[str, Any]] = []
            day_symbols_taken: set[str] = set()
            seen_parts: list[pd.DataFrame] = []
            for ts, current_at_ts in day_alerts.groupby("timestamp", sort=True):
                ts = pd.Timestamp(ts).tz_convert("UTC")
                seen_parts.append(current_at_ts.copy())
                seen = pd.concat(seen_parts, ignore_index=True)

                # Live-style capacity state as of the current signal timestamp.
                closed_before = [t for t in selected_trades if pd.Timestamp(t.get("exit_time")) <= ts]
                open_now = [t for t in selected_trades if pd.Timestamp(t.get("entry_time")) <= ts < pd.Timestamp(t.get("exit_time"))]
                taken_today = len(day_selected)
                if taken_today >= max_trades_per_day:
                    continue
                if len(open_now) >= max_open_positions:
                    continue

                selected_so_far = _v27_select_top_n_with_caps(seen, max_trades_per_day, params)
                if selected_so_far.empty:
                    continue
                selected_so_far["timestamp"] = pd.to_datetime(selected_so_far["timestamp"], utc=True, errors="coerce")
                current_candidates = selected_so_far[selected_so_far["timestamp"].eq(ts)].copy()
                if current_candidates.empty:
                    continue

                # OpenAI review is intentionally NOT called here in V35.7.
                # It is applied once, in batch, before the walk-forward selector above.
                # Calling it here caused one API request per timestamp/candidate in
                # raw replay, which was both expensive and not the requested test.
                if current_candidates.empty:
                    continue

                for _, sig in current_candidates.sort_values(["score", "timestamp"], ascending=[False, True]).iterrows():
                    if len(day_selected) >= max_trades_per_day:
                        break
                    sym = str(sig.get("symbol", "")).upper()
                    if not sym:
                        continue
                    if sum(1 for t in day_selected if str(t.get("symbol", "")).upper() == sym) >= max_orders_per_symbol_per_day:
                        skipped = sig.to_dict(); skipped["selected"] = False; skipped["skip_reason"] = "max_orders_per_symbol_per_day"; skipped_candidate_rows.append(skipped)
                        continue
                    if any(str(t.get("symbol", "")).upper() == sym for t in open_now):
                        skipped = sig.to_dict(); skipped["selected"] = False; skipped["skip_reason"] = "symbol_already_open"; skipped_candidate_rows.append(skipped)
                        continue
                    key = f"{sym}|{day_key}|{pd.Timestamp(sig.get('timestamp')).isoformat()}|{sig.get('side')}|{sig.get('trigger_type')}"
                    if key in taken_trade_keys:
                        continue
                    symbol_frame = signals_by_symbol.get(sym, pd.DataFrame())
                    trade = _signal_to_candidate_from_raw_replay(symbol_frame, sig.get("timestamp"), params)
                    if trade is None:
                        skipped = sig.to_dict(); skipped["selected"] = False; skipped["skip_reason"] = "trade_simulation_failed"; skipped_candidate_rows.append(skipped)
                        continue
                    trade["raw_replay_selected_from_seen_candidates"] = int(len(seen))
                    trade["raw_replay_signal_timestamp"] = sig.get("timestamp")
                    trade["raw_replay_selection_timestamp"] = ts
                    trade["raw_replay_selection_mode"] = "live_worker_seen_so_far_top_n"
                    selected_trades.append(trade)
                    day_selected.append(trade)
                    day_symbols_taken.add(sym)
                    taken_trade_keys.add(key)

        candidates_for_portfolio = pd.DataFrame(selected_trades).sort_values("entry_time").reset_index(drop=True) if selected_trades else pd.DataFrame()
        if skipped_candidate_rows:
            skipped_candidates = pd.DataFrame(skipped_candidate_rows)
        else:
            skipped_candidates = pd.DataFrame()
        # Make Symbol/side kill switch active for full raw-bar replay too.
        # The raw replay path preselects trades before sizing, so the research
        # replay kill-switch helper has to run here instead of only in the classic
        # replay branch.
        v27_kill_skipped = 0
        if not candidates_for_portfolio.empty:
            if "timestamp" not in candidates_for_portfolio.columns:
                if "raw_bar_replay_signal_time" in candidates_for_portfolio.columns:
                    candidates_for_portfolio["timestamp"] = candidates_for_portfolio["raw_bar_replay_signal_time"]
                elif "signal_time" in candidates_for_portfolio.columns:
                    candidates_for_portfolio["timestamp"] = candidates_for_portfolio["signal_time"]
                else:
                    candidates_for_portfolio["timestamp"] = candidates_for_portfolio["entry_time"]
            candidates_for_portfolio, v27_kill_skipped = _v27_apply_symbol_side_kill_switch(candidates_for_portfolio, params)

        # Apply normal sizing/risk controls, but avoid redoing end-of-day ranking.
        params_for_apply = params
        old_mode = getattr(params_for_apply, "backtest_decision_mode", "end_of_day_top_n")
        try:
            params_for_apply.backtest_decision_mode = "raw_bar_replay"
        except Exception:
            pass
        portfolio_trades = apply_portfolio_rules(candidates_for_portfolio, params_for_apply) if not candidates_for_portfolio.empty else pd.DataFrame()
        try:
            params_for_apply.backtest_decision_mode = old_mode
        except Exception:
            pass
        replay_audit = alerts_all
        replay_audit["raw_replay_alert_seen"] = True
        if "prefilter_stats" not in locals():
            prefilter_stats = {"macro_filtered": 0, "stress_filtered": 0, "news_filtered": 0}

    openai_decisions = pd.concat(openai_decisions_all, ignore_index=True) if openai_decisions_all else pd.DataFrame()
    if bool(getattr(params, "openai_trade_filter_enabled", False)):
        if "path" not in openai_diag:
            openai_diag.update({"path": "full_raw_bar_replay_batched_before_walk_forward"})
        openai_diag.update({"api_calls": openai_calls, "reviewed": openai_reviewed, "approved": openai_approved})

    summary = summarize_results(portfolio_trades, params)
    selected_tmp = summary.get("selected_trades", pd.DataFrame())
    if "mfe_mae_summary" not in summary:
        if selected_tmp is not None and not selected_tmp.empty:
            metrics_tmp = summary.get("metrics", {})
            summary["mfe_mae_summary"] = pd.DataFrame([
                {"metric": "avg_mfe_r", "value": metrics_tmp.get("avg_mfe_r", 0), "meaning": "How far trades moved in favor before exit"},
                {"metric": "avg_mae_r", "value": metrics_tmp.get("avg_mae_r", 0), "meaning": "How far trades moved against you before exit"},
                {"metric": "target1_hit_rate", "value": metrics_tmp.get("target1_hit_rate", 0), "meaning": "If low, entries are weak or target1 is too far"},
            ])
        else:
            summary["mfe_mae_summary"] = pd.DataFrame()
    scan_days = len(scan_days_seen)
    selected_count = int(len(summary.get("selected_trades", pd.DataFrame())))
    trade_days = int(summary.get("metrics", {}).get("trade_days", 0))
    summary.update({
        "signals": signals_export,
        "candidates": raw_alerts,
        "raw_replay_alerts_after_filters": replay_audit if "replay_audit" in locals() else pd.DataFrame(),
        "openai_decisions": openai_decisions,
        "candidates_after_openai_filter": pd.DataFrame(),
        "portfolio_trades": portfolio_trades,
        "params": params.to_dict(),
        "symbols": tradable_symbols,
        "feed": feed,
        "start_date": start_date,
        "end_date": end_date,
        "use_news": use_news,
        "execution_timeframe": f"5Min {active_session_mode} raw-bar replay",
        "skipped_symbols": pd.DataFrame(skipped_symbols),
        "diagnostics": {
            "symbols_scanned": len(tradable_symbols),
            "symbols_skipped": len(skipped_symbols),
            "scan_days": scan_days,
            "raw_alerts": raw_alert_count,
            "raw_candidates": int(len(raw_alerts)),
            "openai_candidates_after_filter": None,
            "openai_filter": openai_diag,
            "selected_trades": selected_count,
            "backtest_decision_mode": "full_raw_bar_replay",
            "decision_mode_note": "Full raw-bar replay loads historical 5-minute bars, computes causal features, walks timestamp-by-timestamp, and only lets the selector see alerts available up to the current timestamp. It does not use the V25 candidate replay file.",
            "raw_replay_no_future_candidate_ranking": True,
            "raw_replay_signal_source": "local_or_fetched_raw_5min_bars",
            "raw_replay_feature_note": "Features are precomputed once for speed but use causal rolling/cumulative/shifted formulas. Daily ATR/volume/prev-day context is shifted to prior sessions so current-day final daily data is not visible. Exits are evaluated only after a signal is selected.",
            "prefilter_stats": prefilter_stats if "prefilter_stats" in locals() else {},
            "q_learning_filter": q_learning_diag if "q_learning_diag" in locals() else {"enabled": False},
            "ml_ranker_filter": ml_ranker_diag if "ml_ranker_diag" in locals() else {"enabled": False},
            "v27_kill_switch_skipped_selected_trades": int(v27_kill_skipped) if "v27_kill_skipped" in locals() else 0,
            "trade_days": trade_days,
            "selected_trades_per_scan_day": selected_count / scan_days if scan_days else 0.0,
            "raw_alerts_per_scan_day": raw_alert_count / scan_days if scan_days else 0.0,
            "execution_timeframe": f"5Min {active_session_mode} raw-bar replay",
            "memory_mode": "full_raw_bar_replay_local_bar_store",
            "cache_status": cache_status,
        },
    })
    if export_report:
        summary["report_paths"] = export_backtest_report(summary)
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
    session_mode: str | None = None,
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
    active_session_mode = str(session_mode or getattr(params, "backtest_session_mode", "regular_only") or "regular_only").lower()
    decision_mode = str(getattr(params, "backtest_decision_mode", "end_of_day_top_n") or "end_of_day_top_n").lower()
    if decision_mode in _RAW_BAR_REPLAY_MODES:
        return _run_raw_bar_live_replay(symbols, start_date, end_date, params, feed, use_cache=use_cache, use_news=use_news, export_report=export_report, session_mode=active_session_mode)
    use_v25_replay = (
        str(getattr(params, "strategy_profile", "")).lower() == "symbol_playbook_v25"
        and active_session_mode in {"regular_only", "regular", "regular_hours"}
        and not bool(getattr(params, "v25_allow_generic_symbols", False))
    )
    if use_v25_replay:
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
            all_cache_symbols, "5Min", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=True, session_mode=active_session_mode
        ))
        cache_status.append(client.prefetch_stock_bars(
            all_cache_symbols, "1Day", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=True, session_mode="regular_only"
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
        session_mode=active_session_mode,
    )
    qqq_daily = client.get_stock_bars(
        symbols=["QQQ"],
        timeframe="1Day",
        start=fetch_start,
        end=fetch_end,
        feed=feed,
        adjustment="split",
        use_cache=use_cache,
        session_mode="regular_only",
    )
    if qqq_5m.empty or qqq_daily.empty:
        raise RuntimeError("QQQ data is required for the market filter but was not returned.")
    qqq_context = _reduce_memory(build_qqq_context(qqq_5m, qqq_daily, session_mode=active_session_mode))
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
                session_mode=active_session_mode,
            )
            sym_daily = client.get_stock_bars(
                symbols=[symbol],
                timeframe="1Day",
                start=fetch_start,
                end=fetch_end,
                feed=feed,
                adjustment="split",
                use_cache=use_cache,
                session_mode="regular_only",
            )
            if sym_5m.empty or sym_daily.empty:
                skipped_symbols.append({"symbol": symbol, "reason": "missing_bars"})
                continue

            # V19 feature cache: after the first run, reuse prepared features
            # from disk instead of recomputing intraday, daily and QQQ merges.
            merged = pd.DataFrame()
            feature_cache_path = _feature_cache_path(symbol, feed, fetch_start, fetch_end, active_session_mode)
            if use_cache and bool(getattr(params, "enable_v19_feature_cache", True)) and not use_news:
                merged = _read_feature_cache(feature_cache_path)

            if merged.empty:
                intraday = add_intraday_features(sym_5m, session_mode=active_session_mode)
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

    openai_decisions = pd.DataFrame()
    openai_filter_diagnostics: dict[str, Any] = {"enabled": False}
    candidates_for_portfolio = candidates_all
    if bool(getattr(params, "openai_trade_filter_enabled", False)) and not candidates_all.empty:
        ai_result = review_candidates_with_openai(
            candidates_all,
            max_trades_per_day=int(getattr(params, "max_trades_per_day", 2)),
            model=str(getattr(params, "openai_trade_filter_model", "gpt-5-mini") or "gpt-5-mini"),
            max_candidates_per_day=int(getattr(params, "openai_trade_filter_max_candidates_per_day", 200) or 200),
            min_confidence=float(getattr(params, "openai_trade_filter_min_confidence", 0.0) or 0.0),
            batch_mode=str(getattr(params, "openai_trade_filter_batch_mode", "full_run") or "full_run"),
        )
        candidates_for_portfolio = ai_result.candidates
        openai_decisions = ai_result.decisions
        openai_filter_diagnostics = ai_result.diagnostics

    decision_mode = str(getattr(params, "backtest_decision_mode", "end_of_day_top_n") or "end_of_day_top_n").lower()
    if decision_mode in {"live_simulated", "live", "walk_forward", "seen_so_far_top_n"} and not candidates_for_portfolio.empty:
        candidates_for_portfolio = _v27_select_live_simulated_top_n(candidates_for_portfolio, int(getattr(params, "max_trades_per_day", 2) or 2), params)

    portfolio_trades = apply_portfolio_rules(candidates_for_portfolio, params) if not candidates_for_portfolio.empty else pd.DataFrame()
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
            "openai_decisions": openai_decisions,
            "candidates_after_openai_filter": candidates_for_portfolio,
            "portfolio_trades": portfolio_trades,
            "params": params.to_dict(),
            "symbols": tradable_symbols,
            "feed": feed,
            "start_date": start_date,
            "end_date": end_date,
            "use_news": use_news,
            "execution_timeframe": f"5Min {active_session_mode}",
            "skipped_symbols": pd.DataFrame(skipped_symbols),
            "diagnostics": {
                "symbols_scanned": len(tradable_symbols),
                "symbols_skipped": len(skipped_symbols),
                "scan_days": scan_days,
                "raw_alerts": raw_alert_count,
                "raw_candidates": int(len(candidates_all)),
                "openai_candidates_after_filter": int(len(candidates_for_portfolio)) if bool(getattr(params, "openai_trade_filter_enabled", False)) else None,
                "openai_filter": openai_filter_diagnostics,
                "selected_trades": selected_count,
                "backtest_decision_mode": str(getattr(params, "backtest_decision_mode", "end_of_day_top_n") or "end_of_day_top_n"),
                "decision_mode_note": "live_simulated processes candidates chronologically and only ranks candidates seen up to each timestamp; end_of_day_top_n preserves the original full-day research ranking.",
                "trade_days": trade_days,
                "selected_trades_per_scan_day": selected_count / scan_days if scan_days else 0.0,
                "raw_alerts_per_scan_day": raw_alert_count / scan_days if scan_days else 0.0,
                "execution_timeframe": f"5Min {active_session_mode}",
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
