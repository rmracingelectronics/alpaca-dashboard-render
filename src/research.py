from __future__ import annotations

import json
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alpaca_rest import AlpacaDataClient, pad_start_for_indicators
from .config import AlpacaSettings, RESEARCH_PACKS_DIR, StrategyParams
from .indicators import add_daily_features, add_intraday_features, build_qqq_context, merge_market_context
from .strategy import compute_signals, simulate_candidates
from .symbols import MARKET_SYMBOLS, WATCHLISTS, parse_symbols


def _plus_one_day(date_str: str) -> str:
    return (pd.Timestamp(date_str) + pd.Timedelta(days=1)).date().isoformat()


def _dt_string() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))


def _reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


def _file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except FileNotFoundError:
        return 0.0


def get_symbols_for_preset(preset: str, custom_symbols: str | None = None) -> list[str]:
    return parse_symbols(custom_symbols, preset=preset)


def preload_research_data(
    preset: str = "edge_core_40",
    start: str = "2022-01-01",
    end: str = "2026-06-18",
    feed: str = "iex",
    adjustment: str = "split",
    custom_symbols: str | None = None,
) -> dict[str, Any]:
    """Fetch 5-minute and daily bars once into the compressed local store.

    This is intentionally separated from backtesting. The first run downloads the
    missing Alpaca data; later backtests/research builds read from disk and only
    fetch missing ranges.
    """
    settings = AlpacaSettings()
    client = AlpacaDataClient(settings)
    symbols = get_symbols_for_preset(preset, custom_symbols)
    all_symbols = list(dict.fromkeys(MARKET_SYMBOLS + symbols))
    fetch_start = pad_start_for_indicators(start, days=70)
    fetch_end = _plus_one_day(end)
    started = time.time()

    status_5m = client.prefetch_stock_bars(
        all_symbols, "5Min", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True
    )
    status_1d = client.prefetch_stock_bars(
        all_symbols, "1Day", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True
    )

    # Summarize local rows and size without loading every file fully more than once.
    coverage = local_cache_status(all_symbols, start, end, feed=feed, adjustment=adjustment)
    elapsed = time.time() - started
    return {
        "preset": preset,
        "symbol_count": len(symbols),
        "symbols": symbols,
        "market_symbols": MARKET_SYMBOLS,
        "start": start,
        "end": end,
        "fetch_start_for_indicators": fetch_start,
        "fetch_end": fetch_end,
        "feed": feed,
        "adjustment": adjustment,
        "elapsed_seconds": round(elapsed, 2),
        "status_5m": status_5m,
        "status_1d": status_1d,
        "coverage_summary": coverage.get("summary", {}),
    }


def local_cache_status(
    symbols: list[str],
    start: str,
    end: str,
    feed: str = "iex",
    adjustment: str = "split",
) -> dict[str, Any]:
    settings = AlpacaSettings()
    client = AlpacaDataClient(settings)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(_plus_one_day(end), tz="UTC")
    rows: list[dict[str, Any]] = []
    total_size = 0.0
    for symbol in list(dict.fromkeys([s.upper() for s in symbols if s])):
        for timeframe in ["5Min", "1Day"]:
            path = client._local_store_path(symbol, timeframe, feed, adjustment)  # internal path but useful here
            size_mb = _file_size_mb(path)
            total_size += size_mb
            df = client._read_local_symbol(symbol, timeframe, feed, adjustment)
            if df.empty:
                rows.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "rows_in_range": 0,
                    "first_timestamp": "",
                    "last_timestamp": "",
                    "file_mb": round(size_mb, 3),
                    "path": str(path),
                    "status": "missing",
                })
                continue
            mask = (df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)
            part = df.loc[mask]
            rows.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "rows_in_range": int(len(part)),
                "first_timestamp": str(part["timestamp"].min()) if not part.empty else "",
                "last_timestamp": str(part["timestamp"].max()) if not part.empty else "",
                "file_mb": round(size_mb, 3),
                "path": str(path),
                "status": "ok" if not part.empty else "no_rows_for_requested_range",
            })
    status_df = pd.DataFrame(rows)
    return {
        "rows": rows,
        "summary": {
            "symbol_timeframe_files": int(len(status_df)),
            "total_file_mb": round(total_size, 2),
            "missing_files": int((status_df["status"] == "missing").sum()) if not status_df.empty else 0,
            "empty_range_files": int((status_df["status"] == "no_rows_for_requested_range").sum()) if not status_df.empty else 0,
        },
    }


def make_research_params(
    min_score: float = 45.0,
    direction_mode: str = "long_only",
    max_candidates_per_symbol_day: int = 8,
) -> StrategyParams:
    """Broad candidate generator for research, not live trading.

    It deliberately opens up the final strategy gates so the research dataset
    contains winners, losers, and near-misses. We then discover which features
    actually separate the winners from the losers.
    """
    return StrategyParams(
        strategy_profile="research_v16_broad_candidates",
        direction_mode=direction_mode,
        initial_account_value=10_000.0,
        risk_per_trade_dollars=100.0,
        requested_risk_percent=1.0,
        risk_per_trade_pct=0.01,
        min_candidate_score=float(min_score),
        max_trades_per_day=999,
        max_open_positions=999,
        max_alerts_per_symbol_per_day=int(max_candidates_per_symbol_day),
        candle_pattern_mode="off",
        enable_mean_reversion=True,
        enable_or_retest=True,
        enable_or_retest_only_rejection=False,
        enable_v8_regime_filters=False,
        enable_v12_core_filter=False,
        enable_v14_long_period_filters=False,
        v12_morning_only=False,
        primary_start="09:45",
        primary_end="14:30",
        afternoon_start="14:30",
        afternoon_end="15:30",
        min_avg_20d_dollar_volume=30_000_000.0,
        min_current_5m_dollar_volume=200_000.0,
        min_daily_atr_pct=0.40,
        max_daily_atr_pct=12.0,
        min_gap_percent=0.0,
        min_rvol_reason=0.90,
        min_day_relative_strength=0.0,
        min_open_relative_strength=0.0,
        rvol_min=0.80,
        max_vwap_extension_atr=2.50,
        max_candle_range_atr=3.0,
        max_entry_chase_atr=0.30,
    )


RESEARCH_COLUMNS = [
    "symbol", "side", "signal_time_et", "entry_time_et", "exit_time_et", "session_date", "entry_hour_et",
    "trigger_type", "setup_family", "candidate_score", "supporting_score", "quality",
    "entry_candle_pattern", "candle_pattern_score", "entry_candle_ok", "opposing_candle_warning_at_entry",
    "entry_price", "stop_price", "target1", "target2", "risk_per_share", "r_multiple", "mfe_r", "mae_r",
    "target1_hit", "target2_hit", "exit_reason", "duration_minutes",
    "gap_percent", "rvol_time_of_day", "day_relative_strength", "open_relative_strength", "stock_day_change_percent", "qqq_day_change_percent",
    "vwap_extension_atr", "daily_atr14_percent", "low_followthrough_mode", "v8_trade_context", "v8_regime_ok", "v8_candle_quality_ok",
]


def _add_research_labels(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    for col in ["mfe_r", "mae_r", "r_multiple", "candidate_score", "rvol_time_of_day", "day_relative_strength", "open_relative_strength", "vwap_extension_atr", "daily_atr14_percent"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["winner_r_gt_0"] = out["r_multiple"] > 0
    out["profitable_after_costs"] = out["r_multiple"] > 0.05
    out["hit_0_5r"] = out["mfe_r"] >= 0.50
    out["hit_0_75r"] = out["mfe_r"] >= 0.75
    out["hit_1_0r"] = out["mfe_r"] >= 1.00
    out["lost_more_than_0_5r"] = out["mae_r"] <= -0.50
    out["bad_trade"] = (out["r_multiple"] <= -0.35) | ((out["mfe_r"] < 0.35) & (out["mae_r"] <= -0.50))
    out["excellent_trade"] = (out["r_multiple"] >= 0.50) | (out["mfe_r"] >= 1.0)
    out["month"] = pd.to_datetime(out["entry_time_et"], errors="coerce").dt.to_period("M").astype(str)
    out["year"] = pd.to_datetime(out["entry_time_et"], errors="coerce").dt.year
    return out


def build_research_pack(
    preset: str = "edge_core_40",
    start: str = "2022-01-01",
    end: str = "2026-06-18",
    feed: str = "iex",
    adjustment: str = "split",
    direction_mode: str = "long_only",
    min_score: float = 45.0,
    custom_symbols: str | None = None,
    preload_missing: bool = True,
) -> dict[str, Any]:
    settings = AlpacaSettings()
    client = AlpacaDataClient(settings)
    symbols = get_symbols_for_preset(preset, custom_symbols)
    all_symbols = list(dict.fromkeys(MARKET_SYMBOLS + symbols))
    fetch_start = pad_start_for_indicators(start, days=70)
    fetch_end = _plus_one_day(end)
    if preload_missing:
        preload_research_data(preset=preset, start=start, end=end, feed=feed, adjustment=adjustment, custom_symbols=custom_symbols)

    params = make_research_params(min_score=min_score, direction_mode=direction_mode)
    started = time.time()

    qqq_5m = client.get_stock_bars(["QQQ"], "5Min", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
    qqq_1d = client.get_stock_bars(["QQQ"], "1Day", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
    qqq_context = build_qqq_context(qqq_5m, qqq_1d)
    start_ts = pd.Timestamp(start, tz="America/New_York")
    end_ts = pd.Timestamp(_plus_one_day(end), tz="America/New_York")

    candidate_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    raw_signal_counts: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            sym_5m = client.get_stock_bars([symbol], "5Min", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
            sym_1d = client.get_stock_bars([symbol], "1Day", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
            if sym_5m.empty or sym_1d.empty:
                skipped.append({"symbol": symbol, "reason": "missing_bars"})
                continue
            intraday = add_intraday_features(sym_5m)
            intraday = add_daily_features(intraday, sym_1d)
            merged = merge_market_context(intraday, qqq_context)
            signals = compute_signals(merged, params)
            signals = signals[(signals["timestamp_ny"] >= start_ts) & (signals["timestamp_ny"] < end_ts)].copy()
            buy_count = int(signals.get("buy_alert", pd.Series(False, index=signals.index)).fillna(False).astype(bool).sum()) if not signals.empty else 0
            raw_signal_counts.append({"symbol": symbol, "raw_alerts": buy_count, "rows_scanned": int(len(signals))})
            if signals.empty or buy_count == 0:
                continue
            candidates = simulate_candidates(signals, params)
            if not candidates.empty:
                cols = [c for c in RESEARCH_COLUMNS if c in candidates.columns]
                candidate_frames.append(_reduce_memory(candidates[cols].copy()))
        except Exception as exc:
            skipped.append({"symbol": symbol, "reason": f"error: {exc}"})

    candidates_all = pd.concat(candidate_frames, ignore_index=True).sort_values("entry_time_et").reset_index(drop=True) if candidate_frames else pd.DataFrame()
    candidates_all = _add_research_labels(candidates_all)

    pack_name = f"research_pack_{_safe_name(preset)}_{start}_{end}_{_dt_string()}"
    out_dir = RESEARCH_PACKS_DIR / pack_name
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = out_dir / "research_candidates.csv.gz"
    candidates_all.to_csv(candidates_path, index=False, compression="gzip")

    summary_by_symbol = _summary_table(candidates_all, "symbol")
    summary_by_setup = _summary_table(candidates_all, "setup_family")
    summary_by_trigger = _summary_table(candidates_all, "trigger_type")
    summary_by_candle = _summary_table(candidates_all, "entry_candle_pattern")
    summary_by_month = _summary_table(candidates_all, "month")
    for name, df in [
        ("summary_by_symbol.csv", summary_by_symbol),
        ("summary_by_setup.csv", summary_by_setup),
        ("summary_by_trigger.csv", summary_by_trigger),
        ("summary_by_candle.csv", summary_by_candle),
        ("summary_by_month.csv", summary_by_month),
        ("raw_signal_counts.csv", pd.DataFrame(raw_signal_counts)),
        ("skipped_symbols.csv", pd.DataFrame(skipped)),
        ("feature_dictionary.csv", feature_dictionary()),
    ]:
        df.to_csv(out_dir / name, index=False)

    manifest = {
        "pack_type": "candidate_research_dataset",
        "version": "v16",
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "preset": preset,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "market_symbols": MARKET_SYMBOLS,
        "start": start,
        "end": end,
        "feed": feed,
        "adjustment": adjustment,
        "direction_mode": direction_mode,
        "min_candidate_score": min_score,
        "candidate_rows": int(len(candidates_all)),
        "elapsed_seconds": round(time.time() - started, 2),
        "params": asdict(params),
        "important_note": "Upload this ZIP, not the raw local_bars folder. It contains compact candidate features/outcomes for analysis.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    zip_path = RESEARCH_PACKS_DIR / f"{pack_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file in sorted(out_dir.iterdir()):
            zf.write(file, arcname=file.name)

    return {
        "zip_path": str(zip_path),
        "folder": str(out_dir),
        "candidate_rows": int(len(candidates_all)),
        "zip_mb": round(_file_size_mb(zip_path), 2),
        "elapsed_seconds": manifest["elapsed_seconds"],
        "manifest": manifest,
    }


def _summary_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "candidates", "win_rate", "avg_r", "avg_mfe_r", "avg_mae_r", "hit_0_5r_rate", "excellent_rate", "bad_rate"])
    grouped = df.groupby(group_col, dropna=False)
    out = grouped.agg(
        candidates=("r_multiple", "size"),
        win_rate=("winner_r_gt_0", "mean"),
        avg_r=("r_multiple", "mean"),
        median_r=("r_multiple", "median"),
        avg_mfe_r=("mfe_r", "mean"),
        avg_mae_r=("mae_r", "mean"),
        hit_0_5r_rate=("hit_0_5r", "mean"),
        hit_1_0r_rate=("hit_1_0r", "mean"),
        excellent_rate=("excellent_trade", "mean"),
        bad_rate=("bad_trade", "mean"),
    ).reset_index()
    return out.sort_values(["avg_r", "candidates"], ascending=[False, False])


def feature_dictionary() -> pd.DataFrame:
    rows = [
        ("r_multiple", "Final simulated trade result in R units after exits/slippage."),
        ("mfe_r", "Maximum favorable excursion in R; how far the trade moved in favor."),
        ("mae_r", "Maximum adverse excursion in R; how far the trade moved against the entry."),
        ("hit_0_5r", "True if the candidate reached +0.5R at any point."),
        ("hit_1_0r", "True if the candidate reached +1.0R at any point."),
        ("bad_trade", "True if the candidate had materially poor R/MFE/MAE behaviour."),
        ("excellent_trade", "True if the candidate reached strong MFE or final R."),
        ("candidate_score", "Rule-based score at signal time."),
        ("setup_family", "Broad setup family such as trend_pullback or vwap_reclaim_reversal."),
        ("trigger_type", "Specific trigger subtype used by the strategy engine."),
        ("entry_candle_pattern", "Candle classification at the signal bar."),
        ("rvol_time_of_day", "Volume relative to typical volume for that time of day."),
        ("day_relative_strength", "Stock intraday change minus QQQ intraday change."),
        ("open_relative_strength", "Stock change from open minus QQQ change from open."),
        ("vwap_extension_atr", "Distance from VWAP measured in 5-minute ATR units."),
        ("daily_atr14_percent", "Daily ATR(14) as percent of price."),
        ("low_followthrough_mode", "Whether daily ATR regime suggests weak continuation."),
    ]
    return pd.DataFrame(rows, columns=["feature", "description"])

# ---------------------------------------------------------------------------
# V17 OPPORTUNITY DISCOVERY DATASET
# ---------------------------------------------------------------------------
# This is intentionally independent from the current strategy signal engine.
# It does NOT call compute_signals() or simulate_candidates(). It treats each
# intraday 5-minute candle as a possible decision point, calculates only
# features available at that moment, then labels what happened afterwards.

OPPORTUNITY_FEATURE_COLUMNS = [
    "symbol", "side", "dataset_split", "signal_time_et", "entry_time_et", "session_date", "time_str", "entry_hour_et",
    "entry_price", "risk_per_share", "signal_close", "signal_high", "signal_low", "signal_volume",
    "future_horizon_bars", "mfe_r", "mae_r", "best_r", "worst_r", "hit_0_5r", "hit_1_0r", "hit_1_5r", "hit_2_0r", "hit_stop", "first_touch", "opportunity_label",
    "open", "high", "low", "close", "volume", "current_5m_dollar_volume", "avg_20d_dollar_volume",
    "ema9", "ema20", "ema50", "ema9_above_ema20", "ema20_above_ema50", "close_above_vwap", "close_above_ema9", "close_above_ema20",
    "session_vwap", "vwap_extension_atr", "atr5m14", "daily_atr14_percent", "gap_percent", "stock_day_change_percent", "stock_change_from_open",
    "qqq_day_change_percent", "qqq_change_from_open", "qqq_15min_change_percent", "day_relative_strength", "open_relative_strength",
    "rvol_time_of_day", "body_pct", "upper_wick_pct", "lower_wick_pct", "candle_close_position", "candle_pattern_primary",
    "opening_range_high", "opening_range_low", "opening_30_high", "opening_30_low", "intraday_high_so_far", "intraday_low_so_far",
    "distance_to_or_high_atr", "distance_to_prev_high_atr", "distance_to_hod_atr", "distance_to_lod_atr", "range_position_day",
    "qqq_close", "qqq_session_vwap", "qqq_ema9", "qqq_ema20", "qqq_15m_close", "qqq_15m_ema50", "market_filter_pass",
]


def _ensure_opportunity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add strategy-independent features used by the opportunity discovery dataset."""
    if df.empty:
        return df
    out = df.copy()
    for col in ["open", "high", "low", "close", "volume", "atr5m14", "session_vwap", "prev_close", "session_open"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["stock_day_change_percent"] = (out["close"] - out["prev_close"]) / out["prev_close"].replace(0, np.nan) * 100.0
    out["stock_change_from_open"] = (out["close"] - out["session_open"]) / out["session_open"].replace(0, np.nan) * 100.0
    out["gap_percent"] = (out["session_open"] - out["prev_close"]) / out["prev_close"].replace(0, np.nan) * 100.0
    if "qqq_day_change_percent" in out.columns:
        out["day_relative_strength"] = out["stock_day_change_percent"] - out["qqq_day_change_percent"]
    else:
        out["day_relative_strength"] = np.nan
    if "qqq_change_from_open" in out.columns:
        out["open_relative_strength"] = out["stock_change_from_open"] - out["qqq_change_from_open"]
    else:
        out["open_relative_strength"] = np.nan
    out["vwap_extension_atr"] = (out["close"] - out["session_vwap"]) / out["atr5m14"].replace(0, np.nan)
    out["ema9_above_ema20"] = out["ema9"] > out["ema20"]
    out["ema20_above_ema50"] = out["ema20"] > out["ema50"]
    out["close_above_vwap"] = out["close"] > out["session_vwap"]
    out["close_above_ema9"] = out["close"] > out["ema9"]
    out["close_above_ema20"] = out["close"] > out["ema20"]
    atr = out["atr5m14"].replace(0, np.nan)
    out["distance_to_or_high_atr"] = (out["close"] - out["opening_range_high"]) / atr
    out["distance_to_prev_high_atr"] = (out["close"] - out["prev_day_high"]) / atr
    out["distance_to_hod_atr"] = (out["close"] - out["intraday_high_so_far"]) / atr
    out["distance_to_lod_atr"] = (out["close"] - out["intraday_low_so_far"]) / atr
    day_range = (out["intraday_high_so_far"] - out["intraday_low_so_far"]).replace(0, np.nan)
    out["range_position_day"] = (out["close"] - out["intraday_low_so_far"]) / day_range
    return out


def _future_extremes_for_session(group: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """Calculate next-bar entry and future high/low after each signal bar.

    Signal at row i means features are known at candle close i. Entry is next
    candle open i+1. Future path is bars i+1 through i+horizon_bars, clipped by
    session end. This avoids using future data in features.
    """
    g = group.sort_values("timestamp").copy().reset_index(drop=True)
    high = pd.to_numeric(g["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(g["low"], errors="coerce").to_numpy(dtype="float64")
    open_ = pd.to_numeric(g["open"], errors="coerce").to_numpy(dtype="float64")
    n = len(g)
    entry_open = np.full(n, np.nan, dtype="float64")
    fut_high = np.full(n, np.nan, dtype="float64")
    fut_low = np.full(n, np.nan, dtype="float64")
    bars_available = np.zeros(n, dtype="int16")
    first_touch_1r_stop_long: list[str] = ["none"] * n
    first_touch_1r_stop_short: list[str] = ["none"] * n

    for i in range(n - 1):
        start = i + 1
        end = min(n, i + 1 + horizon_bars)
        if start >= end:
            continue
        entry_open[i] = open_[start]
        h_slice = high[start:end]
        l_slice = low[start:end]
        fut_high[i] = np.nanmax(h_slice) if h_slice.size else np.nan
        fut_low[i] = np.nanmin(l_slice) if l_slice.size else np.nan
        bars_available[i] = end - start

    g["entry_price_next_open"] = entry_open
    g["future_high"] = fut_high
    g["future_low"] = fut_low
    g["future_bars_available"] = bars_available
    return g


def _classify_opportunities(base: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """Expand each base signal row into long and short opportunity rows."""
    if base.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    # Risk unit intentionally independent from old strategy. It normalizes the
    # opportunity label using local volatility, not strategy stop logic.
    risk = np.maximum(pd.to_numeric(base["atr5m14"], errors="coerce") * 0.75, pd.to_numeric(base["close"], errors="coerce") * 0.003)
    risk = risk.replace(0, np.nan)
    entry = pd.to_numeric(base["entry_price_next_open"], errors="coerce")
    fut_high = pd.to_numeric(base["future_high"], errors="coerce")
    fut_low = pd.to_numeric(base["future_low"], errors="coerce")

    for side in ["long", "short"]:
        out = base.copy()
        out["side"] = side
        out["entry_price"] = entry
        out["risk_per_share"] = risk
        if side == "long":
            out["mfe_r"] = (fut_high - entry) / risk
            out["mae_r"] = (fut_low - entry) / risk
            out["best_r"] = out["mfe_r"]
            out["worst_r"] = out["mae_r"]
        else:
            out["mfe_r"] = (entry - fut_low) / risk
            out["mae_r"] = (entry - fut_high) / risk
            out["best_r"] = out["mfe_r"]
            out["worst_r"] = out["mae_r"]
        out["future_horizon_bars"] = int(horizon_bars)
        out["hit_0_5r"] = out["mfe_r"] >= 0.5
        out["hit_1_0r"] = out["mfe_r"] >= 1.0
        out["hit_1_5r"] = out["mfe_r"] >= 1.5
        out["hit_2_0r"] = out["mfe_r"] >= 2.0
        out["hit_stop"] = out["mae_r"] <= -1.0
        # This is not a full trade simulator. It is a clean opportunity label.
        # Excellent means clear forward opportunity. Bad means stop risk or poor
        # forward movement. The final backtest must still validate a rule.
        out["opportunity_label"] = np.select(
            [
                (out["mfe_r"] >= 1.5) & (out["mae_r"] > -0.75),
                (out["mfe_r"] >= 1.0) & (out["mae_r"] > -1.0),
                (out["mfe_r"] >= 0.5) & (out["mae_r"] > -1.0),
                (out["mae_r"] <= -1.0) & (out["mfe_r"] < 0.5),
            ],
            ["excellent", "good", "scalp", "bad"],
            default="neutral",
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _sample_opportunity_upload(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    """Keep upload small while preserving all rare useful cases and a balanced negative sample."""
    if df.empty or len(df) <= max_rows:
        return df
    rng = np.random.default_rng(random_state)
    parts: list[pd.DataFrame] = []
    # Preserve the most informative classes first.
    for label, cap_frac in [("excellent", 0.28), ("good", 0.24), ("bad", 0.24), ("scalp", 0.14), ("neutral", 0.10)]:
        group = df[df["opportunity_label"] == label]
        if group.empty:
            continue
        cap = max(1000, int(max_rows * cap_frac))
        if len(group) > cap:
            group = group.sample(n=cap, random_state=random_state)
        parts.append(group)
    sampled = pd.concat(parts, ignore_index=True) if parts else df.sample(n=max_rows, random_state=random_state)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.sort_values(["signal_time_et", "symbol", "side"]).reset_index(drop=True)


def _opportunity_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=[group_col])
    g = df.groupby(group_col, dropna=False)
    out = g.agg(
        rows=("mfe_r", "size"),
        excellent_rate=("opportunity_label", lambda s: (s == "excellent").mean()),
        good_or_better_rate=("opportunity_label", lambda s: s.isin(["excellent", "good"]).mean()),
        scalp_or_better_rate=("opportunity_label", lambda s: s.isin(["excellent", "good", "scalp"]).mean()),
        bad_rate=("opportunity_label", lambda s: (s == "bad").mean()),
        avg_mfe_r=("mfe_r", "mean"),
        avg_mae_r=("mae_r", "mean"),
        med_mfe_r=("mfe_r", "median"),
        med_mae_r=("mae_r", "median"),
        hit_1r_rate=("hit_1_0r", "mean"),
        stop_rate=("hit_stop", "mean"),
    ).reset_index()
    return out.sort_values(["good_or_better_rate", "rows"], ascending=[False, False])


def opportunity_feature_dictionary() -> pd.DataFrame:
    rows = [
        ("opportunity_label", "Forward outcome label independent from the old strategy: excellent/good/scalp/neutral/bad."),
        ("mfe_r", "Maximum favorable excursion in R units over the future horizon."),
        ("mae_r", "Maximum adverse excursion in R units over the future horizon."),
        ("risk_per_share", "Volatility-normalized risk unit: max(0.75 * 5m ATR14, 0.3% of price)."),
        ("entry_price", "Next 5-minute candle open after the signal candle close."),
        ("side", "Hypothetical long or short opportunity."),
        ("rvol_time_of_day", "Current volume divided by median volume for that time slot over prior sessions."),
        ("day_relative_strength", "Stock day change minus QQQ day change at signal time."),
        ("open_relative_strength", "Stock change from open minus QQQ change from open at signal time."),
        ("vwap_extension_atr", "Distance from VWAP measured in 5-minute ATR units."),
        ("candle_pattern_primary", "Rule-based candle class computed from OHLCV only."),
        ("distance_to_hod_atr", "Distance from high-of-day-so-far measured in ATR units."),
        ("range_position_day", "Where close sits inside current intraday range, 0 low to 1 high."),
    ]
    return pd.DataFrame(rows, columns=["feature", "description"])




def _assign_dataset_split_from_ts(ts: pd.Series, train_end: str, validate_end: str) -> pd.Series:
    dt = pd.to_datetime(ts, errors="coerce").dt.tz_localize(None).dt.normalize()
    train_cut = pd.Timestamp(train_end).normalize()
    val_cut = pd.Timestamp(validate_end).normalize()
    return pd.Series(
        np.select(
            [dt <= train_cut, dt <= val_cut],
            ["train", "validate"],
            default="test",
        ),
        index=ts.index,
    )

def build_opportunity_dataset(
    preset: str = "edge_core_40",
    start: str = "2022-01-01",
    end: str = "2026-06-18",
    feed: str = "iex",
    adjustment: str = "split",
    custom_symbols: str | None = None,
    preload_missing: bool = True,
    horizon_bars: int = 12,
    start_time: str = "09:40",
    end_time: str = "14:30",
    min_avg_20d_dollar_volume: float = 20_000_000.0,
    min_5m_dollar_volume: float = 50_000.0,
    max_upload_rows: int = 200_000,
    split_train_end: str = "2024-12-31",
    split_validate_end: str = "2025-12-31",
) -> dict[str, Any]:
    """Build strategy-independent opportunity discovery dataset.

    It scans every eligible 5-minute candle for each symbol and labels the future
    long/short opportunity. The upload ZIP contains a stratified compact sample,
    top opportunities, summaries, and a manifest. Raw local bars stay on the
    user's machine.
    """
    settings = AlpacaSettings()
    client = AlpacaDataClient(settings)
    symbols = get_symbols_for_preset(preset, custom_symbols)
    all_symbols = list(dict.fromkeys(MARKET_SYMBOLS + symbols))
    fetch_start = pad_start_for_indicators(start, days=90)
    fetch_end = _plus_one_day(end)
    if preload_missing:
        preload_research_data(preset=preset, start=start, end=end, feed=feed, adjustment=adjustment, custom_symbols=custom_symbols)

    started = time.time()
    qqq_5m = client.get_stock_bars(["QQQ"], "5Min", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
    qqq_1d = client.get_stock_bars(["QQQ"], "1Day", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
    qqq_context = build_qqq_context(qqq_5m, qqq_1d)
    start_ts = pd.Timestamp(start, tz="America/New_York")
    end_ts = pd.Timestamp(_plus_one_day(end), tz="America/New_York")

    upload_frames: list[pd.DataFrame] = []
    top_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    total_rows_before_sample = 0

    for symbol in symbols:
        try:
            sym_5m = client.get_stock_bars([symbol], "5Min", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
            sym_1d = client.get_stock_bars([symbol], "1Day", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
            if sym_5m.empty or sym_1d.empty:
                skipped.append({"symbol": symbol, "reason": "missing_bars"})
                continue
            intraday = add_intraday_features(sym_5m)
            intraday = add_daily_features(intraday, sym_1d)
            merged = merge_market_context(intraday, qqq_context)
            merged = _ensure_opportunity_features(merged)
            # Only broad reality filters, not strategy filters. We avoid impossible
            # illiquid bars but do not require old setup triggers.
            mask = (
                (merged["timestamp_ny"] >= start_ts)
                & (merged["timestamp_ny"] < end_ts)
                & (merged["time_str"] >= start_time)
                & (merged["time_str"] <= end_time)
                & (merged["avg_20d_dollar_volume"] >= float(min_avg_20d_dollar_volume))
                & (merged["current_5m_dollar_volume"] >= float(min_5m_dollar_volume))
                & (merged["atr5m14"] > 0)
            )
            base = merged.loc[mask].copy()
            if base.empty:
                skipped.append({"symbol": symbol, "reason": "no_eligible_bars"})
                continue
            future_frames = []
            for _, session_group in base.groupby("session_date", sort=False):
                # Need entire session to calculate future bars, not just eligible
                # rows. Use merged session group then select base timestamps after.
                session_date = session_group["session_date"].iloc[0]
                full_session = merged[merged["session_date"] == session_date].copy()
                fut = _future_extremes_for_session(full_session, horizon_bars=horizon_bars)
                eligible_ts = set(session_group["timestamp"])
                fut = fut[fut["timestamp"].isin(eligible_ts)].copy()
                future_frames.append(fut)
            if not future_frames:
                continue
            fut_all = pd.concat(future_frames, ignore_index=True)
            fut_all = fut_all[fut_all["future_bars_available"] >= max(3, min(horizon_bars, 6))].copy()
            if fut_all.empty:
                continue
            opp = _classify_opportunities(fut_all, horizon_bars=horizon_bars)
            if opp.empty:
                continue
            opp["signal_time_et"] = pd.to_datetime(opp["timestamp_ny"]).astype(str)
            opp["dataset_split"] = _assign_dataset_split_from_ts(opp["timestamp_ny"], split_train_end, split_validate_end)
            # Entry time is next bar timestamp, approximate from signal + 5 min.
            opp["entry_time_et"] = (pd.to_datetime(opp["timestamp_ny"]) + pd.Timedelta(minutes=5)).astype(str)
            opp["entry_hour_et"] = pd.to_datetime(opp["timestamp_ny"]).dt.hour
            opp["signal_close"] = opp["close"]
            opp["signal_high"] = opp["high"]
            opp["signal_low"] = opp["low"]
            opp["signal_volume"] = opp["volume"]
            total_rows_before_sample += int(len(opp))
            cols = [c for c in OPPORTUNITY_FEATURE_COLUMNS if c in opp.columns]
            opp_small = _reduce_memory(opp[cols].copy())
            # Preserve top opportunities separately for easy analysis.
            top = opp_small[opp_small["opportunity_label"].isin(["excellent", "good", "bad"])].copy()
            if len(top) > 6000:
                top = top.sample(n=6000, random_state=42)
            top_frames.append(top)
            # Per-symbol sample to avoid one highly active symbol dominating.
            per_symbol_cap = max(2000, int(max_upload_rows / max(len(symbols), 1)))
            upload_frames.append(_sample_opportunity_upload(opp_small, max_rows=per_symbol_cap))
            summary_frames.append(_opportunity_summary(opp_small, "opportunity_label").assign(symbol=symbol))
        except Exception as exc:
            skipped.append({"symbol": symbol, "reason": f"error: {exc}"})

    upload_df = pd.concat(upload_frames, ignore_index=True) if upload_frames else pd.DataFrame()
    upload_df = _sample_opportunity_upload(upload_df, max_rows=max_upload_rows) if not upload_df.empty else upload_df
    top_df = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    if not top_df.empty:
        top_df = top_df.sort_values(["opportunity_label", "mfe_r"], ascending=[True, False]).reset_index(drop=True)
    summary_by_symbol = _opportunity_summary(upload_df, "symbol")
    summary_by_side = _opportunity_summary(upload_df, "side")
    summary_by_split = _opportunity_summary(upload_df, "dataset_split")
    summary_by_candle = _opportunity_summary(upload_df, "candle_pattern_primary")
    summary_by_hour = _opportunity_summary(upload_df, "entry_hour_et")
    if not upload_df.empty:
        # signal_time_et may contain mixed DST offsets (-05:00 and -04:00).
        # Parse as UTC first, then convert back to New York time before making month buckets.
        month_ts = pd.to_datetime(upload_df["signal_time_et"], errors="coerce", utc=True)
        upload_df["month"] = month_ts.dt.tz_convert("America/New_York").dt.to_period("M").astype(str)
    else:
        upload_df["month"] = ""
    summary_by_month = _opportunity_summary(upload_df, "month") if not upload_df.empty else pd.DataFrame()
    if not upload_df.empty and {"dataset_split", "side"}.issubset(upload_df.columns):
        summary_by_split_side = _opportunity_summary(upload_df.assign(split_side=upload_df["dataset_split"].astype(str) + "_" + upload_df["side"].astype(str)), "split_side")
    else:
        summary_by_split_side = pd.DataFrame()
    if not upload_df.empty and {"dataset_split", "candle_pattern_primary"}.issubset(upload_df.columns):
        summary_by_split_candle = _opportunity_summary(upload_df.assign(split_candle=upload_df["dataset_split"].astype(str) + "_" + upload_df["candle_pattern_primary"].astype(str)), "split_candle")
    else:
        summary_by_split_candle = pd.DataFrame()

    pack_name = f"opportunity_dataset_{_safe_name(preset)}_{start}_{end}_{_dt_string()}"
    out_dir = RESEARCH_PACKS_DIR / pack_name
    out_dir.mkdir(parents=True, exist_ok=True)
    upload_df.to_csv(out_dir / "opportunity_samples.csv.gz", index=False, compression="gzip")
    top_df.to_csv(out_dir / "top_labeled_opportunities.csv.gz", index=False, compression="gzip")
    for name, df in [
        ("summary_by_symbol.csv", summary_by_symbol),
        ("summary_by_side.csv", summary_by_side),
        ("summary_by_split.csv", summary_by_split),
        ("summary_by_split_side.csv", summary_by_split_side),
        ("summary_by_candle.csv", summary_by_candle),
        ("summary_by_split_candle.csv", summary_by_split_candle),
        ("summary_by_hour.csv", summary_by_hour),
        ("summary_by_month.csv", summary_by_month),
        ("skipped_symbols.csv", pd.DataFrame(skipped)),
        ("feature_dictionary.csv", opportunity_feature_dictionary()),
    ]:
        df.to_csv(out_dir / name, index=False)

    manifest = {
        "pack_type": "strategy_independent_opportunity_dataset",
        "version": "v17",
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "preset": preset,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "start": start,
        "end": end,
        "feed": feed,
        "adjustment": adjustment,
        "horizon_bars": horizon_bars,
        "start_time": start_time,
        "end_time": end_time,
        "min_avg_20d_dollar_volume": min_avg_20d_dollar_volume,
        "min_5m_dollar_volume": min_5m_dollar_volume,
        "total_rows_before_upload_sampling": int(total_rows_before_sample),
        "upload_rows": int(len(upload_df)),
        "top_labeled_rows": int(len(top_df)),
        "max_upload_rows": int(max_upload_rows),
        "split_train_end": split_train_end,
        "split_validate_end": split_validate_end,
        "split_definition": {"train": f"<= {split_train_end}", "validate": f"> {split_train_end} and <= {split_validate_end}", "test": f"> {split_validate_end}"},
        "elapsed_seconds": round(time.time() - started, 2),
        "important_note": "This dataset is independent from the current strategy. It labels broad 5-minute long/short opportunity windows using only future outcomes; features are available at signal time.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    zip_path = RESEARCH_PACKS_DIR / f"{pack_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file in sorted(out_dir.iterdir()):
            zf.write(file, arcname=file.name)
    return {
        "zip_path": str(zip_path),
        "folder": str(out_dir),
        "upload_rows": int(len(upload_df)),
        "top_labeled_rows": int(len(top_df)),
        "zip_mb": round(_file_size_mb(zip_path), 2),
        "elapsed_seconds": manifest["elapsed_seconds"],
        "manifest": manifest,
    }
