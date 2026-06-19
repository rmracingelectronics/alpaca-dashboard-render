from __future__ import annotations

import json
import math
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .alpaca_rest import AlpacaDataClient, pad_start_for_indicators
from .config import AlpacaSettings, ML_BACKTESTS_DIR, ML_DATASETS_DIR, ML_MODELS_DIR
from .indicators import add_daily_features, add_intraday_features, build_qqq_context, merge_market_context
from .research import get_symbols_for_preset, preload_research_data, _plus_one_day, _safe_name, _file_size_mb
from .symbols import MARKET_SYMBOLS

NY_TZ = "America/New_York"
DEFAULT_TARGET_R_VALUES = (0.5, 0.75, 1.0, 1.5, 2.0)

NUMERIC_FEATURES = [
    "signal_minute_of_day", "entry_hour_et", "day_of_week", "month_num",
    "open", "high", "low", "close", "volume", "current_5m_dollar_volume", "avg_20d_dollar_volume",
    "ema9", "ema20", "ema50", "rsi2", "rsi14", "atr5m14", "daily_atr14_percent",
    "gap_percent", "stock_day_change_percent", "stock_change_from_open", "qqq_day_change_percent", "qqq_change_from_open", "qqq_15min_change_percent",
    "day_relative_strength", "open_relative_strength", "directional_day_relative_strength", "directional_open_relative_strength",
    "directional_stock_day_change_percent", "directional_stock_change_from_open", "directional_gap_percent",
    "rvol_time_of_day", "vwap_extension_atr", "abs_vwap_extension_atr", "directional_vwap_extension_atr",
    "body_pct", "upper_wick_pct", "lower_wick_pct", "candle_close_position", "directional_candle_close_position",
    "range_position_day", "directional_range_position_day", "distance_to_or_high_atr", "directional_distance_to_or_high_atr",
    "distance_to_prev_high_atr", "directional_distance_to_prev_high_atr", "distance_to_hod_atr", "distance_to_lod_atr",
    "qqq_daily_atr14_percent", "qqq_rsi2", "qqq_close_above_vwap", "qqq_ema9_above_ema20", "market_filter_pass",
    "ema9_above_ema20", "ema20_above_ema50", "close_above_vwap", "close_above_ema9", "close_above_ema20",
]

CATEGORICAL_FEATURES = ["symbol", "side", "candle_pattern_primary", "time_bucket"]

IDENTITY_COLUMNS = [
    "symbol", "side", "dataset_split", "session_date", "signal_time_et", "entry_time_et", "time_str", "entry_hour_et",
    "entry_price", "risk_per_share", "entry_bar_index", "future_bars_available",
]

OUTCOME_PREFIXES = ["target_", "bars_to_target_", "outcome_r_"]


def _dt_string() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def _parse_target_values(text: str | Iterable[float] | None) -> list[float]:
    if text is None:
        return list(DEFAULT_TARGET_R_VALUES)
    if isinstance(text, str):
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
        vals = [float(p) for p in parts]
    else:
        vals = [float(x) for x in text]
    vals = sorted(set(round(v, 4) for v in vals if v > 0))
    return vals or list(DEFAULT_TARGET_R_VALUES)


def _target_tag(value: float) -> str:
    return str(value).replace(".", "_")


def _assign_split_from_ts(ts: pd.Series, train_end: str, validate_end: str) -> pd.Series:
    dt = pd.to_datetime(ts, errors="coerce").dt.tz_localize(None).dt.normalize()
    train_cut = pd.Timestamp(train_end).normalize()
    val_cut = pd.Timestamp(validate_end).normalize()
    return pd.Series(
        np.select([dt <= train_cut, dt <= val_cut], ["train", "validate"], default="test"),
        index=ts.index,
    )


def _first_touch_for_session(g: pd.DataFrame, horizon_bars: int, target_values: list[float], risk_atr_mult: float, min_risk_pct: float) -> dict[str, Any]:
    """Calculate next-open entry and conservative first-touch outcomes for one session.

    Signal candle i enters at candle i+1 open. Future path uses candles i+1 .. i+horizon.
    If target and stop are both touched in the same candle, stop is assumed first.
    """
    g = g.sort_values("timestamp").reset_index(drop=True)
    n = len(g)
    high = pd.to_numeric(g["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(g["low"], errors="coerce").to_numpy(dtype="float64")
    open_ = pd.to_numeric(g["open"], errors="coerce").to_numpy(dtype="float64")
    close = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype="float64")
    atr = pd.to_numeric(g["atr5m14"], errors="coerce").to_numpy(dtype="float64")

    entry = np.full(n, np.nan, dtype="float64")
    risk = np.full(n, np.nan, dtype="float64")
    bars_avail = np.zeros(n, dtype="int16")
    entry_idx = np.full(n, -1, dtype="int16")
    fut_close = np.full(n, np.nan, dtype="float64")

    # side -> arrays
    out: dict[str, Any] = {
        "entry_price": entry,
        "risk_per_share": risk,
        "future_bars_available": bars_avail,
        "entry_bar_index": entry_idx,
    }

    for side in ["long", "short"]:
        prefix = f"{side}_"
        out[prefix + "mfe_r"] = np.full(n, np.nan, dtype="float32")
        out[prefix + "mae_r"] = np.full(n, np.nan, dtype="float32")
        out[prefix + "final_r"] = np.full(n, np.nan, dtype="float32")
        out[prefix + "bars_to_stop"] = np.full(n, -1, dtype="int16")
        for tv in target_values:
            tag = _target_tag(tv)
            out[prefix + f"target_{tag}_before_stop"] = np.zeros(n, dtype="bool")
            out[prefix + f"stop_before_target_{tag}"] = np.zeros(n, dtype="bool")
            out[prefix + f"bars_to_target_{tag}"] = np.full(n, -1, dtype="int16")
            out[prefix + f"outcome_r_{tag}"] = np.full(n, np.nan, dtype="float32")

    for i in range(n - 1):
        start = i + 1
        end = min(n, i + 1 + horizon_bars)
        if start >= end:
            continue
        e = open_[start]
        r = max(atr[i] * risk_atr_mult if np.isfinite(atr[i]) else np.nan, close[i] * min_risk_pct if np.isfinite(close[i]) else np.nan)
        if not np.isfinite(e) or not np.isfinite(r) or r <= 0:
            continue
        entry[i] = e
        risk[i] = r
        bars_avail[i] = end - start
        entry_idx[i] = start
        fut_close[i] = close[end - 1]

        for side in ["long", "short"]:
            prefix = f"{side}_"
            if side == "long":
                mfe = (np.nanmax(high[start:end]) - e) / r
                mae = (np.nanmin(low[start:end]) - e) / r
                final_r = (fut_close[i] - e) / r if np.isfinite(fut_close[i]) else np.nan
                stop_level = e - r
            else:
                mfe = (e - np.nanmin(low[start:end])) / r
                mae = (e - np.nanmax(high[start:end])) / r
                final_r = (e - fut_close[i]) / r if np.isfinite(fut_close[i]) else np.nan
                stop_level = e + r
            out[prefix + "mfe_r"][i] = mfe
            out[prefix + "mae_r"][i] = mae
            out[prefix + "final_r"][i] = final_r

            stop_bar = -1
            for k in range(start, end):
                if side == "long":
                    stop_hit = low[k] <= stop_level
                else:
                    stop_hit = high[k] >= stop_level
                if stop_hit:
                    stop_bar = k - start + 1
                    break
            out[prefix + "bars_to_stop"][i] = stop_bar

            for tv in target_values:
                tag = _target_tag(tv)
                target_level = e + (tv * r) if side == "long" else e - (tv * r)
                target_bar = -1
                stop_first = False
                target_first = False
                for k in range(start, end):
                    if side == "long":
                        stop_hit = low[k] <= stop_level
                        target_hit = high[k] >= target_level
                    else:
                        stop_hit = high[k] >= stop_level
                        target_hit = low[k] <= target_level
                    # Conservative same-bar handling: stop first.
                    if stop_hit and target_hit:
                        stop_first = True
                        break
                    if stop_hit:
                        stop_first = True
                        break
                    if target_hit:
                        target_first = True
                        target_bar = k - start + 1
                        break
                out[prefix + f"target_{tag}_before_stop"][i] = target_first
                out[prefix + f"stop_before_target_{tag}"][i] = stop_first
                out[prefix + f"bars_to_target_{tag}"][i] = target_bar
                if target_first:
                    outcome = float(tv)
                elif stop_first:
                    outcome = -1.0
                else:
                    outcome = float(np.clip(final_r, -1.0, float(tv))) if np.isfinite(final_r) else np.nan
                out[prefix + f"outcome_r_{tag}"][i] = outcome
    return out


def _time_bucket(time_str: pd.Series) -> pd.Series:
    # Compact categories. Keep model interpretable and avoid one-hot per exact minute.
    t = time_str.astype(str)
    return pd.Series(
        np.select(
            [t < "10:00", t < "11:00", t < "12:00", t < "13:30", t < "14:30"],
            ["0940_0955", "1000_1055", "1100_1155", "1200_1325", "1330_1425"],
            default="late",
        ),
        index=time_str.index,
    )


def _prepare_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["open", "high", "low", "close", "volume", "atr5m14", "session_vwap", "prev_close", "session_open"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["stock_day_change_percent"] = (out["close"] - out["prev_close"]) / out["prev_close"].replace(0, np.nan) * 100.0
    out["stock_change_from_open"] = (out["close"] - out["session_open"]) / out["session_open"].replace(0, np.nan) * 100.0
    out["gap_percent"] = (out["session_open"] - out["prev_close"]) / out["prev_close"].replace(0, np.nan) * 100.0
    out["day_relative_strength"] = out["stock_day_change_percent"] - out.get("qqq_day_change_percent", np.nan)
    out["open_relative_strength"] = out["stock_change_from_open"] - out.get("qqq_change_from_open", np.nan)
    atr = out["atr5m14"].replace(0, np.nan)
    out["vwap_extension_atr"] = (out["close"] - out["session_vwap"]) / atr
    out["abs_vwap_extension_atr"] = out["vwap_extension_atr"].abs()
    out["ema9_above_ema20"] = (out["ema9"] > out["ema20"]).astype("int8")
    out["ema20_above_ema50"] = (out["ema20"] > out["ema50"]).astype("int8")
    out["close_above_vwap"] = (out["close"] > out["session_vwap"]).astype("int8")
    out["close_above_ema9"] = (out["close"] > out["ema9"]).astype("int8")
    out["close_above_ema20"] = (out["close"] > out["ema20"]).astype("int8")
    out["distance_to_or_high_atr"] = (out["close"] - out["opening_range_high"]) / atr
    out["distance_to_prev_high_atr"] = (out["close"] - out["prev_day_high"]) / atr
    out["distance_to_hod_atr"] = (out["close"] - out["intraday_high_so_far"]) / atr
    out["distance_to_lod_atr"] = (out["close"] - out["intraday_low_so_far"]) / atr
    day_range = (out["intraday_high_so_far"] - out["intraday_low_so_far"]).replace(0, np.nan)
    out["range_position_day"] = (out["close"] - out["intraday_low_so_far"]) / day_range
    out["qqq_close_above_vwap"] = (out.get("qqq_close", np.nan) > out.get("qqq_session_vwap", np.nan)).astype("int8")
    out["qqq_ema9_above_ema20"] = (out.get("qqq_ema9", np.nan) > out.get("qqq_ema20", np.nan)).astype("int8")
    out["market_filter_pass"] = out.get("market_filter_pass", False).fillna(False).astype("int8") if "market_filter_pass" in out.columns else 0
    out["time_bucket"] = _time_bucket(out["time_str"])
    out["signal_minute_of_day"] = out["timestamp_ny"].dt.hour * 60 + out["timestamp_ny"].dt.minute
    out["entry_hour_et"] = out["timestamp_ny"].dt.hour
    out["day_of_week"] = out["timestamp_ny"].dt.dayofweek
    out["month_num"] = out["timestamp_ny"].dt.month
    return out


def _side_rows(base: pd.DataFrame, ft: dict[str, Any], side: str, target_values: list[float]) -> pd.DataFrame:
    prefix = f"{side}_"
    out = base.copy()
    out["side"] = side
    out["entry_price"] = ft["entry_price"]
    out["risk_per_share"] = ft["risk_per_share"]
    out["future_bars_available"] = ft["future_bars_available"]
    out["entry_bar_index"] = ft["entry_bar_index"]
    out["mfe_r"] = ft[prefix + "mfe_r"]
    out["mae_r"] = ft[prefix + "mae_r"]
    out["final_r"] = ft[prefix + "final_r"]
    out["bars_to_stop"] = ft[prefix + "bars_to_stop"]
    for tv in target_values:
        tag = _target_tag(tv)
        out[f"target_{tag}_before_stop"] = ft[prefix + f"target_{tag}_before_stop"]
        out[f"stop_before_target_{tag}"] = ft[prefix + f"stop_before_target_{tag}"]
        out[f"bars_to_target_{tag}"] = ft[prefix + f"bars_to_target_{tag}"]
        out[f"outcome_r_{tag}"] = ft[prefix + f"outcome_r_{tag}"]
    # Direction-normalized features. These let one model learn long and short behaviour together.
    mult = 1.0 if side == "long" else -1.0
    out["directional_vwap_extension_atr"] = mult * out["vwap_extension_atr"]
    out["directional_day_relative_strength"] = mult * out["day_relative_strength"]
    out["directional_open_relative_strength"] = mult * out["open_relative_strength"]
    out["directional_stock_day_change_percent"] = mult * out["stock_day_change_percent"]
    out["directional_stock_change_from_open"] = mult * out["stock_change_from_open"]
    out["directional_gap_percent"] = mult * out["gap_percent"]
    out["directional_range_position_day"] = out["range_position_day"] if side == "long" else (1.0 - out["range_position_day"])
    out["directional_candle_close_position"] = out["candle_close_position"] if side == "long" else (1.0 - out["candle_close_position"])
    out["directional_distance_to_or_high_atr"] = out["distance_to_or_high_atr"] if side == "long" else -out["distance_to_or_high_atr"]
    out["directional_distance_to_prev_high_atr"] = out["distance_to_prev_high_atr"] if side == "long" else -out["distance_to_prev_high_atr"]
    out["signal_time_et"] = out["timestamp_ny"].astype(str)
    out["entry_time_et"] = (out["timestamp_ny"] + pd.Timedelta(minutes=5)).astype(str)
    return out


def _reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


def build_first_touch_ml_dataset(
    preset: str = "edge_core_40",
    start: str = "2022-01-01",
    end: str = "2026-06-18",
    feed: str = "iex",
    adjustment: str = "split",
    custom_symbols: str | None = None,
    preload_missing: bool = False,
    horizon_bars: int = 12,
    start_time: str = "09:40",
    end_time: str = "14:30",
    risk_atr_mult: float = 0.75,
    min_risk_pct: float = 0.003,
    min_avg_20d_dollar_volume: float = 20_000_000.0,
    min_5m_dollar_volume: float = 50_000.0,
    target_values: str | Iterable[float] | None = None,
    split_train_end: str = "2024-12-31",
    split_validate_end: str = "2025-12-31",
    direction_mode: str = "long_short",
) -> dict[str, Any]:
    """Build per-symbol first-touch ML dataset from local bars.

    This dataset fixes the V18 research flaw: labels require target-before-stop.
    It does not use the current strategy to decide candidates; every eligible
    5-minute candle becomes a possible long and/or short row.
    """
    started = time.time()
    target_values_list = _parse_target_values(target_values)
    settings = AlpacaSettings()
    client = AlpacaDataClient(settings)
    symbols = get_symbols_for_preset(preset, custom_symbols)
    fetch_start = pad_start_for_indicators(start, days=90)
    fetch_end = _plus_one_day(end)
    if preload_missing:
        preload_research_data(preset=preset, start=start, end=end, feed=feed, adjustment=adjustment, custom_symbols=custom_symbols)
    qqq_5m = client.get_stock_bars(["QQQ"], "5Min", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
    qqq_1d = client.get_stock_bars(["QQQ"], "1Day", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
    qqq_context = build_qqq_context(qqq_5m, qqq_1d)

    run_name = f"first_touch_ml_{_safe_name(preset)}_{start}_{end}_h{horizon_bars}_{_dt_string()}"
    out_dir = ML_DATASETS_DIR / run_name
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(_plus_one_day(end), tz=NY_TZ)
    part_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_rows = 0

    for symbol in symbols:
        symbol_started = time.time()
        try:
            sym_5m = client.get_stock_bars([symbol], "5Min", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
            sym_1d = client.get_stock_bars([symbol], "1Day", fetch_start, fetch_end, feed=feed, adjustment=adjustment, use_cache=True)
            if sym_5m.empty or sym_1d.empty:
                skipped.append({"symbol": symbol, "reason": "missing_bars"})
                continue
            intraday = add_intraday_features(sym_5m)
            intraday = add_daily_features(intraday, sym_1d)
            merged = merge_market_context(intraday, qqq_context)
            merged = _prepare_base_features(merged)
            frames = []
            for session_date, full_session in merged.groupby("session_date", sort=False):
                full_session = full_session.sort_values("timestamp").copy()
                ft = _first_touch_for_session(full_session, horizon_bars, target_values_list, risk_atr_mult, min_risk_pct)
                for side in ["long", "short"]:
                    if direction_mode == "long_only" and side != "long":
                        continue
                    if direction_mode == "short_only" and side != "short":
                        continue
                    rows = _side_rows(full_session, ft, side, target_values_list)
                    frames.append(rows)
            if not frames:
                skipped.append({"symbol": symbol, "reason": "no_sessions"})
                continue
            all_rows = pd.concat(frames, ignore_index=True)
            mask = (
                (all_rows["timestamp_ny"] >= start_ts)
                & (all_rows["timestamp_ny"] < end_ts)
                & (all_rows["time_str"] >= start_time)
                & (all_rows["time_str"] <= end_time)
                & (all_rows["avg_20d_dollar_volume"] >= float(min_avg_20d_dollar_volume))
                & (all_rows["current_5m_dollar_volume"] >= float(min_5m_dollar_volume))
                & (all_rows["future_bars_available"] >= max(3, min(horizon_bars, 6)))
                & (all_rows["entry_price"].notna())
                & (all_rows["risk_per_share"].notna())
                & (all_rows["risk_per_share"] > 0)
            )
            all_rows = all_rows.loc[mask].copy()
            if all_rows.empty:
                skipped.append({"symbol": symbol, "reason": "no_eligible_rows"})
                continue
            all_rows["dataset_split"] = _assign_split_from_ts(all_rows["timestamp_ny"], split_train_end, split_validate_end)
            keep_cols = list(dict.fromkeys(
                IDENTITY_COLUMNS
                + NUMERIC_FEATURES
                + CATEGORICAL_FEATURES
                + ["mfe_r", "mae_r", "final_r", "bars_to_stop"]
                + [f"target_{_target_tag(tv)}_before_stop" for tv in target_values_list]
                + [f"stop_before_target_{_target_tag(tv)}" for tv in target_values_list]
                + [f"bars_to_target_{_target_tag(tv)}" for tv in target_values_list]
                + [f"outcome_r_{_target_tag(tv)}" for tv in target_values_list]
            ))
            keep_cols = [c for c in keep_cols if c in all_rows.columns]
            all_rows = _reduce_memory(all_rows[keep_cols])
            part_path = parts_dir / f"{symbol}.pkl.gz"
            all_rows.to_pickle(part_path, compression="gzip")
            split_counts = all_rows["dataset_split"].value_counts(dropna=False).to_dict()
            total_rows += int(len(all_rows))
            part_summaries.append({
                "symbol": symbol,
                "rows": int(len(all_rows)),
                "file_mb": round(_file_size_mb(part_path), 3),
                "elapsed_seconds": round(time.time() - symbol_started, 2),
                "train_rows": int(split_counts.get("train", 0)),
                "validate_rows": int(split_counts.get("validate", 0)),
                "test_rows": int(split_counts.get("test", 0)),
            })
            print(json.dumps({"symbol": symbol, "rows": int(len(all_rows)), "elapsed_seconds": round(time.time()-symbol_started, 2)}), flush=True)
        except Exception as exc:
            skipped.append({"symbol": symbol, "reason": f"error: {exc}"})

    summary_df = pd.DataFrame(part_summaries)
    summary_df.to_csv(out_dir / "part_summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped_symbols.csv", index=False)
    feature_dict = pd.DataFrame(
        [(c, "numeric_feature") for c in NUMERIC_FEATURES]
        + [(c, "categorical_feature") for c in CATEGORICAL_FEATURES]
        + [(f"target_{_target_tag(tv)}_before_stop", f"Label: target {tv}R hit before 1R stop within horizon") for tv in target_values_list]
        + [(f"outcome_r_{_target_tag(tv)}", f"Conservative first-touch R outcome for target {tv}R vs 1R stop") for tv in target_values_list],
        columns=["feature", "description"],
    )
    feature_dict.to_csv(out_dir / "feature_dictionary.csv", index=False)
    manifest = {
        "pack_type": "first_touch_ml_dataset",
        "version": "v20",
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "preset": preset,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "start": start,
        "end": end,
        "feed": feed,
        "adjustment": adjustment,
        "horizon_bars": horizon_bars,
        "target_values": target_values_list,
        "risk_atr_mult": risk_atr_mult,
        "min_risk_pct": min_risk_pct,
        "start_time": start_time,
        "end_time": end_time,
        "split_train_end": split_train_end,
        "split_validate_end": split_validate_end,
        "direction_mode": direction_mode,
        "total_rows": int(total_rows),
        "elapsed_seconds": round(time.time() - started, 2),
        "parts_dir": str(parts_dir),
        "important_note": "Features use only data known at signal candle close. Labels use conservative first-touch target-before-stop sequencing after next-bar entry.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return {"dataset_folder": str(out_dir), "parts_dir": str(parts_dir), "total_rows": int(total_rows), "elapsed_seconds": manifest["elapsed_seconds"], "manifest": manifest}


def _load_dataset_parts(dataset_folder: str | Path, columns: list[str] | None = None) -> Iterable[pd.DataFrame]:
    folder = Path(dataset_folder)
    parts_dir = folder / "parts"
    for path in sorted(parts_dir.glob("*.pkl.gz")):
        df = pd.read_pickle(path)
        if columns is not None:
            cols = [c for c in columns if c in df.columns]
            df = df[cols].copy()
        yield df


def _stratified_sample(df: pd.DataFrame, label_col: str, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if df.empty or len(df) <= max_rows:
        return df
    pos = df[df[label_col].astype(bool)]
    neg = df[~df[label_col].astype(bool)]
    half = max_rows // 2
    parts = []
    if not pos.empty:
        parts.append(pos.sample(n=min(len(pos), half), random_state=random_state))
    remaining = max_rows - sum(len(p) for p in parts)
    if not neg.empty and remaining > 0:
        parts.append(neg.sample(n=min(len(neg), remaining), random_state=random_state))
    out = pd.concat(parts, ignore_index=True) if parts else df.sample(n=max_rows, random_state=random_state)
    if len(out) < max_rows:
        rest = df.drop(out.index, errors="ignore")
        if not rest.empty:
            out = pd.concat([out, rest.sample(n=min(len(rest), max_rows - len(out)), random_state=random_state)], ignore_index=True)
    return out.sample(frac=1, random_state=random_state).reset_index(drop=True)


def _load_sampled_dataset(dataset_folder: str | Path, target_col: str, max_train_rows: int, max_eval_rows_per_split: int, random_state: int = 42) -> pd.DataFrame:
    use_cols = list(dict.fromkeys(IDENTITY_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [
        target_col, target_col.replace("target_", "outcome_r_").replace("_before_stop", ""),
        "mfe_r", "mae_r", "final_r", "bars_to_stop",
    ]))
    split_frames = {"train": [], "validate": [], "test": []}
    load_cols = list(dict.fromkeys(use_cols + ["dataset_split"]))
    for part in _load_dataset_parts(dataset_folder, columns=load_cols):
        if part.empty or target_col not in part.columns:
            continue
        for split in ["train", "validate", "test"]:
            sub = part[part["dataset_split"] == split].copy()
            if sub.empty:
                continue
            cap = max_train_rows if split == "train" else max_eval_rows_per_split
            # per part cap proportional; final cap applied later
            split_frames[split].append(_stratified_sample(sub, target_col, max(1000, min(len(sub), cap // 8)), random_state=random_state))
    frames = []
    for split, parts in split_frames.items():
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True)
        cap = max_train_rows if split == "train" else max_eval_rows_per_split
        merged = _stratified_sample(merged, target_col, cap, random_state=random_state)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _metrics_for_selection(df: pd.DataFrame, outcome_col: str, risk_dollars: float = 100.0) -> dict[str, Any]:
    if df.empty:
        return {"trades": 0, "win_rate": np.nan, "avg_r": np.nan, "profit_factor": np.nan, "pnl_dollars": 0.0, "bad_rate": np.nan}
    r = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0.0)
    wins = r[r > 0]
    losses = r[r < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "trades": int(len(df)),
        "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()),
        "median_r": float(r.median()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else np.nan),
        "pnl_dollars": float(r.sum() * risk_dollars),
        "bad_rate": float((r <= -0.75).mean()),
        "gross_r": float(r.sum()),
    }


def _daily_rank_filter(df: pd.DataFrame, prob_col: str, max_trades_per_day: int, max_per_symbol_day: int = 1) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    x["session_date"] = x["session_date"].astype(str)
    x = x.sort_values(["session_date", prob_col], ascending=[True, False])
    if max_per_symbol_day > 0:
        x = x.groupby(["session_date", "symbol"], as_index=False, sort=False).head(max_per_symbol_day)
    if max_trades_per_day > 0:
        x = x.groupby("session_date", as_index=False, sort=False).head(max_trades_per_day)
    return x.reset_index(drop=True)


def train_opportunity_model(
    dataset_folder: str | Path,
    target_r: float = 0.75,
    model_type: str = "extra_trees",
    max_train_rows: int = 300_000,
    max_eval_rows_per_split: int = 150_000,
    random_state: int = 42,
    max_trades_per_day: int = 3,
    risk_dollars: float = 100.0,
) -> dict[str, Any]:
    """Train an interpretable tree-based opportunity model and evaluate by split."""
    try:
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import roc_auc_score, average_precision_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.tree import DecisionTreeClassifier, export_text
    except Exception as exc:
        raise RuntimeError("ML training requires scikit-learn and joblib. Run: pip install scikit-learn joblib") from exc

    tag = _target_tag(target_r)
    target_col = f"target_{tag}_before_stop"
    outcome_col = f"outcome_r_{tag}"
    data = _load_sampled_dataset(dataset_folder, target_col, max_train_rows, max_eval_rows_per_split, random_state=random_state)
    if data.empty:
        raise RuntimeError(f"No dataset rows loaded from {dataset_folder}. Build first-touch dataset first.")
    if outcome_col not in data.columns:
        raise RuntimeError(f"Missing {outcome_col}; rebuild dataset with target-r value {target_r}.")

    # Ensure only available feature columns are used.
    numeric = [c for c in NUMERIC_FEATURES if c in data.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in data.columns]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in categorical:
        data[col] = data[col].astype(str).fillna("missing")
    data[target_col] = data[target_col].fillna(False).astype(bool)

    train = data[data["dataset_split"] == "train"].copy()
    if train.empty:
        raise RuntimeError("Training split is empty.")
    X_train = train[numeric + categorical]
    y_train = train[target_col].astype(int)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50))]), categorical),
        ],
        sparse_threshold=0.3,
    )
    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=100, class_weight="balanced_subsample", n_jobs=-1, random_state=random_state)
    else:
        clf = ExtraTreesClassifier(n_estimators=400, max_depth=10, min_samples_leaf=80, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=random_state)
    model = Pipeline([("preprocess", preprocess), ("model", clf)])
    model.fit(X_train, y_train)

    out_dir = ML_MODELS_DIR / f"opportunity_model_{Path(dataset_folder).name}_target{tag}_{_dt_string()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")

    # Score sampled train/validate/test.
    scored_frames = []
    metrics_rows = []
    threshold_rows = []
    for split, split_df in data.groupby("dataset_split", sort=False):
        split_df = split_df.copy()
        X = split_df[numeric + categorical]
        proba = model.predict_proba(X)[:, 1]
        split_df["ml_probability"] = proba
        scored_frames.append(split_df)
        y = split_df[target_col].astype(int)
        try:
            auc = roc_auc_score(y, proba) if y.nunique() > 1 else np.nan
            ap = average_precision_score(y, proba) if y.nunique() > 1 else np.nan
        except Exception:
            auc = np.nan
            ap = np.nan
        base_metrics = _metrics_for_selection(split_df, outcome_col, risk_dollars)
        base_metrics.update({"split": split, "selection": "all_sampled", "auc": auc, "average_precision": ap, "positive_rate": float(y.mean())})
        metrics_rows.append(base_metrics)
        for threshold in np.round(np.arange(0.45, 0.96, 0.025), 3):
            selected = split_df[split_df["ml_probability"] >= threshold].copy()
            ranked = _daily_rank_filter(selected, "ml_probability", max_trades_per_day=max_trades_per_day, max_per_symbol_day=1)
            m = _metrics_for_selection(ranked, outcome_col, risk_dollars)
            m.update({"split": split, "threshold": float(threshold), "max_trades_per_day": max_trades_per_day, "selection": "threshold_daily_rank"})
            threshold_rows.append(m)

    scored_sample = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    scored_sample.to_csv(out_dir / "scored_sample.csv.gz", index=False, compression="gzip")
    metrics_df = pd.DataFrame(metrics_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    metrics_df.to_csv(out_dir / "sample_split_metrics.csv", index=False)
    threshold_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    # Feature importances.
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
        imp.to_csv(out_dir / "feature_importance.csv", index=False)
    except Exception as exc:
        imp = pd.DataFrame({"feature": [], "importance": []})
        (out_dir / "feature_importance_error.txt").write_text(str(exc))

    # Shallow surrogate decision tree for rule extraction.
    try:
        Xtr = model.named_steps["preprocess"].transform(X_train)
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(200, int(len(train) * 0.003)), class_weight="balanced", random_state=random_state)
        tree.fit(Xtr, y_train)
        names = list(model.named_steps["preprocess"].get_feature_names_out())
        rules_text = export_text(tree, feature_names=names, max_depth=4)
        (out_dir / "learned_rule_tree.txt").write_text(rules_text)
    except Exception as exc:
        (out_dir / "learned_rule_tree_error.txt").write_text(str(exc))

    # Suggested thresholds: positive avg R in train/validate/test with reasonable count.
    suggestion = threshold_df.pivot_table(index="threshold", columns="split", values=["avg_r", "profit_factor", "trades"], aggfunc="first") if not threshold_df.empty else pd.DataFrame()
    recommended_threshold = None
    if not threshold_df.empty:
        candidates = []
        for threshold, g in threshold_df.groupby("threshold"):
            by_split = {row["split"]: row for _, row in g.iterrows()}
            if not all(s in by_split for s in ["train", "validate", "test"]):
                continue
            if by_split["train"]["trades"] < 50 or by_split["validate"]["trades"] < 20 or by_split["test"]["trades"] < 10:
                continue
            if by_split["train"]["avg_r"] > 0 and by_split["validate"]["avg_r"] > 0 and by_split["test"]["avg_r"] > 0:
                score = min(by_split["train"]["avg_r"], by_split["validate"]["avg_r"], by_split["test"]["avg_r"]) * math.log1p(min(by_split["train"]["trades"], by_split["validate"]["trades"], by_split["test"]["trades"]))
                candidates.append((score, threshold))
        if candidates:
            recommended_threshold = float(sorted(candidates, reverse=True)[0][1])
    manifest = {
        "model_type": model_type,
        "dataset_folder": str(dataset_folder),
        "target_r": float(target_r),
        "target_col": target_col,
        "outcome_col": outcome_col,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "max_train_rows": max_train_rows,
        "max_eval_rows_per_split": max_eval_rows_per_split,
        "random_state": random_state,
        "recommended_threshold": recommended_threshold,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "important_note": "Thresholds are evaluated by train/validate/test split. Use backtest_learned_strategy.py on the full per-symbol dataset before trusting any threshold.",
    }
    (out_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    # Small uploadable diagnostics ZIP; model.joblib is intentionally excluded.
    diagnostics_zip = ML_MODELS_DIR / f"{out_dir.name}_diagnostics.zip"
    with zipfile.ZipFile(diagnostics_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file in sorted(out_dir.iterdir()):
            if file.name == "model.joblib":
                continue
            zf.write(file, arcname=file.name)
    return {
        "model_folder": str(out_dir),
        "diagnostics_zip": str(diagnostics_zip),
        "recommended_threshold": recommended_threshold,
        "metrics": metrics_df.to_dict("records"),
        "threshold_rows": int(len(threshold_df)),
    }


def backtest_learned_strategy(
    dataset_folder: str | Path,
    model_folder: str | Path,
    threshold: float | None = None,
    target_r: float | None = None,
    risk_dollars: float = 100.0,
    max_trades_per_day: int = 3,
    max_per_symbol_day: int = 1,
    min_probability: float | None = None,
) -> dict[str, Any]:
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("Backtest requires joblib/scikit-learn. Run: pip install scikit-learn joblib") from exc
    model_folder = Path(model_folder)
    manifest = json.loads((model_folder / "model_manifest.json").read_text())
    model = joblib.load(model_folder / "model.joblib")
    target_r = float(target_r if target_r is not None else manifest.get("target_r", 0.75))
    threshold = float(threshold if threshold is not None else (manifest.get("recommended_threshold") if manifest.get("recommended_threshold") is not None else 0.65))
    if min_probability is not None:
        threshold = float(min_probability)
    tag = _target_tag(target_r)
    outcome_col = f"outcome_r_{tag}"
    numeric = [c for c in manifest.get("numeric_features", NUMERIC_FEATURES)]
    categorical = [c for c in manifest.get("categorical_features", CATEGORICAL_FEATURES)]
    selected_parts = []
    started = time.time()
    for part in _load_dataset_parts(dataset_folder):
        if part.empty or outcome_col not in part.columns:
            continue
        for col in numeric:
            if col in part.columns:
                part[col] = pd.to_numeric(part[col], errors="coerce")
        for col in categorical:
            if col in part.columns:
                part[col] = part[col].astype(str).fillna("missing")
        feature_cols = [c for c in numeric + categorical if c in part.columns]
        proba = model.predict_proba(part[feature_cols])[:, 1]
        part = part.copy()
        part["ml_probability"] = proba
        sel = part[part["ml_probability"] >= threshold].copy()
        if not sel.empty:
            selected_parts.append(sel)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    ranked = _daily_rank_filter(selected, "ml_probability", max_trades_per_day=max_trades_per_day, max_per_symbol_day=max_per_symbol_day) if not selected.empty else selected
    if not ranked.empty:
        ranked["pnl_dollars"] = pd.to_numeric(ranked[outcome_col], errors="coerce").fillna(0.0) * float(risk_dollars)
        ranked["equity_curve"] = ranked["pnl_dollars"].cumsum() + 10_000.0
    out_dir = ML_BACKTESTS_DIR / f"learned_strategy_{Path(dataset_folder).name}_target{tag}_thr{str(threshold).replace('.', '_')}_{_dt_string()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_dir / "selected_trades.csv.gz", index=False, compression="gzip")
    selected.head(200000).to_csv(out_dir / "raw_selected_over_threshold_sample.csv.gz", index=False, compression="gzip")

    summaries = []
    if not ranked.empty:
        for group_col, name in [("dataset_split", "summary_by_split.csv"), ("symbol", "summary_by_symbol.csv"), ("side", "summary_by_side.csv"), ("entry_hour_et", "summary_by_hour.csv"), ("candle_pattern_primary", "summary_by_candle.csv")]:
            rows = []
            for key, g in ranked.groupby(group_col, dropna=False):
                m = _metrics_for_selection(g, outcome_col, risk_dollars)
                m[group_col] = key
                rows.append(m)
            pd.DataFrame(rows).to_csv(out_dir / name, index=False)
    overall = _metrics_for_selection(ranked, outcome_col, risk_dollars)
    overall.update({"threshold": threshold, "target_r": target_r, "risk_dollars": risk_dollars, "max_trades_per_day": max_trades_per_day, "elapsed_seconds": round(time.time() - started, 2)})
    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2, default=str))
    zip_path = ML_BACKTESTS_DIR / f"{out_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file in sorted(out_dir.iterdir()):
            zf.write(file, arcname=file.name)
    return {"report_folder": str(out_dir), "zip_path": str(zip_path), "summary": overall, "zip_mb": round(_file_size_mb(zip_path), 2)}

# -----------------------------------------------------------------------------
# V21: expected-value ML research model
# -----------------------------------------------------------------------------
# V20 intentionally remains available above, but its classifier workflow used a
# balanced diagnostic sample and a target-hit probability. That was useful for
# checking whether first-touch labels existed, but it overestimated real strategy
# performance. V21 trains on natural rows and ranks by predicted expected R.

LEAKAGE_SAFE_NUMERIC_FEATURES = [
    # time / calendar
    "signal_minute_of_day", "entry_hour_et", "day_of_week", "month_num",
    # normalized liquidity / volatility
    "rsi2", "rsi14", "daily_atr14_percent", "gap_percent",
    "stock_day_change_percent", "stock_change_from_open",
    "qqq_day_change_percent", "qqq_change_from_open", "qqq_15min_change_percent",
    "day_relative_strength", "open_relative_strength",
    "directional_day_relative_strength", "directional_open_relative_strength",
    "directional_stock_day_change_percent", "directional_stock_change_from_open", "directional_gap_percent",
    "rvol_time_of_day", "vwap_extension_atr", "abs_vwap_extension_atr", "directional_vwap_extension_atr",
    # candle shape / intraday position
    "body_pct", "upper_wick_pct", "lower_wick_pct", "candle_close_position", "directional_candle_close_position",
    "range_position_day", "directional_range_position_day", "distance_to_or_high_atr", "directional_distance_to_or_high_atr",
    "distance_to_prev_high_atr", "directional_distance_to_prev_high_atr", "distance_to_hod_atr", "distance_to_lod_atr",
    # market context
    "qqq_daily_atr14_percent", "qqq_rsi2", "qqq_close_above_vwap", "qqq_ema9_above_ema20", "market_filter_pass",
    # local trend booleans
    "ema9_above_ema20", "ema20_above_ema50", "close_above_vwap", "close_above_ema9", "close_above_ema20",
]

LEAKAGE_SAFE_CATEGORICAL_FEATURES = ["side", "candle_pattern_primary", "time_bucket"]
LEAKAGE_SAFE_CATEGORICAL_WITH_SYMBOL = ["symbol", "side", "candle_pattern_primary", "time_bucket"]


def _natural_sample(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    """Random sample without rebalancing labels. Keeps the market's natural base rate."""
    if df.empty or max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def _load_natural_dataset(
    dataset_folder: str | Path,
    outcome_col: str,
    max_train_rows: int,
    max_eval_rows_per_split: int,
    random_state: int = 42,
    include_symbol_feature: bool = False,
) -> pd.DataFrame:
    cats = LEAKAGE_SAFE_CATEGORICAL_WITH_SYMBOL if include_symbol_feature else LEAKAGE_SAFE_CATEGORICAL_FEATURES
    use_cols = list(dict.fromkeys(IDENTITY_COLUMNS + LEAKAGE_SAFE_NUMERIC_FEATURES + cats + [
        outcome_col, "mfe_r", "mae_r", "final_r", "bars_to_stop", "dataset_split",
    ]))
    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "validate": [], "test": []}
    for part in _load_dataset_parts(dataset_folder, columns=use_cols):
        if part.empty or outcome_col not in part.columns:
            continue
        for split in ["train", "validate", "test"]:
            sub = part[part["dataset_split"] == split].copy()
            if not sub.empty:
                split_frames[split].append(sub)
    frames = []
    for split, parts in split_frames.items():
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True)
        cap = max_train_rows if split == "train" else max_eval_rows_per_split
        frames.append(_natural_sample(merged, cap, random_state=random_state))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _metrics_for_predicted_ev_selection(df: pd.DataFrame, outcome_col: str, risk_dollars: float = 100.0) -> dict[str, Any]:
    return _metrics_for_selection(df, outcome_col, risk_dollars)


def _sweep_ev_thresholds(
    scored_df: pd.DataFrame,
    outcome_col: str,
    score_col: str,
    thresholds: list[float],
    max_trades_per_day: int,
    risk_dollars: float,
) -> pd.DataFrame:
    rows = []
    for split, split_df in scored_df.groupby("dataset_split", sort=False):
        for threshold in thresholds:
            selected = split_df[pd.to_numeric(split_df[score_col], errors="coerce") >= threshold].copy()
            ranked = _daily_rank_filter(selected, score_col, max_trades_per_day=max_trades_per_day, max_per_symbol_day=1)
            m = _metrics_for_selection(ranked, outcome_col, risk_dollars)
            m.update({"split": split, "threshold": float(threshold), "max_trades_per_day": max_trades_per_day, "selection": "ev_threshold_daily_rank"})
            rows.append(m)
    return pd.DataFrame(rows)


def _choose_ev_threshold(threshold_df: pd.DataFrame) -> float | None:
    """Choose threshold from train+validate only. Test is never used for selection."""
    if threshold_df.empty:
        return None
    candidates = []
    for threshold, g in threshold_df.groupby("threshold"):
        by_split = {row["split"]: row for _, row in g.iterrows()}
        if not all(s in by_split for s in ["train", "validate"]):
            continue
        tr = by_split["train"]
        va = by_split["validate"]
        if tr["trades"] < 100 or va["trades"] < 30:
            continue
        if not (tr["avg_r"] > 0 and va["avg_r"] > 0):
            continue
        if not (tr["profit_factor"] > 1.05 and va["profit_factor"] > 1.02):
            continue
        # Prefer robust validation expectancy and adequate trade count over max train profit.
        score = float(va["avg_r"]) * math.log1p(float(va["trades"])) + 0.25 * float(tr["avg_r"])
        candidates.append((score, float(threshold)))
    if not candidates:
        return None
    return float(sorted(candidates, reverse=True)[0][1])


def train_ev_opportunity_model(
    dataset_folder: str | Path,
    target_r: float = 0.75,
    model_type: str = "extra_trees_regressor",
    max_train_rows: int = 500_000,
    max_eval_rows_per_split: int = 250_000,
    random_state: int = 42,
    max_trades_per_day: int = 3,
    risk_dollars: float = 100.0,
    include_symbol_feature: bool = False,
) -> dict[str, Any]:
    """Train an expected-R model using natural class distribution.

    Unlike the V20 classifier, this model predicts outcome R directly and the
    diagnostic threshold sweep is based on natural samples, not 50/50 balanced
    samples. Absolute price/EMA/volume levels are excluded by default because
    they caused symbol-specific overfit in V20.
    """
    try:
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.tree import DecisionTreeRegressor, export_text
    except Exception as exc:
        raise RuntimeError("EV model training requires scikit-learn and joblib. Run: pip install scikit-learn joblib") from exc

    tag = _target_tag(target_r)
    outcome_col = f"outcome_r_{tag}"
    data = _load_natural_dataset(
        dataset_folder,
        outcome_col=outcome_col,
        max_train_rows=max_train_rows,
        max_eval_rows_per_split=max_eval_rows_per_split,
        random_state=random_state,
        include_symbol_feature=include_symbol_feature,
    )
    if data.empty:
        raise RuntimeError(f"No dataset rows loaded from {dataset_folder}.")
    if outcome_col not in data.columns:
        raise RuntimeError(f"Missing {outcome_col}; rebuild dataset with target-r value {target_r}.")

    numeric = [c for c in LEAKAGE_SAFE_NUMERIC_FEATURES if c in data.columns]
    categorical_source = LEAKAGE_SAFE_CATEGORICAL_WITH_SYMBOL if include_symbol_feature else LEAKAGE_SAFE_CATEGORICAL_FEATURES
    categorical = [c for c in categorical_source if c in data.columns]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in categorical:
        data[col] = data[col].astype(str).fillna("missing")
    data[outcome_col] = pd.to_numeric(data[outcome_col], errors="coerce").fillna(0.0).clip(-1.25, float(target_r))

    train = data[data["dataset_split"] == "train"].copy()
    if train.empty:
        raise RuntimeError("Training split is empty.")
    X_train = train[numeric + categorical]
    y_train = train[outcome_col].astype(float)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50))]), categorical),
        ],
        sparse_threshold=0.3,
    )
    if model_type == "random_forest_regressor":
        reg = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=120, max_features="sqrt", n_jobs=-1, random_state=random_state)
    else:
        reg = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=120, max_features="sqrt", n_jobs=-1, random_state=random_state)
    model = Pipeline([("preprocess", preprocess), ("model", reg)])
    model.fit(X_train, y_train)

    out_dir = ML_MODELS_DIR / f"ev_model_{Path(dataset_folder).name}_target{tag}_{_dt_string()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")

    scored_frames = []
    metrics_rows = []
    for split, split_df in data.groupby("dataset_split", sort=False):
        split_df = split_df.copy()
        X = split_df[numeric + categorical]
        pred = model.predict(X)
        split_df["predicted_ev_r"] = pred
        scored_frames.append(split_df)
        r = split_df[outcome_col]
        try:
            mae = float(mean_absolute_error(r, pred))
            r2 = float(r2_score(r, pred))
        except Exception:
            mae = np.nan
            r2 = np.nan
        m = _metrics_for_selection(split_df, outcome_col, risk_dollars)
        m.update({"split": split, "selection": "natural_sample_all", "mae": mae, "r2": r2, "rows": int(len(split_df))})
        metrics_rows.append(m)
    scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()

    # Thresholds are in predicted R. Include low thresholds because real edge may be small.
    thresholds = [round(x, 3) for x in np.arange(-0.05, 0.251, 0.01)]
    threshold_df = _sweep_ev_thresholds(scored, outcome_col, "predicted_ev_r", thresholds, max_trades_per_day, risk_dollars)
    recommended_threshold = _choose_ev_threshold(threshold_df)

    scored.head(300000).to_csv(out_dir / "scored_natural_sample.csv.gz", index=False, compression="gzip")
    pd.DataFrame(metrics_rows).to_csv(out_dir / "natural_sample_metrics.csv", index=False)
    threshold_df.to_csv(out_dir / "ev_threshold_sweep_natural_sample.csv", index=False)

    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
        imp.to_csv(out_dir / "ev_feature_importance.csv", index=False)
    except Exception as exc:
        (out_dir / "ev_feature_importance_error.txt").write_text(str(exc))

    try:
        Xtr = model.named_steps["preprocess"].transform(X_train)
        tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=max(300, int(len(train) * 0.004)), random_state=random_state)
        tree.fit(Xtr, y_train)
        names = list(model.named_steps["preprocess"].get_feature_names_out())
        (out_dir / "ev_rule_tree.txt").write_text(export_text(tree, feature_names=names, max_depth=4))
    except Exception as exc:
        (out_dir / "ev_rule_tree_error.txt").write_text(str(exc))

    manifest = {
        "model_family": "expected_value_regression_v21",
        "model_type": model_type,
        "dataset_folder": str(dataset_folder),
        "target_r": float(target_r),
        "outcome_col": outcome_col,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "include_symbol_feature": bool(include_symbol_feature),
        "max_train_rows": max_train_rows,
        "max_eval_rows_per_split": max_eval_rows_per_split,
        "random_state": random_state,
        "recommended_threshold": recommended_threshold,
        "threshold_selection_note": "Threshold selected from train+validate natural samples only; test is reported but not used to choose threshold.",
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "correction_from_v20": "Natural sampling, predicted expected R, no absolute price/EMA/volume features by default, no symbol feature by default.",
    }
    (out_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    diagnostics_zip = ML_MODELS_DIR / f"{out_dir.name}_diagnostics.zip"
    with zipfile.ZipFile(diagnostics_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file in sorted(out_dir.iterdir()):
            if file.name == "model.joblib":
                continue
            zf.write(file, arcname=file.name)
    return {
        "model_folder": str(out_dir),
        "diagnostics_zip": str(diagnostics_zip),
        "recommended_threshold": recommended_threshold,
        "metrics": pd.DataFrame(metrics_rows).to_dict("records"),
        "threshold_rows": int(len(threshold_df)),
        "important_note": "If recommended_threshold is null, do not backtest blindly. Train/validate did not show a robust EV threshold.",
    }


def backtest_ev_strategy(
    dataset_folder: str | Path,
    model_folder: str | Path,
    threshold: float | None = None,
    target_r: float | None = None,
    risk_dollars: float = 100.0,
    max_trades_per_day: int = 3,
    max_per_symbol_day: int = 1,
) -> dict[str, Any]:
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("Backtest requires joblib/scikit-learn. Run: pip install scikit-learn joblib") from exc
    model_folder = Path(model_folder)
    manifest = json.loads((model_folder / "model_manifest.json").read_text())
    model = joblib.load(model_folder / "model.joblib")
    target_r = float(target_r if target_r is not None else manifest.get("target_r", 0.75))
    threshold = manifest.get("recommended_threshold") if threshold is None else threshold
    if threshold is None:
        raise RuntimeError("This EV model has no recommended_threshold. Inspect diagnostics or pass --threshold manually.")
    threshold = float(threshold)
    tag = _target_tag(target_r)
    outcome_col = f"outcome_r_{tag}"
    numeric = [c for c in manifest.get("numeric_features", LEAKAGE_SAFE_NUMERIC_FEATURES)]
    categorical = [c for c in manifest.get("categorical_features", LEAKAGE_SAFE_CATEGORICAL_FEATURES)]
    selected_parts = []
    started = time.time()
    for part in _load_dataset_parts(dataset_folder):
        if part.empty or outcome_col not in part.columns:
            continue
        for col in numeric:
            if col in part.columns:
                part[col] = pd.to_numeric(part[col], errors="coerce")
        for col in categorical:
            if col in part.columns:
                part[col] = part[col].astype(str).fillna("missing")
        feature_cols = [c for c in numeric + categorical if c in part.columns]
        part = part.copy()
        part["predicted_ev_r"] = model.predict(part[feature_cols])
        sel = part[pd.to_numeric(part["predicted_ev_r"], errors="coerce") >= threshold].copy()
        if not sel.empty:
            selected_parts.append(sel)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    ranked = _daily_rank_filter(selected, "predicted_ev_r", max_trades_per_day=max_trades_per_day, max_per_symbol_day=max_per_symbol_day) if not selected.empty else selected
    if not ranked.empty:
        ranked["pnl_dollars"] = pd.to_numeric(ranked[outcome_col], errors="coerce").fillna(0.0) * float(risk_dollars)
        ranked["equity_curve"] = ranked["pnl_dollars"].cumsum() + 10_000.0
    out_dir = ML_BACKTESTS_DIR / f"ev_strategy_{Path(dataset_folder).name}_target{tag}_thr{str(threshold).replace('.', '_').replace('-', 'neg')}_{_dt_string()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_dir / "selected_trades.csv.gz", index=False, compression="gzip")
    selected.head(200000).to_csv(out_dir / "raw_selected_over_threshold_sample.csv.gz", index=False, compression="gzip")
    if not ranked.empty:
        for group_col, name in [("dataset_split", "summary_by_split.csv"), ("symbol", "summary_by_symbol.csv"), ("side", "summary_by_side.csv"), ("entry_hour_et", "summary_by_hour.csv"), ("candle_pattern_primary", "summary_by_candle.csv"), ("time_bucket", "summary_by_time_bucket.csv")]:
            rows = []
            for key, g in ranked.groupby(group_col, dropna=False):
                m = _metrics_for_selection(g, outcome_col, risk_dollars)
                m[group_col] = key
                rows.append(m)
            pd.DataFrame(rows).to_csv(out_dir / name, index=False)
    overall = _metrics_for_selection(ranked, outcome_col, risk_dollars)
    overall.update({"threshold": threshold, "target_r": target_r, "risk_dollars": risk_dollars, "max_trades_per_day": max_trades_per_day, "elapsed_seconds": round(time.time() - started, 2)})
    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2, default=str))
    zip_path = ML_BACKTESTS_DIR / f"{out_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file in sorted(out_dir.iterdir()):
            zf.write(file, arcname=file.name)
    return {"report_folder": str(out_dir), "zip_path": str(zip_path), "summary": overall, "zip_mb": round(_file_size_mb(zip_path), 2)}
