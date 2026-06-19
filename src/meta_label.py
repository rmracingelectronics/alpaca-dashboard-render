from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .alpaca_rest import AlpacaDataClient, pad_start_for_indicators
from .config import AlpacaSettings, StrategyParams, ML_DATASETS_DIR, ML_MODELS_DIR, ML_BACKTESTS_DIR
from .indicators import add_intraday_features, add_daily_features, build_qqq_context, merge_market_context
from .research import get_symbols_for_preset, _plus_one_day, _file_size_mb
from .strategy import compute_signals
from .ml_research import _first_touch_for_session, _target_tag

NY_TZ = "America/New_York"

CORE_NUMERIC_FEATURES = [
    "candidate_score", "signal_minute_of_day", "entry_hour_et", "day_of_week", "month_num",
    "rvol_time_of_day", "daily_atr14_percent", "qqq_daily_atr14_percent", "qqq_15min_change_percent",
    "stock_day_change_percent", "stock_change_from_open", "gap_percent", "day_relative_strength", "open_relative_strength",
    "rs_accel_3", "rs_accel_6", "open_rs_accel_3", "time_since_open_atr_move",
    "vwap_extension_atr", "abs_vwap_extension_atr", "candle_range_atr", "body_pct", "upper_wick_pct", "lower_wick_pct", "candle_close_position",
    "range_position_day", "distance_to_or_high_atr", "distance_to_prev_high_atr", "distance_to_hod_atr", "distance_to_lod_atr",
    "stock_realized_vol_20", "qqq_realized_vol_20", "qqq_vol_z_780", "stock_vol_z_780",
    "ema9_above_ema20", "ema20_above_ema50", "close_above_vwap", "close_above_ema9", "close_above_ema20",
    "qqq_close_above_vwap", "qqq_ema9_above_ema20", "market_filter_pass",
    "v8_regime_ok", "v13_micro_quality_ok", "v14_filter_ok", "v15_quality_ok", "low_followthrough_context",
]

CORE_CATEGORICAL_FEATURES = [
    "side", "trigger_type", "setup_family", "candle_pattern_primary", "v8_trade_context", "time_bucket",
]


def _dt_string() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def _assign_split(ts: pd.Series, train_end: str, validate_end: str) -> pd.Series:
    dt = pd.to_datetime(ts, errors="coerce", utc=True).dt.tz_convert(NY_TZ).dt.tz_localize(None).dt.normalize()
    train_cut = pd.Timestamp(train_end).normalize()
    val_cut = pd.Timestamp(validate_end).normalize()
    return pd.Series(np.select([dt <= train_cut, dt <= val_cut], ["train", "validate"], default="test"), index=ts.index)


def _time_bucket_from_str(time_str: pd.Series) -> pd.Series:
    t = time_str.astype(str)
    return pd.Series(
        np.select(
            [t < "10:00", t < "11:00", t < "12:00", t < "13:30", t < "14:30"],
            ["0940_0955", "1000_1055", "1100_1155", "1200_1325", "1330_1425"],
            default="late",
        ),
        index=time_str.index,
    )


def add_core_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime/context features for meta-labeling on core rule candidates.

    These are leakage-safe: all values use data available at the signal candle close.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.sort_values(["symbol", "timestamp"]).copy()
    for col in ["close", "session_open", "session_vwap", "atr5m14", "qqq_close", "qqq_session_open", "qqq_session_vwap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    atr = out["atr5m14"].replace(0, np.nan)
    out["abs_vwap_extension_atr"] = out.get("vwap_extension_atr", (out["close"] - out["session_vwap"]) / atr).abs()
    if "candle_range" in out.columns:
        out["candle_range_atr"] = pd.to_numeric(out["candle_range"], errors="coerce") / atr
    else:
        out["candle_range_atr"] = (pd.to_numeric(out["high"], errors="coerce") - pd.to_numeric(out["low"], errors="coerce")) / atr
    if "time_bucket" not in out.columns and "time_str" in out.columns:
        out["time_bucket"] = _time_bucket_from_str(out["time_str"])
    if "timestamp_ny" in out.columns:
        out["signal_minute_of_day"] = out["timestamp_ny"].dt.hour * 60 + out["timestamp_ny"].dt.minute
        out["entry_hour_et"] = out["timestamp_ny"].dt.hour
        out["day_of_week"] = out["timestamp_ny"].dt.dayofweek
        out["month_num"] = out["timestamp_ny"].dt.month
    out["time_since_open_atr_move"] = (out["close"] - out["session_open"]).abs() / atr
    if "day_relative_strength" in out.columns:
        out["rs_accel_3"] = out.groupby("symbol")["day_relative_strength"].diff(3)
        out["rs_accel_6"] = out.groupby("symbol")["day_relative_strength"].diff(6)
    if "open_relative_strength" in out.columns:
        out["open_rs_accel_3"] = out.groupby("symbol")["open_relative_strength"].diff(3)
    # Rolling realized volatility in percent over last 20 5-min bars; z vs roughly 10 trading days.
    out["stock_ret_5m"] = out.groupby("symbol")["close"].pct_change()
    out["stock_realized_vol_20"] = out.groupby("symbol")["stock_ret_5m"].transform(lambda s: s.rolling(20, min_periods=10).std() * 100.0)
    out["stock_vol_z_780"] = out.groupby("symbol")["stock_realized_vol_20"].transform(lambda s: (s - s.rolling(780, min_periods=120).mean()) / s.rolling(780, min_periods=120).std())
    # QQQ columns repeat per symbol; diff within each symbol is sufficient after merge_asof.
    out["qqq_ret_5m"] = out.groupby("symbol")["qqq_close"].pct_change()
    out["qqq_realized_vol_20"] = out.groupby("symbol")["qqq_ret_5m"].transform(lambda s: s.rolling(20, min_periods=10).std() * 100.0)
    out["qqq_vol_z_780"] = out.groupby("symbol")["qqq_realized_vol_20"].transform(lambda s: (s - s.rolling(780, min_periods=120).mean()) / s.rolling(780, min_periods=120).std())
    # Make booleans numeric for models.
    for col in ["v8_regime_ok", "v13_micro_quality_ok", "v14_filter_ok", "v15_quality_ok", "low_followthrough_context", "market_filter_pass"]:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool).astype("int8")
    return out


def _prepare_qqq_context(client: AlpacaDataClient, fetch_start: str, fetch_end: str, feed: str, use_cache: bool) -> pd.DataFrame:
    qqq_5m = client.get_stock_bars(["QQQ"], "5Min", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=use_cache)
    qqq_1d = client.get_stock_bars(["QQQ"], "1Day", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=use_cache)
    if qqq_5m.empty or qqq_1d.empty:
        raise RuntimeError("QQQ data is required for core meta dataset.")
    return build_qqq_context(qqq_5m, qqq_1d)


def _add_first_touch_labels(frame: pd.DataFrame, target_values: list[float], horizon_bars: int, risk_atr_mult: float, min_risk_pct: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    parts = []
    for _, g in frame.groupby("session_date", sort=False):
        ft = _first_touch_for_session(g, horizon_bars=horizon_bars, target_values=target_values, risk_atr_mult=risk_atr_mult, min_risk_pct=min_risk_pct)
        sg = g.sort_values("timestamp").reset_index(drop=True).copy()
        sg["entry_price_ft"] = ft["entry_price"]
        sg["risk_per_share_ft"] = ft["risk_per_share"]
        sg["future_bars_available"] = ft["future_bars_available"]
        for side in ["long", "short"]:
            prefix = f"{side}_"
            sg[prefix + "mfe_r"] = ft[prefix + "mfe_r"]
            sg[prefix + "mae_r"] = ft[prefix + "mae_r"]
            sg[prefix + "final_r"] = ft[prefix + "final_r"]
            sg[prefix + "bars_to_stop"] = ft[prefix + "bars_to_stop"]
            for tv in target_values:
                tag = _target_tag(tv)
                for suffix in [f"target_{tag}_before_stop", f"stop_before_target_{tag}", f"bars_to_target_{tag}", f"outcome_r_{tag}"]:
                    sg[prefix + suffix] = ft[prefix + suffix]
        parts.append(sg)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _unify_side_outcomes(cands: pd.DataFrame, target_values: list[float]) -> pd.DataFrame:
    out = cands.copy()
    for base_col in ["mfe_r", "mae_r", "final_r", "bars_to_stop"]:
        out[base_col] = np.where(out["side"].eq("short"), out.get("short_" + base_col), out.get("long_" + base_col))
    for tv in target_values:
        tag = _target_tag(tv)
        for suffix in [f"target_{tag}_before_stop", f"stop_before_target_{tag}", f"bars_to_target_{tag}", f"outcome_r_{tag}"]:
            out[suffix] = np.where(out["side"].eq("short"), out.get("short_" + suffix), out.get("long_" + suffix))
    return out


def build_core_meta_dataset(
    preset: str = "edge_core_40",
    symbols: list[str] | None = None,
    start: str = "2022-01-01",
    end: str = "2026-06-18",
    feed: str = "iex",
    min_score: float = 80.0,
    direction_mode: str = "long_only",
    target_r_values: Iterable[float] = (0.5, 0.6, 0.75, 1.0, 1.5),
    horizon_bars: int = 12,
    split_train_end: str = "2024-12-31",
    split_validate_end: str = "2025-12-31",
    use_cache: bool = True,
) -> dict[str, Any]:
    """Build meta-label dataset only from the verified core rule triggers.

    This is the decisive pivot recommended by the feedback: primary rules propose
    high-quality bets; ML only meta-labels those candidates.
    """
    target_values = sorted(set(float(x) for x in target_r_values))
    settings = AlpacaSettings()
    client = AlpacaDataClient(settings)
    fetch_start = pad_start_for_indicators(start, days=70)
    fetch_end = _plus_one_day(end)
    symbol_list = symbols or get_symbols_for_preset(preset)
    symbol_list = [s.upper() for s in symbol_list if s and s.upper() != "QQQ"]
    qqq_context = _prepare_qqq_context(client, fetch_start, fetch_end, feed, use_cache)
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(_plus_one_day(end), tz=NY_TZ)

    params = StrategyParams(
        strategy_profile="adaptive_v19_verified_core",
        direction_mode=direction_mode,
        min_candidate_score=float(min_score),
        enable_opportunity_v18=False,
        enable_or_retest=False,
        enable_vwap_reclaim_reversal=False,
        enable_mean_reversion=False,
        candle_pattern_mode="off",
        v12_morning_only=True,
        max_trades_per_day=10,
        max_alerts_per_symbol_per_day=99,
    )
    all_candidates = []
    symbol_status = []
    started = time.time()
    for symbol in symbol_list:
        t0 = time.time()
        try:
            sym_5m = client.get_stock_bars([symbol], "5Min", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=use_cache)
            sym_1d = client.get_stock_bars([symbol], "1Day", fetch_start, fetch_end, feed=feed, adjustment="split", use_cache=use_cache)
            if sym_5m.empty or sym_1d.empty:
                symbol_status.append({"symbol": symbol, "status": "missing_bars", "rows": 0, "elapsed_seconds": round(time.time() - t0, 2)})
                continue
            intraday = add_intraday_features(sym_5m)
            intraday = add_daily_features(intraday, sym_1d)
            merged = merge_market_context(intraday, qqq_context)
            merged = add_core_meta_features(merged)
            signals = compute_signals(merged, params)
            signals = signals[(signals["timestamp_ny"] >= start_ts) & (signals["timestamp_ny"] < end_ts)].copy()
            if signals.empty:
                symbol_status.append({"symbol": symbol, "status": "no_rows", "rows": 0, "elapsed_seconds": round(time.time() - t0, 2)})
                continue
            labeled = _add_first_touch_labels(signals, target_values, horizon_bars=horizon_bars, risk_atr_mult=1.0, min_risk_pct=0.0015)
            core_mask = labeled.get("buy_alert", False).fillna(False).astype(bool)
            # High-conviction verified core only: trend/micro pullback, not broad vwap/or/retest/opportunity modules.
            trig = labeled.get("trigger_type", "").astype(str)
            core_mask = core_mask & trig.str.contains("trend_pullback|micro_pullback", case=False, na=False)
            core_mask = core_mask & (pd.to_numeric(labeled.get("candidate_score", 0), errors="coerce") >= float(min_score))
            cands = labeled.loc[core_mask].copy()
            if cands.empty:
                symbol_status.append({"symbol": symbol, "status": "no_core_candidates", "rows": 0, "elapsed_seconds": round(time.time() - t0, 2)})
                continue
            cands = _unify_side_outcomes(cands, target_values)
            cands["dataset_split"] = _assign_split(cands["timestamp"], split_train_end, split_validate_end)
            cands["signal_time_et"] = cands["timestamp_ny"].astype(str)
            cands["core_candidate_family"] = "verified_core_pullback"
            # Keep compact but rich columns.
            keep_cols = []
            for col in [
                "symbol", "side", "dataset_split", "session_date", "signal_time_et", "timestamp", "timestamp_ny", "time_str",
                "entry_price_ft", "risk_per_share_ft", "future_bars_available", "candidate_score", "trigger_type", "setup_family", "v8_trade_context",
                "candle_pattern_primary", "core_candidate_family",
            ] + CORE_NUMERIC_FEATURES + CORE_CATEGORICAL_FEATURES + ["mfe_r", "mae_r", "final_r", "bars_to_stop"]:
                if col in cands.columns and col not in keep_cols:
                    keep_cols.append(col)
            for tv in target_values:
                tag = _target_tag(tv)
                for col in [f"target_{tag}_before_stop", f"stop_before_target_{tag}", f"bars_to_target_{tag}", f"outcome_r_{tag}"]:
                    if col in cands.columns and col not in keep_cols:
                        keep_cols.append(col)
            cands = cands[keep_cols].copy()
            all_candidates.append(cands)
            symbol_status.append({"symbol": symbol, "status": "ok", "rows": int(len(cands)), "elapsed_seconds": round(time.time() - t0, 2)})
        except Exception as exc:
            symbol_status.append({"symbol": symbol, "status": "error", "error": str(exc), "rows": 0, "elapsed_seconds": round(time.time() - t0, 2)})
    dataset = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    out_dir = ML_DATASETS_DIR / f"core_meta_{preset}_{start}_{end}_score{str(min_score).replace('.', '_')}_{_dt_string()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_dir / "core_meta_candidates.csv.gz", index=False, compression="gzip")
    pd.DataFrame(symbol_status).to_csv(out_dir / "symbol_status.csv", index=False)
    if not dataset.empty:
        # Summaries.
        summary_rows = []
        tag = _target_tag(0.75 if 0.75 in target_values else target_values[0])
        out_col = f"outcome_r_{tag}"
        for group_col in ["dataset_split", "symbol", "trigger_type", "candle_pattern_primary", "entry_hour_et"]:
            if group_col in dataset.columns:
                for key, g in dataset.groupby(group_col, dropna=False):
                    r = pd.to_numeric(g[out_col], errors="coerce").fillna(0)
                    summary_rows.append({"group": group_col, "value": key, "rows": len(g), "avg_r_0_75": float(r.mean()), "win_rate_0_75": float((r > 0).mean()), "bad_rate_0_75": float((r <= -0.999).mean())})
        pd.DataFrame(summary_rows).to_csv(out_dir / "core_meta_summary.csv", index=False)
    manifest = {
        "dataset_family": "v24_core_meta_labeling",
        "preset": preset,
        "symbols": symbol_list,
        "start": start,
        "end": end,
        "feed": feed,
        "min_score": float(min_score),
        "direction_mode": direction_mode,
        "target_r_values": target_values,
        "horizon_bars": int(horizon_bars),
        "split_train_end": split_train_end,
        "split_validate_end": split_validate_end,
        "rows": int(len(dataset)),
        "elapsed_seconds": round(time.time() - started, 2),
        "important_note": "Meta-labeling dataset: only verified-core trend/micro-pullback candidates are included. ML should filter these, not mine all candles.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    zip_path = ML_DATASETS_DIR / f"{out_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file in sorted(out_dir.iterdir()):
            zf.write(file, arcname=file.name)
    return {"dataset_folder": str(out_dir), "zip_path": str(zip_path), "rows": int(len(dataset)), "symbol_status_rows": len(symbol_status), "elapsed_seconds": round(time.time() - started, 2), "zip_mb": round(_file_size_mb(zip_path), 2)}


def _metrics(df: pd.DataFrame, outcome_col: str, risk_dollars: float = 100.0) -> dict[str, Any]:
    if df is None or df.empty or outcome_col not in df.columns:
        return {"trades": 0, "win_rate": 0.0, "avg_r": 0.0, "median_r": 0.0, "profit_factor": 0.0, "pnl_dollars": 0.0, "bad_rate": 0.0, "gross_r": 0.0}
    r = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0.0)
    wins = r[r > 0]
    losses = r[r < 0]
    pf = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 1e-12 else (float("inf") if wins.sum() > 0 else 0.0)
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()),
        "median_r": float(r.median()),
        "profit_factor": pf,
        "pnl_dollars": float(r.sum() * risk_dollars),
        "bad_rate": float((r <= -0.999).mean()),
        "gross_r": float(r.sum()),
    }


def _daily_rank_filter(df: pd.DataFrame, score_col: str, max_trades_per_day: int = 3, max_per_symbol_day: int = 1) -> pd.DataFrame:
    # Preserve the input columns even when no rows pass a threshold.
    # Otherwise downstream code that checks dataset_split/symbol/etc. can crash
    # with KeyError on an empty generic DataFrame.
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy().reset_index(drop=True)
    d = df.copy()
    d["session_date"] = d["session_date"].astype(str)
    d = d.sort_values(["session_date", score_col], ascending=[True, False]).copy()
    # One per symbol/day first.
    d["symbol_rank_day"] = d.groupby(["session_date", "symbol"]).cumcount() + 1
    d = d[d["symbol_rank_day"] <= int(max_per_symbol_day)]
    d["day_rank"] = d.groupby("session_date").cumcount() + 1
    return d[d["day_rank"] <= int(max_trades_per_day)].sort_values(["session_date", "day_rank"]).reset_index(drop=True)


def _choose_threshold(threshold_df: pd.DataFrame, min_train_trades: int = 40, min_validate_trades: int = 15) -> float | None:
    if threshold_df.empty:
        return None
    piv = threshold_df.pivot_table(index="threshold", columns="split", values=["trades", "avg_r", "profit_factor"], aggfunc="first")
    candidates = []
    for thr in sorted(threshold_df["threshold"].dropna().unique()):
        row = threshold_df[threshold_df["threshold"] == thr]
        t = row[row["split"] == "train"]
        v = row[row["split"] == "validate"]
        if t.empty or v.empty:
            continue
        t0 = t.iloc[0]
        v0 = v.iloc[0]
        if int(t0["trades"]) >= min_train_trades and int(v0["trades"]) >= min_validate_trades and float(t0["avg_r"]) > 0 and float(v0["avg_r"]) > 0 and float(t0["profit_factor"]) >= 1.1 and float(v0["profit_factor"]) >= 1.05:
            candidates.append((float(thr), float(v0["avg_r"]), int(v0["trades"])))
    if not candidates:
        return None
    # Prefer robust validation EV with enough trades; not necessarily max threshold.
    candidates = sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0]


def train_core_meta_model(
    dataset_folder: str | Path,
    target_r: float = 0.75,
    risk_dollars: float = 100.0,
    max_trades_per_day: int = 3,
    include_symbol_feature: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
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
        raise RuntimeError("Core meta model requires scikit-learn and joblib. Run: pip install scikit-learn joblib") from exc
    dataset_folder = Path(dataset_folder)
    path = dataset_folder / "core_meta_candidates.csv.gz"
    if not path.exists():
        raise RuntimeError(f"Missing {path}. Build the core meta dataset first.")
    df = pd.read_csv(path)
    tag = _target_tag(target_r)
    outcome_col = f"outcome_r_{tag}"
    if outcome_col not in df.columns:
        raise RuntimeError(f"Missing {outcome_col} in core meta dataset.")
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0.0).clip(-1.25, float(target_r))
    numeric = [c for c in CORE_NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CORE_CATEGORICAL_FEATURES if c in df.columns]
    if include_symbol_feature and "symbol" in df.columns:
        categorical.append("symbol")
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical:
        df[col] = df[col].astype(str).fillna("missing")
    train = df[df["dataset_split"] == "train"].copy()
    if len(train) < 50:
        raise RuntimeError(f"Too few train candidates ({len(train)}). Lower min score or expand core candidate families.")
    X_train = train[numeric + categorical]
    y_train = train[outcome_col].astype(float)
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), categorical),
    ], sparse_threshold=0.3)
    reg = ExtraTreesRegressor(n_estimators=500, max_depth=8, min_samples_leaf=max(8, int(len(train) * 0.02)), max_features="sqrt", n_jobs=-1, random_state=random_state)
    model = Pipeline([("preprocess", preprocess), ("model", reg)])
    model.fit(X_train, y_train)
    scored = []
    metrics_rows = []
    for split, g0 in df.groupby("dataset_split", sort=False):
        g = g0.copy()
        g["predicted_ev_r"] = model.predict(g[numeric + categorical])
        scored.append(g)
        m = _metrics(g, outcome_col, risk_dollars)
        try:
            m["mae"] = float(mean_absolute_error(g[outcome_col], g["predicted_ev_r"]))
            m["r2"] = float(r2_score(g[outcome_col], g["predicted_ev_r"]))
        except Exception:
            m["mae"] = np.nan
            m["r2"] = np.nan
        m.update({"split": split, "selection": "all_core_candidates", "rows": int(len(g))})
        metrics_rows.append(m)
    scored_df = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
    thresholds = [round(x, 3) for x in np.arange(-0.10, 0.401, 0.01)]
    rows = []
    for thr in thresholds:
        sel = scored_df[pd.to_numeric(scored_df["predicted_ev_r"], errors="coerce") >= thr].copy()
        sel_rank = _daily_rank_filter(sel, "predicted_ev_r", max_trades_per_day=max_trades_per_day, max_per_symbol_day=1)
        for split in ["train", "validate", "test"]:
            g = sel_rank[sel_rank["dataset_split"] == split]
            m = _metrics(g, outcome_col, risk_dollars)
            m.update({"threshold": thr, "split": split})
            rows.append(m)
    threshold_df = pd.DataFrame(rows)
    rec = _choose_threshold(threshold_df)
    out_dir = ML_MODELS_DIR / f"core_meta_model_{dataset_folder.name}_target{tag}_{_dt_string()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    pd.DataFrame(metrics_rows).to_csv(out_dir / "core_meta_metrics.csv", index=False)
    threshold_df.to_csv(out_dir / "core_meta_threshold_sweep.csv", index=False)
    scored_df.to_csv(out_dir / "scored_core_candidates.csv.gz", index=False, compression="gzip")
    try:
        names = model.named_steps["preprocess"].get_feature_names_out()
        imp = pd.DataFrame({"feature": names, "importance": model.named_steps["model"].feature_importances_}).sort_values("importance", ascending=False)
        imp.to_csv(out_dir / "core_meta_feature_importance.csv", index=False)
    except Exception as exc:
        (out_dir / "feature_importance_error.txt").write_text(str(exc))
    try:
        Xtr = model.named_steps["preprocess"].transform(X_train)
        tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=max(10, int(len(train) * 0.05)), random_state=random_state)
        tree.fit(Xtr, y_train)
        (out_dir / "core_meta_rule_tree.txt").write_text(export_text(tree, feature_names=list(model.named_steps["preprocess"].get_feature_names_out()), max_depth=4))
    except Exception as exc:
        (out_dir / "rule_tree_error.txt").write_text(str(exc))
    manifest = {
        "model_family": "v24_core_meta_labeling",
        "dataset_folder": str(dataset_folder),
        "target_r": float(target_r),
        "outcome_col": outcome_col,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "recommended_threshold": rec,
        "risk_dollars": risk_dollars,
        "max_trades_per_day": max_trades_per_day,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "note": "ML meta-labeling only on verified-core candidates. If recommended_threshold is null, core candidates did not produce robust train/validate EV after meta filtering.",
    }
    (out_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    zip_path = ML_MODELS_DIR / f"{out_dir.name}_diagnostics.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file in sorted(out_dir.iterdir()):
            if file.name == "model.joblib":
                continue
            zf.write(file, arcname=file.name)
    return {"model_folder": str(out_dir), "diagnostics_zip": str(zip_path), "recommended_threshold": rec, "metrics": metrics_rows, "threshold_rows": int(len(threshold_df)), "rows": int(len(df))}


def backtest_core_meta_strategy(dataset_folder: str | Path, model_folder: str | Path, threshold: float | None = None, risk_dollars: float = 100.0, max_trades_per_day: int = 3) -> dict[str, Any]:
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("Backtest requires joblib/scikit-learn. Run: pip install scikit-learn joblib") from exc
    dataset_folder = Path(dataset_folder)
    model_folder = Path(model_folder)
    manifest = json.loads((model_folder / "model_manifest.json").read_text())
    threshold = manifest.get("recommended_threshold") if threshold is None else threshold
    if threshold is None:
        raise RuntimeError("This core meta model has no recommended threshold. Inspect diagnostics or pass --threshold manually.")
    model = joblib.load(model_folder / "model.joblib")
    df = pd.read_csv(dataset_folder / "core_meta_candidates.csv.gz")
    outcome_col = manifest["outcome_col"]
    numeric = [c for c in manifest.get("numeric_features", []) if c in df.columns]
    categorical = [c for c in manifest.get("categorical_features", []) if c in df.columns]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical:
        df[col] = df[col].astype(str).fillna("missing")
    df["predicted_ev_r"] = model.predict(df[numeric + categorical])
    sel = df[pd.to_numeric(df["predicted_ev_r"], errors="coerce") >= float(threshold)].copy()
    ranked = _daily_rank_filter(sel, "predicted_ev_r", max_trades_per_day=max_trades_per_day, max_per_symbol_day=1)
    if not ranked.empty:
        ranked["pnl_dollars"] = pd.to_numeric(ranked[outcome_col], errors="coerce").fillna(0.0) * float(risk_dollars)
        ranked["equity_curve"] = 10000.0 + ranked["pnl_dollars"].cumsum()
    out_dir = ML_BACKTESTS_DIR / f"core_meta_strategy_{dataset_folder.name}_thr{str(threshold).replace('.', '_').replace('-', 'neg')}_{_dt_string()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_dir / "selected_trades.csv.gz", index=False, compression="gzip")
    sel.head(200000).to_csv(out_dir / "raw_selected_over_threshold_sample.csv.gz", index=False, compression="gzip")
    for group_col, name in [("dataset_split", "summary_by_split.csv"), ("symbol", "summary_by_symbol.csv"), ("side", "summary_by_side.csv"), ("trigger_type", "summary_by_trigger.csv"), ("entry_hour_et", "summary_by_hour.csv"), ("candle_pattern_primary", "summary_by_candle.csv"), ("v8_trade_context", "summary_by_context.csv")]:
        if group_col in ranked.columns and not ranked.empty:
            rows = []
            for key, g in ranked.groupby(group_col, dropna=False):
                m = _metrics(g, outcome_col, risk_dollars)
                m[group_col] = key
                rows.append(m)
            pd.DataFrame(rows).to_csv(out_dir / name, index=False)
    summary = _metrics(ranked, outcome_col, risk_dollars)
    summary.update({"threshold": float(threshold), "risk_dollars": risk_dollars, "max_trades_per_day": max_trades_per_day, "outcome_col": outcome_col})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    zip_path = ML_BACKTESTS_DIR / f"{out_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file in sorted(out_dir.iterdir()):
            zf.write(file, arcname=file.name)
    return {"report_folder": str(out_dir), "zip_path": str(zip_path), "summary": summary, "zip_mb": round(_file_size_mb(zip_path), 2)}
