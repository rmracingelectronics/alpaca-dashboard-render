from __future__ import annotations

import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ML_BACKTESTS_DIR, ML_MODELS_DIR
from .ml_research import (
    _target_tag,
    _load_dataset_parts,
    _metrics_for_selection,
    _daily_rank_filter,
    _natural_sample,
    LEAKAGE_SAFE_NUMERIC_FEATURES,
    LEAKAGE_SAFE_CATEGORICAL_FEATURES,
)
from .research import _file_size_mb


def _dt_string() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _str(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[col].astype(str).fillna("")


def add_event_family_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add realistic event-family columns to first-touch rows.

    These are not final trade rules. They are candidate event gates to remove
    millions of random candles before ML tries to learn expectancy.
    Every condition uses only features available at signal-candle close.
    """
    out = df.copy()
    tb = _str(out, "time_bucket")
    side = _str(out, "side")
    candle = _str(out, "candle_pattern_primary")

    daily_atr = _num(out, "daily_atr14_percent")
    rvol = _num(out, "rvol_time_of_day")
    dvwap = _num(out, "directional_vwap_extension_atr")
    avwap = _num(out, "abs_vwap_extension_atr")
    drange = _num(out, "directional_range_position_day")
    dclosepos = _num(out, "directional_candle_close_position")
    dopenrs = _num(out, "directional_open_relative_strength")
    ddayrs = _num(out, "directional_day_relative_strength")
    dstock_open = _num(out, "directional_stock_change_from_open")
    dgap = _num(out, "directional_gap_percent")
    gap = _num(out, "gap_percent").abs()
    qqq_open = _num(out, "qqq_change_from_open")
    qqq_day = _num(out, "qqq_day_change_percent")
    qtrend = np.where(side.eq("long"), qqq_open, -qqq_open)

    close_above_vwap = _num(out, "close_above_vwap")
    side_vwap_ok = np.where(side.eq("long"), close_above_vwap > 0, close_above_vwap <= 0)
    ema_align = np.where(side.eq("long"), (_num(out, "ema9_above_ema20") > 0) & (_num(out, "ema20_above_ema50") > 0), (_num(out, "ema9_above_ema20") <= 0) & (_num(out, "ema20_above_ema50") <= 0))

    is_reject = candle.str.contains("rejection", case=False, na=False)
    is_cont = candle.str.contains("continuation", case=False, na=False)
    is_neutral = candle.str.contains("neutral", case=False, na=False)

    out["event_opening_low_atr_near_vwap"] = (tb.eq("0940_0955") & (daily_atr.between(0.4, 2.8)) & (avwap <= 1.1) & (dclosepos >= 0.45) & (rvol.between(0.4, 3.0)))
    out["event_opening_rs_push"] = (tb.eq("0940_0955") & (daily_atr <= 4.5) & (ddayrs >= 0.25) & (dstock_open >= 0.15) & (dclosepos >= 0.55) & (avwap <= 2.5))
    out["event_opening_gap_fade"] = (tb.eq("0940_0955") & (gap >= 2.0) & (dgap <= -1.0) & (dstock_open >= 0.20) & (dopenrs >= 0.15) & (avwap <= 2.5))
    out["event_vwap_controlled_reclaim_reject"] = (tb.isin(["0940_0955", "1000_1055", "1100_1155"]) & (avwap <= 0.65) & (dclosepos >= 0.55) & side_vwap_ok & (rvol >= 0.5))
    out["event_or_break_continuation"] = (tb.isin(["1000_1055", "1100_1155"]) & (_num(out, "directional_distance_to_or_high_atr") >= 0.0) & (drange >= 0.70) & (dvwap.between(0.0, 2.0)) & ((ddayrs >= 0.25) | (qtrend >= 0.10)))
    out["event_rs_pullback_vwap_ema"] = (tb.isin(["1000_1055", "1100_1155", "1200_1325"]) & (ddayrs >= 0.50) & (dvwap.between(-0.50, 1.25)) & ((side_vwap_ok) | ema_align) & (daily_atr.between(0.8, 4.5)))
    out["event_high_gap_controlled_continuation"] = (gap >= 4.0) & (dgap >= 1.5) & (dstock_open >= 0.25) & (dvwap.between(-0.25, 3.5)) & ((ddayrs >= 0.25) | (rvol >= 1.2))
    out["event_rejection_at_range_edge"] = (tb.isin(["1000_1055", "1100_1155", "1200_1325"]) & is_reject & (dclosepos >= 0.55) & ((drange >= 0.75) | (drange <= 0.25)) & (avwap <= 2.5))
    out["event_late_trend_followthrough"] = (tb.isin(["1330_1425", "late"]) & (ddayrs >= 0.75) & (dstock_open >= 0.50) & (dvwap.between(0.0, 2.0)) & (qtrend >= 0.0) & (daily_atr <= 4.5))
    out["event_compact_continuation_candle"] = (tb.isin(["1000_1055", "1100_1155", "1200_1325"]) & is_cont & (dclosepos >= 0.60) & (dvwap.between(-0.25, 1.75)) & ((ddayrs >= 0.20) | (qtrend >= 0.05)))

    event_cols = [c for c in out.columns if c.startswith("event_")]
    if event_cols:
        out["event_count"] = out[event_cols].sum(axis=1).astype("int16")
        # first matched event name for model/reporting
        arr = np.full(len(out), "none", dtype=object)
        for c in event_cols:
            mask = out[c].astype(bool).to_numpy() & (arr == "none")
            arr[mask] = c.replace("event_", "")
        out["event_family"] = arr
    else:
        out["event_count"] = 0
        out["event_family"] = "none"
    return out


def _event_columns() -> list[str]:
    return [
        "event_opening_low_atr_near_vwap",
        "event_opening_rs_push",
        "event_opening_gap_fade",
        "event_vwap_controlled_reclaim_reject",
        "event_or_break_continuation",
        "event_rs_pullback_vwap_ema",
        "event_high_gap_controlled_continuation",
        "event_rejection_at_range_edge",
        "event_late_trend_followthrough",
        "event_compact_continuation_candle",
    ]


def scan_event_patterns(dataset_folder: str | Path, target_r: float = 0.75, risk_dollars: float = 100.0, max_rows_per_event_sample: int = 200000) -> dict[str, Any]:
    tag = _target_tag(target_r)
    outcome_col = f"outcome_r_{tag}"
    rows = []
    samples = []
    started = time.time()
    use_cols = list(dict.fromkeys([
        "symbol", "side", "dataset_split", "session_date", "signal_time_et", "entry_time_et", "entry_hour_et", "time_bucket", "candle_pattern_primary",
        outcome_col, "mfe_r", "mae_r", "final_r", "bars_to_stop",
        *LEAKAGE_SAFE_NUMERIC_FEATURES, *LEAKAGE_SAFE_CATEGORICAL_FEATURES
    ]))
    for part in _load_dataset_parts(dataset_folder, columns=use_cols):
        if part.empty or outcome_col not in part.columns:
            continue
        part = add_event_family_columns(part)
        event_cols = _event_columns()
        for ev in event_cols:
            sub = part[part[ev].astype(bool)].copy()
            if sub.empty:
                continue
            for split, g in sub.groupby("dataset_split", sort=False):
                m = _metrics_for_selection(g, outcome_col, risk_dollars)
                m.update({"event": ev.replace("event_", ""), "split": split})
                rows.append(m)
            if len(samples) < 40:
                keep = sub.sample(n=min(len(sub), max_rows_per_event_sample // 40), random_state=42) if len(sub) > max_rows_per_event_sample // 40 else sub
                samples.append(keep)
        any_ev = part[part[event_cols].any(axis=1)].copy()
        if not any_ev.empty:
            for split, g in any_ev.groupby("dataset_split", sort=False):
                m = _metrics_for_selection(g, outcome_col, risk_dollars)
                m.update({"event": "ANY_EVENT_UNION", "split": split})
                rows.append(m)
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.groupby(["event", "split"], as_index=False).agg({
            "trades":"sum", "gross_r":"sum", "pnl_dollars":"sum"
        })
        # recompute approximate avg from sums; PF/win need samples, so use original group weighted means in simple way via second pass not needed for scan triage
        report["avg_r"] = report["gross_r"] / report["trades"].replace(0, np.nan)
        report = report.sort_values(["event", "split"])
    out_dir = ML_MODELS_DIR / f"event_scan_{Path(dataset_folder).name}_target{tag}_{_dt_string()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "event_split_summary.csv", index=False)
    if samples:
        pd.concat(samples, ignore_index=True).head(max_rows_per_event_sample).to_csv(out_dir / "event_sample.csv.gz", index=False, compression="gzip")
    manifest = {"dataset_folder": str(dataset_folder), "target_r": target_r, "outcome_col": outcome_col, "elapsed_seconds": round(time.time()-started,2), "note":"Event scan removes random all-candle noise. Use only events positive in train and validate for model training."}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    zip_path = ML_MODELS_DIR / f"{out_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(out_dir.iterdir()):
            zf.write(f, arcname=f.name)
    return {"report_folder": str(out_dir), "zip_path": str(zip_path), "summary_rows": int(len(report)), "elapsed_seconds": manifest["elapsed_seconds"], "zip_mb": round(_file_size_mb(zip_path),2)}


def _load_event_dataset(dataset_folder: str | Path, outcome_col: str, max_train_rows: int, max_eval_rows_per_split: int, random_state: int) -> pd.DataFrame:
    use_cols = list(dict.fromkeys([
        "symbol", "side", "dataset_split", "session_date", "signal_time_et", "entry_time_et", "entry_hour_et", "time_bucket", "candle_pattern_primary",
        outcome_col, "mfe_r", "mae_r", "final_r", "bars_to_stop", *LEAKAGE_SAFE_NUMERIC_FEATURES, *LEAKAGE_SAFE_CATEGORICAL_FEATURES
    ]))
    frames = {"train": [], "validate": [], "test": []}
    for part in _load_dataset_parts(dataset_folder, columns=use_cols):
        if part.empty or outcome_col not in part.columns:
            continue
        part = add_event_family_columns(part)
        event_cols = _event_columns()
        part = part[part[event_cols].any(axis=1)].copy()
        if part.empty:
            continue
        for split in frames:
            sub = part[part["dataset_split"].eq(split)]
            if sub.empty:
                continue
            cap = max_train_rows if split == "train" else max_eval_rows_per_split
            frames[split].append(_natural_sample(sub, max(2000, min(len(sub), cap // 8)), random_state=random_state))
    outs=[]
    for split, parts in frames.items():
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True)
        cap = max_train_rows if split == "train" else max_eval_rows_per_split
        outs.append(_natural_sample(merged, cap, random_state=random_state))
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


def train_event_ev_model(dataset_folder: str | Path, target_r: float = 0.75, max_train_rows: int = 500000, max_eval_rows_per_split: int = 250000, max_trades_per_day: int = 3, risk_dollars: float = 100.0, random_state: int = 42) -> dict[str, Any]:
    try:
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except Exception as exc:
        raise RuntimeError("Event EV training requires scikit-learn and joblib. Run: pip install scikit-learn joblib") from exc
    tag = _target_tag(target_r)
    outcome_col = f"outcome_r_{tag}"
    data = _load_event_dataset(dataset_folder, outcome_col, max_train_rows, max_eval_rows_per_split, random_state)
    if data.empty:
        raise RuntimeError("No event rows loaded. Build first-touch dataset first.")
    numeric = [c for c in LEAKAGE_SAFE_NUMERIC_FEATURES + ["event_count"] if c in data.columns]
    categorical = [c for c in LEAKAGE_SAFE_CATEGORICAL_FEATURES + ["event_family"] if c in data.columns]
    for c in numeric:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    for c in categorical:
        data[c] = data[c].astype(str).fillna("missing")
    data[outcome_col] = pd.to_numeric(data[outcome_col], errors="coerce").fillna(0.0).clip(-1.25, float(target_r))
    train = data[data["dataset_split"].eq("train")].copy()
    X_train = train[numeric + categorical]
    y_train = train[outcome_col].astype(float)
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50))]), categorical),
    ], sparse_threshold=0.3)
    reg = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=100, max_features="sqrt", n_jobs=-1, random_state=random_state)
    model = Pipeline([("preprocess", pre), ("model", reg)])
    model.fit(X_train, y_train)
    out_dir = ML_MODELS_DIR / f"event_ev_model_{Path(dataset_folder).name}_target{tag}_{_dt_string()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    scored_frames=[]; metrics=[]
    for split, g in data.groupby("dataset_split", sort=False):
        g=g.copy(); g["predicted_ev_r"] = model.predict(g[numeric+categorical]); scored_frames.append(g)
        m=_metrics_for_selection(g, outcome_col, risk_dollars)
        try: m.update({"mae": float(mean_absolute_error(g[outcome_col], g["predicted_ev_r"])), "r2": float(r2_score(g[outcome_col], g["predicted_ev_r"]))})
        except Exception: pass
        m.update({"split":split,"selection":"event_sample_all","rows":len(g)})
        metrics.append(m)
    scored=pd.concat(scored_frames, ignore_index=True)
    rows=[]
    for split,g in scored.groupby("dataset_split", sort=False):
        for threshold in [round(x,3) for x in np.arange(-0.05,0.251,0.01)]:
            sel=g[pd.to_numeric(g["predicted_ev_r"], errors="coerce")>=threshold]
            ranked=_daily_rank_filter(sel,"predicted_ev_r",max_trades_per_day=max_trades_per_day,max_per_symbol_day=1)
            m=_metrics_for_selection(ranked,outcome_col,risk_dollars)
            m.update({"split":split,"threshold":float(threshold),"selection":"event_ev_threshold_daily_rank","max_trades_per_day":max_trades_per_day})
            rows.append(m)
    threshold_df=pd.DataFrame(rows)
    recommended=None
    candidates=[]
    for thr,g in threshold_df.groupby("threshold"):
        d={r["split"]:r for _,r in g.iterrows()}
        if "train" in d and "validate" in d and d["train"]["trades"]>=100 and d["validate"]["trades"]>=30:
            if d["train"]["avg_r"]>0 and d["validate"]["avg_r"]>0 and d["train"]["profit_factor"]>1.05 and d["validate"]["profit_factor"]>1.02:
                candidates.append((float(d["validate"]["avg_r"])*math.log1p(float(d["validate"]["trades"])), float(thr)))
    if candidates:
        recommended=sorted(candidates, reverse=True)[0][1]
    pd.DataFrame(metrics).to_csv(out_dir / "event_sample_metrics.csv", index=False)
    threshold_df.to_csv(out_dir / "event_threshold_sweep.csv", index=False)
    scored.head(300000).to_csv(out_dir / "scored_event_sample.csv.gz", index=False, compression="gzip")
    try:
        imp=pd.DataFrame({"feature":model.named_steps["preprocess"].get_feature_names_out(),"importance":model.named_steps["model"].feature_importances_}).sort_values("importance",ascending=False)
        imp.to_csv(out_dir / "event_feature_importance.csv", index=False)
    except Exception as exc:
        (out_dir / "event_feature_importance_error.txt").write_text(str(exc))
    manifest={"model_family":"event_expected_value_v22","dataset_folder":str(dataset_folder),"target_r":target_r,"outcome_col":outcome_col,"numeric_features":numeric,"categorical_features":categorical,"recommended_threshold":recommended,"created_at_utc":pd.Timestamp.utcnow().isoformat(),"note":"Trains only on realistic event-family rows, not all candles."}
    (out_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    zip_path=ML_MODELS_DIR / f"{out_dir.name}_diagnostics.zip"
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
        for f in sorted(out_dir.iterdir()):
            if f.name != "model.joblib": zf.write(f,arcname=f.name)
    return {"model_folder":str(out_dir),"diagnostics_zip":str(zip_path),"recommended_threshold":recommended,"metrics":metrics,"threshold_rows":len(threshold_df),"important_note":"If recommended_threshold is null, event EV did not validate. Upload diagnostics."}
