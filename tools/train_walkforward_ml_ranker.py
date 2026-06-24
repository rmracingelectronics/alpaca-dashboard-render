from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.config import ML_BACKTESTS_DIR, ML_MODELS_DIR
from src.ml_ranker_policy import (
    MLRankerConfig,
    add_live_safe_features,
    bootstrap_summary,
    live_style_select_by_score,
    save_ranker_model,
    scan_thresholds,
    score_candidates,
    summarize_trades,
    train_ranker_model,
)


def _dt_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def _month_starts(start: pd.Timestamp, end: pd.Timestamp, step_months: int) -> list[pd.Timestamp]:
    cur = pd.Timestamp(start).normalize().replace(day=1)
    out = []
    while cur <= end:
        out.append(cur)
        cur = cur + pd.DateOffset(months=int(step_months))
    return out


def _parse_thresholds(raw: str) -> list[float]:
    vals = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals or [-0.10, -0.05, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def _prepare_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Dataset is empty: {path}")
    if "session_date" not in df.columns:
        raise RuntimeError("Dataset must contain session_date. Rebuild with build_ml_ranker_dataset.py.")
    if "r_multiple" not in df.columns:
        raise RuntimeError("Dataset must contain r_multiple outcomes.")
    df = add_live_safe_features(df)
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.date.astype(str)
    df = df[pd.to_datetime(df["session_date"], errors="coerce").notna()].copy()
    return df.sort_values(["session_date", "_sort_ts", "symbol"]).reset_index(drop=True)


def _window_summary_row(window_id: int, phase: str, selected: pd.DataFrame, reviewed: pd.DataFrame, extra: dict) -> dict:
    row = summarize_trades(selected, fixed_risk_dollars=float(extra.get("fixed_risk", 100.0)))
    row.update({
        "window_id": int(window_id),
        "phase": str(phase),
        "reviewed_candidates": int(len(reviewed)) if reviewed is not None else 0,
        "approved_candidates": int(len(selected)) if selected is not None else 0,
    })
    row.update(extra)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Train/test a PyBroker-style walk-forward ML ranker on live-safe candidate rows.")
    p.add_argument("--dataset", required=True, help="Path to candidates.csv from build_ml_ranker_dataset.py or V37 dataset builder.")
    p.add_argument("--start", default="", help="Optional first test date. Defaults to dataset start plus train window.")
    p.add_argument("--end", default="", help="Optional last test date. Defaults to dataset end.")
    p.add_argument("--train-months", type=int, default=18)
    p.add_argument("--validation-months", type=int, default=3)
    p.add_argument("--test-months", type=int, default=2)
    p.add_argument("--step-months", type=int, default=2)
    p.add_argument("--lookahead-days", type=int, default=1, help="Gap between train/validation/test to prevent leakage.")
    p.add_argument("--model-type", default="extra_trees_regressor", choices=["extra_trees_regressor", "random_forest_regressor", "hist_gbdt_regressor", "extra_trees_classifier", "random_forest_classifier"])
    p.add_argument("--target", default="utility_r", choices=["utility_r", "r_multiple", "win"])
    p.add_argument("--kappa", type=float, default=0.25)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-samples-leaf", type=int, default=8)
    p.add_argument("--min-train-rows", type=int, default=250)
    p.add_argument("--thresholds", default="-0.10,-0.05,0,0.02,0.05,0.10,0.15,0.20,0.30")
    p.add_argument("--top-trades-per-day", type=int, default=1)
    p.add_argument("--max-symbol-per-day", type=int, default=1)
    p.add_argument("--fixed-risk", type=float, default=100.0)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--name", default="")
    args = p.parse_args()

    df = _prepare_dataset(Path(args.dataset))
    dates = pd.to_datetime(df["session_date"], errors="coerce")
    data_start = dates.min().normalize()
    data_end = dates.max().normalize()
    test_start_default = data_start + pd.DateOffset(months=int(args.train_months) + int(args.validation_months)) + pd.Timedelta(days=int(args.lookahead_days))
    test_start = pd.Timestamp(args.start).normalize() if args.start else test_start_default
    test_end_limit = pd.Timestamp(args.end).normalize() if args.end else data_end
    thresholds = _parse_thresholds(args.thresholds)

    name = args.name or f"walkforward_ml_ranker_{Path(args.dataset).parent.name}_{_dt_stamp()}"
    out_dir = ML_BACKTESTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_root = ML_MODELS_DIR / f"{name}_models"
    model_root.mkdir(parents=True, exist_ok=True)

    cfg = MLRankerConfig(
        model_type=args.model_type,
        target=args.target,
        kappa=float(args.kappa),
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        random_seed=int(args.random_seed),
        min_train_rows=int(args.min_train_rows),
        top_trades_per_day=int(args.top_trades_per_day),
        max_symbol_per_day=int(args.max_symbol_per_day),
        bootstrap_samples=int(args.bootstrap_samples),
    )

    all_selected = []
    all_reviewed = []
    window_rows = []
    scan_rows = []
    feature_rows = []
    windows = _month_starts(test_start, test_end_limit, int(args.step_months))
    window_id = 0
    for ts0 in windows:
        test_start_w = pd.Timestamp(ts0).normalize()
        test_end_w = min(test_start_w + pd.DateOffset(months=int(args.test_months)) - pd.Timedelta(days=1), test_end_limit)
        val_end = test_start_w - pd.Timedelta(days=int(args.lookahead_days) + 1)
        val_start = val_end - pd.DateOffset(months=int(args.validation_months)) + pd.Timedelta(days=1)
        train_end = val_start - pd.Timedelta(days=int(args.lookahead_days) + 1)
        train_start = train_end - pd.DateOffset(months=int(args.train_months)) + pd.Timedelta(days=1)
        if test_start_w > test_end_limit:
            break
        mask_train_core = (dates >= train_start) & (dates <= train_end)
        mask_val = (dates >= val_start) & (dates <= val_end)
        mask_train_full = (dates >= train_start) & (dates <= val_end)
        mask_test = (dates >= test_start_w) & (dates <= test_end_w)
        train_core = df[mask_train_core].copy()
        val = df[mask_val].copy()
        train_full = df[mask_train_full].copy()
        test = df[mask_test].copy()
        if len(train_core) < int(args.min_train_rows) or val.empty or test.empty:
            continue
        window_id += 1
        try:
            model_val = train_ranker_model(train_core, cfg)
            val_scored = score_candidates(val, model_val)
            val_scan = scan_thresholds(val_scored, thresholds=thresholds, top_trades_per_day=int(args.top_trades_per_day), max_symbol_per_day=int(args.max_symbol_per_day), fixed_risk_dollars=float(args.fixed_risk))
            # Require positive validation R if possible. If none positive, use best objective but mark it clearly.
            val_positive = val_scan[val_scan["total_r"] > 0].copy()
            best = (val_positive if not val_positive.empty else val_scan).iloc[0].to_dict()
            threshold = float(best["threshold"])
            model_test = train_ranker_model(train_full, cfg)
            model_folder = model_root / f"window_{window_id:03d}"
            save_ranker_model(model_test, model_folder)
            reviewed_test = score_candidates(test, model_test)
            selected_test = live_style_select_by_score(reviewed_test, threshold=threshold, top_trades_per_day=int(args.top_trades_per_day), max_symbol_per_day=int(args.max_symbol_per_day))
            reviewed_test["window_id"] = window_id
            reviewed_test["selected_by_ml"] = False
            if not selected_test.empty:
                key_cols = [c for c in ["symbol", "timestamp", "entry_time", "session_date"] if c in reviewed_test.columns and c in selected_test.columns]
                if key_cols:
                    selected_keys = set(map(tuple, selected_test[key_cols].astype(str).to_numpy()))
                    reviewed_test["selected_by_ml"] = [tuple(x) in selected_keys for x in reviewed_test[key_cols].astype(str).to_numpy()]
            selected_test["window_id"] = window_id
            selected_test["ml_threshold"] = threshold
            selected_test["ml_model_folder"] = str(model_folder)
            if not selected_test.empty:
                all_selected.append(selected_test)
            all_reviewed.append(reviewed_test)
            extra = {
                "fixed_risk": float(args.fixed_risk),
                "train_start": str(train_start.date()), "train_end": str(train_end.date()),
                "validation_start": str(val_start.date()), "validation_end": str(val_end.date()),
                "test_start": str(test_start_w.date()), "test_end": str(test_end_w.date()),
                "train_rows": int(len(train_core)), "validation_rows": int(len(val)), "test_rows": int(len(test)),
                "chosen_threshold": threshold, "validation_total_r_at_threshold": float(best.get("total_r", 0.0)),
                "validation_positive_threshold_found": bool(not val_positive.empty),
                "model_folder": str(model_folder),
            }
            row = _window_summary_row(window_id, "test", selected_test, reviewed_test, extra)
            row.update(bootstrap_summary(selected_test, samples=int(args.bootstrap_samples), seed=int(args.random_seed) + window_id))
            window_rows.append(row)
            val_scan["window_id"] = window_id
            val_scan["phase"] = "validation_threshold_scan"
            val_scan["train_start"] = str(train_start.date())
            val_scan["validation_start"] = str(val_start.date())
            val_scan["validation_end"] = str(val_end.date())
            scan_rows.append(val_scan)
            fi = pd.DataFrame(model_test.get("feature_importance", []))
            if not fi.empty:
                fi["window_id"] = window_id
                feature_rows.append(fi)
        except Exception as exc:
            window_rows.append({
                "window_id": window_id + 1,
                "phase": "error",
                "error": str(exc),
                "train_start": str(train_start.date()), "train_end": str(train_end.date()),
                "validation_start": str(val_start.date()), "validation_end": str(val_end.date()),
                "test_start": str(test_start_w.date()), "test_end": str(test_end_w.date()),
            })
            continue

    selected = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    reviewed = pd.concat(all_reviewed, ignore_index=True) if all_reviewed else pd.DataFrame()
    window_summary = pd.DataFrame(window_rows)
    threshold_scan = pd.concat(scan_rows, ignore_index=True) if scan_rows else pd.DataFrame()
    feature_importance = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()

    selected.to_csv(out_dir / "selected_trades.csv", index=False)
    reviewed.to_csv(out_dir / "reviewed_candidates.csv", index=False)
    window_summary.to_csv(out_dir / "window_summary.csv", index=False)
    threshold_scan.to_csv(out_dir / "threshold_scan.csv", index=False)
    if not feature_importance.empty:
        fi_summary = feature_importance.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
        feature_importance.to_csv(out_dir / "feature_importance_by_window.csv", index=False)
        fi_summary.to_csv(out_dir / "feature_importance.csv", index=False)
    overall = summarize_trades(selected, fixed_risk_dollars=float(args.fixed_risk))
    overall.update(bootstrap_summary(selected, samples=int(args.bootstrap_samples), seed=int(args.random_seed)))
    overall.update({
        "dataset": str(Path(args.dataset).resolve()),
        "windows_run": int((window_summary["phase"] == "test").sum()) if not window_summary.empty and "phase" in window_summary.columns else 0,
        "model_type": str(args.model_type),
        "target": str(args.target),
        "train_months": int(args.train_months),
        "validation_months": int(args.validation_months),
        "test_months": int(args.test_months),
        "lookahead_days": int(args.lookahead_days),
        "top_trades_per_day": int(args.top_trades_per_day),
        "max_symbol_per_day": int(args.max_symbol_per_day),
    })
    pd.DataFrame([overall]).to_csv(out_dir / "walkforward_summary.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({"args": vars(args), "config": cfg.__dict__, "overall": overall, "output_dir": str(out_dir)}, indent=2, default=str), encoding="utf-8")

    # Final model trained on the whole dataset for optional live-paper use after validation.
    final_model = train_ranker_model(df, cfg)
    final_model_dir = model_root / "final_model_all_data"
    final_model_path = save_ranker_model(final_model, final_model_dir)
    print(f"Walk-forward output: {out_dir}")
    print(f"Final model: {final_model_path}")
    print(pd.DataFrame([overall]).to_string(index=False))


if __name__ == "__main__":
    main()
