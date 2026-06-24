from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.config import ML_BACKTESTS_DIR
from src.ml_ranker_policy import (
    add_live_safe_features,
    bootstrap_summary,
    live_style_select_by_score,
    load_ranker_model,
    score_candidates,
    summarize_trades,
)


def _dt_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest a saved walk-forward ML ranker final model on a candidate dataset/period.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, help="Path to model.json or model folder from train_walkforward_ml_ranker.py")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--top-trades-per-day", type=int, default=1)
    p.add_argument("--max-symbol-per-day", type=int, default=1)
    p.add_argument("--fixed-risk", type=float, default=100.0)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--name", default="")
    args = p.parse_args()

    df = pd.read_csv(args.dataset)
    df = add_live_safe_features(df)
    if "session_date" not in df.columns:
        raise RuntimeError("Dataset must contain session_date.")
    d = pd.to_datetime(df["session_date"], errors="coerce")
    if args.start:
        df = df[d >= pd.Timestamp(args.start)].copy()
        d = pd.to_datetime(df["session_date"], errors="coerce")
    if args.end:
        df = df[d <= pd.Timestamp(args.end)].copy()
    model = load_ranker_model(args.model)
    reviewed = score_candidates(df, model)
    selected = live_style_select_by_score(reviewed, threshold=float(args.threshold), top_trades_per_day=int(args.top_trades_per_day), max_symbol_per_day=int(args.max_symbol_per_day))
    if not selected.empty:
        selected["ml_threshold"] = float(args.threshold)
    summary = summarize_trades(selected, fixed_risk_dollars=float(args.fixed_risk))
    summary.update(bootstrap_summary(selected, samples=int(args.bootstrap_samples)))
    summary.update({
        "dataset": str(Path(args.dataset).resolve()),
        "model": str(Path(args.model).resolve()),
        "start": str(args.start), "end": str(args.end),
        "threshold": float(args.threshold),
        "top_trades_per_day": int(args.top_trades_per_day),
        "max_symbol_per_day": int(args.max_symbol_per_day),
        "reviewed_candidates": int(len(reviewed)),
        "approved_candidates": int(len(selected)),
    })
    name = args.name or f"ml_ranker_backtest_{_dt_stamp()}"
    out_dir = ML_BACKTESTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(out_dir / "reviewed_candidates.csv", index=False)
    selected.to_csv(out_dir / "selected_trades.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({"args": vars(args), "summary": summary}, indent=2, default=str), encoding="utf-8")
    print(f"Backtest output: {out_dir}")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
