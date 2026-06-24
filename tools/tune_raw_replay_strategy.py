from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy_tuner import (  # noqa: E402
    bootstrap_metrics,
    prepare_candidate_frame,
    save_json,
    tune_train_validate_holdout,
)


def main() -> int:
    p = argparse.ArgumentParser(description="No-leak train/validate/holdout tuner for raw-bar live-replay candidate datasets.")
    p.add_argument("--dataset", required=True, help="Path to candidates.csv built from raw-bar live replay.")
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--validate-start", required=True)
    p.add_argument("--validate-end", required=True)
    p.add_argument("--holdout-start")
    p.add_argument("--holdout-end")
    p.add_argument("--trials", type=int, default=750, help="Number of random parameter configs to try, in addition to seed configs.")
    p.add_argument("--top-train-keep", type=int, default=60, help="Only this many best train configs are evaluated on validation/holdout.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed-risk", type=float, default=100.0)
    p.add_argument("--name", default="raw_tuner_run")
    args = p.parse_args()

    dataset = Path(args.dataset)
    outdir = ROOT / "data" / "tuning_runs" / args.name
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset)
    df = prepare_candidate_frame(df)
    res = tune_train_validate_holdout(
        df,
        train_start=args.train_start,
        train_end=args.train_end,
        validate_start=args.validate_start,
        validate_end=args.validate_end,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        n_random=args.trials,
        top_train_keep=args.top_train_keep,
        seed=args.seed,
        fixed_risk=args.fixed_risk,
    )

    res["train_trials"].to_csv(outdir / "all_train_trials.csv", index=False)
    res["validation_results"].to_csv(outdir / "validation_results.csv", index=False)
    cfg = res["best_config"]
    bundle = res["best_bundle"]
    if cfg is None:
        print("No valid config found.")
        return 2
    save_json(outdir / "best_config.json", cfg.to_dict())
    save_json(outdir / "manifest.json", {"dataset": str(dataset), "periods": res["periods"], "trials": args.trials, "top_train_keep": args.top_train_keep, "seed": args.seed, "fixed_risk": args.fixed_risk})
    for name in ["train", "validate", "holdout"]:
        trades = bundle.get(f"{name}_trades")
        metrics = bundle.get(f"{name}_metrics")
        if trades is not None:
            trades.to_csv(outdir / f"best_{name}_selected_trades.csv", index=False)
            trades.to_csv(outdir / f"best_{name}_trade_decision_report.csv", index=False)
        if metrics is not None:
            pd.DataFrame([metrics]).to_csv(outdir / f"best_{name}_summary.csv", index=False)
    hold = bundle.get("holdout_trades")
    if hold is not None and not hold.empty:
        boot = bootstrap_metrics(hold, n=1000, seed=args.seed)
        if not boot.empty:
            boot.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_csv(outdir / "holdout_bootstrap_summary.csv")
    print(f"Output folder: {outdir}")
    print("Best config:")
    print(pd.Series(cfg.to_dict()).to_string())
    print("\nValidation metrics:")
    print(pd.Series(bundle["validate_metrics"]).to_string())
    if args.holdout_start or args.holdout_end:
        print("\nHoldout metrics:")
        print(pd.Series(bundle["holdout_metrics"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
