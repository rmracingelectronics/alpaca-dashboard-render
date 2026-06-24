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
    run_walkforward_tuning,
    save_json,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Rolling walk-forward strategy tuner: tune on past, validate, test next unseen window.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--train-days", type=int, default=504)
    p.add_argument("--validate-days", type=int, default=126)
    p.add_argument("--test-days", type=int, default=63)
    p.add_argument("--lookahead-days", type=int, default=1)
    p.add_argument("--trials-per-window", type=int, default=350)
    p.add_argument("--top-train-keep", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed-risk", type=float, default=100.0)
    p.add_argument("--name", default="walkforward_tuning_run")
    args = p.parse_args()

    outdir = ROOT / "data" / "tuning_runs" / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    df = prepare_candidate_frame(pd.read_csv(args.dataset))
    res = run_walkforward_tuning(
        df,
        train_days=args.train_days,
        validate_days=args.validate_days,
        test_days=args.test_days,
        lookahead_days=args.lookahead_days,
        start=args.start,
        end=args.end,
        n_random=args.trials_per_window,
        top_train_keep=args.top_train_keep,
        seed=args.seed,
        fixed_risk=args.fixed_risk,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    res["window_summary"].to_csv(outdir / "walkforward_window_summary.csv", index=False)
    res["chosen_configs"].to_csv(outdir / "chosen_configs.csv", index=False)
    res["selected_trades"].to_csv(outdir / "walkforward_selected_trades.csv", index=False)
    res["selected_trades"].to_csv(outdir / "walkforward_trade_decision_report.csv", index=False)
    pd.DataFrame([res["overall"]]).to_csv(outdir / "walkforward_overall_summary.csv", index=False)
    save_json(outdir / "manifest.json", {
        "dataset": args.dataset,
        "start": args.start,
        "end": args.end,
        "train_days": args.train_days,
        "validate_days": args.validate_days,
        "test_days": args.test_days,
        "lookahead_days": args.lookahead_days,
        "trials_per_window": args.trials_per_window,
        "top_train_keep": args.top_train_keep,
        "seed": args.seed,
        "fixed_risk": args.fixed_risk,
        "windows": res["windows"],
    })
    boot = bootstrap_metrics(res["selected_trades"], n=1000, seed=args.seed)
    if not boot.empty:
        boot.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_csv(outdir / "walkforward_bootstrap_summary.csv")
    print(f"Output folder: {outdir}")
    print("Walk-forward overall:")
    print(pd.DataFrame([res["overall"]]).to_string(index=False))
    if not res["window_summary"].empty:
        print("\nWindow summary:")
        print(res["window_summary"][["window", "train_start", "train_end", "validate_start", "validate_end", "test_start", "test_end", "validate_total_r", "validate_trades", "test_total_r", "test_trades", "test_profit_factor"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
