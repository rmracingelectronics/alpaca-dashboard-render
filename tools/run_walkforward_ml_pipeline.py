from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="One-command pipeline: build live-safe ML dataset, then run walk-forward ML ranker.")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-06-20")
    p.add_argument("--preset", default="v25_playbook")
    p.add_argument("--symbols", default="")
    p.add_argument("--feed", default="iex")
    p.add_argument("--session-mode", default="regular_only")
    p.add_argument("--direction", default="long_short")
    p.add_argument("--dataset-name", default="")
    p.add_argument("--name", default="walkforward_ml_ranker_pipeline")
    p.add_argument("--train-months", type=int, default=18)
    p.add_argument("--validation-months", type=int, default=3)
    p.add_argument("--test-months", type=int, default=2)
    p.add_argument("--step-months", type=int, default=2)
    p.add_argument("--lookahead-days", type=int, default=1)
    p.add_argument("--model-type", default="extra_trees_regressor")
    p.add_argument("--target", default="utility_r")
    p.add_argument("--top-trades-per-day", type=int, default=1)
    p.add_argument("--max-symbol-per-day", type=int, default=1)
    p.add_argument("--min-train-rows", type=int, default=250)
    p.add_argument("--thresholds", default="-0.10,-0.05,0,0.02,0.05,0.10,0.15,0.20,0.30")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_name = args.dataset_name or f"ml_ranker_{args.start}_{args.end}"
    build_cmd = [
        sys.executable, str(root / "tools" / "build_ml_ranker_dataset.py"),
        "--start", args.start, "--end", args.end, "--preset", args.preset, "--feed", args.feed,
        "--session-mode", args.session_mode, "--direction", args.direction,
        "--min-score", "0", "--candle-mode", "off", "--news-filter", "off", "--qqq-stress-filter", "off",
        "--use-news-proxy", "off", "--max-candidates-per-day", "999", "--max-symbol-candidates-per-day", "999",
        "--name", dataset_name,
    ]
    if args.symbols:
        build_cmd.extend(["--symbols", args.symbols])
    run(build_cmd)

    dataset_path = root / "data" / "ml_datasets" / dataset_name / "candidates.csv"
    train_cmd = [
        sys.executable, str(root / "tools" / "train_walkforward_ml_ranker.py"),
        "--dataset", str(dataset_path), "--train-months", str(args.train_months),
        "--validation-months", str(args.validation_months), "--test-months", str(args.test_months),
        "--step-months", str(args.step_months), "--lookahead-days", str(args.lookahead_days),
        "--model-type", args.model_type, "--target", args.target,
        "--top-trades-per-day", str(args.top_trades_per_day), "--max-symbol-per-day", str(args.max_symbol_per_day),
        "--min-train-rows", str(args.min_train_rows), "--thresholds", args.thresholds, "--name", args.name,
    ]
    run(train_cmd)


if __name__ == "__main__":
    main()
