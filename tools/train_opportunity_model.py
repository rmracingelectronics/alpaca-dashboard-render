from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_research import train_opportunity_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML model on first-touch opportunity dataset.")
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--target-r", type=float, default=0.75)
    parser.add_argument("--model-type", default="extra_trees", choices=["extra_trees", "random_forest"])
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--max-eval-rows-per-split", type=int, default=150_000)
    parser.add_argument("--max-trades-per-day", type=int, default=3)
    parser.add_argument("--risk-dollars", type=float, default=100.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    result = train_opportunity_model(
        dataset_folder=args.dataset_folder,
        target_r=args.target_r,
        model_type=args.model_type,
        max_train_rows=args.max_train_rows,
        max_eval_rows_per_split=args.max_eval_rows_per_split,
        random_state=args.random_state,
        max_trades_per_day=args.max_trades_per_day,
        risk_dollars=args.risk_dollars,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
