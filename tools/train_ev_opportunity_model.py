from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_research import train_ev_opportunity_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train V21 expected-value opportunity model using natural first-touch rows.")
    ap.add_argument("--dataset-folder", required=True)
    ap.add_argument("--target-r", type=float, default=0.75)
    ap.add_argument("--model-type", default="extra_trees_regressor", choices=["extra_trees_regressor", "random_forest_regressor"])
    ap.add_argument("--max-train-rows", type=int, default=500000)
    ap.add_argument("--max-eval-rows-per-split", type=int, default=250000)
    ap.add_argument("--max-trades-per-day", type=int, default=3)
    ap.add_argument("--risk-dollars", type=float, default=100.0)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--include-symbol-feature", action="store_true", help="Allow symbol one-hot features. Off by default to reduce overfit.")
    args = ap.parse_args()
    result = train_ev_opportunity_model(**vars(args))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
