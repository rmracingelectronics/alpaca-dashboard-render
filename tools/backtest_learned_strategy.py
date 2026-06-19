from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_research import backtest_learned_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest learned ML strategy against full first-touch dataset.")
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--model-folder", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-r", type=float, default=None)
    parser.add_argument("--risk-dollars", type=float, default=100.0)
    parser.add_argument("--max-trades-per-day", type=int, default=3)
    parser.add_argument("--max-per-symbol-day", type=int, default=1)
    args = parser.parse_args()
    result = backtest_learned_strategy(
        dataset_folder=args.dataset_folder,
        model_folder=args.model_folder,
        threshold=args.threshold,
        target_r=args.target_r,
        risk_dollars=args.risk_dollars,
        max_trades_per_day=args.max_trades_per_day,
        max_per_symbol_day=args.max_per_symbol_day,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
