from __future__ import annotations
import argparse, json
from src.meta_label import backtest_core_meta_strategy


def main():
    p = argparse.ArgumentParser(description="Backtest V24 meta-label strategy.")
    p.add_argument("--dataset-folder", required=True)
    p.add_argument("--model-folder", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--risk-dollars", type=float, default=100.0)
    p.add_argument("--max-trades-per-day", type=int, default=3)
    args = p.parse_args()
    result = backtest_core_meta_strategy(args.dataset_folder, args.model_folder, threshold=args.threshold, risk_dollars=args.risk_dollars, max_trades_per_day=args.max_trades_per_day)
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
