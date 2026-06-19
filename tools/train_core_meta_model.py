from __future__ import annotations
import argparse, json
from src.meta_label import train_core_meta_model


def main():
    p = argparse.ArgumentParser(description="Train V24 meta-label model on verified core candidates.")
    p.add_argument("--dataset-folder", required=True)
    p.add_argument("--target-r", type=float, default=0.75)
    p.add_argument("--risk-dollars", type=float, default=100.0)
    p.add_argument("--max-trades-per-day", type=int, default=3)
    p.add_argument("--include-symbol-feature", action="store_true")
    args = p.parse_args()
    result = train_core_meta_model(args.dataset_folder, target_r=args.target_r, risk_dollars=args.risk_dollars, max_trades_per_day=args.max_trades_per_day, include_symbol_feature=args.include_symbol_feature)
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
