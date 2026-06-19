from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_research import build_first_touch_ml_dataset, train_opportunity_model, backtest_learned_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full V20 first-touch ML pipeline.")
    parser.add_argument("--preset", default="edge_core_40")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-06-18")
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--target-r", type=float, default=0.75)
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--max-eval-rows-per-split", type=int, default=150_000)
    parser.add_argument("--max-trades-per-day", type=int, default=3)
    parser.add_argument("--risk-dollars", type=float, default=100.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--preload-missing", action="store_true")
    args = parser.parse_args()

    dataset = build_first_touch_ml_dataset(
        preset=args.preset,
        start=args.start,
        end=args.end,
        feed=args.feed,
        preload_missing=args.preload_missing,
        horizon_bars=args.horizon_bars,
        target_values="0.5,0.75,1.0,1.5,2.0",
    )
    model = train_opportunity_model(
        dataset_folder=dataset["dataset_folder"],
        target_r=args.target_r,
        max_train_rows=args.max_train_rows,
        max_eval_rows_per_split=args.max_eval_rows_per_split,
        max_trades_per_day=args.max_trades_per_day,
        risk_dollars=args.risk_dollars,
    )
    report = backtest_learned_strategy(
        dataset_folder=dataset["dataset_folder"],
        model_folder=model["model_folder"],
        threshold=args.threshold,
        risk_dollars=args.risk_dollars,
        max_trades_per_day=args.max_trades_per_day,
    )
    print(json.dumps({"dataset": dataset, "model": model, "backtest": report}, indent=2, default=str))

if __name__ == "__main__":
    main()
