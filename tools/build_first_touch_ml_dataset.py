from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_research import build_first_touch_ml_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build first-touch ML opportunity dataset from local bars.")
    parser.add_argument("--preset", default="edge_core_40")
    parser.add_argument("--symbols", default=None, help="Optional comma-separated symbols instead of preset.")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-06-18")
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--adjustment", default="split")
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument("--start-time", default="09:40")
    parser.add_argument("--end-time", default="14:30")
    parser.add_argument("--risk-atr-mult", type=float, default=0.75)
    parser.add_argument("--min-risk-pct", type=float, default=0.003)
    parser.add_argument("--min-avg-20d-dollar-volume", type=float, default=20_000_000.0)
    parser.add_argument("--min-5m-dollar-volume", type=float, default=50_000.0)
    parser.add_argument("--target-r-values", default="0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--split-train-end", default="2024-12-31")
    parser.add_argument("--split-validate-end", default="2025-12-31")
    parser.add_argument("--direction", default="long_short", choices=["long_only", "short_only", "long_short"])
    parser.add_argument("--preload-missing", action="store_true", help="Fetch missing bars from Alpaca if not already cached.")
    args = parser.parse_args()

    result = build_first_touch_ml_dataset(
        preset=args.preset,
        custom_symbols=args.symbols,
        start=args.start,
        end=args.end,
        feed=args.feed,
        adjustment=args.adjustment,
        preload_missing=args.preload_missing,
        horizon_bars=args.horizon_bars,
        start_time=args.start_time,
        end_time=args.end_time,
        risk_atr_mult=args.risk_atr_mult,
        min_risk_pct=args.min_risk_pct,
        min_avg_20d_dollar_volume=args.min_avg_20d_dollar_volume,
        min_5m_dollar_volume=args.min_5m_dollar_volume,
        target_values=args.target_r_values,
        split_train_end=args.split_train_end,
        split_validate_end=args.split_validate_end,
        direction_mode=args.direction,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
