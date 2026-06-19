from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import build_opportunity_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a strategy-independent intraday opportunity discovery dataset.")
    parser.add_argument("--preset", default="edge_core_40")
    parser.add_argument("--symbols", default=None, help="Optional comma-separated symbols overriding preset.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--adjustment", default="split")
    parser.add_argument("--horizon-bars", type=int, default=12, help="Future 5-minute bars used to label opportunity outcome. 12 = about 1 hour.")
    parser.add_argument("--start-time", default="09:40")
    parser.add_argument("--end-time", default="14:30")
    parser.add_argument("--min-avg-20d-dollar-volume", type=float, default=20_000_000)
    parser.add_argument("--min-5m-dollar-volume", type=float, default=50_000)
    parser.add_argument("--max-upload-rows", type=int, default=200_000, help="Maximum sampled rows in upload file to respect upload size limits.")
    parser.add_argument("--split-train-end", default="2024-12-31", help="Last date included in discovery/training split.")
    parser.add_argument("--split-validate-end", default="2025-12-31", help="Last date included in validation split; later rows are final test.")
    parser.add_argument("--no-preload", action="store_true", help="Do not fetch missing data; use only local cache.")
    args = parser.parse_args()

    result = build_opportunity_dataset(
        preset=args.preset,
        custom_symbols=args.symbols,
        start=args.start,
        end=args.end,
        feed=args.feed,
        adjustment=args.adjustment,
        preload_missing=not args.no_preload,
        horizon_bars=args.horizon_bars,
        start_time=args.start_time,
        end_time=args.end_time,
        min_avg_20d_dollar_volume=args.min_avg_20d_dollar_volume,
        min_5m_dollar_volume=args.min_5m_dollar_volume,
        max_upload_rows=args.max_upload_rows,
        split_train_end=args.split_train_end,
        split_validate_end=args.split_validate_end,
    )
    print(json.dumps(result, indent=2))
    print("\nUpload this ZIP here:")
    print(result["zip_path"])


if __name__ == "__main__":
    main()
