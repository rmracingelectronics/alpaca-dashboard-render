from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import build_research_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact uploadable research ZIP from local OHLCV bars.")
    parser.add_argument("--preset", default="edge_core_40")
    parser.add_argument("--symbols", default="", help="Optional comma-separated custom symbols. Overrides preset when supplied.")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-06-18")
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--adjustment", default="split")
    parser.add_argument("--direction", default="long_only", choices=["long_only", "long_short"])
    parser.add_argument("--min-score", type=float, default=45.0, help="Broad candidate score floor for research. Lower = more candidates/larger pack.")
    parser.add_argument("--no-preload", action="store_true", help="Do not fetch missing bars; use only local cache.")
    args = parser.parse_args()

    custom = args.symbols if args.symbols.strip() else None
    result = build_research_pack(
        preset=args.preset,
        start=args.start,
        end=args.end,
        feed=args.feed,
        adjustment=args.adjustment,
        direction_mode=args.direction,
        min_score=args.min_score,
        custom_symbols=custom,
        preload_missing=not args.no_preload,
    )
    print(json.dumps(result, indent=2, default=str))
    print("\nUpload this ZIP here:")
    print(result["zip_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
