from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import preload_research_data, local_cache_status, get_symbols_for_preset
from src.symbols import MARKET_SYMBOLS


def main() -> int:
    parser = argparse.ArgumentParser(description="Download OHLCV bars once into the compressed local cache.")
    parser.add_argument("--preset", default="edge_core_40")
    parser.add_argument("--symbols", default="", help="Optional comma-separated custom symbols. Overrides preset when supplied.")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-06-18")
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--adjustment", default="split")
    parser.add_argument("--status-only", action="store_true", help="Do not download; only print local cache coverage.")
    args = parser.parse_args()

    custom = args.symbols if args.symbols.strip() else None
    symbols = get_symbols_for_preset(args.preset, custom)
    all_symbols = list(dict.fromkeys(MARKET_SYMBOLS + symbols))

    if args.status_only:
        result = local_cache_status(all_symbols, args.start, args.end, args.feed, args.adjustment)
    else:
        result = preload_research_data(args.preset, args.start, args.end, args.feed, args.adjustment, custom)

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
