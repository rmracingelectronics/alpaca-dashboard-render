from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import get_symbols_for_preset, local_cache_status
from src.symbols import MARKET_SYMBOLS


def main() -> int:
    parser = argparse.ArgumentParser(description="Print local OHLCV cache coverage.")
    parser.add_argument("--preset", default="edge_core_40")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-06-18")
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--adjustment", default="split")
    parser.add_argument("--csv", default="", help="Optional CSV output path for detailed coverage rows.")
    args = parser.parse_args()
    custom = args.symbols if args.symbols.strip() else None
    symbols = list(dict.fromkeys(MARKET_SYMBOLS + get_symbols_for_preset(args.preset, custom)))
    result = local_cache_status(symbols, args.start, args.end, args.feed, args.adjustment)
    print(json.dumps(result["summary"], indent=2, default=str))
    if args.csv:
        import pandas as pd
        pd.DataFrame(result["rows"]).to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
