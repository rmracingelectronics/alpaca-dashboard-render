from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alpaca_rest import AlpacaDataClient, pad_start_for_indicators
from src.symbols import WATCHLISTS, parse_symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="Preload Alpaca OHLCV bars into the local disk cache.")
    parser.add_argument("--preset", default="edge_core_40", choices=sorted(WATCHLISTS.keys()))
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbols. Overrides preset if supplied.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--timeframes", default="5Min,1Day")
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols, preset=args.preset)
    symbols = list(dict.fromkeys(["QQQ"] + [s.upper() for s in symbols]))
    fetch_start = pad_start_for_indicators(args.start, days=55)
    fetch_end = args.end
    client = AlpacaDataClient()

    print(f"Preloading {len(symbols)} symbols from {fetch_start} to {fetch_end} feed={args.feed}")
    for timeframe in [x.strip() for x in args.timeframes.split(",") if x.strip()]:
        status = client.prefetch_stock_bars(symbols, timeframe, fetch_start, fetch_end, feed=args.feed, adjustment="split", use_cache=True)
        print(timeframe, status)
    print("Done. Later backtests over overlapping periods should read from data/local_bars.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
