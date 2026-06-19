from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_research import backtest_ev_strategy


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest V21 expected-value learned strategy on full local ML dataset.")
    ap.add_argument("--dataset-folder", required=True)
    ap.add_argument("--model-folder", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--target-r", type=float, default=None)
    ap.add_argument("--risk-dollars", type=float, default=100.0)
    ap.add_argument("--max-trades-per-day", type=int, default=3)
    ap.add_argument("--max-per-symbol-day", type=int, default=1)
    args = ap.parse_args()
    result = backtest_ev_strategy(**vars(args))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
