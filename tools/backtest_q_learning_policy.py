from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import pandas as pd

from src.config import ML_BACKTESTS_DIR
from src.q_learning_policy import backtest_q_policy, load_q_model


def _dt_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest a trained Q-learning trade/skip policy on a candidate CSV.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, help="Path to model.json")
    p.add_argument("--start", default="", help="Optional session start date filter.")
    p.add_argument("--end", default="", help="Optional session end date filter.")
    p.add_argument("--top-trades-per-day", type=int, default=1)
    p.add_argument("--max-symbol-per-day", type=int, default=1)
    p.add_argument("--min-edge", type=float, default=None)
    p.add_argument("--min-state-count", type=int, default=None)
    p.add_argument("--fixed-risk", type=float, default=100.0)
    p.add_argument("--name", default="")
    args = p.parse_args()

    df = pd.read_csv(args.dataset)
    if "session_date" in df.columns:
        dates = pd.to_datetime(df["session_date"], errors="coerce")
        if args.start:
            df = df[dates >= pd.Timestamp(args.start)].copy()
            dates = pd.to_datetime(df["session_date"], errors="coerce")
        if args.end:
            df = df[dates <= pd.Timestamp(args.end)].copy()
    model = load_q_model(args.model)
    selected, summary, reviewed = backtest_q_policy(
        df, model,
        top_trades_per_day=int(args.top_trades_per_day),
        max_symbol_per_day=int(args.max_symbol_per_day),
        min_edge=args.min_edge,
        min_state_count=args.min_state_count,
        fixed_risk_dollars=float(args.fixed_risk),
    )
    name = args.name or f"q_policy_backtest_{_dt_stamp()}"
    out_dir = ML_BACKTESTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "selected_trades.csv", index=False)
    reviewed.to_csv(out_dir / "reviewed_candidates.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({"args": vars(args), "summary": summary}, indent=2, default=str), encoding="utf-8")
    print(f"Backtest output folder: {out_dir}")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
