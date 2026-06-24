from __future__ import annotations

import argparse
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


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).replace(";", ",").split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(float(x.strip())) for x in str(raw).replace(";", ",").split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Scan Q-policy thresholds/min-state-counts on a validation/test period.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--edges", default="-0.05,0,0.02,0.05,0.1,0.15,0.2,0.3")
    p.add_argument("--min-state-counts", default="1,3,5,8,12,20")
    p.add_argument("--top-trades", default="1,2,3")
    p.add_argument("--fixed-risk", type=float, default=100.0)
    p.add_argument("--name", default="")
    args = p.parse_args()
    df = pd.read_csv(args.dataset)
    if "session_date" in df.columns:
        dates = pd.to_datetime(df["session_date"], errors="coerce")
        if args.start:
            df = df[dates >= pd.Timestamp(args.start)].copy(); dates = pd.to_datetime(df["session_date"], errors="coerce")
        if args.end:
            df = df[dates <= pd.Timestamp(args.end)].copy()
    model = load_q_model(args.model)
    rows = []
    best_selected = pd.DataFrame()
    best_key = None
    best_score = -999999.0
    for edge in _parse_float_list(args.edges):
        for msc in _parse_int_list(args.min_state_counts):
            for topn in _parse_int_list(args.top_trades):
                selected, summary, reviewed = backtest_q_policy(df, model, top_trades_per_day=topn, max_symbol_per_day=1, min_edge=edge, min_state_count=msc, fixed_risk_dollars=float(args.fixed_risk))
                summary.update({"min_edge": edge, "min_state_count": msc, "top_trades_per_day": topn})
                rows.append(summary)
                # Prefer profit but penalize too few trades and drawdown.
                score = float(summary.get("total_r", 0.0)) + 0.05 * float(summary.get("trades", 0)) + 0.2 * float(summary.get("max_drawdown_r", 0.0))
                if score > best_score:
                    best_score = score; best_key = (edge, msc, topn); best_selected = selected.copy()
    out = pd.DataFrame(rows).sort_values(["total_r", "profit_factor", "trades"], ascending=[False, False, False])
    name = args.name or f"q_policy_threshold_scan_{_dt_stamp()}"
    out_dir = ML_BACKTESTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "threshold_scan.csv", index=False)
    best_selected.to_csv(out_dir / "best_selected_trades.csv", index=False)
    print(f"Output folder: {out_dir}")
    print(f"Best key by score: {best_key}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
