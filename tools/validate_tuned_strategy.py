from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy_tuner import (  # noqa: E402
    bootstrap_metrics,
    evaluate_config,
    load_config,
    prepare_candidate_frame,
    save_json,
    split_by_date,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate a tuned config on any date range without changing it.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True, help="Path to best_config.json")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--fixed-risk", type=float, default=100.0)
    p.add_argument("--name", default="validate_tuned_strategy")
    args = p.parse_args()

    outdir = ROOT / "data" / "tuning_runs" / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    df = prepare_candidate_frame(pd.read_csv(args.dataset))
    cfg = load_config(args.config)
    part = split_by_date(df, args.start, args.end)
    metrics, trades = evaluate_config(part, cfg, fixed_risk=args.fixed_risk)
    trades.to_csv(outdir / "selected_trades.csv", index=False)
    trades.to_csv(outdir / "trade_decision_report.csv", index=False)
    pd.DataFrame([metrics]).to_csv(outdir / "summary.csv", index=False)
    boot = bootstrap_metrics(trades, n=1000)
    if not boot.empty:
        boot.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_csv(outdir / "bootstrap_summary.csv")
    save_json(outdir / "manifest.json", {"dataset": args.dataset, "config": args.config, "start": args.start, "end": args.end, "fixed_risk": args.fixed_risk})
    print(f"Output folder: {outdir}")
    print(pd.DataFrame([metrics]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
