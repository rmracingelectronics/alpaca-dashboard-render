"""Quick verification that fixed-dollar risk changes the trade result.
Run from the project folder:
    python tools/risk_sanity_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import StrategyParams
from src.strategy import apply_portfolio_rules

candidate = {
    "entry_time": pd.Timestamp("2026-01-02 15:00:00+00:00"),
    "exit_time": pd.Timestamp("2026-01-02 16:00:00+00:00"),
    "session_date": pd.Timestamp("2026-01-02").date(),
    "candidate_score": 90,
    "entry_price": 100.0,
    "risk_per_share": 2.0,
    "pnl_per_share": 1.0,
    "r_multiple": 0.5,
    "symbol": "TEST",
}

for risk in [25, 100, 200, 500]:
    params = StrategyParams(
        initial_account_value=10000,
        risk_per_trade_dollars=risk,
        risk_per_trade_pct=risk / 10000,
        requested_risk_percent=(risk / 10000) * 100,
    )
    out = apply_portfolio_rules(pd.DataFrame([candidate]), params)
    row = out.iloc[0]
    print(
        f"Risk ${risk:>3}: shares={row['shares']:.4f}, "
        f"actual_risk=${row['actual_dollars_at_risk']:.2f}, "
        f"pnl=${row['pnl_dollars']:.2f}"
    )
