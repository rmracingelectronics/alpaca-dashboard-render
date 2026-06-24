from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.config import ML_DATASETS_DIR
from src.ml_ranker_policy import add_live_safe_features

# Reuse the raw-bar live-replay dataset builder from V37 so all deterministic
# signal/risk/outcome logic stays in one place.
from tools.build_q_learning_dataset import build_from_csv, build_from_raw_replay  # type: ignore


def _dt_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a live-safe ML ranker dataset from raw-bar replay candidates or an existing CSV.")
    p.add_argument("--source-csv", default="", help="Existing candidate CSV with r_multiple outcomes. Optional.")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-06-20")
    p.add_argument("--preset", default="v25_playbook")
    p.add_argument("--symbols", default="", help="Comma-separated custom symbols. Overrides preset.")
    p.add_argument("--feed", default="iex")
    p.add_argument("--session-mode", default="regular_only", choices=["regular_only", "extended_hours", "twenty_four_five"])
    p.add_argument("--direction", default="long_short", choices=["long_short", "long_only", "short_only"])
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--candle-mode", default="off")
    p.add_argument("--mean-reversion", default="on")
    p.add_argument("--or-retest", default="on")
    p.add_argument("--news-filter", default="off")
    p.add_argument("--qqq-stress-filter", default="off")
    p.add_argument("--qqq-stress-threshold", type=float, default=4.2)
    p.add_argument("--use-news-proxy", default="off")
    p.add_argument("--slippage-bps", type=float, default=3.0)
    p.add_argument("--base-live-gate", default="off", choices=["off", "v358", "v359", "v364"], help="Optional deterministic gate before ML dataset creation.")
    p.add_argument("--max-candidates-per-day", type=int, default=999)
    p.add_argument("--max-symbol-candidates-per-day", type=int, default=999)
    p.add_argument("--name", default="")
    args = p.parse_args()

    if args.source_csv:
        candidates = build_from_csv(Path(args.source_csv))
        diagnostics = {"source_csv": str(Path(args.source_csv).resolve())}
    else:
        candidates, diagnostics = build_from_raw_replay(args)

    if candidates.empty:
        raise RuntimeError("No candidate rows with r_multiple outcomes were created.")
    if "r_multiple" not in candidates.columns:
        raise RuntimeError("Dataset must contain r_multiple outcomes.")

    candidates = add_live_safe_features(candidates)
    r = pd.to_numeric(candidates["r_multiple"], errors="coerce").fillna(0.0)
    candidates["target_r_multiple"] = r
    candidates["target_win"] = (r > 0).astype(int)
    candidates["target_utility_r_kappa025"] = r - 0.5 * 0.25 * (r ** 2)

    name = args.name or f"ml_ranker_candidates_{args.start}_{args.end}_{_dt_stamp()}"
    out_dir = ML_DATASETS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "candidates.csv"
    candidates.to_csv(out_csv, index=False)

    summary = {
        "rows": int(len(candidates)),
        "symbols": int(candidates.get("symbol", pd.Series(dtype=str)).astype(str).str.upper().nunique()),
        "start": str(args.start),
        "end": str(args.end),
        "win_rate_if_all_traded": float((r > 0).mean() * 100.0),
        "avg_r_if_all_traded": float(r.mean()),
        "total_r_if_all_traded": float(r.sum()),
        "profit_factor_if_all_traded": float(r[r > 0].sum() / abs(r[r < 0].sum())) if float(r[r < 0].sum()) < 0 else 999.0,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "dataset_summary.csv", index=False)
    manifest = {
        "dataset_type": "walkforward_ml_ranker_trade_candidates",
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "args": vars(args),
        "summary": summary,
        "diagnostics": diagnostics,
        "columns": list(candidates.columns),
        "no_lookahead_note": "Feature columns are candidate/timestamp fields. r_multiple is an outcome label used only for training/evaluation.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"ML ranker dataset saved: {out_csv}")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
