from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_ranker_policy import live_style_select_by_score, summarize_trades, bootstrap_summary  # noqa: E402


def _parse_float_list(s: str):
    return [float(x.strip()) for x in str(s).split(',') if x.strip()]


def _parse_int_list(s: str):
    return [int(x.strip()) for x in str(s).split(',') if x.strip()]


def _date_filter(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df.copy()
    if "session_date" in out.columns:
        d = pd.to_datetime(out["session_date"], errors="coerce")
    elif "timestamp" in out.columns:
        d = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.tz_localize(None)
    else:
        return out
    if start:
        out = out[d >= pd.Timestamp(start)].copy()
        d = pd.to_datetime(out.get("session_date", out.get("timestamp")), errors="coerce")
    if end:
        out = out[d <= pd.Timestamp(end)].copy()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Scan thresholds using out-of-sample walk-forward predictions only; no final-model leakage.")
    p.add_argument("--reviewed-candidates", "--reviewed", dest="reviewed_candidates", required=True, help="reviewed_candidates.csv from train_walkforward_ml_ranker.py")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--predicted-r-thresholds", default="-0.05,0,0.02,0.05,0.08,0.10,0.15,0.20")
    p.add_argument("--win-prob-thresholds", default="0.50,0.52,0.55,0.58,0.60")
    p.add_argument("--top-trades", default="1,2,3")
    p.add_argument("--max-symbol-per-day", type=int, default=1)
    p.add_argument("--fixed-risk", type=float, default=100.0)
    p.add_argument("--name", default="walkforward_ranker_oos_threshold_scan")
    args = p.parse_args()

    df = pd.read_csv(args.reviewed_candidates)
    df = _date_filter(df, args.start, args.end)
    if df.empty:
        raise SystemExit("No reviewed candidate rows after date filtering.")
    if "ml_pred_r" not in df.columns:
        if "ml_predicted_r" in df.columns:
            df["ml_pred_r"] = df["ml_predicted_r"]
        else:
            raise SystemExit("reviewed_candidates must contain ml_pred_r or ml_predicted_r.")
    if "ml_pred_win_prob" not in df.columns:
        if "ml_win_prob" in df.columns:
            df["ml_pred_win_prob"] = df["ml_win_prob"]
        else:
            df["ml_pred_win_prob"] = pd.NA

    rows = []
    best = None
    best_score = -1e18
    for min_r in _parse_float_list(args.predicted_r_thresholds):
        for min_wp in _parse_float_list(args.win_prob_thresholds):
            for topn in _parse_int_list(args.top_trades):
                sel = live_style_select_by_score(df, score_col="ml_pred_r", threshold=min_r, top_trades_per_day=topn, max_symbol_per_day=args.max_symbol_per_day, min_win_prob=min_wp)
                summ = summarize_trades(sel, fixed_risk_dollars=args.fixed_risk)
                score = summ.get("total_r", 0) + min(max(summ.get("profit_factor", 0) - 1, -1), 3) - abs(summ.get("max_drawdown_r", 0)) * 0.35
                if summ.get("trades", 0) < 10:
                    score -= (10 - summ.get("trades", 0)) * 0.5
                row = {**summ, "top_trades_per_day": topn, "max_symbol_per_day": args.max_symbol_per_day, "min_predicted_r": min_r, "min_win_prob": min_wp, "selection_score": score, "reviewed_candidates": len(df)}
                rows.append(row)
                if score > best_score:
                    best_score = score
                    best = (row, sel)
    outdir = ROOT / "data" / "ml_backtests" / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    scan = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    scan.to_csv(outdir / "threshold_scan_oos.csv", index=False)
    if best:
        best[1].to_csv(outdir / "best_selected_trades_oos.csv", index=False)
        best_row = best[0]
        best_row.update(bootstrap_summary(best[1], samples=1000))
        pd.DataFrame([best_row]).to_csv(outdir / "best_summary_oos.csv", index=False)
    print(f"Output folder: {outdir}")
    print(scan.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
