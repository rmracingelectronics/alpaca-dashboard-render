from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import pandas as pd

from src.backtest import run_backtest
from src.config import ML_DATASETS_DIR, StrategyParams
from src.symbols import parse_symbols


def _dt_stamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def _clean_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Keep rows with an actual simulated outcome.
    if "r_multiple" in out.columns:
        out["r_multiple"] = pd.to_numeric(out["r_multiple"], errors="coerce")
        out = out.dropna(subset=["r_multiple"])
    if "session_date" not in out.columns:
        if "timestamp" in out.columns:
            out["session_date"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date.astype(str)
        elif "entry_time" in out.columns:
            out["session_date"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date.astype(str)
    # Normalize common fields so later tools are independent of report version.
    if "candidate_score" not in out.columns and "score" in out.columns:
        out["candidate_score"] = out["score"]
    if "timestamp" not in out.columns and "raw_bar_replay_signal_time" in out.columns:
        out["timestamp"] = out["raw_bar_replay_signal_time"]
    return out.reset_index(drop=True)


def build_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _clean_candidates(df)


def build_from_raw_replay(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    symbols = parse_symbols(args.symbols, preset=args.preset)
    if not symbols:
        raise RuntimeError("No tradable symbols resolved. Check --preset or --symbols.")
    params = StrategyParams(
        strategy_profile="symbol_playbook_v25",
        backtest_decision_mode="full_raw_bar_replay",
        backtest_session_mode=args.session_mode,
        v25_allow_generic_symbols=bool(args.symbols and args.symbols.strip()),
        direction_mode=args.direction,
        max_trades_per_day=int(args.max_candidates_per_day),
        max_open_positions=int(args.max_candidates_per_day),
        max_alerts_per_symbol_per_day=int(args.max_symbol_candidates_per_day),
        min_candidate_score=float(args.min_score),
        candle_pattern_mode=args.candle_mode,
        enable_mean_reversion=args.mean_reversion.lower() in {"on", "true", "yes", "1"},
        enable_or_retest=args.or_retest.lower() in {"on", "true", "yes", "1"},
        v27_news_filter_mode=args.news_filter,
        v27_market_stress_mode=args.qqq_stress_filter,
        v27_qqq_stress_abs_change_pct=float(args.qqq_stress_threshold),
        slippage_bps=float(args.slippage_bps),
        position_sizing_mode="fixed_dollar_risk",
        risk_per_trade_dollars=100.0,
    )
    # Start with all custom live filters off unless explicitly requested. The Q
    # learner should see a broad enough candidate set to learn what to reject.
    if args.base_live_gate == "v364":
        params.enable_v364_professional_momentum_filter = True
    elif args.base_live_gate == "v359":
        params.enable_v359_live_hunter_filter = True
    elif args.base_live_gate == "v358":
        params.enable_v358_live_quality_filter = True

    summary = run_backtest(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        params=params,
        feed=args.feed,
        use_cache=True,
        use_news=args.use_news_proxy.lower() in {"on", "true", "yes", "1"},
        export_report=False,
        session_mode=args.session_mode,
    )
    candidates = summary.get("portfolio_trades", pd.DataFrame())
    if candidates is None or candidates.empty:
        # Some report versions use selected_trades inside summarize_results.
        candidates = summary.get("selected_trades", pd.DataFrame())
    return _clean_candidates(candidates), summary.get("diagnostics", {})


def main() -> None:
    p = argparse.ArgumentParser(description="Build a live-safe Q-learning candidate dataset from raw replay or an existing all-candidates CSV.")
    p.add_argument("--source-csv", default="", help="Existing CSV with candidate rows and r_multiple outcomes. Optional.")
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
    p.add_argument("--base-live-gate", default="off", choices=["off", "v358", "v359", "v364"], help="Optional deterministic gate applied before ML dataset creation.")
    p.add_argument("--max-candidates-per-day", type=int, default=999)
    p.add_argument("--max-symbol-candidates-per-day", type=int, default=999)
    p.add_argument("--name", default="")
    args = p.parse_args()

    if args.source_csv:
        source = Path(args.source_csv)
        candidates = build_from_csv(source)
        diagnostics = {"source_csv": str(source)}
    else:
        candidates, diagnostics = build_from_raw_replay(args)

    if candidates.empty:
        raise RuntimeError("No candidate rows with r_multiple outcomes were created.")

    name = args.name or f"q_learning_candidates_{args.start}_{args.end}_{_dt_stamp()}"
    out_dir = ML_DATASETS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "candidates.csv"
    candidates.to_csv(out_csv, index=False)
    manifest = {
        "dataset_type": "q_learning_trade_candidates",
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "rows": int(len(candidates)),
        "symbols": sorted(candidates.get("symbol", pd.Series(dtype=str)).astype(str).str.upper().unique().tolist()),
        "start": str(args.start),
        "end": str(args.end),
        "feed": str(args.feed),
        "session_mode": str(args.session_mode),
        "source_csv": str(args.source_csv or ""),
        "diagnostics": diagnostics,
        "columns": list(candidates.columns),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    # Quick class balance / outcome summary.
    r = pd.to_numeric(candidates["r_multiple"], errors="coerce").fillna(0.0)
    pd.DataFrame([{
        "rows": int(len(candidates)),
        "win_rate": float((r > 0).mean() * 100.0),
        "avg_r": float(r.mean()),
        "total_r_if_all_traded": float(r.sum()),
        "profit_factor_if_all_traded": float(r[r > 0].sum() / abs(r[r < 0].sum())) if float(r[r < 0].sum()) < 0 else 999.0,
    }]).to_csv(out_dir / "dataset_summary.csv", index=False)
    print(f"Dataset saved: {out_csv}")
    print(f"Rows: {len(candidates)}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
