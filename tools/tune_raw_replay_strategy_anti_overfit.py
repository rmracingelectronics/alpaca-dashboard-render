from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

from src.strategy_tuner import StrategyTuneConfig, generate_configs, metrics_from_trades, prepare_candidate_frame, save_json, select_live_style, split_by_date, bootstrap_metrics


def selected_ts(trades: pd.DataFrame) -> pd.Series:
    if trades is None or trades.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    col = "_ts" if "_ts" in trades.columns else ("timestamp" if "timestamp" in trades.columns else None)
    return pd.to_datetime(trades[col], utc=True, errors="coerce") if col else pd.Series(pd.NaT, index=trades.index)


def enrich_metrics(m: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
    out = dict(m)
    if trades is None or trades.empty:
        out.update(dict(active_months=0, positive_months=0, positive_month_pct=0.0, active_years=0, positive_years=0, positive_year_pct=0.0, symbol_count=0, max_symbol_trade_share=0.0, max_symbol_r_share=0.0, setup_count=0, max_setup_trade_share=0.0, max_setup_r_share=0.0, anti_overfit_score=-9999.0))
        return out
    df = trades.copy()
    ts = selected_ts(df)
    # Convert to a timezone-naive calendar representation before to_period().
    # This avoids thousands of pandas timezone warnings during large tuning runs.
    if getattr(ts.dt, "tz", None) is not None:
        ts_local = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    else:
        ts_local = ts
    df["__month"] = ts_local.dt.to_period("M").astype(str)
    df["__year"] = ts_local.dt.year.astype("Int64").astype(str)
    df["__r"] = pd.to_numeric(df.get("r_multiple", 0.0), errors="coerce").fillna(0.0)
    mr = df.groupby("__month", sort=False)["__r"].sum()
    yr = df.groupby("__year", sort=False)["__r"].sum()
    sym = df.get("_symbol", df.get("symbol", pd.Series("UNKNOWN", index=df.index))).astype(str).str.upper().replace({"":"UNKNOWN"})
    sc = sym.value_counts()
    pos_mask = df["__r"] > 0
    gross_pos = float(df.loc[pos_mask, "__r"].sum())
    sym_pos = df.loc[pos_mask].assign(__sym=sym.loc[pos_mask]).groupby("__sym", sort=False)["__r"].sum() if gross_pos > 0 else pd.Series(dtype=float)
    trig = df.get("_trigger", df.get("trigger_type", pd.Series("UNKNOWN", index=df.index))).astype(str).str.lower()
    setup = pd.Series("other", index=df.index, dtype="object")
    setup.loc[trig.str.contains("vwap_pullback", na=False)] = "vwap_pullback"
    setup.loc[trig.str.contains("gap_cont", na=False)] = "gap_cont"
    setup.loc[trig.str.contains("sweep|reclaim|reject", regex=True, na=False)] = "sweep_reclaim_reject"
    setup.loc[trig.str.contains("orb|opening|retest|10_or", regex=True, na=False)] = "or_retest"
    setup.loc[trig.str.contains("late_trend|trend_follow", regex=True, na=False)] = "late_trend"
    setup_counts = setup.value_counts()
    setup_pos = df.loc[pos_mask].assign(__setup=setup.loc[pos_mask]).groupby("__setup", sort=False)["__r"].sum() if gross_pos > 0 else pd.Series(dtype=float)
    trades_n = int(m.get("trades", 0) or 0)
    out.update({
        "active_months": int(len(mr)),
        "positive_months": int((mr > 0).sum()) if len(mr) else 0,
        "positive_month_pct": float(100.0 * (mr > 0).sum() / len(mr)) if len(mr) else 0.0,
        "active_years": int(len(yr)),
        "positive_years": int((yr > 0).sum()) if len(yr) else 0,
        "positive_year_pct": float(100.0 * (yr > 0).sum() / len(yr)) if len(yr) else 0.0,
        "symbol_count": int(len(sc)),
        "max_symbol_trade_share": float(sc.max() / trades_n) if trades_n else 0.0,
        "max_symbol_r_share": float(sym_pos.max() / gross_pos) if gross_pos > 0 and not sym_pos.empty else 0.0,
        "setup_count": int(len(setup_counts)),
        "max_setup_trade_share": float(setup_counts.max() / trades_n) if trades_n else 0.0,
        "max_setup_r_share": float(setup_pos.max() / gross_pos) if gross_pos > 0 and not setup_pos.empty else 0.0,
    })
    out["anti_overfit_score"] = anti_score(out)
    return out


def anti_score(m: dict[str, Any]) -> float:
    trades = int(m.get("trades", 0) or 0)
    total_r = float(m.get("total_r", 0.0) or 0.0)
    pf = float(m.get("profit_factor", 0.0) or 0.0)
    exp = float(m.get("expectancy_r", 0.0) or 0.0)
    dd = abs(float(m.get("max_drawdown_r", 0.0) or 0.0))
    pm = float(m.get("positive_month_pct", 0.0) or 0.0) / 100.0
    sym_share = float(m.get("max_symbol_trade_share", 0.0) or 0.0)
    sym_r_share = float(m.get("max_symbol_r_share", 0.0) or 0.0)
    setup_share = float(m.get("max_setup_trade_share", 0.0) or 0.0)
    active_months = int(m.get("active_months", 0) or 0)
    return float(total_r + 2*np.log1p(max(0, min(pf,10)-1)) + 2*exp + 3*pm - 0.85*dd - max(0,20-trades)*1.4 - max(0,4-active_months)*1.5 - max(0,sym_share-0.55)*8 - max(0,sym_r_share-0.70)*8 - max(0,setup_share-0.80)*3)


def eval_cfg(df: pd.DataFrame, cfg: StrategyTuneConfig, fixed_risk: float) -> tuple[dict[str, Any], pd.DataFrame]:
    trades = select_live_style(df, cfg)
    m = metrics_from_trades(trades, fixed_risk=fixed_risk)
    m = enrich_metrics(m, trades)
    m.update(cfg.to_dict())
    return m, trades


def acceptable(m: dict[str, Any], a: argparse.Namespace) -> bool:
    return int(m.get("trades",0) or 0) >= a.min_validation_trades and int(m.get("trade_days",0) or 0) >= a.min_validation_trade_days and int(m.get("active_months",0) or 0) >= a.min_validation_active_months and float(m.get("total_r",0) or 0) >= a.min_validation_total_r and float(m.get("profit_factor",0) or 0) >= a.min_validation_pf and float(m.get("expectancy_r",0) or 0) > 0 and abs(float(m.get("max_drawdown_r",0) or 0)) <= a.max_validation_dd_r and float(m.get("max_symbol_trade_share",0) or 0) <= a.max_symbol_trade_share and float(m.get("max_symbol_r_share",0) or 0) <= a.max_symbol_r_share and float(m.get("positive_month_pct",0) or 0) >= a.min_positive_month_pct


def reasons(m: dict[str, Any], a: argparse.Namespace) -> str:
    r=[]
    if int(m.get("trades",0) or 0) < a.min_validation_trades: r.append("too_few_trades")
    if int(m.get("trade_days",0) or 0) < a.min_validation_trade_days: r.append("too_few_trade_days")
    if int(m.get("active_months",0) or 0) < a.min_validation_active_months: r.append("too_few_active_months")
    if float(m.get("total_r",0) or 0) < a.min_validation_total_r: r.append("negative_total_r")
    if float(m.get("profit_factor",0) or 0) < a.min_validation_pf: r.append("low_pf")
    if abs(float(m.get("max_drawdown_r",0) or 0)) > a.max_validation_dd_r: r.append("high_drawdown")
    if float(m.get("max_symbol_trade_share",0) or 0) > a.max_symbol_trade_share: r.append("symbol_trade_concentration")
    if float(m.get("max_symbol_r_share",0) or 0) > a.max_symbol_r_share: r.append("symbol_profit_concentration")
    if float(m.get("positive_month_pct",0) or 0) < a.min_positive_month_pct: r.append("unstable_months")
    return ";".join(r)


def main() -> int:
    p=argparse.ArgumentParser(description="V37.3 anti-overfit train/validation/holdout tuner.")
    p.add_argument("--dataset", required=True); p.add_argument("--train-start", required=True); p.add_argument("--train-end", required=True); p.add_argument("--validate-start", required=True); p.add_argument("--validate-end", required=True); p.add_argument("--holdout-start"); p.add_argument("--holdout-end")
    p.add_argument("--trials", type=int, default=1500); p.add_argument("--top-train-keep", type=int, default=120); p.add_argument("--seed", type=int, default=42); p.add_argument("--fixed-risk", type=float, default=100.0); p.add_argument("--name", default="anti_overfit_tuner_run")
    p.add_argument("--min-train-trades", type=int, default=30); p.add_argument("--min-validation-trades", type=int, default=20); p.add_argument("--min-validation-trade-days", type=int, default=10); p.add_argument("--min-validation-active-months", type=int, default=4); p.add_argument("--min-validation-total-r", type=float, default=0.0); p.add_argument("--min-validation-pf", type=float, default=1.20); p.add_argument("--max-validation-dd-r", type=float, default=6.0); p.add_argument("--max-symbol-trade-share", type=float, default=0.55); p.add_argument("--max-symbol-r-share", type=float, default=0.70); p.add_argument("--min-positive-month-pct", type=float, default=50.0); p.add_argument("--allow-unacceptable-best", action="store_true")
    args=p.parse_args()
    outdir=ROOT/"data"/"tuning_runs"/args.name; outdir.mkdir(parents=True, exist_ok=True)
    df=prepare_candidate_frame(pd.read_csv(args.dataset)); train=split_by_date(df,args.train_start,args.train_end); val=split_by_date(df,args.validate_start,args.validate_end); hold=split_by_date(df,args.holdout_start,args.holdout_end) if args.holdout_start or args.holdout_end else pd.DataFrame()
    train_rows=[]; cfgs=generate_configs(args.trials,args.seed)
    for cfg in cfgs:
        m,_=eval_cfg(train,cfg,args.fixed_risk); m["train_enough_trades"]=int(m.get("trades",0) or 0)>=args.min_train_trades; train_rows.append(m)
    train_df=pd.DataFrame(train_rows).sort_values("anti_overfit_score",ascending=False); train_df.to_csv(outdir/"all_train_trials.csv",index=False)
    finalists=train_df[train_df["trades"].fillna(0).astype(int)>=args.min_train_trades].head(args.top_train_keep)
    if finalists.empty: finalists=train_df.head(args.top_train_keep)
    val_rows=[]
    for _,row in finalists.iterrows():
        cfg=StrategyTuneConfig.from_dict(row.to_dict()); m,tr=eval_cfg(val,cfg,args.fixed_risk); m["acceptable_anti_overfit"]=acceptable(m,args); m["rejection_reasons"]=reasons(m,args); val_rows.append(m)
    val_df=pd.DataFrame(val_rows).sort_values(["acceptable_anti_overfit","anti_overfit_score","total_r","profit_factor","trades"],ascending=[False,False,False,False,False]); val_df.to_csv(outdir/"validation_results.csv",index=False)
    if val_df.empty: raise RuntimeError("No validation results")
    best=val_df.iloc[0].to_dict(); best_cfg=StrategyTuneConfig.from_dict(best); bm,btr=eval_cfg(val,best_cfg,args.fixed_risk); bm["acceptable_anti_overfit"]=acceptable(bm,args); bm["rejection_reasons"]=reasons(bm,args); btr.to_csv(outdir/"best_validation_selected_trades.csv",index=False); btr.to_csv(outdir/"best_validation_trade_decision_report.csv",index=False); pd.DataFrame([bm]).to_csv(outdir/"best_validation_summary.csv",index=False)
    hm={}
    if not hold.empty:
        hm,htr=eval_cfg(hold,best_cfg,args.fixed_risk); hm["acceptable_anti_overfit"]=acceptable(hm,args); hm["rejection_reasons"]=reasons(hm,args); htr.to_csv(outdir/"best_holdout_selected_trades.csv",index=False); htr.to_csv(outdir/"best_holdout_trade_decision_report.csv",index=False); pd.DataFrame([hm]).to_csv(outdir/"best_holdout_summary.csv",index=False); boot=bootstrap_metrics(htr,n=1000,seed=args.seed) if not htr.empty else pd.DataFrame();
        if not boot.empty: boot.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).to_csv(outdir/"holdout_bootstrap_summary.csv")
    if bm.get("acceptable_anti_overfit") or args.allow_unacceptable_best: save_json(outdir/"best_config.json",best_cfg.to_dict())
    else: (outdir/"NO_ACCEPTABLE_CONFIG_FOUND.txt").write_text("No config passed anti-overfit gates. best_config.json was deliberately not written.\n",encoding="utf-8")
    save_json(outdir/"manifest.json",{"version":"v37.3_anti_overfit","dataset":args.dataset,"args":vars(args),"best_validation":bm,"best_holdout":hm})
    print(f"Output folder: {outdir}"); print("Best validation metrics:"); print(pd.DataFrame([bm]).to_string(index=False))
    if hm: print("Best-config holdout metrics:"); print(pd.DataFrame([hm]).to_string(index=False))
    if not bm.get("acceptable_anti_overfit") and not args.allow_unacceptable_best: print("No acceptable config found. best_config.json was NOT written.")
    return 0

if __name__=="__main__": raise SystemExit(main())
