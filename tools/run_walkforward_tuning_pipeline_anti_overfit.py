from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.strategy_tuner import prepare_candidate_frame, split_by_date, metrics_from_trades, save_json, bootstrap_metrics, StrategyTuneConfig
# Reuse helper functions from the train/validation tuner script.
from tools.tune_raw_replay_strategy_anti_overfit import eval_cfg, acceptable, reasons


def windows(df: pd.DataFrame, start: str, end: str, train_days: int, validate_days: int, test_days: int, lookahead_days: int):
    dates = pd.Series(pd.to_datetime(df["_date"], errors="coerce").dropna().unique()).sort_values().reset_index(drop=True)
    if start: dates = dates[dates >= pd.Timestamp(start)]
    if end: dates = dates[dates <= pd.Timestamp(end)]
    dates = dates.reset_index(drop=True); out=[]; i=0
    while i + train_days + validate_days + test_days + 2*lookahead_days <= len(dates):
        tr_s=dates.iloc[i]; tr_e=dates.iloc[i+train_days-1]
        va_s=dates.iloc[i+train_days+lookahead_days]; va_e=dates.iloc[i+train_days+lookahead_days+validate_days-1]
        te_s=dates.iloc[i+train_days+lookahead_days+validate_days+lookahead_days]; te_e=dates.iloc[i+train_days+lookahead_days+validate_days+lookahead_days+test_days-1]
        out.append({"train_start":str(tr_s.date()),"train_end":str(tr_e.date()),"validate_start":str(va_s.date()),"validate_end":str(va_e.date()),"test_start":str(te_s.date()),"test_end":str(te_e.date())})
        i += test_days
    return out


def main() -> int:
    p=argparse.ArgumentParser(description="V37.3 rolling anti-overfit walk-forward tuner.")
    p.add_argument("--dataset", required=True); p.add_argument("--start", default=""); p.add_argument("--end", default=""); p.add_argument("--train-days", type=int, default=504); p.add_argument("--validate-days", type=int, default=126); p.add_argument("--test-days", type=int, default=63); p.add_argument("--lookahead-days", type=int, default=1); p.add_argument("--trials-per-window", type=int, default=900); p.add_argument("--top-train-keep", type=int, default=100); p.add_argument("--seed", type=int, default=42); p.add_argument("--fixed-risk", type=float, default=100.0); p.add_argument("--name", default="walkforward_anti_overfit_tuning")
    p.add_argument("--min-train-trades", type=int, default=30); p.add_argument("--min-validation-trades", type=int, default=15); p.add_argument("--min-validation-trade-days", type=int, default=8); p.add_argument("--min-validation-active-months", type=int, default=3); p.add_argument("--min-validation-total-r", type=float, default=0.0); p.add_argument("--min-validation-pf", type=float, default=1.20); p.add_argument("--max-validation-dd-r", type=float, default=6.0); p.add_argument("--max-symbol-trade-share", type=float, default=0.60); p.add_argument("--max-symbol-r-share", type=float, default=0.75); p.add_argument("--min-positive-month-pct", type=float, default=50.0); p.add_argument("--allow-unacceptable-windows", action="store_true")
    args=p.parse_args(); outdir=ROOT/"data"/"tuning_runs"/args.name; outdir.mkdir(parents=True,exist_ok=True)
    df=prepare_candidate_frame(pd.read_csv(args.dataset)); wins=windows(df,args.start,args.end,args.train_days,args.validate_days,args.test_days,args.lookahead_days)
    if not wins: raise RuntimeError("No windows generated")
    win_rows=[]; selected=[]; configs=[]
    for i,w in enumerate(wins, start=1):
        # Call the single tuner into a window subfolder, then load its results.
        sub_name=f"{args.name}_window_{i:02d}"
        cmd=[sys.executable,"tools/tune_raw_replay_strategy_anti_overfit.py","--dataset",args.dataset,"--train-start",w["train_start"],"--train-end",w["train_end"],"--validate-start",w["validate_start"],"--validate-end",w["validate_end"],"--holdout-start",w["test_start"],"--holdout-end",w["test_end"],"--trials",str(args.trials_per_window),"--top-train-keep",str(args.top_train_keep),"--seed",str(args.seed+i*1000),"--fixed-risk",str(args.fixed_risk),"--name",sub_name,"--min-train-trades",str(args.min_train_trades),"--min-validation-trades",str(args.min_validation_trades),"--min-validation-trade-days",str(args.min_validation_trade_days),"--min-validation-active-months",str(args.min_validation_active_months),"--min-validation-total-r",str(args.min_validation_total_r),"--min-validation-pf",str(args.min_validation_pf),"--max-validation-dd-r",str(args.max_validation_dd_r),"--max-symbol-trade-share",str(args.max_symbol_trade_share),"--max-symbol-r-share",str(args.max_symbol_r_share),"--min-positive-month-pct",str(args.min_positive_month_pct)]
        if args.allow_unacceptable_windows: cmd.append("--allow-unacceptable-best")
        subprocess.call(cmd, cwd=str(ROOT))
        sub=ROOT/"data"/"tuning_runs"/sub_name
        val_sum=pd.read_csv(sub/"best_validation_summary.csv").iloc[0].to_dict() if (sub/"best_validation_summary.csv").exists() else {}
        hold_sum=pd.read_csv(sub/"best_holdout_summary.csv").iloc[0].to_dict() if (sub/"best_holdout_summary.csv").exists() else {}
        if (sub/"best_holdout_selected_trades.csv").exists():
            tr=pd.read_csv(sub/"best_holdout_selected_trades.csv"); tr["walkforward_window"]=i; selected.append(tr)
        cfg={}
        if (sub/"best_config.json").exists():
            cfg=json.loads((sub/"best_config.json").read_text()); cfg["window"]=i; configs.append(cfg)
        row={"window":i,**w,"validation_acceptable":bool(val_sum.get("acceptable_anti_overfit",False)),"validate_trades":val_sum.get("trades",0),"validate_total_r":val_sum.get("total_r",0.0),"validate_pf":val_sum.get("profit_factor",0.0),"validate_rejection":val_sum.get("rejection_reasons","")}
        row.update({"test_trades":hold_sum.get("trades",0),"test_total_r":hold_sum.get("total_r",0.0),"test_pf":hold_sum.get("profit_factor",0.0),"test_win_rate":hold_sum.get("win_rate",0.0),"test_rejection":hold_sum.get("rejection_reasons","")})
        win_rows.append(row); print(f"Window {i}/{len(wins)} test R={row['test_total_r']} trades={row['test_trades']} acceptable={row['validation_acceptable']}")
    wf=pd.DataFrame(win_rows); wf.to_csv(outdir/"walkforward_window_summary.csv",index=False)
    if configs: pd.DataFrame(configs).to_csv(outdir/"chosen_configs.csv",index=False)
    allsel=pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(); allsel.to_csv(outdir/"walkforward_selected_trades.csv",index=False); allsel.to_csv(outdir/"walkforward_trade_decision_report.csv",index=False)
    overall=metrics_from_trades(allsel,args.fixed_risk); overall["windows"]=len(wf); overall["positive_windows"]=int((pd.to_numeric(wf["test_total_r"],errors="coerce").fillna(0)>0).sum()); overall["positive_window_pct"]=float(100*overall["positive_windows"]/max(1,len(wf)))
    pd.DataFrame([overall]).to_csv(outdir/"walkforward_overall_summary.csv",index=False)
    if not allsel.empty:
        boot=bootstrap_metrics(allsel,n=1000,seed=args.seed)
        if not boot.empty: boot.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).to_csv(outdir/"walkforward_bootstrap_summary.csv")
    save_json(outdir/"manifest.json",{"version":"v37.3_anti_overfit_walkforward","args":vars(args),"windows":wins,"overall":overall})
    print(f"Output folder: {outdir}"); print(pd.DataFrame([overall]).to_string(index=False)); return 0

if __name__=="__main__": raise SystemExit(main())
