from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse, json
from src.event_ml import train_event_ev_model

def main():
    p=argparse.ArgumentParser(description='Train EV model only on realistic event-family rows.')
    p.add_argument('--dataset-folder', required=True)
    p.add_argument('--target-r', type=float, default=0.75)
    p.add_argument('--max-train-rows', type=int, default=500000)
    p.add_argument('--max-eval-rows-per-split', type=int, default=250000)
    p.add_argument('--max-trades-per-day', type=int, default=3)
    p.add_argument('--risk-dollars', type=float, default=100.0)
    args=p.parse_args()
    res=train_event_ev_model(args.dataset_folder,args.target_r,args.max_train_rows,args.max_eval_rows_per_split,args.max_trades_per_day,args.risk_dollars)
    print(json.dumps(res,indent=2,default=str))
if __name__=='__main__': main()
