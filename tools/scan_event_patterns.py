from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse, json
from src.event_ml import scan_event_patterns

def main():
    p=argparse.ArgumentParser(description='Scan first-touch dataset for event-family edges by train/validate/test split.')
    p.add_argument('--dataset-folder', required=True)
    p.add_argument('--target-r', type=float, default=0.75)
    p.add_argument('--risk-dollars', type=float, default=100.0)
    args=p.parse_args()
    print(json.dumps(scan_event_patterns(args.dataset_folder,args.target_r,args.risk_dollars),indent=2,default=str))
if __name__=='__main__': main()
