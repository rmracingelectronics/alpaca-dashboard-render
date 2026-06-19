from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse, json, math, zipfile, time
from typing import Any
import numpy as np
import pandas as pd

from src.config import ML_MODELS_DIR
from src.research import _file_size_mb
from src.ml_research import _target_tag, _load_dataset_parts, LEAKAGE_SAFE_NUMERIC_FEATURES, LEAKAGE_SAFE_CATEGORICAL_FEATURES
from src.event_ml import add_event_family_columns


def _bucketize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def num(c): return pd.to_numeric(out[c], errors='coerce') if c in out.columns else pd.Series(np.nan, index=out.index)
    out['daily_atr_bucket'] = pd.cut(num('daily_atr14_percent'), [-999,0.8,1.5,2.5,4.0,6.0,999], labels=['atr_lt_0_8','atr_0_8_1_5','atr_1_5_2_5','atr_2_5_4','atr_4_6','atr_gt_6']).astype(str)
    out['dvwap_bucket'] = pd.cut(num('directional_vwap_extension_atr'), [-999,-1.0,-0.25,0.25,0.75,1.5,3.0,999], labels=['dvwap_lt_m1','dvwap_m1_m025','dvwap_m025_025','dvwap_025_075','dvwap_075_15','dvwap_15_3','dvwap_gt_3']).astype(str)
    out['rs_bucket'] = pd.cut(num('directional_day_relative_strength'), [-999,-1.0,-0.25,0.25,0.75,1.5,3.0,999], labels=['rs_lt_m1','rs_m1_m025','rs_m025_025','rs_025_075','rs_075_15','rs_15_3','rs_gt_3']).astype(str)
    out['rvol_bucket'] = pd.cut(num('rvol_time_of_day'), [-999,0.5,0.8,1.2,1.8,3.0,999], labels=['rvol_lt_0_5','rvol_0_5_0_8','rvol_0_8_1_2','rvol_1_2_1_8','rvol_1_8_3','rvol_gt_3']).astype(str)
    out['rangepos_bucket'] = pd.cut(num('directional_range_position_day'), [-999,0.25,0.5,0.75,0.9,999], labels=['range_lt_025','range_025_05','range_05_075','range_075_09','range_gt_09']).astype(str)
    return out


def _agg_chunk(g: pd.DataFrame, keys: list[str], outcome_col: str, tag: str) -> pd.DataFrame:
    y = pd.to_numeric(g[outcome_col], errors='coerce').fillna(0.0)
    tmp = g[keys + ['dataset_split']].copy()
    tmp['_count'] = 1
    tmp['_gross_r'] = y.values
    tmp['_win_count'] = (y.values > 0).astype(int)
    tmp['_bad_count'] = (y.values < 0).astype(int)
    tmp['_gross_win_r'] = np.where(y.values > 0, y.values, 0.0)
    tmp['_gross_loss_r'] = np.where(y.values < 0, -y.values, 0.0)
    ag = tmp.groupby(keys + ['dataset_split'], dropna=False, observed=False).agg(
        trades=('_count','sum'), gross_r=('_gross_r','sum'), win_count=('_win_count','sum'), bad_count=('_bad_count','sum'), gross_win_r=('_gross_win_r','sum'), gross_loss_r=('_gross_loss_r','sum')
    ).reset_index()
    ag['pattern_type'] = tag
    return ag


def _combine(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    group_cols = ['pattern_type'] + keys + ['dataset_split']
    df = df.groupby(group_cols, dropna=False, observed=False).agg(
        trades=('trades','sum'), gross_r=('gross_r','sum'), win_count=('win_count','sum'), bad_count=('bad_count','sum'), gross_win_r=('gross_win_r','sum'), gross_loss_r=('gross_loss_r','sum')
    ).reset_index()
    df['avg_r'] = df['gross_r'] / df['trades'].replace(0, np.nan)
    df['win_rate'] = df['win_count'] / df['trades'].replace(0, np.nan)
    df['bad_rate'] = df['bad_count'] / df['trades'].replace(0, np.nan)
    df['profit_factor'] = df['gross_win_r'] / df['gross_loss_r'].replace(0, np.nan)
    return df


def _score_candidates(summary: pd.DataFrame, keys: list[str], min_train:int, min_validate:int, min_test:int) -> pd.DataFrame:
    if summary.empty:
        return summary
    idx = ['pattern_type'] + keys
    piv_avg = summary.pivot_table(index=idx, columns='dataset_split', values='avg_r', aggfunc='first')
    piv_tr = summary.pivot_table(index=idx, columns='dataset_split', values='trades', aggfunc='first').fillna(0)
    piv_pf = summary.pivot_table(index=idx, columns='dataset_split', values='profit_factor', aggfunc='first')
    out = piv_avg.copy()
    out.columns = [f'avg_r_{c}' for c in out.columns]
    for c in piv_tr.columns:
        out[f'trades_{c}'] = piv_tr[c]
    for c in piv_pf.columns:
        out[f'pf_{c}'] = piv_pf[c]
    for c in ['train','validate','test']:
        if f'avg_r_{c}' not in out.columns: out[f'avg_r_{c}'] = np.nan
        if f'trades_{c}' not in out.columns: out[f'trades_{c}'] = 0
        if f'pf_{c}' not in out.columns: out[f'pf_{c}'] = np.nan
    out = out.reset_index()
    out['passes_train_validate'] = (out['trades_train']>=min_train) & (out['trades_validate']>=min_validate) & (out['avg_r_train']>0) & (out['avg_r_validate']>0) & (out['pf_train']>1.02) & (out['pf_validate']>1.01)
    out['passes_all_three'] = out['passes_train_validate'] & (out['trades_test']>=min_test) & (out['avg_r_test']>0) & (out['pf_test']>1.01)
    out['robust_score'] = out[['avg_r_train','avg_r_validate','avg_r_test']].min(axis=1) * np.log1p(out[['trades_train','trades_validate','trades_test']].min(axis=1).clip(lower=0))
    out = out.sort_values(['passes_all_three','passes_train_validate','robust_score','avg_r_validate'], ascending=[False,False,False,False])
    return out


def scan_sub_event_patterns(dataset_folder: str, target_r: float=0.75, risk_dollars: float=100.0, min_train:int=150, min_validate:int=50, min_test:int=25) -> dict[str,Any]:
    started=time.time()
    tag=_target_tag(target_r)
    outcome_col=f'outcome_r_{tag}'
    use_cols=list(dict.fromkeys(['symbol','side','dataset_split','session_date','signal_time_et','entry_time_et','entry_hour_et','time_bucket','candle_pattern_primary',outcome_col,'mfe_r','mae_r','final_r','bars_to_stop',*LEAKAGE_SAFE_NUMERIC_FEATURES,*LEAKAGE_SAFE_CATEGORICAL_FEATURES]))
    pattern_defs=[
        ('event_side',['event_family','side']),
        ('event_side_timebucket',['event_family','side','time_bucket']),
        ('event_side_hour',['event_family','side','entry_hour_et']),
        ('event_side_candle',['event_family','side','candle_pattern_primary']),
        ('event_side_time_candle',['event_family','side','time_bucket','candle_pattern_primary']),
        ('event_side_atr',['event_family','side','daily_atr_bucket']),
        ('event_side_dvwap',['event_family','side','dvwap_bucket']),
        ('event_side_rs',['event_family','side','rs_bucket']),
        ('event_side_rvol',['event_family','side','rvol_bucket']),
        ('event_side_rangepos',['event_family','side','rangepos_bucket']),
        ('symbol_event_side',['symbol','event_family','side']),
        ('symbol_event_side_time',['symbol','event_family','side','time_bucket']),
    ]
    buckets={name:[] for name,_ in pattern_defs}
    event_rows=0; total_rows=0
    for part in _load_dataset_parts(dataset_folder, columns=use_cols):
        if part.empty or outcome_col not in part.columns: continue
        total_rows += len(part)
        part=add_event_family_columns(part)
        part=part[part['event_count']>0].copy()
        if part.empty: continue
        part=_bucketize(part)
        event_rows += len(part)
        for name,keys in pattern_defs:
            buckets[name].append(_agg_chunk(part, keys, outcome_col, name))
    out_dir=ML_MODELS_DIR / f'sub_event_scan_{Path(dataset_folder).name}_target{tag}_{pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_frames=[]; summary_info=[]
    for name,keys in pattern_defs:
        summ=_combine(buckets[name], keys)
        if not summ.empty:
            summ.to_csv(out_dir / f'{name}_split_summary.csv', index=False)
            cand=_score_candidates(summ, keys, min_train, min_validate, min_test)
            cand.to_csv(out_dir / f'{name}_candidates.csv', index=False)
            top=cand[(cand['passes_train_validate'])].head(200).copy()
            if not top.empty:
                candidate_frames.append(top)
            summary_info.append({'pattern_type':name,'summary_rows':int(len(summ)),'candidate_rows':int(len(cand)),'train_validate_pass':int(cand['passes_train_validate'].sum()) if 'passes_train_validate' in cand else 0,'all_three_pass':int(cand['passes_all_three'].sum()) if 'passes_all_three' in cand else 0})
    allc=pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    if not allc.empty:
        allc=allc.sort_values(['passes_all_three','robust_score','avg_r_validate'], ascending=[False,False,False])
    allc.to_csv(out_dir / 'ALL_ACCEPTED_SUBPATTERNS.csv', index=False)
    pd.DataFrame(summary_info).to_csv(out_dir / 'scan_summary.csv', index=False)
    manifest={'dataset_folder':str(dataset_folder),'target_r':target_r,'outcome_col':outcome_col,'total_rows_seen':int(total_rows),'event_rows_seen':int(event_rows),'min_train':min_train,'min_validate':min_validate,'min_test':min_test,'elapsed_seconds':round(time.time()-started,2),'note':'Sub-event scan searches event+side+time/candle/bucket pockets. Treat as discovery; next step is confirm with a real rule backtest.'}
    (out_dir/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    zip_path=ML_MODELS_DIR / f'{out_dir.name}.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
        for f in sorted(out_dir.iterdir()): zf.write(f,arcname=f.name)
    return {'report_folder':str(out_dir),'zip_path':str(zip_path),'accepted_subpatterns':int(len(allc)),'all_three_pass':int(allc['passes_all_three'].sum()) if not allc.empty and 'passes_all_three' in allc else 0,'event_rows_seen':int(event_rows),'total_rows_seen':int(total_rows),'elapsed_seconds':manifest['elapsed_seconds'],'zip_mb':round(_file_size_mb(zip_path),2)}


def main():
    p=argparse.ArgumentParser(description='Scan narrower event+side+time/candle/feature-bucket patterns on first-touch dataset.')
    p.add_argument('--dataset-folder', required=True)
    p.add_argument('--target-r', type=float, default=0.75)
    p.add_argument('--risk-dollars', type=float, default=100.0)
    p.add_argument('--min-train', type=int, default=150)
    p.add_argument('--min-validate', type=int, default=50)
    p.add_argument('--min-test', type=int, default=25)
    args=p.parse_args()
    print(json.dumps(scan_sub_event_patterns(args.dataset_folder,args.target_r,args.risk_dollars,args.min_train,args.min_validate,args.min_test), indent=2, default=str))

if __name__=='__main__':
    main()
