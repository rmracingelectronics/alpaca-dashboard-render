from __future__ import annotations
import argparse
import os
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np


def _read_report(path: str) -> pd.DataFrame:
    path = str(path)
    if os.path.isdir(path):
        candidates = [os.path.join(path, 'selected_trade_market_conditions.csv'), os.path.join(path, 'selected_trades.csv')]
        for c in candidates:
            if os.path.exists(c):
                return pd.read_csv(c)
        raise FileNotFoundError(f'No selected trade CSV found in folder: {path}')
    if path.lower().endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            target = next((n for n in names if n.endswith('selected_trade_market_conditions.csv')), None)
            if target is None:
                target = next((n for n in names if n.endswith('selected_trades.csv')), None)
            if target is None:
                raise FileNotFoundError(f'No selected trade CSV found in zip: {path}')
            with z.open(target) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index, dtype='float64')


def _prep(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out['source_report'] = source
    out['r_multiple'] = _num(out, 'r_multiple')
    out['win'] = out['r_multiple'] > 0
    out['side'] = out.get('side', '').astype(str).str.lower()
    side_mult = out['side'].map({'long': 1.0, 'short': -1.0}).fillna(0.0)
    for c in ['day_relative_strength','open_relative_strength','vwap_extension_atr','rvol_time_of_day','daily_atr14_percent','gap_percent','qqq_day_change_percent','candidate_score']:
        out[c] = _num(out, c)
    out['dir_rs'] = side_mult * out['day_relative_strength']
    out['dir_open_rs'] = side_mult * out['open_relative_strength']
    out['dir_vwap'] = side_mult * out['vwap_extension_atr']
    out['abs_rs'] = out['day_relative_strength'].abs()
    out['abs_vwap'] = out['vwap_extension_atr'].abs()
    out['abs_gap'] = out['gap_percent'].abs()
    out['abs_qqq'] = out['qqq_day_change_percent'].abs()
    out['trigger_type'] = out.get('trigger_type', '').astype(str)
    out['entry_candle_pattern'] = out.get('entry_candle_pattern', '').astype(str)
    return out


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        r = g['r_multiple']
        gp = r[r > 0].sum(); gl = -r[r < 0].sum()
        row = {c: v for c, v in zip(group_cols, keys)}
        row.update({
            'trades': int(len(g)),
            'total_r': float(r.sum()),
            'win_rate': float((r > 0).mean() * 100.0),
            'profit_factor': float(gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0),
            'avg_r': float(r.mean()),
            'median_dir_rs': float(g['dir_rs'].median()),
            'median_dir_vwap': float(g['dir_vwap'].median()),
            'median_rvol': float(g['rvol_time_of_day'].median()),
            'median_daily_atr_pct': float(g['daily_atr14_percent'].median()),
            'median_abs_qqq': float(g['abs_qqq'].median()),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['total_r','trades'], ascending=[False,False])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports', nargs='+', required=True, help='Report zip/folder/CSV paths to inspect')
    ap.add_argument('--output', default='data/failure_reversal_analysis')
    args = ap.parse_args()
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for r in args.reports:
        frames.append(_prep(_read_report(r), os.path.basename(str(r).rstrip('/\\'))))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / 'all_selected_trades_with_directional_indicators.csv', index=False)
    _summary(df, ['source_report']).to_csv(outdir / 'report_summary.csv', index=False)
    _summary(df, ['side','trigger_type']).to_csv(outdir / 'side_trigger_summary.csv', index=False)
    _summary(df, ['symbol','side','trigger_type']).to_csv(outdir / 'symbol_side_trigger_summary.csv', index=False)
    _summary(df, ['entry_candle_pattern','side']).to_csv(outdir / 'candle_side_summary.csv', index=False)

    failed = df[df['r_multiple'] < 0].copy()
    failed['opposite_watch'] = np.select(
        [
            (failed['side'].eq('long') & ((failed['day_relative_strength'] < -0.4) | ((failed['qqq_day_change_percent'] < -0.25) & (failed['open_relative_strength'] < 0.25)) | (failed['vwap_extension_atr'] < -0.5))),
            (failed['side'].eq('short') & ((failed['day_relative_strength'] > 0.4) | ((failed['qqq_day_change_percent'] > 0.25) & (failed['open_relative_strength'] > -0.25)) | (failed['vwap_extension_atr'] > 0.5))),
        ],
        ['failed_long_possible_short_watch','failed_short_possible_long_watch'],
        default='skip_not_obvious_reverse'
    )
    failed.to_csv(outdir / 'failed_trade_opposite_watch_diagnostics.csv', index=False)
    print(f'Wrote analysis files to {outdir}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
