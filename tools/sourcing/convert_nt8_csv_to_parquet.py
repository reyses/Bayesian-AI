"""
Convert NT8 CSV exports to parquet, stitching contracts via roll calendar
and partitioning strictly by CME session-day boundaries.

Reads from: DATA/RAW_NT8/{contract}/{tf}/*.csv
Writes to:  DATA/ATLAS_NT8/{tf}/{session_day}.parquet
"""

import os
import sys
import glob
import re
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core_v2.sessions import session_day_array, session_day
from DATA.pipeline.databento_to_atlas import _build_roll_calendar, _front_for_day

RAW_ROOT = Path('DATA/RAW_NT8')
ATLAS_OUT = Path('DATA/ATLAS_NT8')

QCODE_REV = {'03': 'H', '06': 'M', '09': 'U', '12': 'Z'}

def nt8_folder_to_symbol(folder_name: str) -> str:
    """Map 'MNQ_06-26' -> 'MNQM6'"""
    m = re.match(r'^MNQ_(\d{2})-(\d{2})$', folder_name)
    if m:
        month, year = m.groups()
        return f'MNQ{QCODE_REV[month]}{year[-1]}'
    return folder_name

def read_nt8_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
    expected = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    if not expected.issubset(df.columns):
        raise ValueError(f'{path}: missing cols. Have {list(df.columns)}')
    return df

def process_tf(tf: str, roll_events: list, verify: bool = False) -> list[dict]:
    print(f"\nProcessing TF: {tf}")
    csv_paths = list(RAW_ROOT.glob(f'*/{tf}/*.csv'))
    if not csv_paths:
        print(f"No CSVs found for {tf}")
        return []

    day_contract_data = {}
    
    for path in tqdm(csv_paths, desc=f'Reading {tf} CSVs'):
        contract_folder = path.parent.parent.name
        contract_sym = nt8_folder_to_symbol(contract_folder)
        
        df = read_nt8_csv(path)
        if len(df) == 0:
            continue
            
        df['timestamp'] = df['timestamp'].astype('int64')
        if tf == '1m':
            df['timestamp'] += 59  # top-of-min -> bar-close
            
        for c in ['open', 'high', 'low', 'close']:
            df[c] = df[c].astype('float64')
        df['volume'] = df['volume'].astype('int64')
        
        # Partition by session_day
        df['_day'] = session_day_array(df['timestamp'].to_numpy())
        for day_str, group in df.groupby('_day'):
            if day_str not in day_contract_data:
                day_contract_data[day_str] = {}
            if contract_sym not in day_contract_data[day_str]:
                day_contract_data[day_str][contract_sym] = []
            day_contract_data[day_str][contract_sym].append(group.drop(columns=['_day']))
            
    out_dir = ATLAS_OUT / tf
    out_dir.mkdir(parents=True, exist_ok=True)
    
    total_bars = 0
    days_processed = 0
    manifest = []
    
    for day_str in tqdm(sorted(day_contract_data.keys()), desc=f'Writing {tf}'):
        contracts_dict = day_contract_data[day_str]
        
        for c in contracts_dict:
            c_df = pd.concat(contracts_dict[c], ignore_index=True)
            c_df = c_df.drop_duplicates(subset='timestamp', keep='last').sort_values('timestamp').reset_index(drop=True)
            contracts_dict[c] = c_df
            
        day_date = datetime.strptime(day_str, '%Y_%m_%d').date()
        front_sym = _front_for_day(day_date, roll_events)
        
        if front_sym in contracts_dict:
            chosen_contract = front_sym
            fallback = False
        else:
            chosen_contract = max(contracts_dict.keys(), key=lambda c: contracts_dict[c]['volume'].sum())
            fallback = True
            
        final_df = contracts_dict[chosen_contract]
        out_path = out_dir / f"{day_str}.parquet"
        
        if verify and tf == '1s' and day_str == '2026_03_20':
            # Run parity test
            s1 = final_df.copy()
            s1['bucket'] = (s1['timestamp'] // 5) * 5 + 4
            g = s1.groupby('bucket', sort=True).agg(
                open=('open', 'first'),
                high=('high', 'max'),
                low=('low', 'min'),
                close=('close', 'last'),
                volume=('volume', 'sum'),
            ).reset_index().rename(columns={'bucket': 'timestamp'})
            g['timestamp'] = g['timestamp'].astype('int64')
            g['volume'] = g['volume'].astype('int64')
            g = g[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            existing_path = 'DATA/ATLAS_NT8/5s/2026_03_20.parquet'
            if os.path.exists(existing_path):
                existing = pd.read_parquet(existing_path)
                print(f'\n=== VERIFY 2026_03_20 Parity ===')
                print(f'  Rebin from 1s: {len(g)} rows, vol total {int(g["volume"].sum()):,}')
                print(f'  Existing 5s:   {len(existing)} rows, vol total {int(existing["volume"].sum()):,}')
                merged = g.merge(existing, on='timestamp', how='inner', suffixes=('_rebin', '_exist'))
                print(f'  Matching ts: {len(merged)}')
                if len(merged) > 0:
                    for c in ['open', 'high', 'low', 'close', 'volume']:
                        diff = (merged[f'{c}_rebin'] - merged[f'{c}_exist']).abs()
                        print(f'  {c}: max abs diff = {diff.max():.4f}, mean abs diff = {diff.mean():.4f}')
                print(f'  Session day mapping test:')
                sample_ts = s1['timestamp'].iloc[0]
                computed_day = session_day(sample_ts)
                print(f'    ts={sample_ts} mapped to {computed_day} (expected {day_str})')
                if computed_day == day_str:
                    print(f'    PASS')
                else:
                    print(f'    FAIL')
        
        final_df.to_parquet(out_path, index=False)
        total_bars += len(final_df)
        days_processed += 1
        
        manifest.append({
            'day': day_str,  # matching the databento manifest col
            'chosen': chosen_contract,
            'calendar_fallback': fallback,
            'bars': len(final_df),
        })
        
    print(f"  -> Saved {days_processed} days, {total_bars:,} bars to {out_dir}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true',
                    help='Compare rebin output against existing parquet for 2026_03_20')
    args = ap.parse_args()

    print('Building NT8 Parquet from Raw CSVs...')
    roll_events = _build_roll_calendar(2020, 2035)

    # Process 1s and 1m. Other TFs will be built by build_timeframes.py.
    manifest_1s = process_tf('1s', roll_events, verify=args.verify)
    manifest_1m = process_tf('1m', roll_events, verify=args.verify)
    
    # Save roll manifest if generated
    if manifest_1s:
        man_path = ATLAS_OUT / 'roll_manifest.csv'
        man_df = pd.DataFrame(manifest_1s)
        man_df['rolled'] = man_df['chosen'] != man_df['chosen'].shift(1)
        man_df.loc[0, 'rolled'] = False
        
        # Merge with existing
        if man_path.exists():
            old = pd.read_csv(man_path)
            man_df = pd.concat([old, man_df]).drop_duplicates(subset='day', keep='last')
            man_df = man_df.sort_values('day').reset_index(drop=True)
            
        man_df.to_csv(man_path, index=False)
        print(f"\nSaved roll manifest to {man_path}")
        
        n_rolls = int(man_df['rolled'].sum())
        print(f"Manifest has {len(man_df)} days, {n_rolls} roll seams.")
        
    # Write report
    report_path = Path('reports/findings/nt8_convert_report.txt')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("NT8 CONVERTER REPORT\n")
        f.write("====================\n\n")
        if manifest_1s:
            f.write(f"1s: {len(manifest_1s)} days ({manifest_1s[0]['day']} -> {manifest_1s[-1]['day']})\n")
        if manifest_1m:
            f.write(f"1m: {len(manifest_1m)} days ({manifest_1m[0]['day']} -> {manifest_1m[-1]['day']})\n")

if __name__ == '__main__':
    main()
