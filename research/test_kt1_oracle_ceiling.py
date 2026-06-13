import os, sys, json
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq

def load_segments(path='artifacts/stage2_year_segments.json'):
    print("Loading segments...")
    with open(path, 'r') as f:
        segs = json.load(f)
    print(f"Loaded {len(segs):,} segments.")
    return segs

def match_trades_to_segments(trades, segs, atlas_dir):
    # Group segments by day
    segs_by_day = {}
    for s in segs:
        segs_by_day.setdefault(s['day'], []).append(s)
    
    # Sort segments by start_idx
    for d in segs_by_day:
        segs_by_day[d].sort(key=lambda x: x['raw_start_idx'])

    matched_status = []
    matched_tier = []
    
    # Process trades day by day
    for day, g in tqdm(trades.groupby('day'), desc=f"Matching trades ({atlas_dir})"):
        if day not in segs_by_day:
            matched_status.extend([None]*len(g))
            matched_tier.extend([None]*len(g))
            continue
            
        p = os.path.join(atlas_dir, "1s", f"{day}.parquet")
        if not os.path.exists(p):
            # Try NT8
            p2 = p.replace('ATLAS', 'ATLAS_NT8')
            if os.path.exists(p2):
                p = p2
            else:
                matched_status.extend([None]*len(g))
                matched_tier.extend([None]*len(g))
                continue
            
        fts = pq.read_table(p, columns=['timestamp']).to_pandas()['timestamp'].values
        
        day_segs = segs_by_day[day]
        seg_starts = np.array([s['raw_start_idx'] for s in day_segs])
        seg_ends = np.array([s['raw_end_idx'] for s in day_segs])
        
        for ets in g['entry_ts']:
            pos = np.searchsorted(fts, ets, side='right') - 1
            if pos < 0:
                matched_status.append(None)
                matched_tier.append(None)
                continue
                
            s_idx = np.searchsorted(seg_starts, pos, side='right') - 1
            if s_idx >= 0 and pos <= seg_ends[s_idx]:
                matched_status.append(day_segs[s_idx].get('status', 'UNKNOWN'))
                matched_tier.append(day_segs[s_idx].get('volatility_tier', -1))
            else:
                matched_status.append('GAP')
                matched_tier.append(-1)
                
    trades['seg_status'] = matched_status
    trades['seg_tier'] = matched_tier
    return trades

def report_oracle(df, name):
    print(f"\n--- {name} ORACLE CEILING ---")
    valid = df[df['seg_status'].notna()]
    print(f"Matched {len(valid):,}/{len(df):,} trades to segments.")
    
    stats = []
    for st, g in valid.groupby('seg_status'):
        stats.append((st, len(g), g['pnl_usd'].mean(), g['pnl_usd'].sum()))
    stats.sort(key=lambda x: x[1], reverse=True)
    
    for st, n, mean, tot in stats:
        print(f"  {st:12s} | n={n:5d} ({n/len(valid)*100:4.1f}%) | mean ${mean:7.2f} | tot ${tot:9.2f}")
        
    print("\n  By Volatility Tier (PRISTINE=1-2, RECOVERED=3-8, CHAOS=9):")
    for tier, g in valid.groupby('seg_tier'):
        print(f"    Tier {tier} | n={len(g):5d} | mean ${g['pnl_usd'].mean():7.2f} | tot ${g['pnl_usd'].sum():9.2f}")

def main():
    IS = pd.read_csv('reports/findings/strategy_runs/nmp_fade_raw_is_atr4.csv')
    OOS = pd.read_csv('reports/findings/strategy_runs/nmp_fade_raw_oos_atr4.csv')
    
    segs = load_segments()
    
    IS = match_trades_to_segments(IS, segs, 'DATA/ATLAS')
    OOS = match_trades_to_segments(OOS, segs, 'DATA/ATLAS_NT8')
    
    report_oracle(IS, 'IN-SAMPLE')
    report_oracle(OOS, 'OUT-OF-SAMPLE (OOS)')
    
if __name__ == '__main__':
    main()
