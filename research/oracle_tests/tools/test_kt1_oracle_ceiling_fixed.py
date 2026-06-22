"""KT1 oracle-ceiling — CORRECTED join (2026-06-13, Claude verification of Gemini's MSG/KT1).

Gemini's research/test_kt1_oracle_ceiling.py had TWO defects:
  1. JOIN BUG: it read the 1s parquet (pos = 1s-bar index, up to ~69k/day) and compared to
     segment raw_start_idx/raw_end_idx, which the segmenter built in 5s space (~0-16.4k/day).
     => ~76% of the day is past 5s-bar 16.4k, so most trades fell in 'GAP' (58.6% IS) and the
     rest were mis-assigned. Verified: raw_end_idx max 16,419 == 5s bars (16,458), != 1s (69,092).
  2. MISLABEL: report_oracle() computes MEAN PnL per coarse status (PRISTINE/RECOVERED/CHAOS),
     which is the AVERAGE per group, NOT an oracle CEILING (best selectable subset with hindsight).
  Plus the OOS half is vacuous: the segment corpus is 2025 IS; 2026 OOS trades have no segments.

This script fixes the join (read 5s) and restricts to IS. It reports the VALID coarse
stratifier + a crude oracle peek. The TRUE KT1 ceiling (best selectable SUBSET over fine
regime cells / a daisy-chain best-trade oracle) is still TODO — flagged below.
"""
import os, json
import numpy as np, pandas as pd
import pyarrow.parquet as pq

def main():
    IS = pd.read_csv('reports/findings/strategy_runs/nmp_fade_raw_is_atr4.csv')
    segs = json.load(open('artifacts/stage2_year_segments.json'))
    by_day = {}
    for s in segs:
        by_day.setdefault(s['day'], []).append(s)
    for d in by_day:
        by_day[d].sort(key=lambda x: x['raw_start_idx'])

    status, tier = [], []
    for day, g in IS.groupby('day'):
        p = os.path.join('DATA/ATLAS/5s', f"{day}.parquet")   # 5s = the segment's native space
        if day not in by_day or not os.path.exists(p):
            status += [None]*len(g); tier += [None]*len(g); continue
        fts = pq.read_table(p, columns=['timestamp']).to_pandas()['timestamp'].values
        ds = by_day[day]
        ss = np.array([s['raw_start_idx'] for s in ds]); se = np.array([s['raw_end_idx'] for s in ds])
        for ets in g['entry_ts']:
            pos = np.searchsorted(fts, ets, side='right') - 1
            if pos < 0:
                status.append(None); tier.append(None); continue
            si = np.searchsorted(ss, pos, side='right') - 1
            if si >= 0 and pos <= se[si]:
                status.append(ds[si].get('status', 'UNK')); tier.append(ds[si].get('volatility_tier', -1))
            else:
                status.append('GAP'); tier.append(-1)
    IS['seg_status'] = status; IS['seg_tier'] = tier
    v = IS[IS['seg_status'].notna()]
    print(f"CORRECTED (5s) join: matched {len(v):,}/{len(IS):,} | "
          f"GAP {100*(v['seg_status']=='GAP').mean():.1f}% (Gemini's broken 1s join was 58.6%)")
    print("mean pnl by segment status (coarse stratifier, valid join):")
    for st, g in v.groupby('seg_status'):
        print(f"  {st:11s} n={len(g):6d} ({100*len(g)/len(v):4.1f}%)  mean ${g['pnl_usd'].mean():7.2f}")
    dm = IS.groupby('day')['pnl_usd'].mean()
    print(f"\noracle peek (coarse): best DAY mean ${dm.max():.2f}, net-positive days {100*(dm>0).mean():.0f}%, "
          f"top-quartile-day mean ${dm[dm > dm.quantile(.75)].mean():.2f}")
    print("\nTODO (true KT1 ceiling): best selectable SUBSET over FINE regime cells (regime_buckets"
          " w/ corrected metric) or a daisy-chain best-trade oracle — coarse status does NOT bound it.")

if __name__ == '__main__':
    main()
