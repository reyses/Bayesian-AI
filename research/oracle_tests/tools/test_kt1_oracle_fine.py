import os, json
import numpy as np, pandas as pd
import pyarrow.parquet as pq

RNG = np.random.RandomState(42); NB = 4000

def dayblock_mean_ci(df):
    per_day = df.groupby('day')['pnl_usd'].agg(['sum', 'count'])
    pd_arr = per_day.values
    def mean_est(rows):
        s, c = rows[:, 0].sum(), rows[:, 1].sum()
        return s / c if c > 0 else np.nan
    
    pt = mean_est(pd_arr)
    nd = len(pd_arr)
    boots = [mean_est(pd_arr[RNG.randint(0, nd, nd)]) for _ in range(NB)]
    boots = [b for b in boots if not np.isnan(b)]
    ci = np.percentile(boots, [2.5, 97.5]) if boots else [np.nan, np.nan]
    return pt, ci

def main():
    IS = pd.read_csv('reports/findings/strategy_runs/nmp_fade_raw_is_atr4.csv')
    segs = json.load(open('artifacts/stage2_year_segments.json'))
    by_day = {}
    for s in segs:
        by_day.setdefault(s['day'], []).append(s)
    for d in by_day:
        by_day[d].sort(key=lambda x: x['raw_start_idx'])

    status, tier, length, terms, root_cell = [], [], [], [], []
    for day, g in IS.groupby('day'):
        p = os.path.join('DATA/ATLAS/5s', f"{day}.parquet")
        if day not in by_day or not os.path.exists(p):
            status += [None]*len(g); tier += [None]*len(g)
            length += [None]*len(g); terms += [None]*len(g); root_cell += [None]*len(g)
            continue
        fts = pq.read_table(p, columns=['timestamp']).to_pandas()['timestamp'].values
        ds = by_day[day]
        ss = np.array([s['raw_start_idx'] for s in ds]); se = np.array([s['raw_end_idx'] for s in ds])
        for ets in g['entry_ts']:
            pos = np.searchsorted(fts, ets, side='right') - 1
            if pos < 0:
                status.append(None); tier.append(None); length.append(None); terms.append(None); root_cell.append(None)
                continue
            si = np.searchsorted(ss, pos, side='right') - 1
            if si >= 0 and pos <= se[si]:
                s = ds[si]
                status.append(s.get('status', 'UNK'))
                tier.append(s.get('volatility_tier', -1))
                length.append(s.get('length', -1))
                terms.append(s.get('surviving_terms_count', -1))
                ac = s.get('active_grid_cells', [])
                root_cell.append(ac[0] if ac else -1)
            else:
                status.append('GAP'); tier.append(-1); length.append(-1); terms.append(-1); root_cell.append(-1)
                
    IS['seg_status'] = status
    IS['seg_tier'] = tier
    IS['seg_len'] = length
    IS['seg_terms'] = terms
    IS['seg_root'] = root_cell
    
    v = IS[IS['seg_status'].notna() & (IS['seg_status'] != 'GAP')].copy()
    
    # Bucket by (status, tier, length_quartile)
    v['len_q'] = pd.qcut(v['seg_len'], 4, duplicates='drop')
    v['fine_regime_1'] = v['seg_status'].astype(str) + "_" + v['seg_tier'].astype(str) + "_" + v['len_q'].astype(str)
    
    # Bucket by root cell
    v['fine_regime_2'] = v['seg_root'].astype(str)
    
    # Bucket by (tier, terms)
    v['fine_regime_3'] = v['seg_tier'].astype(str) + "_" + v['seg_terms'].astype(str)

    best_mean = -9999
    best_subset = None
    best_n = 0
    best_name = ""

    for regime_col in ['fine_regime_1', 'fine_regime_2', 'fine_regime_3']:
        for name, g in v.groupby(regime_col):
            if len(g) >= 200:
                m = g['pnl_usd'].mean()
                if m > best_mean:
                    best_mean = m
                    best_subset = g
                    best_n = len(g)
                    best_name = f"{regime_col} = {name}"
                    
    if best_subset is not None:
        pt, ci = dayblock_mean_ci(best_subset)
        print(f"CEILING: ${pt:.2f}, n={best_n}, CI[{ci[0]:.2f},{ci[1]:.2f}]")
        print(f"VERDICT: {'CEILING-POSITIVE' if ci[0] > 0 else 'CEILING-FLAT'}")
        print(f"  (Subset: {best_name})")
    else:
        print("No subset with n >= 200 found.")

if __name__ == '__main__':
    main()
