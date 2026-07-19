import os
import sys
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BUILDERS = os.path.abspath(os.path.join(HERE, '..', '..', 'exit_dojo', 'builders'))
TOOLS = os.path.abspath(os.path.join(HERE, '..', '..', 'exit_dojo', 'tools'))
sys.path.insert(0, BUILDERS)
sys.path.insert(0, TOOLS)

import episode_builder as eb
import telescope_packet_builder as tb
import stop_reenter_sim as srs
sw = srs.sw

def get_engagements(split):
    econ = pd.read_parquet(eb.ECON_DRIFT_PATH, columns=['ts', 'day', 'det', 'is_long', 'P', 'split'])
    thr = float(np.percentile(econ.loc[econ.split == 'train', 'P'].values, tb.P_PCTL))
    if split == 'train':
        sub = econ[(econ.split == 'train') & (econ.P >= thr) & (econ.day.str[:4] == '2024')].copy()
    else:
        sub = econ[(econ.split == 'test') & (econ.P >= thr) & (econ.day.str[:4].isin(['2025', '2026']))].copy()
    
    sub = sub.sort_values(['day', 'is_long', 'ts', 'det']).reset_index(drop=True)
    last = {}
    keep = []
    for r in sub.itertuples():
        k = (r.day, bool(r.is_long))
        if k in last and r.ts - last[k] <= tb.DEDUP_S:
            continue
        last[k] = r.ts
        keep.append(r.Index)
    dd = sub.loc[keep].reset_index(drop=True)
    return dd

import select_wrongdir as swl

def scan_engagements(eng):
    day_engs, all_terminals = swl.scan(eng)
    flat = []
    for d, arr in day_engs.items():
        for i, a in enumerate(arr):
            # day_engs items are dictionaries
            flat.append(a)
    return pd.DataFrame(flat)

if __name__ == '__main__':
    print("Loading delta map...")
    delta_df = pd.read_parquet(os.path.join(HERE, '..', '..', '..', 'DATA', 'ATLAS', 'order_flow_delta_5s.parquet'))
    delta_df['ts_sec'] = delta_df.index.astype(np.int64) // 10**9
    delta_map = dict(zip(delta_df['ts_sec'], delta_df['delta']))

    print("Loading train engagements...")
    train_eng = get_engagements('train')
    train_df = scan_engagements(train_eng)
    
    print("Loading test engagements...")
    test_eng = get_engagements('test')
    test_df = scan_engagements(test_eng)

    def eval_threshold(df, thr_adv):
        # thr_adv is how much adverse delta we tolerate before blocking
        # e.g. thr_adv = 200 means we block Long if delta <= -200
        d = np.array([delta_map.get(int(ts), 0.0) for ts in df['ts']])
        is_long = df['is_long'].values
        # block if adverse delta exceeds thr_adv
        # meaning for long, block if d <= -thr_adv. For short, block if d >= thr_adv
        adverse_delta = np.where(is_long, -d, d)
        
        passes = adverse_delta < thr_adv
        
        terminals = df['terminal'].values
        good = terminals >= 4.0
        
        base_rate = good.mean()
        pass_rate = good[passes].mean() if passes.sum() > 0 else 0.0
        vol_retained = passes.mean()
        
        return base_rate, pass_rate, vol_retained, passes
        
    print(f"Train base N={len(train_df)}")
    best_thr = None
    best_delta = -999
    
    for t in [0, 50, 100, 150, 200, 250, 300, 400]:
        base, p_rate, vol, _ = eval_threshold(train_df, t)
        print(f"Train thr={t}: base={base:.4f} pass={p_rate:.4f} vol={vol:.4f}")
        if vol >= 0.30 and (p_rate - base) > best_delta:
            best_delta = p_rate - base
            best_thr = t
            
    print(f"\nChosen threshold: {best_thr}")
    
    # Evaluate on test
    base, p_rate, vol, passes = eval_threshold(test_df, best_thr)
    print(f"Test result thr={best_thr}: base={base:.4f} pass={p_rate:.4f} vol={vol:.4f}")
    
    # bootstrap CI for good-rate delta
    terminals = test_df['terminal'].values
    passes_mask = passes
    np.random.seed(12345)
    
    # Fast Bootstrap over days
    df_test = pd.DataFrame({
        'day': test_df['day'],
        'terminal': terminals,
        'passes': passes_mask
    })
    day_groups = [group for _, group in df_test.groupby('day')]
    n_days = len(day_groups)
    
    # Pre-calculate sums and counts per day
    day_stats = np.zeros((n_days, 3)) # [num_good, num_passes, num_good_passes]
    day_counts = np.zeros(n_days)
    
    for i, group in enumerate(day_groups):
        t = group['terminal'].values
        p = group['passes'].values
        g = t >= 4.0
        
        day_stats[i, 0] = g.sum()
        day_stats[i, 1] = p.sum()
        day_stats[i, 2] = g[p].sum() if p.sum() > 0 else 0
        day_counts[i] = len(group)
        
    deltas = []
    
    for _ in range(4000):
        idx = np.random.randint(0, n_days, size=n_days)
        tot = day_counts[idx].sum()
        
        tot_good = day_stats[idx, 0].sum()
        tot_passes = day_stats[idx, 1].sum()
        tot_good_passes = day_stats[idx, 2].sum()
        
        b_base = tot_good / tot
        b_pass = tot_good_passes / tot_passes if tot_passes > 0 else b_base
        deltas.append(b_pass - b_base)
        
    deltas = np.sort(deltas)
    ci_lo = deltas[int(0.025 * 4000)]
    ci_hi = deltas[int(0.975 * 4000)]
    
    print(f"Test CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    
    res = f"""# FOOTPRINT-IMB Spinout Report

## Extraction
- **Rule**: {best_thr} adverse delta cutoff
- **Volume Cost**: Retains {vol*100:.1f}% of engagements
- **Base Good Rate**: {base:.4f}
- **Filtered Good Rate**: {p_rate:.4f}
- **Delta CI**: [{ci_lo:.4f}, {ci_hi:.4f}]
"""
    
    with open('../reports/footprint_imbalance_spinout.md', 'w') as f:
        f.write(res)
