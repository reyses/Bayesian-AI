"""Pure entry-accuracy diagnostic — no exit logic.

User question (2026-05-17): without considering exits, how accurate are
entries? Strip out R-trigger / trail / target — just measure the entry's
forward trajectory and ask: did this entry go where we predicted?

For each leg (entry at pivot, exit at next pivot — pure leg structure):

  Trajectory metrics (favorable move from entry, signed by leg dir):
    favorable_at_T(t)   = max favorable price - entry, for first t seconds
    final_amplitude     = leg MFE (= leg_amplitude as previously)
    time_to_1R          = first t at which favorable >= R
    time_to_MFE         = t at which leg's max favorable is reached
    max_retrace_pct     = max retrace during the leg / final amplitude
                          (intra-leg whipsaw indicator)

  Hit rates (target-style):
    pct_reach_0.5R, 1R, 2R, 3R, 4R favorable before next pivot

Stratify all of the above by entry composite signal.

The key question: does the composite at entry predict not just final
amplitude but ALSO the path quality (clean ride vs whipsaw)?
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_zigzag_baseline import compute_atr, TICK_SIZE


DOLLAR_PER_POINT = 2.0
TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')

TIMEPOINTS_S = [60, 120, 180, 300, 600, 900]   # 1, 2, 3, 5, 10, 15 min
R_MULTIPLES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def derive_pivot_events(truth_day):
    piv = truth_day[truth_day['is_pivot'] == 1].sort_values('timestamp')
    if len(piv) == 0:
        return []
    ts = piv['timestamp'].values.astype(np.int64)
    pd_ = piv['pivot_dir'].values
    pp_ = piv['pivot_price'].values
    groups = [[0]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > 90:
            groups.append([i])
        else:
            groups[-1].append(i)
    out = []
    for grp in groups:
        ts_c = int(np.median(ts[grp]))
        vals, counts = np.unique(pd_[grp], return_counts=True)
        d = str(vals[np.argmax(counts)])
        p = float(np.mean(pp_[grp]))
        out.append((ts_c, d, p))
    return out


def analyze_leg(closes5s, ts5s, entry_ts, entry_price, leg_dir, end_ts, r_price):
    """Return dict of entry-trajectory metrics (no exit logic)."""
    mask = (ts5s >= entry_ts) & (ts5s <= end_ts)
    closes_leg = closes5s[mask]
    ts_leg = ts5s[mask]
    if len(closes_leg) == 0:
        return None

    # Favorable move at each bar (signed by leg direction)
    if leg_dir == 'LONG':
        favorable = closes_leg - entry_price   # >0 = favorable
        running_max_fav = np.maximum.accumulate(favorable)
        # Adverse = how far below entry we went (LONG leg, should be 0 since
        # we entered at the LOW pivot, but include for completeness)
        running_min_fav = np.minimum.accumulate(favorable)
    else:
        favorable = entry_price - closes_leg
        running_max_fav = np.maximum.accumulate(favorable)
        running_min_fav = np.minimum.accumulate(favorable)

    rel_ts = ts_leg - entry_ts   # seconds since entry

    # Favorable at fixed time points
    fav_at_t = {}
    for t in TIMEPOINTS_S:
        # find last index where rel_ts <= t
        idx = np.searchsorted(rel_ts, t, side='right') - 1
        if idx < 0:
            fav_at_t[t] = np.nan
        else:
            fav_at_t[t] = float(running_max_fav[idx])

    # Time to reach R multiples
    time_to_R = {}
    for m in R_MULTIPLES:
        threshold = m * r_price
        hits = np.where(running_max_fav >= threshold)[0]
        if len(hits) == 0:
            time_to_R[m] = np.nan
        else:
            time_to_R[m] = float(rel_ts[hits[0]])

    final_amp = float(running_max_fav[-1])

    # Time to MFE (peak)
    time_to_mfe = float(rel_ts[np.argmax(favorable)])

    # Max within-leg adverse from running_max_fav (whipsaw indicator)
    # i.e., max drawdown from running peak during the leg
    if leg_dir == 'LONG':
        drawdown = running_max_fav - favorable
    else:
        drawdown = running_max_fav - favorable
    max_intra_dd = float(drawdown.max())

    # MAE from entry (should be ~0 for oracle entry at pivot extreme)
    mae_from_entry = float(-running_min_fav.min())   # absolute adverse value

    out_d = {
        'entry_ts': int(entry_ts),
        'leg_dir': leg_dir,
        'r_price': float(r_price),
        'leg_end_ts': int(end_ts),
        'leg_duration_s': int(end_ts - entry_ts),
        'final_amplitude_pts': final_amp,
        'final_amplitude_usd': final_amp * DOLLAR_PER_POINT,
        'time_to_mfe_s': time_to_mfe,
        'max_intra_dd_pts': max_intra_dd,
        'max_intra_dd_usd': max_intra_dd * DOLLAR_PER_POINT,
        'mae_from_entry_pts': mae_from_entry,
        'whipsaw_ratio': max_intra_dd / max(final_amp, 1e-9),
    }
    for t in TIMEPOINTS_S:
        out_d[f'fav_at_{t}s_pts'] = fav_at_t[t]
        out_d[f'fav_at_{t}s_usd'] = fav_at_t[t] * DOLLAR_PER_POINT
    for m in R_MULTIPLES:
        out_d[f'time_to_{m:.1f}R'] = time_to_R[m]
        out_d[f'hit_{m:.1f}R'] = int(not np.isnan(time_to_R[m]))
    return out_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--entry-legs',
                    default='reports/findings/regret_oracle/composite_entry_analyzer.csv',
                    help='From composite_entry_analyzer.py — gives entry composite signals')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_entry_accuracy.csv')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/composite_entry_accuracy.txt')
    args = ap.parse_args()

    print('Loading truth + entry composites...')
    truth = pd.read_parquet(args.truth)
    entries = pd.read_csv(args.entry_legs)
    print(f'  truth {len(truth):,}  entries {len(entries):,}')

    rows = []
    for day in tqdm(sorted(truth['day'].unique()), desc='days'):
        bars5s_path = NT8_5S_DIR / f'{day}.parquet'
        bars1m_path = NT8_1M_DIR / f'{day}.parquet'
        if not bars5s_path.exists() or not bars1m_path.exists():
            continue
        bars1m = pd.read_parquet(bars1m_path).sort_values('timestamp').reset_index(drop=True)
        bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
        atr_pts = compute_atr(bars1m, 14)
        min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * TRAIN_ATR_MULT)))
        r_price = min_rev_ticks * TICK_SIZE

        truth_day = truth[truth['day'] == day]
        events = derive_pivot_events(truth_day)
        if len(events) < 2:
            continue
        closes5s = bars5s['close'].values.astype(np.float64)
        ts5s = bars5s['timestamp'].values.astype(np.int64)

        for k in range(len(events) - 1):
            entry_ts, leg_dir, entry_price = events[k]
            next_ts = events[k + 1][0]
            r = analyze_leg(closes5s, ts5s, entry_ts, entry_price, leg_dir,
                              next_ts, r_price)
            if r is None:
                continue
            r['day'] = day
            r['atr_pts'] = atr_pts
            rows.append(r)

    df = pd.DataFrame(rows)
    # Merge entry composite signals
    df_merged = df.merge(
        entries[['day', 'entry_ts', 'entry_zone', 'entry_p_b6_match',
                 'entry_p_mid', 'entry_p_early', 'entry_p_late']],
        on=['day', 'entry_ts'], how='inner',
    )
    df_merged.to_csv(args.out, index=False)
    print(f'\nWrote: {args.out}  ({len(df_merged):,} legs)')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('PURE ENTRY ACCURACY — no exit logic, just leg trajectories from entry')
    out('=' * 78)
    out(f'Legs: {len(df_merged):,}   Days: {df_merged["day"].nunique()}')

    # === Headline trajectory stats (all legs) ===
    out('')
    out('--- Aggregate trajectory (all legs) ---')
    out(f'  Final amplitude:    mean ${df_merged["final_amplitude_usd"].mean():.2f}   '
        f'median ${df_merged["final_amplitude_usd"].median():.2f}   '
        f'p25 ${df_merged["final_amplitude_usd"].quantile(0.25):.2f}   '
        f'p75 ${df_merged["final_amplitude_usd"].quantile(0.75):.2f}')
    out(f'  Time to MFE:        mean {df_merged["time_to_mfe_s"].mean()/60:.1f}m   '
        f'median {df_merged["time_to_mfe_s"].median()/60:.1f}m')
    out(f'  Max intra-leg DD:   mean ${df_merged["max_intra_dd_usd"].mean():.2f}   '
        f'median ${df_merged["max_intra_dd_usd"].median():.2f}')
    out(f'  Whipsaw ratio:      mean {df_merged["whipsaw_ratio"].mean():.2f}   '
        f'(DD / final amp; 0 = clean ride, 1 = pure whipsaw)')
    out(f'  MAE from entry:     mean ${df_merged["mae_from_entry_pts"].mean() * DOLLAR_PER_POINT:.2f}   '
        f'(should be ~0 since entries are at pivot extremes)')

    # === Favorable move at fixed times ===
    out('')
    out('--- Favorable move at fixed time-after-entry ($/contract) ---')
    out(f'  {"timepoint":<12}  {"mean":>9}  {"median":>9}  '
        f'{"p25":>9}  {"p75":>9}')
    for t in TIMEPOINTS_S:
        col = f'fav_at_{t}s_usd'
        v = df_merged[col].dropna()
        out(f'  t = {t/60:>5.1f}m   ${v.mean():>+7.2f}   ${v.median():>+7.2f}   '
            f'${v.quantile(0.25):>+7.2f}   ${v.quantile(0.75):>+7.2f}')

    # === Hit rates ===
    out('')
    out('--- Hit rate at favorable R-multiple thresholds ---')
    out(f'  R = ${r_price * DOLLAR_PER_POINT:.2f} (= 4×ATR, the indicator threshold)')
    out(f'  {"threshold":<10}  {"hit %":>7}  {"time median":>12}  {"time p75":>10}')
    for m in R_MULTIPLES:
        hit_col = f'hit_{m:.1f}R'
        time_col = f'time_to_{m:.1f}R'
        n_hit = int(df_merged[hit_col].sum())
        pct = n_hit / len(df_merged) * 100
        time_med = df_merged[time_col].dropna().median()
        time_p75 = df_merged[time_col].dropna().quantile(0.75)
        out(f'  {m:.1f}R         {pct:>6.1f}%  '
            f'{time_med/60 if not np.isnan(time_med) else float("nan"):>10.1f}m  '
            f'{time_p75/60 if not np.isnan(time_p75) else float("nan"):>9.1f}m')

    # === Stratified by entry zone ===
    out('')
    out('--- BY ENTRY ZONE ---')
    out(f'  {"zone":<12}  {"n":>5}  {"%legs":>6}  {"med_amp":>9}  '
        f'{"hit_1R":>7}  {"hit_2R":>7}  {"hit_3R":>7}  '
        f'{"med_t_1R":>9}  {"whipsaw":>8}')
    for zone, sub in df_merged.groupby('entry_zone'):
        n = len(sub)
        med_amp = sub['final_amplitude_usd'].median()
        hit_1r = sub['hit_1.0R'].mean() * 100
        hit_2r = sub['hit_2.0R'].mean() * 100
        hit_3r = sub['hit_3.0R'].mean() * 100
        med_t_1r = sub['time_to_1.0R'].dropna().median() / 60 if sub['time_to_1.0R'].notna().any() else float('nan')
        whipsaw = sub['whipsaw_ratio'].median()
        out(f'  {zone:<12}  {n:>5}  {n/len(df_merged)*100:>5.1f}%  '
            f'${med_amp:>7.2f}  '
            f'{hit_1r:>6.1f}%  {hit_2r:>6.1f}%  {hit_3r:>6.1f}%  '
            f'{med_t_1r:>8.1f}m  {whipsaw:>7.2f}')

    out('')
    out('Reading the table:')
    out('  hit_1R%   = % of legs that reach favorable move >= 1R = 4×ATR')
    out('  hit_2R%   = % of legs that reach >= 2R = 8×ATR — "big winners"')
    out('  hit_3R%   = % of legs that reach >= 3R = 12×ATR — "monster legs"')
    out('  med_t_1R  = median time (minutes) from entry to first reach 1R favorable')
    out('  whipsaw   = median (max intra-leg DD) / (final amplitude)')

    # === Stratified by B6 directional match at entry ===
    out('')
    out('--- BY B6 DIRECTIONAL MATCH at entry ---')
    df_merged['b6_bucket'] = pd.cut(df_merged['entry_p_b6_match'],
                                      bins=[0, 0.30, 0.50, 0.70, 1.0],
                                      labels=['<0.30', '0.30-0.50',
                                              '0.50-0.70', '>=0.70'])
    out(f'  {"bucket":<12}  {"n":>5}  {"med_amp":>9}  '
        f'{"hit_1R":>7}  {"hit_2R":>7}  {"hit_3R":>7}  '
        f'{"med_t_1R":>9}  {"whipsaw":>8}')
    for b, sub in df_merged.groupby('b6_bucket', observed=True):
        if len(sub) == 0:
            continue
        med_amp = sub['final_amplitude_usd'].median()
        hit_1r = sub['hit_1.0R'].mean() * 100
        hit_2r = sub['hit_2.0R'].mean() * 100
        hit_3r = sub['hit_3.0R'].mean() * 100
        med_t_1r = sub['time_to_1.0R'].dropna().median() / 60 if sub['time_to_1.0R'].notna().any() else float('nan')
        whipsaw = sub['whipsaw_ratio'].median()
        out(f'  {str(b):<12}  {len(sub):>5}  ${med_amp:>7.2f}  '
            f'{hit_1r:>6.1f}%  {hit_2r:>6.1f}%  {hit_3r:>6.1f}%  '
            f'{med_t_1r:>8.1f}m  {whipsaw:>7.2f}')

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.report}')


if __name__ == '__main__':
    main()
