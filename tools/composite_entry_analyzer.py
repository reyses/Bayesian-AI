"""Composite Entry Analyzer — does entry-time composite predict leg quality?

The trail-tightening and target-placement sims both failed because R-trigger
is structurally optimal for "exit X below peak" given current signal
precision. But the composite zones DO carry pivot-proximity information
(2-3x lift). Maybe that information can help at ENTRY rather than EXIT:

  Question: at the start of each leg (entry timestamp = pivot timestamp),
  what does the composite zone/B5/B6 say? Does it predict the leg's
  forward amplitude / P&L?

  If YES (correlation):
    -> Sizing simulator: bigger position on predicted-strong legs
    -> Entry-suppression: skip predicted-weak legs

  If NO (no correlation):
    -> Composite signals are NOT useful for entry decisions either
    -> The signals describe IN-LEG state, not entry-quality

Per-leg outputs:
  entry_zone, entry_P_b5_MID, entry_P_b6_LONG, entry_P_b6_SHORT
  leg_amplitude (peak - entry for LONG, entry - trough for SHORT, in $)
  leg_pnl_at_R = leg_amplitude - R (= baseline R-trigger P&L)

Stratify by entry signals → mean / median / win-rate per group.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--cloud',
                    default='reports/findings/regret_oracle/pivot_probability_cloud.parquet')
    ap.add_argument('--b5',
                    default='reports/findings/regret_oracle/b5_leg_phase_OOS_NT8.parquet')
    ap.add_argument('--b6',
                    default='reports/findings/regret_oracle/b6_proba_OOS_NT8.parquet')
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_entry_analyzer.csv')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/composite_entry_analyzer.txt')
    args = ap.parse_args()

    print('Loading inputs...')
    truth = pd.read_parquet(args.truth)
    cloud = pd.read_parquet(args.cloud)
    b5 = pd.read_parquet(args.b5)
    b6 = pd.read_parquet(args.b6)
    print(f'  truth {len(truth):,}  cloud {len(cloud):,}  b5 {len(b5):,}  b6 {len(b6):,}')

    rows = []
    for day in tqdm(sorted(truth['day'].unique()), desc='days'):
        bars1m_path = NT8_1M_DIR / f'{day}.parquet'
        bars5s_path = NT8_5S_DIR / f'{day}.parquet'
        if not bars1m_path.exists() or not bars5s_path.exists():
            continue
        bars1m = pd.read_parquet(bars1m_path).sort_values('timestamp').reset_index(drop=True)
        bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
        atr_pts = compute_atr(bars1m, 14)
        min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * TRAIN_ATR_MULT)))
        r_price = min_rev_ticks * TICK_SIZE

        truth_day = truth[truth['day'] == day]
        cloud_day = cloud[cloud['day'] == day].sort_values('timestamp').reset_index(drop=True)
        b5_day = b5[b5['day'] == day].sort_values('timestamp').reset_index(drop=True)
        b6_day = b6[b6['day'] == day].sort_values('timestamp').reset_index(drop=True)

        events = derive_pivot_events(truth_day)
        if len(events) < 2:
            continue

        cloud_ts = cloud_day['timestamp'].values.astype(np.int64)
        b5_ts = b5_day['timestamp'].values.astype(np.int64)
        b6_ts = b6_day['timestamp'].values.astype(np.int64)

        closes5s = bars5s['close'].values.astype(np.float64)
        ts5s = bars5s['timestamp'].values.astype(np.int64)

        K = args.K
        p_long_col  = f'p_PIVOT_TO_LONG_{K}m'
        p_short_col = f'p_PIVOT_TO_SHORT_{K}m'

        for k in range(len(events) - 1):
            entry_ts, leg_dir, entry_price = events[k]
            next_ts, _, next_price = events[k + 1]

            # Forward leg amplitude (truth)
            leg_mask = (ts5s >= entry_ts) & (ts5s <= next_ts)
            leg_closes = closes5s[leg_mask]
            if len(leg_closes) == 0:
                continue
            if leg_dir == 'LONG':
                leg_extreme = float(leg_closes.max())
            else:
                leg_extreme = float(leg_closes.min())
            leg_amplitude_pts = abs(leg_extreme - entry_price)
            leg_amplitude_usd = leg_amplitude_pts * DOLLAR_PER_POINT
            # P&L using R-trigger baseline = amplitude - R
            pnl_at_R_pts = leg_amplitude_pts - r_price
            pnl_at_R_usd = pnl_at_R_pts * DOLLAR_PER_POINT
            leg_duration_min = (next_ts - entry_ts) / 60.0

            # Entry-time composite snapshot — find the 1m bar at or just
            # after the pivot ts (since pivot may occur mid-1m-bar)
            ci = int(np.searchsorted(cloud_ts, entry_ts, side='left'))
            ci = min(max(ci, 0), len(cloud_ts) - 1)
            entry_zone = str(cloud_day['zone'].iloc[ci])
            entry_cloud_state = str(cloud_day['cloud_state'].iloc[ci]) \
                if 'cloud_state' in cloud_day.columns else 'NA'

            # Entry-time B5 (leg-phase probs)
            bi = int(np.searchsorted(b5_ts, entry_ts, side='left'))
            bi = min(max(bi, 0), len(b5_day) - 1)
            entry_p_mid   = float(b5_day['p_phase_MID'].iloc[bi])    if 'p_phase_MID'   in b5_day.columns else np.nan
            entry_p_early = float(b5_day['p_phase_EARLY'].iloc[bi])  if 'p_phase_EARLY' in b5_day.columns else np.nan
            entry_p_late  = float(b5_day['p_phase_LATE'].iloc[bi])   if 'p_phase_LATE'  in b5_day.columns else np.nan

            # Entry-time B6 directional
            b6i = int(np.searchsorted(b6_ts, entry_ts, side='left'))
            b6i = min(max(b6i, 0), len(b6_day) - 1)
            entry_p_b6_long  = float(b6_day[p_long_col].iloc[b6i])  if p_long_col  in b6_day.columns else np.nan
            entry_p_b6_short = float(b6_day[p_short_col].iloc[b6i]) if p_short_col in b6_day.columns else np.nan
            # B6 directional MATCH: did B6 predict our leg's direction?
            # If we just entered LONG leg, B6 P(PIVOT_TO_LONG) high = "we ARE coming off a LONG pivot" = good sign
            entry_p_b6_match = entry_p_b6_long if leg_dir == 'LONG' else entry_p_b6_short

            rows.append({
                'day': day,
                'entry_ts': entry_ts,
                'leg_dir': leg_dir,
                'entry_price': entry_price,
                'leg_amplitude_pts': leg_amplitude_pts,
                'leg_amplitude_usd': leg_amplitude_usd,
                'leg_duration_min': leg_duration_min,
                'r_price': r_price,
                'pnl_at_R_usd': pnl_at_R_usd,
                'entry_zone': entry_zone,
                'entry_cloud_state': entry_cloud_state,
                'entry_p_mid': entry_p_mid,
                'entry_p_early': entry_p_early,
                'entry_p_late': entry_p_late,
                'entry_p_b6_long': entry_p_b6_long,
                'entry_p_b6_short': entry_p_b6_short,
                'entry_p_b6_match': entry_p_b6_match,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f'\nWrote: {args.out}  ({len(df):,} legs)')

    # === Report ===
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('COMPOSITE ENTRY ANALYZER — does entry-time signal predict leg quality?')
    out('=' * 78)
    out(f'Legs: {len(df):,}   Days: {df["day"].nunique()}')
    out(f'R-trigger P&L per leg:   mean ${df["pnl_at_R_usd"].mean():+.2f}   '
        f'median ${df["pnl_at_R_usd"].median():+.2f}')
    out(f'Leg amplitude:           mean ${df["leg_amplitude_usd"].mean():.2f}   '
        f'median ${df["leg_amplitude_usd"].median():.2f}')
    out(f'  pos amplitude > R:     {(df["pnl_at_R_usd"] > 0).mean()*100:.1f}% of legs (profitable)')

    # Per entry_zone
    out('')
    out('--- BY ENTRY ZONE (composite at leg start) ---')
    out(f'  {"zone":<14}  {"n":>6}  {"% legs":>7}  '
        f'{"amp_mean":>10}  {"amp_median":>11}  '
        f'{"pnl_mean":>10}  {"pnl_median":>11}  {"%win":>6}')
    for zone, sub in df.groupby('entry_zone'):
        amp = sub['leg_amplitude_usd']
        pnl = sub['pnl_at_R_usd']
        out(f'  {zone:<14}  {len(sub):>6}  {len(sub)/len(df)*100:>6.1f}%  '
            f'${amp.mean():>9.2f}  ${amp.median():>10.2f}  '
            f'${pnl.mean():>+9.2f}  ${pnl.median():>+10.2f}  '
            f'{(pnl > 0).mean()*100:>5.1f}%')

    out('')
    out('--- BY B5 entry P(MID) bucket (proxy for "deep in trend") ---')
    df['b5_mid_bucket'] = pd.cut(df['entry_p_mid'], bins=[0, 0.2, 0.3, 0.4, 0.5, 1.0],
                                  labels=['<0.20', '0.20-0.30', '0.30-0.40', '0.40-0.50', '>=0.50'])
    out(f'  {"bucket":<12}  {"n":>6}  '
        f'{"amp_mean":>10}  {"pnl_mean":>10}  {"%win":>6}')
    for b, sub in df.groupby('b5_mid_bucket', observed=True):
        if len(sub) == 0: continue
        amp = sub['leg_amplitude_usd']; pnl = sub['pnl_at_R_usd']
        out(f'  {str(b):<12}  {len(sub):>6}  '
            f'${amp.mean():>9.2f}  ${pnl.mean():>+9.2f}  '
            f'{(pnl > 0).mean()*100:>5.1f}%')

    # B6 directional match — if we entered LONG, was B6 P(PIVOT_TO_LONG) high?
    out('')
    out('--- BY B6 directional MATCH at entry ---')
    out('  (B6 P(LONG) at entry of a LONG leg, P(SHORT) at entry of SHORT leg)')
    df['b6_match_bucket'] = pd.cut(df['entry_p_b6_match'], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
                                    labels=['<0.30', '0.30-0.40', '0.40-0.50',
                                             '0.50-0.60', '0.60-0.70', '>=0.70'])
    out(f'  {"bucket":<12}  {"n":>6}  '
        f'{"amp_mean":>10}  {"pnl_mean":>10}  {"%win":>6}')
    for b, sub in df.groupby('b6_match_bucket', observed=True):
        if len(sub) == 0: continue
        amp = sub['leg_amplitude_usd']; pnl = sub['pnl_at_R_usd']
        out(f'  {str(b):<12}  {len(sub):>6}  '
            f'${amp.mean():>9.2f}  ${pnl.mean():>+9.2f}  '
            f'{(pnl > 0).mean()*100:>5.1f}%')

    # Correlation tests
    out('')
    out('--- CORRELATION: entry signal vs leg P&L ---')
    for col in ['entry_p_mid', 'entry_p_early', 'entry_p_late',
                'entry_p_b6_match', 'entry_p_b6_long', 'entry_p_b6_short']:
        sub = df.dropna(subset=[col, 'pnl_at_R_usd'])
        if len(sub) < 10:
            continue
        from scipy.stats import spearmanr, pearsonr
        try:
            rho_s, p_s = spearmanr(sub[col], sub['pnl_at_R_usd'])
            rho_p, p_p = pearsonr(sub[col], sub['pnl_at_R_usd'])
            out(f'  {col:<22}  Pearson r={rho_p:+.4f} (p={p_p:.3f})  '
                f'Spearman r={rho_s:+.4f} (p={p_s:.3f})')
        except Exception as e:
            out(f'  {col}: error {e}')

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.report}')


if __name__ == '__main__':
    main()
