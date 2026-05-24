"""
sigma_cascade.py -- Multi-TF z-score alignment detector ("resonance cascade")
=============================================================================

Operationalizes the "extreme sigma in all TF / energy buildup" hypothesis using
the precomputed features in DATA/ATLAS/FEATURES_5s/. No SFE recompute needed.

For each 5s timestamp the parquet already carries `<TF>_z_se` for TF in
{15s, 1m, 5m, 15m, 1h, 1D}. We:
  1. Mark each TF as STRETCHED-LONG (z_se > +threshold), STRETCHED-SHORT
     (z_se < -threshold), or NEUTRAL.
  2. Count alignment per direction:
        n_long_aligned  = #TFs simultaneously stretched LONG
        n_short_aligned = #TFs simultaneously stretched SHORT
        max_alignment   = max(n_long, n_short)
        cascade_dir     = direction with more votes ('LONG' / 'SHORT' / 'NEUTRAL')
  3. For each bar, measure forward 1m-close return over `--lookahead-minutes`
     in the cascade direction (= alignment-direction-signed price move).
  4. Aggregate WR / mean / median profit by alignment level, IS vs OOS.

Hypothesis: WR climbs monotonically with alignment count. If 5/6 alignment shows
e.g. 60%+ WR with positive mean, the cascade signal is real.

Usage:
    python tools/sigma_cascade.py                                 # all days, default
    python tools/sigma_cascade.py --threshold 2.0                 # default sigma threshold
    python tools/sigma_cascade.py --threshold 1.5 --lookahead-minutes 10
    python tools/sigma_cascade.py --tfs 5m 15m 1h 1D              # macro-only
    python tools/sigma_cascade.py --day 2026-02-09                # single day diagnostic
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_TFS = ['15s', '1m', '5m', '15m', '1h', '1D']
DEFAULT_THRESHOLD = 2.0
DEFAULT_LOOKAHEAD_MIN = 5

ATLAS_ROOT = 'DATA/ATLAS'
FEATURES_DIR = os.path.join(ATLAS_ROOT, 'FEATURES_5s')

# MNQ
DOLLAR_PER_POINT = 2.0


# ── Core ──────────────────────────────────────────────────────────────────

def load_features_day(features_dir: str, day_label: str) -> pd.DataFrame | None:
    p = os.path.join(features_dir, f'{day_label}.parquet')
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)


def load_1m_day(atlas_root: str, day_label: str) -> pd.DataFrame | None:
    p = os.path.join(atlas_root, '1m', f'{day_label}.parquet')
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)


def compute_alignment(df_features: pd.DataFrame,
                      tfs: list[str],
                      threshold: float) -> pd.DataFrame:
    """Per-row multi-TF z-alignment.

    Missing `<TF>_z_se` columns are silently treated as 0 (= not stretched), so the
    function tolerates feature-set mismatches without exploding.
    """
    n = len(df_features)
    if n == 0:
        return pd.DataFrame()

    z_cols = []
    z_arrays = []
    for tf in tfs:
        col = f'{tf}_z_se'
        if col in df_features.columns:
            z_arrays.append(df_features[col].values.astype(np.float64))
            z_cols.append(col)
        else:
            z_arrays.append(np.zeros(n, dtype=np.float64))
            z_cols.append(f'{col} (missing)')

    Z = np.stack(z_arrays, axis=1)                 # shape (n, n_tfs)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    abs_Z = np.abs(Z)

    n_long = (Z > threshold).sum(axis=1).astype(np.int32)
    n_short = (Z < -threshold).sum(axis=1).astype(np.int32)
    max_align = np.maximum(n_long, n_short).astype(np.int32)

    cascade_dir = np.where(n_long > n_short, 'LONG',
                           np.where(n_short > n_long, 'SHORT', 'NEUTRAL'))

    return pd.DataFrame({
        'timestamp': df_features['timestamp'].values.astype(np.int64),
        'n_long_aligned': n_long,
        'n_short_aligned': n_short,
        'max_alignment': max_align,
        'cascade_dir': cascade_dir,
        'max_abs_z': abs_Z.max(axis=1),
        'sum_abs_z': abs_Z.sum(axis=1),
    })


def attach_lookahead(alignment_df: pd.DataFrame,
                     df_1m: pd.DataFrame | None,
                     lookahead_seconds: int) -> pd.DataFrame:
    """Add forward-return columns. profit_pts = direction_sign * (future_close - current_close)."""
    if df_1m is None or len(df_1m) == 0 or alignment_df.empty:
        return alignment_df

    ts_1m = df_1m['timestamp'].values.astype(np.int64)
    closes_1m = df_1m['close'].values.astype(np.float64)
    align_ts = alignment_df['timestamp'].values.astype(np.int64)

    cur_idx = np.clip(np.searchsorted(ts_1m, align_ts, side='right') - 1, 0, len(closes_1m) - 1)
    fut_idx = np.clip(np.searchsorted(ts_1m, align_ts + lookahead_seconds, side='right') - 1,
                       0, len(closes_1m) - 1)

    cur_price = closes_1m[cur_idx]
    fut_price = closes_1m[fut_idx]
    ret_pts = fut_price - cur_price

    direction_sign = np.where(alignment_df['cascade_dir'].values == 'LONG', +1,
                       np.where(alignment_df['cascade_dir'].values == 'SHORT', -1, 0)).astype(np.int8)
    profit_pts = direction_sign * ret_pts

    out = alignment_df.copy()
    out['cur_price'] = cur_price
    out['fut_price'] = fut_price
    out['ret_pts'] = ret_pts
    out['direction_sign'] = direction_sign
    out['profit_pts'] = profit_pts
    out['profitable'] = (profit_pts > 0).astype(np.int8)
    return out


# ── Reporting ─────────────────────────────────────────────────────────────

def _is_2025(day: str) -> bool:  return day.startswith('2025_')
def _is_2026(day: str) -> bool:  return day.startswith('2026_')


def print_distribution(df: pd.DataFrame, n_tfs: int):
    print(f'\n=== ALIGNMENT DISTRIBUTION (|z_se| > threshold) ===')
    print(f'{"Aligned":>8} {"Bars":>12} {"% total":>10}')
    total = len(df)
    for k in range(n_tfs + 1):
        c = int((df['max_alignment'] == k).sum())
        print(f'{k:>8} {c:>12,} {100.0*c/total:>9.2f}%')


def print_predictive(df: pd.DataFrame, n_tfs: int, label: str, lookahead_min: int):
    print(f'\n=== PREDICTIVE — {label} (lookahead = {lookahead_min}m, direction-signed) ===')
    print(f'{"Align":>5} {"N":>10} {"WR%":>7} {"Mean$":>10} {"Median$":>10} {"%Long":>8} {"%Short":>8}')
    sub_all = df[df['direction_sign'] != 0]
    for k in range(1, n_tfs + 1):
        sub = sub_all[sub_all['max_alignment'] == k]
        if len(sub) == 0:
            continue
        wr = sub['profitable'].mean() * 100.0
        mean_pts = sub['profit_pts'].mean()
        median_pts = sub['profit_pts'].median()
        n_l = int((sub['cascade_dir'] == 'LONG').sum())
        n_s = int((sub['cascade_dir'] == 'SHORT').sum())
        print(f'{k:>5} {len(sub):>10,} {wr:>6.1f}% '
              f'${mean_pts*DOLLAR_PER_POINT:>+8.2f} '
              f'${median_pts*DOLLAR_PER_POINT:>+8.2f} '
              f'{100.0*n_l/len(sub):>7.1f}% {100.0*n_s/len(sub):>7.1f}%')


def main():
    ap = argparse.ArgumentParser(description='Multi-TF z-score alignment cascade detector')
    ap.add_argument('--features-dir', default=FEATURES_DIR)
    ap.add_argument('--atlas', default=ATLAS_ROOT)
    ap.add_argument('--tfs', nargs='+', default=DEFAULT_TFS,
                    help='TFs to use (default: 15s 1m 5m 15m 1h 1D)')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                    help='|z_se| above this counts as stretched (default 2.0)')
    ap.add_argument('--lookahead-minutes', type=int, default=DEFAULT_LOOKAHEAD_MIN,
                    help='Forward-return measurement window (default 5m)')
    ap.add_argument('--day', default=None,
                    help='Single-day diagnostic (YYYY-MM-DD); else processes all days')
    ap.add_argument('--out-prefix', default='reports/findings/sigma_cascade')
    args = ap.parse_args()

    if args.day:
        days = [args.day.replace('-', '_')]
    else:
        files = sorted(glob.glob(os.path.join(args.features_dir, '*.parquet')))
        days = [os.path.splitext(os.path.basename(p))[0] for p in files]

    if not days:
        print(f'No feature parquets in {args.features_dir}')
        return

    print('=' * 80)
    print('SIGMA CASCADE — multi-TF z-score alignment')
    print(f'Features dir: {args.features_dir}')
    print(f'TFs: {args.tfs}  |  |z| threshold: {args.threshold}  |  lookahead: {args.lookahead_minutes}m')
    print(f'Days: {len(days)}')
    print('=' * 80)

    blocks = []
    iterator = days if len(days) == 1 else tqdm(days, desc='days')
    for day_label in iterator:
        df_feat = load_features_day(args.features_dir, day_label)
        if df_feat is None:
            continue
        df_1m = load_1m_day(args.atlas, day_label)
        align = compute_alignment(df_feat, args.tfs, args.threshold)
        align = attach_lookahead(align, df_1m, args.lookahead_minutes * 60)
        align['day'] = day_label
        blocks.append(align)

    if not blocks:
        print('No data loaded.')
        return

    df = pd.concat(blocks, ignore_index=True)
    n_tfs = len(args.tfs)

    print_distribution(df, n_tfs)

    if 'profit_pts' in df.columns:
        is_df = df[df['day'].apply(_is_2025)]
        oos_df = df[df['day'].apply(_is_2026)]
        if len(is_df):  print_predictive(is_df, n_tfs, 'IS (2025)', args.lookahead_minutes)
        if len(oos_df): print_predictive(oos_df, n_tfs, 'OOS (2026)', args.lookahead_minutes)

    # Save aggregate summary CSV
    out_csv = f'{args.out_prefix}_z{args.threshold}_la{args.lookahead_minutes}m.csv'
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary_rows = []
    for split_name, split_df in [('IS', df[df['day'].apply(_is_2025)]),
                                  ('OOS', df[df['day'].apply(_is_2026)])]:
        if len(split_df) == 0:
            continue
        for k in range(0, n_tfs + 1):
            sub = split_df[split_df['max_alignment'] == k]
            sub_dir = sub[sub['direction_sign'] != 0] if k > 0 else sub
            row = {
                'split':           split_name,
                'tfs':             ','.join(args.tfs),
                'threshold':       args.threshold,
                'lookahead_min':   args.lookahead_minutes,
                'alignment':       k,
                'n_bars':          int(len(sub)),
                'n_directional':   int(len(sub_dir)),
                'wr_pct':          float(sub_dir['profitable'].mean() * 100.0) if len(sub_dir) else 0.0,
                'mean_profit_pts': float(sub_dir['profit_pts'].mean()) if len(sub_dir) else 0.0,
                'median_profit_pts': float(sub_dir['profit_pts'].median()) if len(sub_dir) else 0.0,
                'mean_profit_usd': float(sub_dir['profit_pts'].mean() * DOLLAR_PER_POINT) if len(sub_dir) else 0.0,
                'n_long':          int((sub_dir['cascade_dir'] == 'LONG').sum()) if len(sub_dir) else 0,
                'n_short':         int((sub_dir['cascade_dir'] == 'SHORT').sum()) if len(sub_dir) else 0,
            }
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv}')


if __name__ == '__main__':
    main()
