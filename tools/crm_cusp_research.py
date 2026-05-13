"""CRM-cusp edge research — does |z| local-maximum predict reversion?

Question (user, 2026-05-10):
    "Enter at the cusp of the CRM, since we can calculate based on probability
     if it's at max and ready to return to CRM"

Cusp definition (no lookahead):
    At each 1m close t, look back 2 bars. If
        |z[t-2]| < |z[t-1]|   AND   |z[t-1]| > |z[t]|
    then bar t-1 was a |z| local max. We know this at t (one-bar lag).
    Reversion direction = -sign(z[t-1]).

Comparison signals at the same 1m closes:
    A. "z above threshold" (the current NMP seed at z_thr=1.8) — no cusp
    B. "cusp confirmed" at the same z bucket
    C. "cusp confirmed AND |z[t-1]| in band [lo, hi]" (validated NMP retune band)

Forward return (signed by reversion direction):
    fwd_5m, fwd_15m, fwd_30m, fwd_60m — close-to-close at the 1m grid in ticks
    1 MNQ tick = 0.25, tick value $0.50 → ticks * $0.50 = $.

Output: reports/findings/cusp_research/
    summary.csv     — per (signal, z-bucket, horizon) cell: n, hit-rate, mean ticks
    cusp_vs_threshold_h15m_z_band_1.5_1.8.png
    pf_wr_by_bucket.png

Run:
    python -m tools.crm_cusp_research --is-only --tfs 1m --horizons 5,15,30,60
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'DATA' / 'ATLAS'
OUT_DIR = ROOT / 'reports' / 'findings' / 'cusp_research'

# IS / OOS split: same as 2D regime labels (60/20/20). For "honest cusp edge"
# we evaluate IS-only first; OOS is a second, separate run.
IS_END_DATE = '2025-10-31'   # IS ends here
OOS_START_DATE = '2026-01-01'   # OOS starts (Jan-Feb 2026)

Z_BUCKETS = [(0.8, 1.2), (1.2, 1.5), (1.5, 1.8), (1.8, 2.2),
                  (2.2, 2.8), (2.8, 3.5), (3.5, 99.0)]

# Tier 1: native-cadence horizons in 1m bars (5/15/30/60 minutes)
HORIZONS_MIN = [5, 15, 30, 60]


def list_days(split: str) -> List[str]:
    days = sorted(p.stem for p in (DATA_DIR / '5s').glob('*.parquet'))
    if split == 'is':
        return [d for d in days if d.replace('_', '-') <= IS_END_DATE]
    if split == 'oos':
        return [d for d in days if d.replace('_', '-') >= OOS_START_DATE]
    return days


def load_day_1m(day_str: str) -> pd.DataFrame:
    """Load 5s close + 1m z_se; downsample to 1m close grid."""
    ohlc_path = DATA_DIR / '5s' / f'{day_str}.parquet'
    z_path = DATA_DIR / 'FEATURES_5s_v2' / 'L3_1m' / f'{day_str}.parquet'
    if not (ohlc_path.exists() and z_path.exists()):
        return pd.DataFrame()

    ohlc = pd.read_parquet(ohlc_path, columns=['timestamp', 'close'])
    z = pd.read_parquet(z_path, columns=['timestamp', 'L3_1m_z_se_15'])
    df = ohlc.merge(z, on='timestamp', how='inner')

    # Downsample to 1m close boundaries (timestamp % 60 == 0 AND last 5s of that minute)
    df['minute'] = df['timestamp'] // 60
    last_idx = df.groupby('minute')['timestamp'].idxmax()
    df_1m = df.loc[last_idx].sort_values('timestamp').reset_index(drop=True)
    df_1m = df_1m.rename(columns={'L3_1m_z_se_15': 'z'})
    return df_1m[['timestamp', 'close', 'z']]


def label_signals_and_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns:
        is_thr  — |z[t]| >= 1.5 (NMP seed, current bar)
        is_cusp — |z[t-1]| was a local max (confirmed at t)
        z_at_decision — for thr: z[t]; for cusp: z[t-1]
        dir_rev — reversion direction (+1 LONG / -1 SHORT) based on z_at_decision
        fwd_{h}m_signed — close-to-close return in ticks signed by dir_rev
    """
    df = df.copy()
    z = df['z'].values
    n = len(z)

    # Cusp: confirmed local-max at t-1 (compare t-2, t-1, t)
    abs_z = np.abs(z)
    is_cusp = np.zeros(n, dtype=bool)
    is_cusp[2:] = (abs_z[1:-1] > abs_z[:-2]) & (abs_z[1:-1] > abs_z[2:])
    df['is_cusp'] = is_cusp
    df['z_at_cusp'] = np.where(is_cusp, np.roll(z, 1), np.nan)

    # Threshold: current |z| >= 1.5 (any growing-z entry, no cusp check)
    df['is_thr'] = abs_z >= 1.5

    # Convert prices to ticks (4 ticks = $1, tick = 0.25)
    close_t = df['close'].values
    for h in HORIZONS_MIN:
        fwd_close = np.full(n, np.nan)
        fwd_close[:-h] = close_t[h:]
        ret_ticks = (fwd_close - close_t) / 0.25
        # Sign by reversion direction at the decision time
        # For cusp: dir_rev = -sign(z_at_cusp). For thr: dir_rev = -sign(z[t]).
        sign_cusp = -np.sign(df['z_at_cusp'].fillna(0).values)
        sign_thr = -np.sign(z)
        df[f'fwd_{h}m_cusp_ticks'] = np.where(is_cusp, ret_ticks * sign_cusp, np.nan)
        df[f'fwd_{h}m_thr_ticks'] = np.where(df['is_thr'].values, ret_ticks * sign_thr, np.nan)

    return df


def bucket_z(z: float) -> str:
    az = abs(z)
    for lo, hi in Z_BUCKETS:
        if lo <= az < hi:
            return f'[{lo:.1f},{hi:.1f})'
    return 'oob'


def pf_wr(returns: np.ndarray) -> float:
    """Profit-factor-based WR per CLAUDE.md: (sum_winners / |sum_losers|) - 1."""
    r = returns[~np.isnan(returns)]
    if len(r) == 0:
        return np.nan
    win = r[r > 0].sum()
    loss = -r[r < 0].sum()
    if loss == 0:
        return np.inf if win > 0 else np.nan
    return win / loss - 1.0


def summarize(all_days: pd.DataFrame) -> pd.DataFrame:
    """Build cell-by-cell summary across (signal, z-bucket, horizon)."""
    rows = []
    for signal in ['cusp', 'thr']:
        mask_col = 'is_cusp' if signal == 'cusp' else 'is_thr'
        z_col = 'z_at_cusp' if signal == 'cusp' else 'z'
        sub = all_days[all_days[mask_col]].copy()
        if len(sub) == 0:
            continue
        sub['z_bucket'] = sub[z_col].apply(bucket_z)
        for bucket, grp in sub.groupby('z_bucket'):
            for h in HORIZONS_MIN:
                col = f'fwd_{h}m_{signal}_ticks'
                r = grp[col].dropna().values
                if len(r) == 0:
                    continue
                rows.append({
                    'signal': signal,
                    'z_bucket': bucket,
                    'horizon_min': h,
                    'n': len(r),
                    'hit_rate': float((r > 0).mean()),
                    'mean_ticks': float(r.mean()),
                    'median_ticks': float(np.median(r)),
                    'mean_dollars': float(r.mean() * 0.50),
                    'pf_wr': pf_wr(r),
                    'q25_ticks': float(np.percentile(r, 25)),
                    'q75_ticks': float(np.percentile(r, 75)),
                })
    return pd.DataFrame(rows)


def _sort_buckets(buckets):
    return sorted([b for b in buckets if b != 'oob'],
                       key=lambda b: float(b.strip('[').split(',')[0]))


def plot_pf_wr_by_bucket(summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, len(HORIZONS_MIN), figsize=(4 * len(HORIZONS_MIN), 5), sharey=True)
    for ax, h in zip(axes, HORIZONS_MIN):
        sub = summary[summary['horizon_min'] == h]
        buckets = _sort_buckets(sub['z_bucket'].unique())
        x = np.arange(len(buckets))
        w = 0.4
        cusp = sub[sub['signal'] == 'cusp'].set_index('z_bucket').reindex(buckets)
        thr = sub[sub['signal'] == 'thr'].set_index('z_bucket').reindex(buckets)
        ax.bar(x - w/2, cusp['pf_wr'], w, label='cusp', color='#2e7d32')
        ax.bar(x + w/2, thr['pf_wr'], w, label='thr only', color='#777')
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=45, ha='right', fontsize=8)
        ax.axhline(0, color='black', lw=0.6)
        ax.set_title(f'h={h}m')
        ax.grid(True, alpha=0.3)
        if h == HORIZONS_MIN[0]:
            ax.set_ylabel('PF-WR = sum_W/|sum_L| - 1')
            ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('PF-WR by |z| bucket — cusp-confirmed vs threshold-only')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def plot_n_by_bucket(summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, len(HORIZONS_MIN), figsize=(4 * len(HORIZONS_MIN), 5), sharey=True)
    for ax, h in zip(axes, HORIZONS_MIN):
        sub = summary[summary['horizon_min'] == h]
        buckets = _sort_buckets(sub['z_bucket'].unique())
        x = np.arange(len(buckets))
        w = 0.4
        cusp = sub[sub['signal'] == 'cusp'].set_index('z_bucket').reindex(buckets)
        thr = sub[sub['signal'] == 'thr'].set_index('z_bucket').reindex(buckets)
        ax.bar(x - w/2, cusp['n'], w, label='cusp', color='#2e7d32')
        ax.bar(x + w/2, thr['n'], w, label='thr only', color='#777')
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'h={h}m  N samples')
        ax.grid(True, alpha=0.3)
        if h == HORIZONS_MIN[0]:
            ax.set_ylabel('N')
            ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('Sample count by |z| bucket — cusp signal is rarer than threshold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['is', 'oos', 'both'], default='both')
    ap.add_argument('--max-days', type=int, default=0,
                    help='Cap days per split for speed; 0 = all')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = ['is', 'oos'] if args.split == 'both' else [args.split]

    for split in splits:
        days = list_days(split)
        if args.max_days:
            days = days[:args.max_days]
        if not days:
            print(f'[{split}] no days found, skipping')
            continue

        per_day_dfs = []
        for d in tqdm(days, desc=f'cusp/{split}'):
            df = load_day_1m(d)
            if df.empty:
                continue
            df = label_signals_and_forward(df)
            df['day'] = d
            per_day_dfs.append(df)

        if not per_day_dfs:
            continue
        all_df = pd.concat(per_day_dfs, ignore_index=True)
        summary = summarize(all_df)
        summary = summary.sort_values(['signal', 'horizon_min', 'z_bucket']).reset_index(drop=True)
        summary.to_csv(OUT_DIR / f'summary_{split}.csv', index=False)
        print(f'[{split}] {len(all_df)} 1m bars, {all_df["is_cusp"].sum()} cusps, '
                  f'{all_df["is_thr"].sum()} thr-only')

        plot_pf_wr_by_bucket(summary, OUT_DIR / f'pf_wr_by_bucket_{split}.png')
        plot_n_by_bucket(summary, OUT_DIR / f'n_by_bucket_{split}.png')

        # Top-line numbers for quick scan
        print(f'\n=== {split.upper()} TOP-LINE ===')
        for h in HORIZONS_MIN:
            cusp_band = summary[(summary['signal'] == 'cusp')
                                          & (summary['horizon_min'] == h)
                                          & (summary['z_bucket'] == '[1.5,1.8)')]
            thr_band = summary[(summary['signal'] == 'thr')
                                         & (summary['horizon_min'] == h)
                                         & (summary['z_bucket'] == '[1.5,1.8)')]
            if not cusp_band.empty and not thr_band.empty:
                c = cusp_band.iloc[0]
                t = thr_band.iloc[0]
                print(f'  h={h}m  [1.5,1.8) | cusp: n={c["n"]:5d} pf_wr={c["pf_wr"]:+.3f} '
                          f'mean=${c["mean_dollars"]:+.2f}'
                          f' | thr: n={t["n"]:5d} pf_wr={t["pf_wr"]:+.3f} '
                          f'mean=${t["mean_dollars"]:+.2f}')

    print(f'\nOutputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
