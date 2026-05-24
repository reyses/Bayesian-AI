"""
v2_features_overlay_viz.py — Visual overlay of features against price.

User asked for a way to OVERLAY signals visually and observe interactions
that describe a state. This tool picks representative days from each 2D
regime and produces a stacked plot:

  Top panel    : 5m close with horizontal regime label
  Middle panel : top features as faint colored lines overlaid (z-scored)
  Bottom panel : "chord" features (top 3 from chord EDA) as bold traces

One PNG per (regime, day) into reports/findings/v2_features_overlay/.

Use it to literally scroll through the days and see which feature
combinations co-activate at notable price moves.

Usage:
  python tools/v2_features_overlay_viz.py
  python tools/v2_features_overlay_viz.py --days-per-regime 3
  python tools/v2_features_overlay_viz.py --features L1_5m_body L2_5m_price_velocity_w
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import (
    load_regime_labels, REGIME_2D_ORDER, REGIME_2D_COLORS,
)
from tools.v2_features_lookback_eda import load_shortlist


DEFAULT_BASE_TF = '5m'
DEFAULT_DAYS_PER_REGIME = 2
DEFAULT_TOP_K = 12
DEFAULT_CHORD_FEATURES = (
    'L1_4h_body', 'L1_4h_price_velocity_1b', 'L3_4h_z_low_w'
)


def pick_representative_days(merged: pd.DataFrame, regime: str,
                              n_days: int, seed: int = 42) -> list[str]:
    """Pick `n_days` from `regime` — sample uniformly from those available."""
    sub = merged[merged['regime_2d'] == regime]
    days = sorted(sub['date'].unique())
    if len(days) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(days) <= n_days:
        return days
    idx = rng.choice(len(days), size=n_days, replace=False)
    return [days[i] for i in sorted(idx)]


def zscore(v: np.ndarray) -> np.ndarray:
    mu = np.nanmean(v)
    sd = np.nanstd(v)
    if sd < 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sd


def plot_one_day(day_df: pd.DataFrame, top_features: list[str],
                  chord_features: list[str], regime: str,
                  out_path: str, base_tf: str):
    """Plot one day's price + feature overlay."""
    if len(day_df) == 0:
        return

    # Get x-axis as datetime
    ts = day_df['ts_int'].values.astype(np.int64)
    x_dates = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts]

    fig = plt.figure(figsize=(14, 10), facecolor='white')
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2.5, 2], hspace=0.05)

    # Panel 1: price + vwap references
    ax1 = fig.add_subplot(gs[0, 0])
    close = day_df['close'].values
    ax1.plot(x_dates, close, color='black', lw=1.2, label='close')
    for vc, c, ls in [(f'L2_{base_tf}_vwap_w', '#888', '-'),
                       ('L2_1h_vwap_w', '#1565c0', '-'),
                       ('L2_4h_vwap_w', '#d32f2f', '--')]:
        if vc in day_df.columns:
            v = day_df[vc].values
            if not np.all(np.isnan(v)):
                ax1.plot(x_dates, v, color=c, lw=0.9, ls=ls, alpha=0.7,
                          label=vc.replace('L2_', '').replace('_w', ''))
    ax1.set_ylabel('Price')
    ax1.set_title(f'{day_df["date"].iloc[0]}  —  regime: {regime}',
                    color=REGIME_2D_COLORS.get(regime, '#000'),
                    fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left', fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelbottom=False)

    # Panel 2: top features (faint, z-scored)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cmap = plt.cm.tab20
    for i, feat in enumerate(top_features):
        if feat not in day_df.columns:
            continue
        v = day_df[feat].values
        z = zscore(v)
        ax2.plot(x_dates, z, color=cmap(i % 20), lw=0.8, alpha=0.6,
                  label=feat.replace('L2_', '').replace('L1_', '').replace('L3_', ''))
    ax2.axhline(0, color='gray', lw=0.4)
    ax2.axhline(1, color='gray', lw=0.3, ls=':')
    ax2.axhline(-1, color='gray', lw=0.3, ls=':')
    ax2.set_ylabel('Top-12 features (z)')
    ax2.legend(loc='upper left', fontsize=6, ncol=4)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelbottom=False)

    # Panel 3: chord features (bold)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    chord_colors = ['#d32f2f', '#1565c0', '#7b1fa2']
    for i, feat in enumerate(chord_features):
        if feat not in day_df.columns:
            continue
        v = day_df[feat].values
        z = zscore(v)
        ax3.plot(x_dates, z, color=chord_colors[i % len(chord_colors)],
                  lw=1.6, alpha=0.9,
                  label=feat.replace('L2_', '').replace('L1_', '').replace('L3_', ''))
    ax3.axhline(0, color='gray', lw=0.4)
    ax3.axhline(1, color='gray', lw=0.3, ls=':')
    ax3.axhline(-1, color='gray', lw=0.3, ls=':')
    ax3.set_ylabel('Chord (z)')
    ax3.legend(loc='upper left', fontsize=8, ncol=3)
    ax3.grid(True, alpha=0.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))

    fig.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default=DEFAULT_BASE_TF)
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--layer1-dir',
                        default='reports/findings/v2_features_regime_eda')
    parser.add_argument('--rank-by', default='lookback_corr',
                        choices=['cohen_d', 'lookback_corr', 'forward_corr'])
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                        help='Number of top features to overlay (panel 2)')
    parser.add_argument('--chord-features', nargs='+',
                        default=list(DEFAULT_CHORD_FEATURES),
                        help='Three features to highlight in panel 3 (the chord)')
    parser.add_argument('--days-per-regime', type=int, default=DEFAULT_DAYS_PER_REGIME)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_overlay')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Features Overlay Viz — visual state characterization")
    print(f"  Base TF: {args.base_tf}  Split: {args.split}")
    print(f"  Top features: {args.top_k}  Chord: {args.chord_features}")
    print(f"  Days per regime: {args.days_per_regime}")
    print(f"{'='*70}")

    # Shortlist
    top_features = load_shortlist(args.layer1_dir, args.top_k, args.rank_by)

    # Load + merge
    print(f"\n--- Loading data ---")
    base_df = load_atlas_tf(args.data, args.base_tf)
    if pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
        ts_int = base_df['timestamp'].astype('int64') // 10**9
    else:
        ts_int = base_df['timestamp'].astype(np.int64)
    base_df = base_df.copy()
    base_df['ts_int'] = ts_int
    dt_la = pd.to_datetime(ts_int, unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
    base_df['date'] = dt_la.dt.date.astype(str)

    labels_df = load_regime_labels(args.labels_csv).copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]
    merged = base_df.merge(
        labels_df[['date', 'regime_2d', 'split']], on='date', how='inner')
    if args.split.upper() != 'ALL':
        merged = merged[merged['split'] == args.split.upper()].reset_index(drop=True)
    print(f"  After split={args.split}: {len(merged):,} bars")

    ts_int = merged['ts_int'].values.astype(np.int64)
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int.min()), int(ts_int.max())), verbose=False,
    )
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)

    print(f"\n--- Plotting representative days per regime ---")
    rng_seed = args.seed
    saved = 0
    for regime in REGIME_2D_ORDER:
        days = pick_representative_days(full, regime, args.days_per_regime,
                                          seed=rng_seed)
        rng_seed += 1
        for day in days:
            day_df = full[full['date'] == day].sort_values('ts_int').reset_index(drop=True)
            if len(day_df) < 50:
                print(f"  [skip] {regime} / {day}: only {len(day_df)} bars")
                continue
            slug = f"{regime}__{day.replace('-', '_')}"
            out_path = os.path.join(args.output_dir, f'{slug}.png')
            plot_one_day(day_df, top_features, args.chord_features,
                          regime, out_path, args.base_tf)
            print(f"  [saved] {out_path}")
            saved += 1

    print(f"\nDone. {saved} PNGs in {args.output_dir}/")


if __name__ == '__main__':
    main()
