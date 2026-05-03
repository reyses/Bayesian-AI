"""
v2_features_volume_variation_eda.py — Layer 2D-vol: volume × variation states.

User: "what about volume and the other features, they must describe something"
User: "how about high variation in a low volume state?"

The prior layers ranked by Cohen-d on regime separation or |corr| with past
price moves. Volume features (vol_velocity_1b, vol_mean_w, vol_sigma_w,
vol_accel_w, ...) didn't make those shortlists because they DON'T correlate
with price velocity — and that's exactly the point. They describe activity
intensity, not direction.

This tool characterizes the 4 quadrants of (volume × variation):

  HIGH_VOL × HIGH_VAR : active directional movement OR capitulation
  HIGH_VOL × LOW_VAR  : compression / absorption — often precedes breakout
  LOW_VOL × HIGH_VAR  : noise / fakeout — moves without participation
  LOW_VOL × LOW_VAR   : dead zone — quiet absorption or transition

For each pair (vol_feature, var_feature) at the same TF:
  - Bin both into quantiles (default 3 → 9 cells)
  - Per cell: regime_2d distribution, n bars, mean fwd return, WR,
              mean MFE, mean MAE
  - Highlight the 4 corner cells specifically

Volume measures available (per TF):
  L1_<tf>_vol_velocity_1b   bar-to-bar Δvolume
  L1_<tf>_vol_accel_1b      d(vol_velocity)/db
  L2_<tf>_vol_mean_w        rolling mean volume (intensity proxy)
  L2_<tf>_vol_sigma_w       rolling std volume (volume volatility)
  L2_<tf>_vol_velocity_w    smoothed volume momentum
  L2_<tf>_vol_accel_w       smoothed volume acceleration

Variation measures (per TF):
  L1_<tf>_bar_range         (high - low) per bar
  L2_<tf>_price_sigma_w     rolling std of price
  L3_<tf>_swing_noise_w     max draw / range proxy

Outputs:
  reports/findings/v2_volume_variation/
    pair_summary.csv           — per (vol_feat, var_feat, tf) row
    corners.csv                — 4-corner cell stats for each pair
    top_pairs/<slug>.csv       — full 9-cell tables for top pairs
    top_pairs/<slug>.png       — heatmap visualizations
    summary.md
"""

from __future__ import annotations
import argparse
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    TF_HIERARCHY_V2, load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import (
    load_regime_labels, REGIME_2D_ORDER,
)


DEFAULT_BASE_TF = '5m'
DEFAULT_FORWARD_N = 12
DEFAULT_QUANTILES = 3


# Per-TF feature templates (matches features_v2 schema)
def vol_features_for_tf(tf: str) -> list[str]:
    """All volume-related features for one TF."""
    return [
        f'L1_{tf}_vol_velocity_1b',
        f'L1_{tf}_vol_accel_1b',
        f'L2_{tf}_vol_mean_w',
        f'L2_{tf}_vol_sigma_w',
        f'L2_{tf}_vol_velocity_w',
        f'L2_{tf}_vol_accel_w',
    ]


def var_features_for_tf(tf: str) -> list[str]:
    """All variation/dispersion-related features for one TF."""
    return [
        f'L1_{tf}_bar_range',
        f'L2_{tf}_price_sigma_w',
        f'L3_{tf}_swing_noise_w',
    ]


def quantile_bins(values: np.ndarray, q: int) -> np.ndarray:
    valid = ~np.isnan(values)
    out = np.full(len(values), -1, dtype=np.int8)
    if valid.sum() < q * 5:
        return out
    qs = np.quantile(values[valid], np.linspace(0, 1, q + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    bin_idx = np.digitize(values[valid], qs[1:-1])
    out[valid] = bin_idx.astype(np.int8)
    return out


def analyze_vol_var_pair(vol: np.ndarray, var: np.ndarray, regimes: np.ndarray,
                          fwd: np.ndarray, q: int) -> tuple[pd.DataFrame, dict]:
    """Per-cell stats for one (vol, var) pair.

    Returns (cells_df, corner_dict) where corner_dict has the 4 corners.
    """
    bv = quantile_bins(vol, q)
    bx = quantile_bins(var, q)
    valid = (bv >= 0) & (bx >= 0) & ~np.isnan(fwd)
    if valid.sum() < q * q * 30:
        return pd.DataFrame(), {}

    cells = []
    for iv in range(q):
        for ix in range(q):
            mask = valid & (bv == iv) & (bx == ix)
            n = int(mask.sum())
            if n == 0:
                continue
            cell_regimes = regimes[mask]
            regime_counts = pd.Series(cell_regimes).value_counts()
            regime_pcts = regime_counts / n
            dominant_regime = regime_counts.idxmax()
            f = fwd[mask]
            cells.append({
                'vol_q': iv, 'var_q': ix, 'n': n,
                'dominant_regime': dominant_regime,
                'dominant_pct': float(regime_pcts.iloc[0]),
                'mean_fwd': float(f.mean()),
                'std_fwd': float(f.std(ddof=1)) if n > 1 else 0.0,
                'win_rate': float((f > 0).mean()),
                **{f'pct_{r}': float(regime_pcts.get(r, 0.0))
                   for r in REGIME_2D_ORDER},
            })
    cells_df = pd.DataFrame(cells)

    # 4 corners: (LOW=0, HIGH=q-1)
    high_q = q - 1
    corners = {}
    for label, (vq, xq) in {
        'LOW_VOL_LOW_VAR':   (0, 0),
        'LOW_VOL_HIGH_VAR':  (0, high_q),
        'HIGH_VOL_LOW_VAR':  (high_q, 0),
        'HIGH_VOL_HIGH_VAR': (high_q, high_q),
    }.items():
        row = cells_df[(cells_df['vol_q'] == vq) & (cells_df['var_q'] == xq)]
        if len(row) > 0:
            corners[label] = row.iloc[0].to_dict()
    return cells_df, corners


def plot_vol_var_heatmap(cells_df: pd.DataFrame, vol_name: str, var_name: str,
                          q: int, out_path: str):
    """Two heatmaps: mean_fwd and dominant regime concentration."""
    if len(cells_df) == 0:
        return
    grid_fwd = np.full((q, q), np.nan)
    grid_pct = np.full((q, q), np.nan)
    grid_n = np.zeros((q, q), dtype=int)
    grid_regime = np.full((q, q), '', dtype=object)
    for _, c in cells_df.iterrows():
        v, x = int(c['vol_q']), int(c['var_q'])
        # Y axis = volume (low at bottom), X axis = variation (low at left)
        grid_fwd[v, x] = c['mean_fwd']
        grid_pct[v, x] = c['dominant_pct']
        grid_n[v, x] = c['n']
        grid_regime[v, x] = c['dominant_regime']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # Panel A: forward return
    vmax = np.nanmax(np.abs(grid_fwd)) if not np.all(np.isnan(grid_fwd)) else 1
    im0 = axes[0].imshow(grid_fwd, cmap='RdBu_r', origin='lower',
                          vmin=-vmax, vmax=vmax, aspect='auto')
    for v in range(q):
        for x in range(q):
            if np.isnan(grid_fwd[v, x]):
                continue
            axes[0].text(x, v,
                          f"fwd={grid_fwd[v, x]:+.1f}\nn={grid_n[v, x]}\n"
                          f"WR={cells_df[(cells_df.vol_q==v) & (cells_df.var_q==x)].win_rate.iloc[0]:.0%}",
                          ha='center', va='center', fontsize=8,
                          color='white' if abs(grid_fwd[v, x]) > vmax / 2 else 'black')
    axes[0].set_xticks(range(q))
    axes[0].set_yticks(range(q))
    axes[0].set_xticklabels([f'Q{i}' for i in range(q)])
    axes[0].set_yticklabels([f'Q{i}' for i in range(q)])
    axes[0].set_xlabel(f'{var_name}  (variation quantile)')
    axes[0].set_ylabel(f'{vol_name}  (volume quantile)')
    axes[0].set_title(f'Mean forward return per cell')
    plt.colorbar(im0, ax=axes[0], label='mean fwd return')

    # Panel B: dominant regime label
    im1 = axes[1].imshow(grid_pct, cmap='viridis', origin='lower',
                          vmin=0.16, vmax=1.0, aspect='auto')
    for v in range(q):
        for x in range(q):
            if grid_regime[v, x] == '':
                continue
            r = grid_regime[v, x]
            pct = grid_pct[v, x]
            axes[1].text(x, v, f"{r}\n{pct:.0%}", ha='center', va='center',
                          fontsize=8,
                          color='white' if pct > 0.5 else 'black')
    axes[1].set_xticks(range(q))
    axes[1].set_yticks(range(q))
    axes[1].set_xticklabels([f'Q{i}' for i in range(q)])
    axes[1].set_yticklabels([f'Q{i}' for i in range(q)])
    axes[1].set_xlabel(f'{var_name}  (variation quantile)')
    axes[1].set_ylabel(f'{vol_name}  (volume quantile)')
    axes[1].set_title(f'Dominant regime per cell + concentration')
    plt.colorbar(im1, ax=axes[1], label='dominant_pct')

    fig.suptitle(f'{vol_name}  x  {var_name}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default=DEFAULT_BASE_TF)
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--tfs', nargs='+',
                        default=['5m', '15m', '1h', '4h'],
                        help='Which TFs to consider for vol×var pairs')
    parser.add_argument('--quantiles', type=int, default=DEFAULT_QUANTILES)
    parser.add_argument('--forward-n', type=int, default=DEFAULT_FORWARD_N)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--top-pairs-to-plot', type=int, default=10)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_volume_variation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    top_dir = os.path.join(args.output_dir, 'top_pairs')
    os.makedirs(top_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"  V2 Volume x Variation EDA")
    print(f"  Base TF: {args.base_tf}  Split: {args.split}")
    print(f"  TFs to consider: {args.tfs}")
    print(f"  Quantiles: {args.quantiles}")
    print(f"{'='*70}")

    # Build feature pairs (same TF)
    pairs = []
    for tf in args.tfs:
        for vol, var in product(vol_features_for_tf(tf), var_features_for_tf(tf)):
            pairs.append((tf, vol, var))
    print(f"\n  Total (vol, var) pairs to analyze: {len(pairs)}")

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

    close = full['close'].values.astype(np.float64)
    n = len(close)
    fwd = np.full(n, np.nan)
    if n > args.forward_n:
        fwd[:-args.forward_n] = close[args.forward_n:] - close[:-args.forward_n]
    regimes = full['regime_2d'].values.astype(str)

    # Iterate pairs
    pair_rows = []
    corner_rows = []
    cell_dfs = {}

    for (tf, vol, var) in pairs:
        if vol not in full.columns or var not in full.columns:
            continue
        v = full[vol].values.astype(np.float64)
        x = full[var].values.astype(np.float64)
        cells_df, corners = analyze_vol_var_pair(v, x, regimes, fwd, args.quantiles)
        if len(cells_df) == 0:
            continue
        cell_dfs[(tf, vol, var)] = cells_df

        # Pair-level summary
        max_abs_fwd = float(cells_df['mean_fwd'].abs().max())
        max_pct = float(cells_df['dominant_pct'].max())
        wr_spread = float(cells_df['win_rate'].max() - cells_df['win_rate'].min())
        pair_rows.append({
            'tf': tf, 'vol_feature': vol, 'var_feature': var,
            'max_abs_mean_fwd': max_abs_fwd,
            'max_dominant_pct': max_pct,
            'wr_spread': wr_spread,
        })

        # Corner stats
        for label, c in corners.items():
            corner_rows.append({
                'tf': tf, 'vol_feature': vol, 'var_feature': var,
                'corner': label,
                'n': c['n'],
                'dominant_regime': c['dominant_regime'],
                'dominant_pct': c['dominant_pct'],
                'mean_fwd': c['mean_fwd'],
                'win_rate': c['win_rate'],
            })

    pair_df = pd.DataFrame(pair_rows)
    pair_path = os.path.join(args.output_dir, 'pair_summary.csv')
    pair_df.to_csv(pair_path, index=False)
    print(f"\n  [saved] {pair_path}")

    corner_df = pd.DataFrame(corner_rows)
    corner_path = os.path.join(args.output_dir, 'corners.csv')
    corner_df.to_csv(corner_path, index=False)
    print(f"  [saved] {corner_path}")

    # Print: 4 corners summary across all pairs
    print(f"\n--- 4 corners — what each describes ---")
    for label in ['LOW_VOL_HIGH_VAR', 'HIGH_VOL_LOW_VAR',
                   'HIGH_VOL_HIGH_VAR', 'LOW_VOL_LOW_VAR']:
        sub = corner_df[corner_df['corner'] == label]
        if len(sub) == 0:
            continue
        # Aggregate: which regime tends to dominate this corner?
        regime_tally = sub['dominant_regime'].value_counts()
        avg_fwd = sub['mean_fwd'].mean()
        avg_wr = sub['win_rate'].mean()
        avg_pct = sub['dominant_pct'].mean()
        print(f"\n  {label} (across {len(sub)} (vol,var) pairs):")
        print(f"    Avg mean_fwd: {avg_fwd:+.2f}  Avg WR: {avg_wr:.1%}  "
              f"Avg dominant_pct: {avg_pct:.0%}")
        print(f"    Most-common dominant regimes: "
              f"{', '.join(f'{r}({c})' for r, c in regime_tally.head(3).items())}")

    # Specific drilldown: LOW_VOL × HIGH_VAR — the user's question
    print(f"\n--- LOW_VOL x HIGH_VAR drilldown (per pair) ---")
    lvhv = corner_df[corner_df['corner'] == 'LOW_VOL_HIGH_VAR'] \
        .sort_values('mean_fwd', ascending=False).head(15)
    print(f"  {'tf':>3}  {'vol':>30}  {'var':>30}  {'n':>5}  "
          f"{'regime':>14}  {'pct':>5}  {'fwd':>7}  {'WR':>5}")
    for _, r in lvhv.iterrows():
        vol_short = r['vol_feature'].split('_', 1)[1]
        var_short = r['var_feature'].split('_', 1)[1]
        print(f"  {r['tf']:>3}  {vol_short[:30]:>30}  {var_short[:30]:>30}  "
              f"{r['n']:>5}  {r['dominant_regime']:>14}  "
              f"{r['dominant_pct']:>5.0%}  {r['mean_fwd']:>+7.2f}  "
              f"{r['win_rate']:>5.0%}")

    print(f"\n--- HIGH_VOL x LOW_VAR drilldown (compression candidates) ---")
    hvlv = corner_df[corner_df['corner'] == 'HIGH_VOL_LOW_VAR'] \
        .sort_values('mean_fwd', ascending=False).head(15)
    for _, r in hvlv.iterrows():
        vol_short = r['vol_feature'].split('_', 1)[1]
        var_short = r['var_feature'].split('_', 1)[1]
        print(f"  {r['tf']:>3}  {vol_short[:30]:>30}  {var_short[:30]:>30}  "
              f"{r['n']:>5}  {r['dominant_regime']:>14}  "
              f"{r['dominant_pct']:>5.0%}  {r['mean_fwd']:>+7.2f}  "
              f"{r['win_rate']:>5.0%}")

    # Save top pairs (CSV + heatmap)
    top_pairs = pair_df.sort_values('max_abs_mean_fwd', ascending=False) \
        .head(args.top_pairs_to_plot)
    print(f"\n--- Top {args.top_pairs_to_plot} pairs by max-cell |fwd return| ---")
    saved = 0
    for _, r in top_pairs.iterrows():
        key = (r['tf'], r['vol_feature'], r['var_feature'])
        cells_df = cell_dfs.get(key)
        if cells_df is None or len(cells_df) == 0:
            continue
        slug = (f"{r['tf']}__{r['vol_feature']}__x__{r['var_feature']}").replace('/', '_')[:140]
        cells_df.to_csv(os.path.join(top_dir, f'{slug}.csv'), index=False)
        plot_vol_var_heatmap(cells_df, r['vol_feature'], r['var_feature'],
                              args.quantiles, os.path.join(top_dir, f'{slug}.png'))
        saved += 1
    print(f"  [saved] {saved} top pair grids in {top_dir}/")

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 Volume x Variation EDA — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Base TF:** `{args.base_tf}`  **Split:** `{args.split}`\n")
        f.write(f"**TFs analyzed:** {args.tfs}\n")
        f.write(f"**Quantiles:** {args.quantiles}\n")
        f.write(f"**Forward N:** {args.forward_n} bars\n\n")

        f.write("## 4-corner summary (averaged across all (vol, var) pairs)\n\n")
        f.write("| Corner | Avg mean_fwd | Avg WR | Avg dom. pct | Top regimes |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for label in ['LOW_VOL_HIGH_VAR', 'HIGH_VOL_LOW_VAR',
                       'HIGH_VOL_HIGH_VAR', 'LOW_VOL_LOW_VAR']:
            sub = corner_df[corner_df['corner'] == label]
            if len(sub) == 0:
                continue
            regime_tally = sub['dominant_regime'].value_counts().head(3)
            f.write(f"| {label} | {sub['mean_fwd'].mean():+.2f} | "
                    f"{sub['win_rate'].mean():.1%} | "
                    f"{sub['dominant_pct'].mean():.0%} | "
                    f"{', '.join(f'{r}({c})' for r, c in regime_tally.items())} |\n")

        f.write("\n## LOW_VOL x HIGH_VAR top 15 (your question — fakeout territory)\n\n")
        f.write(lvhv.to_string(index=False))
        f.write("\n\n## HIGH_VOL x LOW_VAR top 15 (compression / breakout candidates)\n\n")
        f.write(hvlv.to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
