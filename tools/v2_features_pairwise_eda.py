"""
v2_features_pairwise_eda.py — Layer 2 of the regime EDA stack.

For pairs of features taken from a curated shortlist (top-K from Layer 1),
characterize price behavior in each (X_quantile × Y_quantile) cell.
This answers: "when feature X is HIGH and feature Y is LOW, what does
price do?".

Pruning strategy (combinatorial explosion control):
  - Layer 1 (`v2_features_regime_eda.py`) produces a feature shortlist.
    By default we pick the top-K by max(|cohen_d|) across regime pairs OR
    by |corr_lookback_return|. Default K=20 → 190 pairs.
  - For each pair, compute per-cell (q_X × q_Y) stats. Default 3 quantiles.

Per (X_q, Y_q) cell, we compute:
  - n bars
  - mean & std of future_return (close[t+N] - close[t])
  - mean & std of concurrent dislocation (close - vwap_5m_w)
  - mean win rate (sign(future_return) > 0 fraction)

Joint-vs-additive structure:
  We compare the joint cell mean against the additive prediction
  (X_q marginal mean + Y_q marginal mean - global mean). The residual
  measures "interaction" — non-additive structure. Pairs with high
  interaction residuals carry information beyond the marginals.

Outputs:
  reports/findings/v2_features_pairwise_eda/
    pair_summary.csv         — per-pair: max cell return, max win-rate, interaction score
    top_pairs/<pairname>.csv — per-pair grid of cell stats (for the top pairs)
    top_pairs/<pairname>.png — heatmap of mean future return per cell
    summary.md               — narrative

Usage:
  python tools/v2_features_pairwise_eda.py
  python tools/v2_features_pairwise_eda.py --top-k 15 --quantiles 5
  python tools/v2_features_pairwise_eda.py --rank-by lookback_corr
"""

from __future__ import annotations
import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import load_regime_labels


DEFAULT_BASE_TF = '5m'
DEFAULT_FORWARD_N = 12
DEFAULT_QUANTILES = 3
DEFAULT_TOP_K = 20
DEFAULT_TOP_PAIRS_TO_PLOT = 12


def load_layer1_shortlist(layer1_dir: str, top_k: int, rank_by: str) -> list[str]:
    """Pick top-K features from Layer 1 outputs based on `rank_by`.

    rank_by:
      'cohen_d'        → max |cohen_d| across all regime pairs
      'lookback_corr'  → |corr_lookback_return|
      'forward_corr'   → |corr_forward_return|
      'concurrent_abs' → |corr_concurrent_abs|
    """
    sep_path = os.path.join(layer1_dir, 'regime_separation.csv')
    corr_path = os.path.join(layer1_dir, 'price_correlations.csv')

    if rank_by == 'cohen_d':
        sep = pd.read_csv(sep_path)
        ranking = sep.groupby('feature')['abs_d'].max().sort_values(ascending=False)
    elif rank_by == 'lookback_corr':
        corr = pd.read_csv(corr_path)
        corr['abs_lb'] = corr['corr_lookback_return'].abs()
        ranking = corr.set_index('feature')['abs_lb'].sort_values(ascending=False)
    elif rank_by == 'forward_corr':
        corr = pd.read_csv(corr_path)
        corr['abs_fw'] = corr['corr_forward_return'].abs()
        ranking = corr.set_index('feature')['abs_fw'].sort_values(ascending=False)
    elif rank_by == 'concurrent_abs':
        corr = pd.read_csv(corr_path)
        corr['abs_ca'] = corr['corr_concurrent_abs'].abs()
        ranking = corr.set_index('feature')['abs_ca'].sort_values(ascending=False)
    else:
        raise ValueError(f"unknown rank_by: {rank_by}")

    return ranking.head(top_k).index.tolist()


def quantile_bins(values: np.ndarray, q: int) -> np.ndarray:
    """Return integer quantile labels 0..q-1 for each value, NaN gets -1."""
    valid = ~np.isnan(values)
    out = np.full(len(values), -1, dtype=np.int8)
    if valid.sum() < q * 5:
        return out
    qs = np.quantile(values[valid], np.linspace(0, 1, q + 1))
    qs[0] -= 1e-9   # ensure leftmost included
    qs[-1] += 1e-9
    bin_idx = np.digitize(values[valid], qs[1:-1])
    out[valid] = bin_idx.astype(np.int8)
    return out


def analyze_pair(x: np.ndarray, y: np.ndarray, fwd: np.ndarray,
                  conc: np.ndarray, q: int) -> dict:
    """Per-cell stats + interaction score for one (X, Y) pair."""
    bx = quantile_bins(x, q)
    by = quantile_bins(y, q)
    valid = (bx >= 0) & (by >= 0) & ~np.isnan(fwd)
    if valid.sum() < q * q * 5:
        return {'cells': pd.DataFrame(), 'interaction': float('nan'),
                'max_abs_mean_fwd': float('nan'), 'max_win_rate': float('nan'),
                'min_win_rate': float('nan'), 'n_total': int(valid.sum())}

    cells = []
    global_fwd = fwd[valid].mean()

    # Marginal means (for additive baseline)
    marg_x = {bi: float(fwd[valid & (bx == bi)].mean())
              if (valid & (bx == bi)).sum() > 0 else global_fwd
              for bi in range(q)}
    marg_y = {bi: float(fwd[valid & (by == bi)].mean())
              if (valid & (by == bi)).sum() > 0 else global_fwd
              for bi in range(q)}

    interaction_sq_sum = 0.0
    n_cells_with_data = 0

    for ix in range(q):
        for iy in range(q):
            mask = valid & (bx == ix) & (by == iy)
            n = int(mask.sum())
            if n == 0:
                continue
            f = fwd[mask]
            c = conc[mask] if conc is not None else None
            wr = float((f > 0).mean()) if n else float('nan')
            mean_fwd = float(f.mean())
            mean_conc = float(c.mean()) if c is not None else float('nan')
            additive_pred = marg_x[ix] + marg_y[iy] - global_fwd
            interaction = mean_fwd - additive_pred
            interaction_sq_sum += interaction ** 2
            n_cells_with_data += 1
            cells.append({
                'x_q': ix, 'y_q': iy, 'n': n,
                'mean_fwd': mean_fwd, 'std_fwd': float(f.std(ddof=1)) if n > 1 else 0.0,
                'win_rate': wr,
                'mean_conc': mean_conc,
                'additive_pred': additive_pred,
                'interaction': interaction,
            })
    cells_df = pd.DataFrame(cells)

    interaction_rms = np.sqrt(interaction_sq_sum / max(n_cells_with_data, 1))
    return {
        'cells': cells_df,
        'interaction': float(interaction_rms),
        'max_abs_mean_fwd': float(cells_df['mean_fwd'].abs().max()) if len(cells_df) else float('nan'),
        'max_win_rate': float(cells_df['win_rate'].max()) if len(cells_df) else float('nan'),
        'min_win_rate': float(cells_df['win_rate'].min()) if len(cells_df) else float('nan'),
        'n_total': int(valid.sum()),
    }


def plot_pair_heatmap(cells_df: pd.DataFrame, x_name: str, y_name: str,
                       q: int, out_path: str):
    """Heatmap of mean_fwd per cell + win rate annotation."""
    if len(cells_df) == 0:
        return
    grid = np.full((q, q), np.nan)
    wr_grid = np.full((q, q), np.nan)
    n_grid = np.full((q, q), 0)
    for _, c in cells_df.iterrows():
        grid[int(c['y_q']), int(c['x_q'])] = c['mean_fwd']
        wr_grid[int(c['y_q']), int(c['x_q'])] = c['win_rate']
        n_grid[int(c['y_q']), int(c['x_q'])] = c['n']
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = np.nanmax(np.abs(grid)) if not np.all(np.isnan(grid)) else 1
    im = ax.imshow(grid, cmap='RdBu_r', origin='lower',
                    vmin=-vmax, vmax=vmax, aspect='auto')
    for iy in range(q):
        for ix in range(q):
            if np.isnan(grid[iy, ix]):
                continue
            ax.text(ix, iy,
                    f"fwd={grid[iy, ix]:+.2f}\nWR={wr_grid[iy, ix]:.0%}\nn={int(n_grid[iy, ix])}",
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(grid[iy, ix]) > vmax / 2 else 'black')
    ax.set_xticks(range(q))
    ax.set_yticks(range(q))
    ax.set_xticklabels([f'Q{i}' for i in range(q)])
    ax.set_yticklabels([f'Q{i}' for i in range(q)])
    ax.set_xlabel(f'{x_name}  (quantile)')
    ax.set_ylabel(f'{y_name}  (quantile)')
    ax.set_title(f'Mean forward return per (X,Y) bin\n{x_name} × {y_name}')
    plt.colorbar(im, ax=ax, label='mean forward return')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default=DEFAULT_BASE_TF)
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--layer1-dir',
                        default='reports/findings/v2_features_regime_eda',
                        help='Layer 1 outputs directory (regime_separation.csv, '
                             'price_correlations.csv)')
    parser.add_argument('--rank-by', default='lookback_corr',
                        choices=['cohen_d', 'lookback_corr', 'forward_corr',
                                  'concurrent_abs'])
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                        help='How many features from Layer 1 to consider for pairs')
    parser.add_argument('--quantiles', type=int, default=DEFAULT_QUANTILES,
                        help='Number of quantile bins per feature')
    parser.add_argument('--forward-n', type=int, default=DEFAULT_FORWARD_N,
                        help='Forward return window in bars')
    parser.add_argument('--split', default='IS')
    parser.add_argument('--top-pairs-to-plot', type=int,
                        default=DEFAULT_TOP_PAIRS_TO_PLOT,
                        help='Number of top-interaction pairs to save heatmap+CSV for')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_pairwise_eda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    top_pairs_dir = os.path.join(args.output_dir, 'top_pairs')
    os.makedirs(top_pairs_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"  V2 Features × Price — Layer 2 (pairwise)")
    print(f"  Base TF: {args.base_tf}  Split: {args.split}")
    print(f"  Top-K from Layer 1: {args.top_k} (ranked by {args.rank_by})")
    print(f"  Quantiles per feature: {args.quantiles}")
    print(f"  Forward N: {args.forward_n}")
    print(f"{'='*70}")

    # Layer 1 shortlist
    print(f"\n--- Loading Layer 1 shortlist ---")
    shortlist = load_layer1_shortlist(args.layer1_dir, args.top_k, args.rank_by)
    print(f"  Top-{args.top_k} features (by {args.rank_by}):")
    for f in shortlist:
        print(f"    {f}")

    # Load + merge data
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
    print(f"  {args.base_tf}: {len(base_df):,} bars")

    labels_df = load_regime_labels(args.labels_csv)
    labels_df = labels_df.copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]
    merged = base_df.merge(
        labels_df[['date', 'direction_axis', 'variation_axis', 'regime_2d', 'split']],
        on='date', how='inner')
    if args.split.upper() != 'ALL':
        merged = merged[merged['split'] == args.split.upper()].reset_index(drop=True)
    print(f"  After split={args.split}: {len(merged):,} bars")

    ts_int = merged['ts_int'].values.astype(np.int64)
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int.min()), int(ts_int.max())), verbose=False,
    )
    print(f"  v2 features: {len(features_5s):,} 5s rows")
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)

    # Anchor signals
    base_close = full['close'].values.astype(np.float64)
    n = len(base_close)
    fwd = np.full(n, np.nan)
    if n > args.forward_n:
        fwd[:-args.forward_n] = base_close[args.forward_n:] - base_close[:-args.forward_n]
    vwap_col = f'L2_{args.base_tf}_vwap_w'
    conc = (base_close - full[vwap_col].values.astype(np.float64)) \
        if vwap_col in full.columns else np.full(n, 0.0)

    # Validate shortlist features exist
    shortlist = [f for f in shortlist if f in full.columns]
    print(f"\n  Shortlist features present in data: {len(shortlist)}")

    # Pairwise analysis
    pairs = list(combinations(shortlist, 2))
    print(f"\n--- Analyzing {len(pairs)} pairs ---")
    pair_rows = []
    pair_cell_dfs = {}

    for x_name, y_name in pairs:
        x = full[x_name].values.astype(np.float64)
        y = full[y_name].values.astype(np.float64)
        result = analyze_pair(x, y, fwd, conc, args.quantiles)
        pair_rows.append({
            'x_feature': x_name,
            'y_feature': y_name,
            'pair': f'{x_name}__x__{y_name}',
            'interaction_rms': result['interaction'],
            'max_abs_mean_fwd': result['max_abs_mean_fwd'],
            'max_win_rate': result['max_win_rate'],
            'min_win_rate': result['min_win_rate'],
            'wr_spread': (result['max_win_rate'] - result['min_win_rate'])
                          if not (np.isnan(result['max_win_rate'])
                                   or np.isnan(result['min_win_rate']))
                          else float('nan'),
            'n_total': result['n_total'],
        })
        pair_cell_dfs[(x_name, y_name)] = result['cells']

    pair_df = pd.DataFrame(pair_rows).sort_values(
        'interaction_rms', ascending=False).reset_index(drop=True)
    pair_path = os.path.join(args.output_dir, 'pair_summary.csv')
    pair_df.to_csv(pair_path, index=False)
    print(f"  [saved] {pair_path}")

    # Print top pairs by interaction
    print(f"\n  Top {args.top_pairs_to_plot} pairs by interaction RMS:")
    for _, r in pair_df.head(args.top_pairs_to_plot).iterrows():
        print(f"    {r['x_feature'][:30]:>30} × {r['y_feature'][:30]:<30} "
              f"interaction={r['interaction_rms']:+.2f}  "
              f"WR_spread={r['wr_spread']:.0%}  "
              f"max|fwd|={r['max_abs_mean_fwd']:.2f}")

    print(f"\n  Top {args.top_pairs_to_plot} pairs by WR spread:")
    by_wr = pair_df.sort_values('wr_spread', ascending=False).head(args.top_pairs_to_plot)
    for _, r in by_wr.iterrows():
        print(f"    {r['x_feature'][:30]:>30} × {r['y_feature'][:30]:<30} "
              f"WR=[{r['min_win_rate']:.0%}..{r['max_win_rate']:.0%}]  "
              f"interaction={r['interaction_rms']:+.2f}")

    # Save top pairs (CSV + heatmap)
    top_pairs_to_save = pair_df.head(args.top_pairs_to_plot)['pair'].tolist()
    saved = 0
    for _, r in pair_df.head(args.top_pairs_to_plot).iterrows():
        pair_key = (r['x_feature'], r['y_feature'])
        cells_df = pair_cell_dfs.get(pair_key, pd.DataFrame())
        if len(cells_df) == 0:
            continue
        # Use safe filename
        slug = (r['x_feature'].replace('/', '_') + '__x__'
                + r['y_feature'].replace('/', '_'))[:120]
        cells_path = os.path.join(top_pairs_dir, f'{slug}.csv')
        cells_df.to_csv(cells_path, index=False)
        png_path = os.path.join(top_pairs_dir, f'{slug}.png')
        plot_pair_heatmap(cells_df, r['x_feature'], r['y_feature'],
                           args.quantiles, png_path)
        saved += 1
    print(f"\n  [saved] {saved} top-pair grids in {top_pairs_dir}/")

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 Pairwise Feature × Price EDA — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Base TF:** `{args.base_tf}`\n")
        f.write(f"**Split:** `{args.split}`\n")
        f.write(f"**Top-K from Layer 1:** {args.top_k} (ranked by `{args.rank_by}`)\n")
        f.write(f"**Quantiles:** {args.quantiles} per feature\n")
        f.write(f"**Forward N:** {args.forward_n} bars "
                f"({args.forward_n * (5 if args.base_tf == '5m' else 1)} min)\n\n")
        f.write(f"## Layer 1 shortlist\n\n")
        for s in shortlist:
            f.write(f"- `{s}`\n")
        f.write(f"\n## Top pairs by interaction RMS\n\n")
        f.write("Higher = more non-additive structure (the pair tells you more "
                "than the sum of its marginals).\n\n")
        f.write(pair_df.head(args.top_pairs_to_plot).to_string(index=False))
        f.write("\n\n")
        f.write(f"## Top pairs by WR spread\n\n")
        f.write(f"WR spread = max_cell_WR − min_cell_WR. Larger = pair "
                f"more strongly differentiates winning from losing bars.\n\n")
        f.write(by_wr.to_string(index=False))
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
