"""
v2_features_chord_eda.py — Layer 3 reframed as "chord finder".

Music metaphor: pairs are intervals, triples are chords. We're not asking
"do 3 features predict better than 2" (mixing-colors marginal-value
question). We're asking "do certain triplets co-activate into a
recognizable fingerprint" — a chord that maps to a specific regime or
price behavior.

For each candidate triplet (X, Y, Z) from a top-K shortlist:
  1. Bin each feature into Q quantiles -> Q^3 cells.
  2. Per cell, compute:
       - regime distribution (% of bars in this cell that are UP_SMOOTH,
         UP_CHOPPY, ..., FLAT_CHOPPY)
       - dominant regime + its concentration (max %)
       - mean forward return
       - win rate
       - cell support (n bars)
  3. Score the triplet by:
       - chord_purity: mean dominant_regime_pct across all populated cells
         (high = cells reliably map to specific regimes)
       - chord_signal: max|cell_mean_fwd| × min(cell_n / 50, 1)
         (high = at least one cell has strong, well-supported price reaction)
       - chord_entropy: average regime entropy per cell
         (low = cells are regime-concentrated)

Output: ranking of triplets by chord_purity + chord_signal. Cells of the
top triplets exported individually so we can see "what does this chord
mean" in regime + price terms.

Usage:
  python tools/v2_features_chord_eda.py
  python tools/v2_features_chord_eda.py --top-k 10 --quantiles 3
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

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import (
    load_regime_labels, REGIME_2D_ORDER,
)
from tools.v2_features_lookback_eda import load_shortlist


DEFAULT_BASE_TF = '5m'
DEFAULT_FORWARD_N = 12
DEFAULT_TOP_K = 10        # C(10, 3) = 120 triplets
DEFAULT_QUANTILES = 3     # 27 cells per triplet
DEFAULT_MIN_CELL_N = 50


def quantile_bins(values: np.ndarray, q: int) -> np.ndarray:
    """Integer quantile labels 0..q-1, NaN -> -1."""
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


def analyze_chord(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   regimes: np.ndarray, fwd: np.ndarray, q: int,
                   min_cell_n: int) -> dict:
    """For one triplet, compute per-cell regime + price stats."""
    bx = quantile_bins(x, q)
    by = quantile_bins(y, q)
    bz = quantile_bins(z, q)
    valid = (bx >= 0) & (by >= 0) & (bz >= 0) & ~np.isnan(fwd)
    if valid.sum() < q ** 3 * 5:
        return {'cells': pd.DataFrame(), 'chord_purity': float('nan'),
                'chord_signal': float('nan'), 'chord_entropy': float('nan'),
                'n_populated_cells': 0}

    cells = []
    for ix in range(q):
        for iy in range(q):
            for iz in range(q):
                mask = valid & (bx == ix) & (by == iy) & (bz == iz)
                n = int(mask.sum())
                if n < min_cell_n:
                    continue
                # Regime distribution within cell
                cell_regimes = regimes[mask]
                regime_counts = pd.Series(cell_regimes).value_counts()
                regime_pcts = regime_counts / n
                dominant_regime = regime_counts.idxmax()
                dominant_pct = float(regime_pcts.iloc[0])
                # Entropy
                p = regime_pcts.values
                p = p[p > 0]
                entropy = float(-(p * np.log2(p)).sum())
                # Price
                f = fwd[mask]
                cells.append({
                    'x_q': ix, 'y_q': iy, 'z_q': iz,
                    'n': n,
                    'dominant_regime': dominant_regime,
                    'dominant_pct': dominant_pct,
                    'entropy': entropy,
                    'mean_fwd': float(f.mean()),
                    'std_fwd': float(f.std(ddof=1)) if n > 1 else 0.0,
                    'win_rate': float((f > 0).mean()),
                    # Per-regime % (top 6 regimes)
                    **{f'pct_{r}': float(regime_pcts.get(r, 0.0))
                       for r in REGIME_2D_ORDER},
                })

    cells_df = pd.DataFrame(cells)
    if len(cells_df) == 0:
        return {'cells': cells_df, 'chord_purity': float('nan'),
                'chord_signal': float('nan'), 'chord_entropy': float('nan'),
                'n_populated_cells': 0}

    chord_purity = float(cells_df['dominant_pct'].mean())
    chord_entropy = float(cells_df['entropy'].mean())
    # chord_signal: max |mean_fwd| weighted by sample size
    weights = np.minimum(cells_df['n'] / 200, 1.0)
    weighted_abs_fwd = (cells_df['mean_fwd'].abs() * weights)
    chord_signal = float(weighted_abs_fwd.max())

    return {
        'cells': cells_df,
        'chord_purity': chord_purity,
        'chord_signal': chord_signal,
        'chord_entropy': chord_entropy,
        'n_populated_cells': len(cells_df),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default=DEFAULT_BASE_TF)
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--layer1-dir',
                        default='reports/findings/v2_features_regime_eda')
    parser.add_argument('--rank-by', default='cohen_d',
                        choices=['cohen_d', 'lookback_corr', 'forward_corr'],
                        help='Layer 1 ranking — cohen_d is best for regime-fingerprint hunting')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--quantiles', type=int, default=DEFAULT_QUANTILES)
    parser.add_argument('--forward-n', type=int, default=DEFAULT_FORWARD_N)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-cell-n', type=int, default=DEFAULT_MIN_CELL_N)
    parser.add_argument('--top-chords-to-save', type=int, default=15)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_chord_eda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    chords_dir = os.path.join(args.output_dir, 'top_chords')
    os.makedirs(chords_dir, exist_ok=True)

    n_pairs = args.top_k * (args.top_k - 1) * (args.top_k - 2) // 6
    print(f"{'='*70}")
    print(f"  V2 Features Chord EDA (3-feature regime fingerprint hunt)")
    print(f"  Base TF: {args.base_tf}  Split: {args.split}")
    print(f"  Top-K: {args.top_k} -> C(K,3) = {n_pairs} triplets")
    print(f"  Quantiles: {args.quantiles} -> {args.quantiles**3} cells per triplet")
    print(f"  Min cell support: {args.min_cell_n}")
    print(f"{'='*70}")

    # Shortlist
    shortlist = load_shortlist(args.layer1_dir, args.top_k, args.rank_by)
    print(f"\n  Shortlist (by {args.rank_by}):")
    for f in shortlist:
        print(f"    {f}")

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
    shortlist = [f for f in shortlist if f in full.columns]

    close = full['close'].values.astype(np.float64)
    n = len(close)
    fwd = np.full(n, np.nan)
    if n > args.forward_n:
        fwd[:-args.forward_n] = close[args.forward_n:] - close[:-args.forward_n]
    regimes = full['regime_2d'].values.astype(str)

    # Iterate triplets
    triplets = list(combinations(shortlist, 3))
    print(f"\n--- Analyzing {len(triplets)} triplets ---")
    chord_rows = []
    triplet_cell_dfs = {}

    for (a, b, c) in triplets:
        x = full[a].values.astype(np.float64)
        y = full[b].values.astype(np.float64)
        z = full[c].values.astype(np.float64)
        result = analyze_chord(x, y, z, regimes, fwd, args.quantiles,
                                args.min_cell_n)
        chord_rows.append({
            'x': a, 'y': b, 'z': c,
            'chord_purity': result['chord_purity'],
            'chord_signal': result['chord_signal'],
            'chord_entropy': result['chord_entropy'],
            'n_cells': result['n_populated_cells'],
        })
        triplet_cell_dfs[(a, b, c)] = result['cells']

    chord_df = pd.DataFrame(chord_rows)
    chord_df = chord_df.sort_values('chord_purity', ascending=False).reset_index(drop=True)
    out_path = os.path.join(args.output_dir, 'chord_summary.csv')
    chord_df.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}")

    # Print rankings
    print(f"\n  Top {args.top_chords_to_save} chords by purity (cells map cleanly to one regime):")
    print(f"    {'features':>80}  {'purity':>7} {'signal':>7} {'entropy':>7} {'cells':>5}")
    for _, r in chord_df.head(args.top_chords_to_save).iterrows():
        feats = f"{r['x'][:25]} + {r['y'][:25]} + {r['z'][:25]}"
        print(f"    {feats[:80]:>80}  {r['chord_purity']:>7.1%} "
              f"{r['chord_signal']:>+7.2f} {r['chord_entropy']:>7.2f} "
              f"{int(r['n_cells']):>5}")

    print(f"\n  Top {args.top_chords_to_save} chords by signal (max-cell |fwd return|):")
    by_sig = chord_df.sort_values('chord_signal', ascending=False).head(args.top_chords_to_save)
    for _, r in by_sig.iterrows():
        feats = f"{r['x'][:25]} + {r['y'][:25]} + {r['z'][:25]}"
        print(f"    {feats[:80]:>80}  signal={r['chord_signal']:>+7.2f} "
              f"purity={r['chord_purity']:>6.1%} cells={int(r['n_cells'])}")

    # Save top-N chord cell tables
    saved = 0
    union_top = pd.concat([chord_df.head(args.top_chords_to_save),
                            by_sig]).drop_duplicates(subset=['x', 'y', 'z'])
    for _, r in union_top.iterrows():
        cells_df = triplet_cell_dfs.get((r['x'], r['y'], r['z']),
                                         pd.DataFrame())
        if len(cells_df) == 0:
            continue
        slug = f"{r['x']}__x__{r['y']}__x__{r['z']}".replace('/', '_')[:140]
        cells_path = os.path.join(chords_dir, f'{slug}.csv')
        cells_df.to_csv(cells_path, index=False)
        saved += 1
    print(f"\n  [saved] {saved} top-chord cell tables in {chords_dir}/")

    # Top-3 chord drilldowns: show what each cell means
    print(f"\n--- Top 3 chord drilldowns (top cells) ---")
    for _, r in chord_df.head(3).iterrows():
        cells_df = triplet_cell_dfs.get((r['x'], r['y'], r['z']))
        if cells_df is None or len(cells_df) == 0:
            continue
        print(f"\n  CHORD: {r['x']} + {r['y']} + {r['z']}")
        print(f"    purity={r['chord_purity']:.1%}, signal={r['chord_signal']:.2f}, "
              f"cells={r['n_cells']}")
        # Top 5 cells by dominant_pct
        top_cells = cells_df.sort_values('dominant_pct', ascending=False).head(5)
        for _, c in top_cells.iterrows():
            print(f"    cell ({c['x_q']},{c['y_q']},{c['z_q']}) n={c['n']:>5}: "
                  f"{c['dominant_regime']:>14} ({c['dominant_pct']:.0%})  "
                  f"mean_fwd={c['mean_fwd']:>+7.2f}  WR={c['win_rate']:.0%}")

    # Markdown
    md_path = os.path.join(args.output_dir, 'top_chords.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 feature chords (3-feature regime fingerprints) — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Base TF:** `{args.base_tf}`  **Split:** `{args.split}`\n")
        f.write(f"**Top-K:** {args.top_k} -> C(K,3) = {len(triplets)} triplets\n")
        f.write(f"**Quantiles:** {args.quantiles}\n")
        f.write(f"**Min cell support:** {args.min_cell_n} bars\n\n")
        f.write("## Top chords by purity\n\n")
        f.write("Higher purity = cells reliably map to one specific regime "
                "(the chord encodes a regime fingerprint).\n\n")
        f.write(chord_df.head(args.top_chords_to_save).to_string(index=False))
        f.write("\n\n")
        f.write("## Top chords by signal\n\n")
        f.write("Max cell |mean_fwd| × min(n/200, 1). Higher = at least one cell "
                "has a strong, well-supported price reaction.\n\n")
        f.write(by_sig.to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
