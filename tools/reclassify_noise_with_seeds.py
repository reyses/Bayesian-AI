"""Re-classify a NOISE-bucket cell using the FULL 20-primitive standalone-research
SeedPrimitiveLibrary instead of the 13-primitive simple-shapes library.

Rationale (locked 2026-05-10 morning): at the note level, mean span is
~2.2 min = 26 5s bars — plenty of data to fit richer templates (V's, U's,
STEP, skewed exponentials, oscillators). The simple-shapes-only architecture
deliberately excluded compound shapes because at coarse scales they
"decompose into legs at deeper levels". But the note level IS the leaf —
there's nowhere deeper. Compound shapes are valid primitive labels here.

Pipeline:
    1. Load all-notes.csv, filter to (note_shape, parent_shape) cell
    2. Extract raw 5s close trajectory per note, resample to N=16
    3. Classify against 20-primitive SeedPrimitiveLibrary at r>=0.75
    4. Report sub-shape distribution + per-sub-shape P(fwd_up)

Input cell defaults to NOISE-after-STEEP_LINEAR_DOWN at note level
(the canary cell from 2026-05-09 evening, n=9,561 / 4.69% of all notes).

USAGE
    python tools/reclassify_noise_with_seeds.py
    python tools/reclassify_noise_with_seeds.py --shape NOISE --parent STEEP_LINEAR_UP
    python tools/reclassify_noise_with_seeds.py --threshold 0.75
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.research.seeds import SeedPrimitiveLibrary
from tools.segment_day_motif_melody import _load_5s


def attach_parent_shape(notes: pd.DataFrame, measures: pd.DataFrame) -> pd.DataFrame:
    measures = measures.copy()
    measures['parent_chain'] = measures['parent_chain'].fillna('').astype(str)
    measures['idx'] = measures['idx'].astype(int)
    plk = measures.set_index(['day', 'parent_chain', 'idx'])['shape'].to_dict()
    notes = notes.copy()
    notes['parent_chain'] = notes['parent_chain'].fillna('').astype(str)
    def _gp(row):
        pc = row['parent_chain']
        if not pc: return None
        toks = pc.split('/')
        try:
            return plk.get((row['day'], '/'.join(toks[:-1]), int(toks[-1])))
        except ValueError:
            return None
    notes['parent_shape'] = notes.apply(_gp, axis=1)
    return notes


def extract_trajectory(day: str, start_ts: int, end_ts: int,
                        cache: dict, n_points: int = 16) -> np.ndarray:
    if day not in cache:
        df = _load_5s(day)
        if df.empty:
            cache[day] = None
        else:
            cache[day] = (df['timestamp'].values.astype(np.int64),
                           df['close'].values.astype(np.float64))
    if cache[day] is None:
        return np.full(n_points, np.nan)
    ts, close = cache[day]
    i_a = int(np.searchsorted(ts, start_ts))
    i_b = int(np.searchsorted(ts, end_ts))
    if i_b - i_a < 2:
        return np.full(n_points, np.nan)
    seg = close[i_a:i_b + 1]
    src_x = np.linspace(0, 1, len(seg))
    tgt_x = np.linspace(0, 1, n_points)
    return np.interp(tgt_x, src_x, seg)


def classify_against_seeds(traj_raw: np.ndarray, lib: SeedPrimitiveLibrary,
                            threshold: float) -> tuple[str, float]:
    """Returns (best_shape, pearson_r) at >= threshold else ('NOISE', best_r)."""
    seg = np.asarray(traj_raw, dtype=float)
    if len(seg) != lib.N:
        return 'NOISE', 0.0
    mn, mx = seg.min(), seg.max()
    if mx - mn < 1e-12:
        return 'FLATLINE', 1.0
    normed = (seg - mn) / (mx - mn)
    best, best_r = 'NOISE', -2.0
    for name, tpl in lib.shapes.items():
        if tpl.std() < 1e-12:
            continue
        r = float(np.corrcoef(normed, tpl)[0, 1])
        if r > best_r:
            best, best_r = name, r
    if best_r < threshold:
        return 'NOISE', best_r
    return best, best_r


def attach_fwd_return(df: pd.DataFrame, horizon_s: int, cache: dict) -> pd.DataFrame:
    """Add fwd_return (close at end_ts + horizon_s) − close at end_ts)."""
    df = df.copy()
    df['fwd_return'] = np.nan
    for day, sub in df.groupby('day'):
        if day not in cache or cache[day] is None:
            continue
        ts, close = cache[day]
        end_ts = sub['end_ts'].astype(np.int64).values
        tgt_ts = end_ts + horizon_s
        i_end = np.searchsorted(ts, end_ts)
        i_tgt = np.searchsorted(ts, tgt_ts)
        valid = (i_end < len(ts)) & (i_tgt < len(ts))
        base = np.where(valid, close[np.clip(i_end, 0, len(ts)-1)], np.nan)
        target = np.where(valid, close[np.clip(i_tgt, 0, len(ts)-1)], np.nan)
        df.loc[sub.index, 'fwd_return'] = target - base
    return df


def beta_posterior(k: int, n: int) -> tuple[float, float, float]:
    a, b = k + 1, n - k + 1
    return float(a / (a + b)), float(beta_dist.ppf(0.025, a, b)), float(beta_dist.ppf(0.975, a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', default='NOISE')
    ap.add_argument('--parent', default='STEEP_LINEAR_DOWN')
    ap.add_argument('--threshold', type=float, default=0.75,
                     help='SeedPrimitiveLibrary uses 0.75 by default; tighten for cleaner subshapes')
    ap.add_argument('--n-points', type=int, default=16)
    ap.add_argument('--horizon-s', type=int, default=30,
                     help='Forward return horizon for P(fwd>0) (note-level default 30s)')
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/noise_after_down_inspection')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    notes = pd.read_csv('reports/findings/segments/simple_bulk_v2/all_notes.csv')
    measures = pd.read_csv('reports/findings/segments/simple_bulk_v2/all_measures.csv')
    notes = attach_parent_shape(notes, measures)
    cell = notes[(notes['shape'] == args.shape)
                  & (notes['parent_shape'] == args.parent)].reset_index(drop=True)
    print(f'Cell ({args.shape}, parent={args.parent}) n={len(cell):,}')

    lib = SeedPrimitiveLibrary(N=args.n_points)
    print(f'Library has {len(lib.shapes)} primitives')
    cache = {}

    print('Extracting + classifying trajectories...')
    sub_shapes = []
    sub_rs = []
    for _, r in tqdm(cell.iterrows(), total=len(cell)):
        traj = extract_trajectory(r['day'], int(r['start_ts']), int(r['end_ts']),
                                    cache, n_points=args.n_points)
        if not np.all(np.isfinite(traj)):
            sub_shapes.append('NA'); sub_rs.append(0.0); continue
        sh, r_val = classify_against_seeds(traj, lib, args.threshold)
        sub_shapes.append(sh); sub_rs.append(r_val)

    cell['seed_shape'] = sub_shapes
    cell['seed_r'] = sub_rs

    # Attach forward returns
    cell = attach_fwd_return(cell, args.horizon_s, cache)

    print(f'\nSub-shape distribution (threshold r >= {args.threshold}):')
    counts = Counter([s for s in sub_shapes if s != 'NA'])
    valid_n = len([s for s in sub_shapes if s != 'NA'])
    for sh, n in counts.most_common():
        print(f'  {sh:<25s} {n:>6d}  ({100*n/valid_n:.1f}%)')

    # P(fwd>0) per sub-shape (filter nan returns)
    valid_cell = cell.dropna(subset=['fwd_return'])
    print(f'\nP(fwd_return > 0 | sub-shape, parent={args.parent}):')
    rows = []
    for sh, sub in valid_cell.groupby('seed_shape'):
        n = len(sub)
        if n < 30:
            continue
        k = int((sub['fwd_return'] > 0).sum())
        p, lo, hi = beta_posterior(k, n)
        mean_ret = float(sub['fwd_return'].mean())
        rows.append({
            'seed_shape': sh, 'n': n, 'k_up': k,
            'p_up_mean': round(p, 4),
            'ci95_lo': round(lo, 4),
            'ci95_hi': round(hi, 4),
            'mean_fwd_return': round(mean_ret, 3),
        })
    out = pd.DataFrame(rows).sort_values('n', ascending=False)
    print(out.to_string(index=False))

    csv_path = os.path.join(args.out_dir,
                              f'reclassified_{args.shape}_under_{args.parent}.csv')
    out.to_csv(csv_path, index=False)
    cell[['day', 'idx', 'parent_chain', 'shape', 'parent_shape',
           'seed_shape', 'seed_r', 'length_min', 'fwd_return']].to_csv(
        os.path.join(args.out_dir,
                      f'reclassified_per_note_{args.shape}_under_{args.parent}.csv'),
        index=False)

    # Render: sub-shape p_up bar + library template centroid grid
    if not out.empty:
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        # Bar of p_up with CIs
        top = out.head(15)
        y = np.arange(len(top))[::-1]
        p = top['p_up_mean'].values
        lo = top['ci95_lo'].values
        hi = top['ci95_hi'].values
        colors = ['#43A047' if pi > 0.5 else '#E53935' for pi in p]
        axes[0].barh(y, p, xerr=[p - lo, hi - p], color=colors, alpha=0.85)
        axes[0].axvline(0.5, color='black', lw=0.6, alpha=0.5)
        for yi, (_, row) in zip(y, top.iterrows()):
            axes[0].text(row['p_up_mean'] + 0.005, yi,
                          f'  p={row["p_up_mean"]:.3f}  n={row["n"]}',
                          va='center', fontsize=8)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(top['seed_shape'])
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel('P(fwd_return > 0)')
        axes[0].set_title(
            f'{args.shape}-after-{args.parent} reclassified with 20-primitive seeds\n'
            f'P(fwd>0 | sub-shape)  horizon={args.horizon_s}s', fontsize=11)
        axes[0].grid(True, axis='x', alpha=0.3)

        # Library template centroids overlaid
        cmap = plt.get_cmap('tab20')
        for i, (name, tpl) in enumerate(lib.shapes.items()):
            axes[1].plot(tpl, color=cmap(i % 20), lw=1.4, alpha=0.85, label=name)
        axes[1].set_title(f'20-primitive seed library (N={lib.N})', fontsize=11)
        axes[1].set_xlabel('resampled time'); axes[1].set_ylabel('normalized 0-1')
        axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                        fontsize=7, ncol=1)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = os.path.join(args.out_dir,
                                f'reclassified_{args.shape}_under_{args.parent}.png')
        plt.savefig(out_png, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()
