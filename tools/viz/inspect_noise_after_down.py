"""Inspect the NOISE-after-STEEP_LINEAR_DOWN cell at note level.

Two outputs:
    1. Cluster trajectories — extract the 5s close path of each NOISE note
       segment, resample to fixed length, normalize 0-1, hierarchical-cluster
       on Pearson distance. Show centroids + cluster sizes.
    2. Sample renders in context — pick K random instances, render the
       parent measure (STEEP_LINEAR_DOWN, 15s anchor) + the NOISE child
       (5s anchor) + the forward 30s window.

USAGE
    python tools/inspect_noise_after_down.py
    python tools/inspect_noise_after_down.py --n-clusters 6 --n-examples 9
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import _load_5s


def attach_parent_shape(notes: pd.DataFrame, measures: pd.DataFrame) -> pd.DataFrame:
    measures = measures.copy()
    measures['parent_chain'] = measures['parent_chain'].fillna('').astype(str)
    measures['idx'] = measures['idx'].astype(int)
    plk = measures.set_index(['day', 'parent_chain', 'idx'])['shape'].to_dict()
    notes = notes.copy()
    notes['parent_chain'] = notes['parent_chain'].fillna('').astype(str)
    def get_p(row):
        pc = row['parent_chain']
        if not pc:
            return None
        toks = pc.split('/')
        try:
            return plk.get((row['day'], '/'.join(toks[:-1]), int(toks[-1])))
        except ValueError:
            return None
    notes['parent_shape'] = notes.apply(get_p, axis=1)
    return notes


def extract_trajectory(day: str, start_ts: int, end_ts: int,
                        cache: dict, n_points: int = 16) -> np.ndarray:
    """Raw 5s close, resampled + normalized 0-1."""
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
    resampled = np.interp(tgt_x, src_x, seg)
    mn, mx = resampled.min(), resampled.max()
    if mx - mn < 1e-9:
        return np.zeros(n_points)
    return (resampled - mn) / (mx - mn)


def render_clusters(traj_mat: np.ndarray, labels: np.ndarray, out_png: str,
                     keep: set):
    cluster_ids_sorted = sorted(keep) + [0]
    n_panels = len(cluster_ids_sorted)
    cols = 3
    rows = max(1, int(np.ceil(n_panels / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4),
                              squeeze=False)
    cmap = plt.get_cmap('tab10')
    panel = 0
    for ci, cid in enumerate(cluster_ids_sorted):
        mask = labels == cid
        n = int(mask.sum())
        if n == 0:
            continue
        ax = axes.ravel()[panel]; panel += 1
        traj_in = traj_mat[mask]
        color = cmap(ci % 10) if cid != 0 else '#888'
        for t in traj_in[:60]:
            ax.plot(t, color=color, lw=0.4, alpha=0.18)
        centroid = traj_in.mean(axis=0)
        ax.plot(centroid, color=color if cid != 0 else '#444',
                lw=2.6, alpha=1.0, label='centroid')
        # describe shape
        diffs = np.diff(centroid)
        n_flips = int(((diffs[1:] * diffs[:-1]) < 0).sum())
        net = float(centroid[-1] - centroid[0])
        peak_idx = int(np.argmax(np.abs(centroid - 0.5)))
        cid_str = 'NOISE_residual' if cid == 0 else f'subC{cid}'
        ax.set_title(f'{cid_str}  n={n}  ({100*n/len(traj_mat):.1f}%)\n'
                     f'net={net:+.2f}  flips={n_flips}  peak@{peak_idx}/15',
                     fontsize=10)
        ax.set_xlabel('resampled time (16 pts)')
        ax.set_ylabel('normalized 5s close')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=8)
    for j in range(panel, len(axes.ravel())):
        axes.ravel()[j].set_visible(False)
    fig.suptitle(f'NOISE-after-STEEP_LINEAR_DOWN at note level — '
                  f'sub-shape clustering of {len(traj_mat):,} trajectories\n'
                  f'(Pearson-distance hierarchical clustering, normalized 5s close)',
                  fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)


def render_example_in_context(day: str, note_row: pd.Series,
                                parent_row: pd.Series, cache: dict,
                                ax) -> None:
    """Render parent measure + child note + forward window."""
    if day not in cache or cache[day] is None:
        return
    ts, close = cache[day]
    p_start = int(parent_row['start_ts'])
    p_end = int(parent_row['end_ts'])
    n_start = int(note_row['start_ts'])
    n_end = int(note_row['end_ts'])
    fwd_end = n_end + 30  # 30s forward
    win_start = p_start - 60
    win_end = fwd_end + 30
    i_a = int(np.searchsorted(ts, win_start))
    i_b = int(np.searchsorted(ts, win_end))
    if i_b - i_a < 4:
        return
    sub_ts = ts[i_a:i_b]
    sub_close = close[i_a:i_b]
    sub_dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in sub_ts]
    ax.plot(sub_dt, sub_close, color='black', lw=0.7, alpha=0.85)

    # parent span tint (red — STEEP_LINEAR_DOWN)
    p_start_dt = datetime.fromtimestamp(p_start, tz=timezone.utc)
    p_end_dt = datetime.fromtimestamp(p_end, tz=timezone.utc)
    ax.axvspan(p_start_dt, p_end_dt, color='#E53935', alpha=0.10,
                label=f'parent: STEEP_LINEAR_DOWN ({parent_row["length_min"]*60:.0f}s)')
    # child span tint (gray — NOISE)
    n_start_dt = datetime.fromtimestamp(n_start, tz=timezone.utc)
    n_end_dt = datetime.fromtimestamp(n_end, tz=timezone.utc)
    ax.axvspan(n_start_dt, n_end_dt, color='#9E9E9E', alpha=0.35,
                label=f'NOISE child ({note_row["length_min"]*60:.0f}s)')
    # forward window (green/red shading by direction)
    fwd_end_dt = datetime.fromtimestamp(fwd_end, tz=timezone.utc)
    i_n_end = int(np.searchsorted(ts, n_end))
    i_fwd_end = int(np.searchsorted(ts, fwd_end))
    if i_fwd_end > i_n_end and i_fwd_end < len(close):
        net = close[i_fwd_end] - close[i_n_end]
        col = '#43A047' if net > 0 else '#E53935'
        ax.axvspan(n_end_dt, fwd_end_dt, color=col, alpha=0.12,
                    label=f'fwd 30s: {net:+.2f}')
    ax.set_title(f'{day}  {datetime.fromtimestamp(n_start, tz=timezone.utc).strftime("%H:%M:%S")}',
                  fontsize=9)
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-points', type=int, default=16)
    ap.add_argument('--threshold', type=float, default=0.85,
                     help='Pearson r cluster threshold (cut at 1-r)')
    ap.add_argument('--min-cluster-size', type=int, default=200)
    ap.add_argument('--n-examples', type=int, default=9,
                     help='Number of in-context example panels')
    ap.add_argument('--max-traj', type=int, default=4000,
                     help='Cap n trajectories used for clustering (n^2 distance matrix)')
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/noise_after_down_inspection')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    notes = pd.read_csv('reports/findings/segments/simple_bulk_v2/all_notes.csv')
    measures = pd.read_csv('reports/findings/segments/simple_bulk_v2/all_measures.csv')
    notes = attach_parent_shape(notes, measures)

    cell = notes[(notes['shape'] == 'NOISE')
                  & (notes['parent_shape'] == 'STEEP_LINEAR_DOWN')].reset_index(drop=True)
    print(f'Cell n = {len(cell):,}')

    # Cluster on a sub-sample if too big (n^2 distance matrix)
    if len(cell) > args.max_traj:
        cluster_sample = cell.sample(n=args.max_traj, random_state=42).reset_index(drop=True)
        print(f'Subsampling {args.max_traj:,} for clustering (n^2 RAM-bound)')
    else:
        cluster_sample = cell

    print(f'Extracting trajectories ({args.n_points} points each)...')
    cache = {}
    trajectories = []
    keep_idx = []
    for i, r in tqdm(cluster_sample.iterrows(), total=len(cluster_sample)):
        traj = extract_trajectory(r['day'], int(r['start_ts']), int(r['end_ts']),
                                    cache, n_points=args.n_points)
        if np.all(np.isfinite(traj)) and traj.std() > 1e-9:
            trajectories.append(traj)
            keep_idx.append(i)
    traj_mat = np.array(trajectories)
    print(f'Valid trajectories: {len(traj_mat):,}')

    # Pairwise corr distance
    print('Computing pairwise Pearson distances...')
    corr = np.corrcoef(traj_mat)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, 2)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    cluster_labels = fcluster(Z, t=(1.0 - args.threshold), criterion='distance')

    from collections import Counter
    label_counts = Counter(cluster_labels)
    keep = {l for l, n in label_counts.items() if n >= args.min_cluster_size}
    final_labels = np.array([l if l in keep else 0 for l in cluster_labels])
    n_clusters = len(keep)
    n_residual = (final_labels == 0).sum()
    print(f'Clusters with n>={args.min_cluster_size}: {n_clusters}')
    print(f'Residual (small clusters): {n_residual}')
    print(f'\nCluster summary:')
    for cid in [0] + sorted(keep):
        mask = final_labels == cid
        n = int(mask.sum())
        if n == 0:
            continue
        avg = traj_mat[mask].mean(axis=0)
        diffs = np.diff(avg)
        n_flips = int(((diffs[1:] * diffs[:-1]) < 0).sum())
        net = float(avg[-1] - avg[0])
        cid_str = 'NOISE_resid' if cid == 0 else f'subC{cid}'
        print(f'  {cid_str:>11s}  n={n:>5d}  ({100*n/len(traj_mat):.1f}%)  '
              f'net={net:+.2f}  flips={n_flips}')

    # Render cluster centroids
    out_clusters = os.path.join(args.out_dir, 'sub_shape_clusters.png')
    render_clusters(traj_mat, final_labels, out_clusters, keep)
    print(f'\nClusters chart -> {out_clusters}')

    # Save mapping
    cluster_sample_keep = cluster_sample.iloc[keep_idx].copy()
    cluster_sample_keep['subshape_id'] = final_labels
    cluster_sample_keep[['day', 'idx', 'parent_chain', 'shape', 'parent_shape',
                          'length_min', 'subshape_id']].to_csv(
        os.path.join(args.out_dir, 'subshape_mapping.csv'), index=False)

    # Example renders in context
    print(f'\nRendering {args.n_examples} in-context examples...')
    examples = cell.sample(n=args.n_examples, random_state=7).reset_index(drop=True)
    cols = 3
    rows = int(np.ceil(args.n_examples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4),
                              squeeze=False)
    measures_idx = measures.set_index(['day', 'parent_chain', 'idx']).to_dict('index')
    for ax, (_, r) in zip(axes.ravel(), examples.iterrows()):
        # find parent measure row
        toks = r['parent_chain'].split('/')
        parent_chain_above = '/'.join(toks[:-1])
        try:
            parent_idx = int(toks[-1])
            parent_row = measures_idx.get((r['day'], parent_chain_above, parent_idx))
        except ValueError:
            parent_row = None
        if parent_row is None:
            ax.set_visible(False); continue
        parent_row = pd.Series(parent_row)
        render_example_in_context(r['day'], r, parent_row, cache, ax)
    for j in range(args.n_examples, len(axes.ravel())):
        axes.ravel()[j].set_visible(False)
    fig.suptitle('NOISE-after-STEEP_LINEAR_DOWN — sample occurrences in context\n'
                  '(red span = parent STEEP_LINEAR_DOWN measure  /  '
                  'gray span = NOISE child note  /  '
                  'green or red trailing = forward 30s direction)',
                  fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_examples = os.path.join(args.out_dir, 'examples_in_context.png')
    plt.savefig(out_examples, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Examples chart -> {out_examples}')


if __name__ == '__main__':
    main()
