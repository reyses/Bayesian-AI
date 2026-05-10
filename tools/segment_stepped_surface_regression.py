"""Stepped surface (hyper-volume) regression per level.

For each level, treats every segment as a sample with:
    features (X): shape (categorical), skew (categorical), length_min,
                   mean_sigma, r, segment_slope_pts_per_min, parent_shape
    target  (Y): forward return measured at end_ts over a level-specific horizon

Outputs per level:
    1. Marginal expectations: E[fwd_return | shape], CI, n  (csv + chart)
    2. Surface: E[fwd_return | shape × length_bucket] heatmap
    3. Surface: E[fwd_return | shape × sigma_bucket]  heatmap
    4. Surface: E[fwd_return | shape × parent_shape]  heatmap
    5. OLS regression with one-hot shape + length + sigma + r  (coef table)
    6. MFE/MAE per shape (forward best/worst excursion)

Horizons (forward, after segment end):
    phrase (15m segment): 60min forward
    motif  (5m  segment): 30min forward
    sub_motif (1m segment): 10min forward
    measure (15s segment): 2min forward
    note  (5s segment): 30s forward

Outputs to reports/findings/segments/stepped_surface_reg/<level>__*.csv/.png

USAGE
    python tools/segment_stepped_surface_regression.py
    python tools/segment_stepped_surface_regression.py --level note  # just one level
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import _load_5s


LEVELS_HORIZONS = {
    'phrase':    60 * 60,   # 60 min forward
    'motif':     30 * 60,
    'sub_motif': 10 * 60,
    'measure':   2 * 60,
    'note':      30,
}

PARENT = {'phrase': None, 'motif': 'phrase', 'sub_motif': 'motif',
          'measure': 'sub_motif', 'note': 'measure'}


def compute_forward_returns(level_df: pd.DataFrame, horizon_s: int,
                             cache: dict) -> pd.DataFrame:
    """For each row, look up close at end_ts and at end_ts + horizon_s on
    the day's 5s bar. Add fwd_return, fwd_mfe, fwd_mae columns (in points).

    Vectorized per-day: compute all i_end and i_tgt at once, then loop
    only for the windowed MFE/MAE.
    """
    out = level_df.copy()
    out['parent_chain'] = out['parent_chain'].fillna('').astype(str)
    out['fwd_return'] = np.nan
    out['fwd_mfe'] = np.nan
    out['fwd_mae'] = np.nan
    for day, sub in tqdm(out.groupby('day'), desc='fwd-returns'):
        if day not in cache:
            df_5s = _load_5s(day)
            if df_5s.empty:
                cache[day] = None
                continue
            ts = df_5s['timestamp'].values.astype(np.int64)
            close = df_5s['close'].values.astype(np.float64)
            # Cache cumulative max/min for fast windowed MFE/MAE? Not trivial because
            # window starts vary. Stick with per-row windowed pass but avoid full copy.
            cache[day] = (ts, close)
        if cache[day] is None:
            continue
        ts, close = cache[day]
        end_ts_arr = sub['end_ts'].astype(np.int64).values
        tgt_ts_arr = end_ts_arr + horizon_s
        i_end = np.searchsorted(ts, end_ts_arr)
        i_tgt = np.searchsorted(ts, tgt_ts_arr)
        valid_mask = (i_end < len(ts)) & (i_tgt < len(ts))
        # forward return vectorized
        base = np.where(valid_mask, close[np.clip(i_end, 0, len(ts)-1)], np.nan)
        target = np.where(valid_mask, close[np.clip(i_tgt, 0, len(ts)-1)], np.nan)
        fwd_ret = target - base
        # MFE / MAE: per-row loop (varying window)
        n = len(sub)
        mfe = np.full(n, np.nan); mae = np.full(n, np.nan)
        for k in range(n):
            if not valid_mask[k]:
                continue
            ie, it = int(i_end[k]), int(i_tgt[k])
            if it > ie:
                w = close[ie:it + 1]
                b = w[0]
                mfe[k] = float(w.max() - b)
                mae[k] = float(w.min() - b)
            else:
                mfe[k] = 0.0
                mae[k] = 0.0
        out.loc[sub.index, 'fwd_return'] = fwd_ret
        out.loc[sub.index, 'fwd_mfe'] = mfe
        out.loc[sub.index, 'fwd_mae'] = mae
    return out


def marginal_table(df: pd.DataFrame, group_col: str = 'shape') -> pd.DataFrame:
    """E[fwd_return | group] with bootstrap-free SE-based CI."""
    rows = []
    for g, sub in df.groupby(group_col):
        y = sub['fwd_return'].dropna().values
        if len(y) < 5:
            continue
        m = float(np.mean(y))
        se = float(np.std(y, ddof=1) / np.sqrt(len(y)))
        rows.append({
            group_col: g, 'n': len(y),
            'mean':   round(m, 3),
            'se':     round(se, 3),
            'ci95_lo': round(m - 1.96 * se, 3),
            'ci95_hi': round(m + 1.96 * se, 3),
            'mfe_mean': round(float(np.mean(sub['fwd_mfe'].dropna())), 3),
            'mae_mean': round(float(np.mean(sub['fwd_mae'].dropna())), 3),
        })
    return pd.DataFrame(rows).sort_values('n', ascending=False)


def surface_table(df: pd.DataFrame, axis_a: str, axis_b: str,
                   metric: str = 'fwd_return') -> pd.DataFrame:
    """Pivot table E[metric | axis_a, axis_b]."""
    sub = df.dropna(subset=[metric, axis_a, axis_b]).copy()
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(index=axis_a, columns=axis_b, values=metric,
                              aggfunc='mean')
    counts = sub.pivot_table(index=axis_a, columns=axis_b, values=metric,
                              aggfunc='count')
    return pivot, counts


def render_heatmap(pivot: pd.DataFrame, counts: pd.DataFrame, title: str,
                    out_png: str):
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(pivot.columns) + 4),
                                       max(5, 0.4 * len(pivot.index) + 3)))
    vmin, vmax = float(pivot.min().min()), float(pivot.max().max())
    vabs = max(abs(vmin), abs(vmax))
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-vabs, vmax=vabs,
                    aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            n = counts.values[i, j] if counts is not None else 0
            if not np.isnan(v):
                ax.text(j, i, f'{v:+.2f}\nn={int(n) if not np.isnan(n) else 0}',
                         ha='center', va='center', fontsize=7,
                         color='black' if abs(v) < 0.5 * vabs else 'white')
    plt.colorbar(im, ax=ax, label='mean fwd_return (pts)')
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)


def attach_parent_shape(child_df: pd.DataFrame,
                         parent_df: pd.DataFrame) -> pd.DataFrame:
    parent_df = parent_df.copy()
    parent_df['parent_chain'] = parent_df['parent_chain'].fillna('').astype(str)
    parent_df['idx'] = parent_df['idx'].astype(int)
    parent_lookup = parent_df.set_index(
        ['day', 'parent_chain', 'idx']
    )['shape'].to_dict()
    child_df = child_df.copy()
    child_df['parent_chain'] = child_df['parent_chain'].fillna('').astype(str)

    def get_parent(row):
        pc = row['parent_chain']
        if not pc:
            return None
        toks = pc.split('/')
        try:
            return parent_lookup.get(
                (row['day'], '/'.join(toks[:-1]), int(toks[-1])))
        except ValueError:
            return None

    child_df['parent_shape'] = child_df.apply(get_parent, axis=1)
    return child_df


def ols_with_shape_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Simple regression: fwd_return ~ shape (one-hot) + length_min + mean_sigma + r.
    Outputs coef + se + p for each shape effect."""
    sub = df.dropna(subset=['fwd_return', 'shape', 'length_min',
                              'mean_sigma', 'r']).copy()
    if sub.empty:
        return pd.DataFrame()
    X_dummies = pd.get_dummies(sub['shape'], prefix='shape', drop_first=True).astype(float)
    X_cont = sub[['length_min', 'mean_sigma', 'r']].astype(float).values
    X_dummies_arr = X_dummies.values
    # standardize continuous for stable OLS
    mu = X_cont.mean(axis=0); sd = X_cont.std(axis=0)
    sd[sd < 1e-9] = 1.0
    X_cont_z = (X_cont - mu) / sd
    X = np.hstack([np.ones((len(sub), 1)), X_dummies_arr, X_cont_z])
    y = sub['fwd_return'].astype(float).values
    # OLS via lstsq
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    resid = y - yhat
    dof = max(1, len(y) - X.shape[1])
    sigma2 = float((resid @ resid) / dof)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
    t_stat = coef / np.where(se > 0, se, 1.0)
    names = (['intercept']
              + list(X_dummies.columns)
              + ['length_min_z', 'mean_sigma_z', 'r_z'])
    rows = []
    for name, c, s, t in zip(names, coef, se, t_stat):
        rows.append({
            'feature': name,
            'coef': round(float(c), 4),
            'se': round(float(s), 4),
            't_stat': round(float(t), 2),
            'sig': '***' if abs(t) > 3.29 else '**' if abs(t) > 2.58
                    else '*' if abs(t) > 1.96 else '',
        })
    return pd.DataFrame(rows)


def quintile_bucket(s: pd.Series, k: int = 5, prefix: str = 'q') -> pd.Series:
    return pd.qcut(s.rank(method='first'), q=k,
                    labels=[f'{prefix}{i+1}' for i in range(k)])


def process_level(level: str, csv_path: str, parent_csv: str | None,
                   out_dir: str, horizon_s: int, cache: dict,
                   max_rows: int | None = None):
    print(f'\n=== {level} (horizon {horizon_s}s) ===')
    df = pd.read_csv(csv_path)
    if max_rows is not None and len(df) > max_rows:
        # Stratified sample by shape if too big (notes are 200k)
        df = (df.groupby('shape', group_keys=False)
                 .apply(lambda x: x.sample(n=min(len(x),
                                                   int(max_rows * len(x) / len(df))),
                                            random_state=42)))
    print(f'  loaded {len(df):,} rows')

    df = compute_forward_returns(df, horizon_s, cache)
    valid = df.dropna(subset=['fwd_return'])
    print(f'  with valid fwd_return: {len(valid):,}')

    if parent_csv and os.path.exists(parent_csv):
        parent_df = pd.read_csv(parent_csv)
        df = attach_parent_shape(df, parent_df)
        valid = df.dropna(subset=['fwd_return'])

    # 1. marginal by shape
    marg = marginal_table(valid, 'shape')
    marg.to_csv(os.path.join(out_dir, f'{level}_marginal_by_shape.csv'), index=False)
    print(f'  marginal by shape -> {len(marg)} rows')

    # 2. surface shape × length_bucket
    valid['length_bucket'] = quintile_bucket(valid['length_min'], k=5, prefix='Lq')
    pivot1, count1 = surface_table(valid, 'shape', 'length_bucket')
    pivot1.to_csv(os.path.join(out_dir, f'{level}_surface_shape_length.csv'))
    render_heatmap(pivot1, count1,
                    f'[{level}] E[fwd_return | shape × length_quintile]  '
                    f'horizon={horizon_s}s',
                    os.path.join(out_dir, f'{level}_surface_shape_length.png'))

    # 3. surface shape × sigma_bucket
    valid['sigma_bucket'] = quintile_bucket(valid['mean_sigma'].fillna(
        valid['mean_sigma'].median()), k=5, prefix='Sq')
    pivot2, count2 = surface_table(valid, 'shape', 'sigma_bucket')
    pivot2.to_csv(os.path.join(out_dir, f'{level}_surface_shape_sigma.csv'))
    render_heatmap(pivot2, count2,
                    f'[{level}] E[fwd_return | shape × sigma_quintile]  '
                    f'horizon={horizon_s}s',
                    os.path.join(out_dir, f'{level}_surface_shape_sigma.png'))

    # 4. surface shape × parent_shape (if parent attached)
    if 'parent_shape' in valid.columns and valid['parent_shape'].notna().any():
        pp = valid.dropna(subset=['parent_shape'])
        pivot3, count3 = surface_table(pp, 'parent_shape', 'shape')
        pivot3.to_csv(os.path.join(out_dir, f'{level}_surface_parent_self.csv'))
        render_heatmap(pivot3, count3,
                        f'[{level}] E[fwd_return | parent_shape × shape]  '
                        f'horizon={horizon_s}s',
                        os.path.join(out_dir, f'{level}_surface_parent_self.png'))

    # 5. OLS coef table
    ols = ols_with_shape_dummies(valid)
    ols.to_csv(os.path.join(out_dir, f'{level}_ols_coefs.csv'), index=False)
    print(f'  OLS coefs -> {len(ols)} rows')

    # 6. shape × MFE/MAE
    mfe_table = (valid.groupby('shape')
                  .agg(n=('fwd_mfe', 'count'),
                        mfe_mean=('fwd_mfe', 'mean'),
                        mfe_p90=('fwd_mfe', lambda x: x.quantile(0.9)),
                        mae_mean=('fwd_mae', 'mean'),
                        mae_p10=('fwd_mae', lambda x: x.quantile(0.1)))
                  .round(3).reset_index().sort_values('n', ascending=False))
    mfe_table.to_csv(os.path.join(out_dir, f'{level}_mfe_mae.csv'), index=False)

    return marg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default='reports/findings/segments/simple_bulk_v2')
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/stepped_surface_reg')
    ap.add_argument('--level', default=None,
                     help='Run only this level (default: all)')
    ap.add_argument('--max-note-rows', type=int, default=80000,
                     help='Subsample note level to keep runtime manageable')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    levels = [args.level] if args.level else list(LEVELS_HORIZONS.keys())
    cache = {}

    for level in levels:
        csv = os.path.join(args.in_dir, f'all_{level}s.csv')
        if not os.path.exists(csv):
            print(f'  skip [{level}] (no csv)')
            continue
        parent_lvl = PARENT[level]
        parent_csv = (os.path.join(args.in_dir, f'all_{parent_lvl}s.csv')
                       if parent_lvl else None)
        max_rows = args.max_note_rows if level == 'note' else None
        process_level(level, csv, parent_csv, args.out_dir,
                       LEVELS_HORIZONS[level], cache, max_rows=max_rows)


if __name__ == '__main__':
    main()
