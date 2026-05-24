"""Bayesian shape probability tables per level.

Builds two complementary tables per level:

    A. Shape priors (no conditioning)
       P(shape) at each level — how often each primitive shape occurs

    B. Forward-direction posteriors with Beta-Binomial smoothing
       P(fwd_return > 0 | shape, conditioning)
       conditioning options:
         - none (just shape)
         - skew (shape × skew)
         - sigma_bucket (shape × sigma quintile)
         - parent_shape (shape × parent_shape)

Smoothing: Beta(1,1) Jeffreys-style prior. Posterior mean = (k + 1) / (n + 2).
We also report 95% credible interval from the Beta posterior.

The tables are the SUBSTRATE for the V0 hierarchical Bayesian model we'll
calibrate later. Each level can be queried independently OR composed.

Output (per level):
    reports/findings/segments/bayes_tables/<level>_priors.csv
    reports/findings/segments/bayes_tables/<level>_p_up_given_shape.csv
    reports/findings/segments/bayes_tables/<level>_p_up_given_shape_skew.csv
    reports/findings/segments/bayes_tables/<level>_p_up_given_shape_sigma.csv
    reports/findings/segments/bayes_tables/<level>_p_up_given_shape_parent.csv
    reports/findings/segments/bayes_tables/<level>_priors.png
    reports/findings/segments/bayes_tables/<level>_p_up.png

Forward-return horizons (re-uses values from surface regression tool):
    phrase 60m, motif 30m, sub_motif 10m, measure 2m, note 30s

USAGE
    python tools/segment_bayes_tables.py
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
from scipy.stats import beta as beta_dist
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.segment_day_motif_melody import _load_5s


LEVELS_HORIZONS = {
    'phrase':    60 * 60,
    'motif':     30 * 60,
    'sub_motif': 10 * 60,
    'measure':   2 * 60,
    'note':      30,
}
PARENT = {'phrase': None, 'motif': 'phrase', 'sub_motif': 'motif',
          'measure': 'sub_motif', 'note': 'measure'}


def attach_fwd_return_and_parent(df: pd.DataFrame, horizon_s: int,
                                   parent_df: pd.DataFrame | None,
                                   cache: dict) -> pd.DataFrame:
    out = df.copy()
    out['parent_chain'] = out['parent_chain'].fillna('').astype(str)
    out['idx'] = out['idx'].astype(int)
    out['fwd_return'] = np.nan
    for day, sub in tqdm(out.groupby('day'), desc=f'fwd-h={horizon_s}'):
        if day not in cache:
            df_5s = _load_5s(day)
            if df_5s.empty:
                cache[day] = None
                continue
            cache[day] = (df_5s['timestamp'].values.astype(np.int64),
                           df_5s['close'].values.astype(np.float64))
        if cache[day] is None:
            continue
        ts, close = cache[day]
        end_ts = sub['end_ts'].astype(np.int64).values
        tgt_ts = end_ts + horizon_s
        i_end = np.searchsorted(ts, end_ts)
        i_tgt = np.searchsorted(ts, tgt_ts)
        valid = (i_end < len(ts)) & (i_tgt < len(ts))
        base = np.where(valid, close[np.clip(i_end, 0, len(ts)-1)], np.nan)
        target = np.where(valid, close[np.clip(i_tgt, 0, len(ts)-1)], np.nan)
        out.loc[sub.index, 'fwd_return'] = target - base

    if parent_df is not None:
        parent_df = parent_df.copy()
        parent_df['parent_chain'] = parent_df['parent_chain'].fillna('').astype(str)
        parent_df['idx'] = parent_df['idx'].astype(int)
        parent_lookup = parent_df.set_index(
            ['day', 'parent_chain', 'idx'])['shape'].to_dict()
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
        out['parent_shape'] = out.apply(get_parent, axis=1)
    return out


def beta_posterior(k: int, n: int) -> tuple[float, float, float]:
    """Beta(1,1) Jeffreys posterior. Returns (p_mean, ci95_lo, ci95_hi)."""
    a = k + 1; b = n - k + 1
    p_mean = a / (a + b)
    lo = beta_dist.ppf(0.025, a, b)
    hi = beta_dist.ppf(0.975, a, b)
    return float(p_mean), float(lo), float(hi)


def shape_priors(df: pd.DataFrame) -> pd.DataFrame:
    counts = df['shape'].value_counts()
    n = len(df)
    rows = []
    for sh, c in counts.items():
        p, lo, hi = beta_posterior(int(c), n)
        rows.append({
            'shape': sh, 'count': int(c), 'total': n,
            'p_mean': round(p, 4), 'ci95_lo': round(lo, 4),
            'ci95_hi': round(hi, 4),
        })
    return pd.DataFrame(rows).sort_values('count', ascending=False)


def p_up_given(df: pd.DataFrame, group_cols: list,
                min_n: int = 10) -> pd.DataFrame:
    sub = df.dropna(subset=['fwd_return'] + group_cols)
    rows = []
    for keys, g in sub.groupby(group_cols):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        n = len(g)
        if n < min_n:
            continue
        k = int((g['fwd_return'] > 0).sum())
        p, lo, hi = beta_posterior(k, n)
        # mean magnitude
        mean_ret = float(g['fwd_return'].mean())
        rec = {'n': n, 'k_up': k, 'p_up_mean': round(p, 4),
                'ci95_lo': round(lo, 4), 'ci95_hi': round(hi, 4),
                'mean_fwd_return': round(mean_ret, 3)}
        for col, val in zip(group_cols, keys):
            rec[col] = val
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df_out = pd.DataFrame(rows)
    cols = group_cols + ['n', 'k_up', 'p_up_mean', 'ci95_lo', 'ci95_hi',
                          'mean_fwd_return']
    return df_out[cols].sort_values(['n'], ascending=False)


def render_prior_chart(priors: pd.DataFrame, title: str, out_png: str):
    if priors.empty:
        return
    top = priors.head(15)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(top) + 2)))
    y = np.arange(len(top))[::-1]
    ax.barh(y, top['p_mean'], xerr=[top['p_mean'] - top['ci95_lo'],
                                       top['ci95_hi'] - top['p_mean']],
             color='#1E88E5', alpha=0.85, ecolor='#0D47A1')
    for yi, (_, row) in zip(y, top.iterrows()):
        ax.text(row['p_mean'] + 0.005, yi,
                  f' p={row["p_mean"]:.3f}  n={row["count"]}',
                  va='center', fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(top['shape'])
    ax.set_xlabel('P(shape)')
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)


def render_p_up_chart(df: pd.DataFrame, title: str, out_png: str,
                        max_rows: int = 25):
    if df.empty:
        return
    top = df.head(max_rows)
    label_col = top.columns[0]  # group label
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top) + 2)))
    y = np.arange(len(top))[::-1]
    p = top['p_up_mean'].values
    lo = top['ci95_lo'].values
    hi = top['ci95_hi'].values
    colors = ['#43A047' if pi > 0.5 else '#E53935' for pi in p]
    ax.barh(y, p, xerr=[p - lo, hi - p], color=colors, alpha=0.85)
    ax.axvline(0.5, color='black', lw=0.6, alpha=0.5)
    for yi, (_, row) in zip(y, top.iterrows()):
        label = row[label_col] if label_col in row else 'NA'
        ax.text(row['p_up_mean'] + 0.005, yi,
                  f' p={row["p_up_mean"]:.3f}  n={row["n"]}',
                  va='center', fontsize=7)
    labels = top[label_col].astype(str).values
    if len(top.columns) > 6:  # multi-key
        keys = [c for c in top.columns
                 if c not in ('n', 'k_up', 'p_up_mean', 'ci95_lo', 'ci95_hi',
                               'mean_fwd_return')]
        labels = top[keys].astype(str).agg(' × '.join, axis=1).values
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('P(fwd_return > 0)')
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)


def process_level(level: str, csv_path: str, parent_csv: str | None,
                   out_dir: str, horizon_s: int, cache: dict):
    print(f'\n=== {level} (horizon {horizon_s}s) ===')
    df = pd.read_csv(csv_path)
    parent_df = pd.read_csv(parent_csv) if parent_csv and os.path.exists(parent_csv) else None
    df = attach_fwd_return_and_parent(df, horizon_s, parent_df, cache)
    print(f'  loaded {len(df):,}; valid fwd_return: {df["fwd_return"].notna().sum():,}')

    # Bucketize sigma by quintile (within level)
    sigma = df['mean_sigma'].copy()
    valid_sigma = sigma.dropna()
    if len(valid_sigma) >= 5:
        try:
            df['sigma_bucket'] = pd.qcut(sigma.rank(method='first'), q=5,
                                          labels=['Sq1', 'Sq2', 'Sq3', 'Sq4', 'Sq5'])
        except ValueError:
            df['sigma_bucket'] = 'Sq3'
    else:
        df['sigma_bucket'] = 'Sq3'

    # 1. shape priors
    priors = shape_priors(df)
    priors.to_csv(os.path.join(out_dir, f'{level}_priors.csv'), index=False)
    render_prior_chart(priors, f'[{level}] shape priors  (n={len(df):,})',
                        os.path.join(out_dir, f'{level}_priors.png'))

    # 2a. P_up | shape
    p_shape = p_up_given(df, ['shape'])
    p_shape.to_csv(os.path.join(out_dir, f'{level}_p_up_given_shape.csv'), index=False)
    render_p_up_chart(p_shape, f'[{level}] P(fwd>0 | shape)  horizon={horizon_s}s',
                       os.path.join(out_dir, f'{level}_p_up.png'))

    # 2b. P_up | shape × skew
    p_skew = p_up_given(df, ['shape', 'skew'])
    p_skew.to_csv(os.path.join(out_dir, f'{level}_p_up_given_shape_skew.csv'),
                   index=False)

    # 2c. P_up | shape × sigma_bucket
    df['sigma_bucket'] = df['sigma_bucket'].astype(str)
    p_sig = p_up_given(df, ['shape', 'sigma_bucket'])
    p_sig.to_csv(os.path.join(out_dir, f'{level}_p_up_given_shape_sigma.csv'),
                   index=False)

    # 2d. P_up | shape × parent_shape (if parent attached)
    if 'parent_shape' in df.columns and df['parent_shape'].notna().any():
        sub = df.dropna(subset=['parent_shape'])
        p_par = p_up_given(sub, ['shape', 'parent_shape'])
        p_par.to_csv(os.path.join(out_dir,
                                    f'{level}_p_up_given_shape_parent.csv'),
                      index=False)
        # Render top-25 most-extreme p_up cells (away from 0.5)
        if not p_par.empty:
            p_par_sorted = p_par.copy()
            p_par_sorted['extreme'] = (p_par_sorted['p_up_mean'] - 0.5).abs()
            p_par_sorted = p_par_sorted.sort_values('extreme', ascending=False).head(25)
            render_p_up_chart(p_par_sorted,
                               f'[{level}] P(fwd>0 | shape × parent_shape) — top extremes',
                               os.path.join(out_dir,
                                             f'{level}_p_up_parent_extremes.png'))

    print(f'  written priors + 4 conditional tables for {level}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default='reports/findings/segments/simple_bulk_v2')
    ap.add_argument('--out-dir', default='reports/findings/segments/bayes_tables')
    ap.add_argument('--level', default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    levels = [args.level] if args.level else list(LEVELS_HORIZONS.keys())
    cache = {}
    for level in levels:
        csv = os.path.join(args.in_dir, f'all_{level}s.csv')
        if not os.path.exists(csv):
            print(f'  skip [{level}] (no csv)'); continue
        parent_lvl = PARENT[level]
        parent_csv = (os.path.join(args.in_dir, f'all_{parent_lvl}s.csv')
                       if parent_lvl else None)
        process_level(level, csv, parent_csv, args.out_dir,
                       LEVELS_HORIZONS[level], cache)


if __name__ == '__main__':
    main()
