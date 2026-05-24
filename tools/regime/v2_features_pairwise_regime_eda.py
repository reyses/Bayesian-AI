"""
v2_features_pairwise_regime_eda.py — Pair interaction stratified by regime.

User question: "if we are in a high variable lower TF (bigger bars) do
features change their behavior or the way they interact with each other?".

Two structural questions per pair:

  1. Does corr(X, Y) — the LINEAR INTERACTION between two features —
     change across regimes? If yes, the features are not "independent
     siblings" but regime-dependent relatives.

  2. Does the JOINT CELL pattern change? When binning (X, Y) into
     (X_q, Y_q), does the cell that predicts UP move in one regime
     predict DOWN in another? Does the interaction-RMS (non-additivity)
     differ?

For each (pair, regime) cell:
  corr_xy            Pearson r between X and Y within that regime
  interaction_rms    Within-regime non-additivity score (Layer A2 metric)
  max_abs_cell_fwd   Max |mean_fwd| across the 9 (X_q, Y_q) cells
  best_cell_id       Which (X_q, Y_q) cell carries the strongest signal

Top output: pairs with the BIGGEST corr(X, Y) range across regimes —
these are pairs whose relationship STRUCTURE changes character by
regime, not just whose price reaction changes.

Pruning: top-12 features from Layer 1 by Cohen-d → C(12, 2) = 66 pairs ×
6 regimes × 9 cells = 3,564 cells.

Outputs:
  reports/findings/v2_features_pairwise_regime/
    pair_regime_summary.csv        — long form (pair, regime, corr_xy, etc.)
    pivot_corr_xy.csv              — pair × regime matrix of corr(X,Y)
    top_structural_changers.csv    — pairs whose corr(X,Y) flips sign or has biggest range
    cells/<pair>__<regime>.csv     — per-cell detail for each (pair, regime) in top changers
    summary.md
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
DEFAULT_QUANTILES = 3
DEFAULT_TOP_K = 12


def quantile_bins_within(values: np.ndarray, q: int) -> np.ndarray:
    """Quantile bin labels using only the SUBSET of values passed in.

    Important: bins are computed within the regime so the cells are regime-
    relative (not global). Otherwise rare regimes would have empty cells
    when binned by global quantiles.
    """
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


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 30:
        return float('nan')
    a = a[mask]; b = b[mask]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def analyze_pair_within_regime(x: np.ndarray, y: np.ndarray, fwd: np.ndarray,
                                  q: int, min_cell_n: int) -> dict:
    """Per-(pair, regime) stats. x/y/fwd are already filtered to the regime."""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(fwd))
    if valid.sum() < q * q * min_cell_n:
        return {'corr_xy': float('nan'), 'interaction_rms': float('nan'),
                'max_abs_cell_fwd': float('nan'), 'best_cell': None,
                'cells': pd.DataFrame(), 'n': int(valid.sum())}

    x_v = x[valid]; y_v = y[valid]; fwd_v = fwd[valid]

    # Linear inter-feature correlation within regime
    corr_xy = safe_corr(x_v, y_v)

    # Quantile bins within regime
    bx = quantile_bins_within(x_v, q)
    by = quantile_bins_within(y_v, q)

    cells = []
    cell_means = {}
    global_mean = float(np.mean(fwd_v))

    # Marginals
    marg_x = {bi: float(fwd_v[bx == bi].mean())
              if (bx == bi).sum() > 0 else global_mean
              for bi in range(q)}
    marg_y = {bi: float(fwd_v[by == bi].mean())
              if (by == bi).sum() > 0 else global_mean
              for bi in range(q)}

    interaction_sq_sum = 0.0
    n_cells = 0
    for ix in range(q):
        for iy in range(q):
            mask = (bx == ix) & (by == iy)
            n_cell = int(mask.sum())
            if n_cell < min_cell_n:
                continue
            f = fwd_v[mask]
            mean_fwd = float(f.mean())
            wr = float((f > 0).mean())
            additive_pred = marg_x[ix] + marg_y[iy] - global_mean
            interaction = mean_fwd - additive_pred
            interaction_sq_sum += interaction ** 2
            n_cells += 1
            cells.append({
                'x_q': ix, 'y_q': iy, 'n': n_cell,
                'mean_fwd': mean_fwd, 'win_rate': wr,
                'interaction': interaction,
            })
    cells_df = pd.DataFrame(cells)
    interaction_rms = (np.sqrt(interaction_sq_sum / max(n_cells, 1))
                       if n_cells > 0 else float('nan'))

    if len(cells_df) > 0:
        max_abs_idx = cells_df['mean_fwd'].abs().idxmax()
        best_cell = (int(cells_df.loc[max_abs_idx, 'x_q']),
                     int(cells_df.loc[max_abs_idx, 'y_q']))
        max_abs_cell_fwd = float(cells_df.loc[max_abs_idx, 'mean_fwd'])
    else:
        best_cell = None
        max_abs_cell_fwd = float('nan')

    return {
        'corr_xy': corr_xy,
        'interaction_rms': interaction_rms,
        'max_abs_cell_fwd': max_abs_cell_fwd,
        'best_cell': best_cell,
        'cells': cells_df,
        'n': int(valid.sum()),
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
                        choices=['cohen_d', 'lookback_corr', 'forward_corr'])
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--quantiles', type=int, default=DEFAULT_QUANTILES)
    parser.add_argument('--forward-n', type=int, default=DEFAULT_FORWARD_N)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--min-cell-n', type=int, default=20)
    parser.add_argument('--top-changers-to-detail', type=int, default=10)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_pairwise_regime')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cells_dir = os.path.join(args.output_dir, 'cells')
    os.makedirs(cells_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"  V2 Features: Pairwise interaction × regime")
    print(f"  Base TF: {args.base_tf}  Split: {args.split}")
    print(f"  Top-K: {args.top_k} -> C(K,2)={args.top_k*(args.top_k-1)//2} pairs "
          f"× {len(REGIME_2D_ORDER)} regimes")
    print(f"  Quantiles: {args.quantiles}  Min cell n: {args.min_cell_n}")
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

    pairs = list(combinations(shortlist, 2))
    print(f"\n--- Analyzing {len(pairs)} pairs × {len(REGIME_2D_ORDER)} regimes ---")

    pair_regime_rows = []
    cell_dfs = {}  # (pair, regime) -> cells_df

    for x_name, y_name in pairs:
        x = full[x_name].values.astype(np.float64)
        y = full[y_name].values.astype(np.float64)
        for regime in REGIME_2D_ORDER:
            mask = regimes == regime
            if mask.sum() < args.quantiles * args.quantiles * args.min_cell_n:
                continue
            r = analyze_pair_within_regime(x[mask], y[mask], fwd[mask],
                                            args.quantiles, args.min_cell_n)
            if np.isnan(r['corr_xy']):
                continue
            pair_regime_rows.append({
                'x_feature': x_name,
                'y_feature': y_name,
                'pair': f'{x_name}__x__{y_name}',
                'regime_2d': regime,
                'n': r['n'],
                'corr_xy': r['corr_xy'],
                'interaction_rms': r['interaction_rms'],
                'max_abs_cell_fwd': r['max_abs_cell_fwd'],
                'best_cell_xq': r['best_cell'][0] if r['best_cell'] else -1,
                'best_cell_yq': r['best_cell'][1] if r['best_cell'] else -1,
            })
            cell_dfs[(x_name, y_name, regime)] = r['cells']

    pr_df = pd.DataFrame(pair_regime_rows)
    pr_path = os.path.join(args.output_dir, 'pair_regime_summary.csv')
    pr_df.to_csv(pr_path, index=False)
    print(f"  [saved] {pr_path} ({len(pr_df)} (pair, regime) rows)")

    # Pivot: pair × regime → corr_xy
    pv_corr = pr_df.pivot(index='pair', columns='regime_2d', values='corr_xy')
    pv_corr = pv_corr.reindex(columns=[r for r in REGIME_2D_ORDER
                                          if r in pv_corr.columns])
    pv_corr_path = os.path.join(args.output_dir, 'pivot_corr_xy.csv')
    pv_corr.to_csv(pv_corr_path)
    print(f"  [saved] {pv_corr_path}")

    # Pivot: pair × regime → interaction_rms
    pv_irms = pr_df.pivot(index='pair', columns='regime_2d',
                            values='interaction_rms')
    pv_irms = pv_irms.reindex(columns=[r for r in REGIME_2D_ORDER
                                          if r in pv_irms.columns])
    pv_irms_path = os.path.join(args.output_dir, 'pivot_interaction_rms.csv')
    pv_irms.to_csv(pv_irms_path)

    # Identify pairs whose corr(X, Y) FLIPS sign across regimes
    print(f"\n--- Structural changers: pairs whose corr(X,Y) flips/swings across regimes ---")
    structural_rows = []
    for pair_name, row in pv_corr.iterrows():
        vals = row.dropna().values
        if len(vals) < 2:
            continue
        signs = np.sign(vals)
        n_pos = int((signs > 0).sum())
        n_neg = int((signs < 0).sum())
        has_flip = bool(n_pos > 0 and n_neg > 0)
        structural_rows.append({
            'pair': pair_name,
            'n_regimes': len(vals),
            'min_corr_xy': float(vals.min()),
            'max_corr_xy': float(vals.max()),
            'corr_xy_range': float(vals.max() - vals.min()),
            'sign_flip': has_flip,
            'n_pos_regimes': n_pos,
            'n_neg_regimes': n_neg,
        })
    s_df = pd.DataFrame(structural_rows).sort_values(
        'corr_xy_range', ascending=False)
    s_path = os.path.join(args.output_dir, 'top_structural_changers.csv')
    s_df.to_csv(s_path, index=False)
    print(f"  [saved] {s_path}")

    print(f"\n  Top 20 pairs with biggest corr(X,Y) range across regimes:")
    print(f"    {'pair':>62}  {'n_reg':>5}  {'min_r':>6} {'max_r':>6} "
          f"{'range':>5}  {'flip':>4}")
    for _, r in s_df.head(20).iterrows():
        print(f"    {r['pair'][:62]:>62}  {int(r['n_regimes']):>5}  "
              f"{r['min_corr_xy']:>+6.3f} {r['max_corr_xy']:>+6.3f} "
              f"{r['corr_xy_range']:>5.3f}  {'YES' if r['sign_flip'] else 'no':>4}")

    print(f"\n  Pairs whose corr(X,Y) FLIPS sign across regimes:")
    flippers = s_df[s_df['sign_flip']].head(15)
    for _, r in flippers.iterrows():
        print(f"    {r['pair'][:60]:>60}  pos={int(r['n_pos_regimes'])}  "
              f"neg={int(r['n_neg_regimes'])}  range={r['corr_xy_range']:.3f}")

    # Save per-(pair, regime) cell detail for top changers
    print(f"\n--- Detail tables for top {args.top_changers_to_detail} structural changers ---")
    saved = 0
    for _, r in s_df.head(args.top_changers_to_detail).iterrows():
        pair_name = r['pair']
        x_feat, y_feat = pair_name.split('__x__')
        # Save one CSV per regime, plus a combined view
        combined = []
        for regime in REGIME_2D_ORDER:
            cells = cell_dfs.get((x_feat, y_feat, regime), pd.DataFrame())
            if len(cells) == 0:
                continue
            cells = cells.copy()
            cells['regime_2d'] = regime
            combined.append(cells)
        if combined:
            comb_df = pd.concat(combined, ignore_index=True)
            slug = pair_name.replace('/', '_')[:120]
            comb_path = os.path.join(cells_dir, f'{slug}.csv')
            comb_df.to_csv(comb_path, index=False)
            saved += 1
    print(f"  [saved] {saved} pair detail tables in {cells_dir}/")

    # Drilldown print: top 5 changers — per-regime corr_xy + best cell
    print(f"\n--- Top 5 structural-changer drilldowns ---")
    for _, r in s_df.head(5).iterrows():
        pair_name = r['pair']
        x_feat, y_feat = pair_name.split('__x__')
        print(f"\n  PAIR: {x_feat} x {y_feat}")
        print(f"    corr(X,Y) range across regimes: {r['min_corr_xy']:+.3f} to "
              f"{r['max_corr_xy']:+.3f}  (flip: {r['sign_flip']})")
        sub = pr_df[(pr_df['x_feature'] == x_feat) &
                    (pr_df['y_feature'] == y_feat)].sort_values('regime_2d')
        for _, pr_r in sub.iterrows():
            print(f"    {pr_r['regime_2d']:>14}  n={int(pr_r['n']):>5}  "
                  f"corr(X,Y)={pr_r['corr_xy']:+.3f}  "
                  f"interaction_rms={pr_r['interaction_rms']:+.3f}  "
                  f"max|cell_fwd|={pr_r['max_abs_cell_fwd']:+.2f}  "
                  f"best_cell=(Q{int(pr_r['best_cell_xq'])},"
                  f"Q{int(pr_r['best_cell_yq'])})")

    # Markdown
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features: pairwise interaction × regime — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Base TF:** `{args.base_tf}`  **Split:** `{args.split}`  "
                f"**Top-K:** {args.top_k}  **Quantiles:** {args.quantiles}\n\n")
        f.write("## Top 30 pairs by corr(X,Y) range across regimes\n\n")
        f.write(s_df.head(30).to_string(index=False))
        f.write("\n\n## corr(X,Y) pivot — pair × regime\n\n")
        f.write(pv_corr.round(3).to_string())
        f.write("\n\n## interaction_rms pivot — pair × regime\n\n")
        f.write(pv_irms.round(3).to_string())
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
