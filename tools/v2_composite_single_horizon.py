"""
v2 Composite Directional — Single-Horizon Refit
================================================
Apples-to-apples composite. Every voter (1m, 5m, 15m, 1h, 4h) fits an OLS
predicting the SAME target: signed MFE at the common cadence (default 5m,
using the common cadence's regime-determined direction × MFE — same
target standalone L uses, just measured at one common horizon).

Why: the previous L-aggregator showed 56-60% composite accuracy because
each TF predicted its own future window (1m → 1h ahead, 1h → 8h ahead).
When projected onto 5m bars, horizons didn't align. Here we eliminate
that mismatch — every TF predicts "what's the signed MFE of the next
2-hour window starting at THIS 5m bar".

Pipeline:
  1. Load 5m OHLC. Run regime detection + oracle MFE/MAE at 5m cadence
     → produces (signed_mfe[i], regime_id[i]) for each 5m bar.
  2. For each base TF in voter set:
       a. Load 5m's v2 feature row at each 5m bar (the same source we'd
          use for that TF, just sampled at 5m cadence).
       b. Subset features to the columns belonging to that TF (23 cols).
       c. Fit OLS on (TF-features → signed_mfe at 5m cadence).
       d. Score on test split, save predictions.
  3. Composite layer: same combinators as L-aggregator (majority vote,
     strict-all, magnitude-weighted, confidence-gated).

Each voter sees only its own TF's features — that's what makes them
distinct voters. They all predict the same target, so combinator
results are now meaningful across voters.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from tools.research.data import load_atlas_tf
from tools.research.imr import compute_price_imr, detect_regimes, compute_regime_oracle
from tools.research.features_v2 import (
    TF_HIERARCHY_V2, FEATURE_NAMES_V2,
    load_v2_features, align_v2_to_base_tf,
)
from config.oracle_config import ORACLE_LOOKAHEAD_BARS


DEFAULT_VOTER_TFS = ['1m', '5m', '15m', '1h', '4h']
DEFAULT_COMMON_TF = '5m'


def get_tf_columns_in_aligned(aligned_df: pd.DataFrame, tf: str) -> list[str]:
    """Pull the 23 v2 feature column names belonging to one TF from the
    aligned-feature DataFrame.
    """
    cols = []
    for fname in FEATURE_NAMES_V2:
        if fname.endswith('_1b') or fname in ('bar_range', 'body'):
            cn = f'L1_{tf}_{fname}'
        elif fname.endswith('_w'):
            cn = f'L2_{tf}_{fname}'
        else:
            cn = f'L3_{tf}_{fname}_w'
        if cn in aligned_df.columns:
            cols.append(cn)
    return cols


def build_common_target(common_tf: str, atlas_root: str,
                        context_days: int, analysis_days: int,
                        verbose: bool = True) -> dict:
    """Run common-cadence regime detection + oracle MFE/MAE → signed_mfe target.

    Returns dict with timestamps, signed_mfe, regime_ids, etc.
    """
    base_df = load_atlas_tf(atlas_root, common_tf)
    if base_df.empty:
        raise FileNotFoundError(f"no OHLC for common TF {common_tf}")
    if verbose:
        print(f"  Common TF {common_tf}: {len(base_df):,} bars")

    price_imr = compute_price_imr(base_df, context_days, analysis_days)
    regime_ids, regime_meta = detect_regimes(price_imr)
    lookahead = ORACLE_LOOKAHEAD_BARS.get(common_tf, 16)
    bar_indices, mfes, maes, directions = compute_regime_oracle(
        base_df, regime_ids, regime_meta, lookahead=lookahead)

    if verbose:
        print(f"  Oracle: {len(mfes)} bars with MFE/MAE (lookahead={lookahead}, "
              f"{len(regime_meta)} regimes)")

    # Signed MFE = MFE * +1 if regime is LONG, -1 if SHORT
    sign_per = np.array([+1.0 if d == 'LONG' else -1.0 for d in directions])
    signed_mfe = mfes * sign_per

    base_ts = base_df['timestamp'].values.astype(np.int64)
    if pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
        base_ts = base_ts // 10**9

    target_ts = base_ts[bar_indices]

    return {
        'common_tf': common_tf,
        'base_df': base_df,
        'target_ts': target_ts,
        'signed_mfe': signed_mfe,
        'mfes': mfes,
        'maes': maes,
        'directions': directions,
        'lookahead': lookahead,
        'n_bars': len(signed_mfe),
    }


def fit_voter(tf: str, target: dict, v2_dir: str, atlas_root: str,
              verbose: bool = True) -> dict:
    """Fit OLS for one voter TF predicting common-cadence signed_mfe.

    Voter sees only its TF's 23 feature columns (sampled at common cadence).
    """
    if verbose:
        print(f"\n--- Voter: {tf} ---")

    target_ts = target['target_ts']
    Y = target['signed_mfe']

    ts_min = int(target_ts.min())
    ts_max = int(target_ts.max())
    features_5s = load_v2_features(
        v2_dir=v2_dir, atlas_root=atlas_root, day_strs=None,
        ts_range=(ts_min, ts_max), verbose=False,
    )
    if verbose:
        print(f"  v2 features: {len(features_5s):,} 5s rows")

    # Reindex features onto common-cadence target timestamps
    aligned = align_v2_to_base_tf(features_5s, target_ts)

    # Pull just this voter's TF columns
    voter_cols = get_tf_columns_in_aligned(aligned, tf)
    if not voter_cols:
        return {'tf': tf, 'error': 'no_voter_columns'}
    if verbose:
        print(f"  Voter sees {len(voter_cols)} feature columns")
    X = aligned[voter_cols].values

    # Drop rows with all-NaN voter features (occurs at the very start)
    valid = ~np.isnan(X).all(axis=1)
    X = X[valid]
    Y_v = Y[valid]
    target_ts_v = target_ts[valid]
    # NaN-fill remaining
    X = np.nan_to_num(X, nan=0.0)

    if len(Y_v) < 200:
        return {'tf': tf, 'error': f'too_few ({len(Y_v)})'}

    # Time-ordered 60/20/20
    n = len(Y_v)
    n_tr = int(n * 0.6)
    n_va = int(n * 0.2)
    X_tr, X_va, X_te = X[:n_tr], X[n_tr:n_tr + n_va], X[n_tr + n_va:]
    Y_tr, _Y_va, Y_te = Y_v[:n_tr], Y_v[n_tr:n_tr + n_va], Y_v[n_tr + n_va:]
    ts_te = target_ts_v[n_tr + n_va:]

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    ols = LinearRegression().fit(X_tr_sc, Y_tr)
    pred_te = ols.predict(X_te_sc)

    r2_te = ols.score(X_te_sc, Y_te)
    actual_dir = np.sign(Y_te)
    pred_dir = np.sign(pred_te)
    valid_dir = actual_dir != 0
    acc = float((actual_dir[valid_dir] == pred_dir[valid_dir]).mean()) \
        if valid_dir.any() else float('nan')
    base_acc = float(max((actual_dir[valid_dir] == 1).mean(),
                          (actual_dir[valid_dir] == -1).mean())) \
        if valid_dir.any() else 0.5

    if verbose:
        print(f"  Test R²: {r2_te:.4f}, acc: {acc:.1%} "
              f"(baseline {base_acc:.1%}, lift {acc-base_acc:+.1%})")

    return {
        'tf': tf,
        'n_train': n_tr,
        'n_val': n_va,
        'n_test': len(Y_te),
        'r2_test': r2_te,
        'accuracy_test': acc,
        'baseline_acc': base_acc,
        'lift': acc - base_acc,
        'pred_test': pred_te,
        'actual_test': Y_te,
        'ts_test': ts_te,
    }


def composite_evaluate(voter_results: list[dict], baseline_acc: float,
                       conf_threshold: float, verbose: bool = True) -> dict:
    """Compute combinator metrics over the (test-region, common-cadence) preds.

    All voters share target_ts → straight stack into matrix.
    """
    voters = [r for r in voter_results if 'error' not in r]
    if len(voters) < 2:
        return {'error': 'need_two_voters'}

    # Voters should already share ts_test (all from same target). Sanity check.
    ts_ref = voters[0]['ts_test']
    for v in voters[1:]:
        assert len(v['ts_test']) == len(ts_ref) and np.array_equal(v['ts_test'], ts_ref), \
            f"ts_test mismatch for voter {v['tf']}"

    pred_matrix = np.column_stack([v['pred_test'] for v in voters])
    dir_matrix = np.sign(pred_matrix).astype(np.int8)
    actual = np.sign(voters[0]['actual_test'])
    valid = actual != 0
    n_voters = len(voters)
    tf_order = [v['tf'] for v in voters]

    out = {'voters': tf_order, 'baseline_acc': baseline_acc, 'n_test': int(valid.sum())}

    # Majority vote stratified
    n_long = (dir_matrix == 1).sum(axis=1)
    n_short = (dir_matrix == -1).sum(axis=1)
    consensus = np.zeros(len(actual), dtype=np.int8)
    consensus[n_long > n_short] = 1
    consensus[n_short > n_long] = -1
    consensus_strength = np.where(consensus == 1, n_long,
                                    np.where(consensus == -1, n_short, 0))
    maj_rows = []
    for k in range(1, n_voters + 1):
        m = (consensus_strength >= k) & valid & (consensus != 0)
        if m.sum() == 0:
            continue
        sub_acc = float((consensus[m] == actual[m]).mean())
        maj_rows.append({'min_agree': k, 'n': int(m.sum()),
                         'pct_data': float(m.sum() / valid.sum() * 100),
                         'accuracy': sub_acc, 'lift': sub_acc - baseline_acc})
    out['majority'] = maj_rows
    if verbose:
        print(f"\n--- Composite: majority vote (n_voters={n_voters}) ---")
        for r in maj_rows:
            print(f"  >= {r['min_agree']} agree: n={r['n']:>5}, "
                  f"acc={r['accuracy']:.1%}, lift={r['lift']:+.1%}, "
                  f"%data={r['pct_data']:.1f}%")

    # Strict-all
    long_all = ((dir_matrix == 1) | (dir_matrix == 0)).all(axis=1) & (n_long >= 1)
    short_all = ((dir_matrix == -1) | (dir_matrix == 0)).all(axis=1) & (n_short >= 1)
    strict = np.zeros(len(actual), dtype=np.int8)
    strict[long_all] = 1
    strict[short_all] = -1
    m_strict = (strict != 0) & valid
    if m_strict.sum() > 0:
        s_acc = float((strict[m_strict] == actual[m_strict]).mean())
        out['strict'] = {'n': int(m_strict.sum()),
                         'pct_data': float(m_strict.sum() / valid.sum() * 100),
                         'accuracy': s_acc, 'lift': s_acc - baseline_acc}
        if verbose:
            print(f"\n--- Composite: strict-all ---")
            print(f"  n={out['strict']['n']:>5}, acc={s_acc:.1%}, "
                  f"lift={s_acc-baseline_acc:+.1%}, %data={out['strict']['pct_data']:.1f}%")

    # Magnitude-weighted (each voter normalized to unit std)
    stds = pred_matrix.std(axis=0, keepdims=True)
    stds = np.where(stds > 1e-9, stds, 1.0)
    weighted = (pred_matrix / stds).sum(axis=1)
    consensus_w = np.sign(weighted).astype(np.int8)
    w_rows = []
    for thr in [0.0, 1.0, 2.0, 3.0, 5.0]:
        m = (np.abs(weighted) > thr) & valid & (consensus_w != 0)
        if m.sum() == 0:
            continue
        sub_acc = float((consensus_w[m] == actual[m]).mean())
        w_rows.append({'threshold_w': thr, 'n': int(m.sum()),
                       'pct_data': float(m.sum() / valid.sum() * 100),
                       'accuracy': sub_acc, 'lift': sub_acc - baseline_acc})
    out['weighted'] = w_rows
    if verbose:
        print(f"\n--- Composite: magnitude-weighted ---")
        for r in w_rows:
            print(f"  |w|>{r['threshold_w']}: n={r['n']:>5}, "
                  f"acc={r['accuracy']:.1%}, lift={r['lift']:+.1%}, "
                  f"%data={r['pct_data']:.1f}%")

    # Confgated majority
    n_long_cg = np.zeros(len(actual), dtype=np.int32)
    n_short_cg = np.zeros(len(actual), dtype=np.int32)
    for j in range(pred_matrix.shape[1]):
        active = np.abs(pred_matrix[:, j]) > conf_threshold
        n_long_cg += (active & (dir_matrix[:, j] == 1)).astype(np.int32)
        n_short_cg += (active & (dir_matrix[:, j] == -1)).astype(np.int32)
    consensus_cg = np.zeros(len(actual), dtype=np.int8)
    consensus_cg[n_long_cg > n_short_cg] = 1
    consensus_cg[n_short_cg > n_long_cg] = -1
    m_cg = (consensus_cg != 0) & valid
    if m_cg.sum() > 0:
        cg_acc = float((consensus_cg[m_cg] == actual[m_cg]).mean())
        out['confgated'] = {'threshold': conf_threshold, 'n': int(m_cg.sum()),
                             'pct_data': float(m_cg.sum() / valid.sum() * 100),
                             'accuracy': cg_acc, 'lift': cg_acc - baseline_acc}
        if verbose:
            print(f"\n--- Composite: confgated-majority |p|>{conf_threshold} ---")
            print(f"  n={out['confgated']['n']:>5}, acc={cg_acc:.1%}, "
                  f"lift={cg_acc-baseline_acc:+.1%}, %data={out['confgated']['pct_data']:.1f}%")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--voter-tfs', nargs='+', default=DEFAULT_VOTER_TFS)
    parser.add_argument('--common-tf', default=DEFAULT_COMMON_TF)
    parser.add_argument('--context-days', type=int, default=21)
    parser.add_argument('--analysis-days', type=int, default=0)
    parser.add_argument('--conf-threshold', type=float, default=20.0)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_composite_single_horizon')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Composite — Single-Horizon Refit")
    print(f"  Common cadence: {args.common_tf}")
    print(f"  Voter TFs: {args.voter_tfs}")
    print(f"  Conf threshold: {args.conf_threshold}")
    print(f"{'='*70}")

    # 1. Build common-cadence target
    print(f"\n--- Step 1: build target at {args.common_tf} ---")
    target = build_common_target(args.common_tf, args.data,
                                   args.context_days, args.analysis_days,
                                   verbose=True)

    # 2. Fit one voter per TF
    print(f"\n--- Step 2: fit voters ---")
    voter_results = []
    for tf in args.voter_tfs:
        try:
            r = fit_voter(tf, target, v2_dir=args.cache, atlas_root=args.data,
                          verbose=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {'tf': tf, 'error': str(e)}
        voter_results.append(r)

    # Per-voter summary CSV
    rows = []
    for r in voter_results:
        if 'error' in r:
            rows.append({'tf': r['tf'], 'error': r['error']})
            continue
        rows.append({k: r[k] for k in
                     ('tf', 'n_train', 'n_test', 'r2_test',
                      'accuracy_test', 'baseline_acc', 'lift')})
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(args.output_dir, 'per_voter_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  [saved] {summary_path}")
    print(summary_df.to_string(index=False))

    # 3. Composite
    valid_voters = [r for r in voter_results if 'error' not in r]
    if len(valid_voters) >= 2:
        baseline_acc = float(np.mean([v['baseline_acc'] for v in valid_voters]))
        comp = composite_evaluate(valid_voters, baseline_acc,
                                    conf_threshold=args.conf_threshold,
                                    verbose=True)
        # Persist combinators
        if 'majority' in comp:
            pd.DataFrame(comp['majority']).to_csv(
                os.path.join(args.output_dir, 'composite_majority.csv'), index=False)
        if 'strict' in comp:
            pd.DataFrame([comp['strict']]).to_csv(
                os.path.join(args.output_dir, 'composite_strict.csv'), index=False)
        if 'weighted' in comp:
            pd.DataFrame(comp['weighted']).to_csv(
                os.path.join(args.output_dir, 'composite_weighted.csv'), index=False)
        if 'confgated' in comp:
            pd.DataFrame([comp['confgated']]).to_csv(
                os.path.join(args.output_dir, 'composite_confgated.csv'), index=False)

        # Markdown narrative
        md_path = os.path.join(args.output_dir, 'summary.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# V2 Single-Horizon Composite — "
                    f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
            f.write(f"**Common cadence:** `{args.common_tf}`\n")
            f.write(f"**Voter TFs:** {args.voter_tfs}\n")
            f.write(f"**Target:** signed MFE × regime direction at "
                    f"{args.common_tf} (lookahead={target['lookahead']} bars)\n")
            f.write(f"**Baseline (avg majority class):** {baseline_acc:.1%}\n\n")
            f.write("## Per-voter\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            f.write("## Composite — majority vote\n\n")
            f.write(pd.DataFrame(comp.get('majority', [])).to_string(index=False))
            f.write("\n\n")
            if 'strict' in comp:
                f.write("## Composite — strict-all\n\n")
                f.write(pd.DataFrame([comp['strict']]).to_string(index=False))
                f.write("\n\n")
            f.write("## Composite — magnitude-weighted\n\n")
            f.write(pd.DataFrame(comp.get('weighted', [])).to_string(index=False))
            f.write("\n\n")
            if 'confgated' in comp:
                f.write(f"## Composite — confgated (|pred| > {args.conf_threshold})\n\n")
                f.write(pd.DataFrame([comp['confgated']]).to_string(index=False))
                f.write("\n")
        print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
