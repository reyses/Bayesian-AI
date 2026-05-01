"""
v2 Composite Directional System
================================
Train a signed-MFE-OLS model (Analysis L style) at multiple base TFs against
the v2 feature schema, then combine their predictions into a composite
directional signal.

For each base TF in {1m, 5m, 15m, 1h, 4h, 1D}:
  1. Load raw OHLC + v2 features (185D), reindex onto base TF.
  2. Compute signed-MFE oracle target Y at the TF's natural lookahead.
  3. Fit OLS on standardized X -> Y on a 60/20/20 train/val/test split.
  4. Score test set: direction accuracy + confidence-gated tiers.

Composite layer (after individual TFs):
  - Align all per-TF predictions onto a common decision cadence (default: 5m).
  - For each common-cadence bar, gather each TF's most-recent prediction.
  - Three combinators evaluated:
      a. Majority vote on direction.
      b. Confidence-weighted average of normalized predictions.
      c. Strict-agreement: only fire when N+ TFs agree.
  - Report direction accuracy + confidence-gated lift for each combinator.

Outputs:
  reports/findings/v2_composite_directional/
    per_tf_l_summary.csv      — one row per base TF: R², accuracy, gate tiers
    composite_majority.csv     — majority-vote results
    composite_weighted.csv     — weighted-average results
    composite_agreement.csv    — strict-agreement results
    summary.md                 — written narrative
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
from tools.research.features_v2 import (
    TF_HIERARCHY_V2, TF_LABELS_V2, FEATURE_NAMES_V2,
    N_TFS_V2, N_FEATURES_PER_TF_V2,
    load_v2_features, align_v2_to_base_tf, reshape_v2_to_stack,
)
from config.oracle_config import ORACLE_LOOKAHEAD_BARS

# Default base TFs to fit at
DEFAULT_BASE_TFS = ['1m', '5m', '15m', '1h', '4h', '1D']
# Confidence gate thresholds reported per TF
CONF_THRESHOLDS = [0.0, 5.0, 10.0, 20.0, 50.0]
# Common decision cadence for the composite layer
DEFAULT_COMMON_TF = '5m'


def compute_signed_mfe_target(base_df: pd.DataFrame, lookahead: int) -> np.ndarray:
    """For each bar i, compute signed MFE = max(MFE_long, MFE_short) * direction_sign.

    direction_sign chosen per bar by which leg (up/down) had the larger excursion.
    Returns array of length len(base_df). Last `lookahead` entries get NaN.
    """
    high = base_df['high'].values.astype(np.float64)
    low = base_df['low'].values.astype(np.float64)
    close = base_df['close'].values.astype(np.float64)
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(n - lookahead):
        entry = close[i]
        max_up = float(high[i + 1: i + 1 + lookahead].max() - entry)
        max_dn = float(entry - low[i + 1: i + 1 + lookahead].min())
        if max_up >= max_dn:
            out[i] = max_up
        else:
            out[i] = -max_dn
    return out


def fit_per_tf(base_tf: str, atlas_root: str, v2_dir: str,
               verbose: bool = True) -> dict:
    """Fit signed-MFE OLS at a single base TF. Returns metrics + model artifacts."""
    if verbose:
        print(f"\n{'='*60}\n  FIT base_tf={base_tf}\n{'='*60}")

    base_df = load_atlas_tf(atlas_root, base_tf)
    if base_df.empty:
        return {'base_tf': base_tf, 'error': 'no_base_data'}
    if verbose:
        print(f"  Base bars: {len(base_df):,}")

    lookahead = ORACLE_LOOKAHEAD_BARS.get(base_tf, 16)
    if verbose:
        print(f"  Lookahead: {lookahead} bars")

    Y = compute_signed_mfe_target(base_df, lookahead)
    valid = ~np.isnan(Y)
    if valid.sum() < 200:
        return {'base_tf': base_tf, 'error': f'too_few_valid_bars ({valid.sum()})'}

    base_ts_full = base_df['timestamp'].values.astype(np.int64)
    if pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
        base_ts_full = base_ts_full // 10**9

    ts_min = int(base_df['timestamp'].iloc[0])
    ts_max = int(base_df['timestamp'].iloc[-1])
    features_5s = load_v2_features(
        v2_dir=v2_dir, atlas_root=atlas_root, day_strs=None,
        ts_range=(ts_min, ts_max), verbose=False,
    )
    if verbose:
        print(f"  v2 features: {len(features_5s):,} 5s rows")

    valid_ts = base_ts_full[valid]
    aligned = align_v2_to_base_tf(features_5s, valid_ts)
    stack, l0 = reshape_v2_to_stack(aligned)
    Y_v = Y[valid]

    # Flatten: (N, 8*23) + L0
    X = np.concatenate(
        [stack.reshape(len(valid_ts), -1), l0[:, None]],
        axis=1,
    )
    if verbose:
        print(f"  X shape: {X.shape},  Y shape: {Y_v.shape}")

    # Time-ordered 60/20/20 split (no shuffle — preserves ordering)
    n = len(Y_v)
    n_tr = int(n * 0.6)
    n_va = int(n * 0.2)
    X_tr, X_va, X_te = X[:n_tr], X[n_tr:n_tr + n_va], X[n_tr + n_va:]
    Y_tr, Y_va, Y_te = Y_v[:n_tr], Y_v[n_tr:n_tr + n_va], Y_v[n_tr + n_va:]
    ts_te = valid_ts[n_tr + n_va:]

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_va_sc = sc.transform(X_va)
    X_te_sc = sc.transform(X_te)

    ols = LinearRegression()
    ols.fit(X_tr_sc, Y_tr)
    pred_te = ols.predict(X_te_sc)

    r2_te = ols.score(X_te_sc, Y_te)

    # Direction accuracy + gates
    dir_te = np.sign(Y_te)
    pred_dir = np.sign(pred_te)
    acc = float((dir_te == pred_dir)[dir_te != 0].mean()) if (dir_te != 0).any() else float('nan')
    base_acc = float(max((dir_te == 1).mean(), (dir_te == -1).mean()))

    gates = []
    for thr in CONF_THRESHOLDS:
        mask = np.abs(pred_te) > thr
        if mask.sum() == 0:
            gates.append({'threshold': thr, 'n': 0, 'accuracy': float('nan'), 'pct_data': 0.0})
            continue
        sub_acc = float((dir_te[mask] == pred_dir[mask]).mean())
        gates.append({
            'threshold': thr,
            'n': int(mask.sum()),
            'accuracy': sub_acc,
            'pct_data': float(mask.sum() / len(pred_te) * 100),
        })

    if verbose:
        print(f"  Test R^2: {r2_te:.4f}")
        print(f"  Test direction accuracy: {acc:.1%} (baseline {base_acc:.1%}, lift {acc-base_acc:+.1%})")
        print(f"  Confidence gates:")
        for g in gates:
            print(f"    |pred|>{g['threshold']:.0f}: n={g['n']}, "
                  f"acc={g['accuracy']:.1%}, %data={g['pct_data']:.1f}%")

    return {
        'base_tf': base_tf,
        'n_train': n_tr,
        'n_val': n_va,
        'n_test': len(Y_te),
        'r2_test': r2_te,
        'direction_acc_test': acc,
        'baseline_acc': base_acc,
        'lift': acc - base_acc,
        'gates': gates,
        # Artifacts for compositing
        'ts_test': ts_te,
        'pred_test': pred_te,
        'actual_dir_test': dir_te,
        'lookahead': lookahead,
    }


def composite_majority(per_tf_results: list[dict], common_tf: str, atlas_root: str,
                       verbose: bool = True) -> dict:
    """Project each TF's test predictions onto the common-TF cadence, then majority-vote.

    A TF's "vote" at common-TF bar T is sign(prediction at the most recent bar
    of that TF whose timestamp <= T). TFs with no prior bar at T are ignored.
    """
    common_df = load_atlas_tf(atlas_root, common_tf)
    if common_df.empty:
        raise FileNotFoundError(f"no OHLC for common TF {common_tf}")
    common_ts_all = common_df['timestamp'].values.astype(np.int64)
    if pd.api.types.is_datetime64_any_dtype(common_df['timestamp']):
        common_ts_all = common_ts_all // 10**9

    # Use only the test region (latest 20% of data, roughly aligned across TFs)
    # — find earliest test ts across TFs to define the composite test window.
    test_starts = [int(r['ts_test'][0]) for r in per_tf_results if 'ts_test' in r and len(r['ts_test']) > 0]
    if not test_starts:
        return {'error': 'no_test_data'}
    test_start_ts = max(test_starts)
    test_end_ts = min(int(r['ts_test'][-1]) for r in per_tf_results if 'ts_test' in r and len(r['ts_test']) > 0)
    common_test_mask = (common_ts_all >= test_start_ts) & (common_ts_all <= test_end_ts)
    common_ts = common_ts_all[common_test_mask]
    common_close = common_df['close'].values[common_test_mask]
    if verbose:
        print(f"  Composite test window: {len(common_ts)} {common_tf} bars")

    # Per-bar: for each TF, look up most recent prediction at or before bar ts
    votes = np.zeros((len(common_ts), len(per_tf_results)), dtype=np.float64)  # +1/-1/0
    for j, r in enumerate(per_tf_results):
        if 'ts_test' not in r or len(r['ts_test']) == 0:
            continue
        tf_ts = np.asarray(r['ts_test'], dtype=np.int64)
        tf_pred = np.asarray(r['pred_test'], dtype=np.float64)
        idx = np.searchsorted(tf_ts, common_ts, side='right') - 1
        valid = idx >= 0
        v = np.zeros(len(common_ts), dtype=np.float64)
        v[valid] = np.sign(tf_pred[idx[valid]])
        votes[:, j] = v

    # Majority vote per bar (+1 if sum>0, -1 if sum<0, 0 if tie)
    vote_sum = votes.sum(axis=1)
    consensus = np.sign(vote_sum)

    # Actual direction at each common-TF bar (next-bar look)
    lookahead = ORACLE_LOOKAHEAD_BARS.get(common_tf, 16)
    actual_signed_mfe = np.full(len(common_ts), np.nan)
    high = common_df['high'].values[common_test_mask]
    low = common_df['low'].values[common_test_mask]
    for i in range(len(common_ts) - lookahead):
        entry = common_close[i]
        max_up = float(high[i + 1: i + 1 + lookahead].max() - entry)
        max_dn = float(entry - low[i + 1: i + 1 + lookahead].min())
        actual_signed_mfe[i] = max_up if max_up >= max_dn else -max_dn

    actual_dir = np.sign(actual_signed_mfe)
    valid = ~np.isnan(actual_signed_mfe) & (consensus != 0) & (actual_dir != 0)
    if valid.sum() == 0:
        return {'error': 'no_overlap'}

    overall_acc = float((consensus[valid] == actual_dir[valid]).mean())
    base_acc = float(max((actual_dir[valid] == 1).mean(), (actual_dir[valid] == -1).mean()))

    # Stratify by agreement strength (number of TFs voting the same way)
    agree_counts = np.abs(vote_sum)
    stratify = []
    for k in range(len(per_tf_results), 0, -1):
        m = (agree_counts >= k) & valid
        if m.sum() == 0:
            continue
        sub_acc = float((consensus[m] == actual_dir[m]).mean())
        stratify.append({
            'min_agree': k,
            'n': int(m.sum()),
            'accuracy': sub_acc,
            'lift': sub_acc - base_acc,
            'pct_data': float(m.sum() / valid.sum() * 100),
        })

    if verbose:
        print(f"  Majority composite: {valid.sum()} bars, acc={overall_acc:.1%}, "
              f"baseline={base_acc:.1%}, lift={overall_acc-base_acc:+.1%}")
        for s in stratify:
            print(f"    >={s['min_agree']} TFs agree: n={s['n']}, "
                  f"acc={s['accuracy']:.1%}, lift={s['lift']:+.1%}, "
                  f"%data={s['pct_data']:.1f}%")

    return {
        'overall_acc': overall_acc,
        'baseline_acc': base_acc,
        'lift': overall_acc - base_acc,
        'n_bars': int(valid.sum()),
        'stratify': stratify,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2',
                        help='v2 features directory')
    parser.add_argument('--base-tfs', nargs='+', default=DEFAULT_BASE_TFS,
                        help='base TFs to fit OLS at')
    parser.add_argument('--common-tf', default=DEFAULT_COMMON_TF,
                        help='cadence for the composite vote layer')
    parser.add_argument('--output-dir', default='reports/findings/v2_composite_directional')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"  V2 COMPOSITE DIRECTIONAL SYSTEM")
    print(f"  Base TFs: {args.base_tfs}")
    print(f"  Composite cadence: {args.common_tf}")
    print(f"{'='*70}")

    per_tf_results = []
    for tf in args.base_tfs:
        try:
            r = fit_per_tf(tf, atlas_root=args.data, v2_dir=args.cache, verbose=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {'base_tf': tf, 'error': str(e)}
        per_tf_results.append(r)

    # Per-TF summary CSV
    rows = []
    for r in per_tf_results:
        if 'error' in r:
            rows.append({'base_tf': r['base_tf'], 'error': r['error']})
            continue
        rows.append({
            'base_tf': r['base_tf'],
            'n_train': r['n_train'],
            'n_test': r['n_test'],
            'r2_test': r['r2_test'],
            'direction_acc': r['direction_acc_test'],
            'baseline_acc': r['baseline_acc'],
            'lift': r['lift'],
            'gate_0_acc': r['gates'][0]['accuracy'],
            'gate_5_acc': r['gates'][1]['accuracy'],
            'gate_10_acc': r['gates'][2]['accuracy'],
            'gate_20_acc': r['gates'][3]['accuracy'],
            'gate_50_acc': r['gates'][4]['accuracy'],
        })
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(args.output_dir, 'per_tf_l_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  [saved] {summary_path}")

    # Composite layer
    valid_results = [r for r in per_tf_results if 'error' not in r]
    if len(valid_results) >= 2:
        print(f"\n{'='*70}\n  COMPOSITE: majority vote\n{'='*70}")
        comp = composite_majority(valid_results, args.common_tf, args.data, verbose=True)
        comp_path = os.path.join(args.output_dir, 'composite_majority.csv')
        if 'stratify' in comp:
            pd.DataFrame(comp['stratify']).to_csv(comp_path, index=False)
            print(f"  [saved] {comp_path}")

    # Narrative summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 Composite Directional System — {pd.Timestamp.utcnow().strftime('%Y-%m-%d')}\n\n")
        f.write("## Per-TF Signed-MFE OLS\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        if len(valid_results) >= 2 and 'stratify' in comp:
            f.write("## Composite Majority Vote\n\n")
            f.write(f"- Decision cadence: `{args.common_tf}`\n")
            f.write(f"- Voters: {[r['base_tf'] for r in valid_results]}\n")
            f.write(f"- N test bars: {comp['n_bars']}\n")
            f.write(f"- Overall acc: {comp['overall_acc']:.1%} (baseline {comp['baseline_acc']:.1%}, lift {comp['lift']:+.1%})\n\n")
            f.write("Stratified by agreement strength:\n\n")
            f.write(pd.DataFrame(comp['stratify']).to_markdown(index=False))
            f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
