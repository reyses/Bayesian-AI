"""
v2 Composite Quantile Directional System
=========================================
Distributional alternative to v2_composite_directional.py.

The OLS approach (Analysis L) predicts the conditional MEAN of signed MFE.
That's a single point summary of a distribution that has rich structure:
the conditional spread, skew, and tail behavior carry signal that the mean
collapses. Direction sign(mean) is tradable but conservative; sign(lower
quartile) is more aggressive when the distribution is well-localized.

This tool fits per-TF quantile regressions of signed MFE on the 185D v2
feature vector at multiple quantiles {0.1, 0.25, 0.5, 0.75, 0.9}, then
defines a directional signal:

    LONG  if Q_0.25 > 0        (lower quartile is positive)
    SHORT if Q_0.75 < 0        (upper quartile is negative)
    FLAT  otherwise            (distribution straddles zero)

This makes "no signal" a first-class outcome — you only act when the
conditional distribution is unambiguous on its sign.

Composite layer: align per-TF signals onto a common cadence (default 5m)
and require strict / N-of-K agreement before firing.

Usage:
    python tools/v2_composite_quantile.py
    python tools/v2_composite_quantile.py --base-tfs 5m 15m 1h
    python tools/v2_composite_quantile.py --quick  # fewer trees, faster fits
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.preprocessing import StandardScaler

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf, reshape_v2_to_stack,
)
from config.oracle_config import ORACLE_LOOKAHEAD_BARS


DEFAULT_BASE_TFS = ['1m', '5m', '15m', '1h', '4h', '1D']
DEFAULT_COMMON_TF = '5m'
DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


# ── Quantile model backend selection ─────────────────────────────────────

def _get_quantile_backend(quick: bool = False):
    """Return a (fit_fn, name) tuple. Prefers lightgbm if available, else sklearn GBM."""
    try:
        import lightgbm as lgb

        def fit_lgb(X_tr, y_tr, X_va, y_va, alpha):
            params = {
                'objective': 'quantile',
                'alpha': alpha,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_data_in_leaf': 50,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
            }
            n_estimators = 60 if quick else 200
            model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
            return model

        return fit_lgb, 'lightgbm'
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor

        def fit_skl(X_tr, y_tr, X_va, y_va, alpha):
            n_estimators = 50 if quick else 150
            model = GradientBoostingRegressor(
                loss='quantile', alpha=alpha,
                n_estimators=n_estimators, max_depth=4,
                learning_rate=0.05, subsample=0.7, random_state=0,
            )
            model.fit(X_tr, y_tr)
            return model

        return fit_skl, 'sklearn-gbm'


# ── Target ───────────────────────────────────────────────────────────────

def compute_signed_mfe_target(base_df: pd.DataFrame, lookahead: int) -> np.ndarray:
    """For each bar i, signed MFE = max(MFE_long, MFE_short) * direction_sign."""
    high = base_df['high'].values.astype(np.float64)
    low = base_df['low'].values.astype(np.float64)
    close = base_df['close'].values.astype(np.float64)
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(n - lookahead):
        entry = close[i]
        max_up = float(high[i + 1: i + 1 + lookahead].max() - entry)
        max_dn = float(entry - low[i + 1: i + 1 + lookahead].min())
        out[i] = max_up if max_up >= max_dn else -max_dn
    return out


# ── Per-TF fit ───────────────────────────────────────────────────────────

def fit_quantile_per_tf(base_tf: str, atlas_root: str, v2_dir: str,
                        quantiles: tuple, fit_fn, backend_name: str,
                        verbose: bool = True) -> dict:
    """Fit quantile regressions at one base TF. Returns predictions + metrics."""
    if verbose:
        print(f"\n{'='*60}\n  FIT base_tf={base_tf} ({backend_name})\n{'='*60}")

    base_df = load_atlas_tf(atlas_root, base_tf)
    if base_df.empty:
        return {'base_tf': base_tf, 'error': 'no_base_data'}
    if verbose:
        print(f"  Base bars: {len(base_df):,}")

    lookahead = ORACLE_LOOKAHEAD_BARS.get(base_tf, 16)
    Y = compute_signed_mfe_target(base_df, lookahead)
    valid = ~np.isnan(Y)
    if valid.sum() < 500:
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

    X = np.concatenate(
        [stack.reshape(len(valid_ts), -1), l0[:, None]],
        axis=1,
    )

    # Time-ordered 60/20/20 split
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

    if verbose:
        print(f"  Split: train={n_tr}, val={n_va}, test={len(Y_te)}")
        print(f"  Y range: [{Y_tr.min():.1f}, {Y_tr.max():.1f}], std={Y_tr.std():.2f}")

    # Fit one model per quantile
    preds_te = {}
    for q in quantiles:
        if verbose:
            print(f"  Fitting Q_{q} ...", flush=True)
        m = fit_fn(X_tr_sc, Y_tr, X_va_sc, Y_va, q)
        preds_te[q] = m.predict(X_te_sc)

    # Distributional signal
    q_low = preds_te[0.25] if 0.25 in preds_te else preds_te[min(quantiles)]
    q_med = preds_te[0.5] if 0.5 in preds_te else preds_te[quantiles[len(quantiles)//2]]
    q_high = preds_te[0.75] if 0.75 in preds_te else preds_te[max(quantiles)]

    signal = np.zeros(len(Y_te), dtype=np.int8)
    signal[q_low > 0] = 1     # LONG: lower quartile positive
    signal[q_high < 0] = -1   # SHORT: upper quartile negative
    # else FLAT (0)

    actual_dir = np.sign(Y_te)

    # Score
    n_long = int((signal == 1).sum())
    n_short = int((signal == -1).sum())
    n_flat = int((signal == 0).sum())
    n_active = n_long + n_short

    mask_active = signal != 0
    if mask_active.sum() > 0 and (actual_dir != 0).any():
        m = mask_active & (actual_dir != 0)
        active_acc = float((signal[m] == actual_dir[m]).mean()) if m.any() else float('nan')
    else:
        active_acc = float('nan')

    base_acc = float(max((actual_dir == 1).mean(), (actual_dir == -1).mean()))

    # Mean-quantile fallback comparison (the OLS-style baseline)
    mean_dir = np.sign(q_med)
    mean_acc = float((mean_dir == actual_dir)[(actual_dir != 0)].mean()) \
        if (actual_dir != 0).any() else float('nan')

    if verbose:
        print(f"  Signal split: LONG={n_long} ({n_long/len(Y_te)*100:.1f}%) "
              f"SHORT={n_short} ({n_short/len(Y_te)*100:.1f}%) "
              f"FLAT={n_flat} ({n_flat/len(Y_te)*100:.1f}%)")
        print(f"  Active accuracy:  {active_acc:.1%} ({n_active} bars, "
              f"{n_active/len(Y_te)*100:.1f}% of test)")
        print(f"  Median-only acc:  {mean_acc:.1%} (sign of Q_0.5, all bars)")
        print(f"  Baseline (majority): {base_acc:.1%}")
        if not np.isnan(active_acc):
            print(f"  Lift on active: {active_acc - base_acc:+.1%}")

    return {
        'base_tf': base_tf,
        'n_train': n_tr,
        'n_val': n_va,
        'n_test': len(Y_te),
        'lookahead': lookahead,
        'signal_long_pct': n_long / len(Y_te) * 100,
        'signal_short_pct': n_short / len(Y_te) * 100,
        'signal_flat_pct': n_flat / len(Y_te) * 100,
        'active_n': n_active,
        'active_acc': active_acc,
        'median_only_acc': mean_acc,
        'baseline_acc': base_acc,
        'lift_active': active_acc - base_acc if not np.isnan(active_acc) else float('nan'),
        # Artifacts for compositing
        'ts_test': ts_te,
        'signal_test': signal,
        'q_med_test': q_med,
        'q_low_test': q_low,
        'q_high_test': q_high,
        'actual_dir_test': actual_dir,
    }


# ── Composite layer ──────────────────────────────────────────────────────

def composite_strict(per_tf_results: list[dict], common_tf: str, atlas_root: str,
                     verbose: bool = True) -> dict:
    """Project each TF's signals onto common cadence, score N-of-K agreement.

    For each common-TF bar, look up the most recent signal from each TF.
    Stratify accuracy by how many TFs agree (and how many were active).
    """
    common_df = load_atlas_tf(atlas_root, common_tf)
    if common_df.empty:
        return {'error': f'no_OHLC_for_{common_tf}'}
    common_ts_all = common_df['timestamp'].values.astype(np.int64)
    if pd.api.types.is_datetime64_any_dtype(common_df['timestamp']):
        common_ts_all = common_ts_all // 10**9

    # Composite test window = intersection of per-TF test ranges
    starts = [int(r['ts_test'][0]) for r in per_tf_results
              if 'ts_test' in r and len(r['ts_test']) > 0]
    ends = [int(r['ts_test'][-1]) for r in per_tf_results
            if 'ts_test' in r and len(r['ts_test']) > 0]
    if not starts:
        return {'error': 'no_test_data'}
    test_start = max(starts)
    test_end = min(ends)
    common_test_mask = (common_ts_all >= test_start) & (common_ts_all <= test_end)
    common_ts = common_ts_all[common_test_mask]
    common_close = common_df['close'].values[common_test_mask]
    common_high = common_df['high'].values[common_test_mask]
    common_low = common_df['low'].values[common_test_mask]
    if verbose:
        print(f"  Composite test window: {len(common_ts)} {common_tf} bars")

    # Project per-TF signals onto common cadence
    n_voters = len(per_tf_results)
    signal_matrix = np.zeros((len(common_ts), n_voters), dtype=np.int8)
    voter_names = []
    for j, r in enumerate(per_tf_results):
        voter_names.append(r['base_tf'])
        if 'ts_test' not in r or len(r['ts_test']) == 0:
            continue
        tf_ts = np.asarray(r['ts_test'], dtype=np.int64)
        tf_signal = np.asarray(r['signal_test'], dtype=np.int8)
        idx = np.searchsorted(tf_ts, common_ts, side='right') - 1
        valid = idx >= 0
        s = np.zeros(len(common_ts), dtype=np.int8)
        s[valid] = tf_signal[idx[valid]]
        signal_matrix[:, j] = s

    # Common-TF actual direction
    lookahead = ORACLE_LOOKAHEAD_BARS.get(common_tf, 16)
    actual_signed_mfe = np.full(len(common_ts), np.nan)
    for i in range(len(common_ts) - lookahead):
        entry = common_close[i]
        max_up = float(common_high[i + 1: i + 1 + lookahead].max() - entry)
        max_dn = float(entry - common_low[i + 1: i + 1 + lookahead].min())
        actual_signed_mfe[i] = max_up if max_up >= max_dn else -max_dn
    actual_dir = np.sign(actual_signed_mfe)

    valid_y = ~np.isnan(actual_signed_mfe) & (actual_dir != 0)

    # Combinator: take consensus = majority vote among NON-FLAT voters
    n_long_voters = (signal_matrix == 1).sum(axis=1)
    n_short_voters = (signal_matrix == -1).sum(axis=1)
    n_active_voters = n_long_voters + n_short_voters
    consensus = np.zeros(len(common_ts), dtype=np.int8)
    consensus[n_long_voters > n_short_voters] = 1
    consensus[n_short_voters > n_long_voters] = -1
    # ties -> 0 (FLAT)

    base_acc = float(max((actual_dir[valid_y] == 1).mean(), (actual_dir[valid_y] == -1).mean()))

    # Stratify by min agreement (how many TFs agreed on the consensus side)
    rows = []
    max_voters = signal_matrix.shape[1]
    for k in range(1, max_voters + 1):
        # consensus side has at least k votes
        consensus_strength = np.where(consensus == 1, n_long_voters,
                                       np.where(consensus == -1, n_short_voters, 0))
        m = (consensus_strength >= k) & valid_y & (consensus != 0)
        if m.sum() == 0:
            continue
        sub_acc = float((consensus[m] == actual_dir[m]).mean())
        rows.append({
            'min_agree': k,
            'n': int(m.sum()),
            'accuracy': sub_acc,
            'lift': sub_acc - base_acc,
            'pct_test_data': float(m.sum() / valid_y.sum() * 100),
        })

    # Strict-all variant: every non-flat voter agrees AND no voter dissents
    n_long_all = ((signal_matrix == 1) | (signal_matrix == 0)).all(axis=1) & (n_long_voters >= 1)
    n_short_all = ((signal_matrix == -1) | (signal_matrix == 0)).all(axis=1) & (n_short_voters >= 1)
    strict = np.zeros(len(common_ts), dtype=np.int8)
    strict[n_long_all] = 1
    strict[n_short_all] = -1
    m_strict = (strict != 0) & valid_y
    if m_strict.sum() > 0:
        strict_acc = float((strict[m_strict] == actual_dir[m_strict]).mean())
        strict_row = {
            'min_agree': -1,  # marker for "strict-all"
            'n': int(m_strict.sum()),
            'accuracy': strict_acc,
            'lift': strict_acc - base_acc,
            'pct_test_data': float(m_strict.sum() / valid_y.sum() * 100),
        }
        rows.append(strict_row)

    if verbose:
        print(f"  Baseline (majority class): {base_acc:.1%}")
        print(f"  Composite stratification:")
        for r in rows:
            label = 'strict-all' if r['min_agree'] == -1 else f">= {r['min_agree']} agree"
            print(f"    {label:>12}: n={r['n']:>6}, acc={r['accuracy']:.1%}, "
                  f"lift={r['lift']:+.1%}, %data={r['pct_test_data']:.1f}%")

    return {
        'baseline_acc': base_acc,
        'n_test_bars': int(valid_y.sum()),
        'voters': voter_names,
        'stratify': rows,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tfs', nargs='+', default=DEFAULT_BASE_TFS)
    parser.add_argument('--common-tf', default=DEFAULT_COMMON_TF)
    parser.add_argument('--output-dir', default='reports/findings/v2_composite_quantile')
    parser.add_argument('--quick', action='store_true',
                        help='Fewer trees per quantile fit (faster, less accurate)')
    parser.add_argument('--quantiles', nargs='+', type=float, default=list(DEFAULT_QUANTILES))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fit_fn, backend_name = _get_quantile_backend(quick=args.quick)
    print(f"{'='*70}")
    print(f"  V2 COMPOSITE QUANTILE DIRECTIONAL SYSTEM")
    print(f"  Base TFs: {args.base_tfs}")
    print(f"  Common cadence: {args.common_tf}")
    print(f"  Quantiles: {args.quantiles}")
    print(f"  Backend: {backend_name}{' (quick mode)' if args.quick else ''}")
    print(f"{'='*70}")

    quantiles = tuple(args.quantiles)
    per_tf_results = []
    for tf in args.base_tfs:
        try:
            r = fit_quantile_per_tf(tf, atlas_root=args.data, v2_dir=args.cache,
                                    quantiles=quantiles, fit_fn=fit_fn,
                                    backend_name=backend_name, verbose=True)
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
            'n_test': r['n_test'],
            'long_pct': r['signal_long_pct'],
            'short_pct': r['signal_short_pct'],
            'flat_pct': r['signal_flat_pct'],
            'active_acc': r['active_acc'],
            'median_only_acc': r['median_only_acc'],
            'baseline_acc': r['baseline_acc'],
            'lift_active': r['lift_active'],
        })
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(args.output_dir, 'per_tf_quantile_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  [saved] {summary_path}")
    print(summary_df.to_string(index=False))

    # Composite
    valid_results = [r for r in per_tf_results if 'error' not in r]
    if len(valid_results) >= 2:
        print(f"\n{'='*70}\n  COMPOSITE: strict + N-of-K agreement\n{'='*70}")
        comp = composite_strict(valid_results, args.common_tf, args.data, verbose=True)
        if 'stratify' in comp:
            comp_path = os.path.join(args.output_dir, 'composite_quantile.csv')
            pd.DataFrame(comp['stratify']).to_csv(comp_path, index=False)
            print(f"  [saved] {comp_path}")

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 Composite Quantile Directional — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d')}\n\n")
        f.write(f"**Backend:** `{backend_name}`"
                f"{'  (quick mode)' if args.quick else ''}\n\n")
        f.write(f"**Quantiles:** {list(quantiles)}\n\n")
        f.write(f"**Decision rule:** LONG if Q_0.25 > 0; "
                f"SHORT if Q_0.75 < 0; FLAT otherwise.\n\n")
        f.write("## Per-TF\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        if len(valid_results) >= 2 and 'stratify' in comp:
            f.write("## Composite\n\n")
            f.write(f"- Decision cadence: `{args.common_tf}`\n")
            f.write(f"- Voters: {comp['voters']}\n")
            f.write(f"- Test bars: {comp['n_test_bars']}\n")
            f.write(f"- Baseline (majority class): {comp['baseline_acc']:.1%}\n\n")
            f.write("Stratified by agreement strength (`-1` = strict-all):\n\n")
            f.write(pd.DataFrame(comp['stratify']).to_markdown(index=False))
            f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
