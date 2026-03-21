#!/usr/bin/env python
"""
Golden Path Physics — what physics state separates real moves from noise?

Uses signal_log data (all candidates, not just traded) to find the
discriminant boundary between oracle-profitable signals and noise.

Approach:
  1. Conditional means: avg physics state for MEGA vs SCALP vs NOISE
  2. Effect size (Cohen's d): which features have largest separation
  3. Logistic regression: P(profitable) = sigmoid(w . physics)
  4. Decision boundary: thresholds that maximize precision at 98%+ WR

Usage:
  python tools/research_golden_physics.py               # IS (default)
  python tools/research_golden_physics.py --oos          # OOS
  python tools/research_golden_physics.py --save         # save report
"""
import argparse
import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd

# Physics features available in signal_log
PHYSICS_COLS = [
    'hurst', 'tunnel_prob', 'mom_rev_ratio', 'band_speed',
    'micro_z', 'macro_z', 'depth',
    'F_momentum', 'F_reversion', 'velocity', 'sigma',
]


def _load_signal_log(mode='is'):
    """Load and concatenate all signal_log shards."""
    base = os.path.join('reports', mode, 'shards')
    pattern = os.path.join(base, 'signal_log_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        single = os.path.join('reports', mode, 'signal_log.csv')
        if os.path.exists(single):
            files = [single]
    if not files:
        print(f"[ERROR] No signal_log found in {base}")
        sys.exit(1)
    frames = [pd.read_csv(f) for f in files]
    sl = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(sl):,} signal_log records from {len(files)} shards ({mode})")
    return sl


def _prep_data(sl):
    """Prepare features + targets from signal_log.

    Builds two targets:
      is_real:       binary (MEGA/SCALP=1, NOISE=0)
      signal_quality: 0-10 continuous scale based on oracle_pnl percentile
                      0 = pure noise (oracle_pnl=0)
                      10 = top 1% oracle_pnl (ideal MEGA)
    """
    # Only keep records with physics data
    available = [c for c in PHYSICS_COLS if c in sl.columns]
    if len(available) < 4:
        print(f"  [!] Only {len(available)} physics columns found: {available}")
        return None, None, None

    keep_cols = available + ['oracle_label']
    if 'oracle_pnl' in sl.columns:
        keep_cols.append('oracle_pnl')
    if 'gate' in sl.columns:
        keep_cols.append('gate')

    df = sl[keep_cols].copy()
    df = df.dropna(subset=available)

    # Binary target
    df['is_real'] = (df['oracle_label'].isin(['MEGA', 'SCALP'])).astype(int)

    # Continuous 0-10 signal quality scale
    # Based on oracle_pnl: 0=noise, linear scale to 10=top percentile
    if 'oracle_pnl' in df.columns:
        pnl = df['oracle_pnl'].clip(lower=0)
        # Top 1% of non-zero PnL defines the ceiling (score=10)
        nonzero = pnl[pnl > 0]
        if len(nonzero) > 0:
            ceiling = nonzero.quantile(0.99)
            df['signal_quality'] = (pnl / ceiling * 10).clip(0, 10).round(1)
        else:
            df['signal_quality'] = 0.0
        # Quality labels
        df['quality_bin'] = pd.cut(df['signal_quality'],
                                    bins=[-0.1, 0, 1, 3, 5, 7, 10.1],
                                    labels=['0_noise', '1_weak', '2_minor', '3_moderate', '4_strong', '5_mega'])
    else:
        df['signal_quality'] = df['is_real'].astype(float) * 5.0
        df['quality_bin'] = df['oracle_label']

    # Absolute values for directional fields
    for col in ['F_momentum', 'F_reversion', 'velocity']:
        if col in df.columns:
            df[f'abs_{col}'] = df[col].abs()

    return df, available, df['is_real']


def section_conditional_means(df, features):
    """Section 1: Physics profile across signal quality scale (0-10)."""
    print(f"\n{'='*80}")
    print(f"  SECTION 1: Physics Profile by Signal Quality (0=noise, 10=ideal MEGA)")
    print(f"{'='*80}")

    n_total = len(df)
    if 'quality_bin' in df.columns:
        print(f"\n  Signal quality distribution:")
        for qbin, grp in df.groupby('quality_bin', observed=True):
            pct = len(grp) / n_total * 100
            bar = '#' * min(40, int(pct))
            print(f"    {str(qbin):<12} {len(grp):>8,}  ({pct:>5.1f}%)  {bar}")

    # Per oracle class
    print(f"\n  Oracle class counts:")
    for cls in ['MEGA', 'SCALP', 'NOISE']:
        n = (df['oracle_label'] == cls).sum()
        print(f"    {cls:<8} {n:>8,}  ({n/n_total*100:.1f}%)")

    # Physics means by quality bin
    if 'quality_bin' in df.columns:
        bins = [b for b in df['quality_bin'].cat.categories if df[df['quality_bin'] == b].shape[0] > 0]
        print(f"\n  {'Feature':<16}", end='')
        for b in bins:
            label = str(b)[:8]
            print(f" {label:>8}", end='')
        print()
        print(f"  {'-'*16}", end='')
        for _ in bins:
            print(f" {'-'*8}", end='')
        print()

        for feat in features:
            print(f"  {feat:<16}", end='')
            for b in bins:
                mask = df['quality_bin'] == b
                val = df.loc[mask, feat].mean() if mask.sum() > 0 else 0
                print(f" {val:>8.3f}", end='')
            print()

    # MEGA focus: what makes a MEGA different from everything else?
    mega = df[df['oracle_label'] == 'MEGA']
    rest = df[df['oracle_label'] != 'MEGA']
    if len(mega) >= 10:
        print(f"\n  -- MEGA vs EVERYTHING ELSE --")
        print(f"  {'Feature':<16} {'MEGA':>10} {'Other':>10} {'Delta':>10}")
        print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
        for feat in features:
            m_mean = mega[feat].mean()
            r_mean = rest[feat].mean()
            delta = m_mean - r_mean
            marker = ' <--' if abs(delta) > 0.1 * max(abs(m_mean), abs(r_mean), 0.001) else ''
            print(f"  {feat:<16} {m_mean:>10.4f} {r_mean:>10.4f} {delta:>+10.4f}{marker}")


def section_effect_size(df, features):
    """Section 2: Cohen's d — standardized effect size for each physics feature."""
    print(f"\n{'='*80}")
    print(f"  SECTION 2: Effect Size (Cohen's d) -- Which Features Separate Best?")
    print(f"{'='*80}")
    print(f"\n  |d| > 0.8 = large, 0.5 = medium, 0.2 = small")

    real = df[df['is_real'] == 1]
    noise = df[df['is_real'] == 0]

    results = []
    for feat in features:
        r_vals = real[feat].dropna()
        n_vals = noise[feat].dropna()
        if len(r_vals) < 10 or len(n_vals) < 10:
            continue
        r_mean, r_std = r_vals.mean(), r_vals.std()
        n_mean, n_std = n_vals.mean(), n_vals.std()
        pooled_std = np.sqrt((r_std**2 + n_std**2) / 2)
        if pooled_std < 1e-12:
            continue
        d = (r_mean - n_mean) / pooled_std
        results.append({
            'feature': feat,
            'd': d,
            'abs_d': abs(d),
            'real_mean': r_mean,
            'noise_mean': n_mean,
            'direction': 'real HIGHER' if d > 0 else 'real LOWER',
        })

    results.sort(key=lambda x: -x['abs_d'])
    print(f"\n  {'Feature':<16} {'Cohen d':>8} {'|d|':>6} {'Real mean':>10} {'Noise mean':>10}  Direction")
    print(f"  {'-'*16} {'-'*8} {'-'*6} {'-'*10} {'-'*10}  {'-'*12}")
    for r in results:
        size = 'LARGE' if r['abs_d'] > 0.8 else 'MEDIUM' if r['abs_d'] > 0.5 else 'small'
        print(f"  {r['feature']:<16} {r['d']:>+8.3f} {r['abs_d']:>6.3f} "
              f"{r['real_mean']:>10.4f} {r['noise_mean']:>10.4f}  {r['direction']}  ({size})")

    return results


def section_quality_regression(df, features, export=False):
    """Section 3: Linear regression — predict continuous signal quality (0-10).

    Fits: quality = w0 + w1*feat1_scaled + w2*feat2_scaled + ...
    Exports weights to checkpoints/quality_weights.json for use in ExecutionEngine.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score, mean_absolute_error
    except ImportError:
        print("\n  [!] sklearn not available, skipping regression")
        return

    print(f"\n{'='*80}")
    print(f"  SECTION 3: Quality Regression -- quality(0-10) = w . physics")
    print(f"{'='*80}")

    if 'signal_quality' not in df.columns:
        print("  No signal_quality column — need oracle_pnl in signal_log")
        return

    X = df[features].fillna(0).values
    y = df['signal_quality'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge regression (regularized to avoid overfitting sparse features)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"\n  R-squared:  {r2:.3f}")
    print(f"  MAE:        {mae:.2f} quality points")
    if r2 < 0.05:
        print(f"  --> Weak: physics explains <5% of signal quality variance")
    elif r2 < 0.15:
        print(f"  --> Moderate: physics has some predictive power")
    else:
        print(f"  --> Strong: physics meaningfully predicts signal quality")

    # Feature weights (standardized — comparable across features)
    coefs = list(zip(features, model.coef_))
    coefs.sort(key=lambda x: -abs(x[1]))
    print(f"\n  Feature weights (standardized, positive = higher quality):")
    print(f"  {'Feature':<16} {'Weight':>8}  {'Impact':>8}")
    print(f"  {'-'*16} {'-'*8}  {'-'*8}")
    for feat, w in coefs:
        bar_len = min(30, int(abs(w) * 5))
        bar = '+' * bar_len if w > 0 else '-' * bar_len
        print(f"  {feat:<16} {w:>+8.4f}  {bar}")

    print(f"\n  Intercept: {model.intercept_:.4f}")

    # Validation: avg predicted quality by actual oracle class
    df_check = df.copy()
    df_check['pred_quality'] = y_pred
    print(f"\n  Predicted quality by oracle class:")
    for cls in ['NOISE', 'SCALP', 'MEGA']:
        mask = df_check['oracle_label'] == cls
        if mask.sum() > 0:
            avg_pred = df_check.loc[mask, 'pred_quality'].mean()
            avg_actual = df_check.loc[mask, 'signal_quality'].mean()
            print(f"    {cls:<8} predicted={avg_pred:.2f}  actual={avg_actual:.2f}  n={mask.sum():,}")

    # Top-10 discrimination: does the model rank MEGAs highest?
    df_check = df_check.sort_values('pred_quality', ascending=False)
    top_1pct = df_check.head(max(1, len(df_check) // 100))
    mega_in_top = (top_1pct['oracle_label'] == 'MEGA').sum()
    real_in_top = (top_1pct['is_real'] == 1).sum()
    print(f"\n  Top 1% by predicted quality ({len(top_1pct):,} candidates):")
    print(f"    MEGA:  {mega_in_top:,}  ({mega_in_top/len(top_1pct)*100:.1f}%)")
    print(f"    Real:  {real_in_top:,}  ({real_in_top/len(top_1pct)*100:.1f}%)")
    base_mega = (df['oracle_label'] == 'MEGA').mean()
    print(f"    Lift vs base rate: {mega_in_top/len(top_1pct)/base_mega:.1f}x" if base_mega > 0 else "")

    # Export weights
    if export:
        import json
        weights = {
            'features': features,
            'weights': {feat: round(float(w), 6) for feat, w in zip(features, model.coef_)},
            'intercept': round(float(model.intercept_), 6),
            'scaler_mean': {feat: round(float(m), 6) for feat, m in zip(features, scaler.mean_)},
            'scaler_std': {feat: round(float(s), 6) for feat, s in zip(features, scaler.scale_)},
            'r2': round(r2, 4),
            'mae': round(mae, 4),
            'n_samples': len(df),
        }
        out_path = os.path.join('checkpoints', 'quality_weights.json')
        os.makedirs('checkpoints', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(weights, f, indent=2)
        print(f"\n  ** Exported to {out_path} **")
        print(f"  Load in ExecutionEngine for physics-based score competition")


def section_golden_profile(df, features, effect_sizes):
    """Section 4: The golden path physics profile — what to look for."""
    print(f"\n{'='*80}")
    print(f"  SECTION 4: Golden Path Profile -- Target Physics State")
    print(f"{'='*80}")

    if not effect_sizes:
        print("  No effect sizes computed")
        return

    real = df[df['is_real'] == 1]
    noise = df[df['is_real'] == 0]

    # Top discriminating features
    top = [r for r in effect_sizes if r['abs_d'] > 0.1]
    if not top:
        print("\n  No features with meaningful separation (all |d| < 0.1)")
        print("  --> Physics state does NOT predict golden path")
        print("  --> Gate loosening won't help — the bottleneck is elsewhere")
        return

    print(f"\n  Features ranked by discriminative power (|d| > 0.1):\n")
    for r in top:
        feat = r['feature']
        r_med = real[feat].median()
        r_p25 = real[feat].quantile(0.25)
        r_p75 = real[feat].quantile(0.75)
        n_mean = noise[feat].mean()

        print(f"  {feat}:")
        print(f"    Golden path: median={r_med:.4f}  IQR=[{r_p25:.4f}, {r_p75:.4f}]")
        print(f"    Noise:       mean={n_mean:.4f}")
        if r['d'] > 0:
            print(f"    --> Real moves have HIGHER {feat}")
        else:
            print(f"    --> Real moves have LOWER {feat}")
        print()

    # Composite: what % of real moves satisfy all top feature thresholds?
    if len(top) >= 2:
        print(f"  -- Composite filter (all top features in golden-path IQR) --")
        mask_real = pd.Series(True, index=real.index)
        mask_all = pd.Series(True, index=df.index)
        desc = []
        for r in top[:5]:
            feat = r['feature']
            p25 = real[feat].quantile(0.25)
            p75 = real[feat].quantile(0.75)
            mask_real &= (real[feat] >= p25) & (real[feat] <= p75)
            mask_all &= (df[feat] >= p25) & (df[feat] <= p75)
            desc.append(f"{p25:.3f} <= {feat} <= {p75:.3f}")

        n_match_real = mask_real.sum()
        n_match_all = mask_all.sum()
        n_match_noise = (mask_all & (df['is_real'] == 0)).sum()
        precision = n_match_real / n_match_all if n_match_all > 0 else 0

        print(f"    Filter: {' AND '.join(desc)}")
        print(f"    Matches {n_match_all:,} total candidates ({n_match_real:,} real + {n_match_noise:,} noise)")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall: {n_match_real/len(real):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Golden Path Physics Research')
    parser.add_argument('--oos', action='store_true', help='Use OOS data (shorthand for --mode oos)')
    parser.add_argument('--mode', choices=['is', 'oos', 'oos2'], default=None,
                        help='Data mode: is, oos, or oos2')
    parser.add_argument('--save', action='store_true', help='Save report')
    parser.add_argument('--export', action='store_true',
                        help='Export quality_weights.json to checkpoints/')
    args = parser.parse_args()

    mode = args.mode or ('oos' if args.oos else 'is')
    print(f"\n  Golden Path Physics Research ({mode.upper()})")
    print(f"  {'='*50}")

    sl = _load_signal_log(mode)
    df, features, target = _prep_data(sl)
    if df is None:
        return

    section_conditional_means(df, features)
    effect_sizes = section_effect_size(df, features)
    section_quality_regression(df, features, export=args.export)
    section_golden_profile(df, features, effect_sizes)

    if args.save:
        import io
        out_dir = os.path.join('reports', 'research')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'golden_physics_{mode}.txt')
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        sl2 = _load_signal_log(mode)
        df2, feat2, _ = _prep_data(sl2)
        if df2 is not None:
            section_conditional_means(df2, feat2)
            es2 = section_effect_size(df2, feat2)
            section_quality_regression(df2, feat2, export=False)
            section_golden_profile(df2, feat2, es2)
        sys.stdout = old_stdout
        with open(out_path, 'w') as f:
            f.write(buf.getvalue())
        print(f"\n  Saved to {out_path}")


if __name__ == '__main__':
    main()
