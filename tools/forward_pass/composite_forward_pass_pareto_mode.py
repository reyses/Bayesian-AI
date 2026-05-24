"""Pareto-based mode approximation for hardened forward pass P&L.

Histogram modes are noisy with sparse data (31 days). User asked for a
Pareto approximation — fit a continuous distribution to the data and
report the mode of that distribution.

Three mode estimators reported per scheme:
  1. Histogram mode      ($2 bin trade / $25 bin day, per CLAUDE.md)
  2. KDE mode            (Gaussian kernel density estimate, smooth)
  3. Pareto-shape mode   (Generalized Pareto Distribution fit, location)

GPD parameterization (scipy.stats.genpareto):
  shape (xi):  >0 = heavy-tail (Pareto-like)  =0 = exponential  <0 = bounded
  loc:         location parameter (mode for xi >= 0)
  scale:       scale parameter

For asymmetric P&L (mostly small losers + occasional big winners), we
expect xi > 0 (heavy right tail) and loc near the smallest value.

For per-day data we also fit GPD to the EXCESSES OVER A THRESHOLD
(classical Peaks-Over-Threshold approach) since GPD models excesses of
heavy-tail random variables.

NOTE: with N=31 days, GPD fits are noisy. Reporting confidence intervals
on the shape parameter would require bootstrap (not done here for brevity).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def hist_mode(arr, bin_width):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0: return 0.0
    lo = np.floor(a.min() / bin_width) * bin_width
    hi = np.ceil(a.max() / bin_width) * bin_width
    if hi <= lo: return float(a.mean())
    bins = np.arange(lo, hi + bin_width, bin_width)
    counts, edges = np.histogram(a, bins=bins)
    if counts.sum() == 0: return float(a.mean())
    k = int(counts.argmax())
    return float((edges[k] + edges[k + 1]) / 2.0)


def kde_mode(arr, bw_method=None):
    """Mode = peak of Gaussian KDE."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 3: return float('nan')
    kde = stats.gaussian_kde(a, bw_method=bw_method)
    # Evaluate KDE on a grid spanning the data range
    lo, hi = a.min(), a.max()
    grid = np.linspace(lo, hi, 2000)
    density = kde(grid)
    return float(grid[int(density.argmax())])


def gpd_fit(arr):
    """Fit Generalized Pareto Distribution. Returns dict.

    GPD is naturally a model for EXCESSES — apply to (data - min) to
    capture tail shape. shape > 0 indicates heavy right tail.
    """
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 8:
        return None
    try:
        shape, loc, scale = stats.genpareto.fit(a)
        # Pareto-mode: for genpareto with xi=shape:
        #   xi >= 0  -> density is maximized at loc (mode = location)
        #   xi <  0  -> density is monotonically decreasing from loc + scale
        if shape >= 0:
            pareto_mode = loc
        else:
            pareto_mode = loc - scale / shape   # upper endpoint
        # Median of GPD: f(x) = -loc + scale*(2^xi - 1)/xi   for xi!=0
        if abs(shape) < 1e-9:
            median_fit = loc + scale * np.log(2)
        else:
            median_fit = loc + scale * (2 ** shape - 1) / shape
        # Mean (if xi < 1): loc + scale/(1-xi)
        if shape < 1:
            mean_fit = loc + scale / (1 - shape)
        else:
            mean_fit = float('inf')
        return {
            'shape (xi)': float(shape),
            'loc': float(loc),
            'scale': float(scale),
            'pareto_mode': float(pareto_mode),
            'pareto_median': float(median_fit),
            'pareto_mean': float(mean_fit),
        }
    except Exception as e:
        return None


def gpd_pot_fit(arr, threshold_pct=80):
    """Peaks-over-threshold GPD: fit GPD to excesses above the
    `threshold_pct`-th percentile. Classical EVT approach.
    """
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return None
    thr = float(np.percentile(a, threshold_pct))
    excesses = a[a > thr] - thr
    if len(excesses) < 5:
        return None
    try:
        shape, loc, scale = stats.genpareto.fit(excesses, floc=0)
        return {
            'threshold_pct': threshold_pct,
            'threshold_value': thr,
            'n_excesses': len(excesses),
            'shape (xi)': float(shape),
            'scale': float(scale),
            'pareto_mode_above_thr': thr,
            'expected_excess_above_thr': float(scale / (1 - shape)) if shape < 1 else float('inf'),
        }
    except Exception:
        return None


def gbm_ev(pred_R):
    return float(np.clip(max(pred_R - 1.0, 0.0), 0.0, 3.0))


def gbm_quantile(pred_R, rank_pct):
    if rank_pct >= 0.80: return 2.0
    if rank_pct >= 0.50: return 1.5
    if rank_pct >= 0.20: return 0.8
    return 0.5


def hand_aggressive(zone, b6_match):
    if zone == 'AT_PIVOT' or b6_match >= 0.70: return 2.0
    elif zone in ('IMMINENT', 'NEAR_PIVOT', 'NEAR_3m', 'NEAR_5m'): return 1.2
    elif zone == 'CLEAR' and b6_match < 0.50: return 0.0
    else: return 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened.csv')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened_pareto.txt')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f'Loaded {len(df):,} legs across {df["day"].nunique()} days')

    pred = df['pred_amp_R_hardened'].values
    rank_pct = pd.Series(pred).rank(pct=True).values
    schemes = {
        'flat': np.ones(len(df)),
        'gbm_ev': np.array([gbm_ev(p) for p in pred]),
        'gbm_quantile': np.array([gbm_quantile(p, r) for p, r in zip(pred, rank_pct)]),
        'hand_aggressive': np.array([hand_aggressive(z, b)
                                       for z, b in zip(df['entry_zone'], df['entry_p_b6_match'])]),
    }

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('HARDENED FORWARD PASS — Pareto-approximated mode')
    out('=' * 78)
    out(f'Legs: {len(df):,}   Days: {df["day"].nunique()}')
    out('')

    pnl_raw = df['pnl_usd'].values

    # ============================================================
    # PER-LEG: full distribution analysis
    # ============================================================
    out('=== PER-LEG P&L modes ===')
    out(f'{"scheme":<18}  {"hist_mode":>10}  {"kde_mode":>10}  '
        f'{"GPD_mode":>10}  {"GPD_xi":>8}  {"GPD_scale":>10}  {"mean":>8}')
    for name, sizes in schemes.items():
        weighted = pnl_raw * sizes
        taken = sizes > 0
        x = weighted[taken]
        hm = hist_mode(x, bin_width=2.0)
        km = kde_mode(x)
        gpd = gpd_fit(x)
        if gpd is None:
            out(f'{name:<18}  ${hm:>+8.2f}  ${km:>+8.2f}  (GPD fit failed)')
            continue
        out(f'{name:<18}  ${hm:>+8.2f}  ${km:>+8.2f}  '
            f'${gpd["pareto_mode"]:>+8.2f}  '
            f'{gpd["shape (xi)"]:>+7.3f}  '
            f'{gpd["scale"]:>9.2f}  '
            f'${x.mean():>+6.2f}')

    out('')
    out('--- Per-leg GPD interpretation ---')
    for name, sizes in schemes.items():
        weighted = pnl_raw * sizes
        taken = sizes > 0
        x = weighted[taken]
        gpd = gpd_fit(x)
        if gpd is None: continue
        xi = gpd['shape (xi)']
        loc = gpd['loc']
        scale = gpd['scale']
        tail = ('heavy right tail' if xi > 0.1 else
                'roughly exponential' if abs(xi) <= 0.1 else
                'bounded upper tail')
        out(f'  {name:<18}  xi={xi:+.3f} ({tail})   loc=${loc:+.2f}  scale={scale:.2f}')

    # ============================================================
    # PER-DAY: same analysis at the day level
    # ============================================================
    out('')
    out('=== PER-DAY P&L modes ($25 bin histogram, KDE, GPD fit) ===')
    out(f'{"scheme":<18}  {"hist_mode":>10}  {"kde_mode":>10}  '
        f'{"GPD_mode":>10}  {"GPD_xi":>8}  {"GPD_scale":>10}  {"mean":>10}')
    for name, sizes in schemes.items():
        df_copy = df.copy()
        df_copy['wpnl'] = pnl_raw * sizes
        per_day = df_copy.groupby('day')['wpnl'].sum().values
        hm = hist_mode(per_day, bin_width=25.0)
        km = kde_mode(per_day)
        gpd = gpd_fit(per_day)
        if gpd is None:
            out(f'{name:<18}  ${hm:>+8.2f}  ${km:>+8.2f}  (GPD fit failed)')
            continue
        out(f'{name:<18}  ${hm:>+8.2f}  ${km:>+8.2f}  '
            f'${gpd["pareto_mode"]:>+8.2f}  '
            f'{gpd["shape (xi)"]:>+7.3f}  '
            f'{gpd["scale"]:>9.2f}  '
            f'${per_day.mean():>+8.2f}')

    out('')
    out('--- Per-day GPD interpretation ---')
    for name, sizes in schemes.items():
        df_copy = df.copy()
        df_copy['wpnl'] = pnl_raw * sizes
        per_day = df_copy.groupby('day')['wpnl'].sum().values
        gpd = gpd_fit(per_day)
        if gpd is None: continue
        xi = gpd['shape (xi)']
        loc = gpd['loc']
        scale = gpd['scale']
        tail = ('heavy right tail' if xi > 0.1 else
                'roughly exponential' if abs(xi) <= 0.1 else
                'bounded upper tail')
        out(f'  {name:<18}  xi={xi:+.3f} ({tail})  loc=${loc:+.0f}  scale=${scale:.0f}')

    # ============================================================
    # POT (peaks-over-threshold) at p80 for per-day
    # ============================================================
    out('')
    out('=== POT (Peaks-Over-Threshold p80) per-day analysis ===')
    out('  Classical EVT: fits GPD to excesses above the 80th percentile.')
    out(f'  {"scheme":<18}  {"threshold":>10}  {"n_excess":>9}  '
        f'{"xi":>8}  {"scale":>9}  {"expected_excess":>15}')
    for name, sizes in schemes.items():
        df_copy = df.copy()
        df_copy['wpnl'] = pnl_raw * sizes
        per_day = df_copy.groupby('day')['wpnl'].sum().values
        pot = gpd_pot_fit(per_day, threshold_pct=80)
        if pot is None:
            out(f'  {name:<18}  POT fit failed')
            continue
        out(f'  {name:<18}  ${pot["threshold_value"]:>+8.2f}  {pot["n_excesses"]:>9}  '
            f'{pot["shape (xi)"]:>+7.3f}  {pot["scale"]:>9.2f}  '
            f'${pot["expected_excess_above_thr"]:>+12.2f}')

    out('')
    out('Interpretation:')
    out('  - xi > 0.5 means VERY heavy tail (a few days can dominate total P&L)')
    out('  - xi ~ 0   means well-behaved exponential decay')
    out('  - For trading: high xi means the mean is fragile, mode is more trustworthy')
    out('  - For gbm_ev: if GPD xi > 0.3, the strategy is "lottery-like" — most days')
    out('    near the mode, occasional big winners drive the mean. Median > mode is')
    out('    expected.')

    # ============================================================
    # Honest synthesis
    # ============================================================
    out('')
    out('=' * 78)
    out('HONEST SYNTHESIS')
    out('=' * 78)
    out('For trading planning purposes, the appropriate "typical day" estimate')
    out('depends on the GPD xi:')
    out('  - Low xi (xi < 0.2): mean is a reasonable expectation')
    out('  - High xi (xi > 0.3): mode/median is more honest; mean is right-tail driven')

    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
