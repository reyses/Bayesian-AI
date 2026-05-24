"""
Two-chord analysis — distinguish NOISE from REAL TREND CHANGE.

At each zigzag pivot, compute two chord lengths over multiple windows:
  - price_chord    = |close[t] - close[t-W]|  (net price displacement)
  - regression_chord = |fit[t] - fit[t-W]|    (regression mean displacement)

Ratio interpretation:
  - reg/price ≈ 0: price moved but regression didn't → NOISE / mean-reversion
  - reg/price ≈ 1: price and regression moved together → REAL TREND CHANGE
  - in between: partial trend

Hypothesis: pivots with LOW regression-chord-ratio are the cleanest mean-
reversion setups (residual works well). Pivots with HIGH ratio are trend
continuations where residual may mispredict — mean reversion is fake here.

Stratify next-leg-direction success by the ratio to see.

Usage:
    python tools/chord_ratio_analysis.py
    python tools/chord_ratio_analysis.py --day 2025_06_09   # + chart

Output:
    reports/findings/chord_ratio_analysis.md
    charts/chord_ratio_<day>.png (if --day)
"""
import os
import sys
import glob
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.regression_line_cohen_d import zigzag_pivots, cohen_d


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
OUT_MD = 'reports/findings/chord_ratio_analysis.md'
DOLLAR_PER_POINT = 2.0
CHORD_WINDOWS = [10, 20, 60, 180]


def rolling_regression_fitted(closes, window):
    """For each bar i≥W-1, fit OLS over closes[i-W+1..i] and return fitted
    value at i. Returns array of length len(closes); NaN before window."""
    n = len(closes)
    fitted = np.full(n, np.nan)
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        x = np.arange(window, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        dx = x - xm
        denom = (dx * dx).sum()
        if denom < 1e-9:
            continue
        slope = (dx * (y - ym)).sum() / denom
        intercept = ym - slope * xm
        fitted[i] = intercept + slope * (window - 1)
    return fitted


def compute_chord_features(closes, fitted_by_W, pivot_idx, windows):
    """At pivot_idx, compute chord lengths over multiple windows.

    Two definitions of price chord:
      - price_net      = |close[t] - close[t-W]|         (straight-line)
      - price_path     = sum of |diffs| over last W bars (path length)
    And:
      - reg_chord      = |fit[t] - fit[t-W]|             (regression drift)

    Classic Kaufman Efficiency Ratio:
      efficiency = price_net / price_path  (0=chop, 1=pure trend)

    Trend-vs-noise (ours):
      reg_to_path  = reg_chord / price_path    (0=noise, ~1=strong trend)
    """
    out = {}
    for W in windows:
        if pivot_idx < W:
            return None
        # Price path length (total variation over last W bars)
        segment = closes[pivot_idx - W: pivot_idx + 1]  # W+1 points → W diffs
        price_path = float(np.sum(np.abs(np.diff(segment))))
        price_net = float(abs(closes[pivot_idx] - closes[pivot_idx - W]))
        fit = fitted_by_W[W]
        reg_chord = float(abs(fit[pivot_idx] - fit[pivot_idx - W]))
        if np.isnan(reg_chord):
            return None

        # Key ratios:
        efficiency = price_net / max(price_path, 0.1)    # Kaufman ER
        reg_to_path = reg_chord / max(price_path, 0.1)   # trend-vs-noise
        out[f'price_net_{W}'] = price_net
        out[f'price_path_{W}'] = price_path
        out[f'reg_chord_{W}'] = reg_chord
        out[f'efficiency_{W}'] = efficiency
        out[f'reg_to_path_{W}'] = reg_to_path
    return out


def process_day(path, threshold, windows):
    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    closes = df['close'].values.astype(np.float64)
    pivots = zigzag_pivots(closes, threshold)
    if len(pivots) < 3:
        return []
    # Precompute regression fits for each window
    fitted_by_W = {W: rolling_regression_fitted(closes, W) for W in windows}
    events = []
    for i in range(len(pivots) - 1):
        piv_idx = pivots[i]
        next_piv_idx = pivots[i + 1]
        leg_pts = closes[next_piv_idx] - closes[piv_idx]
        feats = compute_chord_features(closes, fitted_by_W, piv_idx, windows)
        if feats is None:
            continue
        events.append({
            'next_direction': 'UP' if leg_pts > 0 else 'DOWN',
            'leg_dollars': abs(leg_pts) * DOLLAR_PER_POINT,
            'feats': feats,
        })
    return events


def analyze(events, label):
    up = [e for e in events if e['next_direction'] == 'UP']
    down = [e for e in events if e['next_direction'] == 'DOWN']
    n_up, n_down = len(up), len(down)
    print(f'{label}: UP={n_up:,}, DOWN={n_down:,}')
    feat_names = list(events[0]['feats'].keys())
    out = {}
    for name in feat_names:
        up_vals = np.array([e['feats'][name] for e in up])
        down_vals = np.array([e['feats'][name] for e in down])
        d = cohen_d(up_vals, down_vals)
        out[name] = {
            'd': d, 'abs_d': abs(d),
            'mean_up': float(up_vals.mean()),
            'mean_down': float(down_vals.mean()),
            'median_up': float(np.median(up_vals)),
            'median_down': float(np.median(down_vals)),
        }
    return out


def stratify_by_ratio(events, ratio_key='reg_to_path_10'):
    """Bucket events by chord ratio and report per-bucket next-leg success."""
    # Buckets for reg_to_path (bounded ~[0,1] — regression chord as fraction
    # of total price path length). Low = noise, high = trend.
    bins = [
        ('VERY_NOISE (<0.05)',      0.00, 0.05),
        ('NOISE (0.05-0.15)',       0.05, 0.15),
        ('MIXED (0.15-0.30)',       0.15, 0.30),
        ('TREND (0.30-0.50)',       0.30, 0.50),
        ('STRONG_TREND (>0.50)',    0.50, float('inf')),
    ]
    out = []
    for label, lo, hi in bins:
        subset = [e for e in events if lo <= e['feats'][ratio_key] < hi]
        if len(subset) < 20:
            out.append({'label': label, 'n': len(subset), 'valid': False})
            continue
        up = sum(1 for e in subset if e['next_direction'] == 'UP')
        down = len(subset) - up
        avg_leg = np.mean([e['leg_dollars'] for e in subset])
        out.append({
            'label': label, 'n': len(subset), 'valid': True,
            'up_pct': up / len(subset) * 100,
            'down_pct': down / len(subset) * 100,
            'avg_leg': float(avg_leg),
        })
    return out


def chart_day(path, windows, out_path, threshold=15.0):
    """Visualize price + regression + both chord lengths for one day."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timezone

    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    closes = df['close'].values.astype(np.float64)
    ts = df['timestamp'].values.astype(np.float64)
    dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]
    W = 60
    fit = rolling_regression_fitted(closes, W)
    pivots = zigzag_pivots(closes, threshold)

    # Compute rolling reg_to_path ratio (regression_chord / price_path_length)
    ratios = np.full(len(closes), np.nan)
    efficiency = np.full(len(closes), np.nan)
    for i in range(W, len(closes)):
        if np.isnan(fit[i]) or np.isnan(fit[i - W]):
            continue
        segment = closes[i - W: i + 1]
        price_path = float(np.sum(np.abs(np.diff(segment))))
        price_net = abs(closes[i] - closes[i - W])
        reg_chord = abs(fit[i] - fit[i - W])
        ratios[i] = reg_chord / max(price_path, 0.1)
        efficiency[i] = price_net / max(price_path, 0.1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                    gridspec_kw={'height_ratios': [2.5, 1]},
                                    sharex=True)
    ax1.plot(dts, closes, color='black', linewidth=0.9, label='Price (1m)')
    ax1.plot(dts, fit, color='tab:blue', linewidth=2.0,
             label=f'Regression mean ({W}-bar)')
    piv_dts = [dts[i] for i in pivots]
    piv_prices = [closes[i] for i in pivots]
    ax1.scatter(piv_dts, piv_prices, c='tab:red', s=20, zorder=5,
                label=f'Zigzag pivots (${threshold:.0f})')
    ax1.set_ylabel('MNQ price', fontsize=11)
    ax1.set_title(f'Chord ratio analysis — {os.path.basename(path).replace(".parquet","")}',
                   fontsize=13)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.25)

    ax2.plot(dts, ratios, color='tab:purple', linewidth=1.0,
             label=f'reg_to_path ({W}-bar) = reg_chord / price_path_length')
    ax2.plot(dts, efficiency, color='tab:orange', linewidth=1.0, alpha=0.7,
             label='efficiency = price_net / price_path')
    ax2.axhline(0.2, color='tab:green', linestyle=':', linewidth=0.8,
                label='NOISE (<0.2)')
    ax2.axhline(0.5, color='tab:red', linestyle=':', linewidth=0.8,
                label='TREND (>0.5)')
    ax2.fill_between(dts, 0, ratios, where=(ratios < 0.2),
                      color='tab:green', alpha=0.15)
    ax2.fill_between(dts, 0, ratios, where=(ratios > 0.5),
                      color='tab:red', alpha=0.15)
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel('chord ratio', fontsize=11)
    ax2.set_xlabel('Time (UTC)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.25)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=15.0)
    ap.add_argument('--day', default=None, help='Day to visualize')
    args = ap.parse_args()

    if args.day:
        out_chart = f'charts/chord_ratio_{args.day}.png'
        path = os.path.join(ATLAS_1M_DIR, f'{args.day}.parquet')
        chart_day(path, CHORD_WINDOWS, out_chart, args.threshold)
        print(f'Wrote chart: {out_chart}')

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))

    is_events = []
    for p in tqdm(is_paths, desc='IS', unit='day'):
        is_events.extend(process_day(p, args.threshold, CHORD_WINDOWS))
    print(f'IS pivot events: {len(is_events):,}')

    oos_events = []
    for p in tqdm(oos_paths, desc='OOS', unit='day'):
        oos_events.extend(process_day(p, args.threshold, CHORD_WINDOWS))
    print(f'OOS pivot events: {len(oos_events):,}')

    is_res = analyze(is_events, 'IS')
    oos_res = analyze(oos_events, 'OOS')

    # Ratio stratification (IS + OOS separately)
    print('\n=== IS reg_to_path_10 stratification ===')
    is_strata = stratify_by_ratio(is_events, 'reg_to_path_10')
    for s in is_strata:
        if s['valid']:
            print(f'  {s["label"]:<30} n={s["n"]:>5,} '
                  f'UP%={s["up_pct"]:>5.1f} DOWN%={s["down_pct"]:>5.1f} '
                  f'avg_leg=${s["avg_leg"]:>5.0f}')
        else:
            print(f'  {s["label"]:<30} n={s["n"]:>5,} (too few)')

    print('\n=== OOS reg_to_path_10 stratification ===')
    oos_strata = stratify_by_ratio(oos_events, 'reg_to_path_10')
    for s in oos_strata:
        if s['valid']:
            print(f'  {s["label"]:<30} n={s["n"]:>5,} '
                  f'UP%={s["up_pct"]:>5.1f} DOWN%={s["down_pct"]:>5.1f} '
                  f'avg_leg=${s["avg_leg"]:>5.0f}')

    # Ratio distribution summary
    print('\n=== reg_to_path_10 distribution (IS) ===')
    ratios = np.array([e['feats']['reg_to_path_10'] for e in is_events])
    for pct in [5, 25, 50, 75, 95]:
        print(f'  p{pct}: {np.percentile(ratios, pct):.3f}')

    # MD
    out = [f'# Two-chord analysis — noise vs trend change', '']
    out.append('**Price chord** = |close[t] − close[t−W]| (net price displacement).')
    out.append('**Regression chord** = |fit[t] − fit[t−W]| (regression-mean '
               'displacement).')
    out.append('**Ratio** = reg_chord / price_chord. Near 0 → noise; near 1 → trend.')
    out.append('')
    out.append(f'Zigzag threshold: ${args.threshold}. Windows: {CHORD_WINDOWS} 1m bars.')
    out.append('')
    out.append(f'IS pivot events: {len(is_events):,} | OOS: {len(oos_events):,}')
    out.append('')

    out.append('## Chord features Cohen d (UP vs DOWN next)')
    out.append('')
    out.append('| Feature | d_IS | d_OOS | Walk-forward |')
    out.append('|---|---:|---:|---|')
    rank = sorted(is_res.keys(), key=lambda k: -is_res[k]['abs_d'])
    for name in rank:
        dis = is_res[name]['d']
        doos = oos_res[name]['d']
        wf = '✓' if (dis * doos > 0 and min(abs(dis), abs(doos)) >= 0.15) else '—'
        out.append(f'| `{name}` | {dis:+.3f} | {doos:+.3f} | {wf} |')
    out.append('')

    out.append('## IS: stratified by reg_to_path_10')
    out.append('')
    out.append('| Ratio bucket | N | UP% | DOWN% | Avg leg $ |')
    out.append('|---|---:|---:|---:|---:|')
    for s in is_strata:
        if s['valid']:
            out.append(f'| {s["label"]} | {s["n"]:,} | {s["up_pct"]:.1f}% | '
                       f'{s["down_pct"]:.1f}% | ${s["avg_leg"]:.0f} |')
    out.append('')

    out.append('## OOS: stratified by reg_to_path_10')
    out.append('')
    out.append('| Ratio bucket | N | UP% | DOWN% | Avg leg $ |')
    out.append('|---|---:|---:|---:|---:|')
    for s in oos_strata:
        if s['valid']:
            out.append(f'| {s["label"]} | {s["n"]:,} | {s["up_pct"]:.1f}% | '
                       f'{s["down_pct"]:.1f}% | ${s["avg_leg"]:.0f} |')
    out.append('')

    out.append('## reg_to_path_10 distribution (IS)')
    out.append('')
    out.append('| Percentile | ratio |')
    out.append('|---|---:|')
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        out.append(f'| p{pct} | {np.percentile(ratios, pct):.3f} |')
    out.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')


if __name__ == '__main__':
    main()
