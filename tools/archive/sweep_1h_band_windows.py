"""
Sweep 1h band windows to find the right sample size.

For each candidate window N ∈ {12, 24, 48, 60, 120, 168, 336}:
  - Compute 1h rolling OLS on every day's 1h bars
  - Measure how price interacts with the ±kσ bands at 5s resolution
  - Aggregate across all IS + OOS days

Metrics per window:
  - median 1h SE ($)          — how wide the ±1σ band actually is
  - % time price within ±1σ   — gaussian target: 68%
  - % time within ±2σ         — gaussian target: 95%
  - % time within ±3σ         — gaussian target: 99.7%
  - Daily max |1h_z_se|       — how far price breaches on an extreme day
  - SE stability (CoV)        — how variable σ itself is across days

Logic:
  A "right" window has price behavior closest to gaussian, with SE stable
  enough that bands don't flail but reactive enough to catch regime shifts.

Usage:
    python tools/sweep_1h_band_windows.py
    python tools/sweep_1h_band_windows.py --windows 24 60 120 168

Output:
    reports/findings/1h_band_window_sweep.md
    charts/1h_band_window_sweep.png
"""
import os
import sys
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATLAS_1H = 'DATA/ATLAS/1h'
ATLAS_5S = 'DATA/ATLAS/5s'
FINDINGS = 'reports/findings/1h_band_window_sweep.md'
CHART = 'charts/1h_band_window_sweep.png'

DEFAULT_WINDOWS = [12, 24, 48, 60, 120, 168, 336]


def rolling_rm_se_full(closes, window):
    """Rolling OLS. Returns arrays (rm, se), NaN where window not full."""
    n = len(closes)
    rm = np.full(n, np.nan, dtype=np.float64)
    se = np.full(n, np.nan, dtype=np.float64)
    x = np.arange(window, dtype=np.float64)
    xm = x.mean()
    dx = x - xm
    denom = float((dx * dx).sum())
    if denom < 1e-12:
        return rm, se
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        ym = y.mean()
        slope = float((dx * (y - ym)).sum() / denom)
        intercept = ym - slope * xm
        fit_last = intercept + slope * (window - 1)
        fits = intercept + slope * x
        resid = y - fits
        sigma = float(np.sqrt((resid ** 2).sum() / max(window - 2, 1)))
        rm[i] = fit_last
        se[i] = sigma
    return rm, se


def analyze_window(files_5s_paths, ts_1h_all, close_1h_all, window):
    """For each day, compute interaction stats at this window."""
    rm_1h, se_1h = rolling_rm_se_full(close_1h_all, window)

    per_day = []
    for p5s in files_5s_paths:
        day = os.path.basename(p5s).replace('.parquet', '')
        try:
            df5s = pd.read_parquet(p5s).sort_values('timestamp').reset_index(drop=True)
        except Exception:
            continue
        ts_5s = df5s['timestamp'].values.astype(np.int64)
        close_5s = df5s['close'].values.astype(np.float64)
        if len(ts_5s) == 0:
            continue
        # Step-fill 1h RM/SE onto 5s
        idx = np.searchsorted(ts_1h_all, ts_5s, side='right') - 1
        idx = np.clip(idx, 0, len(ts_1h_all) - 1)
        rm5 = rm_1h[idx]
        se5 = se_1h[idx]
        # Drop bars where SE not available yet (warm-up)
        ok = ~np.isnan(se5) & (se5 > 0)
        if ok.sum() < 100:
            continue
        rm5 = rm5[ok]
        se5 = se5[ok]
        price = close_5s[ok]
        z = (price - rm5) / se5

        within_1 = (np.abs(z) <= 1).mean()
        within_2 = (np.abs(z) <= 2).mean()
        within_3 = (np.abs(z) <= 3).mean()
        max_abs_z = np.max(np.abs(z))
        median_se = float(np.median(se5))

        per_day.append({
            'day': day,
            'median_se': median_se,
            'pct_within_1': within_1,
            'pct_within_2': within_2,
            'pct_within_3': within_3,
            'max_abs_z': max_abs_z,
        })
    return per_day


def aggregate(per_day):
    if not per_day:
        return None
    arr_se = np.array([d['median_se'] for d in per_day])
    arr_1 = np.array([d['pct_within_1'] for d in per_day])
    arr_2 = np.array([d['pct_within_2'] for d in per_day])
    arr_3 = np.array([d['pct_within_3'] for d in per_day])
    arr_max = np.array([d['max_abs_z'] for d in per_day])
    return {
        'n_days': len(per_day),
        'median_se_med': float(np.median(arr_se)),
        'median_se_p25': float(np.percentile(arr_se, 25)),
        'median_se_p75': float(np.percentile(arr_se, 75)),
        'se_cov': float(arr_se.std() / arr_se.mean()),
        'pct_within_1_mean': float(arr_1.mean()),
        'pct_within_2_mean': float(arr_2.mean()),
        'pct_within_3_mean': float(arr_3.mean()),
        'max_abs_z_p50': float(np.median(arr_max)),
        'max_abs_z_p90': float(np.percentile(arr_max, 90)),
        'max_abs_z_p99': float(np.percentile(arr_max, 99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, nargs='+', default=DEFAULT_WINDOWS)
    args = ap.parse_args()

    # Load all 1h data across all available days, concat
    print('Loading 1h history...')
    all_1h_paths = sorted(glob.glob(os.path.join(ATLAS_1H, '*.parquet')))
    df_1h_all = pd.concat([pd.read_parquet(p) for p in all_1h_paths],
                          ignore_index=True)
    df_1h_all = (df_1h_all.drop_duplicates('timestamp')
                 .sort_values('timestamp').reset_index(drop=True))
    ts_1h_all = df_1h_all['timestamp'].values.astype(np.int64)
    close_1h_all = df_1h_all['close'].values.astype(np.float64)
    print(f'1h bars: {len(ts_1h_all)}  span: '
          f'{pd.Timestamp(ts_1h_all[0], unit="s")} to '
          f'{pd.Timestamp(ts_1h_all[-1], unit="s")}')

    # 5s files for each day
    files_5s = sorted(glob.glob(os.path.join(ATLAS_5S, '*.parquet')))
    print(f'5s files: {len(files_5s)}')

    # Run sweep
    rows = []
    for W in args.windows:
        print(f'\nWindow: {W} bars ({W}h = {W/24:.1f} days)')
        per_day = analyze_window(files_5s, ts_1h_all, close_1h_all, W)
        summary = aggregate(per_day)
        if summary is None:
            print('  no data')
            continue
        rows.append({'window': W, **summary})
        print(f'  days={summary["n_days"]}  median_SE=${summary["median_se_med"]*2:.1f}  '
              f'within+/-1sig={summary["pct_within_1_mean"]*100:.0f}%  '
              f'within+/-2sig={summary["pct_within_2_mean"]*100:.0f}%  '
              f'within+/-3sig={summary["pct_within_3_mean"]*100:.0f}%  '
              f'max|z| p90={summary["max_abs_z_p90"]:.1f}')

    # Build report
    os.makedirs(os.path.dirname(FINDINGS), exist_ok=True)
    lines = []
    lines.append('# 1h band window sweep\n')
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f'1h bars used: {len(ts_1h_all)} total')
    lines.append(f'5s files scanned: {len(files_5s)}\n')
    lines.append('## Gaussian target (for reference)')
    lines.append('- 68% within ±1σ')
    lines.append('- 95% within ±2σ')
    lines.append('- 99.7% within ±3σ')
    lines.append('')
    lines.append('## Per-window results')
    lines.append('')
    lines.append('| Window (h) | Days | Median SE ($) | %within±1σ | %within±2σ | %within±3σ | max\\|z\\| p50 | max\\|z\\| p90 | SE CoV |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        lines.append(
            f'| {r["window"]} | {r["n_days"]} '
            f'| ${r["median_se_med"]*2:.1f} '
            f'| {r["pct_within_1_mean"]*100:.1f}% '
            f'| {r["pct_within_2_mean"]*100:.1f}% '
            f'| {r["pct_within_3_mean"]*100:.1f}% '
            f'| {r["max_abs_z_p50"]:.1f} '
            f'| {r["max_abs_z_p90"]:.1f} '
            f'| {r["se_cov"]:.2f} |'
        )
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    lines.append('Read the %within±1σ column. The **closer to 68%**, the more gaussian '
                 'the price distribution relative to that window. If %within±1σ is '
                 '**much higher than 68%** (e.g. 85-95%), the bands are TOO WIDE — '
                 'the window absorbs more volatility than is locally active. If '
                 '**much lower** (e.g. 40-50%), bands are TOO NARROW — window over-reacts '
                 'to recent noise and bands whip around.')
    lines.append('')

    with open(FINDINGS, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # Chart
    windows = [r['window'] for r in rows]
    within1 = [r['pct_within_1_mean']*100 for r in rows]
    within2 = [r['pct_within_2_mean']*100 for r in rows]
    within3 = [r['pct_within_3_mean']*100 for r in rows]
    median_se = [r['median_se_med']*2 for r in rows]
    max_z_p90 = [r['max_abs_z_p90'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    ax.plot(windows, within1, 'o-', label='within ±1σ', color='tab:blue')
    ax.plot(windows, within2, 's-', label='within ±2σ', color='tab:orange')
    ax.plot(windows, within3, '^-', label='within ±3σ', color='tab:red')
    # Gaussian reference lines
    for target, lbl in [(68, '±1σ target'), (95, '±2σ target'), (99.7, '±3σ target')]:
        ax.axhline(target, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Window (h)')
    ax.set_ylabel('% time price inside band')
    ax.set_title('Band coverage vs window')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log', base=2)

    ax = axes[1]
    ax.plot(windows, median_se, 'o-', color='tab:purple')
    ax.set_xlabel('Window (h)')
    ax.set_ylabel('Median 1h SE ($)')
    ax.set_title('Band width (±1σ size) vs window')
    ax.grid(alpha=0.3)
    ax.set_xscale('log', base=2)

    ax = axes[2]
    ax.plot(windows, max_z_p90, 'o-', color='tab:red')
    ax.set_xlabel('Window (h)')
    ax.set_ylabel('p90 of daily max |z|')
    ax.set_title('Extreme-day band breach magnitude')
    ax.grid(alpha=0.3)
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(CHART), exist_ok=True)
    plt.savefig(CHART, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'\nWrote: {FINDINGS}')
    print(f'Wrote: {CHART}')


if __name__ == '__main__':
    main()
