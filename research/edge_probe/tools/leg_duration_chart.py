#!/usr/bin/env python3
"""Descriptive distribution of MNQ R-trigger leg duration -> PNG for Telegram.
Histogram (bulk) + CDF, with median/mean/mode/IQR annotated. Single data hue;
mean line accented to signal it is tail-inflated. (dataviz skill: form first,
one series so no legend, color last, no dual-axis -> two panels.)"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VEC = 'research/nt8_port/atlas_backtest'
OUT = ('/tmp/claude-1000/-media-moi-WindowsCode-Bayesian-AI/'
       '3b0d97ff-121f-49a8-a569-f3e509b65820/scratchpad/leg_duration.png')

# palette (light): one data hue + ink + a warning accent for the misleading mean
BAR, IQR_FILL, GRID = '#4C78C9', '#E8EEF9', '#E5E7EB'
INK, MUTED, MEAN_C = '#1F2933', '#6B7280', '#E8833A'


def main():
    d = []
    for f in sorted(glob.glob(f'{VEC}/*.parquet')):
        v = pd.read_parquet(f, columns=['bar_ts', 'zz_confirm']).sort_values('bar_ts')
        bts = v['bar_ts'].to_numpy('int64'); zc = v['zz_confirm'].to_numpy()
        rev = np.where(zc != 0)[0]
        for k in range(1, len(rev)):
            dt = (bts[rev[k]] - bts[rev[k - 1]]) / 60.0
            if 0 < dt < 600:
                d.append(dt)
    d = np.array(d); N = len(d)
    med, mean = np.median(d), d.mean()
    q1, q3 = np.percentile(d, [25, 75])
    XMAX = 75

    plt.rcParams.update({'font.size': 13, 'axes.edgecolor': GRID,
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.2), dpi=150,
                                   gridspec_kw={'height_ratios': [2.3, 1], 'hspace': 0.32})
    fig.patch.set_facecolor('white')

    # ---- panel 1: histogram (share of legs) ----
    bins = np.arange(0, XMAX + 2, 2)
    h, edges = np.histogram(d, bins=bins)
    frac = h / N * 100
    ax1.axvspan(q1, q3, color=IQR_FILL, zorder=0)                 # IQR bulk band
    ax1.bar(edges[:-1], frac, width=1.7, align='edge', color=BAR, zorder=2)
    ax1.axvline(med, color=INK, lw=2, zorder=4)
    ax1.axvline(mean, color=MEAN_C, lw=2, ls='--', zorder=4)
    top = frac.max()
    ax1.annotate(f'median {med:.0f} min', (med, top * 0.98), color=INK, fontweight='bold',
                 fontsize=13, ha='left', xytext=(6, 0), textcoords='offset points', va='top')
    ax1.annotate(f'mean {mean:.0f} min\n(tail-inflated)', (mean, top * 0.72), color=MEAN_C,
                 fontweight='bold', fontsize=12, ha='left', xytext=(6, 0), textcoords='offset points')
    ax1.annotate('mode 4–6 min', (5, top), color=MUTED, fontsize=11, ha='center',
                 xytext=(0, 6), textcoords='offset points')
    ax1.annotate(f'middle 50%: {q1:.0f}–{q3:.0f} min', (q3, top * 0.30), color='#3F6DB0',
                 fontsize=11, ha='left', xytext=(8, 0), textcoords='offset points')
    ax1.set_title('MNQ leg duration — most legs are short; the mean is a mirage',
                  fontweight='bold', fontsize=14, loc='left', color=INK, pad=8)
    ax1.set_ylabel('% of legs'); ax1.set_xlim(0, XMAX); ax1.set_ylim(0, top * 1.18)
    ax1.grid(axis='y', color=GRID, lw=0.8); ax1.set_axisbelow(True)
    for s in ('top', 'right'):
        ax1.spines[s].set_visible(False)
    fig.text(0.5, 0.005, f'R-trigger legs (pivot→pivot) · N={N:,} · full ATLAS 2024–2026',
             ha='center', color=MUTED, fontsize=10)

    # ---- panel 2: CDF ----
    xs = np.arange(0, XMAX + 1)
    cdf = np.array([(d <= x).mean() * 100 for x in xs])
    ax2.plot(xs, cdf, color=BAR, lw=2.2)
    for xv, lbl in [(10, '40%'), (20, '66%'), (30, '78%')]:
        yv = (d <= xv).mean() * 100
        ax2.plot([xv, xv], [0, yv], color=GRID, lw=1)
        ax2.plot(xv, yv, 'o', color=BAR, ms=6, mec='white', mew=1.5, zorder=5)
        ax2.annotate(f'{lbl} ≤{xv}m', (xv, yv), color=INK, fontsize=10.5,
                     xytext=(4, -12), textcoords='offset points')
    ax2.axhline(50, color=INK, lw=1, ls=':')
    ax2.set_ylabel('cumulative %'); ax2.set_xlabel('leg duration (minutes)')
    ax2.set_xlim(0, XMAX); ax2.set_ylim(0, 100)
    ax2.grid(axis='y', color=GRID, lw=0.8); ax2.set_axisbelow(True)
    for s in ('top', 'right'):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, facecolor='white', bbox_inches='tight')
    print('saved', OUT, '| N', N, 'median', med, 'mean', round(mean, 1))


if __name__ == '__main__':
    main()
