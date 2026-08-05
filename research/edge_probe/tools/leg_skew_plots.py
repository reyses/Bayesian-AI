#!/usr/bin/env python3
"""Visualize the skew/detrend leg-structure findings (owner 2026-07-28, TG: 'plots
not words, show the example then the plots'). Two PNGs:
  example.png  — one real day: raw price + 30min drift + raw cusps (drifting/skewed),
                 then the DETRENDED series with cusps ALIGNED in a symmetric band.
  findings.png — reversal hazard vs leg_pos (RAW flat/memoryless vs DETRENDED plateau)
                 + consecutive-leg amplitude autocorr (raw r vs detrended r).
dataviz skill: 2 series -> legend + direct labels + linestyle (identity not color-alone);
one hue per series (detrended=blue the finding, raw=amber the artifact); no dual-axis.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, 'research/dojo_forge/tools')
import cubic_regression as cub  # noqa: E402

A1 = 'DATA/ATLAS/1m'; A5 = 'DATA/ATLAS/5s'; R = 10.0
SD = ('/tmp/claude-1000/-media-moi-WindowsCode-Bayesian-AI/'
      '3b0d97ff-121f-49a8-a569-f3e509b65820/scratchpad/')
DET, RAWC, INK, MUTED, GRID, BAND = '#4C78C9', '#E8833A', '#1F2933', '#6B7280', '#E5E7EB', '#E8EEF9'
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': GRID, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})


def zigzag(x, R):
    n = len(x); hi = lo = x[0]; hii = loi = 0; dirn = 0; out = []
    for i in range(1, n):
        p = x[i]
        if np.isnan(p):
            continue
        if dirn >= 0 and p > hi: hi, hii = p, i
        if dirn <= 0 and p < lo: lo, loi = p, i
        if dirn >= 0 and hi - p >= R: out.append((hii, hi, 'peak')); dirn = -1; lo, loi = p, i
        elif dirn <= 0 and p - lo >= R: out.append((loi, lo, 'trough')); dirn = 1; hi, hii = p, i
    return out


def load_day(day):
    m = pd.read_parquet(f'{A1}/{day}.parquet', columns=['timestamp', 'close'])
    ts = m['timestamp'].astype('int64').to_numpy(); c = m['close'].astype(float).to_numpy()
    d5 = pd.read_parquet(f'{A5}/{day}.parquet', columns=['timestamp', 'close']).sort_values('timestamp')
    t5 = d5['timestamp'].astype('int64').to_numpy()
    val, _, _ = cub.rolling(d5['close'].astype(float).to_numpy(), 360, 5)
    k = np.searchsorted(t5, ts, side='right') - 1
    drift = np.where(k >= 0, val[np.clip(k, 0, len(val) - 1)], np.nan)
    return c, drift, c - drift


def hazard_and_amps(days):
    def collect(price_of):
        amps = []; hz = []
        for c, drift, det in days:
            x = price_of(c, det)
            piv = zigzag(np.where(np.isnan(x), np.nanmean(x), x), R)
            for a in range(2, len(piv)):
                i0, v0 = piv[a - 1][0], piv[a - 1][1]; i1, v1 = piv[a][0], piv[a][1]
                pamp = abs(piv[a - 1][1] - piv[a - 2][1])
                if pamp <= 0:
                    continue
                amps.append((abs(v1 - v0), pamp))
                for b in range(i0 + 1, i1):
                    hz.append((abs(x[b] - v0) / pamp, int((i1 - b) <= 3)))
        return np.array(amps), np.array(hz)
    return collect(lambda c, det: c), collect(lambda c, det: det)


def hz_curve(HZ, bins):
    xs, ys = [], []
    for i in range(len(bins) - 1):
        m = (HZ[:, 0] >= bins[i]) & (HZ[:, 0] < bins[i + 1])
        if m.sum() > 200:
            xs.append((bins[i] + bins[i + 1]) / 2); ys.append(HZ[m, 1].mean() * 100)
    return xs, ys


def fig_example(day):
    c, drift, det = load_day(day)
    x = np.arange(len(c))
    praw = zigzag(c, R); pdet = zigzag(np.where(np.isnan(det), np.nanmean(det), det), R)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, gridspec_kw={'hspace': 0.28})
    fig.patch.set_facecolor('white')
    # panel A: raw price + drift + raw cusps
    a1.plot(x, c, color='#90A4AE', lw=1.1, label='price (1m)')
    a1.plot(x, drift, color=RAWC, lw=2, label='30-min drift (skew)')
    for i, v, kind in praw:
        a1.scatter(i, v, marker='v' if kind == 'peak' else '^',
                   color='#C62828' if kind == 'peak' else '#2E7D32', s=55, zorder=5)
    pk = [v for _, v, k in praw if k == 'peak']; trg = [v for _, v, k in praw if k == 'trough']
    if pk: a1.axhline(np.mean(pk), color='#C62828', ls=':', lw=1, alpha=0.5)
    if trg: a1.axhline(np.mean(trg), color='#2E7D32', ls=':', lw=1, alpha=0.5)
    a1.set_title(f'RAW price ({day}) — cusps DRIFT with the skew, no aligned band',
                 fontweight='bold', fontsize=13, loc='left', color=INK)
    a1.set_ylabel('price'); a1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    a1.grid(color=GRID, lw=0.7); a1.set_axisbelow(True)
    for s in ('top', 'right'): a1.spines[s].set_visible(False)
    # panel B: detrended + aligned cusps + band
    pkv = [v for _, v, k in pdet if k == 'peak']; trv = [v for _, v, k in pdet if k == 'trough']
    mp = np.mean(pkv) if pkv else 15; mt = np.mean(trv) if trv else -15
    a2.axhspan(mt, mp, color=BAND, zorder=0, label=f'aligned band [{mt:.0f}, {mp:.0f}] pt')
    a2.axhline(0, color=INK, lw=1)
    a2.plot(x, det, color=DET, lw=1.2)
    for i, v, kind in pdet:
        a2.scatter(i, v, marker='v' if kind == 'peak' else '^',
                   color='#C62828' if kind == 'peak' else '#2E7D32', s=55, zorder=5)
    a2.axhline(mp, color='#C62828', ls=':', lw=1.2); a2.axhline(mt, color='#2E7D32', ls=':', lw=1.2)
    a2.set_title('DETRENDED (price − drift) — cusps ALIGN in a symmetric band',
                 fontweight='bold', fontsize=13, loc='left', color=INK)
    a2.set_ylabel('price − drift (pt)'); a2.set_xlabel('bar (minute of session)')
    a2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    a2.grid(color=GRID, lw=0.7); a2.set_axisbelow(True)
    for s in ('top', 'right'): a2.spines[s].set_visible(False)
    fig.savefig(SD + 'example.png', facecolor='white', bbox_inches='tight')
    print('saved example.png for', day)


def fig_findings(days):
    (RA, RH), (DA, DH) = hazard_and_amps(days)
    bins = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5])
    rx, ry = hz_curve(RH, bins); dx, dy = hz_curve(DH, bins)
    rr = np.corrcoef(RA[:, 0], RA[:, 1])[0, 1]; dr = np.corrcoef(DA[:, 0], DA[:, 1])[0, 1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150, gridspec_kw={'wspace': 0.28})
    fig.patch.set_facecolor('white')
    # panel A: hazard curves
    a1.plot(rx, ry, 's--', color=RAWC, lw=2, ms=7, label='RAW (looks memoryless)')
    a1.plot(dx, dy, 'o-', color=DET, lw=2.4, ms=7, label='DETRENDED (real plateau)')
    a1.axvline(1.0, color=MUTED, ls=':', lw=1)
    a1.annotate('expected\namplitude', (1.0, 20), color=MUTED, fontsize=9, ha='center')
    a1.annotate(f'{dy[-1]:.0f}%', (dx[-1], dy[-1]), color=DET, fontweight='bold',
                xytext=(4, 2), textcoords='offset points')
    a1.set_title('Reversal hazard vs how far into the leg', fontweight='bold', fontsize=13, loc='left')
    a1.set_xlabel('leg_pos = displacement / expected amplitude')
    a1.set_ylabel('P(leg reverses within 3 bars) %')
    a1.legend(loc='upper left', fontsize=10); a1.grid(color=GRID, lw=0.7); a1.set_axisbelow(True)
    a1.set_ylim(0, 95)
    for s in ('top', 'right'): a1.spines[s].set_visible(False)
    # panel B: amplitude autocorr (hexbin detrended) with r's
    idx = np.random.default_rng(0).choice(len(DA), min(6000, len(DA)), replace=False)
    a2.scatter(DA[idx, 1], DA[idx, 0], s=6, color=DET, alpha=0.25, label=f'detrended  r={dr:.2f}')
    ir = np.random.default_rng(1).choice(len(RA), min(6000, len(RA)), replace=False)
    a2.scatter(RA[ir, 1], RA[ir, 0], s=6, color=RAWC, alpha=0.18, label=f'raw  r={rr:.2f}')
    lim = np.percentile(DA[:, 0], 97)
    a2.plot([0, lim], [0, lim], color=INK, lw=1, ls=':')
    a2.set_title('Consecutive leg amplitudes — regular only after de-skewing',
                 fontweight='bold', fontsize=13, loc='left')
    a2.set_xlabel('previous leg amplitude (pt)'); a2.set_ylabel('current leg amplitude (pt)')
    a2.set_xlim(0, lim); a2.set_ylim(0, lim); a2.legend(loc='upper right', fontsize=10)
    a2.grid(color=GRID, lw=0.7); a2.set_axisbelow(True)
    for s in ('top', 'right'): a2.spines[s].set_visible(False)
    fig.savefig(SD + 'findings.png', facecolor='white', bbox_inches='tight')
    print(f'saved findings.png | raw r={rr:.2f} detrended r={dr:.2f}')


def main():
    files = sorted(glob.glob(f'{A1}/*.parquet'))[-150:]
    days = []
    for f in files:
        day = os.path.basename(f)[:10]
        if not os.path.exists(f'{A5}/{day}.parquet'):
            continue
        days.append(load_day(day))
    # example: pick a day with clear structure (decent range + several detrended pivots)
    ex = None
    for f in files[::7]:
        day = os.path.basename(f)[:10]
        if not os.path.exists(f'{A5}/{day}.parquet'):
            continue
        c, dr, det = load_day(day)
        if len(zigzag(np.where(np.isnan(det), np.nanmean(det), det), R)) >= 8:
            ex = day; break
    fig_example(ex or os.path.basename(files[-1])[:10])
    fig_findings(days)


if __name__ == '__main__':
    main()
