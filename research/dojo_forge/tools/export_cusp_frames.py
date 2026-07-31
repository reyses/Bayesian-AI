#!/usr/bin/env python
"""EXPORT CUSP SAMPLE + REVIEW FRAMES for the AG frame-by-frame review loop
(owner 2026-07-31: "frame by frame review all the cusps so we have the true
tops and bottoms" + "CLI AG so we don't waste Claude usage").

Mechanical hindsight-zigzag pivots are the exact extrema; AG's review layer
adds (a) visual QA of the mechanical label and (b) OWNER-TRADEABILITY
curation per OWNER_PROCESS.md. This script gives AG a deterministic input
package: a sampled cusp list (CSV) + one PNG per cusp.

Frame layout per cusp (hindsight is ALLOWED -- these are labels, not signals):
  top panel   : 60 min of 1m-aggregated candles ENDING at the cusp (context)
  bottom panel: 20 min of 5s candles CENTERED on the cusp (10 before / 10
                after) so the reviewer sees the rejection/aftermath.

Usage:
  python export_cusp_frames.py --days 120 --R 30 --sample 150 --seed 11
Outputs:
  research/dojo_forge/gate_state/cusp_review/cusps.csv
  research/dojo_forge/gate_state/cusp_review/frames/<day>_<idx>_<T|B>.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_ground_truth import hindsight_zigzag, D5   # same labels as the stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTD = os.path.join(REPO, 'research', 'dojo_forge', 'gate_state', 'cusp_review')


def candles(ax, o, h, l, c, x0=0):
    x = np.arange(len(o)) + x0
    up = c >= o
    ax.vlines(x, l, h, color=np.where(up, '#2E7D32', '#C62828'), lw=0.7, zorder=2)
    ax.bar(x, np.abs(c - o), bottom=np.minimum(o, c), width=0.7,
           color=np.where(up, '#2E7D32', '#C62828'), edgecolor='none', zorder=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=120)
    ap.add_argument('--R', type=float, default=30.0)
    ap.add_argument('--sample', type=int, default=150)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--min-leg', type=float, default=20.0)
    a = ap.parse_args()

    days = sorted(f[:-8] for f in os.listdir(D5) if f.endswith('.parquet'))[-a.days:]
    rows = []
    for day in tqdm(days, desc='label'):
        df = pd.read_parquet(os.path.join(D5, f'{day}.parquet'))
        if len(df) < 500: continue
        hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
        piv = hindsight_zigzag(hi, lo, a.R)
        for j in range(1, len(piv) - 1):
            i, pr, k = piv[j]
            leg_in = abs(pr - piv[j-1][1]); leg_out = abs(piv[j+1][1] - pr)
            if min(leg_in, leg_out) < a.min_leg: continue
            rows.append(dict(day=day, idx5s=i, ts=int(df['timestamp'].iloc[i]),
                             price=float(pr), kind=k,
                             leg_in_pt=round(leg_in, 2), leg_out_pt=round(leg_out, 2)))
    allc = pd.DataFrame(rows)
    rng = np.random.default_rng(a.seed)
    pick = allc.iloc[sorted(rng.choice(len(allc), min(a.sample, len(allc)), replace=False))]
    os.makedirs(os.path.join(OUTD, 'frames'), exist_ok=True)
    pick.to_csv(os.path.join(OUTD, 'cusps.csv'), index=False)

    for _, r in tqdm(pick.iterrows(), total=len(pick), desc='render'):
        df = pd.read_parquet(os.path.join(D5, f"{r.day}.parquet"))
        i = int(r.idx5s)
        # bottom: 20min of 5s centered (240 bars; 120 pre / 120 post)
        b0, b1 = max(0, i - 120), min(len(df), i + 120)
        z = df.iloc[b0:b1]
        # top: 60min of 1m aggregated, ending at the cusp
        t0 = max(0, i - 720)
        w = df.iloc[t0:i + 1].copy()
        w['g'] = (w['timestamp'] // 60)
        m = w.groupby('g').agg(o=('open', 'first'), h=('high', 'max'),
                               l=('low', 'min'), c=('close', 'last'))
        fig, (axt, axb) = plt.subplots(2, 1, figsize=(11, 7), dpi=100,
                                       gridspec_kw={'height_ratios': [1, 1.4]})
        candles(axt, m['o'].to_numpy(), m['h'].to_numpy(), m['l'].to_numpy(), m['c'].to_numpy())
        axt.axhline(r.price, color='#6A1B9A', lw=1.0, ls='--', alpha=0.8)
        axt.set_title(f"{r.day}  {'TOP' if r.kind=='T' else 'BOTTOM'} @ {r.price:.2f}  "
                      f"(leg in {r.leg_in_pt:.0f}pt / out {r.leg_out_pt:.0f}pt)  -- 1m context",
                      fontsize=9, loc='left')
        candles(axb, z['open'].to_numpy(), z['high'].to_numpy(),
                z['low'].to_numpy(), z['close'].to_numpy(), x0=b0)
        axb.axvline(i, color='#6A1B9A', lw=1.2, ls=':', alpha=0.9)
        axb.axhline(r.price, color='#6A1B9A', lw=1.0, ls='--', alpha=0.8)
        axb.set_title('5s detail -- cusp at dotted line (10min before / 10min after)',
                      fontsize=8, loc='left')
        ts = pd.to_datetime(int(r.ts), unit='s')
        axb.set_xlabel(f"cusp time {ts:%H:%M:%S} UTC", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTD, 'frames', f"{r.day}_{i}_{r.kind}.png"))
        plt.close(fig)

    print(f"\n{len(allc)} cusps labeled mechanically; sampled {len(pick)}")
    print(f"csv    : {os.path.join(OUTD, 'cusps.csv')}")
    print(f"frames : {os.path.join(OUTD, 'frames')}/")


if __name__ == '__main__':
    main()
