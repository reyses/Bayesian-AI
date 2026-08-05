#!/usr/bin/env python3
"""PIVOT-ENTRY LATENCY — t = pvt + n (owner 2026-07-26): the low-risk move is
to fire LATE (after a pivot is confirmed). How small can n be — bars after the
pivot — while the entry is still reliably in the right direction?

CAUSAL zigzag: a pivot at the extreme is only CONFIRMED once price retraces R
points from it; confirmation lands n = (confirm_bar - extreme_bar) bars later.
At confirmation we 'fire' in the reversal direction and measure, to the NEXT
confirmed pivot: direction HIT-RATE and captured points. Sweeping the
retrace threshold R traces the earliness/safety trade-off:
  small R -> small n (early) but more false pivots (lower hit-rate)
  large R -> larger n (late) but cleaner confirmation (higher hit-rate)
The knee = the least-late n that still fires in the right direction reliably.
Raw ATLAS 1m, per-day (no cross-day), causal. CPU-only.
reports/pivot_entry_latency.md + assets/pivot_entry_latency.png.
"""
import glob
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
ATLAS = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
OUT_MD = os.path.join(PROJ, 'reports', 'pivot_entry_latency.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'pivot_entry_latency.png')

R_GRID = [4, 6, 8, 12, 16, 24]     # retrace-confirm threshold (points)
FWD_CAP = 60                        # cap forward horizon (bars) to next pivot


def confirmed_pivots(close, R):
    """Causal zigzag with separate hi/lo tracking. Yield (confirm_idx,
    extreme_idx, direction): +1 = a LOW confirmed (up-leg begins), -1 = HIGH."""
    n = len(close)
    hi = lo = close[0]; hi_i = lo_i = 0; dirn = 0
    for i in range(1, n):
        p = close[i]
        if dirn == 0:
            if p - lo >= R:
                dirn = 1; yield (i, lo_i, 1); hi, hi_i = p, i
            elif hi - p >= R:
                dirn = -1; yield (i, hi_i, -1); lo, lo_i = p, i
            else:
                if p > hi: hi, hi_i = p, i
                if p < lo: lo, lo_i = p, i
        elif dirn == 1:                       # up-leg: track high, confirm HIGH on drop R
            if p > hi:
                hi, hi_i = p, i
            elif hi - p >= R:
                yield (i, hi_i, -1); dirn = -1; lo, lo_i = p, i
        else:                                 # down-leg: track low, confirm LOW on rise R
            if p < lo:
                lo, lo_i = p, i
            elif p - lo >= R:
                yield (i, lo_i, 1); dirn = 1; hi, hi_i = p, i

def main():
    files = sorted(glob.glob(os.path.join(ATLAS, '*.parquet')))
    rows = {R: [] for R in R_GRID}
    for f in files:
        df = pd.read_parquet(f)
        if len(df) < 100:
            continue
        close = df['close'].to_numpy(float)
        for R in R_GRID:
            pivs = list(confirmed_pivots(close, R))
            for j in range(len(pivs) - 1):
                ci, ei, d = pivs[j]
                nxt_ci, nxt_ei, _ = pivs[j + 1]
                n = ci - ei                       # bars pivot -> confirmation
                entry = close[ci]
                # ENTRY-isolated: capture to the leg's EXTREME (peak available)
                cap_ext = d * (close[nxt_ei] - entry)
                # full causal round-trip: exit at next confirmation
                cap_conf = d * (close[min(nxt_ci, ci + FWD_CAP)] - entry)
                phase = n / max(1, (nxt_ei - ei))
                rows[R].append(dict(n=n, hit=int(cap_ext > 0),
                                    cap_ext=cap_ext, cap_conf=cap_conf, phase=phase))

    lines = ['# Pivot-entry latency — t = pvt + n (causal zigzag, ATLAS 1m)',
             f'{len(files)} days. Fire at confirmation; measure to next pivot '
             f'(cap {FWD_CAP} bars).',
             '',
             '| retrace R (pt) | pivots | median n | dir-hit% (to peak) | '
             'captured-to-PEAK (pt) | net round-trip (pt) | phase@entry |',
             '|---|---|---|---|---|---|---|']
    curve = []
    for R in R_GRID:
        r = rows[R]
        if len(r) < 50:
            continue
        med_n = st.median([x['n'] for x in r])
        hit = st.mean([x['hit'] for x in r])
        cap = st.mean([x['cap_ext'] for x in r])
        capc = st.mean([x['cap_conf'] for x in r])
        ph = st.mean([x['phase'] for x in r])
        curve.append((R, len(r), med_n, hit, cap, capc, ph))
        lines.append(f"| {R} | {len(r)} | {med_n:.0f} | {hit:.1%} | {cap:+.1f} "
                     f"| {capc:+.1f} | {ph:.0%} |")
    lines += ['',
              'Reading: hit-rate = fraction of fires whose direction was right '
              'to the next pivot; captured = points won in the fired direction; '
              'phase@entry = how deep into the leg the confirmation lands. The '
              'trade-off is earliness (small n / small R) vs direction '
              'reliability (hit-rate) and phase. Note: hit-rate here is '
              'pivot-to-pivot direction, BEFORE costs (~3.6 tick RT) — a '
              'reliable direction still needs captured > cost to trade.']
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    if curve:
        fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)
        ns = [c[2] for c in curve]; hits = [c[3] * 100 for c in curve]
        caps = [c[4] for c in curve]
        ax1.plot(ns, hits, 'o-', color='tab:blue', label='direction hit-rate %')
        ax1.axhline(50, color='black', lw=1, ls=':')
        ax1.set_xlabel('median n = bars from pivot to confirmation (t = pvt+n)')
        ax1.set_ylabel('direction hit-rate %', color='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(ns, caps, 's--', color='tab:red', label='mean captured pts')
        ax2.set_ylabel('mean captured pts to next pivot', color='tab:red')
        for c in curve:
            ax1.annotate(f'R{c[0]}', (c[2], c[3] * 100), fontsize=8)
        ax1.set_title('How early can we fire after a pivot? (earliness vs reliability)')
        ax1.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG if curve else 'none')


if __name__ == '__main__':
    main()
