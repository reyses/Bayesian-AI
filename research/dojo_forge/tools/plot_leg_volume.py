#!/usr/bin/env python3
"""Render the leg-phase volume/velocity paths as a phone-readable chart
(owner 2026-07-25: charts, not z-score walls). Reuses fspace_gt_volume_leg's
computation; saves reports/assets/leg_phase_volume.png."""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fspace_gt_volume_leg as L  # noqa: E402
import glob
import json
import statistics as st

DOJO = os.path.join(HERE, '..')
OUT = os.path.join(DOJO, 'reports', 'assets', 'leg_phase_volume.png')


def main():
    acc = {k: {p: [] for p in L.PHASES} for k in L.KEYS}
    n_used = 0
    for pkt_path in sorted(glob.glob(os.path.join(L.PACKETS, '*.json'))):
        pkt = json.load(open(pkt_path))
        fvs = [L.parse(fr['text']) for fr in pkt['frames']]
        pxs = [(i, f['px']) for i, f in enumerate(fvs) if 'px' in f]
        if len(pxs) < 8:
            continue
        peak_i = max(pxs, key=lambda t: t[1])[0]
        leg_age = fvs[peak_i].get('leg_age')
        if not leg_age or leg_age < 3:
            continue
        leg_start = peak_i - leg_age
        norm = {}
        ok = True
        for k in L.KEYS:
            vals = [f[k] for f in fvs if k in f]
            if len(vals) < 4 or st.pstdev(vals) == 0:
                ok = False
                break
            norm[k] = (st.mean(vals), st.pstdev(vals))
        if not ok:
            continue
        n_used += 1
        for p in L.PHASES:
            j = round(leg_start + p * leg_age)
            if 0 <= j < len(fvs):
                for k in L.KEYS:
                    if k in fvs[j]:
                        mu, sd = norm[k]
                        acc[k][p].append((fvs[j][k] - mu) / sd)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    styles = {'vol_velocity_1b': ('tab:blue', 'volume (bar)'),
              'vol_velocity_30': ('tab:cyan', 'volume (30-bar)'),
              'price_velocity_30': ('tab:red', 'price velocity')}
    for k, (color, label) in styles.items():
        xs = [p for p in L.PHASES if len(acc[k][p]) >= 20]
        ys = [st.mean(acc[k][p]) for p in xs]
        ax.plot(xs, ys, color=color, label=label, lw=2.5,
                marker='o', ms=4)
    ax.axvline(1.0, color='black', ls='--', lw=1.5)
    ax.axhline(0, color='gray', lw=0.8)
    ax.annotate('THE PEAK', xy=(1.0, 0.35), fontsize=11, ha='center',
                fontweight='bold')
    ax.annotate('volume climax\n(mid-leg)', xy=(0.5, 0.34),
                xytext=(0.25, 0.42), fontsize=10,
                arrowprops=dict(arrowstyle='->'))
    ax.annotate('quiet exhaustion:\nvolume gone, price stalls',
                xy=(0.97, -0.28), xytext=(0.52, -0.42), fontsize=10,
                arrowprops=dict(arrowstyle='->'))
    ax.set_xlabel('leg phase (0 = leg start, 1 = peak, >1 = after the top)')
    ax.set_ylabel('z-score vs episode average')
    ax.set_title(f'How volume and price behave across a winning leg '
                 f'({n_used} episodes)')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.25)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT)
    print(OUT)


if __name__ == '__main__':
    main()
