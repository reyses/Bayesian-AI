#!/usr/bin/env python3
"""WINDOW-SCALE TEST — the dossier's core experiment (owner 2026-07-25):
"we don't see enough signal because our sample windows are not aligned to the
average length of a leg … it might need to be the length of a half leg —
this would make all the Z bigger."

Method: for each dojo episode, locate the peak leg (packet px path + the peak
frame's own leg-age), pull RAW 1m volume from ATLAS, and compute volume
z-scores under three rolling-window scales ending at each bar:
    W30   = 30 bars (the current F-space convention)
    Wleg  = the episode's own leg length
    Whalf = half the leg (min 3 bars)
Then trace each scale's z across leg phase and compare magnitudes at the
mid-leg climax and at the peak. If the owner is right, |z| grows as the
window shrinks toward leg scale.

Run from repo root:
  python research/leg_volume_dossier/tools/window_scale_test.py
Writes reports/window_scale_test.md + assets/window_scale_chart.png.
"""
import glob
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
PACKETS = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'gen0', 'packets')
ATLAS_1M = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
OUT_MD = os.path.join(PROJ, 'reports', 'window_scale_test.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'window_scale_chart.png')

PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')
PHASES = [i / 10 for i in range(0, 11)] + [1.25, 1.5]
MIN_LEG = 4


def parse_frame(text):
    px = leg = None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('local:'):
            m = PX.search(s)
            if m:
                px = float(m.group(1))
            m = LEG.search(s)
            if m:
                leg = float(m.group(1))
    return px, leg


def main():
    day_cache = {}
    scales = ['W30', 'W2leg', 'Wleg', 'Whalf', 'Wpure']
    paths = {s: {p: [] for p in PHASES} for s in scales}
    climax = {s: [] for s in scales}
    at_peak = {s: [] for s in scales}
    n_used = 0
    leg_lens = []

    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        eid = os.path.basename(pkt_path).replace('.json', '')
        parts = eid.split('_')
        day_key, epoch = f"{parts[0]}_{parts[1]}_{parts[2]}", int(parts[3])
        if day_key not in day_cache:
            f = os.path.join(ATLAS_1M, f'{day_key}.parquet')
            day_cache[day_key] = pd.read_parquet(f) if os.path.exists(f) else None
        bars = day_cache[day_key]
        if bars is None:
            continue
        pkt = json.load(open(pkt_path))
        info = [parse_frame(fr['text']) for fr in pkt['frames']]
        pxs = [(i, p) for i, (p, _) in enumerate(info) if p is not None]
        if len(pxs) < 8:
            continue
        peak_i = max(pxs, key=lambda t: t[1])[0]
        leg_age = info[peak_i][1]
        if not leg_age or leg_age < MIN_LEG:
            continue
        # episode minute i -> bar timestamp (packets are minute-cadence from epoch)
        base_min = (epoch // 60) * 60
        vols = bars.set_index('timestamp')['volume']

        def z_at(minute_idx, W):
            ts = base_min + minute_idx * 60
            hist = vols.loc[:ts].tail(W + 1)
            if len(hist) < W + 1:
                return None
            window = hist.iloc[:-1]
            sd = window.std()
            if not sd or sd != sd:
                return None
            return (hist.iloc[-1] - window.mean()) / sd

        Ws = {'W30': 30, 'W2leg': max(8, int(2 * leg_age)),
              'Wleg': max(MIN_LEG, int(leg_age)),
              'Whalf': max(3, int(leg_age // 2))}

        def z_pure(minute_idx):
            # owner 2026-07-25: cross-leg normalization smooths the signal —
            # baseline THIS LEG ONLY (expanding from leg start), no prior-leg
            # contamination in mean or std.
            ts0 = base_min + int(leg_start) * 60
            ts = base_min + minute_idx * 60
            seg = vols.loc[ts0:ts]
            if len(seg) < 4:
                return None
            window = seg.iloc[:-1]
            sd = window.std()
            if not sd or sd != sd:
                return None
            return (seg.iloc[-1] - window.mean()) / sd
        leg_start = peak_i - leg_age
        got_any = False
        for p in PHASES:
            j = round(leg_start + p * leg_age)
            if not (0 <= j < len(pkt['frames'])):
                continue
            for s, W in list(Ws.items()) + [('Wpure', None)]:
                z = z_pure(j) if s == 'Wpure' else z_at(j, W)
                if z is not None:
                    paths[s][p].append(z)
                    got_any = True
                    if 0.4 <= p <= 0.6:
                        climax[s].append(z)
                    if p == 1.0:
                        at_peak[s].append(z)
        if got_any:
            n_used += 1
            leg_lens.append(leg_age)

    # ---- report ----
    def trimmed(xs, frac=0.1):
        if len(xs) < 10:
            return float('nan')
        k = int(len(xs) * frac)
        return st.mean(sorted(xs)[k:-k])

    lines = [
        '# Window-scale test — does leg-scaled windowing amplify the signal?',
        f'{n_used} episodes (raw ATLAS 1m volume); leg length mean '
        f'{st.mean(leg_lens):.1f} / median {st.median(leg_lens):.0f} min.',
        '',
        '## Headline: |z| of the volume climax (mid-leg, phase 0.4-0.6) by window scale',
        '| window | mean z | trimmed10 | median z | n |', '|---|---|---|---|---|',
    ]
    for s in scales:
        xs = climax[s]
        lines.append(f"| {s} | {st.mean(xs):+.2f} | {trimmed(xs):+.2f} "
                     f"| {st.median(xs):+.2f} | {len(xs)} |")
    lines += ['', '## z at the PEAK itself', '| window | mean z | trimmed10 | median z |',
              '|---|---|---|---|']
    for s in scales:
        xs = at_peak[s]
        lines.append(f"| {s} | {st.mean(xs):+.2f} | {trimmed(xs):+.2f} "
                     f"| {st.median(xs):+.2f} |")
    lines += ['', '## Full phase paths (mean z)',
              '| phase | ' + ' | '.join(scales) + ' |',
              '|---' * (len(scales) + 1) + '|']
    for p in PHASES:
        cells = [f"{st.mean(paths[s][p]):+.2f}" if len(paths[s][p]) >= 20 else '—'
                 for s in scales]
        tag = ' **PEAK**' if p == 1.0 else ''
        lines.append(f"| {p:.2f}{tag} | " + ' | '.join(cells) + ' |')
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # ---- chart ----
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    colors = {'W30': 'tab:gray', 'W2leg': 'tab:green', 'Wleg': 'tab:blue', 'Whalf': 'tab:red', 'Wpure': 'tab:purple'}
    labels = {'W30': '30-bar (current)', 'W2leg': '2x leg', 'Wleg': 'leg-length',
              'Whalf': 'half-leg', 'Wpure': 'LEG-PURE (this leg only)'}
    for s in scales:
        xs = [p for p in PHASES if len(paths[s][p]) >= 20]
        ys = [st.mean(paths[s][p]) for p in xs]
        ax.plot(xs, ys, color=colors[s], label=labels[s], lw=2.5, marker='o', ms=4)
    ax.axvline(1.0, color='black', ls='--', lw=1.5)
    ax.axhline(0, color='gray', lw=0.8)
    ax.annotate('THE PEAK', xy=(1.0, ax.get_ylim()[1] * 0.9), fontsize=11,
                ha='center', fontweight='bold')
    ax.set_xlabel('leg phase (0 = leg start, 1 = peak)')
    ax.set_ylabel('raw-volume z-score')
    ax.set_title(f'Same volume, three window scales ({n_used} episodes)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.25)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines[:20]))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
