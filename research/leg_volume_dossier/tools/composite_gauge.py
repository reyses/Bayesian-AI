#!/usr/bin/env python3
"""COMPOSITE GAUGE — the synthesis (owner 2026-07-25: "let's bring this all
together"). Fuses the day's confirmed detectors into one 2-axis instrument:

  SICKNESS = count of active leg-pure dynamics anomalies (the interaction
             set: ldist_std lo, price_accel_1b lo, vol_velocity_30 lo,
             lambda_se_21 hi, price_velocity_30 lo, swing_noise_30 hi)
  VIGOR    = conviction state: FADED if body/range dropped 1 sigma under the
             leg's running norm (the confirmed fade detector), else ALIVE

Cells of (vigor x sickness) -> fwd 3-bar px; day-block CI on the two money
contrasts: [ALIVE, sick 0] vs [FADED, sick >= 2], and the fade-only and
sick-only marginals. Writes reports/composite_gauge.md +
assets/composite_gauge.png.
"""
import glob
import json
import os
import random
import re
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
PACKETS = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'gen0', 'packets')
OUT_MD = os.path.join(PROJ, 'reports', 'composite_gauge.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'composite_gauge.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')

SICK_DETECTORS = [('ldist_std', 'lo'), ('price_accel_1b', 'lo'),
                  ('vol_velocity_30', 'lo'), ('lambda_se_21', 'hi'),
                  ('price_velocity_30', 'lo'), ('swing_noise_30', 'hi')]
NEED = sorted({f for f, _ in SICK_DETECTORS} | {'body', 'bar_range'})
Z_SICK = 2.0
Z_FADE = 1.0
LAG = 2
K = 3
N_BOOT = 2000
SEED = 42


def parse(text):
    feats, px, leg = {}, None, None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('[1m]'):
            for k, v in KV.findall(s):
                if k in NEED:
                    feats[k] = float(v)
        elif s.startswith('local:'):
            m = PX.search(s)
            if m:
                px = float(m.group(1))
            m = LEG.search(s)
            if m:
                leg = float(m.group(1))
    conv = None
    if 'body' in feats and feats.get('bar_range'):
        conv = feats['body'] / feats['bar_range']
    return feats, conv, px, leg


def main():
    obs = []
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        eid = os.path.basename(pkt_path).replace('.json', '')
        day = "_".join(eid.split('_')[:3])
        pkt = json.load(open(pkt_path))
        rows = [parse(fr['text']) for fr in pkt['frames']]
        px_path = [p for _, _, p, _ in rows]
        events, fade_at = {}, None
        prev_leg_start = None
        for i, (feats, conv, px, leg_age) in enumerate(rows):
            if px is None or leg_age is None:
                continue
            leg_start = int(i - leg_age)
            if prev_leg_start is None or abs(leg_start - prev_leg_start) > 1:
                events, fade_at = {}, None
            prev_leg_start = leg_start
            # sickness detectors (leg-pure z on dynamics)
            for fname, tail in SICK_DETECTORS:
                base = [rows[j][0].get(fname) for j in range(max(0, leg_start), i)]
                base = [b for b in base if b is not None]
                v = feats.get(fname)
                if v is None or len(base) < 3:
                    continue
                sd = st.pstdev(base)
                if not sd:
                    continue
                z = (v - st.mean(base)) / sd
                fired = (z >= Z_SICK) if tail == 'hi' else (z <= -Z_SICK)
                if fired and (fname, tail) not in events:
                    events[(fname, tail)] = i
            # vigor: conviction fade (leg-pure)
            cbase = [rows[j][1] for j in range(max(0, leg_start), i)
                     if rows[j][1] is not None]
            if conv is not None and len(cbase) >= 3:
                csd = st.pstdev(cbase)
                if csd and (conv - st.mean(cbase)) / csd <= -Z_FADE and fade_at is None:
                    fade_at = i
            j = i + K
            fwd = (px_path[j] - px
                   if j < len(px_path) and px_path[j] is not None else None)
            if fwd is None:
                continue
            sick = sum(1 for d, t0 in events.items() if (i - t0) >= LAG)
            faded = int(fade_at is not None and (i - fade_at) >= LAG)
            obs.append(dict(day=day, sick=min(sick, 2), faded=faded, fwd=fwd))

    days = sorted({o['day'] for o in obs})
    by_day = {d: [o for o in obs if o['day'] == d] for d in days}

    def cell_means(sample):
        cells = {}
        for o in sample:
            cells.setdefault((o['faded'], o['sick']), []).append(o['fwd'])
        return {k: st.mean(v) for k, v in cells.items() if len(v) >= 5}

    base = cell_means(obs)
    counts = {}
    for o in obs:
        counts[(o['faded'], o['sick'])] = counts.get((o['faded'], o['sick']), 0) + 1

    def contrast(sample):
        good = [o['fwd'] for o in sample if o['faded'] == 0 and o['sick'] == 0]
        bad = [o['fwd'] for o in sample if o['faded'] == 1 and o['sick'] >= 2]
        return (st.mean(bad) - st.mean(good)) if good and bad else None

    c0 = contrast(obs)
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        ss = []
        for d in rng.choices(days, k=len(days)):
            ss.extend(by_day[d])
        b = contrast(ss)
        if b is not None:
            boots.append(b)
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]

    lines = ['# Composite gauge — vigor x sickness (the synthesis)',
             f'N={len(obs)} frame-obs, {len(days)} days.',
             '',
             '| vigor \\ sick | 0 | 1 | 2+ |', '|---|---|---|---|']
    for faded, vlabel in ((0, 'ALIVE'), (1, 'FADED')):
        row = [f'| {vlabel} ']
        for s in (0, 1, 2):
            m = base.get((faded, s))
            n = counts.get((faded, s), 0)
            row.append(f"| {m:+.2f} (n={n}) " if m is not None else "| — ")
        lines.append(''.join(row) + '|')
    lines += ['',
              f'**MONEY CONTRAST** [FADED & sick>=2] − [ALIVE & clean] = '
              f'{c0:+.2f} pts, 95% CI [{lo:+.2f}, {hi:+.2f}] — '
              + ('**SIGNIFICANT**' if hi < 0 else 'not significant'),
              '',
              'Exit-head reading: hold while ALIVE & clean; the gauge arms as '
              'vigor fades; 2+ sickness on a faded leg = the tape has turned.']
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    xs = ['sick 0', 'sick 1', 'sick 2+']
    for faded, color, label in ((0, 'tab:green', 'vigor ALIVE'),
                                (1, 'tab:red', 'vigor FADED')):
        ys = [base.get((faded, s)) for s in (0, 1, 2)]
        ax.plot(xs, [y if y is not None else float('nan') for y in ys],
                'o-', color=color, lw=3, ms=9, label=label)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel(f'mean px, next {K} bars')
    ax.set_title('The composite gauge: what the next 3 minutes pay')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
