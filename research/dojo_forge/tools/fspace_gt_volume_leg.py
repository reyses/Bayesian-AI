#!/usr/bin/env python3
"""Volume progression across the PEAK LEG in PHASE time (owner 2026-07-25:
"the window should be the size of an average leg"). Fixed clock windows smear
leg-native structure; here each episode's peak leg is normalized to phase
0 (leg start, via the peak frame's own `leg age`) → 1 (the peak), sampled at
deciles, plus one leg-length of post-peak in the same units.
Writes reports/fspace_gt_volume_leg.md. CPU-only.
"""
import glob
import json
import os
import re
import statistics as st

DOJO = os.path.join(os.path.dirname(__file__), '..')
PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
OUT = os.path.join(DOJO, 'reports', 'fspace_gt_volume_leg.md')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')
KEYS = ['vol_velocity_1b', 'vol_velocity_30', 'vol_accel_30',
        'price_velocity_30']
PHASES = [i / 10 for i in range(0, 11)] + [1.25, 1.5, 1.75, 2.0]


def parse(text):
    f = {}
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('[1m]'):
            kv = dict(KV.findall(s))
            for k in KEYS:
                if k in kv:
                    f[k] = float(kv[k])
        elif s.startswith('local:'):
            m = PX.search(s)
            if m:
                f['px'] = float(m.group(1))
            m = LEG.search(s)
            if m:
                f['leg_age'] = float(m.group(1))
    return f


def main():
    acc = {k: {p: [] for p in PHASES} for k in KEYS}
    leg_lens = []
    n_used = 0
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        pkt = json.load(open(pkt_path))
        fvs = [parse(fr['text']) for fr in pkt['frames']]
        pxs = [(i, f['px']) for i, f in enumerate(fvs) if 'px' in f]
        if len(pxs) < 8:
            continue
        peak_i = max(pxs, key=lambda t: t[1])[0]
        leg_age = fvs[peak_i].get('leg_age')
        if not leg_age or leg_age < 3:          # too short to phase-sample
            continue
        leg_start = peak_i - leg_age
        norm = {}
        ok = True
        for k in KEYS:
            vals = [f[k] for f in fvs if k in f]
            if len(vals) < 4 or st.pstdev(vals) == 0:
                ok = False
                break
            norm[k] = (st.mean(vals), st.pstdev(vals))
        if not ok:
            continue
        n_used += 1
        leg_lens.append(leg_age)
        for p in PHASES:
            j = round(leg_start + p * leg_age)
            if 0 <= j < len(fvs):
                for k in KEYS:
                    if k in fvs[j]:
                        mu, sd = norm[k]
                        acc[k][p].append((fvs[j][k] - mu) / sd)

    lines = [
        '# Volume across the PEAK LEG — phase time (0=leg start, 1=peak)',
        f'{n_used} episodes; leg length mean {st.mean(leg_lens):.1f} min, '
        f'median {st.median(leg_lens):.0f} (fixed ±5min window was therefore '
        f'~{5 / st.mean(leg_lens):.0%} of a leg on each side — the owner\'s '
        'critique stands).',
        '',
        '| phase | ' + ' | '.join(KEYS) + ' | n |',
        '|---' * (len(KEYS) + 2) + '|',
    ]
    for p in PHASES:
        cells = []
        ns = 0
        for k in KEYS:
            xs = acc[k][p]
            ns = max(ns, len(xs))
            cells.append(f"{st.mean(xs):+.2f}" if len(xs) >= 20 else "—")
        tag = ' **PEAK**' if p == 1.0 else ''
        lines.append(f"| {p:.2f}{tag} | " + " | ".join(cells) + f" | {ns} |")
    lines += [
        '',
        'Reading: phase 0→1 is the leg\'s life; >1 is post-peak in the same '
        'units. Volume climax location within the leg, and the price-velocity '
        'arc, are now leg-native rather than clock-smeared.',
    ]
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
