#!/usr/bin/env python3
"""Volume progression around GROUND-TRUTH (oracle) peaks (owner 2026-07-25).
Aligns frames at offset -5..+5 minutes relative to each episode's true peak
and traces the mean path of volume metrics (z-scored WITHIN episode so
day-scale differences cancel), with price velocity for context.
Answers: do our tops arrive on volume climax, fade, or divergence?
Writes reports/fspace_gt_volume.md. CPU-only.
"""
import glob
import json
import os
import re
import statistics as st

DOJO = os.path.join(os.path.dirname(__file__), '..')
PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
OUT = os.path.join(DOJO, 'reports', 'fspace_gt_volume.md')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
KEYS = ['vol_velocity_1b', 'vol_velocity_30', 'vol_accel_30',
        'price_velocity_30']
WINDOW = 5


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
    return f


def main():
    paths = {k: {o: [] for o in range(-WINDOW, WINDOW + 1)} for k in KEYS}
    n_used = 0
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        pkt = json.load(open(pkt_path))
        fvs = [parse(fr['text']) for fr in pkt['frames']]
        pxs = [(i, f['px']) for i, f in enumerate(fvs) if 'px' in f]
        if len(pxs) < 8:
            continue
        peak_i = max(pxs, key=lambda t: t[1])[0]
        # z-normalize each metric within the episode
        norm = {}
        ok = True
        for k in KEYS:
            vals = [f[k] for f in fvs if k in f]
            if len(vals) < 4 or st.pstdev(vals) == 0:
                ok = False
                break
            mu, sd = st.mean(vals), st.pstdev(vals)
            norm[k] = (mu, sd)
        if not ok:
            continue
        n_used += 1
        for off in range(-WINDOW, WINDOW + 1):
            j = peak_i + off
            if 0 <= j < len(fvs):
                for k in KEYS:
                    if k in fvs[j]:
                        mu, sd = norm[k]
                        paths[k][off].append((fvs[j][k] - mu) / sd)

    lines = [
        '# Volume progression around oracle peaks (within-episode z-scores)',
        f'{n_used} episodes aligned at their true peak (offset 0).',
        '',
        '| offset (min) | ' + ' | '.join(KEYS) + ' | n |',
        '|---' * (len(KEYS) + 2) + '|',
    ]
    for off in range(-WINDOW, WINDOW + 1):
        cells = []
        ns = 0
        for k in KEYS:
            xs = paths[k][off]
            ns = max(ns, len(xs))
            cells.append(f"{st.mean(xs):+.2f}" if xs else "—")
        tag = ' **PEAK**' if off == 0 else ''
        lines.append(f"| {off:+d}{tag} | " + " | ".join(cells) + f" | {ns} |")
    lines += [
        '',
        'Reading: values are z-scores vs the episode\'s own distribution. '
        'A volume CLIMAX at tops shows vol metrics spiking at offset 0; '
        'EXHAUSTION/divergence shows price velocity fading while volume '
        'drains into the peak; post-peak columns show what confirmation the '
        'first minutes after the top offer.',
    ]
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
