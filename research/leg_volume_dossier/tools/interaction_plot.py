#!/usr/bin/env python3
"""INTERACTION ANALYSIS (owner 2026-07-25: "did you do an interaction plot?").
Takes one representative detector per strong family (de-correlated picks) and
measures how they COMBINE:
  1. Anomaly-count curve: fwd 3-bar px by number of simultaneously-active
     detectors (0,1,2,3+), day-block CI — tests the composite-score idea.
  2. Pairwise interaction heatmap: fwd-px delta when BOTH fire vs the
     additive expectation (synergy = both − A − B).
Writes reports/interaction_plot.md + assets/interaction_plot.png.
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
OUT_MD = os.path.join(PROJ, 'reports', 'interaction_plot.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'interaction_plot.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')

# one per family, from the sweep's leaders (de-correlated by construction)
DETECTORS = [('ldist_std', 'lo'), ('price_accel_1b', 'lo'),
             ('vol_velocity_30', 'lo'), ('lambda_se_21', 'hi'),
             ('price_velocity_30', 'lo'), ('swing_noise_30', 'hi')]
Z_EVENT = 2.0
LAG = 2
K = 3
N_BOOT = 2000
SEED = 42


def parse_frame(text):
    feats, px, leg = {}, None, None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('[1m]'):
            for k, v in KV.findall(s):
                feats[k] = float(v)
        elif s.startswith('local:'):
            m = PX.search(s)
            if m:
                px = float(m.group(1))
            m = LEG.search(s)
            if m:
                leg = float(m.group(1))
    return feats, px, leg


def main():
    obs = []          # dict(day, states(tuple of 0/1), fwd)
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        eid = os.path.basename(pkt_path).replace('.json', '')
        day = "_".join(eid.split('_')[:3])
        pkt = json.load(open(pkt_path))
        rows = [parse_frame(fr['text']) for fr in pkt['frames']]
        px_path = [p for _, p, _ in rows]
        events = {}
        prev_leg_start = None
        for i, (feats, px, leg_age) in enumerate(rows):
            if px is None or leg_age is None:
                continue
            leg_start = int(i - leg_age)
            if prev_leg_start is None or abs(leg_start - prev_leg_start) > 1:
                events = {}
            prev_leg_start = leg_start
            for fname, tail in DETECTORS:
                base = [rows[j][0].get(fname) for j in range(max(0, leg_start), i)]
                base = [b for b in base if b is not None]
                v = feats.get(fname)
                if v is None or len(base) < 3:
                    continue
                sd = st.pstdev(base)
                if not sd:
                    continue
                z = (v - st.mean(base)) / sd
                fired = (z >= Z_EVENT) if tail == 'hi' else (z <= -Z_EVENT)
                if fired and (fname, tail) not in events:
                    events[(fname, tail)] = i
            j = i + K
            fwd = (px_path[j] - px
                   if j < len(px_path) and px_path[j] is not None else None)
            if fwd is None:
                continue
            states = tuple(int((d in events) and (i - events[d]) >= LAG)
                           for d in DETECTORS)
            obs.append(dict(day=day, states=states, fwd=fwd))

    days = sorted({o['day'] for o in obs})
    by_day = {d: [o for o in obs if o['day'] == d] for d in days}

    def count_curve(sample):
        buckets = {}
        for o in sample:
            c = min(sum(o['states']), 3)
            buckets.setdefault(c, []).append(o['fwd'])
        return {c: st.mean(v) for c, v in buckets.items() if len(v) >= 5}

    base_curve = count_curve(obs)
    rng = random.Random(SEED)
    boot_curves = {c: [] for c in range(4)}
    for _ in range(N_BOOT):
        ss = []
        for d in rng.choices(days, k=len(days)):
            ss.extend(by_day[d])
        for c, m in count_curve(ss).items():
            boot_curves[c].append(m)
    ci = {}
    for c, xs in boot_curves.items():
        if len(xs) > 100:
            xs.sort()
            ci[c] = (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs))])

    # pairwise synergy
    n_det = len(DETECTORS)
    def mean_fwd(filt):
        xs = [o['fwd'] for o in obs if filt(o['states'])]
        return st.mean(xs) if len(xs) >= 15 else None
    base_all_off = mean_fwd(lambda s: sum(s) == 0)
    syn = [[None] * n_det for _ in range(n_det)]
    for a in range(n_det):
        for b in range(a + 1, n_det):
            m_a = mean_fwd(lambda s: s[a] and not s[b])
            m_b = mean_fwd(lambda s: s[b] and not s[a])
            m_ab = mean_fwd(lambda s: s[a] and s[b])
            if None not in (m_a, m_b, m_ab, base_all_off):
                syn[a][b] = m_ab - (m_a + m_b - base_all_off)

    counts = {min(sum(o['states']), 3): 0 for o in obs}
    for o in obs:
        counts[min(sum(o['states']), 3)] += 1
    lines = ['# Detector interactions',
             f'detectors: {", ".join(f"{f}({t})" for f, t in DETECTORS)}',
             '',
             '## Anomaly-count curve (fwd 3-bar px, day-block 95% CI)',
             '| active detectors | mean fwd px | CI | n |', '|---|---|---|---|']
    for c in sorted(base_curve):
        lo, hi = ci.get(c, (float('nan'), float('nan')))
        lines.append(f"| {c}{'+' if c == 3 else ''} | {base_curve[c]:+.2f} "
                     f"| [{lo:+.2f}, {hi:+.2f}] | {counts.get(c, 0)} |")
    lines += ['', '## Pairwise synergy (both-on minus additive expectation, pts)',
              '| pair | synergy |', '|---|---|']
    for a in range(n_det):
        for b in range(a + 1, n_det):
            if syn[a][b] is not None:
                lines.append(f"| {DETECTORS[a][0]} × {DETECTORS[b][0]} "
                             f"| {syn[a][b]:+.2f} |")
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    cs = sorted(base_curve)
    ys = [base_curve[c] for c in cs]
    yerr = [[base_curve[c] - ci[c][0] for c in cs if c in ci],
            [ci[c][1] - base_curve[c] for c in cs if c in ci]]
    ax.bar([str(c) + ('+' if c == 3 else '') for c in cs], ys,
           yerr=yerr, color=['tab:green' if y > 0 else 'tab:red' for y in ys],
           alpha=0.85, capsize=6)
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel('number of leg-anomaly detectors active at once')
    ax.set_ylabel(f'mean px change over next {K} bars')
    ax.set_title('Stacking anomalies: what the next 3 minutes pay')
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
