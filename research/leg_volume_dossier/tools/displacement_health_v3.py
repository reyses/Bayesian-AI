#!/usr/bin/env python3
"""DISPLACEMENT HEALTH v3 (FAST indicators — the owner's actual design) — leg-phase-aligned pilot (owner 2026-07-25:
"heavy math, let's only do a week or two; if promising then we expand").

Fixes v1's phase-smear: baselines are indexed by LEG PHASE, not minutes-
since-entry. Causal phase estimate: current leg_age / median leg length of
the library (the current leg's final length is unknowable live). Library =
completed legs' displacement-vs-phase profiles (displacement still zeroed at
each leg's OWN start — leg-relative, per the owner's healthy-leg framing).
PILOT SCOPE: first N_DAYS_PILOT curriculum days only.

Writes reports/displacement_health_v3.md + assets/displacement_health_v3.png.
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
OUT_MD = os.path.join(PROJ, 'reports', 'displacement_health_v3.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'displacement_health_v3.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')

# v3: the owner said FAST indicators — bar-level dynamics, not 30-bar rollups
FEATS = ['price_velocity_1b', 'price_accel_1b', 'vol_velocity_1b',
         'vol_accel_1b', 'bar_range', 'body', 'upper_wick', 'lower_wick']
N_DAYS_PILOT = 10
N_LEGS_BASE = 30
Z_SICK = 2.0
K = 3
N_BOOT = 2000
SEED = 42
PHASE_BUCKETS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.3]   # bucket edges (est. phase)


def bucket(phase):
    for i, edge in enumerate(PHASE_BUCKETS):
        if phase <= edge:
            return i
    return len(PHASE_BUCKETS)


def parse_frame(text):
    feats, px, leg = {}, None, None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('[1m]'):
            for k, v in KV.findall(s):
                if k in FEATS:
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
    paths = sorted(glob.glob(os.path.join(PACKETS, '*.json')),
                   key=lambda p: int(os.path.basename(p).split('_')[3]))
    all_days = []
    for p in paths:
        d = "_".join(os.path.basename(p).split('_')[:3])
        if d not in all_days:
            all_days.append(d)
    pilot_days = set(all_days[:N_DAYS_PILOT])

    library = []          # completed legs: list of (phase, {f: leg-relative disp})
    leg_lens = []
    obs = []
    for pkt_path in paths:
        eid = os.path.basename(pkt_path).replace('.json', '')
        day = "_".join(eid.split('_')[:3])
        if day not in pilot_days:
            continue
        pkt = json.load(open(pkt_path))
        rows = [parse_frame(fr['text']) for fr in pkt['frames']]
        px_path = [p for _, p, _ in rows]
        med_leg = st.median(leg_lens) if len(leg_lens) >= 10 else None

        cur_start = None
        cur_profile = []
        for i, (feats, px, leg_age) in enumerate(rows):
            if px is None or leg_age is None:
                continue
            leg_start = int(i - leg_age)
            if cur_start is None or abs(leg_start - cur_start) > 1:
                # a leg ended -> archive its profile with realized phases
                if cur_profile and len(cur_profile) >= 3:
                    L = len(cur_profile)
                    library.append([( (k + 1) / L, d) for k, d in
                                    enumerate(p[1] for p in cur_profile)])
                    leg_lens.append(L)
                cur_start = leg_start
                cur_profile = []
            anchor = rows[max(0, leg_start)][0]
            disp = {f: feats[f] - anchor[f] for f in FEATS
                    if f in feats and f in anchor}
            cur_profile.append((i, disp))

            lib = library[-N_LEGS_BASE:]
            if med_leg and len(lib) >= 10:
                phase_est = min(leg_age / med_leg, 1.5)
                b = bucket(phase_est)
                sick = 0
                for f in FEATS:
                    vals = [d[f] for leg in lib for (ph, d) in leg
                            if bucket(ph) == b and f in d]
                    if len(vals) < 12 or f not in disp:
                        continue
                    sd = st.pstdev(vals)
                    if not sd:
                        continue
                    if abs((disp[f] - st.mean(vals)) / sd) >= Z_SICK:
                        sick += 1
                j = i + K
                fwd = (px_path[j] - px
                       if j < len(px_path) and px_path[j] is not None else None)
                if fwd is not None:
                    obs.append(dict(day=day, sick=min(sick, 4), fwd=fwd))

    days = sorted({o['day'] for o in obs})
    by_day = {d: [o for o in obs if o['day'] == d] for d in days}

    def curve(sample):
        b = {}
        for o in sample:
            b.setdefault(o['sick'], []).append(o['fwd'])
        return {c: st.mean(v) for c, v in b.items() if len(v) >= 5}

    base = curve(obs)
    rng = random.Random(SEED)
    boots = {c: [] for c in range(5)}
    for _ in range(N_BOOT):
        ss = []
        for d in rng.choices(days, k=len(days)):
            ss.extend(by_day[d])
        for c, m in curve(ss).items():
            boots[c].append(m)
    ci = {}
    for c, xs in boots.items():
        if len(xs) > 100:
            xs.sort()
            ci[c] = (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs))])
    counts = {}
    for o in obs:
        counts[o['sick']] = counts.get(o['sick'], 0) + 1

    lines = ['# Displacement health v3 (FAST indicators) — leg-phase-aligned PILOT',
             f'{N_DAYS_PILOT}-day pilot ({len(days)} days with obs); '
             f'displacement zeroed at LEG start; baseline = phase-bucket norms '
             f'from prior {N_LEGS_BASE} legs; causal phase = leg_age / median '
             f'library leg length. N = {len(obs)} frame-obs.',
             '',
             '| unhealthy features | mean fwd px | 95% CI | n |',
             '|---|---|---|---|']
    for c in sorted(base):
        lo, hi = ci.get(c, (float('nan'), float('nan')))
        lines.append(f"| {c}{'+' if c == 4 else ''} | {base[c]:+.2f} "
                     f"| [{lo:+.2f}, {hi:+.2f}] | {counts.get(c, 0)} |")
    verdict = 'PROMISING — expand' if (
        len(base) > 2 and base.get(0, 0) > 0
        and min(base.get(c, 0) for c in base if c >= 2) < 0) else \
        'NOT promising at pilot scale'
    lines += ['', f'**PILOT VERDICT: {verdict}**']
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    cs = sorted(base)
    ys = [base[c] for c in cs]
    yerr = [[base[c] - ci[c][0] if c in ci else 0 for c in cs],
            [ci[c][1] - base[c] if c in ci else 0 for c in cs]]
    ax.bar([str(c) + ('+' if c == 4 else '') for c in cs], ys, yerr=yerr,
           color=['tab:green' if y > 0 else 'tab:red' for y in ys],
           alpha=0.85, capsize=6)
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel('features displaced beyond the phase-matched healthy norm')
    ax.set_ylabel(f'mean px change, next {K} bars')
    ax.set_title(f'Displacement health v3 (FAST indicators) (leg-phase aligned) — {N_DAYS_PILOT}-day pilot')
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
