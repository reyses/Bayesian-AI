#!/usr/bin/env python3
"""TRAILING / SCALE-OUT EXIT POLICIES (owner 2026-07-26: "the policies are not
correct"). The binary exit-at-one-frame class can't capture a ride peak; the
correct class lets winners run and exits on GIVE-BACK from the running peak —
optionally tightening when the leg gauge warns. All causal (running peak uses
past only). Measures captured points vs never-bail, day-block CI, vs the +48
oracle ceiling.

Policies:
  never_bail          : ride to end (the frontier)
  trail(G)            : exit when px <= running_peak - G   (fixed give-back)
  trail_gauge(Gw,Gt)  : give-back Gw normally, tightens to Gt when gauge ARMED
  scale_gauge         : exit HALF when gauge ARMED, trail the rest
Writes reports/trailing_policy.md + assets/trailing_policy.png.
"""
import glob
import json
import os
import random
import re
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
sys.path.insert(0, os.path.join(PROJ, 'pipeline'))
from leg_health_gauge import LegHealthGauge, SICK_DETECTORS  # noqa: E402

PACKETS = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'gen0', 'packets')
OUT_MD = os.path.join(PROJ, 'reports', 'trailing_policy.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'trailing_policy.png')

PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
LEG = re.compile(r'leg age (\d+)m')
NEED = sorted({f for f, _ in SICK_DETECTORS} | {'body', 'bar_range'})
RIDE_FLOOR = 10.0
COST = 1.8            # RT friction pts, charged on any active exit
N_BOOT = 4000
SEED = 42
TRAIL_GRID = [15, 25, 35, 50]


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
    return feats, px, leg


def sim_trail(pxs, give):
    peak = pxs[0]
    for p in pxs[1:]:
        peak = max(peak, p)
        if p <= peak - give:
            return p - COST
    return pxs[-1]


def sim_trail_gauge(rows, pxs, wide, tight):
    g = LegHealthGauge()
    peak = pxs[0]
    for i in range(1, len(rows)):
        feats, px, leg = rows[i]
        if px is None or leg is None:
            continue
        s = g.update(leg_age=leg, feats=feats)
        peak = max(peak, px)
        give = tight if s['armed'] else wide
        if px <= peak - give:
            return px - COST
    return pxs[-1]


def sim_scale_gauge(rows, pxs, wide):
    """Exit half at first ARMED, trail(wide) the other half."""
    g = LegHealthGauge()
    peak = pxs[0]
    half_px = None
    for i in range(1, len(rows)):
        feats, px, leg = rows[i]
        if px is None or leg is None:
            continue
        s = g.update(leg_age=leg, feats=feats)
        peak = max(peak, px)
        if half_px is None and s['armed']:
            half_px = px - COST
        if px <= peak - wide:
            rest = px - COST
            return (half_px + rest) / 2 if half_px is not None else rest
    rest = pxs[-1]
    return (half_px + rest) / 2 if half_px is not None else rest


def main():
    eps = []
    for p in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        eid = os.path.basename(p).replace('.json', '')
        day = "_".join(eid.split('_')[:3])
        rows = [parse(fr['text']) for fr in json.load(open(p))['frames']]
        pxs = [x for _, x, _ in rows if x is not None]
        if len(pxs) < 7 or max(pxs) < RIDE_FLOOR:
            continue
        eps.append(dict(day=day, rows=[r for r in rows if r[1] is not None],
                        pxs=pxs, peak=max(pxs), final=pxs[-1]))

    pol = {'never_bail': lambda e: e['final']}
    for G in TRAIL_GRID:
        pol[f'trail{G}'] = (lambda e, G=G: sim_trail(e['pxs'], G))
    pol['trailG_35/15'] = lambda e: sim_trail_gauge(e['rows'], e['pxs'], 35, 15)
    pol['scaleG_35'] = lambda e: sim_scale_gauge(e['rows'], e['pxs'], 35)

    days = sorted({e['day'] for e in eps})
    by_day = {d: [e for e in eps if e['day'] == d] for d in days}
    oracle = st.mean([st.mean([e['peak'] - e['final'] for e in by_day[d]])
                      for d in days])

    def day_delta(sample, fn):
        per = []
        for d in {e['day'] for e in sample}:
            ds = [e for e in sample if e['day'] == d]
            per.append(st.mean([fn(e) - e['final'] for e in ds]))
        return st.mean(per)

    results = {}
    rng = random.Random(SEED)
    for name, fn in pol.items():
        if name == 'never_bail':
            continue
        base = day_delta(eps, fn)
        bs = sorted(day_delta([e for d in rng.choices(days, k=len(days))
                               for e in by_day[d]], fn) for _ in range(N_BOOT))
        results[name] = (base, bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT)])

    lines = ['# Trailing / scale-out exit policies vs never-bail',
             f'{len(eps)} ride episodes, {len(days)} days. Friction {COST}pt/exit.',
             f'Oracle exit ceiling (peak−final): **+{oracle:.0f} pts/ep**.',
             '',
             '| policy | pts/ep vs never-bail | 95% CI |', '|---|---|---|']
    best = None
    for name, (b, lo, hi) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        star = ' **SIG**' if lo > 0 else ''
        lines.append(f"| {name} | {b:+.1f} | [{lo:+.1f}, {hi:+.1f}]{star} |")
        if lo > 0 and (best is None or b > best[1]):
            best = (name, b, lo, hi)
    lines += ['', ('**Best significant: %s = %+.1f pts/ep vs never-bail '
                   '(CI [%+.1f, %+.1f]) = %.0f%% of the oracle ceiling.**'
                   % (best[0], best[1], best[2], best[3], 100 * best[1] / oracle))
                  if best else
                  '**No trailing policy beats never-bail significantly.**']
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    names = list(results)
    vals = [results[n][0] for n in names]
    err = [[results[n][0] - results[n][1] for n in names],
           [results[n][2] - results[n][0] for n in names]]
    colors = ['tab:green' if results[n][1] > 0 else 'tab:gray' for n in names]
    ax.bar(names, vals, yerr=err, color=colors, alpha=0.85, capsize=5)
    ax.axhline(0, color='black', lw=1)
    ax.axhline(oracle, color='tab:blue', ls='--', lw=1.2,
               label=f'oracle ceiling +{oracle:.0f}')
    ax.set_ylabel('pts/ep captured vs never-bail')
    ax.set_title('Correct policy class: trailing / scale-out (day-block CI)')
    ax.legend()
    ax.tick_params(axis='x', rotation=30)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
