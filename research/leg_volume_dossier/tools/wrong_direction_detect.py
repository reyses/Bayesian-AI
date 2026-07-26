#!/usr/bin/env python3
"""WRONG-DIRECTION DETECTION (owner 2026-07-25): can the FIRST 5 minutes
tell a wrong-direction entry from a good one? Screening pass.

Labels (outcome, full episode): WRONG = final px < 0. GOOD = final px > 0.
Causal features (first W=5 minutes only): early gauge state (max sick count,
faded?, per-detector fired flags), px@W, min px (max adverse), mean bar
conviction, velocity-sign agreement.
Test per feature/detector: P(wrong | signal) vs base rate P(wrong), with
day-block bootstrap CI on the lift. BASELINE TO BEAT: the trivial signal
px@W < 0 (early loss). A detector matters only if it adds to that.
Writes reports/wrong_direction_detect.md + assets/wrong_direction.png.
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
OUT_MD = os.path.join(PROJ, 'reports', 'wrong_direction_detect.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'wrong_direction.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')
ER = re.compile(r'ER10 ([\d.]+)')
FW = re.compile(r'(\w+)-with\(')
FA = re.compile(r'(\w+)-against\(')
NEED = sorted({f for f, _ in SICK_DETECTORS} | {'body', 'bar_range'})
W = 10
N_BOOT = 4000
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
            m = ER.search(s)
            if m:
                feats['ER10'] = float(m.group(1))
            feats['fires_with'] = len(FW.findall(s))
            feats['fires_against'] = len(FA.findall(s))
    return feats, px, leg


def main():
    rows_out = []
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        eid = os.path.basename(pkt_path).replace('.json', '')
        day = "_".join(eid.split('_')[:3])
        frames = json.load(open(pkt_path))['frames']
        rows = [parse(fr['text']) for fr in frames]
        pxs = [p for _, p, _ in rows if p is not None]
        if len(pxs) < W + 3:
            continue
        wrong = pxs[-1] < 0
        g = LegHealthGauge()
        max_sick, faded = 0, False
        fired = set()
        convs, early_px = [], []
        for i, (feats, px, leg_age) in enumerate(rows[:W]):
            if px is None or leg_age is None:
                continue
            s = g.update(leg_age=leg_age, feats=feats)
            max_sick = max(max_sick, s['sick'])
            faded = faded or s['vigor'] == 'FADED'
            fired.update(s['active'])
            if feats.get('bar_range') and 'body' in feats:
                convs.append(feats['body'] / feats['bar_range'])
            early_px.append(px)
        if not early_px:
            continue
        trough = min(early_px)
        recov = early_px[-1] - trough
        under_frac = st.mean([1 if p < 0 else 0 for p in early_px])
        last = rows[:W][-1][0] if rows[:W] else {}
        rows_out.append(dict(
            day=day, wrong=int(wrong),
            no_recovery=int(early_px[-1] < 0 and recov < 5),
            recovering=int(early_px[-1] < 0 and recov >= 10),
            mostly_under=int(under_frac >= 0.7),
            fires_against=int(sum(r[0].get('fires_against', 0) for r in rows[:W]) >= 2),
            fires_with=int(sum(r[0].get('fires_with', 0) for r in rows[:W]) >= 2),
            low_ER=int(last.get('ER10', 1) < 0.2),
            early_neg=int(early_px[-1] < 0),
            deep_adverse=int(min(early_px) < -10),
            sick2=int(max_sick >= 2),
            faded=int(faded),
            conv_neg=int(bool(convs) and st.mean(convs) < 0),
            **{f'd_{f}': int(f in fired) for f, _ in SICK_DETECTORS}))

    days = sorted({r['day'] for r in rows_out})
    by_day = {d: [r for r in rows_out if r['day'] == d] for d in days}
    base = st.mean([r['wrong'] for r in rows_out])
    signals = ['early_neg', 'no_recovery', 'recovering', 'mostly_under',
               'fires_against', 'fires_with', 'low_ER',
               'deep_adverse', 'sick2', 'faded', 'conv_neg'] + \
              [f'd_{f}' for f, _ in SICK_DETECTORS]
    rng = random.Random(SEED)
    results = []
    for sig in signals:
        on = [r['wrong'] for r in rows_out if r[sig]]
        if len(on) < 12:
            continue
        lift0 = st.mean(on) - base
        boots = []
        for _ in range(N_BOOT):
            ss = [r for d in rng.choices(days, k=len(days)) for r in by_day[d]]
            o = [r['wrong'] for r in ss if r[sig]]
            b = st.mean([r['wrong'] for r in ss])
            if o:
                boots.append(st.mean(o) - b)
        boots.sort()
        lo = boots[int(0.025 * len(boots))]
        hi = boots[int(0.975 * len(boots))]
        results.append(dict(sig=sig, n=len(on), p=st.mean(on), lift=lift0,
                            lo=lo, hi=hi, sig_ok=lo > 0))
    results.sort(key=lambda r: -r['lift'])

    lines = [
        f'# Wrong-direction detection — first {W} minutes (screening)',
        f'{len(rows_out)} episodes, {len(days)} days; base P(wrong)={base:.0%}.',
        f'BASELINE TO BEAT: early_neg (px<0 at minute {W}).',
        '',
        '| signal | n on | P(wrong|on) | lift | 95% CI | sig |',
        '|---|---|---|---|---|---|',
    ]
    for r in results:
        lines.append(f"| {r['sig']} | {r['n']} | {r['p']:.0%} | {r['lift']:+.0%} "
                     f"| [{r['lo']:+.0%}, {r['hi']:+.0%}] "
                     f"| {'YES' if r['sig_ok'] else ''} |")
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    top = results[:8]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    names = [r['sig'] for r in top][::-1]
    lifts = [r['lift'] * 100 for r in top][::-1]
    err = [[(r['lift'] - r['lo']) * 100 for r in top][::-1],
           [(r['hi'] - r['lift']) * 100 for r in top][::-1]]
    ax.barh(names, lifts, xerr=err,
            color=['tab:red' if r['sig_ok'] else 'tab:gray' for r in top][::-1],
            alpha=0.85, capsize=5)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel(f'lift in P(wrong direction) vs base {base:.0%} (pct points)')
    ax.set_title(f'What the first {W} minutes say about a wrong entry')
    ax.grid(alpha=0.25, axis='x')
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
