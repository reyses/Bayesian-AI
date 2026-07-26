#!/usr/bin/env python3
"""WINDOW SWEEP for wrong-direction readability (owner 2026-07-26: "the most
recent study found the 10 minute mark to decide"). We only tested W=5 vs 10;
this finds the actual curve: at each decision minute W, how strongly does the
composite wrong-signal discriminate, with day-block CI, and how many episodes
still qualify (longer W -> fewer, and survivorship creeps in).

Signal per W: score = no_recovery + mostly_under + low_ER - d_ldist_std,
threshold >= 2 (the generalizing composite). Reports lift + CI + n_fired.
Writes reports/window_sweep_wrong.md + assets/window_sweep_wrong.png.
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
OUT_MD = os.path.join(PROJ, 'reports', 'window_sweep_wrong.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'window_sweep_wrong.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')
ER = re.compile(r'ER10 ([\d.]+)')
NEED = sorted({f for f, _ in SICK_DETECTORS} | {'body', 'bar_range'})
WINDOWS = [3, 5, 7, 8, 10, 12, 15]
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
    return feats, px, leg


def score_at(rows, W):
    g = LegHealthGauge()
    fired = set()
    early_px = []
    last = {}
    for feats, px, leg_age in rows[:W]:
        if px is None or leg_age is None:
            continue
        s = g.update(leg_age=leg_age, feats=feats)
        fired.update(s['active'])
        early_px.append(px)
        last = feats
    if not early_px:
        return None
    trough = min(early_px)
    no_recovery = int(early_px[-1] < 0 and early_px[-1] - trough < 5)
    mostly_under = int(st.mean([1 if p < 0 else 0 for p in early_px]) >= 0.7)
    low_ER = int(last.get('ER10', 1) < 0.2)
    coil = int('ldist_std' in fired)
    return no_recovery + mostly_under + low_ER - coil


def main():
    episodes = []
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        day = "_".join(os.path.basename(pkt_path).split('_')[:3])
        rows = [parse(fr['text']) for fr in json.load(open(pkt_path))['frames']]
        pxs = [p for _, p, _ in rows if p is not None]
        if len(pxs) < 6:
            continue
        episodes.append(dict(day=day, rows=rows, wrong=int(pxs[-1] < 0),
                             nbars=len(pxs)))

    rng = random.Random(SEED)
    curve = []
    for W in WINDOWS:
        elig = [e for e in episodes if e['nbars'] >= W + 2]
        for e in elig:
            e['_sc'] = score_at(e['rows'], W)
        rows_ok = [e for e in elig if e['_sc'] is not None]
        days = sorted({e['day'] for e in rows_ok})
        by = {d: [e for e in rows_ok if e['day'] == d] for d in days}
        base = st.mean([e['wrong'] for e in rows_ok])
        on = [e['wrong'] for e in rows_ok if e['_sc'] >= 2]
        lift0 = (st.mean(on) - base) if on else 0.0
        boots = []
        for _ in range(N_BOOT):
            ss = [e for d in rng.choices(days, k=len(days)) for e in by[d]]
            o = [e['wrong'] for e in ss if e['_sc'] >= 2]
            if o:
                boots.append(st.mean(o) - st.mean([e['wrong'] for e in ss]))
        boots.sort()
        lo = boots[int(0.025 * len(boots))] if boots else 0
        hi = boots[int(0.975 * len(boots))] if boots else 0
        curve.append(dict(W=W, n_elig=len(rows_ok), n_fired=len(on),
                          p_wrong=(st.mean(on) if on else float('nan')),
                          base=base, lift=lift0, lo=lo, hi=hi, sig=lo > 0))

    lines = ['# Window sweep — when does the wrong-direction signal switch on?',
             'composite (no_recovery+mostly_under+low_ER-coil) >= 2, day-block CI.',
             '',
             '| minute W | eps eligible | fired | P(wrong|on) | base | lift | 95% CI | sig |',
             '|---|---|---|---|---|---|---|---|']
    for c in curve:
        lines.append(f"| {c['W']} | {c['n_elig']} | {c['n_fired']} | "
                     f"{c['p_wrong']:.0%} | {c['base']:.0%} | {c['lift']:+.0%} "
                     f"| [{c['lo']:+.0%}, {c['hi']:+.0%}] "
                     f"| {'YES' if c['sig'] else ''} |")
    first_sig = next((c['W'] for c in curve if c['sig']), None)
    lines += ['', f'**Signal switches on at minute {first_sig}** '
              '(first W where CI clears 0). Note: eligible-episode count falls '
              'as W grows (short episodes drop out) — survivorship, read with '
              'the n column.']
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    Ws = [c['W'] for c in curve]
    lifts = [c['lift'] * 100 for c in curve]
    err = [[(c['lift'] - c['lo']) * 100 for c in curve],
           [(c['hi'] - c['lift']) * 100 for c in curve]]
    colors = ['tab:red' if c['sig'] else 'tab:gray' for c in curve]
    ax.errorbar(Ws, lifts, yerr=err, fmt='o-', color='black', ecolor='gray',
                capsize=5, zorder=1)
    ax.scatter(Ws, lifts, c=colors, s=90, zorder=2)
    ax.axhline(0, color='black', lw=1)
    if first_sig:
        ax.axvline(first_sig, color='tab:red', ls='--', lw=1.2)
    ax.set_xlabel('decision minute W')
    ax.set_ylabel('wrong-direction lift (pct points vs base)')
    ax.set_title('When does the losing-trade signal become readable?')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
