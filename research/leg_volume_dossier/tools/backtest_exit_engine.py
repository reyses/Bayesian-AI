#!/usr/bin/env python3
"""Backtest LegExitEngine vs never-bail on the census ride episodes.
Reports mean pts/ep vs never-bail (day-block CI) AND tail metrics (the engine's
real value is disaster protection, not mean capture). Small grid on (FLOOR,
TIGHT). CPU-only. Writes reports/exit_engine_backtest.md + chart.
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
from leg_exit_engine import LegExitEngine, COST  # noqa: E402
from leg_health_gauge import SICK_DETECTORS       # noqa: E402

PACKETS = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'gen0', 'packets')
OUT_MD = os.path.join(PROJ, 'reports', 'exit_engine_backtest.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'exit_engine_backtest.png')

PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
LEG = re.compile(r'leg age (\d+)m')
NEED = sorted({f for f, _ in SICK_DETECTORS} | {'body', 'bar_range'})
RIDE_FLOOR = 10.0
N_BOOT = 4000
SEED = 42
GRID = [(50, 15), (50, 20), (40, 12), (60, 20)]


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


def run(rows, pxs, floor, tight):
    eng = LegExitEngine(floor=floor, tight=tight)
    for feats, px, leg in rows:
        if px is None or leg is None:
            continue
        d = eng.update(px, leg, feats)
        if d['action'] == 'EXIT':
            return px - COST, d['reason']
    return pxs[-1], 'ride-to-end'


def main():
    eps = []
    for p in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        day = "_".join(os.path.basename(p).split('_')[:3])
        rows = [parse(fr['text']) for fr in json.load(open(p))['frames']]
        pxs = [x for _, x, _ in rows if x is not None]
        if len(pxs) < 7 or max(pxs) < RIDE_FLOOR:
            continue
        eps.append(dict(day=day, rows=[r for r in rows if r[1] is not None],
                        pxs=pxs, final=pxs[-1]))
    days = sorted({e['day'] for e in eps})

    lines = ['# LegExitEngine backtest vs never-bail (census ride eps)',
             f'{len(eps)} ride episodes, {len(days)} days.',
             '',
             '| FLOOR/TIGHT | mean vs NB | 95% CI | exit rate | '
             'worst ep (NB) | worst ep (engine) |',
             '|---|---|---|---|---|---|']
    best = None
    for floor, tight in GRID:
        for e in eps:
            pnl, reason = run(e['rows'], e['pxs'], floor, tight)
            e['_d'] = pnl - e['final']
            e['_ex'] = int(reason != 'ride-to-end')
            e['_eng'] = pnl
        by_day = {d: [e for e in eps if e['day'] == d] for d in days}
        daymean = {d: st.mean([e['_d'] for e in by_day[d]]) for d in days}
        base = st.mean([daymean[d] for d in days])
        rng = random.Random(SEED)
        bs = sorted(st.mean([daymean[d] for d in rng.choices(days, k=len(days))])
                    for _ in range(N_BOOT))
        lo, hi = bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT)]
        exit_rate = st.mean([e['_ex'] for e in eps])
        worst_nb = min(e['final'] for e in eps)
        worst_eng = min(e['_eng'] for e in eps)
        tag = ' **beats NB**' if lo > 0 else ''
        lines.append(f"| {floor}/{tight} | {base:+.1f} | [{lo:+.1f}, {hi:+.1f}] "
                     f"| {exit_rate:.0%} | {worst_nb:+.0f} | {worst_eng:+.0f} |{tag}")
        if best is None or base > best[1]:
            best = ((floor, tight), base, lo, hi, worst_nb, worst_eng)

    (bf, bt), bmean, blo, bhi, wnb, weng = best
    lines += ['',
              f'Best config {bf}/{bt}: mean {bmean:+.1f} pts/ep vs never-bail '
              f'(CI [{blo:+.1f}, {bhi:+.1f}]).',
              f'TAIL: worst single episode never-bail {wnb:+.0f} pts vs engine '
              f'{weng:+.0f} pts — the floor caps the disaster by '
              f'{weng - wnb:+.0f} pts.',
              '',
              'Read honestly: if mean CI includes/below 0, the engine does NOT '
              'beat never-bail on average (expected — every component did not) '
              'but the tail column shows whether the catastrophic floor earns '
              'its small average cost as disaster insurance.']
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # chart: per-episode engine vs never-bail scatter for best config
    for e in eps:
        pnl, reason = run(e['rows'], e['pxs'], bf, bt)
        e['_eng'] = pnl
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.scatter([e['final'] for e in eps], [e['_eng'] for e in eps],
               alpha=0.5, s=25)
    lim = [min(e['final'] for e in eps) - 10, max(e['final'] for e in eps) + 10]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlabel('never-bail px/ep')
    ax.set_ylabel('LegExitEngine px/ep')
    ax.set_title(f'Engine vs never-bail per episode (floor {bf}/tight {bt})\n'
                 'below diagonal = engine gave up gains; left tail = floor saved a disaster')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
