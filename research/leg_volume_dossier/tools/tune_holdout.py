#!/usr/bin/env python3
"""TUNE/HOLDOUT VALIDATION (owner protocol, 2026-07-25): "use 1/4 of the data
to tune randomly and see if it generalizes on the rest."

- Days split RANDOMLY (seeded; day = independence unit): 1/4 TUNE, 3/4 TEST.
- TUNE: grid-search gauge knobs (Z_SICK, Z_FADE, SICK_ARM, LAG) to maximize
  the economic objective on tune days: mean per-episode delta of
  [exit at first ARMED bar] minus [never-bail (episode final px)].
- TEST: knobs LOCKED; on holdout days measure (a) the same delta with
  day-block bootstrap CI, (b) causal timing vs ground truth: distribution of
  (peak_bar − armed_bar) in minutes.
Generalization verdict: holdout CI. This is still within the 25 burned days —
TRUE fresh-day confirmation stays reserved for lockbox-conveyor days.
Writes reports/tune_holdout.md + assets/tune_holdout.png.
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
import leg_health_gauge as G                      # noqa: E402
from leg_health_gauge import LegHealthGauge      # noqa: E402

PACKETS = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'gen0', 'packets')
OUT_MD = os.path.join(PROJ, 'reports', 'tune_holdout.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'tune_holdout.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')
NEED = sorted({f for f, _ in G.SICK_DETECTORS} | {'body', 'bar_range'})
SEED = 42
N_BOOT = 4000

GRID = dict(Z_SICK=[1.5, 2.0, 2.5], Z_FADE=[0.5, 1.0, 1.5],
            SICK_ARM=[1, 2, 3], LAG=[1, 2, 3])


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


def load_episodes():
    eps = []
    for p in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        eid = os.path.basename(p).replace('.json', '')
        day = "_".join(eid.split('_')[:3])
        rows = [parse(fr['text']) for fr in json.load(open(p))['frames']]
        pxs = [(i, x) for i, (_, x, _) in enumerate(rows) if x is not None]
        if len(pxs) < 8:
            continue
        peak_i, _ = max(pxs, key=lambda t: t[1])
        final_px = pxs[-1][1]
        eps.append(dict(day=day, rows=rows, peak=peak_i, final=final_px))
    return eps


def run_policy(ep, knobs):
    """Exit at first ARMED bar; else never-bail. Returns (pnl, armed_bar)."""
    G.Z_SICK, G.Z_FADE, G.LAG = knobs['Z_SICK'], knobs['Z_FADE'], knobs['LAG']
    arm = knobs['SICK_ARM']
    g = LegHealthGauge()
    for i, (feats, px, leg_age) in enumerate(ep['rows']):
        if px is None or leg_age is None:
            continue
        s = g.update(leg_age=leg_age, feats=feats)
        if s['vigor'] == 'FADED' and s['sick'] >= arm:
            return px, i
    return ep['final'], None


def objective(eps, knobs):
    deltas = [run_policy(e, knobs)[0] - e['final'] for e in eps]
    return st.mean(deltas) if deltas else float('-inf')


def main():
    eps = load_episodes()
    days = sorted({e['day'] for e in eps})
    rng = random.Random(SEED)
    tune_days = set(rng.sample(days, max(1, len(days) // 4)))
    tune = [e for e in eps if e['day'] in tune_days]
    test = [e for e in eps if e['day'] not in tune_days]

    best, best_obj = None, float('-inf')
    for zs in GRID['Z_SICK']:
        for zf in GRID['Z_FADE']:
            for sa in GRID['SICK_ARM']:
                for lg in GRID['LAG']:
                    k = dict(Z_SICK=zs, Z_FADE=zf, SICK_ARM=sa, LAG=lg)
                    o = objective(tune, k)
                    if o > best_obj:
                        best, best_obj = k, o

    # locked-knob holdout evaluation
    per_day = {}
    leads = []
    for e in test:
        pnl, armed = run_policy(e, best)
        per_day.setdefault(e['day'], []).append(pnl - e['final'])
        if armed is not None:
            leads.append(e['peak'] - armed)
    test_days = sorted(per_day)
    d0 = st.mean([x for d in test_days for x in per_day[d]])
    boots = []
    for _ in range(N_BOOT):
        ss = [x for d in rng.choices(test_days, k=len(test_days))
              for x in per_day[d]]
        boots.append(st.mean(ss))
    boots.sort()
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]
    fired = len(leads)

    lines = [
        '# Tune/holdout validation (owner protocol)',
        f'{len(days)} days -> tune {sorted(tune_days)} ({len(tune)} eps) / '
        f'test {len(test)} eps on {len(test_days)} days. Seed {SEED}.',
        '',
        f'**Tuned knobs**: {best} (tune objective {best_obj:+.2f} pts/ep)',
        '',
        '## HOLDOUT (knobs locked)',
        f'- exit-at-armed vs never-bail: **{d0:+.2f} pts/ep**, '
        f'95% day-block CI [{lo:+.2f}, {hi:+.2f}] — '
        + ('**GENERALIZES**' if lo > 0 else
           'does NOT generalize (CI includes 0)' if hi > 0 else
           '**NEGATIVE — worse than never-bail**'),
        f'- fired on {fired}/{len(test)} episodes; timing vs ground truth: '
        f'median lead {st.median(leads) if leads else float("nan"):+.0f} min '
        f'(negative = fired AFTER the peak), '
        f'IQR [{st.quantiles(leads, n=4)[0] if len(leads) >= 4 else 0:+.0f}, '
        f'{st.quantiles(leads, n=4)[-1] if len(leads) >= 4 else 0:+.0f}]',
        '',
        'NOTE: still within the 25 burned days; lockbox-conveyor fresh days '
        'remain the final confirmation tier.',
    ]
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    ax1.hist(leads, bins=15, color='tab:blue', alpha=0.8)
    ax1.axvline(0, color='black', ls='--', lw=1.5)
    ax1.set_xlabel('minutes before the true peak the signal fired\n(negative = late)')
    ax1.set_title(f'Signal timing vs ground truth (holdout, n={fired})')
    ax1.grid(alpha=0.25)
    ax2.bar(['never-bail', 'exit at ARMED'], [0, d0],
            yerr=[[0, d0 - lo], [0, hi - d0]],
            color=['tab:gray', 'tab:green' if d0 > 0 else 'tab:red'],
            alpha=0.85, capsize=8)
    ax2.axhline(0, color='black', lw=1)
    ax2.set_ylabel('pts/episode vs never-bail')
    ax2.set_title('Holdout economics (knobs locked from tune 1/4)')
    ax2.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
