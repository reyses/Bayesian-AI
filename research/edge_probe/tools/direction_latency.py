#!/usr/bin/env python3
"""DIRECTION-LATENCY CURVE (owner 2026-07-26): once a pivot/entry happens, how
EARLY is the leg's direction detectable? Measures OOS direction-accuracy (AUC)
of the F-space features at bar N-since-entry, for N = 0,1,2,3,5,8,10,12.

If AUC is already high at N~0-2, direction is an AT-PIVOT read and the phase-
0.64 lateness is pure pivot-confirmation lag (speed the detector). If AUC
climbs slowly, direction itself is the late part.

Target = leg direction outcome: wrong (final favorable px < 0) vs right.
Features = rich F-space at frame N. Walk-forward by episode-day, OOS AUC.
CPU-only. reports/direction_latency.md + assets/direction_latency.png.
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
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
PACKETS = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'gen0', 'packets')
OUT_MD = os.path.join(PROJ, 'reports', 'direction_latency.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'direction_latency.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
NS = [0, 1, 2, 3, 5, 8, 10, 12]
N_BOOT = 2000
SEED = 42
DROP = ('price_mean', 'vwap', 'ldist_min', 'ldist_q1', 'ldist_median',
        'ldist_q3', 'ldist_max', 'ldist_mean', 'ldist_level')


def feats(text):
    d = {}
    for ln in text.splitlines():
        if ln.strip().startswith('[1m]'):
            for k, v in KV.findall(ln):
                if not any(k.startswith(e) for e in DROP):
                    d[k] = float(v)
    return d


def pxof(text):
    for ln in text.splitlines():
        if ln.strip().startswith('local:'):
            m = PX.search(ln)
            return float(m.group(1)) if m else None
    return None


def main():
    episodes = []
    keys = set()
    for p in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        day = "_".join(os.path.basename(p).split('_')[:3])
        fr = json.load(open(p))['frames']
        pxs = [pxof(f['text']) for f in fr]
        fe = [feats(f['text']) for f in fr]
        finals = [x for x in pxs if x is not None]
        if len(finals) < max(NS) + 2 or not fe or not fe[0]:
            continue
        wrong = int(finals[-1] < 0)
        episodes.append(dict(day=day, fe=fe, wrong=wrong))
        for d in fe:
            keys.update(d)
    keys = sorted(keys)

    def vec(d):
        return [d.get(k, 0.0) for k in keys]

    days = sorted({e['day'] for e in episodes})
    rng = random.Random(SEED)
    curve = []
    for N in NS:
        # each episode contributes ONE sample: its features at bar N, label wrong
        samples = [(e['day'], vec(e['fe'][N]), e['wrong'])
                   for e in episodes if N < len(e['fe']) and e['fe'][N]]
        per_day_auc = {}
        for ti in range(6, len(days)):
            trd = set(days[:ti]); ted = days[ti]
            tr = [s for s in samples if s[0] in trd]
            te = [s for s in samples if s[0] == ted]
            if len(te) < 4 or len({s[2] for s in tr}) < 2 or len({s[2] for s in te}) < 2:
                continue
            Xtr = np.array([s[1] for s in tr]); ytr = np.array([s[2] for s in tr])
            Xte = np.array([s[1] for s in te]); yte = np.array([s[2] for s in te])
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=120,
                                                 learning_rate=0.05, random_state=SEED)
            clf.fit(Xtr, ytr)
            try:
                per_day_auc[ted] = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
            except ValueError:
                pass
        td = sorted(per_day_auc)
        if not td:
            curve.append((N, float('nan'), float('nan'), float('nan'), 0))
            continue
        mean = st.mean(per_day_auc[d] for d in td) if False else st.mean([per_day_auc[d] for d in td])
        bs = sorted(st.mean([per_day_auc[d] for d in rng.choices(td, k=len(td))])
                    for _ in range(N_BOOT))
        curve.append((N, mean, bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT)], len(td)))

    lines = ['# Direction-latency curve — how early is leg direction detectable?',
             f'{len(episodes)} episodes, {len(days)} days. OOS direction AUC '
             '(predict wrong vs right) using F-space at bar N-since-entry.',
             '',
             '| bars since pivot (N) | OOS AUC | 95% CI | test days |',
             '|---|---|---|---|']
    for N, m, lo, hi, n in curve:
        lines.append(f"| {N} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | {n} |")
    # interpret
    valid = [(N, m) for N, m, lo, hi, n in curve if m == m]
    if valid:
        n0 = valid[0][1]; nlast = valid[-1][1]
        early = any(m >= 0.65 for N, m in valid if N <= 2)
        lines += ['',
                  f'AUC at pivot (N={valid[0][0]}): {n0:.3f}; at N={valid[-1][0]}: {nlast:.3f}.',
                  ('EARLY: direction is largely an at-pivot read -> the phase-0.64 '
                   'lateness is pivot-CONFIRMATION lag, not direction lag. Fix the '
                   'detector speed.' if early else
                   'LATE: direction resolves slowly after the pivot -> direction '
                   'itself is part of the lateness; a faster pivot detector alone '
                   'would enter with weak direction conviction.')]
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    xs = [N for N, m, lo, hi, n in curve if m == m]
    ms = [m for N, m, lo, hi, n in curve if m == m]
    los = [m - lo for N, m, lo, hi, n in curve if m == m]
    his = [hi - m for N, m, lo, hi, n in curve if m == m]
    ax.errorbar(xs, ms, yerr=[los, his], fmt='o-', color='tab:purple', capsize=5)
    ax.axhline(0.5, color='black', lw=1, label='chance')
    ax.axhline(0.86, color='tab:gray', ls='--', label='known ceiling ~0.86')
    ax.set_xlabel('bars since pivot / entry (N)')
    ax.set_ylabel('OOS direction AUC')
    ax.set_title('How early is leg direction detectable after a pivot?')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
