#!/usr/bin/env python3
"""ENTRY-ANCHORED, MAGNITUDE-PRESERVING probe (owner critique 2026-07-26):
price/levels are non-stationary -> ANCHOR at entry (zero out), measure
displacement. Do NOT lazy-z-score (that over-normalizes and destroys the true
magnitude/range). No control-chart spec rules (no spec exists). Frame as
A/B by outcome, measured causally from entry.

Implementation vs the prior rich-probe:
- LEVEL features (SE_*, vol_mean, vol_sigma, swing_noise, price_sigma,
  ldist_std) were EXCLUDED before; now ADDED BACK as entry-anchored
  displacement (value_at_t - value_at_entry) — stationary, magnitude intact.
- STATIONARY features (ratios, probs, velocities, z-scores, band_pos, hurst,
  vr, lambda) kept RAW (magnitude intact).
- NO standardization anywhere. GBM on raw+anchored features (trees preserve
  magnitude via thresholds).
- Outputs: (1) OOS IC vs the +0.050 exclude-levels baseline; (2) A/B cohort:
  good (final px>0) vs bad (final px<0) — entry-anchored feature effect sizes.
CPU-only. reports/probe_entry_anchored.md + assets/probe_entry_anchored.png.
"""
import glob
import json
import math
import os
import random
import re
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
PACKETS = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'gen0', 'packets')
OUT_MD = os.path.join(PROJ, 'reports', 'probe_entry_anchored.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'probe_entry_anchored.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
K = 5
N_BOOT = 4000
SEED = 42
# level (non-stationary) features -> entry-anchor as displacement
LEVEL = ('SE_low_30', 'SE_high_30', 'vol_mean_30', 'vol_sigma_30',
         'swing_noise_30', 'price_sigma_30', 'ldist_std', 'ldist_skew',
         'ldist_kurtosis')
# absolute-price fields: drop entirely (even anchored, they're price scale)
DROP = ('price_mean', 'vwap', 'ldist_min', 'ldist_q1', 'ldist_median',
        'ldist_q3', 'ldist_max', 'ldist_mean', 'ldist_level')


def frame_feats(text):
    d = {}
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('[1m]'):
            for k, v in KV.findall(s):
                if not any(k.startswith(e) for e in DROP):
                    d[k] = float(v)
    return d


def px_of(text):
    for ln in text.splitlines():
        if ln.strip().startswith('local:'):
            m = PX.search(ln)
            return float(m.group(1)) if m else None
    return None


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sp = math.sqrt((st.pvariance(a) + st.pvariance(b)) / 2) or 1e-9
    return (st.mean(a) - st.mean(b)) / sp


def main():
    rows = []          # (day, featvec-dict entry-anchored, fwd, wrong)
    ep_cohort = []      # (entry-anchored feats at ~mid, wrong) for A/B
    keys = set()
    for p in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        day = "_".join(os.path.basename(p).split('_')[:3])
        frames = json.load(open(p))['frames']
        pxs = [px_of(f['text']) for f in frames]
        feats = [frame_feats(f['text']) for f in frames]
        if not feats or not feats[0]:
            continue
        entry = feats[0]
        valid_final = next((x for x in reversed(pxs) if x is not None), None)
        if valid_final is None:
            continue
        wrong = int(valid_final < 0)
        for i in range(len(frames) - K):
            if pxs[i] is None or pxs[i + K] is None or not feats[i]:
                continue
            fe = {}
            for k, v in feats[i].items():
                if k in LEVEL:
                    fe[k + '_dEntry'] = v - entry.get(k, v)   # anchored, raw
                else:
                    fe[k] = v                                  # raw, magnitude kept
            rows.append((day, fe, pxs[i + K] - pxs[i], wrong))
            keys.update(fe)
        mid = len(frames) // 2
        if mid < len(feats) and feats[mid]:
            fm = {}
            for k, v in feats[mid].items():
                fm[(k + '_dEntry') if k in LEVEL else k] = (
                    v - entry.get(k, v) if k in LEVEL else v)
            ep_cohort.append((fm, wrong))
    keys = sorted(keys)

    def vec(d):
        return [d.get(k, 0.0) for k in keys]

    days = sorted({r[0] for r in rows})
    per_day_gic = {}
    for ti in range(8, len(days)):
        tr = [r for r in rows if r[0] in set(days[:ti])]
        te = [r for r in rows if r[0] == days[ti]]
        if len(te) < 20 or len(tr) < 200:
            continue
        Xtr = np.array([vec(d) for _, d, _, _ in tr]); ytr = np.array([f for _, _, f, _ in tr])
        Xte = np.array([vec(d) for _, d, _, _ in te]); yte = np.array([f for _, _, f, _ in te])
        gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=150,
                                            learning_rate=0.05, random_state=SEED)
        gbm.fit(Xtr, ytr)
        gp = gbm.predict(Xte)
        if np.std(gp) > 0 and np.std(yte) > 0:
            per_day_gic[days[ti]] = float(np.corrcoef(gp, yte)[0, 1])

    rng = random.Random(SEED)
    td = sorted(per_day_gic)
    mg = st.mean(per_day_gic.values()) if False else st.mean([per_day_gic[d] for d in td])
    gbs = sorted(st.mean([per_day_gic[d] for d in rng.choices(td, k=len(td))])
                 for _ in range(N_BOOT))
    glo, ghi = gbs[int(0.025 * N_BOOT)], gbs[int(0.975 * N_BOOT)]

    # A/B cohort: good vs bad, entry-anchored feature effect sizes
    good = [f for f, w in ep_cohort if not w]
    bad = [f for f, w in ep_cohort if w]
    ab = []
    for k in keys:
        d = cohen_d([f.get(k, 0.0) for f in bad], [f.get(k, 0.0) for f in good])
        ab.append((k, d))
    ab.sort(key=lambda t: -abs(t[1]))

    lines = [
        '# Entry-anchored, magnitude-preserving probe (owner critique)',
        f'{len(rows):,} frames, {len(days)} days, {len(keys)} features '
        f'(level feats entry-anchored as displacement; NO z-standardization).',
        '',
        f'- **GBM OOS IC (entry-anchored, raw): {mg:+.4f}**, 95% CI '
        f'[{glo:+.4f}, {ghi:+.4f}] '
        + ('SIGNIFICANT' if glo > 0 else 'incl 0'),
        f'- baseline (exclude-levels, z-linear/raw-GBM): +0.027 / +0.050',
        '',
        '## A/B cohort (bad − good, entry-anchored feature effect size, |d|)',
        '| feature | Cohen d (bad vs good) |', '|---|---|',
    ]
    for k, d in ab[:12]:
        lines.append(f"| {k} | {d:+.2f} |")
    lines += ['',
              'Read: whether entry-anchoring the level features (magnitude '
              'kept) raises IC over excluding them tells us if the owner\'s '
              'anchoring fix adds signal. The A/B effect sizes show which '
              'entry-anchored features separate winners from losers — the '
              'magnitude-preserving, spec-free view. Still 16 test days = '
              'underpowered; at-scale build_dataset resolves.']
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    top = ab[:10][::-1]
    ax.barh([k for k, _ in top], [d for _, d in top],
            color=['tab:red' if d > 0 else 'tab:green' for _, d in top], alpha=0.85)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Cohen d: bad-trade minus good-trade (entry-anchored)')
    ax.set_title(f'What separates losers from winners at entry '
                 f'(GBM OOS IC {mg:+.3f})')
    ax.grid(alpha=0.25, axis='x')
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
