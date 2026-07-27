#!/usr/bin/env python3
"""RICH-FEATURE PROBE (blackboard premise test, 2026-07-26). The raw-return
probe found the 1m PRICE sequence efficient (IC ~0). This asks the real
question: does the DERIVED F-space carry causal edge raw price does not?

Same harness/discipline as probe_forward_return, but features = the ~50 rich
1m F-space fields already in the dojo packets (reversion_prob, ldist_*,
lambda_*, velocities, z-scores, band_pos, hurst, ...). Target = forward
K-frame favorable-signed px change (frame-level, thousands of samples).
Walk-forward by EPISODE-DAY (train past days, test next), OOS IC, day-block CI.
Compares a linear (Ridge) and nonlinear (GBM) model; baseline is the raw-return
null (~0).

Verdict:
- rich-feature OOS IC significantly > 0  => F-space carries causal edge (raw
  price does not) => blackboard premise HOLDS; mamba has a real target.
- ~0 => even the engineered state is efficient at this horizon; the premise is
  weak and the hand-built wrong-direction edge may be the ceiling.
CPU-only. Writes reports/probe_rich_features.md + assets/probe_rich_ic.png.
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
PACKETS = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'gen0', 'packets')
OUT_MD = os.path.join(PROJ, 'reports', 'probe_rich_features.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'probe_rich_ic.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
K = 5
N_BOOT = 4000
SEED = 42
# exclude absolute-price-level fields (non-stationary, leak day scale)
EXCLUDE = ('price_mean', 'vwap', 'ldist_min', 'ldist_q1', 'ldist_median',
           'ldist_q3', 'ldist_max', 'ldist_mean', 'ldist_level')


def frame_feats(text):
    # accumulate KV across ALL [1m] lines (packets have an OHLC 'closed-bar'
    # line with no '=' AND a feature line reversion_prob_30=... etc.)
    d = {}
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('[1m]'):
            for k, v in KV.findall(s):
                if not any(k.startswith(e) for e in EXCLUDE):
                    d[k] = float(v)
    return d


def px_of(text):
    for ln in text.splitlines():
        if ln.strip().startswith('local:'):
            m = PX.search(ln)
            return float(m.group(1)) if m else None
    return None


def main():
    rows = []          # (day, featdict, fwd)
    feat_keys = set()
    for p in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        eid = os.path.basename(p).replace('.json', '')
        day = "_".join(eid.split('_')[:3])
        frames = json.load(open(p))['frames']
        pxs = [px_of(f['text']) for f in frames]
        feats = [frame_feats(f['text']) for f in frames]
        for i in range(len(frames) - K):
            if pxs[i] is None or pxs[i + K] is None or not feats[i]:
                continue
            fwd = pxs[i + K] - pxs[i]
            rows.append((day, feats[i], fwd))
            feat_keys.update(feats[i])
    feat_keys = sorted(feat_keys)

    def vec(d):
        return [d.get(k, 0.0) for k in feat_keys]

    days = sorted({r[0] for r in rows})
    # walk-forward by day: expanding train (>=8 days), test next day
    per_day_ic, per_day_gic = {}, {}
    for ti in range(8, len(days)):
        tr_days = set(days[:ti])
        te_day = days[ti]
        tr = [r for r in rows if r[0] in tr_days]
        te = [r for r in rows if r[0] == te_day]
        if len(te) < 20 or len(tr) < 200:
            continue
        Xtr = np.array([vec(d) for _, d, _ in tr]); ytr = np.array([f for _, _, f in tr])
        Xte = np.array([vec(d) for _, d, _ in te]); yte = np.array([f for _, _, f in te])
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
        lin = Ridge(alpha=10.0).fit((Xtr - mu) / sd, ytr)
        lp = lin.predict((Xte - mu) / sd)
        if np.std(lp) > 0 and np.std(yte) > 0:
            per_day_ic[te_day] = float(np.corrcoef(lp, yte)[0, 1])
        gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=150,
                                            learning_rate=0.05, random_state=SEED)
        gbm.fit(Xtr, ytr)
        gp = gbm.predict(Xte)
        if np.std(gp) > 0 and np.std(yte) > 0:
            per_day_gic[te_day] = float(np.corrcoef(gp, yte)[0, 1])

    rng = random.Random(SEED)
    tdays = sorted(per_day_ic)
    lin_ic = list(per_day_ic.values()); gic = list(per_day_gic.values())
    mean_lin = st.mean(lin_ic); mean_g = st.mean(gic) if gic else float('nan')
    bs = sorted(st.mean([per_day_ic[d] for d in rng.choices(tdays, k=len(tdays))])
                for _ in range(N_BOOT))
    lo, hi = bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT)]
    gtdays = sorted(per_day_gic)
    gbs = sorted(st.mean([per_day_gic[d] for d in rng.choices(gtdays, k=len(gtdays))])
                 for _ in range(N_BOOT)) if gtdays else [0, 0]
    glo, ghi = gbs[int(0.025 * len(gbs))], gbs[int(0.975 * len(gbs))]
    sig = lo > 0 or glo > 0

    lines = [
        '# Rich-feature probe — does the F-space carry edge raw price does not?',
        f'{len(rows):,} frames, {len(days)} episode-days, {len(feat_keys)} rich '
        f'features (abs-price-level fields excluded). Fwd {K}-frame px target, '
        'walk-forward by day, OOS.',
        '',
        f'- **LINEAR OOS IC: {mean_lin:+.4f}**, 95% day-block CI [{lo:+.4f}, {hi:+.4f}] '
        + ('SIGNIFICANT' if lo > 0 else 'incl 0'),
        f'- **NONLINEAR (GBM) OOS IC: {mean_g:+.4f}**, 95% CI [{glo:+.4f}, {ghi:+.4f}] '
        + ('SIGNIFICANT' if glo > 0 else 'incl 0'),
        f'- baseline (raw-price probe): IC ~0.005 (not sig)',
        f'- test days: {len(tdays)}',
        '',
        '## Verdict',
        ('The F-space carries causal edge the raw price sequence does NOT — '
         'blackboard premise HOLDS; a learned model on engineered state has a '
         'real, OOS-generalizing target. Next: scale via build_dataset across '
         'ATLAS + confirm; compare learned vs the hand-built wrong-direction '
         'composite (is there structure beyond hand rules?).' if sig else
         'Even the engineered F-space shows no OOS forward-px edge at this '
         'horizon on the packet data. The hand-built wrong-direction/outcome '
         'signals may be the ceiling; a heavy learned loop is not yet '
         'justified. Caveat: 150-episode packets, forward-PX target (not the '
         'trade-OUTCOME target the wrong-dir edge used) — retest at scale + on '
         'the outcome target before concluding.'),
    ]
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.bar(['linear', 'nonlinear(GBM)'], [mean_lin, mean_g],
           yerr=[[mean_lin - lo, mean_g - glo], [hi - mean_lin, ghi - mean_g]],
           color=['tab:blue', 'tab:green'], alpha=0.85, capsize=6)
    ax.axhline(0, color='black', lw=1)
    ax.axhline(0.005, color='tab:gray', ls='--', label='raw-price null ~0.005')
    ax.set_ylabel(f'OOS IC (fwd {K}-frame px)')
    ax.set_title('Rich F-space vs raw price: is there causal edge? (OOS)')
    ax.legend()
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
