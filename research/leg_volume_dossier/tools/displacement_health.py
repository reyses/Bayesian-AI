#!/usr/bin/env python3
"""DISPLACEMENT HEALTH MONITOR (owner design, 2026-07-25): zero every fast
[1m] feature at TRADE START (displacement = f(t) - f(entry)); baseline = the
displacement-at-this-minute norm from the PRIOR 30 legs' healthy (pre-peak)
segments, rolling and causal across the curriculum in time order. Health of
the live trade = how far each feature's displacement sits from that norm
(|z| >= 2 = unhealthy feature); composite = count of unhealthy features.

Test (same endpoint as the whole dossier): fwd 3-bar px by health count,
day-block bootstrap CI. Writes reports/displacement_health.md +
assets/displacement_health.png.
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
OUT_MD = os.path.join(PROJ, 'reports', 'displacement_health.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'displacement_health.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')

# fast features to monitor (displacement-meaningful; skip absolute-price levels)
FEATS = ['price_velocity_30', 'price_accel_30', 'vol_velocity_30',
         'vol_accel_30', 'z_se_30', 'band_pos_30', 'hurst_30',
         'reversion_prob_30', 'vr_exact', 'swing_noise_30', 'ldist_std',
         'lambda_hat_21']
N_LEGS_BASE = 30
Z_SICK = 2.0
K = 3
N_BOOT = 2000
SEED = 42


def parse_frame(text):
    feats, px = {}, None
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
    return feats, px


def main():
    # chronological episode order = causal library order
    eps = []
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json')),
                           key=lambda p: int(os.path.basename(p).split('_')[3])):
        eid = os.path.basename(pkt_path).replace('.json', '')
        day = "_".join(eid.split('_')[:3])
        pkt = json.load(open(pkt_path))
        rows = [parse_frame(fr['text']) for fr in pkt['frames']]
        px_path = [p for _, p in rows]
        valid = [(i, p) for i, p in enumerate(px_path) if p is not None]
        if len(valid) < 10:
            continue
        peak_i, _ = max(valid, key=lambda t: t[1])
        entry = rows[0][0]
        disp = []
        for feats, px in rows:
            disp.append({f: feats[f] - entry[f] for f in FEATS
                         if f in feats and f in entry})
        eps.append(dict(day=day, disp=disp, px=px_path, peak=peak_i))

    # rolling library: displacement profiles of prior legs' PRE-PEAK segments
    library = []          # list of (minute, {f: displacement}) from healthy parts
    obs = []
    for ep in eps:
        # health assessment for THIS trade against the last N_LEGS_BASE legs
        lib = library[-N_LEGS_BASE:]
        if len(lib) >= 10:
            # norm per (minute, feature): mean/sd across library legs
            for m, d in enumerate(ep['disp']):
                if ep['px'][m] is None:
                    continue
                sick = 0
                for f in FEATS:
                    vals = [leg[m][f] for leg in lib
                            if m < len(leg) and f in leg[m]]
                    if len(vals) < 8 or f not in d:
                        continue
                    sd = st.pstdev(vals)
                    if not sd:
                        continue
                    if abs((d[f] - st.mean(vals)) / sd) >= Z_SICK:
                        sick += 1
                j = m + K
                fwd = (ep['px'][j] - ep['px'][m]
                       if j < len(ep['px']) and ep['px'][j] is not None else None)
                if fwd is not None:
                    obs.append(dict(day=ep['day'], sick=min(sick, 4), fwd=fwd))
        # add THIS leg's healthy (pre-peak) displacement profile to the library
        library.append(ep['disp'][:ep['peak'] + 1])

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

    lines = ['# Displacement health monitor (owner design)',
             f'features zeroed at trade start; norm = prior {N_LEGS_BASE} legs\' '
             f'pre-peak displacement at the same minute; sick = |z|>={Z_SICK}.',
             f'N = {len(obs)} frame-obs across {len(days)} days.',
             '',
             '| unhealthy features | mean fwd px | 95% CI | n |',
             '|---|---|---|---|']
    for c in sorted(base):
        lo, hi = ci.get(c, (float('nan'), float('nan')))
        lines.append(f"| {c}{'+' if c == 4 else ''} | {base[c]:+.2f} "
                     f"| [{lo:+.2f}, {hi:+.2f}] | {counts.get(c, 0)} |")
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
    ax.set_xlabel('features displaced beyond the healthy-leg norm (count)')
    ax.set_ylabel(f'mean px change, next {K} bars')
    ax.set_title('Trade health vs the last 30 legs: what the tape pays next')
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
