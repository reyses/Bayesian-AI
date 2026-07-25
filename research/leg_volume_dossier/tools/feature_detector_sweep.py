#!/usr/bin/env python3
"""FEATURE DETECTOR SWEEP (owner 2026-07-25: "this is only one property of
volume — we have other features we should explore"). Runs the leg-pure tail
detector over EVERY numeric [1m] feature the frames carry, both tails,
same causal protocol as the volume test (event >= LAG bars old), same
economic endpoint (fwd px over K bars), day-block bootstrap CI each.

SCREENING, not confirmation: with ~2 tails x ~50 features, a few
CI-significant rows are expected by chance; survivors must be confirmed on
fresh (lockbox-conveyor) days before shipping.

Run: python research/leg_volume_dossier/tools/feature_detector_sweep.py
Writes reports/feature_detector_sweep.md + assets/feature_sweep_chart.png.
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
OUT_MD = os.path.join(PROJ, 'reports', 'feature_detector_sweep.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'feature_sweep_chart.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')

Z_EVENT = 2.0
LAG = 2
K = 3
N_BOOT = 2000
SEED = 42
MIN_SIGNALS = 40           # tails with fewer events are too sparse to rank


def parse_frame(text):
    feats, px, leg = {}, None, None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('[1m]'):
            for k, v in KV.findall(s):
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
    episodes = []
    feat_names = set()
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        eid = os.path.basename(pkt_path).replace('.json', '')
        day = "_".join(eid.split('_')[:3])
        pkt = json.load(open(pkt_path))
        rows = [parse_frame(fr['text']) for fr in pkt['frames']]
        if sum(1 for _, p, _ in rows if p is not None) < 10:
            continue
        episodes.append((day, rows))
        for f, _, _ in rows:
            feat_names.update(f)
    feat_names = sorted(feat_names)

    # per (feature, tail): observations of (day, signal, fwd)
    obs = {(f, t): [] for f in feat_names for t in ('hi', 'lo')}
    for day, rows in episodes:
        px_path = [p for _, p, _ in rows]
        # causal walk: leg-pure baselines per feature
        events = {}                     # (feat, tail) -> minute event fired
        prev_leg_start = None
        for i, (feats, px, leg_age) in enumerate(rows):
            if px is None or leg_age is None:
                continue
            leg_start = int(i - leg_age)
            if prev_leg_start is None or abs(leg_start - prev_leg_start) > 1:
                events = {}             # new leg -> reset detector memory
            prev_leg_start = leg_start
            for fname in feat_names:
                base = [rows[j][0].get(fname) for j in range(max(0, leg_start), i)]
                base = [b for b in base if b is not None]
                v = feats.get(fname)
                if v is None or len(base) < 3:
                    continue
                sd = st.pstdev(base)
                if not sd:
                    continue
                z = (v - st.mean(base)) / sd
                if z >= Z_EVENT and (fname, 'hi') not in events:
                    events[(fname, 'hi')] = i
                if z <= -Z_EVENT and (fname, 'lo') not in events:
                    events[(fname, 'lo')] = i
            j = i + K
            fwd = (px_path[j] - px
                   if j < len(px_path) and px_path[j] is not None else None)
            if fwd is None:
                continue
            for key, fired_at in events.items():
                sig = int((i - fired_at) >= LAG)
                obs[key].append(dict(day=day, signal=sig, fwd=fwd))
            for fname in feat_names:            # never-fired => signal 0
                for t in ('hi', 'lo'):
                    if (fname, t) not in events:
                        obs[(fname, t)].append(dict(day=day, signal=0, fwd=fwd))

    rng = random.Random(SEED)
    results = []
    for key, sample in obs.items():
        n_sig = sum(o['signal'] for o in sample)
        if n_sig < MIN_SIGNALS:
            continue
        days = sorted({o['day'] for o in sample})
        by_day = {d: [o for o in sample if o['day'] == d] for d in days}

        def delta(ss):
            sig = [o['fwd'] for o in ss if o['signal']]
            non = [o['fwd'] for o in ss if not o['signal']]
            if not sig or not non:
                return None
            return st.mean(sig) - st.mean(non)

        d0 = delta(sample)
        boots = []
        for _ in range(N_BOOT):
            ss = []
            for d in rng.choices(days, k=len(days)):
                ss.extend(by_day[d])
            b = delta(ss)
            if b is not None:
                boots.append(b)
        boots.sort()
        lo_ci = boots[int(0.025 * len(boots))]
        hi_ci = boots[int(0.975 * len(boots))]
        results.append(dict(feat=key[0], tail=key[1], n=n_sig, delta=d0,
                            lo=lo_ci, hi=hi_ci,
                            sig=(lo_ci > 0 or hi_ci < 0)))

    results.sort(key=lambda r: -abs(r['delta']))
    n_tests = len(results)
    n_sig_rows = sum(r['sig'] for r in results)
    lines = [
        '# Feature detector sweep — leg-pure tail events vs fwd 3-bar px',
        f'{n_tests} feature-tails tested (>= {MIN_SIGNALS} events each); '
        f'{n_sig_rows} significant at 95% (chance expectation ~{0.05 * n_tests:.1f}). '
        'SCREENING ONLY — survivors need fresh-day confirmation.',
        '',
        '| feature | tail | n events | fwd-px delta | 95% CI | sig |',
        '|---|---|---|---|---|---|',
    ]
    for r in results:
        lines.append(f"| {r['feat']} | {r['tail']} | {r['n']} "
                     f"| {r['delta']:+.2f} | [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                     f"| {'YES' if r['sig'] else ''} |")
    fams = {}
    for r in results:
        fam = r['feat'].split('_')[0]
        if r['feat'].startswith(('price_velocity', 'price_accel', 'price_mean', 'price_sigma')):
            fam = 'price_' + r['feat'].split('_')[1]
        d = fams.setdefault(fam, dict(n=0, sig=0, best=None))
        d['n'] += 1
        d['sig'] += r['sig']
        if d['best'] is None or abs(r['delta']) > abs(d['best']['delta']):
            d['best'] = r
    lines += ['', '## Family rollup (all tails tested)',
              '| family | tails | sig | best row |', '|---|---|---|---|']
    for fam, d in sorted(fams.items(), key=lambda kv: -kv[1]['sig']):
        b = d['best']
        lines.append(f"| {fam} | {d['n']} | {d['sig']} | {b['feat']} {b['tail']} "
                     f"{b['delta']:+.1f} [{b['lo']:+.1f},{b['hi']:+.1f}] |")
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    top = [r for r in results if r['sig']][:12] or results[:12]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    names = [f"{r['feat']} ({r['tail']})" for r in top][::-1]
    deltas = [r['delta'] for r in top][::-1]
    los = [r['delta'] - r['lo'] for r in top][::-1]
    his = [r['hi'] - r['delta'] for r in top][::-1]
    colors = ['tab:red' if d < 0 else 'tab:green' for d in deltas]
    ax.barh(names, deltas, xerr=[los, his], color=colors, alpha=0.8)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel(f'points over the next {K} bars: signal-on minus signal-off')
    ax.set_title('Which feature spikes predict the tape (95% CI, day-block)')
    ax.grid(alpha=0.25, axis='x')
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines[:20]))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
