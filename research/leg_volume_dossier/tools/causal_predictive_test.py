#!/usr/bin/env python3
"""CAUSAL PREDICTIVE TEST — the actionability gate (owner 2026-07-25: "how
can we turn this into actionable data"). Descriptive dossier findings used
peak-aligned hindsight; a tradeable signal must fire CAUSALLY. Test:

  SIGNAL (computable at bar t, no future): within the current leg,
    - a leg-pure volume climax fired (z >= 2 vs this leg's own baseline)
      at least LAG bars ago, AND
    - 1m price velocity has gone flat/negative (<= 0 favorable-signed)
  PREDICTION: the leg's peak occurs within the next K bars.

Measured against base rate P(peak within K bars) at matched leg phases,
with DAY-BLOCK bootstrap CI on the lift (house rules). Also reports the
economics: favorable px change over the K bars after signal vs non-signal
(what acting on it would have been worth, pre-friction).

Run: python research/leg_volume_dossier/tools/causal_predictive_test.py
Writes reports/causal_predictive_test.md.
"""
import glob
import json
import os
import random
import re
import statistics as st

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
PACKETS = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'gen0', 'packets')
ATLAS_1M = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
OUT_MD = os.path.join(PROJ, 'reports', 'causal_predictive_test.md')

PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')
VEL = re.compile(r'\[1m\].*?price_velocity_30=([+-]?\d+(?:\.\d+)?)')

Z_CLIMAX = 2.0
LAG = 2          # climax must be >= this many bars old (leading, not instant)
K = 3            # prediction horizon: peak within next K bars
N_BOOT = 4000
SEED = 42


def parse_frame(text):
    px = leg = vel = None
    m = VEL.search(text)
    if m:
        vel = float(m.group(1))
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('local:'):
            m = PX.search(s)
            if m:
                px = float(m.group(1))
            m = LEG.search(s)
            if m:
                leg = float(m.group(1))
    return px, leg, vel


def main():
    day_cache = {}
    obs = []   # dict(day, signal, peak_within_k, fwd_px)
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        eid = os.path.basename(pkt_path).replace('.json', '')
        parts = eid.split('_')
        day_key, epoch = f"{parts[0]}_{parts[1]}_{parts[2]}", int(parts[3])
        if day_key not in day_cache:
            f = os.path.join(ATLAS_1M, f'{day_key}.parquet')
            day_cache[day_key] = pd.read_parquet(f) if os.path.exists(f) else None
        bars = day_cache[day_key]
        if bars is None:
            continue
        pkt = json.load(open(pkt_path))
        info = [parse_frame(fr['text']) for fr in pkt['frames']]
        px_path = [p for p, _, _ in info]
        valid = [(i, p) for i, p in enumerate(px_path) if p is not None]
        if len(valid) < 10:
            continue
        peak_i, _ = max(valid, key=lambda t: t[1])
        base_min = (epoch // 60) * 60
        vols = bars.set_index('timestamp')['volume']

        # walk forward through frames; maintain causal leg state
        climax_at = None
        cur_leg_start = None
        for i, (px, leg_age, vel) in enumerate(info):
            if px is None or leg_age is None:
                continue
            leg_start = int(i - leg_age)
            if cur_leg_start is None or leg_start != cur_leg_start:
                # new leg detected (leg_age reset) -> reset climax memory
                if cur_leg_start is None or abs(leg_start - cur_leg_start) > 1:
                    climax_at = None
                cur_leg_start = leg_start
            # causal leg-pure z at bar i
            ts0 = base_min + max(0, leg_start) * 60
            ts = base_min + i * 60
            seg = vols.loc[ts0:ts]
            if len(seg) >= 4:
                w = seg.iloc[:-1]
                sd = w.std()
                if sd and sd == sd:
                    z = (seg.iloc[-1] - w.mean()) / sd
                    if z >= Z_CLIMAX and climax_at is None:
                        climax_at = i
            # the causal signal
            signal = (climax_at is not None and (i - climax_at) >= LAG
                      and vel is not None and vel <= 0)
            # forward outcomes (labels use the future; features do not)
            peak_within = int(0 <= (peak_i - i) <= K)
            j = i + K
            fwd = None
            if j < len(px_path) and px_path[j] is not None:
                fwd = px_path[j] - px
            if fwd is not None:
                obs.append(dict(day=day_key, signal=int(signal),
                                peak=peak_within, fwd=fwd))

    days = sorted({o['day'] for o in obs})
    by_day = {d: [o for o in obs if o['day'] == d] for d in days}

    def rates(sample):
        sig = [o for o in sample if o['signal']]
        non = [o for o in sample if not o['signal']]
        p_sig = st.mean([o['peak'] for o in sig]) if sig else float('nan')
        p_non = st.mean([o['peak'] for o in non]) if non else float('nan')
        f_sig = st.mean([o['fwd'] for o in sig]) if sig else float('nan')
        f_non = st.mean([o['fwd'] for o in non]) if non else float('nan')
        return p_sig, p_non, f_sig, f_non

    p_sig, p_non, f_sig, f_non = rates(obs)
    rng = random.Random(SEED)
    lifts, fwd_deltas = [], []
    for _ in range(N_BOOT):
        sample = []
        for d in rng.choices(days, k=len(days)):
            sample.extend(by_day[d])
        a, b, fa, fb = rates(sample)
        if a == a and b == b:
            lifts.append(a - b)
        if fa == fa and fb == fb:
            fwd_deltas.append(fa - fb)
    lifts.sort()
    fwd_deltas.sort()

    def ci(xs):
        return (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs))]) if xs else (0, 0)

    lo, hi = ci(lifts)
    flo, fhi = ci(fwd_deltas)
    n_sig = sum(o['signal'] for o in obs)
    sig_ok = lo > 0
    lines = [
        '# Causal predictive test — is the climax+flat signal actionable?',
        f'signal = leg-pure vol climax (z>={Z_CLIMAX}) fired >={LAG} bars ago '
        f'AND 1m velocity <= 0. Prediction: leg peak within {K} bars.',
        f'N = {len(obs)} frame-observations, {n_sig} signal-on, '
        f'{len(days)} days (day-block bootstrap, {N_BOOT} resamples).',
        '',
        f'| | signal ON | signal OFF |', '|---|---|---|',
        f'| P(peak within {K} bars) | {p_sig:.1%} | {p_non:.1%} |',
        f'| mean fwd px over {K} bars | {f_sig:+.2f} pts | {f_non:+.2f} pts |',
        '',
        f'**LIFT** = {p_sig - p_non:+.1%}, 95% CI [{lo:+.1%}, {hi:+.1%}] — '
        + ('**SIGNIFICANT**' if sig_ok else '**NOT significant** (CI includes 0)'),
        f'**FWD-PX delta** = {f_sig - f_non:+.2f} pts, '
        f'95% CI [{flo:+.2f}, {fhi:+.2f}]',
        '',
        'Decision rule (pre-stated): CI(lift) > 0 => ship as (1) knowledge-pack '
        'v2 line, (2) student feature spec, (3) control-plane strike input. '
        'Else => descriptive-only; do not ship.',
    ]
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
