#!/usr/bin/env python3
"""BAR-ANATOMY VIGOR (owner 2026-07-25: "open vs close displacement +
high/low bar is going to show a pattern that aligns with the concept of
vigor"). Two tests on the [1m] bar anatomy (favorable-signed):
  conviction = body / bar_range        (kept travel; + = with the trade)
  close_pos  = (body + lower_wick) / bar_range   (where it settled, 0..1)
1. VIGOR CURVE: mean conviction & close_pos across leg phase (0..1.5).
2. FADE DETECTOR: conviction drops >= 1 sigma below THIS leg's running norm
   (leg-pure, causal, LAG 2) -> fwd 3-bar px vs otherwise, day-block CI.
Writes reports/bar_anatomy_vigor.md + assets/bar_anatomy_vigor.png.
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
OUT_MD = os.path.join(PROJ, 'reports', 'bar_anatomy_vigor.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'bar_anatomy_vigor.png')

KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')
PHASES = [i / 10 for i in range(0, 11)] + [1.25, 1.5]
Z_FADE = 1.0
LAG = 2
K = 3
N_BOOT = 2000
SEED = 42


def parse(text):
    d, px, leg = {}, None, None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('[1m]'):
            kv = dict(KV.findall(s))
            for k in ('body', 'bar_range', 'upper_wick', 'lower_wick'):
                if k in kv:
                    d[k] = float(kv[k])
        elif s.startswith('local:'):
            m = PX.search(s)
            if m:
                px = float(m.group(1))
            m = LEG.search(s)
            if m:
                leg = float(m.group(1))
    conv = cp = None
    if 'body' in d and d.get('bar_range'):
        conv = d['body'] / d['bar_range']
        if 'lower_wick' in d:
            cp = (d['body'] + d['lower_wick']) / d['bar_range']
    return conv, cp, px, leg


def main():
    curve_conv = {p: [] for p in PHASES}
    curve_cp = {p: [] for p in PHASES}
    obs = []
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        eid = os.path.basename(pkt_path).replace('.json', '')
        day = "_".join(eid.split('_')[:3])
        pkt = json.load(open(pkt_path))
        rows = [parse(fr['text']) for fr in pkt['frames']]
        px_path = [p for _, _, p, _ in rows]
        valid = [(i, p) for i, p in enumerate(px_path) if p is not None]
        if len(valid) < 8:
            continue
        peak_i, _ = max(valid, key=lambda t: t[1])
        leg_age_pk = rows[peak_i][3]
        if leg_age_pk and leg_age_pk >= 4:
            leg_start_pk = int(peak_i - leg_age_pk)
            for ph in PHASES:
                j = round(leg_start_pk + ph * leg_age_pk)
                if 0 <= j < len(rows) and rows[j][0] is not None:
                    curve_conv[ph].append(rows[j][0])
                    if rows[j][1] is not None:
                        curve_cp[ph].append(rows[j][1])
        # causal fade detector across the whole episode
        fade_at = None
        prev_leg_start = None
        for i, (conv, cp, px, leg_age) in enumerate(rows):
            if px is None or leg_age is None:
                continue
            leg_start = int(i - leg_age)
            if prev_leg_start is None or abs(leg_start - prev_leg_start) > 1:
                fade_at = None
            prev_leg_start = leg_start
            base = [rows[j][0] for j in range(max(0, leg_start), i)
                    if rows[j][0] is not None]
            if conv is not None and len(base) >= 3:
                sd = st.pstdev(base)
                if sd and (conv - st.mean(base)) / sd <= -Z_FADE and fade_at is None:
                    fade_at = i
            j = i + K
            fwd = (px_path[j] - px
                   if j < len(px_path) and px_path[j] is not None else None)
            if fwd is not None:
                sig = int(fade_at is not None and (i - fade_at) >= LAG)
                obs.append(dict(day=day, signal=sig, fwd=fwd))

    days = sorted({o['day'] for o in obs})
    by_day = {d: [o for o in obs if o['day'] == d] for d in days}

    def delta(ss):
        a = [o['fwd'] for o in ss if o['signal']]
        b = [o['fwd'] for o in ss if not o['signal']]
        return (st.mean(a) - st.mean(b)) if a and b else None

    d0 = delta(obs)
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        ss = []
        for d in rng.choices(days, k=len(days)):
            ss.extend(by_day[d])
        b = delta(ss)
        if b is not None:
            boots.append(b)
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    n_sig = sum(o['signal'] for o in obs)
    m_on = st.mean([o['fwd'] for o in obs if o['signal']])
    m_off = st.mean([o['fwd'] for o in obs if not o['signal']])

    lines = ['# Bar-anatomy vigor (owner hypothesis)',
             'conviction = body/bar_range (favorable-signed); '
             'fade = conviction 1 sigma under THIS leg\'s running norm.',
             '',
             '## Vigor curve: mean conviction across the peak leg',
             '| phase | conviction | close_pos | n |', '|---|---|---|---|']
    for ph in PHASES:
        xs, ys = curve_conv[ph], curve_cp[ph]
        if len(xs) >= 20:
            lines.append(f"| {ph:.2f}{' **PEAK**' if ph == 1.0 else ''} "
                         f"| {st.mean(xs):+.3f} | "
                         f"{st.mean(ys):.3f} | {len(xs)} |")
    lines += ['', '## Conviction-fade detector (causal, LAG 2)',
              f'signal-on {m_on:+.2f} vs off {m_off:+.2f} pts '
              f'(delta {d0:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}], '
              f'n_sig {n_sig}, {len(days)} days) — '
              + ('SIGNIFICANT' if hi < 0 or lo > 0 else 'not significant')]
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    ph_ok = [p for p in PHASES if len(curve_conv[p]) >= 20]
    ax1.plot(ph_ok, [st.mean(curve_conv[p]) for p in ph_ok], 'o-',
             color='tab:blue', lw=2.5, label='conviction (body/range)')
    ax1.plot(ph_ok, [st.mean(curve_cp[p]) - 0.5 for p in ph_ok], 's--',
             color='tab:orange', lw=2, label='close position − 0.5')
    ax1.axvline(1.0, color='black', ls='--', lw=1.2)
    ax1.axhline(0, color='gray', lw=0.8)
    ax1.set_xlabel('leg phase')
    ax1.set_title('Vigor curve: bar conviction across the leg')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)
    ax2.bar(['fade OFF', 'fade ON'], [m_off, m_on],
            color=['tab:green', 'tab:red'], alpha=0.85)
    ax2.axhline(0, color='black', lw=1)
    ax2.set_ylabel(f'mean px, next {K} bars')
    ax2.set_title(f'Conviction-fade detector (Δ {d0:+.1f}, CI [{lo:+.1f},{hi:+.1f}])')
    ax2.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
