#!/usr/bin/env python3
"""CLIMAX COHORT SPLIT — the dossier's follow-up (2026-07-25): leg-pure
normalization revealed the mid-leg volume climax is a MINORITY-of-legs event
(mean z explodes, median flat). Split legs by whether a leg-pure climax
(z >= Z_CLIMAX in phase 0.25-0.75) occurred and compare how they top:
  - phase distance from climax to peak (does the climax LEAD, and by how much?)
  - post-peak giveback at +0.5 leg (does the climax cohort reverse harder?)
  - peak px vs episode-final px (points the exit would save, per cohort)
Writes reports/climax_cohort.md + assets/climax_cohort_chart.png.
"""
import glob
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
PACKETS = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'gen0', 'packets')
ATLAS_1M = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
OUT_MD = os.path.join(PROJ, 'reports', 'climax_cohort.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'climax_cohort_chart.png')

PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
LEG = re.compile(r'leg age (\d+)m')
Z_CLIMAX = 2.0
MIN_LEG = 4


def parse_frame(text):
    px = leg = None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('local:'):
            m = PX.search(s)
            if m:
                px = float(m.group(1))
            m = LEG.search(s)
            if m:
                leg = float(m.group(1))
    return px, leg


def main():
    day_cache = {}
    cohorts = {True: [], False: []}
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
        px_path = [p for p, _ in info]
        pxs = [(i, p) for i, p in enumerate(px_path) if p is not None]
        if len(pxs) < 8:
            continue
        peak_i, peak_px = max(pxs, key=lambda t: t[1])
        leg_age = info[peak_i][1]
        if not leg_age or leg_age < MIN_LEG:
            continue
        leg_start = int(peak_i - leg_age)
        base_min = (epoch // 60) * 60
        vols = bars.set_index('timestamp')['volume']

        climax_phase = None
        best_z = 0.0
        for j in range(leg_start + 3, peak_i + 1):
            ts0 = base_min + leg_start * 60
            ts = base_min + j * 60
            seg = vols.loc[ts0:ts]
            if len(seg) < 4:
                continue
            window = seg.iloc[:-1]
            sd = window.std()
            if not sd or sd != sd:
                continue
            z = (seg.iloc[-1] - window.mean()) / sd
            phase = (j - leg_start) / leg_age
            if 0.25 <= phase <= 0.75 and z > best_z:
                best_z = z
                climax_phase = phase
        has_climax = best_z >= Z_CLIMAX

        j_half_after = round(peak_i + 0.5 * leg_age)
        px_half_after = (px_path[j_half_after]
                         if j_half_after < len(px_path) and px_path[j_half_after] is not None
                         else None)
        final_px = next(p for p in reversed(px_path) if p is not None)
        cohorts[has_climax].append(dict(
            eid=eid, best_z=best_z, climax_phase=climax_phase,
            lead=((1.0 - climax_phase) * leg_age if climax_phase else None),
            giveback_half=(peak_px - px_half_after
                           if px_half_after is not None else None),
            left_by_neverbail=peak_px - final_px))

    def agg(rows, key):
        xs = [r[key] for r in rows if r.get(key) is not None]
        return (st.mean(xs), st.median(xs), len(xs)) if xs else (0, 0, 0)

    lines = ['# Climax cohort — legs WITH a leg-pure volume climax vs without',
             f'climax = leg-pure z >= {Z_CLIMAX} in phase 0.25-0.75',
             '',
             '| metric | climax cohort | no-climax | ']
    n_c, n_n = len(cohorts[True]), len(cohorts[False])
    lines[-1] = f'| metric | climax (n={n_c}) | no-climax (n={n_n}) |'
    lines.append('|---|---|---|')
    for key, label in [('lead', 'climax→peak lead (minutes)'),
                       ('giveback_half', 'giveback at peak+0.5 leg (pts)'),
                       ('left_by_neverbail', 'peak−final: left by never-bail (pts)')]:
        mc, medc, _ = agg(cohorts[True], key)
        mn, medn, _ = agg(cohorts[False], key)
        lines.append(f"| {label} | {mc:+.1f} (med {medc:+.1f}) "
                     f"| {mn:+.1f} (med {medn:+.1f}) |")
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for has, color, label in [(True, 'tab:red', f'climax legs (n={n_c})'),
                              (False, 'tab:gray', f'no-climax legs (n={n_n})')]:
        xs = [r['left_by_neverbail'] for r in cohorts[has]]
        ax.hist(xs, bins=20, alpha=0.55, color=color, label=label, density=True)
    ax.set_xlabel('points left on the table by never-bail (peak − final)')
    ax.set_ylabel('density')
    ax.set_title('Do climax legs give back more after the top?')
    ax.legend()
    ax.grid(alpha=0.25)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
