#!/usr/bin/env python3
"""RIDE-EDGE GATE SCORER (teacher-before-mamba, owner 2026-07-26).
Scores the LOGIT teacher's exit policy against the pre-registered gate metric
(RIDE_EDGE_GATE_SPEC v2.1 amendment #1):

  metric = per-DAY mean of (episode capture_ratio − same-episode 5m-hold
           capture_ratio), day-level mean, day-block bootstrap CI.

capture_ratio(policy) = px_at_policy_exit / oracle_peak_px  (fraction of the
available favorable move captured). Restricted to RIDE episodes (peak >=
RIDE_FLOOR — there was a move to capture).

Also runs the Q2 "fancy constant" check: teacher vs NEVER-BAIL. If the teacher
does not beat the always-ride constant, the exit policy is state-INdependent
and there is nothing worth distilling into the Mamba (hard fail).

DEV-HOLDOUT ONLY — scores the existing gen-0 census (all 25 curriculum days).
The pre-registered LOCKBOX gate is a separate one-shot, owner-triggered event;
this is the "where does the teacher stand" diagnostic that precedes it.
CPU-only. Writes reports/ride_edge_gate_dev.md.
"""
import glob
import json
import os
import random
import re
import statistics as st

DOJO = os.path.join(os.path.dirname(__file__), '..')
CENSUS = os.path.join(DOJO, 'gate_state', 'acceptance_results_tiered.jsonl')
PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
OUT = os.path.join(DOJO, 'reports', 'ride_edge_gate_dev.md')

PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
RIDE_FLOOR = 10.0      # pts: a real ride existed to be captured
HOLD_5M = 5            # 5m-hold baseline horizon (frames = minutes)
N_BOOT = 4000
SEED = 42


def px_path(eid):
    pkt = json.load(open(os.path.join(PACKETS, f'{eid}.json')))
    out = []
    for fr in pkt['frames']:
        m = PX.search(fr['text'])
        out.append(float(m.group(1)) if m else None)
    return out


def cap(px_at, peak):
    return px_at / peak if peak else 0.0


def main():
    rows = []
    for line in open(CENSUS):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        eid = r['episode_id']
        day = "_".join(eid.split('_')[:3])
        pxs = px_path(eid)
        valid = [(i, p) for i, p in enumerate(pxs) if p is not None]
        if len(valid) < HOLD_5M + 2:
            continue
        peak = max(p for _, p in valid)
        if peak < RIDE_FLOOR:
            continue                                    # not a ride episode
        final = valid[-1][1]
        ef = r.get('exit_frame')
        teach_px = pxs[ef] if (ef is not None and ef < len(pxs)
                               and pxs[ef] is not None) else final
        hold5 = next((p for i, p in valid if i >= HOLD_5M), final)
        rng_i = valid[len(valid) // 3][1]               # crude fixed-horizon ~1/3
        rows.append(dict(
            day=day,
            teacher=cap(teach_px, peak),
            hold5=cap(hold5, peak),
            neverbail=cap(final, peak),
            fixed=cap(rng_i, peak),
            exited=int(ef is not None)))

    days = sorted({r['day'] for r in rows})
    by_day = {d: [r for r in rows if r['day'] == d] for d in days}

    def day_mean(sample, key_a, key_b):
        per = []
        for d in {r['day'] for r in sample}:
            ds = [r for r in sample if r['day'] == d]
            per.append(st.mean([r[key_a] - r[key_b] for r in ds]))
        return st.mean(per) if per else float('nan')

    def boot_ci(key_a, key_b):
        base = day_mean(rows, key_a, key_b)
        rng = random.Random(SEED)
        bs = []
        for _ in range(N_BOOT):
            samp = [r for d in rng.choices(days, k=len(days)) for r in by_day[d]]
            bs.append(day_mean(samp, key_a, key_b))
        bs.sort()
        return base, bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT)]

    gate = boot_ci('teacher', 'hold5')       # THE gate metric
    q2 = boot_ci('teacher', 'neverbail')     # state-dependence vs constant
    nb_vs_hold = boot_ci('neverbail', 'hold5')

    exit_rate = st.mean([r['exited'] for r in rows])
    lines = [
        '# Ride-Edge Gate — DEV-HOLDOUT diagnostic (logit teacher, gen-0 census)',
        f'{len(rows)} ride episodes (peak>={RIDE_FLOOR}pts), {len(days)} days. '
        f'Teacher exit rate on rides: {exit_rate:.0%}.',
        'NOT the lockbox gate — that is a one-shot owner-triggered run. '
        'This says where the teacher stands.',
        '',
        '## Q0 power (rough)',
        f'- {len(days)} days available; day-block CI half-widths below are the '
        'empirical power. If the gate-metric CI half-width > a few ticks of '
        'capture, underpowered.',
        '',
        '## THE GATE METRIC: teacher capture − 5m-hold capture (per-day mean)',
        f'- **{gate[0]:+.3f}**, 95% day-block CI [{gate[1]:+.3f}, {gate[2]:+.3f}] '
        + ('— **beats 5m-hold**' if gate[1] > 0 else
           '— **NOT distinguishable from 5m-hold**' if gate[2] > 0 else
           '— **WORSE than 5m-hold**'),
        '',
        '## Q2 fancy-constant check: teacher − never-bail',
        f'- **{q2[0]:+.3f}**, 95% CI [{q2[1]:+.3f}, {q2[2]:+.3f}] '
        + ('— teacher adds edge OVER the constant (state-dependent)'
           if q2[1] > 0 else
           '— **teacher ≈ never-bail constant — nothing state-dependent '
           'to distill (Q2 HARD-FAIL risk)**' if q2[2] > 0 else
           '— teacher WORSE than the constant (exits destroy capture)'),
        '',
        '## context: never-bail − 5m-hold',
        f'- {nb_vs_hold[0]:+.3f}, 95% CI [{nb_vs_hold[1]:+.3f}, {nb_vs_hold[2]:+.3f}] '
        '(the moat: does riding beat holding-5m on these ride days)',
        '',
        '## Verdict for teacher-before-mamba',
        'If the gate metric ties/loses AND Q2 shows teacher≈never-bail, the '
        'LOGIT teacher is a fancy constant: distilling it yields never-bail, '
        'not a state-dependent ride edge. The teacher needs a distillable '
        'exit-JUDGMENT channel (reasoning/gauge-conditioned) before the Mamba '
        'is justified. Sign is free; the asset is state-dependent magnitude.',
    ]
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
