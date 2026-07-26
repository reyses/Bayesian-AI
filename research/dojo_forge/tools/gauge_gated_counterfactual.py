#!/usr/bin/env python3
"""GAUGE-GATED EXIT — counterfactual re-score (2026-07-26, autonomous).
Before spending GPU on a gen-3 run, test the mechanism on EXISTING data:
replay the gen-0 census, but VETO any recorded EXIT that fired while the leg
gauge was NOT armed (terminal-phase warning off). Effective exit = the first
recorded-EXIT frame that coincides with an ARMED gauge; if none, the trade
rides to the end (never-bail). Then re-score the ride-edge gate metric.

Decision: if gauge-gating moves teacher−never-bail materially positive (a
state-dependent exit edge appears), a real gen-3 GPU run is justified. If it
just collapses to never-bail, the teacher has no exit edge even when
well-timed — and the Mamba's exit job is trivial/unnecessary.
CPU-only. Writes reports/gauge_gated_counterfactual.md.
"""
import json
import os
import random
import re
import statistics as st
import sys

DOJO = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(DOJO, '..', 'leg_volume_dossier', 'pipeline'))
from leg_health_gauge import LegHealthGauge, SICK_DETECTORS  # noqa: E402

CENSUS = os.path.join(DOJO, 'gate_state', 'acceptance_results_tiered.jsonl')
PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
OUT = os.path.join(DOJO, 'reports', 'gauge_gated_counterfactual.md')

PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
LEG = re.compile(r'leg age (\d+)m')
NEED = sorted({f for f, _ in SICK_DETECTORS} | {'body', 'bar_range'})
RIDE_FLOOR = 10.0
HOLD_5M = 5
N_BOOT = 4000
SEED = 42


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
        pkt = json.load(open(os.path.join(PACKETS, f'{eid}.json')))
        parsed = [parse(fr['text']) for fr in pkt['frames']]
        pxs = [p for _, p, _ in parsed]
        valid = [(i, p) for i, p in enumerate(pxs) if p is not None]
        if len(valid) < HOLD_5M + 2:
            continue
        peak = max(p for _, p in valid)
        if peak < RIDE_FLOOR:
            continue
        final = valid[-1][1]
        # per-frame teacher decision from census
        dec = {f['frame_idx']: f.get('decision') for f in r.get('frames', [])}
        # walk the gauge; find first recorded-EXIT frame that is ALSO armed
        g = LegHealthGauge()
        raw_exit = r.get('exit_frame')
        gated_exit = None
        for i, (feats, px, leg_age) in enumerate(parsed):
            if px is None or leg_age is None:
                continue
            s = g.update(leg_age=leg_age, feats=feats)
            if dec.get(i) == 'EXIT' and s['armed'] and gated_exit is None:
                gated_exit = i
        raw_px = pxs[raw_exit] if (raw_exit is not None and pxs[raw_exit] is not None) else final
        gated_px = pxs[gated_exit] if (gated_exit is not None and pxs[gated_exit] is not None) else final
        hold5 = next((p for i, p in valid if i >= HOLD_5M), final)
        rows.append(dict(day=day, peak=peak,
                         raw=raw_px / peak, gated=gated_px / peak,
                         nb=final / peak, hold5=hold5 / peak,
                         raw_ex=int(raw_exit is not None),
                         gated_ex=int(gated_exit is not None)))

    days = sorted({r['day'] for r in rows})
    by_day = {d: [r for r in rows if r['day'] == d] for d in days}

    def dm(sample, a, b):
        per = [st.mean([r[a] - r[b] for r in [x for x in sample if x['day'] == d]])
               for d in {r['day'] for r in sample}]
        return st.mean(per) if per else float('nan')

    def ci(a, b):
        base = dm(rows, a, b)
        rng = random.Random(SEED)
        bs = sorted(dm([r for d in rng.choices(days, k=len(days)) for r in by_day[d]], a, b)
                    for _ in range(N_BOOT))
        return base, bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT)]

    raw_nb = ci('raw', 'nb')
    gated_nb = ci('gated', 'nb')
    gated_hold = ci('gated', 'hold5')
    raw_rate = st.mean([r['raw_ex'] for r in rows])
    gated_rate = st.mean([r['gated_ex'] for r in rows])

    lines = [
        '# Gauge-gated exit — counterfactual on gen-0 census',
        f'{len(rows)} ride episodes, {len(days)} days. Exit rate: raw '
        f'{raw_rate:.0%} -> gauge-gated {gated_rate:.0%}.',
        '',
        '| contrast | mean capture Δ | 95% CI |',
        '|---|---|---|',
        f'| raw teacher − never-bail | {raw_nb[0]:+.3f} | [{raw_nb[1]:+.3f}, {raw_nb[2]:+.3f}] |',
        f'| **gauge-gated − never-bail** | **{gated_nb[0]:+.3f}** | [{gated_nb[1]:+.3f}, {gated_nb[2]:+.3f}] |',
        f'| gauge-gated − 5m-hold | {gated_hold[0]:+.3f} | [{gated_hold[1]:+.3f}, {gated_hold[2]:+.3f}] |',
        '',
        'VERDICT: '
        + ('gauge-gating creates a state-dependent exit edge over never-bail '
           '— gen-3 GPU run JUSTIFIED.' if gated_nb[1] > 0 else
           'gauge-gating removes the value destruction but does NOT beat '
           'never-bail — the teacher has no positive exit edge even when '
           'well-timed on this curriculum. The ride edge is "ride, do not '
           'exit"; the Mamba exit-head is near-trivial. Re-scope before spend.'),
    ]
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
