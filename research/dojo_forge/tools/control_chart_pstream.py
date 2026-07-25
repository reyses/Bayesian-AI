#!/usr/bin/env python3
"""p-stream control chart — quantifies the owner's 'logit jitters' thesis
(2026-07-24) and sizes the CONTROL-PLANE opportunity from existing run data.

Reads tiered census checkpoints (gen-0 + any gen-N partials) and measures,
per policy proposal:
  IN-CONTROL   p decisive (outside the ambiguous band) AND locally stable
  OUT-OF-CONTROL (escalation-eligible): p in [AMBIG_LO, AMBIG_HI], or
               frame-to-frame |dp| > JITTER_DP, or a 0.5-crossing flip
Payoff metric: what fraction of actual EXIT decisions (gen-0's premature-exit
disease) happened on frames the control chart would have escalated — i.e.,
frames where a reasoning layer would have been consulted BEFORE pulling the
trigger. High fraction => the control plane intercepts the disease cheaply.

Writes research/dojo_forge/reports/control_chart_pstream.md. CPU-only.
"""
import glob
import json
import os
import statistics as st

DOJO = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(DOJO, 'reports', 'control_chart_pstream.md')

# Control limits (proposal v0 — to be ratified before any hybrid run; these
# are ANALYSIS knobs here, reported not tuned-on-outcomes):
AMBIG_LO, AMBIG_HI = 0.20, 0.80   # ambiguous band: sensor is not decisive
JITTER_DP = 0.30                  # frame-to-frame swing = unstable process

SOURCES = {
    'gen0_tiered': os.path.join(DOJO, 'gate_state', 'acceptance_results_tiered.jsonl'),
    'gen1_oneshot_partial': os.path.join(DOJO, 'gate_state', 'acceptance_results_gen1_oneshot_partial.jsonl'),
    'gen1_anchor2p': os.path.join(DOJO, 'gate_state', 'acceptance_results_gen1.jsonl'),
}


def analyze(path):
    eps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                eps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n_frames = 0
    dps = []
    ambig = jitter = flips = escal = 0
    exits_total = exits_escalated = 0
    for ep in eps:
        ps = [f['p_exit'] for f in ep.get('frames', []) if f.get('p_exit') is not None]
        decs = [f.get('decision') for f in ep.get('frames', []) if f.get('p_exit') is not None]
        n_frames += len(ps)
        prev = None
        for i, p in enumerate(ps):
            is_ambig = AMBIG_LO <= p <= AMBIG_HI
            is_jit = prev is not None and abs(p - prev) > JITTER_DP
            is_flip = prev is not None and (prev - 0.5) * (p - 0.5) < 0
            ambig += is_ambig
            jitter += is_jit
            flips += is_flip
            esc = is_ambig or is_jit or is_flip
            escal += esc
            if decs[i] == 'EXIT':
                exits_total += 1
                exits_escalated += esc
            if prev is not None:
                dps.append(abs(p - prev))
            prev = p
    return dict(
        episodes=len(eps), frames=n_frames,
        dp_median=(st.median(dps) if dps else 0),
        dp_p90=(st.quantiles(dps, n=10)[-1] if len(dps) >= 10 else 0),
        ambig_rate=ambig / n_frames if n_frames else 0,
        jitter_rate=jitter / n_frames if n_frames else 0,
        flip_rate=flips / n_frames if n_frames else 0,
        escalation_rate=escal / n_frames if n_frames else 0,
        exits=exits_total,
        exits_intercepted=(exits_escalated / exits_total if exits_total else None),
    )


def main():
    lines = [
        '# p-stream control chart — sizing the control plane',
        f'Limits (proposal v0): ambiguous band [{AMBIG_LO},{AMBIG_HI}], '
        f'jitter |dp|>{JITTER_DP}, plus 0.5-crossing flips. '
        'Escalation-eligible = any of the three.',
        '',
        '| source | eps | frames | median dp | p90 dp | ambig% | jitter% | flip% | ESCALATION% | exits | exits intercepted |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for name, path in SOURCES.items():
        if not os.path.exists(path):
            continue
        r = analyze(path)
        if not r['frames']:
            continue
        inter = ('—' if r['exits_intercepted'] is None
                 else f"{r['exits_intercepted']:.0%}")
        lines.append(
            f"| {name} | {r['episodes']} | {r['frames']} | {r['dp_median']:.3f} "
            f"| {r['dp_p90']:.3f} | {r['ambig_rate']:.0%} | {r['jitter_rate']:.0%} "
            f"| {r['flip_rate']:.0%} | **{r['escalation_rate']:.0%}** "
            f"| {r['exits']} | {inter} |")
    lines += [
        '',
        'Reading: ESCALATION% = cost (fraction of frames the reasoning layer '
        'would be consulted). "Exits intercepted" = benefit ceiling (fraction '
        'of trigger-pulls that would have passed through the control plane '
        'first). A hybrid is attractive when intercepted% is high and '
        'escalation% is low.',
    ]
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
