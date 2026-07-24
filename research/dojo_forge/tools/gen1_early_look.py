#!/usr/bin/env python3
"""PARTIAL-RUN early look: gen-1 (GENOME v1) vs gen-0 behavioral comparison.

NOT the gate scorer. No PnL, no CI on $ — behavior only, on the episodes gen-1
has finished so far, matched against the SAME episode_ids in the gen-0 tiered
census. Purpose: detect, hours before the run ends, whether gen-1 is
  (a) exiting less prematurely than gen-0 (the intended fix),
  (b) a never-bail CLONE (exit rate ~0 -> gate will read ~0 ambiguously; the
      failure mode pre-registered by consultant-1's fidelity attack), or
  (c) still churning exits like gen-0.
Writes report to research/dojo_forge/reports/gen1_early_look.md (overwrites —
it's a live partial view, not an artifact of record).

Run: python research/dojo_forge/tools/gen1_early_look.py
"""
import json
import os
import statistics as st

DOJO = os.path.join(os.path.dirname(__file__), '..')
GEN0 = os.path.join(DOJO, 'gate_state', 'acceptance_results_tiered.jsonl')
GEN1 = os.path.join(DOJO, 'gate_state', 'acceptance_results_gen1.jsonl')
OUT = os.path.join(DOJO, 'reports', 'gen1_early_look.md')

# p_exit at which we call a frame "near-trigger" for distribution comparison;
# matches the harness's decision threshold of 0.5 halved to catch approach.
NEAR_TRIGGER_P = 0.25


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial trailing line of a live file
            rows[r['episode_id']] = r
    return rows


def summarize(rows, ids):
    sub = [rows[i] for i in ids]
    exited = [r for r in sub if r['exit_frame'] is not None]
    peaks = []
    for r in sub:
        ps = [f['p_exit'] for f in r.get('frames', [])]
        if ps:
            peaks.append(max(ps))
    return {
        'n': len(sub),
        'exit_rate': len(exited) / len(sub) if sub else 0.0,
        'mean_exit_frame': (st.mean([r['exit_frame'] for r in exited])
                            if exited else None),
        'tainted': sum(1 for r in sub if r.get('tainted')),
        'mean_peak_p': st.mean(peaks) if peaks else 0.0,
        'near_trigger_rate': (sum(1 for p in peaks if p >= NEAR_TRIGGER_P)
                              / len(peaks) if peaks else 0.0),
    }


def main():
    g0, g1 = load(GEN0), load(GEN1)
    common = sorted(set(g0) & set(g1))
    if not common:
        print('no overlapping finished episodes yet')
        return
    s0, s1 = summarize(g0, common), summarize(g1, common)

    if s1['exit_rate'] == 0.0 and s1['near_trigger_rate'] < 0.05:
        verdict = ('NEVER-BAIL CLONE WARNING: gen-1 never exits and rarely '
                   'even approaches the trigger. The gate will read ~0 vs '
                   'never-bail ambiguously (consultant-1 fidelity attack). '
                   'Consider aborting to save GPU-days.')
    elif s1['exit_rate'] == 0.0:
        verdict = ('No exits yet but p_exit does approach the trigger on '
                   f"{s1['near_trigger_rate']:.0%} of episodes — discriminating, "
                   'just conservative. Keep running.')
    elif s1['exit_rate'] < s0['exit_rate']:
        verdict = ('Gen-1 exits less than gen-0 on the same episodes — the '
                   'intended direction. PnL judgment waits for the gate scorer.')
    else:
        verdict = 'Gen-1 exits as much or more than gen-0 — churn not fixed.'

    lines = [
        '# gen-1 EARLY LOOK — PARTIAL RUN, NOT THE GATE',
        f'Matched episodes: {len(common)} (gen-1 finished) of {len(g0)} total',
        '',
        '| metric | gen-0 (same eps) | gen-1 |',
        '|---|---|---|',
        f"| exit rate | {s0['exit_rate']:.0%} | {s1['exit_rate']:.0%} |",
        f"| mean exit frame | {s0['mean_exit_frame']} | {s1['mean_exit_frame']} |",
        f"| mean peak p_exit | {s0['mean_peak_p']:.3f} | {s1['mean_peak_p']:.3f} |",
        f"| near-trigger (peak≥{NEAR_TRIGGER_P}) | {s0['near_trigger_rate']:.0%} | {s1['near_trigger_rate']:.0%} |",
        f"| tainted | {s0['tainted']} | {s1['tainted']} |",
        '',
        f'**Early verdict:** {verdict}',
    ]
    report = '\n'.join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
