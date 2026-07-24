#!/usr/bin/env python3
"""Teacher effectiveness vs truth for the tiered full-depth run (doc 149).

For each episode: the teacher's exit minute = first frame with decision EXIT
(p_exit > 0.5); capture = per_minute_forward_drift[exit_minute] (points from
entry at that minute, sign-adjusted so + = favorable). Compared against:
  - ORACLE: truth's best exit (oracle_capture)
  - RIDE-TO-END: drift[-1] (never bail — the never-bail baseline doc 107 crowned)
  - FIXED-5m: drift[5] (the classic fixed-hold baseline)

Reports (CLAUDE.md discipline): PF-based Trade WR, mode+mean with 4,000-resample
bootstrap CI on the teacher-minus-baseline deltas, explicit significance, N.
Writes research/dojo_forge/reports/tiered_effectiveness_2026-07-23.md.

NOTE drift sign: per_minute_forward_drift is points from entry along the trade
direction per the packet builder (verified: oracle_capture == max(drift)), so
larger = better regardless of is_long.
"""
import json
import os
import glob
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO = os.path.dirname(HERE)
CKPT = os.path.join(DOJO, 'gate_state', 'acceptance_results_tiered.jsonl')
TRUTH_DIR = os.path.join(DOJO, 'reports', 'gen0', 'truth')
OUT = os.path.join(DOJO, 'reports', 'tiered_effectiveness_2026-07-23.md')

random.seed(42)


def bootstrap_ci(deltas, n=4000):
    if not deltas:
        return (0.0, 0.0)
    means = []
    for _ in range(n):
        s = [random.choice(deltas) for _ in deltas]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def pf_wr(captures):
    """CLAUDE.md Trade WR = (sum wins / |sum losses|) - 1."""
    wins = sum(c for c in captures if c > 0)
    losses = abs(sum(c for c in captures if c < 0))
    return (wins / losses - 1) if losses > 0 else float('inf')


def main():
    eps = {}
    with open(CKPT) as fh:
        for line in fh:
            r = json.loads(line)
            eps[r['episode_id']] = r

    rows = []
    for eid, rec in eps.items():
        tp = os.path.join(TRUTH_DIR, f'{eid}.json')
        if not os.path.exists(tp):
            continue
        truth = json.load(open(tp))
        drift = truth['per_minute_forward_drift']
        exit_f = rec.get('exit_frame')
        if exit_f is not None and exit_f < len(drift):
            cap = drift[exit_f]
        else:
            cap = drift[-1]                       # never exited -> rode to end
        rows.append(dict(
            eid=eid, typ=truth.get('type'), n_frames=rec['n_frames_evaluated'],
            exit_minute=exit_f, teacher=cap,
            oracle=truth['oracle_capture'], oracle_minute=truth['oracle_minute'],
            ride_end=drift[-1],
            fixed5=drift[5] if len(drift) > 5 else drift[-1],
        ))

    n = len(rows)
    t = [r['teacher'] for r in rows]
    o = [r['oracle'] for r in rows]
    re_ = [r['ride_end'] for r in rows]
    f5 = [r['fixed5'] for r in rows]
    exited = [r for r in rows if r['exit_minute'] is not None]

    def mean(x): return sum(x) / len(x) if x else 0.0
    def med(x):
        s = sorted(x); return s[len(s) // 2] if s else 0.0

    d_ride = [a - b for a, b in zip(t, re_)]
    d_f5 = [a - b for a, b in zip(t, f5)]
    ci_ride = bootstrap_ci(d_ride)
    ci_f5 = bootstrap_ci(d_f5)

    sig = lambda ci: "SIGNIFICANT" if (ci[0] > 0 or ci[1] < 0) else "NOT significant (CI includes 0)"
    lines = [
        "# Teacher effectiveness — tiered full-depth run (2026-07-23)",
        f"N = {n} episodes (156 packets; missing-truth skipped: {156 - n}). "
        f"Teacher exited in {len(exited)}/{n} episodes "
        f"(median exit minute {med([r['exit_minute'] for r in exited]) if exited else '—'}; "
        f"oracle median minute {med([r['oracle_minute'] for r in rows])}).",
        "",
        "| policy | mean pts | median pts | PF-Trade-WR | capture ratio vs oracle |",
        "|---|---|---|---|---|",
        f"| ORACLE (ceiling) | {mean(o):.2f} | {med(o):.2f} | {pf_wr(o):.3f} | 1.00 |",
        f"| TEACHER (qwen tiered) | {mean(t):.2f} | {med(t):.2f} | {pf_wr(t):.3f} | {mean(t)/mean(o):.3f} |",
        f"| RIDE-TO-END (never-bail) | {mean(re_):.2f} | {med(re_):.2f} | {pf_wr(re_):.3f} | {mean(re_)/mean(o):.3f} |",
        f"| FIXED-5m | {mean(f5):.2f} | {med(f5):.2f} | {pf_wr(f5):.3f} | {mean(f5)/mean(o):.3f} |",
        "",
        f"**Teacher − RIDE-TO-END**: {mean(d_ride):+.2f} pts/episode, 95% CI [{ci_ride[0]:+.2f}, {ci_ride[1]:+.2f}] — {sig(ci_ride)} (N={n}, 4000 resamples)",
        f"**Teacher − FIXED-5m**:   {mean(d_f5):+.2f} pts/episode, 95% CI [{ci_f5[0]:+.2f}, {ci_f5[1]:+.2f}] — {sig(ci_f5)} (N={n})",
        "",
        "## Caveats (honest)",
        "- Points-from-entry per truth drift paths; NOT $-net-of-costs; no CI on the oracle ratio.",
        "- gen-0 teacher = 3 seed genome rules — this is the BASELINE generation, not a tuned one.",
        "- Teacher exit = first p_exit>0.5; threshold sensitivity unexplored (labels are continuous).",
        "- Single run, single seed, deterministic; 7/2884 frames ctx-tainted (0.24%, random).",
        "",
        "## Per-type breakdown",
    ]
    from collections import defaultdict
    byt = defaultdict(list)
    for r in rows:
        byt[r['typ']].append(r)
    for typ, rs in sorted(byt.items()):
        tt = [r['teacher'] for r in rs]; oo = [r['oracle'] for r in rs]; rr = [r['ride_end'] for r in rs]
        lines.append(f"- **{typ}** (N={len(rs)}): teacher {mean(tt):.1f} vs ride-end {mean(rr):.1f} vs oracle {mean(oo):.1f}")

    open(OUT, 'w').write("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"\nwritten: {OUT}")


if __name__ == '__main__':
    main()
