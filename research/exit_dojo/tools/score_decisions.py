"""
Exit Dojo -- decision scorer (research/exit_dojo/tools/score_decisions.py)

Scores an agent's HOLD/EXIT decision transcript for one episode against that
episode's ground-truth sidecar (episodes/truth/ep_NN.json -- never shown to the
agent). See ../README.md for the leakage caveat: pilot scores are single-prompt
(the agent sees all frames at once, attention CAN see the future) and are for
HYPOTHESIS GENERATION only, not a sealed-test result.

Input : research/exit_dojo/reports/decisions/ep_NN.txt  (the agent's raw output,
        one line per frame: "t=<min>: HOLD|EXIT -- <reason>", ending in a
        "SUMMARY: ..." line per the decision contract)
Output: research/exit_dojo/reports/pilot_scorecard.md
Run   : python research/exit_dojo/tools/score_decisions.py
"""
import os
import re
import json
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO_ROOT = os.path.abspath(os.path.join(HERE, '..'))
DECISIONS_DIR = os.path.join(DOJO_ROOT, 'reports', 'decisions')
TRUTH_DIR = os.path.join(DOJO_ROOT, 'episodes', 'truth')
SCORECARD_PATH = os.path.join(DOJO_ROOT, 'reports', 'pilot_scorecard.md')

FIXED_HOLD_MIN = 5            # "fixed-5m hold capture" reference baseline
MIN_ORACLE_FOR_RATIO = 0.5    # pts; below this the oracle ref is too close to 0 for a stable ratio

# "t=<int>: HOLD|EXIT -- <reason>" (accepts an em-dash, en-dash, or plain hyphen as separator;
# accepts an optional "m" minutes suffix after the integer -- every pilot wrote "t=7m:")
LINE_RE = re.compile(r'^\s*t\s*=\s*(\d+)\s*m?\s*:\s*(HOLD|EXIT)\b', re.IGNORECASE)

LEAKAGE_CAVEAT = (
    "**Leakage caveat**: pilot episodes are played single-prompt (the agent receives all "
    "frames in one message with a sequential-commitment contract) -- attention CAN see "
    "future frames, so these scores are OPTIMISTIC and are used ONLY for hypothesis "
    "generation. Any discovered rule must be codified and pass the sealed 2024/2025-26 "
    "harness before belief. A true stepwise-blind runner is a later build if measured LLM "
    "performance is ever wanted."
)


def parse_decisions(path: str):
    """Returns (exit_minute_or_None, n_frame_lines_seen)."""
    exit_min = None
    n = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            n += 1
            t, decision = int(m.group(1)), m.group(2).upper()
            if decision == 'EXIT' and exit_min is None:
                exit_min = t
                break   # contract: nothing after the first EXIT is a valid new decision
    return exit_min, n


def score_episode(decisions_path: str, truth_path: str) -> dict:
    with open(truth_path, encoding='utf-8') as f:
        truth = json.load(f)
    drift = truth['per_minute_forward_drift']
    window = truth['window_minutes']

    exit_min, n_seen = parse_decisions(decisions_path)
    forced = exit_min is None                      # spec: "no-exit = forced close at last frame"
    eff_exit = min(exit_min, window) if exit_min is not None else window
    captured = drift[eff_exit]

    ref5 = drift[min(FIXED_HOLD_MIN, window)]

    lem = truth['label_end_minute']
    oracle_min = lem if (lem is not None and lem <= window) else window
    oracle_ref = drift[oracle_min]
    if abs(oracle_ref) >= MIN_ORACLE_FOR_RATIO:
        ratio = captured / oracle_ref
        ratio_txt = f"{ratio:+.2f}"
    else:
        ratio = float('nan')
        ratio_txt = "n/a"

    return dict(
        episode_id=truth['episode_id'], type=truth['type'],
        agent_exit_min=('none (forced)' if forced else exit_min),
        effective_exit_min=eff_exit, captured=captured, ref_5m=ref5,
        oracle_ref=oracle_ref, oracle_min=oracle_min, ratio=ratio, ratio_txt=ratio_txt,
        forced=forced, n_frames_committed=n_seen, window_minutes=window,
    )


def main():
    files = sorted(glob.glob(os.path.join(DECISIONS_DIR, 'ep_*.txt')))
    if not files:
        print(f'no decisions files found in {DECISIONS_DIR}')
        return

    rows = []
    for fp in files:
        ep_id = os.path.splitext(os.path.basename(fp))[0]
        truth_path = os.path.join(TRUTH_DIR, f'{ep_id}.json')
        if not os.path.exists(truth_path):
            print(f'WARN: no truth sidecar for {ep_id} ({truth_path}), skipping')
            continue
        rows.append(score_episode(fp, truth_path))

    if not rows:
        print('no scorable episodes (decisions present but no matching truth sidecars)')
        return

    lines = []
    A = lines.append
    A('# Exit Dojo -- pilot scorecard')
    A('')
    A(LEAKAGE_CAVEAT)
    A('')
    A('| episode | type | agent exit (min) | captured (pts) | 5m-hold ref (pts) | '
      'oracle (label-end) ref (pts) | capture ratio |')
    A('|---|---|---|---|---|---|---|')
    for r in rows:
        A(f"| {r['episode_id']} | {r['type']} | {r['agent_exit_min']} | {r['captured']:+.2f} | "
          f"{r['ref_5m']:+.2f} | {r['oracle_ref']:+.2f} (@t={r['oracle_min']}) | {r['ratio_txt']} |")

    caps = np.array([r['captured'] for r in rows], dtype=float)
    refs5 = np.array([r['ref_5m'] for r in rows], dtype=float)
    oracles = np.array([r['oracle_ref'] for r in rows], dtype=float)
    ratios = np.array([r['ratio'] for r in rows if np.isfinite(r['ratio'])], dtype=float)

    A('')
    A(f'## Totals (N={len(rows)})')
    A(f'- mean captured: {caps.mean():+.2f} pts | median: {np.median(caps):+.2f} pts')
    A(f'- mean 5m-hold ref: {refs5.mean():+.2f} pts | mean oracle(label-end) ref: '
      f'{oracles.mean():+.2f} pts')
    if len(ratios):
        A(f'- mean capture ratio (n={len(ratios)} with a stable denominator): '
          f'{ratios.mean():+.2f} | median: {np.median(ratios):+.2f}')
    else:
        A(f'- capture ratio: no episode had a stable (|oracle ref| >= {MIN_ORACLE_FOR_RATIO:.1f}pt) '
          f'denominator')
    A('')
    A(f'_N={len(rows)} is a pilot sample for hypothesis generation, not a statistically powered '
      f'claim -- no CI is reported (would be uninformatively wide at this N). See the leakage '
      f'caveat above before acting on any of this._')

    with open(SCORECARD_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f'wrote {SCORECARD_PATH}\n')
    for r in rows:
        print(f"{r['episode_id']:8s} type={r['type']:12s} exit={str(r['agent_exit_min']):12s} "
              f"captured={r['captured']:+7.2f} 5m_ref={r['ref_5m']:+7.2f} "
              f"oracle_ref={r['oracle_ref']:+7.2f} ratio={r['ratio_txt']}")


if __name__ == '__main__':
    main()
