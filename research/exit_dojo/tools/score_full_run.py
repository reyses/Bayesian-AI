"""
Exit Dojo -- full-run scorer (research/exit_dojo/tools/score_full_run.py)

Reads the gate transcripts (reports/full_run/gate_state/<eid>.transcript.jsonl), VERIFIES
the nonce chain per episode (the stepwise-blind audit), and scores each played episode
against its truth sidecar (reports/full_run/truth/<eid>.json -- never served to agents).

Unlike the pilot scorer (score_decisions.py, single-prompt transcripts), scoring here is
gated on the nonce-chain audit: an episode is only scored if its transcript proves the
served path was played sequentially and blind (every commit carries, in order, the exact
nonce the gate served for that frame; frames committed as a gapless 0..k prefix; at most
one EXIT and it is the last commit).

Captured-vs-refs mirrors the pilot: captured drift at the agent's exit, vs a fixed-5m
hold ref and the oracle (label-end) ref. WRONG-SIDE episodes (type 'instantfail' -- the
hindsight label flips against the entry within the first minutes) are ALSO scored on
EXIT-MINUTE PERCENTILE: there the skill is speed (exit before the loss compounds), so we
report how early in the window the agent bailed.

Run: python3.11 research/exit_dojo/tools/score_full_run.py
Output: reports/full_run/scorecard.md
"""
import os
import json
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO_ROOT = os.path.abspath(os.path.join(HERE, '..'))
FULL_RUN_DIR = os.path.join(DOJO_ROOT, 'reports', 'full_run')
GATE_STATE_DIR = os.path.join(FULL_RUN_DIR, 'gate_state')
TRUTH_DIR = os.path.join(FULL_RUN_DIR, 'truth')
SCORECARD = os.path.join(FULL_RUN_DIR, 'scorecard.md')

FIXED_HOLD_MIN = 5             # fixed-5m hold reference (pilot convention)
MIN_ORACLE_FOR_RATIO = 0.5     # pts; below this the oracle ref is too near 0 for a stable ratio


def read_transcript(path: str):
    serves, commits, finish = [], [], None
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e['event'] == 'serve':
                serves.append(e)
            elif e['event'] == 'commit':
                commits.append(e)
            elif e['event'] == 'finish':
                finish = e
    return serves, commits, finish


def audit_nonce_chain(serves, commits):
    """Return (ok: bool, reason: str, exit_frame: int|None). Verifies sequential blind play:
    frames committed as a gapless 0..k prefix; each commit's nonce == that frame's LAST
    serve nonce; at most one EXIT and it is the terminal commit."""
    # last served nonce per frame (a frame is served exactly once in normal play, but be lenient)
    served_nonce = {}
    served_order = []
    for s in serves:
        served_nonce[s['frame']] = s['nonce']
        served_order.append(s['frame'])
    exit_frame = None
    for i, c in enumerate(commits):
        if c['frame'] != i:
            return False, f'commit #{i} is for frame {c["frame"]} (expected gapless prefix {i})', None
        if c['frame'] not in served_nonce:
            return False, f'commit for frame {c["frame"]} was never served', None
        if c['nonce'] != served_nonce[c['frame']]:
            return False, f'nonce mismatch at frame {c["frame"]}', None
        if c['decision'] == 'EXIT':
            if i != len(commits) - 1:
                return False, f'EXIT at frame {c["frame"]} is not the terminal commit', None
            exit_frame = c['frame']
    # served frames must not run ahead of committed+1 (no skip-ahead serve without commit)
    if served_order != sorted(set(served_order)) or (served_order and served_order[0] != 0):
        return False, 'serve order is not a clean 0..k ascending sequence', None
    if len(served_order) not in (len(commits), len(commits) + 1):
        return False, f'{len(served_order)} serves vs {len(commits)} commits (a frame was ' \
                      f'served without being committed, or vice versa)', None
    return True, 'PASS', exit_frame


def score_episode(eid: str):
    tr_path = os.path.join(GATE_STATE_DIR, f'{eid}.transcript.jsonl')
    truth_path = os.path.join(TRUTH_DIR, f'{eid}.json')
    if not os.path.exists(truth_path):
        return dict(eid=eid, audit='NO-TRUTH', scored=False)
    serves, commits, finish = read_transcript(tr_path)
    ok, reason, exit_frame = audit_nonce_chain(serves, commits)
    with open(truth_path, encoding='utf-8') as f:
        truth = json.load(f)
    drift = truth['per_minute_forward_drift']
    window = truth['window_minutes']
    row = dict(eid=eid, type=truth['type'], audit=('PASS' if ok else f'FAIL: {reason}'),
               scored=ok, window=window, n_commits=len(commits))
    if not ok:
        return row

    forced = exit_frame is None
    eff_exit = min(exit_frame, window) if exit_frame is not None else window
    captured = drift[eff_exit]
    ref5 = drift[min(FIXED_HOLD_MIN, window)]
    lem = truth['label_end_minute']
    oracle_min = lem if (lem is not None and lem <= window) else window
    oracle_ref = drift[oracle_min]
    ratio = captured / oracle_ref if abs(oracle_ref) >= MIN_ORACLE_FOR_RATIO else float('nan')

    row.update(exit_min=('none(forced)' if forced else exit_frame), eff_exit=eff_exit,
               captured=captured, ref_5m=ref5, oracle_ref=oracle_ref, oracle_min=oracle_min,
               ratio=ratio, forced=forced,
               # wrong-side (instantfail) speed metric: fraction of window elapsed at exit
               exit_pctile=(eff_exit / window if window > 0 else float('nan')))
    return row


def main():
    truth_files = sorted(glob.glob(os.path.join(TRUTH_DIR, '*.json')))
    eids = [os.path.splitext(os.path.basename(p))[0] for p in truth_files]
    # only score episodes that actually have a transcript
    played = [e for e in eids
              if os.path.exists(os.path.join(GATE_STATE_DIR, f'{e}.transcript.jsonl'))]
    if not played:
        print(f'no played episodes (no transcripts) in {GATE_STATE_DIR}')
        return
    rows = [score_episode(e) for e in played]

    L = []
    A = L.append
    A('# Exit Dojo -- full-run scorecard (stepwise-blind, gate-audited)')
    A('')
    npass = sum(1 for r in rows if r.get('scored'))
    nfail = len(rows) - npass
    A(f'Played episodes: {len(rows)} | nonce-chain audit PASS: {npass} | FAIL: {nfail}')
    A('')
    A('| eid | type | audit | exit(min) | captured(pts) | 5m-hold ref | oracle ref | '
      'ratio | exit %ile |')
    A('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        if not r.get('scored'):
            A(f"| {r['eid']} | {r.get('type','?')} | {r['audit']} | - | - | - | - | - | - |")
            continue
        rt = f"{r['ratio']:+.2f}" if np.isfinite(r['ratio']) else 'n/a'
        A(f"| {r['eid']} | {r['type']} | PASS | {r['exit_min']} | {r['captured']:+.2f} | "
          f"{r['ref_5m']:+.2f} | {r['oracle_ref']:+.2f}(@{r['oracle_min']}) | {rt} | "
          f"{r['exit_pctile']:.2f} |")

    scored = [r for r in rows if r.get('scored')]
    if scored:
        caps = np.array([r['captured'] for r in scored], float)
        A('')
        A(f'## Totals (scored N={len(scored)})')
        A(f"- mean captured {caps.mean():+.2f} pts | median {np.median(caps):+.2f} pts")
        wrong = [r for r in scored if r['type'] == 'instantfail']
        if wrong:
            ep = np.array([r['exit_pctile'] for r in wrong], float)
            em = np.array([r['eff_exit'] for r in wrong], float)
            A(f"- WRONG-SIDE (instantfail) N={len(wrong)}: median exit minute {np.median(em):.1f}, "
              f"median exit %ile of window {np.median(ep):.2f} (lower = faster bail = better)")
    A('')
    A('_Gate-audited stepwise-blind play. Any rule an agent appears to discover here still '
      'must be codified and pass the sealed 2024/2025-26 harness before belief (graduation '
      'firewall) -- a dojo number is never itself a result._')

    os.makedirs(FULL_RUN_DIR, exist_ok=True)
    with open(SCORECARD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'wrote {SCORECARD}')
    for r in rows:
        print(f"  {r['eid']:32s} {r.get('type','?'):11s} audit={r['audit']}"
              + ('' if not r.get('scored') else
                 f" exit={r['exit_min']} cap={r['captured']:+.2f}"))


if __name__ == '__main__':
    main()
