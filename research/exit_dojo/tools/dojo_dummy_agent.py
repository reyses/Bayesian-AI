"""
Exit Dojo -- scripted DUMMY agent (research/exit_dojo/tools/dojo_dummy_agent.py)

A trivial, deterministic python agent that plays ONE episode end-to-end THROUGH the
gate (subprocess calls to dojo_gate.py) -- for VERIFICATION only. It proves the gate's
serve/commit/nonce protocol works and that a stepwise-blind loop completes; it makes
NO use of any lookahead and never touches packet/truth/raw files. It parses only what
`next` prints (the frame text + NONCE), exactly like a real agent would.

Policies:
  --policy hold           HOLD every frame to the window end (no exit)
  --policy exit_at:K       HOLD until frame K, then EXIT at frame K

Run: python3.11 research/exit_dojo/tools/dojo_dummy_agent.py --episode <eid> --policy exit_at:6
"""
import os
import re
import sys
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, 'dojo_gate.py')
PY = sys.executable or 'python3.11'
NONCE_RE = re.compile(r'^NONCE:\s*([0-9a-f]+)\s*$', re.MULTILINE)


def _run(args):
    return subprocess.run([PY, GATE] + args, capture_output=True, text=True)


def play(eid: str, policy: str):
    exit_at = None
    if policy.startswith('exit_at:'):
        exit_at = int(policy.split(':', 1)[1])
    elif policy != 'hold':
        raise SystemExit(f'unknown policy {policy}')

    for step in range(200):   # hard safety bound; real windows are <= ~60 frames
        nx = _run(['next', '--episode', eid])
        out = nx.stdout
        if 'EPISODE CLOSED' in out or 'NO MORE FRAMES' in out:
            break
        if nx.returncode != 0:
            print(f'[dummy] next failed: {nx.stderr}', file=sys.stderr)
            sys.exit(1)
        mo = NONCE_RE.search(out)
        if not mo:
            print(f'[dummy] no NONCE in serve output:\n{out}', file=sys.stderr)
            sys.exit(1)
        nonce = mo.group(1)
        frame_no = _parse_frame_no(out)
        decision = 'EXIT' if (exit_at is not None and frame_no >= exit_at) else 'HOLD'
        reason = f'scripted-{decision.lower()} (dummy policy {policy})'
        cm = _run(['commit', '--episode', eid, '--decision', decision,
                   '--nonce', nonce, '--reason', reason])
        if cm.returncode != 0:
            print(f'[dummy] commit failed: {cm.stderr}', file=sys.stderr)
            sys.exit(1)
        if decision == 'EXIT':
            break
    fin = _run(['finish', '--episode', eid,
                '--summary', f'dummy agent, policy={policy}; scripted verification run.'])
    print(fin.stdout.strip())


def _parse_frame_no(serve_out: str) -> int:
    m = re.search(r'\[t=(\d+)m\]', serve_out)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode', required=True)
    ap.add_argument('--policy', default='hold')
    args = ap.parse_args()
    play(args.episode, args.policy)


if __name__ == '__main__':
    main()
