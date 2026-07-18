"""
Exit Dojo -- the SANDBOX GATE (research/exit_dojo/tools/dojo_gate.py)

Makes stepwise-blindness TRUE BY CONSTRUCTION: an agent can never see frame k+1
until it has committed a decision for frame k carrying the serve-time random nonce.
The gate is the ONLY channel through which an agent may see an episode's frames.

Sub-commands (all take --episode <eid>):
  next
      Serve the next un-served, committable frame. Refuses if the last served frame
      is still un-committed ("commit first"). Refuses (prints EPISODE CLOSED) once a
      binding EXIT has been committed. Each serve mints a fresh random NONCE, records
      it in the append-only transcript, and prints the frame text + `NONCE: <n>`.
  commit --decision HOLD|EXIT --nonce <n> [--frame k] [--reason "..."]
      Accepted ONLY if <n> equals the nonce served for the current (last-served,
      un-committed) frame. Appends the commit to the transcript. The FIRST EXIT is
      BINDING: the gate marks the episode closed and will not serve further frames.
  finish [--summary "..."]
      Records the agent's closing summary and marks the transcript finished.
  status
      Prints a one-line state summary (for fleet resume checks).

State/transcript (append-only) live under reports/full_run/gate_state/:
  <eid>.state.json          machine state (served/committed frames, nonces, closed)
  <eid>.transcript.jsonl    append-only event log (serve|commit|finish) -- the audit

The nonce chain IS the audit: score_full_run.py verifies every commit carries, in
order, the exact nonce the gate served for that frame -> proves the served path was
played sequentially and blind. (Residual risk: an agent could read raw ATLAS itself;
the agent instructions forbid it and any such access shows in its CLI transcript --
this is the graduation firewall the design accepts, see doc 097.)

Truth sidecars (reports/full_run/truth/<eid>.json) are NEVER read or served here.
"""
import os
import sys
import json
import time
import secrets
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO_ROOT = os.path.abspath(os.path.join(HERE, '..'))
# Run directory is the full-run sandbox by DEFAULT (byte-for-byte the original behavior).
# DOJO_RUN_DIR (additive, doc 099) lets a sibling run -- e.g. the wrong-direction dojo --
# reuse this EXACT gate (nonce/serve/commit logic identical) against its own packets +
# gate_state, without forking the security-critical serving code. The fleet subprocess
# sets it in the child env; the agent's gate calls inherit it automatically.
FULL_RUN_DIR = os.environ.get('DOJO_RUN_DIR') or os.path.join(DOJO_ROOT, 'reports', 'full_run')
PACKETS_DIR = os.path.join(FULL_RUN_DIR, 'packets')
GATE_STATE_DIR = os.path.join(FULL_RUN_DIR, 'gate_state')

NONCE_BYTES = 8   # 16 hex chars; unguessable per-serve token binding a commit to its serve


def _paths(eid: str):
    return (os.path.join(PACKETS_DIR, f'{eid}.json'),
            os.path.join(GATE_STATE_DIR, f'{eid}.state.json'),
            os.path.join(GATE_STATE_DIR, f'{eid}.transcript.jsonl'))


def _load_packet(eid: str) -> dict:
    pkt_path, _, _ = _paths(eid)
    if not os.path.exists(pkt_path):
        print(f'ERROR: no packet for episode {eid} ({pkt_path})', file=sys.stderr)
        sys.exit(2)
    with open(pkt_path, encoding='utf-8') as f:
        return json.load(f)


def _load_state(eid: str) -> dict:
    _, st_path, _ = _paths(eid)
    if os.path.exists(st_path):
        with open(st_path, encoding='utf-8') as f:
            return json.load(f)
    return dict(episode_id=eid, served=[], commits=[], pending=None, closed=False,
                exit_frame=None, finished=False, summary=None)


def _save_state(eid: str, state: dict):
    os.makedirs(GATE_STATE_DIR, exist_ok=True)
    _, st_path, _ = _paths(eid)
    tmp = st_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, st_path)


def _log(eid: str, event: dict):
    os.makedirs(GATE_STATE_DIR, exist_ok=True)
    _, _, tr_path = _paths(eid)
    event = dict(ts=time.time(), **event)
    with open(tr_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(event) + '\n')


def cmd_next(eid: str):
    packet = _load_packet(eid)
    frames = packet['frames']
    state = _load_state(eid)

    if state['closed']:
        print('EPISODE CLOSED (a binding EXIT was already committed). Run `finish` with your '
              'summary if you have not.')
        return
    if state['pending'] is not None:
        print(f"ERROR: frame {state['pending']['frame']} was served but not committed. "
              f"Commit HOLD or EXIT for it (with its nonce) before requesting the next frame.",
              file=sys.stderr)
        sys.exit(3)

    k = len(state['commits'])          # next frame index = number of frames already committed
    if k >= len(frames):
        print('NO MORE FRAMES (window exhausted). Run `finish` with your summary.')
        return

    frame = frames[k]
    nonce = secrets.token_hex(NONCE_BYTES)
    state['pending'] = dict(frame=k, nonce=nonce)
    state['served'].append(dict(frame=k, nonce=nonce, ts=time.time()))
    _save_state(eid, state)
    _log(eid, dict(event='serve', frame=k, nonce=nonce))

    if k == 0:
        print(f"=== EPISODE {eid} | {packet['meta']['direction']} | entry P "
              f"{packet['meta']['entry_P']} | {packet['meta']['window_minutes']} frames max ===")
        print(packet['meta']['decision_contract'])
        print(f"({packet['meta']['sign_convention']})")
        print('---')
    print(frame['text'])
    print(f"NONCE: {nonce}")
    print(f"(commit with: python3.11 research/exit_dojo/tools/dojo_gate.py commit "
          f"--episode {eid} --decision HOLD|EXIT --nonce {nonce} --reason \"...\")")


def cmd_commit(eid: str, decision: str, nonce: str, frame: int, reason: str):
    decision = decision.upper()
    if decision not in ('HOLD', 'EXIT'):
        print('ERROR: --decision must be HOLD or EXIT', file=sys.stderr)
        sys.exit(4)
    state = _load_state(eid)
    if state['closed']:
        print('ERROR: episode already closed by a binding EXIT; no further commits accepted.',
              file=sys.stderr)
        sys.exit(5)
    pend = state['pending']
    if pend is None:
        print('ERROR: no frame is currently awaiting a commit. Run `next` first.', file=sys.stderr)
        sys.exit(6)
    if nonce != pend['nonce']:
        print(f"ERROR: nonce mismatch. The current frame ({pend['frame']}) was served with a "
              f"different nonce. You must commit with the exact nonce the gate just printed.",
              file=sys.stderr)
        sys.exit(7)
    if frame is not None and int(frame) != pend['frame']:
        print(f"ERROR: --frame {frame} != current served frame {pend['frame']}.", file=sys.stderr)
        sys.exit(8)

    commit = dict(frame=pend['frame'], nonce=nonce, decision=decision, reason=reason or '')
    state['commits'].append(commit)
    state['pending'] = None
    if decision == 'EXIT':
        state['closed'] = True
        state['exit_frame'] = commit['frame']
    _save_state(eid, state)
    _log(eid, dict(event='commit', **commit))
    if decision == 'EXIT':
        print(f"COMMITTED EXIT at frame {commit['frame']}. Episode CLOSED (first EXIT is binding). "
              f"Run `finish` with your summary.")
    else:
        print(f"COMMITTED HOLD at frame {commit['frame']}. Request the next frame with `next`.")


def cmd_finish(eid: str, summary: str):
    state = _load_state(eid)
    state['finished'] = True
    state['summary'] = summary or ''
    _save_state(eid, state)
    _log(eid, dict(event='finish', summary=summary or '',
                   exit_frame=state['exit_frame'], n_commits=len(state['commits'])))
    print(f"FINISHED episode {eid}. commits={len(state['commits'])} "
          f"exit_frame={state['exit_frame']}.")


def cmd_status(eid: str):
    state = _load_state(eid)
    print(json.dumps(dict(episode_id=eid, n_commits=len(state['commits']),
                          pending=state['pending'], closed=state['closed'],
                          exit_frame=state['exit_frame'], finished=state['finished'])))


def main():
    ap = argparse.ArgumentParser(description='Exit Dojo stepwise-blind sandbox gate')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('next', 'commit', 'finish', 'status'):
        p = sub.add_parser(name)
        p.add_argument('--episode', required=True)
        if name == 'commit':
            p.add_argument('--decision', required=True)
            p.add_argument('--nonce', required=True)
            p.add_argument('--frame', type=int, default=None)
            p.add_argument('--reason', default='')
        if name == 'finish':
            p.add_argument('--summary', default='')
    args = ap.parse_args()

    if args.cmd == 'next':
        cmd_next(args.episode)
    elif args.cmd == 'commit':
        cmd_commit(args.episode, args.decision, args.nonce, args.frame, args.reason)
    elif args.cmd == 'finish':
        cmd_finish(args.episode, args.summary)
    elif args.cmd == 'status':
        cmd_status(args.episode)


if __name__ == '__main__':
    main()
