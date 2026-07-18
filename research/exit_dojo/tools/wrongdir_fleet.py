"""
Wrong-Direction Dojo -- fleet runner (research/exit_dojo/tools/wrongdir_fleet.py)

Task 099. A THIN variant of dojo_fleet.py: same stepwise-blind gate, same scoped-
allowlist launch (NO --dangerously-skip-permissions), resume-safe. The ONLY changes vs
the exit fleet are (1) the reframed agent prompt (bail-on-wrong-direction, not general
exit-timing) and (2) it points the gate at the wrongdir run via the DOJO_RUN_DIR env var
(set in the child subprocess env; the agent's gate calls inherit it).

The agent still interacts ONLY through the gate; it physically cannot look ahead, and the
scoped allowlist `Bash(python3.11 <gate>:*)` denies every other tool call in headless mode
-- which also enforces the no-raw-data firewall by construction (a peek is REFUSED, not
merely forbidden by instruction).

Run:
    python3.11 research/exit_dojo/tools/wrongdir_fleet.py --episodes 1        # one real episode
    python3.11 research/exit_dojo/tools/wrongdir_fleet.py --episodes 200 --parallel 4
    python3.11 research/exit_dojo/tools/wrongdir_fleet.py --only <eid>
"""
import os
import sys
import glob
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO_ROOT = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(DOJO_ROOT, '..', '..'))
WRONGDIR_DIR = os.path.join(DOJO_ROOT, 'reports', 'wrongdir')
PACKETS_DIR = os.path.join(WRONGDIR_DIR, 'packets')
GATE_STATE_DIR = os.path.join(WRONGDIR_DIR, 'gate_state')
SELECTION_JSON = os.path.join(WRONGDIR_DIR, 'selection.json')

GATE_REL = 'research/exit_dojo/tools/dojo_gate.py'   # cwd = ROOT

AGENT_PROMPT = """You are running a WRONG-DIRECTION check on ONE real historical trade replay, \
inside a stepwise-blind sandbox. You interact with it ONLY through a gate program; you physically \
cannot see the next frame until you commit a decision on the current one.

EPISODE ID: {eid}

YOUR ONE JOB: decide, frame by frame, whether this entry is going the WRONG way and will NOT \
recover. The trade is already open in the entry's direction.
- If the entry's thesis is INTACT (it is working, or it is a good trade that merely dips and will \
resolve your way): commit HOLD -- stay in, ride it out even through a dip.
- If the entry is WRONG-DIRECTION (it resolves against you and will not come back): commit EXIT = \
BAIL / flatten NOW. Bail FAST -- the whole point is to cut a wrong-side entry before the loss \
compounds. Your FIRST EXIT is binding and ends the episode.
Do NOT knife a good trade just because it is red for a minute; many good trades dip first. But do \
NOT hold a genuinely wrong-direction entry hoping it comes back.

STRICT RULES:
- Use ONLY the gate commands below (via Bash). Do NOT read any file under \
research/exit_dojo/reports, or DATA/. Do NOT inspect raw parquet, feature stores, or label files. \
Your only window into the episode is the gate's output.
- Play ONE frame at a time: request a frame, decide, commit, repeat.
- Every price number is favorable-signed points from entry (entry = 0.00): positive = the trade is \
working, negative = it is going against you.

LOOP (repeat until the gate says EPISODE CLOSED or NO MORE FRAMES):
1. Run:  python3.11 {gate} next --episode {eid}
2. Read the frame text. Note the printed `NONCE: <n>`.
3. Decide HOLD (thesis intact) or EXIT (wrong-direction -> bail) using ONLY the frames seen so far:
     python3.11 {gate} commit --episode {eid} --decision HOLD --nonce <n> --reason "short reason"
   (use --decision EXIT to bail instead).
4. If you committed EXIT, or the gate says CLOSED / NO MORE FRAMES, stop looping.

FINISH: once the loop ends, run:
     python3.11 {gate} finish --episode {eid} --summary "2-3 sentences: what told you it was \
wrong-direction and you bailed (or why you judged the thesis intact and held), and what you'd \
watch next time"

Begin now. Do not narrate at length between commands; just play the loop."""


def discover_claude_bin():
    if os.environ.get('CLAUDE_BIN'):
        return os.environ['CLAUDE_BIN']
    pats = [
        os.path.join(os.environ.get('APPDATA', ''), 'Claude', 'claude-code', '*', 'claude.exe'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Claude', 'claude-code', '*', 'claude.exe'),
    ]
    cands = []
    for p in pats:
        cands.extend(glob.glob(p))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.basename(os.path.dirname(p)))
    return cands[-1]


def is_finished(eid: str) -> bool:
    st = os.path.join(GATE_STATE_DIR, f'{eid}.state.json')
    if os.path.exists(st):
        try:
            with open(st, encoding='utf-8') as f:
                if json.load(f).get('finished'):
                    return True
        except Exception:
            pass
    tr = os.path.join(GATE_STATE_DIR, f'{eid}.transcript.jsonl')
    if os.path.exists(tr):
        with open(tr, encoding='utf-8') as f:
            for line in f:
                if '"event": "finish"' in line or '"event":"finish"' in line:
                    return True
    return False


def load_runnable(only=None):
    with open(SELECTION_JSON, encoding='utf-8') as f:
        sel = json.load(f)
    eids = [e['eid'] for e in sel['episodes']]
    if only:
        eids = [e for e in eids if e == only]
    return [e for e in eids if os.path.exists(os.path.join(PACKETS_DIR, f'{e}.json'))]


def run_one(eid: str, claude_bin: str, timeout: int, trace: bool = False):
    prompt = AGENT_PROMPT.format(eid=eid, gate=GATE_REL)
    # SECURITY: scoped allowlist ONLY (NO --dangerously-skip-permissions), same as the
    # exit fleet -- every non-gate tool call is denied in headless mode.
    gate_rule = f'Bash(python3.11 {GATE_REL}:*)'
    cmd = [claude_bin, '-p', prompt, '--model', 'sonnet', '--allowedTools', gate_rule]
    if trace:
        cmd += ['--output-format', 'stream-json', '--verbose']
    # point the reused gate at the wrongdir run (inherited by the agent's gate subprocesses)
    child_env = dict(os.environ, DOJO_RUN_DIR=WRONGDIR_DIR)
    log_path = os.path.join(GATE_STATE_DIR, f'{eid}.agent_stdout.txt')
    os.makedirs(GATE_STATE_DIR, exist_ok=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout, env=child_env)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f'# exit_code={r.returncode} dt={time.time()-t0:.0f}s\n')
            f.write('## STDOUT\n'); f.write(r.stdout or '')
            f.write('\n## STDERR\n'); f.write(r.stderr or '')
        return eid, r.returncode, is_finished(eid), time.time() - t0, None
    except subprocess.TimeoutExpired:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f'# TIMEOUT after {timeout}s\n')
        return eid, -1, is_finished(eid), time.time() - t0, 'timeout'
    except Exception as ex:
        return eid, -2, is_finished(eid), time.time() - t0, str(ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--only', default=None)
    ap.add_argument('--claude-bin', default=None)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    claude_bin = args.claude_bin or discover_claude_bin()
    if not claude_bin or not os.path.exists(claude_bin):
        print(f'ERROR: claude binary not found (looked up / $CLAUDE_BIN / --claude-bin). '
              f'Got: {claude_bin}', file=sys.stderr)
        sys.exit(2)

    runnable = load_runnable(only=args.only)
    todo = [e for e in runnable if not is_finished(e)][:args.episodes]
    print(f'claude-bin: {claude_bin}')
    print(f'run dir (gate DOJO_RUN_DIR): {WRONGDIR_DIR}')
    print(f'runnable packets: {len(runnable)} | already finished: '
          f'{sum(1 for e in runnable if is_finished(e))} | running now: {len(todo)} '
          f'(parallel={args.parallel}, timeout={args.timeout}s)')
    for e in todo:
        print(f'  - {e}')
    if args.dry_run or not todo:
        print('[dry-run] launching nothing.' if args.dry_run else 'nothing to run.')
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(run_one, e, claude_bin, args.timeout, args.trace): e for e in todo}
        for fut in as_completed(futs):
            eid, rc, fin, dt, err = fut.result()
            tag = 'FINISHED' if fin else 'INCOMPLETE'
            print(f'  [{tag}] {eid} rc={rc} dt={dt:.0f}s' + (f' err={err}' if err else ''))
            results.append((eid, rc, fin, dt, err))

    nfin = sum(1 for _, _, f, _, _ in results if f)
    print(f'done: {nfin}/{len(results)} finished (transcripts closed).')


if __name__ == '__main__':
    main()
