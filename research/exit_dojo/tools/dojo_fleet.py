"""
Exit Dojo -- fleet runner (research/exit_dojo/tools/dojo_fleet.py)

Drives real Sonnet agents through the stepwise-blind gate, one headless CLI session per
episode, N-parallel, resume-safe. Each session is `claude -p "<agent prompt>" --model
sonnet` running with Bash so the agent can call dojo_gate.py -- it can ONLY see frames
through the gate, so it physically cannot look ahead.

Resume-safe: an episode whose gate transcript already has a `finish` event (or whose
state is `finished`) is skipped.

Selection comes from reports/full_run/selection.json (built by telescope_packet_builder).
Only episodes whose packet exists are runnable.

Run:
    python3.11 research/exit_dojo/tools/dojo_fleet.py --episodes 1        # first N runnable
    python3.11 research/exit_dojo/tools/dojo_fleet.py --episodes 200 --parallel 4
    python3.11 research/exit_dojo/tools/dojo_fleet.py --only <eid>        # one specific episode
Flags:
    --claude-bin PATH   override the claude executable (else auto-discovered / $CLAUDE_BIN)
    --timeout SEC       per-episode wall clock (default 900)
    --dry-run           print what would run, launch nothing
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
FULL_RUN_DIR = os.path.join(DOJO_ROOT, 'reports', 'full_run')
PACKETS_DIR = os.path.join(FULL_RUN_DIR, 'packets')
GATE_STATE_DIR = os.path.join(FULL_RUN_DIR, 'gate_state')
SELECTION_JSON = os.path.join(FULL_RUN_DIR, 'selection.json')

GATE_REL = 'research/exit_dojo/tools/dojo_gate.py'   # cwd = ROOT

AGENT_PROMPT = """You are drilling EXIT decisions on ONE real historical trade replay, inside a \
stepwise-blind sandbox. You interact with it ONLY through a gate program; you physically cannot \
see the next frame until you commit a decision on the current one.

EPISODE ID: {eid}

STRICT RULES:
- Use ONLY the three gate commands below (via Bash). Do NOT read any file under \
research/exit_dojo/reports/full_run/packets, .../truth, .../gate_state, or DATA/. Do NOT inspect \
raw parquet, feature stores, or label files. Your only window into the episode is the gate's output.
- Play ONE frame at a time: request a frame, decide, commit, repeat.
- Every price number is favorable-signed points from entry (entry = 0.00): positive = good for the \
position, negative = bad. HOLD = stay in the trade; EXIT = close it now. Your FIRST EXIT is binding \
and ends the episode.

LOOP (repeat until the gate says EPISODE CLOSED or NO MORE FRAMES):
1. Run:  python3.11 {gate} next --episode {eid}
2. Read the frame text. Note the printed `NONCE: <n>`.
3. Decide HOLD or EXIT using ONLY the frames you have seen so far. Then run:
     python3.11 {gate} commit --episode {eid} --decision HOLD --nonce <n> --reason "short reason"
   (use --decision EXIT to close the trade instead).
4. If you committed EXIT, or the gate says CLOSED / NO MORE FRAMES, stop looping.

FINISH: once the loop ends, run:
     python3.11 {gate} finish --episode {eid} --summary "2-3 sentences: what signature drove your \
exit (or why you never exited), and what you'd watch next time"

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
    # highest version dir wins (…/claude-code/<ver>/claude.exe)
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
    # SECURITY (reviewer fix 2026-07-17): NO --dangerously-skip-permissions.
    # The allowlist is scoped to the gate command ONLY — in non-interactive
    # mode every other tool call is denied, which also enforces the
    # no-raw-data-reads firewall by construction (a peek attempt is refused,
    # not merely forbidden by instruction).
    gate_rule = f'Bash(python3.11 {GATE_REL}:*)'
    cmd = [claude_bin, '-p', prompt, '--model', 'sonnet',
           '--allowedTools', gate_rule]
    if trace:
        # stream-json captures the agent's full tool-call trace to the log, making the
        # firewall (no raw-data reads; only gate commands) independently auditable. The
        # nonce chain remains the binding audit; this is belt-and-suspenders. Default OFF
        # keeps the proven text invocation; --trace is opt-in for fleet-scale audits.
        cmd += ['--output-format', 'stream-json', '--verbose']
    log_path = os.path.join(GATE_STATE_DIR, f'{eid}.agent_stdout.txt')
    os.makedirs(GATE_STATE_DIR, exist_ok=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
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
    ap.add_argument('--episodes', type=int, default=1, help='max episodes to run this invocation')
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--only', default=None, help='run one specific eid')
    ap.add_argument('--claude-bin', default=None)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--trace', action='store_true',
                    help='capture the agent tool-call trace (stream-json) for firewall audit')
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
