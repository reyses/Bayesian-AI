#!/usr/bin/env python3
"""Frontier-model dojo runner — Claude (Fable 5) through the SAME tiered
curriculum as the qwen teacher (owner directive 2026-07-24: burn remaining
weekly plan quota on a frontier baseline before it resets).

Comparability contract:
  - Context: IDENTICAL to eval_native_tiered (imports build_user_content —
    pinned anchor + 20-min 1m/5m closed-bar history + full NOW). No drift.
  - System prompt: same genome system prompt as the gate runs.
  - Stateless per frame like the harness: one headless `claude -p` call per
    frame with the full tiered prompt. NO conversation carry (a resumed
    conversation would add hidden state the qwen harness doesn't have).
  - Readout: generation-based DECISION parse (no logits over the CLI).
    TRUNCATED/unparseable is recorded loudly, NEVER silently HOLD (exam v3).
  - Output schema mirrors the tiered ckpt (episode_id, exit_frame, frames[])
    so score_tiered_effectiveness / gen1_early_look style tooling reads it.

Run:  python research/dojo_forge/pipeline/eval_claude_dojo.py \
        --limit 2 --frames-limit 3          # smoke
      python ... --days 2025_04_08 [--model claude-fable-5]
Requires: ~/.local/bin/claude (stable wrapper). Resume-safe via ckpt jsonl.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO = os.path.join(HERE, '..')
sys.path.insert(0, HERE)
import eval_native_ckpt as base            # noqa: E402  genome + ckpt helpers
from eval_native_tiered import build_user_content  # noqa: E402  THE context

CLAUDE_BIN = os.path.expanduser("~/.local/bin/claude")
DEFAULT_MODEL = "claude-fable-5"
PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
CKPT_DEFAULT = os.path.join(DOJO, 'gate_state', 'claude_dojo.jsonl')
PER_CALL_TIMEOUT_S = 240

ANSWER_SPEC = (
    "\n\nAnswer in EXACTLY this format (no preamble):\n"
    "DECISION: HOLD or EXIT\nCONFIDENCE: 0-100\nREASON: <one line, cite the "
    "rule (e.g. G1.8) or observation driving the decision>")

DEC_RE = re.compile(r"DECISION:\s*(HOLD|EXIT)", re.IGNORECASE)


NEUTRAL_CWD = "/tmp/claude-dojo-neutral"   # NOT the repo: running from the repo
# loads CLAUDE.md + project memory into the subject — which contain our
# CONCLUSIONS (never-bail, ride-only). That is the answer key leaking into the
# exam. The subject must see genome + frame ONLY, exactly like qwen.


def ask_claude(prompt, model):
    os.makedirs(NEUTRAL_CWD, exist_ok=True)
    try:
        r = subprocess.run(
            [CLAUDE_BIN, "-p", prompt, "--model", model,
             "--output-format", "json"],
            capture_output=True, text=True, timeout=PER_CALL_TIMEOUT_S,
            cwd=NEUTRAL_CWD)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0:
        return None, f"rc={r.returncode}:{(r.stderr or '')[:120]}"
    try:
        return json.loads(r.stdout).get("result", ""), None
    except json.JSONDecodeError:
        return None, "bad_json"


def eval_episode(eid, packet, system_prompt, model, frames_limit=None):
    frames = packet.get('frames', [])
    if frames_limit:
        frames = frames[:frames_limit]
    rec_frames, decisions = [], []
    exit_frame = None
    hard_fails = 0
    t0 = time.time()
    for i in range(len(frames)):
        content = frames[i]['text'] if i == 0 else \
            build_user_content(frames, i, decisions)
        prompt = f"{system_prompt}\n\n{content}{ANSWER_SPEC}"
        text, err = ask_claude(prompt, model)
        if err or not text:
            rec_frames.append(dict(frame_idx=i, decision=None, confidence=None,
                                   reason=None, hard_fail=err or "empty"))
            decisions.append(None)
            hard_fails += 1
            continue
        m = DEC_RE.search(text)
        if not m:
            rec_frames.append(dict(frame_idx=i, decision=None, confidence=None,
                                   reason=text[:200], hard_fail="unparseable"))
            decisions.append(None)
            hard_fails += 1
            continue
        decision = m.group(1).upper()
        conf = re.search(r"CONFIDENCE:\s*(\d+)", text)
        reason = re.search(r"REASON:\s*(.+)", text)
        decisions.append(decision)
        rec_frames.append(dict(
            frame_idx=i, decision=decision,
            confidence=int(conf.group(1)) if conf else None,
            reason=(reason.group(1).strip()[:300] if reason else None)))
        if decision == "EXIT" and exit_frame is None:
            exit_frame = i
        print(f"    f{i:02d}: {decision}"
              + (f" ({rec_frames[-1]['reason'][:60]})" if rec_frames[-1]['reason'] else ""),
              flush=True)
    return dict(
        episode_id=eid, engine='claude_cli', model=model,
        readout='generation_v1+tiered_w20', tainted=hard_fails > 0,
        taint_reason=(f"hard_fails:{hard_fails}" if hard_fails else None),
        exit_frame=exit_frame, n_frames_evaluated=len(rec_frames),
        elapsed_s=round(time.time() - t0, 3), ts=time.time(),
        frames=rec_frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', help="comma-separated day filter (e.g. 2025_04_08)")
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--limit', type=int, default=None, help="max NEW episodes")
    ap.add_argument('--frames-limit', type=int, default=None,
                    help="smoke-test: only first N frames per episode")
    ap.add_argument('--ckpt', default=CKPT_DEFAULT)
    args = ap.parse_args()

    if not os.path.exists(CLAUDE_BIN):
        sys.exit(f"claude wrapper missing at {CLAUDE_BIN}")
    system_prompt = (f"Decide to HOLD or EXIT based on the frame. If EXIT, "
                     f"provide a reason.\n\nRULES (Genome):\n{base.load_genome()}")
    files = sorted(glob.glob(os.path.join(PACKETS, "*.json")))
    if args.days:
        want = set(args.days.split(','))
        files = [f for f in files
                 if any(os.path.basename(f).startswith(d) for d in want)]
    completed = base.load_completed(args.ckpt)
    todo = [(os.path.basename(f).replace('.json', ''), f) for f in files]
    todo = [(e, f) for e, f in todo if e not in completed]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[plan] CLAUDE-DOJO model={args.model} episodes={len(todo)} "
          f"frames_limit={args.frames_limit} (resume: {len(completed)} done)",
          flush=True)
    for k, (eid, path) in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {eid}", flush=True)
        rec = eval_episode(eid, json.load(open(path)), system_prompt,
                           args.model, args.frames_limit)
        base.append_checkpoint(args.ckpt, rec)
        print(f"    -> exit_frame={rec['exit_frame']} "
              f"taint={rec['taint_reason']} {rec['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
