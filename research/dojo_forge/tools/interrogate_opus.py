#!/usr/bin/env python3
"""ADAPTIVE INTERROGATION — Opus questions, qwen answers (owner 2026-07-24:
"you can also use opus for interrogation"). Unlike interview_memo.py's fixed
probes, Opus reads each answer and crafts the next question live, hunting the
actual confusion. Use in the session stage when fixed probes are ambiguous.

Loop (max --rounds): Opus gets the goal + transcript -> asks ONE question ->
qwen (reasoning enabled, exam chat machinery) answers -> repeat. Opus closes
with a VERDICT: UNDERSTANDS / MISUNDERSTANDS: <what> / CANNOT: <what>, plus
its recommended overlay change.

usage: interrogate_opus.py --goal "why does it write mottos instead of
       data-bearing memos" [--rounds 5] [--run-tag sprintN] [--out-tag t1]
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(DOJO, 'pipeline'))
sys.path.insert(0, HERE)
import eval_native_ckpt as base            # noqa: E402
from exam_day import chat, visible         # noqa: E402

CLAUDE = os.path.expanduser('~/.local/bin/claude')
OPUS = 'claude-opus-4-8'
QWEN_SYSTEM = ("You are the trading teacher being interviewed about your own "
               "journal MEMOs. Answer honestly and concretely. Keep <think> "
               "under 150 words.")


def ask_opus(prompt):
    r = subprocess.run([CLAUDE, '-p', prompt, '--model', OPUS,
                        '--output-format', 'json'],
                       capture_output=True, text=True, timeout=180)
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith('{'):
            return json.loads(line).get('result', '').strip()
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal', required=True)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--db', default=os.path.join(DOJO, 'gate_state',
                                                 'teacher_memory_v2.db'))
    ap.add_argument('--run-tag', default=None)
    ap.add_argument('--num-ctx', type=int, default=16384)
    ap.add_argument('--out-tag', default='latest')
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    q = ("SELECT text FROM memos WHERE created_run=? ORDER BY id DESC LIMIT 5"
         if args.run_tag else "SELECT text FROM memos ORDER BY id DESC LIMIT 5")
    memos = [r[0] for r in con.execute(
        q, (args.run_tag,) if args.run_tag else ())]

    from llama_cpp import Llama
    n_layers = base.preflight_vram(-1, args.num_ctx)
    llm = Llama(model_path=base.DEFAULT_BLOB_LINUX, n_gpu_layers=n_layers,
                n_ctx=args.num_ctx, n_batch=base.N_BATCH, seed=42,
                temperature=0.0, logits_all=False, flash_attn=True,
                verbose=False)

    transcript = []
    memos_block = "\n".join(f"- {m[:160]}" for m in memos) or "(none)"
    for rnd in range(1, args.rounds + 1):
        tlog = "\n".join(f"{w}: {t}" for w, t in transcript) or "(none yet)"
        opus_prompt = (
            f"You are interrogating a 14B trading-teacher LLM about a failure. "
            f"GOAL: {args.goal}\nIts recent journal memos:\n{memos_block}\n"
            f"Interrogation so far:\n{tlog}\n\n"
            f"Round {rnd}/{args.rounds}. If you have enough evidence, reply "
            f"EXACTLY:\nVERDICT: UNDERSTANDS|MISUNDERSTANDS: <what>|CANNOT: "
            f"<what>\nFIX: <one concrete overlay/task-structure change>\n"
            f"Otherwise reply with ONE question to ask it next (question only, "
            f"no preamble).")
        move = ask_opus(opus_prompt)
        if not move:
            transcript.append(("OPUS", "(no reply — API failure)"))
            break
        if move.upper().startswith("VERDICT"):
            transcript.append(("OPUS", move))
            break
        transcript.append(("OPUS-Q", move))
        ans = visible(chat(llm, QWEN_SYSTEM, [('user', move)]))
        transcript.append(("QWEN-A", ans))
        print(f"[r{rnd}] Q: {move[:100]}\n     A: {ans[:140]}", flush=True)

    out = os.path.join(DOJO, 'reports',
                       f'interrogation_opus_{args.out_tag}.md')
    with open(out, 'w') as f:
        f.write(f"# Opus interrogation — goal: {args.goal}\n\n")
        for w, t in transcript:
            f.write(f"**{w}:** {t}\n\n")
    print(f"[saved] {out}")


if __name__ == '__main__':
    main()
