#!/usr/bin/env python3
"""Interrogate the teacher: WHY did it exit early? (owner ask, 2026-07-23)

Replays the EXACT tiered context of chosen (episode, frame) pairs — same anchor +
20-min 1m/5m closed-bar history + full NOW frame, same seed/temp — but with the
reasoning bypass REMOVED: the model generates its <think> chain + a final
explanation citing which genome rule / market feature drove the EXIT.

Honesty caveat (printed into the report): post-hoc explanations are plausible
stories, not guaranteed causes of the logit — treat as hypothesis generators for
genome mutation design, cross-checked against the drift data.

Usage: python ask_teacher_why.py [--targets N] [--max-tokens 500]
Writes research/dojo_forge/reports/teacher_why_2026-07-23.md
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DOJO, 'pipeline'))
import eval_native_ckpt as base
from eval_native_tiered import build_user_content

CKPT = os.path.join(DOJO, 'gate_state', 'acceptance_results_tiered.jsonl')
TRUTH = os.path.join(DOJO, 'reports', 'gen0', 'truth')
PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
OUT = os.path.join(DOJO, 'reports', 'teacher_why_2026-07-23.md')


def pick_targets(eps, n):
    """Premature exits ranked by points left on the table."""
    targets = []
    for eid, rec in eps.items():
        ef = rec.get('exit_frame')
        if ef is None:
            continue
        t = json.load(open(os.path.join(TRUTH, f'{eid}.json')))
        om, drift = t['oracle_minute'], t['per_minute_forward_drift']
        if om - ef >= 5:
            cap = drift[ef] if ef < len(drift) else drift[-1]
            targets.append((t['oracle_capture'] - cap, eid, ef, om,
                            rec['frames'][ef]['p_exit']))
    targets.sort(reverse=True)
    return targets[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', type=int, default=10)
    ap.add_argument('--max-tokens', type=int, default=500)
    ap.add_argument('--num-ctx', type=int, default=13312)   # +1k for generation room
    args = ap.parse_args()

    eps = {}
    with open(CKPT) as fh:
        for line in fh:
            r = json.loads(line)
            eps[r['episode_id']] = r
    targets = pick_targets(eps, args.targets)
    print(f"interrogating {len(targets)} premature exits", flush=True)

    system_prompt = (f"Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason."
                     f"\n\nRULES (Genome):\n{base.load_genome()}")

    from llama_cpp import Llama
    blob = base.DEFAULT_BLOB_LINUX
    llm = Llama(model_path=blob, n_gpu_layers=-1, n_ctx=args.num_ctx,
                n_batch=base.N_BATCH, seed=42, temperature=0.0,
                logits_all=False, flash_attn=True, verbose=False)

    lines = ["# Teacher interrogation — why the premature exits? (2026-07-23)",
             "Replayed the EXACT tiered context of the worst premature exits with the",
             "reasoning bypass removed. CAVEAT: post-hoc explanations are plausible",
             "stories, not guaranteed causes — hypothesis generators for mutation design.",
             ""]
    for k, (left, eid, ef, om, p) in enumerate(targets, 1):
        packet = json.load(open(os.path.join(PACKETS, f'{eid}.json')))
        frames = packet['frames']
        decisions = [f.get('decision') for f in eps[eid]['frames']]
        content = frames[ef]['text'] if ef == 0 else build_user_content(frames, ef, decisions)
        ask = (content + "\n\nYou are at this decision point. State your decision "
               "(HOLD or EXIT) and then explain in <=120 words: which specific "
               "Genome rule(s) and which specific market features in the frames "
               "drove it. Quote the rule you applied.")
        prompt = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                  f"<|im_start|>user\n{ask}<|im_end|>\n<|im_start|>assistant\n")
        out = llm(prompt, max_tokens=args.max_tokens, temperature=0.0, seed=42,
                  stop=["<|im_end|>"])
        text = out['choices'][0]['text'].strip()
        # strip <think> block for the report but keep a one-line note of its gist
        vis = text.split('</think>')[-1].strip() if '</think>' in text else text
        lines += [f"## {k}. {eid} — exited m{ef} (p_exit={p:.3f}), oracle m{om}, "
                  f"left {left:.0f} pts",
                  vis, ""]
        print(f"[{k}/{len(targets)}] {eid} m{ef}: {vis[:100]!r}", flush=True)

    open(OUT, 'w').write("\n".join(lines) + "\n")
    print(f"written: {OUT}")


if __name__ == '__main__':
    main()
