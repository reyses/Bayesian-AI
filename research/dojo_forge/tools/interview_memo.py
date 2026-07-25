#!/usr/bin/env python3
"""MEMO-COMPREHENSION INTERVIEW (owner 2026-07-24: "we also have the exam tool
and the interview to understand the reasoning — it could be it simply does not
understand"). Run when sprint retros fail: distinguishes CANNOT from
MISUNDERSTANDS before another blind prompt rewrite.

Reuses exam_day's chat machinery (reasoning ENABLED). Four probes:
  P1 concept   — "what makes a trading memo useful to your future self?"
  P2 critique  — show one of ITS OWN motto memos; ask what's wrong with it
  P3 contrast  — motto vs the #9-style memo; which is better and why
  P4 rewrite   — give it a REAL frame + its motto; demand a rewritten memo
                 with a concrete magnitude from that frame
Reading the transcript: if P1-P3 are correct but P4 fails, it UNDERSTANDS but
cannot execute under the decision load (-> restructure task, e.g., separate
memo pass). If P1-P3 fail, it does not understand (-> teach the concept in the
overlay with worked examples, or the capability is absent at this scale).

usage: interview_memo.py [--db PATH] [--run-tag TAG] [--num-ctx 16384]
Writes research/dojo_forge/reports/interview_memo_<ts>.md (ts passed via
--out-tag to avoid Date.now in workflows; default 'latest').
"""
import argparse
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(DOJO, 'pipeline'))
sys.path.insert(0, HERE)
import eval_native_ckpt as base            # noqa: E402
from exam_day import chat, visible         # noqa: E402

GOOD_EXAMPLE = ('reversion_prob_30 split 0.80(1m)/0.97(5m) during gb>40% '
                'resolved as continuation - multi-TF split favored holding')

SYSTEM = ("You are a trading teacher's coach reviewing the teacher's own "
          "journal. Answer directly and concretely. Keep <think> under 150 "
          "words.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.path.join(DOJO, 'gate_state',
                                                 'teacher_memory_v2.db'))
    ap.add_argument('--run-tag', default=None)
    ap.add_argument('--num-ctx', type=int, default=16384)
    ap.add_argument('--out-tag', default='latest')
    ap.add_argument('--frame-file', default=None,
                    help="optional: a file holding one frame's NOW text for P4")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    if args.run_tag:
        rows = con.execute("SELECT text FROM memos WHERE created_run=? "
                           "ORDER BY id DESC LIMIT 3", (args.run_tag,)).fetchall()
    else:
        rows = con.execute("SELECT text FROM memos ORDER BY id DESC LIMIT 3").fetchall()
    own_memos = [r[0] for r in rows] or ["(no memos in bank)"]

    frame_txt = ""
    if args.frame_file and os.path.exists(args.frame_file):
        frame_txt = open(args.frame_file, encoding='utf-8').read()[:6000]

    probes = [
        ("P1_concept",
         "You keep a journal of trading MEMOs for your future self. In 3 "
         "sentences: what properties make a memo USEFUL when you read it days "
         "later on a different trading day?"),
        ("P2_critique",
         f"Here is a memo you wrote: \"{own_memos[0]}\". In 2-3 sentences: "
         "what is wrong with it as a note to your future self?"),
        ("P3_contrast",
         f"Memo A: \"{own_memos[0]}\"\nMemo B: \"{GOOD_EXAMPLE}\"\n"
         "Which is more useful to retrieve on a future day, and WHY, in 2-3 "
         "sentences?"),
        ("P4_rewrite",
         (f"Market frame:\n{frame_txt}\n\n" if frame_txt else "")
         + f"Rewrite this memo so it contains at least one concrete magnitude "
           f"(a feature value, signed-points level, or duration) "
           f"{'from the frame above ' if frame_txt else 'from your recent trading '}"
           f"and what it resolved into: \"{own_memos[0]}\""),
    ]

    from llama_cpp import Llama
    n_layers = base.preflight_vram(-1, args.num_ctx)
    llm = Llama(model_path=base.DEFAULT_BLOB_LINUX, n_gpu_layers=n_layers,
                n_ctx=args.num_ctx, n_batch=base.N_BATCH, seed=42,
                temperature=0.0, logits_all=False, flash_attn=True,
                verbose=False)
    out_lines = [f"# memo-comprehension interview ({args.out_tag})",
                 f"own memos sampled: {len(own_memos)}", ""]
    for tag, q in probes:
        ans = visible(chat(llm, SYSTEM, [('user', q)]))
        out_lines += [f"## {tag}", f"Q: {q[:200]}", f"A: {ans}", ""]
        print(f"[{tag}] {ans[:160]}", flush=True)
    out = os.path.join(DOJO, 'reports', f'interview_memo_{args.out_tag}.md')
    with open(out, 'w') as f:
        f.write("\n".join(out_lines))
    print(f"[saved] {out}")


if __name__ == '__main__':
    main()
