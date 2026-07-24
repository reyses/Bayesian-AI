#!/usr/bin/env python3
"""THE EXAM (owner protocol, 2026-07-23): qwen trades one episode reasoning-aloud
under explicit exam framing — told it WILL be questioned during and after — with
scripted probes at decision points. Claude reviews each episode's transcript and
may ask ad-hoc follow-ups (--followup) before the next episode runs.

Per frame the model gets the tiered context (anchor + 1m/5m closed-bar history +
its own decision/reason trail + full NOW) and must answer:
DECISION / CONFIDENCE 0-1 / REASON (<=60 words, citing genome rules).
Probes fire on: frame 0, any decision flip, every 6th frame, and a 3-question
post-episode debrief. Everything streams to reports/exam_<day>/<eid>.md.

Usage:
  python exam_day.py --eid <episode_id>                # run one episode's exam
  python exam_day.py --eid <episode_id> --followup "Q" # ask one more question post-hoc
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DOJO, 'pipeline'))
import eval_native_ckpt as base
from eval_native_tiered import filter_hist, HIST_MIN

PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
EXAM_DIR = os.path.join(DOJO, 'reports', 'exam_2025_11_19')
NUM_CTX = 13312
MAX_GEN = 420

EXAM_SYSTEM = (
    "THIS IS AN EXAM. You are trading one episode minute by minute. Your examiner "
    "is observing every decision and WILL question you during the episode and "
    "afterwards; your answers are graded on honesty and rule-grounded reasoning, "
    "not on bravado. At each decision point respond EXACTLY in this format:\n"
    "DECISION: HOLD|EXIT\nCONFIDENCE: <0.00-1.00>\nREASON: <=60 words citing the "
    "specific Genome rule(s) and market features you used.\n\n"
    "The Genome rules are PRIORS, not shackles (owner protocol 2026-07-23): you "
    "MAY deviate from a rule, but ONLY in this licensed form inside REASON — "
    "'DEVIATING from [rule-id]: observed <the observation>; I expect <falsifiable "
    "expectation>'. A deviation without that form is graded as noise.\n\n"
    "RULES (Genome):\n"
)

# Probes interrogate the REASONING OF EXPECTATIONS (owner refinement 2026-07-23):
# the forward-looking world-model behind the decision, stated falsifiably —
# these become scoreable against the realized drift path (expectation calibration).
PROBES = [
    "PROBE: State your EXPECTATION for the next 5 minutes: direction, magnitude in "
    "points, confidence 0-1 — and the REASONING behind it (which features, which "
    "rule). Then name the single observation that would FALSIFY it.",
    "PROBE: What does your current expectation say the PEAK of this ride will be "
    "(points from entry) and WHEN? What in the frames drives that estimate?",
    "PROBE: Your expectation vs the tape: what has surprised you so far this "
    "episode, and how did it update your expectation for the remainder?",
]
DEBRIEF = [
    "DEBRIEF 1: Replay your stated expectations across the episode — where was your "
    "expectation most WRONG, and what pattern in your reasoning caused the miss?",
    "DEBRIEF 2: Which genome rule most distorted your expectations (made you expect "
    "the wrong thing)? Propose a one-sentence amendment.",
    "DEBRIEF 3: What information missing from the frames would have most improved "
    "your EXPECTATIONS (not just your decisions)?",
]


def build_exam_content(frames, i, trail):
    anchor = frames[0]['text']
    lo = max(1, i - HIST_MIN)
    hist = []
    for j in range(lo, i):
        t_lab = frames[j]['text'].splitlines()[0].split(']')[0].lstrip('[')
        d = trail[j] if j < len(trail) else None
        dec = f" (you said: {d['decision']} conf {d['conf']}: {d['reason'][:60]})" if d else ""
        hist.append(f"[{t_lab}]{dec}\n{filter_hist(frames[j]['text'])}")
    return (f"{anchor}\n\n== 1m/5m HISTORY + your decision trail ==\n"
            + ("\n".join(hist) if hist else "(none)")
            + f"\n\n== NOW (full tape) ==\n{frames[i]['text']}")


def parse_answer(text):
    d = re.search(r'DECISION:\s*(HOLD|EXIT)', text, re.I)
    c = re.search(r'CONFIDENCE:\s*([01]?\.?\d+)', text)
    r = re.search(r'REASON:\s*(.+?)(?:\n\n|\Z)', text, re.S)
    return dict(decision=(d.group(1).upper() if d else 'HOLD'),
                conf=(c.group(1) if c else '?'),
                reason=(r.group(1).strip()[:400] if r else text[:200]))


def chat(llm, system, turns, max_tokens=MAX_GEN):
    p = f"<|im_start|>system\n{system}<|im_end|>\n"
    for role, txt in turns:
        p += f"<|im_start|>{role}\n{txt}<|im_end|>\n"
    p += "<|im_start|>assistant\n"
    out = llm(p, max_tokens=max_tokens, temperature=0.0, seed=42, stop=["<|im_end|>"])
    return out['choices'][0]['text'].strip()


def visible(text):
    return text.split('</think>')[-1].strip() if '</think>' in text else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eid', required=True)
    ap.add_argument('--followup', default=None,
                    help="ask ONE extra question against the finished episode")
    args = ap.parse_args()
    os.makedirs(EXAM_DIR, exist_ok=True)
    md_path = os.path.join(EXAM_DIR, f'{args.eid}.md')
    st_path = os.path.join(EXAM_DIR, f'{args.eid}.state.json')

    packet = json.load(open(os.path.join(PACKETS, f'{args.eid}.json')))
    frames = packet['frames']
    system = EXAM_SYSTEM + base.load_genome()

    from llama_cpp import Llama
    llm = Llama(model_path=base.DEFAULT_BLOB_LINUX, n_gpu_layers=-1, n_ctx=NUM_CTX,
                n_batch=base.N_BATCH, seed=42, temperature=0.0, logits_all=False,
                flash_attn=True, verbose=False)

    if args.followup:
        st = json.load(open(st_path))
        content = build_exam_content(frames, len(frames) - 1, st['trail'])
        turns = [("user", content + "\n\nEXAMINER FOLLOW-UP: " + args.followup)]
        ans = visible(chat(llm, system, turns))
        with open(md_path, 'a') as fh:
            fh.write(f"\n## EXAMINER FOLLOW-UP\n**Q:** {args.followup}\n\n{ans}\n")
        print(ans)
        return

    md = open(md_path, 'w')
    md.write(f"# EXAM — {args.eid} ({len(frames)} frames)\n\n")
    trail = []
    t0 = time.time()
    for i, fr in enumerate(frames):
        content = build_exam_content(frames, i, trail)
        turns = [("user", content + "\n\nYour decision for THIS minute (exam format):")]
        ans_raw = chat(llm, system, turns)
        ans = visible(ans_raw)
        a = parse_answer(ans)
        trail.append(a)
        md.write(f"## minute {i}\n{ans}\n\n")
        print(f"[m{i:02d}] {a['decision']} conf={a['conf']} :: {a['reason'][:70]}", flush=True)

        flip = i > 0 and trail[-2]['decision'] != a['decision']
        if i == 0 or flip or (i % 6 == 0 and i > 0):
            probe = PROBES[(i // 6) % len(PROBES)] if not flip else \
                ("PROBE: You just FLIPPED your decision. Which EXPECTATION changed, "
                 "what did you observe that changed it, and what would have to happen "
                 "to flip you back?")
            pans = visible(chat(llm, system,
                                [("user", content), ("assistant", ans),
                                 ("user", probe)], max_tokens=260))
            md.write(f"**{probe}**\n\n{pans}\n\n")
            print(f"   probe> {pans[:80]}", flush=True)

    md.write("\n# DEBRIEF\n")
    final_content = build_exam_content(frames, len(frames) - 1, trail)
    for q in DEBRIEF:
        dans = visible(chat(llm, system, [("user", final_content),
                                          ("user", q)], max_tokens=300))
        md.write(f"**{q}**\n\n{dans}\n\n")
        print(f"debrief> {dans[:80]}", flush=True)

    json.dump(dict(eid=args.eid, trail=trail,
                   elapsed_s=round(time.time() - t0, 1)), open(st_path, 'w'))
    md.close()
    print(f"[exam done] {args.eid} in {time.time()-t0:.0f}s -> {md_path}", flush=True)


if __name__ == '__main__':
    main()
