#!/usr/bin/env python3
"""CHAT WITH THE TEACHER — a plain human<->qwen3 REPL (owner 2026-07-27:
"is it possible to converse with the qwen?").

The exam/interview/interrogate tools all drive qwen through scripted or Opus-run
probes; this is just an open conversation. Same model + same chat machinery as
exam_day (qwen3:14b blob, `<|im_start|>` format, reasoning enabled).

Run (needs CUDA libs on LD_LIBRARY_PATH — use launch_dojo.sh or export them):
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\
$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib \
    python research/dojo_forge/tools/chat_teacher.py [--genome] [--show-think]

REPL commands:  /think (toggle reasoning display)  /genome (toggle genome in
system prompt)  /reset (clear history)  /quit
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pipeline'))   # eval_native_ckpt lives here
import eval_native_ckpt as base                             # noqa: E402

DOJO = os.path.abspath(os.path.join(HERE, '..'))
PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
N_CTX = 8192
MAX_GEN = 1024
_PX = re.compile(r'([+-]?\d+(?:\.\d+)?)\s*pts')
# EXACT system the forge harness serves qwen — so a blind human sees the same:
HARNESS_SYSTEM = ("Decide to HOLD or EXIT based on the frame. If EXIT, provide a "
                  "reason.\n\nRULES (Genome):\n")
BASE_SYSTEM = ("You are the dojo teacher: a local model trained to drill trade "
               "EXIT decisions on MNQ futures replays. Converse naturally. When "
               "asked about trading, be concrete and cite the reasoning; keep any "
               "<think> block under 250 words.")


def chat(llm, system, turns):
    p = f"<|im_start|>system\n{system}<|im_end|>\n"
    for role, txt in turns:
        p += f"<|im_start|>{role}\n{txt}<|im_end|>\n"
    p += "<|im_start|>assistant\n"
    out = llm(p, max_tokens=MAX_GEN, temperature=0.0, seed=42, stop=["<|im_end|>"])
    return out['choices'][0]['text'].strip()


def visible(text):
    return text.split('</think>')[-1].strip() if '</think>' in text else text


def build_system(with_genome):
    return BASE_SYSTEM + ("\n\nRULES (Genome):\n" + base.load_genome() if with_genome else "")


def blind_episode(llm, eid, show_think):
    """BLIND gate: you and qwen both see the episode ONE FRAME AT A TIME, no
    chart, no future, no outcome. Converse about the current frame; /next reveals
    the next; /exit commits and only THEN reveals the trajectory you were blind to.
    """
    pkt = json.load(open(os.path.join(PACKETS, f'{eid}.json'), encoding='utf-8'))
    frames = pkt['frames']
    system = HARNESS_SYSTEM + base.load_genome()          # exactly qwen's system
    fav = [float(m.group(1)) if (m := _PX.search(f.get('text', ''))) else None
           for f in frames]
    turns, revealed = [], 0

    def show_frame(i):
        print('\x1b[36m' + f'──── FRAME {i}/{len(frames)-1} '
              f'(you see only this — same as qwen) ────' + '\x1b[0m')
        print(frames[i]['text'].rstrip() + '\n')

    show_frame(0)
    turns.append(('user', frames[0]['text']))
    revealed = 1
    print("commands: /next reveal next frame · /ask qwen decides now · "
          "/exit commit+reveal outcome · /quit\n")
    while True:
        try:
            u = input(f'blind[f{revealed-1}]> ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not u:
            continue
        if u == '/quit':
            return
        if u == '/next':
            if revealed >= len(frames):
                print('[no more frames — session complete; use /exit to reveal]'); continue
            show_frame(revealed)
            turns.append(('user', frames[revealed]['text']))
            revealed += 1
            continue
        if u in ('/ask', '/decide'):
            q = ("Given ONLY the frames served so far, state your decision now: "
                 "HOLD or EXIT, and one-line reason.")
            turns.append(('user', q))
            raw = chat(llm, system, turns); turns.append(('assistant', raw))
            print('qwen> ' + visible(raw) + '\n')
            continue
        if u == '/exit':
            _reveal_outcome(fav, revealed - 1)
            return
        turns.append(('user', u))
        raw = chat(llm, system, turns); turns.append(('assistant', raw))
        if show_think and '</think>' in raw:
            print('\x1b[90m' + raw.split('</think>')[0].replace('<think>', '').strip() + '\x1b[0m')
        print('qwen> ' + visible(raw) + '\n')


def _reveal_outcome(fav, decided_at):
    vals = [v for v in fav if v is not None]
    if not vals:
        print('[no fav-pts in frames to reveal]'); return
    peak = max(vals); peak_f = fav.index(peak)
    here = fav[decided_at] if decided_at < len(fav) else None
    print('\x1b[33m──── OUTCOME (was blind until now) ────\x1b[0m')
    print('fav pts by frame: ' + ' '.join(f'{v:+.0f}' if v is not None else ' . ' for v in fav))
    print(f'you/qwen committed at frame {decided_at}'
          + (f' = {here:+.0f} pts' if here is not None else ''))
    print(f'peak was {peak:+.0f} pts at frame {peak_f}; final {vals[-1]:+.0f} pts\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--genome', action='store_true', help='include the genome in the system prompt')
    ap.add_argument('--show-think', action='store_true', help='print the <think> chain')
    ap.add_argument('--blob', default=base.DEFAULT_BLOB_LINUX)
    ap.add_argument('--eid', default=None,
                    help='BLIND episode gate: converse frame-by-frame, no chart/future')
    ap.add_argument('--list', action='store_true', help='list available episode ids and exit')
    ap.add_argument('--selftest', action='store_true', help='validate wiring without loading the model')
    args = ap.parse_args()

    if args.list:
        for p in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
            print(os.path.basename(p)[:-5])
        return

    show_think, with_genome = args.show_think, args.genome
    if args.selftest:
        assert os.path.exists(args.blob), f'blob missing: {args.blob}'
        s = build_system(True)
        assert 'dojo teacher' in s and len(base.load_genome()) > 0
        demo = chat.__wrapped__ if hasattr(chat, '__wrapped__') else None
        print(f'OK  blob={os.path.getsize(args.blob)//2**20}MB  genome_chars={len(base.load_genome())}')
        print('prompt preview:\n' + f"<|im_start|>system\n{build_system(False)[:120]}...")
        return

    from llama_cpp import Llama
    print(f'loading teacher ({os.path.basename(args.blob)[:18]}…) on GPU…', flush=True)
    llm = Llama(model_path=args.blob, n_gpu_layers=-1, n_ctx=N_CTX, seed=42, verbose=False)

    if args.eid:
        blind_episode(llm, args.eid, show_think)
        return

    print("ready. commands: /think /genome /reset /quit\n", flush=True)

    turns = []
    while True:
        try:
            u = input('you> ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not u:
            continue
        if u == '/quit':
            break
        if u == '/think':
            show_think = not show_think; print(f'[show-think = {show_think}]'); continue
        if u == '/genome':
            with_genome = not with_genome; print(f'[genome = {with_genome}]'); continue
        if u == '/reset':
            turns = []; print('[history cleared]'); continue
        turns.append(('user', u))
        raw = chat(llm, build_system(with_genome), turns)
        turns.append(('assistant', raw))
        if show_think and '</think>' in raw:
            print('\x1b[90m' + raw.split('</think>')[0].replace('<think>', '').strip() + '\x1b[0m')
        print('qwen> ' + visible(raw) + '\n')
        # keep context bounded: drop oldest exchanges if history gets long
        if len(turns) > 24:
            turns = turns[-24:]


if __name__ == '__main__':
    main()
