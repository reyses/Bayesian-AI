#!/usr/bin/env python3
"""Run qwen on the anticipation packets — judged ONLY on anticipated LEG DIRECTION
(owner 2026-07-27). For each sampled episode, serve the anticipation frame (early
= earliest run-up bar, or fire = at the fire) and ask qwen LONG/SHORT. Compare to
the true leg direction, with gov_dir and cubic-slope sign as baselines.

Writes reports/anticipate/results_<pick>.jsonl + a summary md. CUDA libs needed.
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pipeline'))
import eval_native_ckpt as base                      # noqa: E402

PKT = os.path.join(HERE, '..', 'reports', 'anticipate', 'packets')
OUTD = os.path.join(HERE, '..', 'reports', 'anticipate')
SYSTEM = ("You anticipate the DIRECTION a futures price leg will go, from a "
          "combiner (22 signal streams + pooled P + gov_dir), a cubic curve-"
          "regression (endpoint slope in pts/min = current velocity, curvature = "
          "bend), and price context. The combiner has NOT committed yet — call the "
          "leg EARLY. Reply EXACTLY two lines:\nDIR: LONG|SHORT\nWHY: <<=20 words>\n/no_think")
_GOV = re.compile(r'gov_dir=([+-]?\d)')
_SLP = re.compile(r'slope=([+-]?\d+(?:\.\d+)?)')
_DIR = re.compile(r'DIR:\s*(LONG|SHORT)', re.I)


def chat(llm, turns, max_tokens=256):
    p = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    for role, txt in turns:
        p += f"<|im_start|>{role}\n{txt}<|im_end|>\n"
    p += "<|im_start|>assistant\n"
    out = llm(p, max_tokens=max_tokens, temperature=0.0, seed=42, stop=["<|im_end|>"])
    return out['choices'][0]['text'].strip()


def parse_dir(text):
    t = text.split('</think>')[-1]
    m = _DIR.search(t)
    if m:
        return 1 if m.group(1).upper() == 'LONG' else -1
    if 'LONG' in t.upper() and 'SHORT' not in t.upper():
        return 1
    if 'SHORT' in t.upper() and 'LONG' not in t.upper():
        return -1
    return 0


def pick_frame(frames, how):
    return frames[0] if how == 'early' else frames[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--pick', choices=['early', 'fire'], default='early')
    ap.add_argument('--blob', default=base.DEFAULT_BLOB_LINUX)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(PKT, '*.json')))
    if not files:
        print('no packets'); return
    step = max(1, len(files) // args.n)
    sample = files[::step][:args.n]

    from llama_cpp import Llama
    print(f'loading qwen; scoring {len(sample)} episodes @ pick={args.pick}', flush=True)
    llm = Llama(model_path=args.blob, n_gpu_layers=-1, n_ctx=2048, seed=42, verbose=False)

    outp = os.path.join(OUTD, f'results_{args.pick}.jsonl')
    fh = open(outp, 'w')
    q_ok = g_ok = c_ok = n = 0
    for i, f in enumerate(sample):
        d = json.load(open(f))
        fr = pick_frame(d['frames'], args.pick)
        true = int(d['meta']['leg_dir_true'])
        gov = int(mg.group(1)) if (mg := _GOV.search(fr['text'])) else 0
        slp = float(ms.group(1)) if (ms := _SLP.search(fr['text'])) else 0.0
        cub = 1 if slp > 0 else (-1 if slp < 0 else 0)
        ans = chat(llm, [('user', fr['text'] + '\n\nAnticipate the leg direction now.')])
        qd = parse_dir(ans)
        n += 1
        q_ok += (qd == true); g_ok += (gov == true); c_ok += (cub == true)
        rec = dict(eid=d['episode_id'], k=fr['k'], true=true, qwen=qd, gov=gov,
                   cubic=cub, slope=slp, fire_P=d['meta']['fire_P'])
        fh.write(json.dumps(rec) + '\n'); fh.flush()
        if i < 3 or i % 25 == 0:
            print(f'  [{i+1}/{len(sample)}] {d["episode_id"]} k={fr["k"]} '
                  f'true={true} qwen={qd} gov={gov} cub={cub} | '
                  f'qwen {q_ok/n:.0%} gov {g_ok/n:.0%} cub {c_ok/n:.0%}', flush=True)
    fh.close()
    summ = (f'# Anticipate-combiner — qwen leg-direction ({args.pick} frame)\n'
            f'{n} episodes. Accuracy vs true leg direction:\n'
            f'- **qwen: {q_ok/n:.1%}**\n- gov_dir baseline: {g_ok/n:.1%}\n'
            f'- cubic-slope sign: {c_ok/n:.1%}\n'
            f'(GBM at-scale reference: 0.82-0.88 dir-AUC; combiner 73% right-side)\n')
    open(os.path.join(OUTD, f'summary_{args.pick}.md'), 'w').write(summ)
    print('\n' + summ, flush=True)


if __name__ == '__main__':
    main()
