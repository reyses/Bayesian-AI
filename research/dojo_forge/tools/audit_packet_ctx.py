#!/usr/bin/env python3
"""Offline packet ctx audit (doc 148 verify-then-stop follow-up, 2026-07-22).

Tokenizes every gen-0 packet's cumulative frame prompts with the qwen tokenizer
(vocab_only llama load — CPU, no GPU) and reports, per frame depth k:
  - max / p95 cumulative prompt tokens across all packets
  - the min num_ctx needed to serve frames 0..k clean
  - the KV-cache cost and est. GPU-layer offload at that ctx (RTX 3060 12GB)

Purpose: pick num_ctx from DATA + let the owner choose frame coverage vs speed.
Discovery that motivated it: packets carry 20 telescoping frames (~2.5k tok
each) — the full telescope is ~50k tokens, beyond qwen3's 40,960 window, so
"raise ctx until taint goes away" has no feasible endpoint. Frame coverage is
a spec decision.

Writes research/dojo_forge/reports/packet_ctx_audit.md and prints a summary.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(DOJO))
PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
GENOME = os.path.join(DOJO, 'genome', 'GENOME.md')
BLOB = "/media/moi/WindowsCode/ollama/models/blobs/sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e"
OUT = os.path.join(DOJO, 'reports', 'packet_ctx_audit.md')

# KV cost for qwen3:14b at f16 (measured: 1280MB @ 8192 -> ~160MB per 1k tokens)
KV_MB_PER_1K = 160
LAYER_MB = 205          # per offloaded block (8.6GB Q4 / 41 blocks)
FREE_MB = 11300         # typical free VRAM with lean desktop
MARGIN_MB = 1600        # guardrail margin


def main():
    from llama_cpp import Llama
    llm = Llama(model_path=BLOB, vocab_only=True, verbose=False)
    tok = lambda s: len(llm.tokenize(s.encode('utf-8'), add_bos=False))

    genome = open(GENOME).read() if os.path.exists(GENOME) else ''
    system = ("Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason."
              f"\n\nRULES (Genome):\n{genome}")
    sys_toks = tok(system)

    packs = sorted(glob.glob(os.path.join(PACKETS, '*.json')))
    depth_totals = {}          # k -> list of cumulative tokens at frame k
    for p in packs:
        d = json.load(open(p))
        cum = sys_toks
        for fr in d['frames']:
            cum += tok(fr['text'])
            k = fr['frame']
            depth_totals.setdefault(k, []).append(cum)

    lines = ["# Packet ctx audit (offline tokenizer, doc 148 follow-up)",
             f"packets={len(packs)}  system+genome={sys_toks} tokens",
             "",
             "| frames served | max tokens | p95 | min ctx (max+256) | KV MB | est GPU layers | note |",
             "|---|---|---|---|---|---|---|"]
    print(f"packets={len(packs)} sys_toks={sys_toks}")
    for k in sorted(depth_totals):
        v = sorted(depth_totals[k])
        mx, p95 = v[-1], v[int(len(v) * 0.95) - 1]
        need = mx + 256
        kv = need / 1000 * KV_MB_PER_1K
        layers = int((FREE_MB - kv - MARGIN_MB) / LAYER_MB)
        note = "> qwen 40960 max!" if need > 40960 else ""
        row = (f"| 0..{k} ({k+1}) | {mx} | {p95} | {need} | {kv:.0f} | "
               f"{min(layers, 41)}/41 | {note} |")
        lines.append(row)
        print(row)
    open(OUT, 'w').write("\n".join(lines) + "\n")
    print(f"\nwritten: {OUT}")


if __name__ == '__main__':
    main()
