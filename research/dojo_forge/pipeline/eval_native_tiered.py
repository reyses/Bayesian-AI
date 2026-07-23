#!/usr/bin/env python3
"""Tiered-window native acceptance eval (doc 149, Moises' design 2026-07-22).

Serves ALL frames of every packet (incl. deepest late-ride) in a BOUNDED context
by rebuilding the prompt per frame with tiered history decay:

  [ANCHOR]   frame-0 wide field (all 8 TFs incl. 1h/4h/1D + entry context) — pinned
  [HISTORY]  last HIST_MIN minutes: only the [1m]/[5m] lines (decision context)
             + the decision trail ("minute k: HOLD/EXIT")
  [NOW]      the current minute's full tape incl. 5s/15s (immediate action)

Rationale (owner, via Telegram 2026-07-22): "sub-minute is for taking immediate
action, above-minute is for decisions; drop sub-minute after 1 min — it's
telescopic both ways, the data is kept in the higher TFs."
Measured plateau: ~9.8k tokens at ANY episode depth (20 or 61 min) — vs the
unbounded telescope's 148k max. Fits num_ctx 12288 with headroom.

Reuses eval_native_ckpt.py machinery (loader, logit reader, selftest, ckpt IO,
VRAM guardrail). Per-frame REBUILD (llm.reset + full prefill ~9k tokens): slower
per frame than KV-append, but the only correct way to implement decay.

Usage:
  python eval_native_tiered.py --engine cuda [--num-ctx 12288] [--limit N]
      [--ckpt PATH] [--csv PATH]
"""
import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_native_ckpt as base           # the census harness = shared machinery

DOJO = os.path.dirname(HERE)
DEFAULT_CKPT = os.path.join(DOJO, 'gate_state', 'acceptance_results_tiered.jsonl')
DEFAULT_CSV = os.path.join(DOJO, 'reports', 'acceptance_native_tiered.csv')

# --- tiered-window spec (doc 149; owner-ratified constants) ------------------
HIST_MIN = 20            # minutes of [1m]/[5m]-resolution history retained
KEEP_TFS = ('1m', '5m')  # line tags kept for historical minutes
_TF_LINE = re.compile(r'\[(\d+(?:s|m|h)|1D)\]')


def filter_hist(text):
    """Keep only the [1m]/[5m] CLOSED-BAR lines of a historical minute-frame.

    v2 (post gate-fail 2026-07-22): the indicator-dump lines tokenize at ~1.65
    chars/token (dense numerics) and blew the budget (23k real tokens). The
    owner's ratified intent is "the resolve of 1m, 20 bars" — the BARS. Past
    indicator dumps are derived/rolling values whose current state is visible
    in the NOW frame; dropping them from history loses no bar data."""
    out = []
    for ln in text.splitlines():
        m = _TF_LINE.search(ln)
        if m and m.group(1) in KEEP_TFS and 'closed-bar' in ln:
            out.append(ln)
    return "\n".join(out)


def build_user_content(frames, i, decisions):
    """Tiered context for frame i (i>=1). Frame 0 is served as-is (it IS the anchor)."""
    anchor = frames[0]['text']
    lo = max(1, i - HIST_MIN)
    hist_parts = []
    for j in range(lo, i):
        t_lab = frames[j]['text'].splitlines()[0].split(']')[0].lstrip('[')  # e.g. t=7m
        fl = filter_hist(frames[j]['text'])
        dec = decisions[j] if j < len(decisions) and decisions[j] else "HOLD"
        hist_parts.append(f"[{t_lab}] (1m/5m view; decision then: {dec})\n{fl}")
    hist = "\n".join(hist_parts) if hist_parts else "(none)"
    now = frames[i]['text']
    return (f"{anchor}\n\n== 1m/5m HISTORY (last {min(i-1, HIST_MIN)} min) ==\n{hist}"
            f"\n\n== NOW (full tape) ==\n{now}")


def eval_episode_tiered(llm, reader, id_exit, id_hold, eid, packet, system_prompt,
                        engine, model_name, num_ctx):
    frames = packet.get('frames', [])
    rec_frames = []
    decisions = []
    tainted = False
    taint_reason = None
    exit_frame = None
    t0 = time.time()

    for i, frame in enumerate(frames):
        content = frame['text'] if i == 0 else build_user_content(frames, i, decisions)
        seg = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
               f"<|im_start|>user\n{content}<|im_end|>\n{base.THINK_SUFFIX}")
        llm.reset()                                    # per-frame rebuild (decay)
        toks = llm.tokenize(seg.encode('utf-8'), add_bos=True, special=True)
        if len(toks) >= num_ctx:
            tainted = True
            taint_reason = f"ctx_overflow:{len(toks)}>={num_ctx}"
            rec_frames.append(dict(frame_idx=i, p_exit=None, logit_exit=None,
                                   logit_hold=None, lp_exit=None, lp_hold=None,
                                   prompt_tokens=len(toks), decision=None,
                                   hard_fail="ctx"))
            decisions.append(None)
            continue                                   # later frames may still fit
        llm.eval(toks)
        logits = reader()
        lp_e, lp_h, p_exit = base.logsoftmax_two(logits, id_exit, id_hold)
        decision = "EXIT" if p_exit > 0.5 else "HOLD"
        decisions.append(decision)
        rec_frames.append(dict(frame_idx=i, p_exit=round(p_exit, 6),
                               logit_exit=round(lp_e, 6), logit_hold=round(lp_h, 6),
                               lp_exit=round(lp_e, 6), lp_hold=round(lp_h, 6),
                               prompt_tokens=len(toks), decision=decision))
        if decision == "EXIT" and exit_frame is None:
            exit_frame = i

    return dict(
        episode_id=eid, engine=engine, model=model_name, num_ctx=num_ctx,
        readout=base.READOUT + "+tiered_w20", tainted=tainted,
        taint_reason=taint_reason, exit_frame=exit_frame,
        n_frames_evaluated=len(rec_frames),
        elapsed_s=round(time.time() - t0, 3), ts=time.time(), frames=rec_frames,
    )


def main():
    ap = argparse.ArgumentParser(description="Tiered-window acceptance eval (doc 149)")
    ap.add_argument('--engine', choices=['cpu', 'cuda'], required=True)
    ap.add_argument('--model-blob', default=None)
    ap.add_argument('--n-gpu-layers', type=int, default=None)
    ap.add_argument('--num-ctx', type=int, default=12288)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--packets-dir', default=base.PACKETS_DIR)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    base.NUM_CTX = args.num_ctx                     # keep base guardrail math honest
    num_ctx = args.num_ctx
    if args.model_blob:
        model_blob = args.model_blob
    elif os.path.exists(base.DEFAULT_BLOB_LINUX):
        model_blob = base.DEFAULT_BLOB_LINUX
    else:
        model_blob = base.DEFAULT_BLOB_WSL
    model_name = os.path.basename(model_blob)
    n_gpu_layers = args.n_gpu_layers if args.n_gpu_layers is not None \
        else (-1 if args.engine == 'cuda' else 0)

    packet_files = sorted(glob.glob(os.path.join(args.packets_dir, "*.json")))
    if not packet_files:
        print(f"No packet files in {args.packets_dir}", file=sys.stderr)
        sys.exit(1)

    completed = base.load_completed(args.ckpt)
    base.rebuild_csv(args.csv, list(completed.values()))
    print(f"[resume] {len(completed)} of {len(packet_files)} episodes already done", flush=True)

    system_prompt = (f"Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason."
                     f"\n\nRULES (Genome):\n{base.load_genome()}")

    todo = [(os.path.basename(p).replace('.json', ''), p) for p in packet_files]
    todo = [(eid, p) for eid, p in todo if eid not in completed]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"[plan] TIERED w={HIST_MIN}m engine={args.engine} num_ctx={num_ctx} "
          f"model={model_name}\n[plan] {len(todo)} episodes this pass", flush=True)

    if args.engine == 'cuda':
        n_gpu_layers = base.preflight_vram(n_gpu_layers, num_ctx)

    from llama_cpp import Llama
    print(f"Loading model n_ctx={num_ctx} n_gpu_layers={n_gpu_layers} "
          f"n_batch={base.N_BATCH} logits_all=False ...", flush=True)
    llm = Llama(model_path=model_blob, n_gpu_layers=n_gpu_layers, n_ctx=num_ctx,
                n_batch=base.N_BATCH, seed=42, temperature=0.0, logits_all=False,
                flash_attn=(args.engine == 'cuda'), verbose=False)
    n_vocab = llm.n_vocab()
    reader, method, id_exit, id_hold = base.resolve_and_selftest(llm, n_vocab, system_prompt)

    for k, (eid, path) in enumerate(todo, 1):
        packet = json.load(open(path))
        rec = eval_episode_tiered(llm, reader, id_exit, id_hold, eid, packet,
                                  system_prompt, args.engine, model_name, num_ctx)
        base.append_checkpoint(args.ckpt, rec)
        completed[eid] = rec
        base.rebuild_csv(args.csv, list(completed.values()))
        nT = sum(1 for f in rec['frames'] if f.get('hard_fail'))
        p0 = rec['frames'][0].get('p_exit') if rec['frames'] else None
        print(f"[{k}/{len(todo)}] {eid}: {rec['n_frames_evaluated']} frames "
              f"{rec['elapsed_s']}s p_exit[0]={p0} taintframes={nT}", flush=True)
    print(f"[done] ran {len(todo)} episodes this pass; "
          f"{len(base.load_completed(args.ckpt))}/{len(packet_files)} total complete.", flush=True)


if __name__ == '__main__':
    main()
