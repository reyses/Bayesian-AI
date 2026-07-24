#!/usr/bin/env python3
"""E1 — generation-speed (speculative decoding) bench for the qwen3:14b teacher.

Three arms on the SAME 15 reasoned frames (3 bench episodes x 5 frames), all
temp-0 greedy so the emitted text is deterministic and must be BYTE-IDENTICAL
across arms (that is the lossless-equivalence gate):

  a. BASELINE     plain generation (max_tokens 800), no draft.
  b. PROMPT-LOOKUP llama_cpp.llama_speculative.LlamaPromptLookupDecoding as
                   draft_model= (n-gram lookup from the prompt — zero extra VRAM).
  c. MODEL-DRAFT   a small Qwen3-0.6B GGUF wrapped as a LlamaDraftModel subclass
                   (ModelDraft, incremental draft-KV). llama-cpp-python 0.3.34
                   exposes the draft_model= HOOK + the abstract LlamaDraftModel
                   base, but ships NO concrete model-based class — so we hand-roll
                   one. On a 12GB single GPU the 14B at full offload leaves no room
                   for a GPU-resident draft; this arm documents its actual config
                   (draft device + any 14B offload change) so its number is read in
                   context, not as an apples-to-apples full-offload comparison.

TIMING: the draft lever only speeds the DECODE phase; the ~8-10k-token prefill is
identical across arms. We read llama.cpp's own per-phase counters
(llama_perf_context: t_p_eval_ms prefill, then derive decode wall = total-prefill)
and report BOTH end-to-end tok/s (what a user feels) and decode-isolated tok/s
(the pure lever). Speedup ratio = baseline_decode_toks_per_s vs arm.

Draft forces logits_all=True in the binding (llama.py:344) -> a host-side scores
buffer n_ctx*n_vocab*4B. We size n_ctx tightly (default 11264: max bench prompt
10245 + 800 gen) to keep that buffer ~6.9GB, under free host RAM.

Usage:
  python e1_speculative.py --engine cuda [--n-ctx 11264] [--max-tokens 800]
      [--draft-gguf PATH] [--skip-model-draft] [--limit N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools'))
import bench_common as bc                                    # noqa: E402
import eval_native_ckpt as base                              # noqa: E402  (via bc sys.path)
import llama_cpp                                             # noqa: E402
from llama_cpp.llama_speculative import (LlamaDraftModel,    # noqa: E402
                                         LlamaPromptLookupDecoding)

DEFAULT_N_CTX = 11264      # max bench prompt (10245) + 800 gen + margin
DEFAULT_MAX_TOKENS = 800
PROMPT_LOOKUP_NGRAM = 2    # library defaults; n-gram size for prompt match
PROMPT_LOOKUP_PRED = 10    # tokens proposed per step
MODEL_DRAFT_PRED = 10      # draft tokens the 0.6B proposes per step


# --------------------------------------------------------------- model draft --
class ModelDraft(LlamaDraftModel):
    """Wrap a small Llama as a speculative draft with incremental KV reuse.

    Contract (LlamaDraftModel): __call__(full_input_ids) -> proposed continuation
    tokens. We keep the draft's KV synced to the longest common prefix of the
    accepted sequence (rolling back rejected speculation), eval only the new
    suffix, then greedily roll num_pred draft tokens. The TARGET verifies every
    token, so draft quality affects only SPEED, never the emitted text."""

    def __init__(self, draft_llm, num_pred_tokens=MODEL_DRAFT_PRED):
        self.d = draft_llm
        self.num_pred = int(num_pred_tokens)
        self._reader = None

    def _last_logits(self):
        if self._reader is None:
            self._reader, _ = base.make_last_logits_reader(self.d, self.d.n_vocab())
        return self._reader()

    def __call__(self, input_ids, /, **kwargs):
        d = self.d
        ids = np.asarray(input_ids, dtype=np.intc)
        have = (np.asarray(d._input_ids[:d.n_tokens], dtype=np.intc)
                if d.n_tokens > 0 else np.empty(0, dtype=np.intc))
        m = int(min(len(have), len(ids)))
        k = 0
        while k < m and int(have[k]) == int(ids[k]):
            k += 1
        if k < d.n_tokens:                       # roll back rejected speculation
            d.n_tokens = k
            d._ctx.kv_cache_seq_rm(-1, k, -1)
        if len(ids) > d.n_tokens:                # bring KV up to full accepted seq
            d.eval([int(x) for x in ids[d.n_tokens:]])
        out = []
        for _ in range(self.num_pred):
            tok = int(np.argmax(self._last_logits()))
            out.append(tok)
            d.eval([tok])
        return np.array(out, dtype=np.intc)


# ------------------------------------------------------------------- timing ---
def perf(llm):
    d = llama_cpp.llama_perf_context(llm._ctx.ctx)
    return dict(t_prefill_ms=d.t_p_eval_ms, n_prefill=d.n_p_eval,
                t_decode_ms=d.t_eval_ms, n_decode=d.n_eval)


def run_frame(llm, prompt, max_tokens):
    """Reset -> full prefill+generate (temp 0 greedy). Returns metrics + text."""
    llm.reset()
    llama_cpp.llama_perf_context_reset(llm._ctx.ctx)
    t0 = time.perf_counter()
    out = llm(prompt, max_tokens=max_tokens, temperature=0.0, top_k=1, seed=42,
              stop=["<|im_end|>"])
    total_s = time.perf_counter() - t0
    text = out['choices'][0]['text']
    n_out = out['usage']['completion_tokens']
    pc = perf(llm)
    prefill_s = pc['t_prefill_ms'] / 1000.0
    decode_s = max(total_s - prefill_s, 1e-9)
    return dict(text=text, n_out=n_out, total_s=total_s, prefill_s=prefill_s,
                decode_s=decode_s, e2e_toks=bc.tok_per_s(n_out, total_s),
                decode_toks=bc.tok_per_s(n_out, decode_s),
                text_hash=bc.hash_text(text), perf=pc)


def bench_arm(llm, frames_by_ep, sys_gen, max_tokens, arm):
    rows = []
    for eid, (frames, idxs) in frames_by_ep.items():
        for i in idxs:
            prompt = bc.gen_prompt(frames, i, sys_gen)
            r = run_frame(llm, prompt, max_tokens)
            r.update(eid=eid, frame=i, arm=arm)
            rows.append(r)
            print(f"[{arm}] {eid} f{i:02d}: {r['n_out']} tok  "
                  f"total={r['total_s']:.1f}s prefill={r['prefill_s']:.1f}s "
                  f"decode={r['decode_s']:.1f}s  decode_tok/s={r['decode_toks']:.2f} "
                  f"e2e_tok/s={r['e2e_toks']:.2f}  hash={r['text_hash'][:10]}", flush=True)
    return rows


def summarize(rows):
    n = len(rows)
    dt = np.mean([r['decode_toks'] for r in rows])
    et = np.mean([r['e2e_toks'] for r in rows])
    ds = np.mean([r['decode_s'] for r in rows])
    ts = np.mean([r['total_s'] for r in rows])
    return dict(n=n, mean_decode_toks=dt, mean_e2e_toks=et,
                mean_decode_s=ds, mean_total_s=ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--n-ctx', type=int, default=DEFAULT_N_CTX)
    ap.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument('--draft-gguf', default=os.path.join(bc.MODELS_DIR,
                                                          'Qwen3-0.6B-Q8_0.gguf'))
    ap.add_argument('--draft-engine', choices=['cpu', 'cuda'], default='cpu',
                    help="device for the 0.6B draft (default cpu: the 12GB card is "
                         "full with the 14B at full offload)")
    ap.add_argument('--target-gpu-layers', type=int, default=None,
                    help="override 14B offload for the model-draft arm (to free "
                         "VRAM for a GPU-resident draft); documented in the report")
    ap.add_argument('--skip-model-draft', action='store_true')
    ap.add_argument('--limit', type=int, default=None,
                    help="cap frames PER EPISODE (smoke test)")
    ap.add_argument('--out', default=os.path.join(bc.REPORTS_DIR, 'e1_raw.json'))
    args = ap.parse_args()

    sys_gen = bc.EXAM_SYSTEM_GEN + base.load_genome()

    frames_by_ep = {}
    for eid in bc.BENCH_EPISODES:
        pk = bc.load_packet(eid)
        fr = pk['frames']
        idxs = bc.bench_frame_indices(len(fr))
        if args.limit:
            idxs = idxs[:args.limit]
        frames_by_ep[eid] = (fr, idxs)
    n_frames = sum(len(v[1]) for v in frames_by_ep.values())
    print(f"[plan] E1 arms on {n_frames} frames, max_tokens={args.max_tokens}, "
          f"n_ctx={args.n_ctx}", flush=True)

    result = dict(config=dict(n_ctx=args.n_ctx, max_tokens=args.max_tokens,
                              n_batch=base.N_BATCH, seed=42, temperature=0.0,
                              bench_episodes=bc.BENCH_EPISODES,
                              frames={k: v[1] for k, v in frames_by_ep.items()}),
                  arms={})

    # ---- a + b: single 14B load (baseline then prompt-lookup as draft) --------
    # prompt-lookup needs draft_model set at load (forces logits_all=True), so it
    # cannot share the baseline load. Load baseline first, bench, free, reload.
    print("\n=== ARM a: BASELINE (no draft, logits_all=False) ===", flush=True)
    llm, load_s, mname = bc.load_teacher(args.n_ctx, engine=args.engine)
    print(f"[load] {mname} in {load_s:.1f}s", flush=True)
    rows_a = bench_arm(llm, frames_by_ep, sys_gen, args.max_tokens, 'baseline')
    result['arms']['baseline'] = dict(rows=rows_a, summary=summarize(rows_a),
                                       config=dict(draft=None, logits_all=False))
    llm.close()
    del llm
    import gc; gc.collect(); time.sleep(3)   # let the driver reclaim VRAM before reload

    print("\n=== ARM b: PROMPT-LOOKUP (draft_model=LlamaPromptLookupDecoding) ===",
          flush=True)
    pld = LlamaPromptLookupDecoding(max_ngram_size=PROMPT_LOOKUP_NGRAM,
                                    num_pred_tokens=PROMPT_LOOKUP_PRED)
    llm, load_s, _ = bc.load_teacher(args.n_ctx, engine=args.engine, draft_model=pld)
    print(f"[load] +prompt-lookup in {load_s:.1f}s (logits_all forced True)", flush=True)
    rows_b = bench_arm(llm, frames_by_ep, sys_gen, args.max_tokens, 'prompt_lookup')
    result['arms']['prompt_lookup'] = dict(
        rows=rows_b, summary=summarize(rows_b),
        config=dict(draft='LlamaPromptLookupDecoding',
                    max_ngram_size=PROMPT_LOOKUP_NGRAM,
                    num_pred_tokens=PROMPT_LOOKUP_PRED, logits_all=True))
    llm.close()
    del llm, pld; gc.collect(); time.sleep(3)

    # ---- c: model draft (0.6B) ------------------------------------------------
    if not args.skip_model_draft and os.path.exists(args.draft_gguf):
        print("\n=== ARM c: MODEL-DRAFT (Qwen3-0.6B via ModelDraft wrapper) ===",
              flush=True)
        try:
            from llama_cpp import Llama
            d_ngl = -1 if args.draft_engine == 'cuda' else 0
            draft_llm = Llama(model_path=args.draft_gguf, n_gpu_layers=d_ngl,
                              n_ctx=args.n_ctx, n_batch=base.N_BATCH, seed=42,
                              temperature=0.0, logits_all=False,
                              flash_attn=(args.draft_engine == 'cuda'), verbose=False)
            md = ModelDraft(draft_llm, num_pred_tokens=MODEL_DRAFT_PRED)
            llm, load_s, _ = bc.load_teacher(args.n_ctx, engine=args.engine,
                                             draft_model=md,
                                             n_gpu_layers=args.target_gpu_layers)
            print(f"[load] 14B+0.6B draft (draft on {args.draft_engine}) in "
                  f"{load_s:.1f}s", flush=True)
            rows_c = bench_arm(llm, frames_by_ep, sys_gen, args.max_tokens,
                               'model_draft')
            result['arms']['model_draft'] = dict(
                rows=rows_c, summary=summarize(rows_c),
                config=dict(draft='Qwen3-0.6B-Q8_0', draft_engine=args.draft_engine,
                            num_pred_tokens=MODEL_DRAFT_PRED,
                            target_gpu_layers=args.target_gpu_layers,
                            logits_all=True))
            llm.close(); draft_llm.close()
            del llm, draft_llm, md; gc.collect()
        except Exception as e:  # noqa: BLE001
            print(f"[model_draft] FAILED: {e!r}", flush=True)
            result['arms']['model_draft'] = dict(error=repr(e))
    else:
        reason = ("skipped by flag" if args.skip_model_draft
                  else f"draft gguf not found at {args.draft_gguf}")
        print(f"\n=== ARM c: MODEL-DRAFT skipped ({reason}) ===", flush=True)
        result['arms']['model_draft'] = dict(skipped=reason)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[done] raw -> {args.out}", flush=True)

    # ---- verdicts -------------------------------------------------------------
    base_h = {(r['eid'], r['frame']): r['text_hash'] for r in rows_a}
    print("\n=== SUMMARY ===")
    b = result['arms']['baseline']['summary']
    print(f"baseline: decode {b['mean_decode_toks']:.2f} tok/s, "
          f"e2e {b['mean_e2e_toks']:.2f} tok/s, total {b['mean_total_s']:.1f}s/frame")
    for arm in ('prompt_lookup', 'model_draft'):
        a = result['arms'].get(arm, {})
        if 'summary' not in a:
            print(f"{arm}: {a}")
            continue
        s = a['summary']
        mism = [k for k, h in {(r['eid'], r['frame']): r['text_hash']
                               for r in a['rows']}.items() if base_h.get(k) != h]
        eq = "IDENTICAL" if not mism else f"MISMATCH on {mism}"
        spd = s['mean_decode_toks'] / b['mean_decode_toks']
        e2e_spd = s['mean_e2e_toks'] / b['mean_e2e_toks']
        print(f"{arm}: decode {s['mean_decode_toks']:.2f} tok/s "
              f"(x{spd:.2f} decode, x{e2e_spd:.2f} e2e)  equivalence={eq}")


if __name__ == '__main__':
    main()
