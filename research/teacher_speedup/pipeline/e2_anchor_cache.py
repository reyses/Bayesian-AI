#!/usr/bin/env python3
"""E2 — prefill-speed bench: anchor KV cache (save_state/load_state) vs full rebuild.

Every tiered frame prompt is  [system + genome] + [user: ANCHOR + history + NOW] +
[think-suffix]. The system+genome+anchor prefix is BYTE-identical across all frames
of an episode (~7-8k of the ~8-10k prompt tokens). The acceptance harness rebuilds
and re-prefills the whole thing every frame (eval_native_tiered: llm.reset + full
eval — the correct-but-slow decay implementation). E2 tests reclaiming that:

  (a) BASELINE   per frame: llm.reset() + eval(full_prompt_tokens).
  (b) ANCHOR-CACHED: eval the shared prefix ONCE, llm.save_state(); then per frame
      llm.load_state() + eval(remainder_tokens only).

The cached prefix is the LONGEST COMMON TOKEN PREFIX across all benched frames'
tokenizations (computed empirically, not assumed — BPE boundaries can bite). That
is exactly system+genome+anchor when the anchor text is shared, and we report the
measured prefix length.

CORRECTNESS GATE: the last-position logits (the HOLD/EXIT answer row, read via the
same make_last_logits_reader path the acceptance harness uses) must match baseline
within max|delta| <= 1e-4. Bitwise is the ideal; the tolerance covers fp reduction
reordering when a suffix is evaluated against a restored KV vs a fresh full eval.

Usage:
  python e2_anchor_cache.py --engine cuda [--n-ctx 11264] [--eid EID] [--n-frames 6]
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
import eval_native_ckpt as base                              # noqa: E402

DEFAULT_N_CTX = 11264
DEFAULT_N_FRAMES = 6
LOGIT_TOL = 1e-4


def pick_frames(n_frames_total, k):
    """k evenly spaced reasoned frames (indices >=1) for the episode."""
    idxs = np.unique(np.round(np.linspace(1, n_frames_total - 1, k)).astype(int))
    return [int(x) for x in idxs]


def longest_common_prefix(tok_lists):
    if not tok_lists:
        return 0
    n = min(len(t) for t in tok_lists)
    for p in range(n):
        v = tok_lists[0][p]
        if any(t[p] != v for t in tok_lists):
            return p
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--n-ctx', type=int, default=DEFAULT_N_CTX)
    ap.add_argument('--eid', default=bc.BENCH_EPISODES[0])
    ap.add_argument('--n-frames', type=int, default=DEFAULT_N_FRAMES)
    ap.add_argument('--out', default=os.path.join(bc.REPORTS_DIR, 'e2_raw.json'))
    args = ap.parse_args()

    sys_read = bc.system_prompt_readout()
    pk = bc.load_packet(args.eid)
    frames = pk['frames']
    idxs = pick_frames(len(frames), args.n_frames)
    print(f"[plan] E2 episode {args.eid}: frames {idxs} n_ctx={args.n_ctx}", flush=True)

    # Tokenize every frame's full prompt (add_bos=True, special=True — as acceptance).
    from llama_cpp import Llama  # noqa: F401  (loaded via bc.load_teacher)
    llm, load_s, mname = bc.load_teacher(args.n_ctx, engine=args.engine)
    print(f"[load] {mname} in {load_s:.1f}s", flush=True)
    reader, method, id_exit, id_hold = base.resolve_and_selftest(llm, llm.n_vocab(),
                                                                 sys_read)
    print(f"[readout] {method}", flush=True)

    full_toks = {}
    for i in idxs:
        prompt = bc.logit_prompt(frames, i, sys_read)
        full_toks[i] = llm.tokenize(prompt.encode('utf-8'), add_bos=True, special=True)

    # The cached prefix = longest common token prefix across all benched frames.
    # (Reported so the reader sees exactly how much prefill the cache reclaims.)
    prefix_len = longest_common_prefix([full_toks[i] for i in idxs])
    # Sanity: how much of that is the intended system+anchor boundary?
    anchor_prompt = (f"<|im_start|>system\n{sys_read}<|im_end|>\n"
                     f"<|im_start|>user\n{frames[0]['text']}")
    anchor_toks = llm.tokenize(anchor_prompt.encode('utf-8'), add_bos=True, special=True)
    print(f"[prefix] common-token-prefix={prefix_len}  system+anchor tokens="
          f"{len(anchor_toks)}", flush=True)

    # ---- (a) BASELINE: full per-frame rebuild --------------------------------
    print("\n=== (a) BASELINE full rebuild ===", flush=True)
    base_rows = {}
    for i in idxs:
        toks = full_toks[i]
        llm.reset()
        t0 = time.perf_counter()
        llm.eval(toks)
        dt = time.perf_counter() - t0
        logits = reader()
        base_rows[i] = dict(prefill_s=dt, n_tok=len(toks),
                            logits=np.asarray(logits, dtype=np.float64),
                            hash=bc.hash_logits(logits))
        print(f"[baseline] f{i:02d}: {len(toks)} tok  prefill={dt:.3f}s  "
              f"hash={base_rows[i]['hash'][:10]}", flush=True)

    # ---- (b) ANCHOR-CACHED: eval prefix once, save_state, load+eval remainder --
    print("\n=== (b) ANCHOR-CACHED ===", flush=True)
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(full_toks[idxs[0]][:prefix_len])
    anchor_eval_s = time.perf_counter() - t0
    state = llm.save_state()
    print(f"[anchor] eval {prefix_len} tok once in {anchor_eval_s:.3f}s, saved state "
          f"({state.n_tokens} tok)", flush=True)

    cache_rows = {}
    for i in idxs:
        toks = full_toks[i]
        remainder = toks[prefix_len:]
        t0 = time.perf_counter()
        llm.load_state(state)
        rem_toks = [int(x) for x in remainder]
        llm.eval(rem_toks)
        dt = time.perf_counter() - t0
        logits = reader()
        md = bc.max_abs_delta(logits, base_rows[i]['logits'])
        cache_rows[i] = dict(prefill_s=dt, n_rem=len(remainder),
                             max_abs_delta=md, hash=bc.hash_logits(logits),
                             pass_tol=md <= LOGIT_TOL)
        verdict = 'OK' if md <= LOGIT_TOL else 'FAIL'
        print(f"[cached]   f{i:02d}: rem={len(remainder)} tok  load+eval={dt:.3f}s  "
              f"max|Δlogit|={md:.2e} [{verdict}]", flush=True)

    # ---- summary --------------------------------------------------------------
    mean_base = np.mean([base_rows[i]['prefill_s'] for i in idxs])
    mean_cache = np.mean([cache_rows[i]['prefill_s'] for i in idxs])
    # amortized: one-time anchor eval spread over the benched frames
    mean_cache_amort = mean_cache + anchor_eval_s / len(idxs)
    worst_delta = max(cache_rows[i]['max_abs_delta'] for i in idxs)
    all_pass = all(cache_rows[i]['pass_tol'] for i in idxs)

    result = dict(
        eid=args.eid, config=dict(n_ctx=args.n_ctx, n_batch=base.N_BATCH,
                                  frames=idxs, prefix_len=prefix_len,
                                  anchor_tokens=len(anchor_toks), logit_tol=LOGIT_TOL),
        anchor_eval_s=anchor_eval_s,
        per_frame={str(i): dict(
            n_tok=base_rows[i]['n_tok'], baseline_prefill_s=base_rows[i]['prefill_s'],
            cached_remainder_tok=cache_rows[i]['n_rem'],
            cached_prefill_s=cache_rows[i]['prefill_s'],
            max_abs_delta=cache_rows[i]['max_abs_delta'],
            pass_tol=cache_rows[i]['pass_tol']) for i in idxs},
        summary=dict(mean_baseline_prefill_s=mean_base,
                     mean_cached_prefill_s=mean_cache,
                     mean_cached_amortized_s=mean_cache_amort,
                     speedup_raw=mean_base / mean_cache,
                     speedup_amortized=mean_base / mean_cache_amort,
                     worst_max_abs_delta=worst_delta, all_pass_tol=all_pass))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(result, fh, indent=2)

    s = result['summary']
    print("\n=== SUMMARY ===")
    print(f"mean prefill: baseline {s['mean_baseline_prefill_s']:.3f}s  ->  "
          f"cached {s['mean_cached_prefill_s']:.3f}s  "
          f"(amortized {s['mean_cached_amortized_s']:.3f}s)")
    print(f"speedup: x{s['speedup_raw']:.2f} raw  x{s['speedup_amortized']:.2f} amortized")
    print(f"correctness: worst max|Δlogit|={s['worst_max_abs_delta']:.2e}  "
          f"(tol {LOGIT_TOL}) -> {'ALL PASS' if s['all_pass_tol'] else 'FAIL'}")
    print(f"[done] raw -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
