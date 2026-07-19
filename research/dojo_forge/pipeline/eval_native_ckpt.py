"""
DOJO FORGE — Crash-safe checkpointed native acceptance eval (readout v2).

Derived from pipeline/eval_native.py (accepted native-logit method, comms
139/141/142). qwen3:14b native readout: a closed `</think>` trace makes the next
token the true HOLD/EXIT answer; we read that answer position's logits.

WHY THIS FILE EXISTS
--------------------
1. eval_native.py opens acceptance_native_gen0.csv with mode 'w' (truncates on
   start) and has no resume: a crash after N of 156 episodes loses all N.
2. eval_native.py used create_completion(logprobs=50), which FORCES
   logits_all=True. That allocates an all-positions logit buffer
   (n_ctx*n_vocab*4B ≈ 8192*151936*4 ≈ 5.0 GB HOST RAM). On top of the 9.3 GB
   model that native-OOM-killed the run on 16 GB Windows (exit 255), and
   crashed the memory-constrained WSL VM (see reports/gpu_wsl_build.md).

FIX (readout="last_logits_v2")
------------------------------
Construct Llama with logits_all=False (no big buffer) and read ONLY the last
position's logits via the low-level binding, then take a full-vocab log-softmax
and pull the two exact candidate logprobs (EXIT, HOLD). No create_completion, no
logprobs=50, no top-N floor. Footprint of the logit readout: n_vocab*4B ≈ 0.6 MB
(vs ~5 GB). Exact two-token logprobs are a strictly better instrument than the
truncated top-50, so the old "top-N floor guard" is obsolete and removed.

142 GUARDS still enforced: num_ctx=8192; prompt_tokens >= num_ctx -> ctx
tripwire (frame hard-fail, episode tainted). The floor guard is replaced by an
exact-readout self-test at startup (fail-fast if the binding or token ids are wrong).

CHECKPOINT FILE
---------------
research/dojo_forge/gate_state/acceptance_results.jsonl — one JSON line per
COMPLETED episode, appended immediately (flush+fsync). On restart, completed
episode_ids are skipped. A companion 142-format CSV
(reports/acceptance_native_gen0.csv: eid,frame_idx,p_exit,prompt_tokens,tainted)
is rebuilt from the jsonl on every start (never the source of truth).

Per-episode line schema:
  {"episode_id","engine":"cpu|cuda","model","num_ctx":8192,"readout":"last_logits_v2",
   "tainted","taint_reason","exit_frame","n_frames_evaluated","elapsed_s","ts",
   "frames":[{"frame_idx","p_exit","logit_exit","logit_hold","lp_exit","lp_hold",
              "prompt_tokens","decision"}]}
  frames[].logit_exit / logit_hold are full-vocab log-softmax LOGPROBS (back-compat
  field names; same meaning as the old create_completion logprobs). lp_exit/lp_hold
  are explicit aliases.

INVOCATION
----------
  CPU (Windows):  <dojo_forge>\.venv\Scripts\python.exe pipeline\eval_native_ckpt.py --engine cpu
  CUDA (WSL):     source /home/reyses/venvs/llamacpp-cuda/bin/activate
                  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
                  python pipeline/eval_native_ckpt.py --engine cuda
Both engines write the SAME checkpoint file (CPU run resumable by CUDA and vice-versa).
"""
import os
import sys
import json
import glob
import math
import time
import platform
import argparse

DOJO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PACKETS_DIR = os.path.join(DOJO_ROOT, 'reports', 'gen0', 'packets')
GENOME_PATH = os.path.join(DOJO_ROOT, 'genome', 'GENOME.md')
GATE_STATE_DIR = os.path.join(DOJO_ROOT, 'gate_state')

DEFAULT_CKPT = os.path.join(GATE_STATE_DIR, 'acceptance_results.jsonl')
DEFAULT_CSV = os.path.join(DOJO_ROOT, 'reports', 'acceptance_native_gen0.csv')

NUM_CTX = 8192
READOUT = "last_logits_v2"
MAX_LOGIT_BUFFER_MB = 500.0  # startup HOST-RAM tripwire (logits_all buffer)

# ---- VRAM overallocation guardrails (RTX 3060 12GB; full offload measured 11.7GB) ----
N_BATCH = 256          # prompt-processing CHUNK size. Smaller = smaller VRAM compute
                       # spike per chunk (frame-0 ~5.8k-token prompt processed in ~23
                       # chunks). Halved from 512 to buy VRAM headroom on the 12GB card.
QWEN3_N_BLOCKS = 41    # 40 transformer blocks + output layer (loader dump: block_count=40)
VRAM_PER_LAYER_MB = 210.0    # ~8.4GB Q4 weights / 40 blocks
VRAM_KV_MB_AT_8192 = 1300.0  # KV cache at n_ctx=8192 (scales linearly with n_ctx)
VRAM_COMPUTE_MARGIN_MB = 1600.0  # compute buffers + fragmentation + display headroom

BLOB_NAME = 'sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e'
DEFAULT_BLOB_WIN = r"D:\ollama\models\blobs\{}".format(BLOB_NAME)
DEFAULT_BLOB_WSL = "/mnt/d/ollama/models/blobs/{}".format(BLOB_NAME)

# Assistant scaffold whose closed </think> forces the next token to be the answer.
THINK_SUFFIX = "<|im_start|>assistant\n<think>\nDecision bypassed for native logprobs.\n</think>\n"


def load_genome():
    if os.path.exists(GENOME_PATH):
        with open(GENOME_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


# --------------------------------------------------------------------------- #
#  Low-level last-position logit readout (logits_all=False)                     #
# --------------------------------------------------------------------------- #
def make_last_logits_reader(llm, n_vocab):
    """Return (reader_fn, method_name). reader_fn() -> np.ndarray(n_vocab,) of the
    logits for the most recently decoded token. Probes the bindings that 0.3.34
    actually populates with logits_all=False and locks onto the first that yields
    a finite, non-degenerate vocab row."""
    import numpy as np
    import llama_cpp
    ctx = llm._ctx.ctx

    def _from_ith():
        ptr = llama_cpp.llama_get_logits_ith(ctx, -1)
        return np.ctypeslib.as_array(ptr, shape=(n_vocab,)).astype(np.float64).copy()

    def _from_get():
        ptr = llama_cpp.llama_get_logits(ctx)
        return np.ctypeslib.as_array(ptr, shape=(n_vocab,)).astype(np.float64).copy()

    def _from_eval_logits():
        el = llm.eval_logits
        return np.asarray(el[-1], dtype=np.float64)

    def _from_scores():
        return np.asarray(llm.scores[llm.n_tokens - 1], dtype=np.float64)

    candidates = [("llama_get_logits_ith(-1)", _from_ith),
                  ("llama_get_logits", _from_get),
                  ("eval_logits[-1]", _from_eval_logits),
                  ("scores[n_tokens-1]", _from_scores)]
    for name, fn in candidates:
        try:
            arr = fn()
            if arr is not None and arr.shape == (n_vocab,) \
                    and np.isfinite(arr).all() and float(arr.std()) > 1e-6:
                return fn, name
        except Exception:
            continue
    raise RuntimeError("no binding call returned usable last-position logits "
                       "(tried: %s)" % ", ".join(n for n, _ in candidates))


def preflight_vram(requested_layers, n_ctx):
    """CUDA only: query FREE VRAM via nvidia-smi and pick a safe n_gpu_layers with
    headroom, so a display or stray process holding VRAM can't make full offload
    overallocate and crash the load. Returns a (possibly reduced) layer count;
    aborts if a real 'cuda' run isn't feasible. If nvidia-smi is unavailable, keeps
    the request but WARNS (never silently trusts on a machine that can't be probed)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=15)
        free_mb, total_mb = (int(x) for x in out.strip().splitlines()[0].split(','))
    except Exception as e:  # noqa: BLE001
        print(f"[vram] nvidia-smi unavailable ({e}); cannot preflight. Keeping "
              f"requested={requested_layers}. If load OOMs, pass --n-gpu-layers 35/30.",
              flush=True)
        return requested_layers
    kv_mb = VRAM_KV_MB_AT_8192 * (n_ctx / 8192.0)
    budget = free_mb - kv_mb - VRAM_COMPUTE_MARGIN_MB
    max_fit = int(budget // VRAM_PER_LAYER_MB)
    want = QWEN3_N_BLOCKS if requested_layers < 0 else requested_layers
    safe = min(want, max(0, max_fit))
    print(f"[vram] free={free_mb}MB/{total_mb}MB | KV~{kv_mb:.0f} margin={VRAM_COMPUTE_MARGIN_MB:.0f} "
          f"-> budget {budget:.0f}MB ~= {max_fit} blks; requested {want} -> USING {safe}", flush=True)
    if safe < 20:
        raise SystemExit(
            f"[FATAL] only ~{safe} layers fit in {free_mb}MB free VRAM — that's a mostly-CPU "
            f"run masquerading as 'cuda'. Free VRAM (close the display GPU user / other procs) "
            f"or run --engine cpu deliberately. Refusing.")
    if safe < want:
        print(f"[vram] WARN: reduced offload {want}->{safe} for headroom; the remaining "
              f"{QWEN3_N_BLOCKS - safe} layers run on CPU (slower but safe).", flush=True)
    return safe


def logsoftmax_two(logits, id_exit, id_hold):
    """Full-vocab log-softmax; return (lp_exit, lp_hold, p_exit)."""
    import numpy as np
    m = float(logits.max())
    lse = m + math.log(float(np.exp(logits - m).sum()))
    lp_e = float(logits[id_exit]) - lse
    lp_h = float(logits[id_hold]) - lse
    p = math.exp(lp_e) / (math.exp(lp_e) + math.exp(lp_h))
    return lp_e, lp_h, p


def resolve_and_selftest(llm, n_vocab, system_prompt):
    """Resolve EXIT/HOLD ids, lock the logits reader, self-test on canned frames.
    Fail fast on any anomaly. Returns (reader_fn, method_name, id_exit, id_hold)."""
    id_exit = llm.tokenize(b"EXIT", add_bos=False, special=False)[0]
    id_hold = llm.tokenize(b"HOLD", add_bos=False, special=False)[0]
    tok_exit = llm.detokenize([id_exit]).decode('utf-8', 'ignore')
    tok_hold = llm.detokenize([id_hold]).decode('utf-8', 'ignore')
    if id_exit == id_hold:
        raise SystemExit(f"[FATAL] EXIT and HOLD share first-token id {id_exit}; "
                         f"single-position readout impossible.")

    def eval_canned(frame_text):
        llm.reset()
        seg = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
               f"<|im_start|>user\n{frame_text}<|im_end|>\n{THINK_SUFFIX}")
        llm.eval(llm.tokenize(seg.encode('utf-8'), add_bos=True, special=True))

    eval_canned("Trade just opened. Price sitting right at the mean, low volatility. Action:")
    reader, method = make_last_logits_reader(llm, n_vocab)
    hold_logits = reader()
    _, _, p_hold_frame = logsoftmax_two(hold_logits, id_exit, id_hold)
    hold_argmax = int(hold_logits.argmax())

    eval_canned("Price now +45 ticks in favor, momentum clearly stalling, target reached. Action:")
    exit_logits = reader()
    _, _, p_exit_frame = logsoftmax_two(exit_logits, id_exit, id_hold)
    exit_argmax = int(exit_logits.argmax())

    print(f"[selftest] reader={method}  ID_EXIT={id_exit}({tok_exit!r})  "
          f"ID_HOLD={id_hold}({tok_hold!r})", flush=True)
    print(f"[selftest] HOLD-frame P(EXIT)={p_hold_frame:.6f} "
          f"(argmax={hold_argmax} {llm.detokenize([hold_argmax]).decode('utf-8','ignore')!r})", flush=True)
    print(f"[selftest] EXIT-frame P(EXIT)={p_exit_frame:.6f} "
          f"(argmax={exit_argmax} {llm.detokenize([exit_argmax]).decode('utf-8','ignore')!r})", flush=True)

    if not (p_hold_frame < 0.2 and p_exit_frame > 0.7):
        raise SystemExit(
            f"[FATAL] self-test failed: HOLD-frame P(EXIT)={p_hold_frame:.4f} (want <0.2), "
            f"EXIT-frame P(EXIT)={p_exit_frame:.4f} (want >0.7). Token ids or the "
            f"last-logits binding are wrong for this model/build — refusing to run.")
    print("[selftest] PASS", flush=True)
    return reader, method, id_exit, id_hold


# --------------------------------------------------------------------------- #
#  Episode evaluation (incremental KV, no big logit buffer)                     #
# --------------------------------------------------------------------------- #
def eval_episode(llm, reader, id_exit, id_hold, eid, packet, system_prompt,
                 engine, model_name):
    frames = packet.get('frames', [])
    llm.reset()
    rec_frames = []
    tainted = False
    taint_reason = None
    exit_frame = None
    t0 = time.time()

    for i, frame in enumerate(frames):
        frame_text = frame['text']
        if i == 0:
            seg = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                   f"<|im_start|>user\n{frame_text}<|im_end|>\n{THINK_SUFFIX}")
            toks = llm.tokenize(seg.encode('utf-8'), add_bos=True, special=True)
        else:
            seg = f"<|im_start|>user\n{frame_text}<|im_end|>\n{THINK_SUFFIX}"
            toks = llm.tokenize(seg.encode('utf-8'), add_bos=False, special=True)

        total_tokens = llm.n_tokens + len(toks)
        # Guard: ctx tripwire (prompt_eval_count >= num_ctx).
        if total_tokens >= NUM_CTX:
            tainted = True
            taint_reason = f"ctx_overflow:{total_tokens}>={NUM_CTX}"
            rec_frames.append(dict(frame_idx=i, p_exit=None, logit_exit=None,
                                   logit_hold=None, lp_exit=None, lp_hold=None,
                                   prompt_tokens=total_tokens, decision=None,
                                   hard_fail="ctx"))
            break

        llm.eval(toks)
        logits = reader()
        lp_e, lp_h, p_exit = logsoftmax_two(logits, id_exit, id_hold)
        decision = "EXIT" if p_exit > 0.5 else "HOLD"
        rec_frames.append(dict(frame_idx=i, p_exit=round(p_exit, 6),
                               logit_exit=round(lp_e, 6), logit_hold=round(lp_h, 6),
                               lp_exit=round(lp_e, 6), lp_hold=round(lp_h, 6),
                               prompt_tokens=llm.n_tokens, decision=decision))
        if decision == "EXIT" and exit_frame is None:
            exit_frame = i
        # Commit the chosen decision to the KV cache for the next frame's context.
        dtoks = llm.tokenize(f"{decision}<|im_end|>\n".encode('utf-8'),
                             add_bos=False, special=True)
        if llm.n_tokens + len(dtoks) < NUM_CTX:
            llm.eval(dtoks)

    return dict(
        episode_id=eid, engine=engine, model=model_name, num_ctx=NUM_CTX,
        readout=READOUT, tainted=tainted, taint_reason=taint_reason,
        exit_frame=exit_frame, n_frames_evaluated=len(rec_frames),
        elapsed_s=round(time.time() - t0, 3), ts=time.time(), frames=rec_frames,
    )


# --------------------------------------------------------------------------- #
#  Checkpoint I/O                                                               #
# --------------------------------------------------------------------------- #
def load_completed(ckpt_path):
    done = {}
    if not os.path.exists(ckpt_path):
        return done
    with open(ckpt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from a hard kill — that episode reruns
            if 'episode_id' in rec:
                done[rec['episode_id']] = rec
    return done


def append_checkpoint(ckpt_path, rec):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')
        f.flush()
        os.fsync(f.fileno())


def rebuild_csv(csv_path, records):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("eid,frame_idx,p_exit,prompt_tokens,tainted\n")
        for rec in records:
            eid = rec['episode_id']
            for fr in rec['frames']:
                if fr.get('hard_fail'):
                    f.write(f"{eid},{fr['frame_idx']},NaN,{fr['prompt_tokens']},Y\n")
                else:
                    f.write(f"{eid},{fr['frame_idx']},{fr['p_exit']:.6f},"
                            f"{fr['prompt_tokens']},N\n")


def main():
    ap = argparse.ArgumentParser(description="Crash-safe checkpointed native acceptance eval (v2)")
    ap.add_argument('--engine', choices=['cpu', 'cuda'], required=True)
    ap.add_argument('--model-blob', default=None)
    ap.add_argument('--n-gpu-layers', type=int, default=None,
                    help="Override offload (cuda default -1; OOM fallback 40/35/30)")
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--packets-dir', default=PACKETS_DIR)
    ap.add_argument('--limit', type=int, default=None,
                    help="Process at most N NEW episodes then stop (smoke-test)")
    ap.add_argument('--dry-run', action='store_true',
                    help="Exercise skip/append logic with a MOCK model (no llama_cpp)")
    args = ap.parse_args()

    if args.model_blob:
        model_blob = args.model_blob
    elif platform.system() == 'Windows':
        model_blob = DEFAULT_BLOB_WIN
    else:
        model_blob = DEFAULT_BLOB_WSL
    model_name = os.path.basename(model_blob)
    n_gpu_layers = args.n_gpu_layers if args.n_gpu_layers is not None \
        else (-1 if args.engine == 'cuda' else 0)

    packet_files = sorted(glob.glob(os.path.join(args.packets_dir, "*.json")))
    if not packet_files:
        print(f"No packet files in {args.packets_dir}", file=sys.stderr)
        sys.exit(1)

    completed = load_completed(args.ckpt)
    rebuild_csv(args.csv, list(completed.values()))
    print(f"[resume] {len(completed)} of {len(packet_files)} episodes already in "
          f"{os.path.relpath(args.ckpt, DOJO_ROOT)}", flush=True)

    system_prompt = (f"Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason."
                     f"\n\nRULES (Genome):\n{load_genome()}")

    todo = [(os.path.basename(p).replace('.json', ''), p) for p in packet_files]
    todo = [(eid, p) for eid, p in todo if eid not in completed]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"[plan] engine={args.engine} n_gpu_layers={n_gpu_layers} readout={READOUT} "
          f"model={model_name}\n[plan] {len(todo)} episodes to run this pass "
          f"(limit={args.limit})", flush=True)

    # VRAM overallocation guardrail (cuda only): fit offload to actually-free VRAM.
    if args.engine == 'cuda' and not args.dry_run:
        n_gpu_layers = preflight_vram(n_gpu_layers, NUM_CTX)

    llm = reader = id_exit = id_hold = None
    if not args.dry_run:
        from llama_cpp import Llama
        # logits_all=False (default) => NO all-positions logit buffer. flash_attn on cuda.
        # n_batch=N_BATCH chunks prompt processing to bound the VRAM compute spike.
        print(f"Loading model n_ctx={NUM_CTX} n_gpu_layers={n_gpu_layers} "
              f"n_batch={N_BATCH} logits_all=False ...", flush=True)
        t_load = time.time()
        llm = Llama(model_path=model_blob, n_gpu_layers=n_gpu_layers, n_ctx=NUM_CTX,
                    n_batch=N_BATCH, seed=42, temperature=0.0, logits_all=False,
                    flash_attn=(args.engine == 'cuda'), verbose=False)
        print(f"Model loaded in {time.time()-t_load:.1f}s", flush=True)

        # Memory tripwire: refuse to run if a large logit buffer is in play.
        n_vocab = llm.n_vocab()
        logits_all = bool(getattr(llm, "_logits_all", False))
        buf_mb = (NUM_CTX if logits_all else 1) * n_vocab * 4 / 1e6
        print(f"[mem] logits_all={logits_all} n_vocab={n_vocab} "
              f"logit_buffer~{buf_mb:.1f}MB (cap {MAX_LOGIT_BUFFER_MB}MB)", flush=True)
        if buf_mb > MAX_LOGIT_BUFFER_MB:
            raise SystemExit(f"[FATAL] logit buffer ~{buf_mb:.0f}MB exceeds "
                             f"{MAX_LOGIT_BUFFER_MB}MB cap — would OOM. Ensure logits_all=False.")

        reader, method, id_exit, id_hold = resolve_and_selftest(llm, n_vocab, system_prompt)

    n_run = 0
    for eid, pkt_path in todo:
        with open(pkt_path, 'r', encoding='utf-8') as f:
            packet = json.load(f)
        if args.dry_run:
            rec = dict(episode_id=eid, engine=args.engine, model=model_name,
                       num_ctx=NUM_CTX, readout=READOUT, tainted=False,
                       taint_reason=None, exit_frame=None, n_frames_evaluated=1,
                       elapsed_s=0.0, ts=time.time(),
                       frames=[dict(frame_idx=0, p_exit=0.5, logit_exit=-1.0,
                                    logit_hold=-1.0, lp_exit=-1.0, lp_hold=-1.0,
                                    prompt_tokens=123, decision="HOLD")])
        else:
            rec = eval_episode(llm, reader, id_exit, id_hold, eid, packet,
                               system_prompt, args.engine, model_name)
        append_checkpoint(args.ckpt, rec)
        completed[eid] = rec
        rebuild_csv(args.csv, list(completed.values()))
        n_run += 1
        flag = f"TAINTED({rec['taint_reason']})" if rec['tainted'] else "ok"
        p0 = rec['frames'][0].get('p_exit') if rec['frames'] else None
        print(f"[{n_run}/{len(todo)}] {eid}: {rec['n_frames_evaluated']} frames "
              f"{rec['elapsed_s']}s p_exit[0]={p0} {flag}", flush=True)

    print(f"[done] ran {n_run} episodes this pass; "
          f"{len(completed)}/{len(packet_files)} total complete.", flush=True)


if __name__ == '__main__':
    main()
