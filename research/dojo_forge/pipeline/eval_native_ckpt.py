"""
DOJO FORGE — Crash-safe checkpointed native acceptance eval.

Derived from pipeline/eval_native.py (the accepted native-logit acceptance
runner, comms 139/141/142). Same qwen3:14b native-logit readout — a closed
`</think>` trace makes the next token the true HOLD/EXIT answer, then we pull
the two candidate logprobs from create_completion(logprobs=50).

WHY THIS FILE EXISTS
--------------------
eval_native.py writes acceptance_native_gen0.csv with mode 'w' (truncates on
start) and has no resume: a crash after N of 156 episodes loses all N. This
variant appends ONE JSON line per COMPLETED episode to a checkpoint file
immediately (flush + fsync), and on restart skips every episode already in the
checkpoint. Kill it any time; rerun; it resumes.

142 GUARDS (baked in, engine-agnostic)
--------------------------------------
  1. num_ctx = 8192.
  2. prompt_eval_count tripwire: prompt_tokens >= num_ctx  -> frame hard-fail,
     episode marked tainted (ctx overflow).
  3. top-N floor guard: if EITHER candidate token (EXIT / HOLD) is absent from
     the returned top-N logprobs, that frame hard-fails (do NOT record a floored
     -100 sentinel as a probability). Episode marked tainted.

CHECKPOINT FILE
---------------
  research/dojo_forge/gate_state/acceptance_results.jsonl   (one line / episode)
Each line:
  {
    "episode_id", "engine" (cpu|cuda), "model" (blob basename), "num_ctx",
    "tainted" (bool), "taint_reason" (str|null), "exit_frame" (int|null),
    "n_frames_evaluated" (int), "elapsed_s" (float), "ts" (epoch),
    "frames": [ {"frame_idx","p_exit","logit_exit","logit_hold",
                 "prompt_tokens","decision"} , ... ]
  }
A companion acceptance_native_gen0.csv (142 table: eid,frame_idx,p_exit,
prompt_tokens,tainted) is REBUILT from the jsonl on every start so it is always
consistent with the checkpoint — never the source of truth.

HOW AG INVOKES IT
-----------------
  CPU (Windows, matches the current running batch config):
    <dojo_forge>/.venv/Scripts/python.exe pipeline/eval_native_ckpt.py --engine cpu
  CUDA (WSL, new GPU env):
    source /home/reyses/venvs/llamacpp-cuda/bin/activate
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # see gpu_wsl_build.md
    python pipeline/eval_native_ckpt.py --engine cuda
Both write the SAME checkpoint file, so a CPU run can be stopped and a CUDA run
resumes exactly where it left off (episode-level granularity).
"""
import os
import io
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
CTX_FLOOR_SENTINEL = -100.0  # value create_completion floor / missing-key maps to

# Default model blob (qwen3:14b Q4, ollama store) per platform. Override with --model-blob.
BLOB_NAME = 'sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e'
DEFAULT_BLOB_WIN = r"D:\ollama\models\blobs\{}".format(BLOB_NAME)
DEFAULT_BLOB_WSL = "/mnt/d/ollama/models/blobs/{}".format(BLOB_NAME)


def load_genome():
    if os.path.exists(GENOME_PATH):
        with open(GENOME_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def get_candidate_logprobs(llm, prompt_text):
    """Return (logit_exit, logit_hold, found_exit, found_hold, prompt_eval_count).

    found_* is True only when the candidate token was actually present in the
    returned top-N logprobs (142 floor guard). On a ctx overflow returns the
    sentinel with prompt_eval_count = NUM_CTX + 1 so the caller trips the ctx
    tripwire.
    """
    try:
        response = llm.create_completion(
            prompt_text,
            max_tokens=1,
            logprobs=50,
            temperature=0.0,
        )
    except ValueError as e:
        if "exceed context window" in str(e):
            return CTX_FLOOR_SENTINEL, CTX_FLOOR_SENTINEL, False, False, NUM_CTX + 1
        raise

    prompt_eval_count = response['usage']['prompt_tokens']
    logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]

    def _lookup(*keys):
        for k in keys:
            if k in logprobs:
                return logprobs[k], True
        return CTX_FLOOR_SENTINEL, False

    logit_exit, found_exit = _lookup('EXIT', ' EXIT', 'exit', ' exit')
    logit_hold, found_hold = _lookup('HOLD', ' HOLD', 'hold', ' hold')
    return logit_exit, logit_hold, found_exit, found_hold, prompt_eval_count


def eval_episode(llm, eid, packet, system_prompt, engine, model_name):
    """Run one episode; return the checkpoint record dict."""
    frames = packet.get('frames', [])
    llm.reset()
    prompt_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    rec_frames = []
    tainted = False
    taint_reason = None
    exit_frame = None
    t0 = time.time()

    for i, frame in enumerate(frames):
        frame_text = frame['text']
        # Closed </think> trace => next token is the true HOLD/EXIT answer (comms 141/142).
        prompt_text += (
            f"<|im_start|>user\n{frame_text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\nDecision bypassed for native logprobs.\n</think>\n"
        )

        logit_exit, logit_hold, found_exit, found_hold, pt_tokens = \
            get_candidate_logprobs(llm, prompt_text)

        # Guard 2: ctx tripwire.
        if pt_tokens >= NUM_CTX:
            tainted = True
            taint_reason = f"ctx_overflow:{pt_tokens}>={NUM_CTX}"
            rec_frames.append(dict(frame_idx=i, p_exit=None, logit_exit=None,
                                   logit_hold=None, prompt_tokens=pt_tokens,
                                   decision=None, hard_fail="ctx"))
            break

        # Guard 3: top-N floor — either candidate missing from returned top-N.
        if not found_exit or not found_hold:
            tainted = True
            taint_reason = (f"floor_missing:EXIT_found={found_exit},"
                            f"HOLD_found={found_hold}")
            rec_frames.append(dict(frame_idx=i, p_exit=None,
                                   logit_exit=(logit_exit if found_exit else None),
                                   logit_hold=(logit_hold if found_hold else None),
                                   prompt_tokens=pt_tokens, decision=None,
                                   hard_fail="floor"))
            break

        p_exit = math.exp(logit_exit) / (math.exp(logit_exit) + math.exp(logit_hold))
        decision = "EXIT" if p_exit > 0.5 else "HOLD"
        rec_frames.append(dict(frame_idx=i, p_exit=round(p_exit, 6),
                               logit_exit=round(float(logit_exit), 6),
                               logit_hold=round(float(logit_hold), 6),
                               prompt_tokens=pt_tokens, decision=decision))
        if decision == "EXIT" and exit_frame is None:
            exit_frame = i
        # Append chosen decision to the running context for the next frame.
        prompt_text += f"{decision}<|im_end|>\n"

    return dict(
        episode_id=eid,
        engine=engine,
        model=model_name,
        num_ctx=NUM_CTX,
        tainted=tainted,
        taint_reason=taint_reason,
        exit_frame=exit_frame,
        n_frames_evaluated=len(rec_frames),
        elapsed_s=round(time.time() - t0, 3),
        ts=time.time(),
        frames=rec_frames,
    )


def load_completed(ckpt_path):
    """Return {eid: record} for every complete jsonl line already written."""
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
                # Truncated final line from a hard kill — ignore; that episode reruns.
                continue
            if 'episode_id' in rec:
                done[rec['episode_id']] = rec
    return done


def append_checkpoint(ckpt_path, rec):
    """Atomic-ish append: write one line, flush + fsync so a kill can't lose it."""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')
        f.flush()
        os.fsync(f.fileno())


def rebuild_csv(csv_path, records):
    """Rebuild the 142 CSV table from checkpoint records (always consistent)."""
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
    ap = argparse.ArgumentParser(description="Crash-safe checkpointed native acceptance eval")
    ap.add_argument('--engine', choices=['cpu', 'cuda'], required=True,
                    help="cpu = Windows CPU config (n_gpu_layers=0); "
                         "cuda = WSL GPU config (n_gpu_layers=-1)")
    ap.add_argument('--model-blob', default=None,
                    help="Path to the qwen3:14b GGUF blob (default: platform blob store)")
    ap.add_argument('--n-gpu-layers', type=int, default=None,
                    help="Override offload layers (cuda default -1=all; OOM fallback 40/35/30)")
    ap.add_argument('--ckpt', default=DEFAULT_CKPT, help="Checkpoint jsonl path")
    ap.add_argument('--csv', default=DEFAULT_CSV, help="142-format CSV (rebuilt from ckpt)")
    ap.add_argument('--packets-dir', default=PACKETS_DIR)
    ap.add_argument('--limit', type=int, default=None,
                    help="Smoke-test: process at most N NEW episodes then stop")
    ap.add_argument('--dry-run', action='store_true',
                    help="Exercise skip/append logic with a MOCK model (no llama_cpp / no inference)")
    args = ap.parse_args()

    # Resolve model + offload from engine.
    if args.model_blob:
        model_blob = args.model_blob
    else:
        model_blob = DEFAULT_BLOB_WSL if platform.system() != 'Windows' else DEFAULT_BLOB_WIN
        if args.engine == 'cuda' and platform.system() == 'Windows':
            model_blob = DEFAULT_BLOB_WIN  # cuda selected but on Windows — still Windows path
    model_name = os.path.basename(model_blob)

    if args.n_gpu_layers is not None:
        n_gpu_layers = args.n_gpu_layers
    else:
        n_gpu_layers = -1 if args.engine == 'cuda' else 0

    packet_files = sorted(glob.glob(os.path.join(args.packets_dir, "*.json")))
    if not packet_files:
        print(f"No packet files in {args.packets_dir}", file=sys.stderr)
        sys.exit(1)

    # Resume: load completed episodes, rebuild CSV to match.
    completed = load_completed(args.ckpt)
    rebuild_csv(args.csv, list(completed.values()))
    print(f"[resume] {len(completed)} of {len(packet_files)} episodes already in "
          f"{os.path.relpath(args.ckpt, DOJO_ROOT)}", flush=True)

    system_prompt = (f"Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason."
                     f"\n\nRULES (Genome):\n{load_genome()}")

    # Build the worklist.
    todo = []
    for pkt_path in packet_files:
        eid = os.path.basename(pkt_path).replace('.json', '')
        if eid in completed:
            continue
        todo.append((eid, pkt_path))
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"[plan] engine={args.engine} n_gpu_layers={n_gpu_layers} "
          f"model={model_name}\n[plan] {len(todo)} episodes to run this pass "
          f"(limit={args.limit})", flush=True)

    # Model load (skipped for --dry-run).
    llm = None
    if not args.dry_run:
        from llama_cpp import Llama
        print(f"Loading model n_ctx={NUM_CTX} n_gpu_layers={n_gpu_layers} ...", flush=True)
        t_load = time.time()
        # logits_all=True is REQUIRED: create_completion(logprobs=50) raises ValueError
        # without it (llama.py:1358). CAVEAT for the CUDA/WSL engine: logits_all=True
        # allocates an all-positions logit buffer (n_ctx*n_vocab*4B ≈ 8192*151936*4 ≈ 5.0GB
        # host RAM) at context creation. The default WSL VM (~7.7GB) OOM-crashes on this;
        # give WSL >=12GB via %USERPROFILE%\.wslconfig ([wsl2] memory=12GB) before a CUDA
        # acceptance run. The Windows CPU engine (16GB host) is unaffected. flash_attn on
        # CUDA shrinks the compute buffer (700->307MiB) and speeds prefill.
        llm = Llama(model_path=model_blob, n_gpu_layers=n_gpu_layers, n_ctx=NUM_CTX,
                    n_batch=512, seed=42, temperature=0.0, logits_all=True,
                    flash_attn=(args.engine == 'cuda'))
        print(f"Model loaded in {time.time()-t_load:.1f}s", flush=True)

    n_run = 0
    for eid, pkt_path in todo:
        with open(pkt_path, 'r', encoding='utf-8') as f:
            packet = json.load(f)

        if args.dry_run:
            # MOCK: one clean frame, no inference — proves skip/append plumbing.
            rec = dict(episode_id=eid, engine=args.engine, model=model_name,
                       num_ctx=NUM_CTX, tainted=False, taint_reason=None,
                       exit_frame=None, n_frames_evaluated=1, elapsed_s=0.0,
                       ts=time.time(),
                       frames=[dict(frame_idx=0, p_exit=0.5, logit_exit=-1.0,
                                    logit_hold=-1.0, prompt_tokens=123,
                                    decision="HOLD")])
        else:
            rec = eval_episode(llm, eid, packet, system_prompt, args.engine, model_name)

        append_checkpoint(args.ckpt, rec)
        completed[eid] = rec
        rebuild_csv(args.csv, list(completed.values()))
        n_run += 1
        flag = f"TAINTED({rec['taint_reason']})" if rec['tainted'] else "ok"
        print(f"[{n_run}/{len(todo)}] {eid}: {rec['n_frames_evaluated']} frames "
              f"{rec['elapsed_s']}s {flag}", flush=True)

    print(f"[done] ran {n_run} episodes this pass; "
          f"{len(completed)}/{len(packet_files)} total complete.", flush=True)


if __name__ == '__main__':
    main()
