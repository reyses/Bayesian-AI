#!/usr/bin/env python3
"""teacher_speedup — shared bench machinery (ISOLATED; imports dojo, never edits it).

Everything the E1/E2 harnesses share:
  * load the qwen3:14b teacher with params IDENTICAL to eval_native_tiered defaults
    (n_batch 256, flash_attn on cuda, seed 42, temperature 0, n_gpu_layers -1 via
    the dojo VRAM preflight) — so any timing/equivalence result transfers to prod.
  * build the SAME per-frame tiered prompt the acceptance harness builds, by
    IMPORTING build_user_content / filter_hist from research/dojo_forge/pipeline
    (sys.path insert; no copy of the logic — the dojo file stays the single source).
  * wall-clock timing helpers (tok/s, s/frame).
  * output-hash helpers: hash the exact generated text (E1) and the exact
    last-position logit readout bytes (E2) so equivalence-vs-baseline is a
    byte/float comparison, not eyeballing.

Bench set (FIXED, documented): the first 3 packets alphabetically from
research/dojo_forge/reports/gen0/packets/ :
    2025_01_21_1737469980_S   (SHORT, 20 frames)
    2025_01_21_1737471780_S   (SHORT)
    2025_01_21_1737475200_L   (LONG)
Frame selection for E1 (5 "reasoned" frames/episode): evenly spaced across the
episode at fractions BENCH_FRAME_FRACTIONS of (n_frames-1), clamped to >=1 and
de-duplicated (frame 0 is the wide-field anchor, excluded — the tiered builder
requires i>=1). Prior-frame decisions in the tiered context are pinned to a fixed
"HOLD" trail so the prompt is a pure function of the packet (reproducible hashes),
independent of what any model actually decided.
"""
import hashlib
import os
import sys
import time

import numpy as np

# --- wire in the dojo pipeline (IMPORT ONLY — never modify anything under it) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_TSPEEDUP = os.path.dirname(_HERE)                      # research/teacher_speedup
_DOJO = os.path.join(os.path.dirname(_TSPEEDUP), 'dojo_forge')
_DOJO_PIPELINE = os.path.join(_DOJO, 'pipeline')
if _DOJO_PIPELINE not in sys.path:
    sys.path.insert(0, _DOJO_PIPELINE)

import eval_native_ckpt as base                          # noqa: E402  (loader/logits/genome)
from eval_native_tiered import build_user_content, filter_hist, HIST_MIN  # noqa: E402,F401

MODELS_DIR = os.path.join(_TSPEEDUP, 'models')           # gitignored (*.gguf)
REPORTS_DIR = os.path.join(_TSPEEDUP, 'reports')

# ------------------------------------------------------------------ bench set --
BENCH_EPISODES = [
    '2025_01_21_1737469980_S',
    '2025_01_21_1737471780_S',
    '2025_01_21_1737475200_L',
]
BENCH_FRAME_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)   # -> 5 reasoned frames/ep

# Exam-style framing for E1 free generation (mirrors tools/exam_day.py intent:
# reason under <think>, then emit a compact decision). Kept local so we do NOT
# import exam_day.py (whose module body is main-guarded but which we keep at arm's
# length as an experiment). Generation-speed is prompt-content sensitive, so a
# realistic reasoned prompt is what E1 must time.
EXAM_SYSTEM_GEN = (
    "You are trading one episode of a real historical trade replay, minute by "
    "minute, deciding EXIT vs HOLD. Reason briefly inside a <think> block, then "
    "answer EXACTLY:\nDECISION: HOLD|EXIT\nCONFIDENCE: <0.00-1.00>\nREASON: <=60 "
    "words citing the specific Genome rule(s) and market features you used.\n\n"
    "RULES (Genome):\n"
)
GEN_USER_SUFFIX = "\n\nYour decision for THIS minute (reason, then the exam format):"


def load_packet(eid):
    import json
    path = os.path.join(base.PACKETS_DIR, f'{eid}.json')
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def bench_frame_indices(n_frames):
    """Deterministic 5 evenly-spaced reasoned-frame indices for an episode."""
    idxs = []
    for f in BENCH_FRAME_FRACTIONS:
        i = int(round(f * (n_frames - 1)))
        i = max(1, min(i, n_frames - 1))
        idxs.append(i)
    # de-dupe while preserving order, then keep it sorted for stable reporting
    return sorted(dict.fromkeys(idxs))


def pinned_decisions(n_frames):
    """A fixed HOLD trail so the tiered prompt is a pure function of the packet."""
    return ["HOLD"] * n_frames


def tiered_user_content(frames, i):
    """The EXACT tiered context the acceptance harness builds for frame i>=1."""
    return build_user_content(frames, i, pinned_decisions(len(frames)))


# ---------------------------------------------------------------- prompt build --
def gen_prompt(frames, i, system_gen):
    """Exam-style FREE-GENERATION prompt for frame i (assistant turn left open)."""
    content = tiered_user_content(frames, i)
    return (f"<|im_start|>system\n{system_gen}<|im_end|>\n"
            f"<|im_start|>user\n{content}{GEN_USER_SUFFIX}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def logit_prompt(frames, i, system_prompt):
    """Forced-</think> readout prompt for frame i (matches eval_native_tiered)."""
    content = frames[0]['text'] if i == 0 else tiered_user_content(frames, i)
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{content}<|im_end|>\n{base.THINK_SUFFIX}")


# --------------------------------------------------------------------- loader --
def load_teacher(n_ctx, engine='cuda', draft_model=None, n_gpu_layers=None,
                 verbose=False):
    """Load qwen3:14b with eval_native_tiered's IDENTICAL params.

    n_batch=256, seed=42, temperature=0, flash_attn=(cuda), logits_all=False
    (the binding forces it True whenever draft_model is set — that is a memory
    caveat we surface in the reports, not something we can toggle off).
    n_gpu_layers: -1 -> full offload, fitted to free VRAM by the dojo preflight.
    """
    from llama_cpp import Llama
    model_blob = base.DEFAULT_BLOB_LINUX
    if not os.path.exists(model_blob):
        model_blob = base.DEFAULT_BLOB_WSL
    ngl = n_gpu_layers if n_gpu_layers is not None else (-1 if engine == 'cuda' else 0)
    if engine == 'cuda':
        base.NUM_CTX = n_ctx                                   # keep preflight math honest
        ngl = base.preflight_vram(ngl, n_ctx)
    t0 = time.perf_counter()
    # Draft-model decoding writes full-batch logits; with logits_all=False the
    # scores buffer allocates 0 rows -> "broadcast (n_batch*vocab,) into (0,)"
    # crash (hit in E1 arm b, 2026-07-24). logits_all must follow draft usage.
    llm = Llama(model_path=model_blob, n_gpu_layers=ngl, n_ctx=n_ctx,
                n_batch=base.N_BATCH, seed=42, temperature=0.0,
                logits_all=(draft_model is not None),
                flash_attn=(engine == 'cuda'), draft_model=draft_model,
                verbose=verbose)
    load_s = time.perf_counter() - t0
    return llm, load_s, os.path.basename(model_blob)


def system_prompt_readout():
    """The acceptance-harness system prompt (for E2 logit readout)."""
    return (f"Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason."
            f"\n\nRULES (Genome):\n{base.load_genome()}")


# --------------------------------------------------------------------- timing --
class Stopwatch:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0


def tok_per_s(n_tokens, seconds):
    return (n_tokens / seconds) if seconds > 0 else float('nan')


# --------------------------------------------------------------------- hashing --
def hash_text(text):
    """SHA-256 of exact generated text — temp-0 equivalence must be identical."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def hash_logits(arr, decimals=6):
    """SHA-256 of the rounded logit row — a coarse fingerprint for record-keeping.
    (Bitwise equality of fp reductions is not guaranteed across code paths; the
    load-bearing E2 check is max|delta|, this hash is just a compact receipt.)"""
    a = np.round(np.asarray(arr, dtype=np.float64), decimals)
    return hashlib.sha256(a.tobytes()).hexdigest()


def max_abs_delta(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))
