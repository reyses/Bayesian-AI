# REVIEWER VERDICT — Gate F1: PASS (conditional) · F2 UNBLOCKED · CUDA fix included
**Doc:** 112 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **For:** AG

## 1. Verification (reviewer-reproduced, not taken from the report)
- Re-smoke transcript `gate_state/f1-resmoke-ollama/2025_01_03_..._S`:
  **chain clean = True** by my own checker (every commit nonce == last served).
- Gate code enforces post-finish rejection (ValueError, forge_harness.py:69/84);
  wrong-nonce error event artifact present as pasted.
- JSON-schema binding on the ollama path = structural compliance. Accepted.
- Evidence style this round (pasted artifact lines): exactly right. Keep it.

## 2. VERDICT
**Gate F1: PASS — conditional on the fallback path.** F2 is UNBLOCKED on
ollama+json-schema with the NO-CALIBRATION caveat (no P(EXIT), no prefix
timings) attached to every F2 report until llama.cpp lands.

## 3. The CUDA fix — NO SUDO NEEDED (do this as a parallel task during F2)
Your cu121 wheel failed only because the loader can't see libcudart. The venv
ALREADY CONTAINS it (torch's bundled runtime — reviewer-verified):
`/home/reyses/venvs/bayesian-ai/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12`
Recipe:
```bash
V=/home/reyses/venvs/bayesian-ai/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$V/cuda_runtime/lib:$V/cublas/lib:$V/cudnn/lib:$LD_LIBRARY_PATH
python -c "from llama_cpp import Llama; print('llama_cpp OK')"
```
If additional libs are reported missing, they are almost certainly under
$V/<pkg>/lib as well — append them the same way. Bake the export into the
harness launcher (a named LAUNCHER script, not a magic env in your head).
**Requirement**: llama.cpp path must be LIVE (with the F1.2/F1.4 evidence —
prefix cold/warm table + P(EXIT) series on 2 episodes) BEFORE gen-1 plays,
so calibration exists from gen-1 onward. gen-0 may run on the fallback.

## 4. F2 reminders
Fresh ride-side days only (exclusion set from the three prior selections);
50-episode held-out NEVER played; genome seeded from docs 098/100/107
material; NO self-distillation — failure tapes come back to the reviewer.
Report = doc 113 with artifact lines.
