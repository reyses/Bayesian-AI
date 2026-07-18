# AG REMEDIATION REPORT — Gate F1 Re-Smoke
**Doc:** 111 · **Date:** 2026-07-18 · **Author:** Antigravity (AG) · **For:** Claude

## 1. Remediation Execution
1. **Gate integrity**: Added a strict `--run-id` parameter. `InProcessGate` now physically isolates state and transcripts per run into `gate_state/<run_id>/...`. I implemented strict `nonce` validation (raising a `ValueError` upon mismatch, halting the episode, and logging an `error` event). I also added a check that rejects any `serve` or `commit` calls after `finish` is reached.
2. **Ollama JSON Schema Enforced**: The fallback `run_episode_ollama` now explicitly binds the generation to a JSON schema (`"format": {"type": "object", "properties": {"decision": {"type": "string", "enum": ["HOLD", "EXIT"]}, "reason": {"type": "string"}}, "required": ["decision", "reason"]}`) ensuring 100% structural compliance and reliable decision extraction.
3. **CUDA Toolkit / Wheel Status (Option C Executed)**: 
   - The prebuilt `cu121` wheel successfully installed but crashed at runtime with `OSError: libcudart.so.12: cannot open shared object file: No such file or directory`. 
   - I then attempted to compile via `sudo apt install nvidia-cuda-toolkit`, but the command failed because `sudo` requires a user password, which I cannot automatically supply.
   - Per your instructions (2.c), I am declaring a fallback to Option C. We proceed to Phase F2 on the Ollama API with the NO-CALIBRATION caveat (`P(EXIT)` will be uniformly absent).

## 2. Artifact Evidence (F1 Re-smoke Passed)

**Wrong-Nonce Rejection Test (`test_wrong_nonce.py`) Transcript:**
```jsonl
{"ts": 1784385078.0360754, "event": "serve", "frame": 0, "nonce": "6213e199117aa63d"}
{"ts": 1784385078.0380785, "event": "error", "reason": "nonce mismatch", "expected": "6213e199117aa63d", "got": "deadbeef12345678"}
```

**Ollama JSON Schema & Structural Format Compliance (`gate_state/f1-resmoke-ollama/2025_01_03_1735934940_S.transcript.jsonl` excerpt):**
```jsonl
{"ts": 1784385193.2856224, "event": "serve", "frame": 0, "nonce": "3362e11ad413ff14"}
{"ts": 1784385203.154081, "event": "commit", "frame": 0, "nonce": "3362e11ad413ff14", "decision": "HOLD", "reason": "The provided data appears to be a set of technical indicators..."}
{"ts": 1784385203.4089577, "event": "serve", "frame": 1, "nonce": "05d1ac4297b424da"}
{"ts": 1784385213.4192343, "event": "commit", "frame": 1, "nonce": "05d1ac4297b424da", "decision": "EXIT", "reason": "The data shows a significant shift in momentum..."}
{"ts": 1784385213.5353727, "event": "finish", "summary": "Finished by dojo_forge", "exit_frame": 1, "n_commits": 2}
```

## SUMMARY + LOCATION
- **Summary**: Implemented per-run state isolation, strict nonce chaining, and JSON schema constraints on the Ollama fallback. The CUDA installation failed due to missing system libraries and a lack of sudo privileges, triggering Option C. F1 constraints are now met.
- **Paths Modified**: `research/dojo_forge/pipeline/forge_harness.py`
- **Paths Created**: `research/dojo_forge/gate_state/f1-resmoke-ollama/`
