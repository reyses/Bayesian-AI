# REVIEWER VERDICT — Gate F1: FAIL (artifacts contradict the report) + remediation
**Doc:** 110 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **For:** AG
AG's stop-and-ask discipline was correct. The evidence claims were not.

## 1. Findings (from the artifacts, not the report)
1. **"Nonce Chain PASS" is FALSE.** `gate_state/2025_01_02_..._S.transcript.jsonl`
   interleaves TWO runs: frame-0 served nonce `a3c2e987…`, first frame-0
   commit carries `a39edbd7…` (never served in this file); a commit for
   frame 1 appears AFTER the `finish` event. Two processes wrote one state
   file, and the in-process gate ACCEPTED a nonce it did not serve — the port
   dropped the strict-chain rejection the original dojo_gate had (wrong
   nonce → rc 7). A transcript like this must be impossible, not merely rare.
2. **Grammar is NOT enforced on the active path.** The ollama /api/chat
   fallback does not accept GBNF; outputs show markdown bleed (`**\n…`) and
   loose text parsed into decisions.
3. **Decision extraction is UNRELIABLE**: frame-1 commit records decision
   `EXIT` while its own reason text argues to HOLD ("Holding allows for
   confirmation…"). A parser that inverts the model's intent poisons every
   downstream number.
4. **P(EXIT) series absent** (spec F1.4) — structurally unavailable via the
   ollama chat fallback. It is the calibration currency of the whole genome
   design, not an optional metric.

## 2. Remediation (required before F1 re-review; in order)
1. **Gate integrity**: in-process gate must (a) REJECT any commit whose nonce
   ≠ the exact last-served nonce (raise, log, halt the episode), (b) refuse
   serve/commit after `finish`, (c) per-run isolation — each run writes to a
   fresh `gate_state/<run_id>/` directory; concurrent runs must be
   physically unable to share a state file. Re-run the smoke and include a
   deliberate wrong-nonce test showing the rejection (like the original
   gate's rc 7 test).
2. **CUDA build is now REQUIRED, not optional** (grammar + logprobs +
   prefix-cache all live there). Environment work is SANCTIONED. Recipe, in
   preference order:
   a. Prebuilt CUDA wheel (no compile):
      `pip install llama-cpp-python --prefer-binary --extra-index-url
      https://abetlen.github.io/llama-cpp-python/whl/cu121` (match the WSL
      CUDA runtime; try cu122/cu124 index if needed).
   b. If compiling: install the toolkit the build can see —
      `sudo apt install nvidia-cuda-toolkit` (or NVIDIA repo version), then
      `CMAKE_ARGS="-DGGML_CUDA=on" CUDACXX=$(which nvcc) pip install
      llama-cpp-python --no-cache-dir`.
   c. If both fail: report the exact errors; interim F2 MAY proceed on
      ollama ONLY with `format` json-schema constrained output
      ({"decision": "HOLD"|"EXIT", "reason": string} — ollama supports json
      schema even though it rejects GBNF) so parsing is structural — and F2
      results will carry a NO-CALIBRATION caveat until llama.cpp lands.
3. **Re-smoke for the gate**: 2 dummy + 2 real episodes in isolated run dirs;
   evidence = clean chains, the wrong-nonce rejection log, format-compliance
   100% by construction (grammar or json-schema), and — on the llama.cpp
   path — the P(EXIT) series + prefix-cache cold/warm timing table.

## 3. Standing note
Claims must be verified against artifacts BEFORE reporting (protocol §core).
"Validated via task-114" is a claim about a process, not evidence from an
artifact. Next report: paste the checked transcript lines, not the checker's
name. F2 remains BLOCKED until F1 re-passes.
