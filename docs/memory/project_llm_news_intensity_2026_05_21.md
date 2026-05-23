---
name: LLM News-Intensity Feature for DRS (Phase A scaffold)
description: Dev scaffold built 2026-05-21 to score FOMC/CPI/NFP press releases via local Llama-3.1-8B and inject the score into DRS as a feature; nothing run yet; canonical pipeline byte-identical until validated.
type: project
originSessionId: 2111be4b-5103-4d2d-a1e0-249c678bdcde
---
**Fact.** 2026-05-21 late session: built a fully isolated dev scaffold for an
LLM-scored news-intensity feature feeding DRS. **Nothing has been executed.**
Production paths are byte-identical to before this session.

**Why.** 2026-05-18 DRS canonical verdict: OOS sealed Pearson +0.139 CI
[-0.047, +0.451] — lower bound crosses zero, naive sizing loses -$333/day
CI [-$523, -$163]. The binary event flags (`is_fomc/cpi/nfp/opex`) each
contribute $0 of permutation ΔMAE — they tell DRS an event happened but
not whether it was hawkish/dovish/on-consensus/shock. The continuous LLM
score is meant to subsume them and push OOS lower CI strictly positive.
The verdict itself flagged "LLM-scored news headline intensity" as the
next research direction.

**How to apply.**

1. **Do not touch canonical paths during dev.** The user's hard requirement.
   `tools/sourcing/build_cross_day_features.py`, `drs_canonical_gbm.py`,
   `DATA/CROSS_DAY/cross_day_features{,_with_target}.parquet`,
   `drs_canonical_gbm.pkl`, `forward_pass_full_stack.py`, B10 day-multiplier
   — all unchanged. Everything new lives in:
   - `tools/sourcing/llm_news/` (self-contained module)
   - `tools/sourcing/build_cross_day_features_v2.py` (augmenter, NOT a copy)
   - `tools/sourcing/drs_canonical_gbm_v2.py` (mirror of canonical, dev I/O)
   - `DATA/CROSS_DAY/dev/` (all outputs)
   - `research/llm_news_intensity/` (DMAIC + PDCA + findings)

2. **Execution order (user runs these).**
   - `huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
     Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models`
     → rename to `models/llama-3.1-8b-instruct-q4_k_m.gguf`
   - `pip install llama-cpp-python[cuda]` (or pre-built wheel if build fails)
   - `python -m tools.sourcing.llm_news.cli fetch`
   - `python -m tools.sourcing.llm_news.cli test-synthetic`  ← MUST PASS
   - `python -m tools.sourcing.llm_news.cli score`
   - `python tools/sourcing/build_cross_day_features_v2.py`
   - `python tools/sourcing/drs_canonical_gbm_v2.py`
   - Read `research/llm_news_intensity/findings/{date}_phase_a_results.md`

3. **Phase A gate (single-shot, all 3 required).**
   - IS WF Pearson lower CI ≥ +0.098 (baseline lower bound — must not regress)
   - OOS sealed Pearson lower CI > 0 (was -0.047)
   - `news_intensity_today` ΔMAE ≥ +$30/day (beats `vix_close_prior` at +$17/day)

4. **Decision tree on gate fail.**
   - IS lifts, OOS doesn't → LLM memorized 2024-2025 statements. Kill.
   - IS doesn't lift → prompt eng issue or 8B too small. Iterate prompt
     OR upgrade to Qwen2.5-14B Q4 (~9GB VRAM) or Mistral-Small-24B Q4 (~14GB).
   - ΔMAE under threshold → signal exists but too weak; kill.
   - Score std < 1.5 → don't even retrain DRS; prompt eng failed.

5. **Phase B (cycle_02.md) is only created if Phase A passes.** Adds
   `news_intensity_prior` (yesterday's PM releases, 14:00 ET FOMC etc).
   Bootstrap CI on delta_Pearson(A+B vs A only) must exclude 0.

6. **Promotion to main is a SEPARATE user-approved session.** Includes a
   B10 regression check (re-run forward_pass_full_stack.py IS+OOS; OOS
   $/day delta must be within ±$5/day noise floor). Plan section
   "Promotion to main" has the 10-step procedure.

**Critical files for next session to read first**:
- `~/.claude/plans/i-would-like-to-jaunty-spindle.md` (approved plan)
- `research/llm_news_intensity/project.md` (DMAIC frame)
- `research/llm_news_intensity/cycle_01.md` (PDCA — predictions written
  pre-run, do not edit retroactively per MEMORY PDCA rule)

**Backout cost**: delete `tools/sourcing/llm_news/`, both `_v2.py`,
`DATA/CROSS_DAY/dev/`, `research/llm_news_intensity/`, the `models/` entry
in `.gitignore` (line just added), `tools/sourcing/__init__.py`. Zero
production regression possible.

**Pushback rationale (worth remembering for future LLM-in-system asks)**:
- In-the-loop LLM trade veto: REJECTED. Latency, non-determinism, kills
  CI/bootstrap discipline.
- Babysit / monitoring LLM: marginal value over halt-after-N + DRS + B10 +
  blowout sim + parity checks already in place.
- News intensity scorer: the ONE LLM use case with a clean $/day proof
  path (goes in DRS as a feature, bootstrappable, OOS-validatable).
- LLM-as-feature is the right paradigm; LLM-as-decider is the wrong one.
