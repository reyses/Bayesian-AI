# Local-LLM Capability Ladder — Findings (Pass 1: free-form JSON prompting)

Model: `gemma4:latest` (8B) for text/code/trading, `gemma4:e2b` (5.1B) for vision. Ollama, temp=0,
RTX 3060 12GB. All ground-truthed; scope confined to `research/llm_capability/`.

## Results (pass 1)
| rung | task | accuracy | JSON-format failure | latency/call |
|---|---|---|---|---|
| 2 code | boilerplate, no-magic-numbers | **3/4 compile, 3/3 clean** | (n/a, code block) | 10.1s |
| 1 regime | 2D regime from stats vs manual labels | dir 30% (chance 33%), var 23% (chance 50%), joint 13% | **47%** | 4.4s |
| 3 vision | classify trade chart PNG | **0/10**; never named a direction | **100%** | 6.5s |
| apex | causal trade, 0 lookahead, 90 bars | 0 trades, $0 (forced HOLD) | **100% BAD output** | 2.4s |

## What is real
1. **Code-gen is the one competency.** Bounded, well-specified boilerplate compiles and respects the
   house no-magic-numbers rule. Only miss = token-budget truncation of a docstring.
2. **Latency disqualifies the decider role.** 2.4–10s/call; a 390-bar RTH day ≈ 16 min of inference.
   Confirms the existing graveyard verdict on LLM-as-decider — now quantified.
3. **Free-form JSON prompting collapses.** 47% (labeling) → 100% (vision, apex) unparseable replies.
   This is an ENGINEERING blocker, not (yet) an intelligence verdict.

## The confound (do not over-claim)
The apex's "$0 / 0 trades / 100% BAD" is an **artifact of the format gate** — the model never produced
a valid action, so we never tested trading *reasoning*. Likewise the sub-chance labeling numbers are
inflated by format failures counting as wrong. Pass 1 measures **format compliance**, not reasoning.

## Fix → Pass 2
Use **Ollama constrained decoding** (`format` = JSON schema in the request) to force valid JSON at the
decoder. This isolates reasoning from formatting. Expected: labeling improves (reasoning becomes
measurable), apex finally emits real actions (then we see if it churns / loses), vision likely stays
broken (perception, not formatting). Vision may also need a non-`e2b` multimodal model.

## Verdict so far
For this project the local LLM is a **code-boilerplate assistant**, not a labeler, chart reader, or
decider. Pass 2 (constrained output) will tell us whether the labeling/decider failures are *formatting*
or *reasoning*. Either way, LLM-as-decider stays graveyard for live (latency alone).
