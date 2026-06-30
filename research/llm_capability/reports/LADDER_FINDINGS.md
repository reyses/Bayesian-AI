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

---

# Pass 2 — constrained decoding (`format` schema) + `think:false`

Two confounds removed: (1) free-form JSON → constrained decoder (invalid JSON impossible);
(2) **gemma4 is a reasoning model** — its hidden chain-of-thought (`message.thinking`) ate the
token budget, leaving `content` empty (mis-scored as format failure). `think:false` → straight to answer.

| rung | metric | Pass 1 | Pass 2 | chance |
|---|---|---|---|---|
| regime | direction acc | 30% | **53%** | 33% |
| regime | variation acc | 23% | **47%** | 50% |
| regime | joint | 13% | 27% | ~17% |
| vision | archetype acc | 0% | **30%** | ~14% |
| vision | "read the chart" | 0% | **100%** | — |
| apex | BAD/unparseable | 100% | **0%** | — |
| all | JSON-format failure | 47–100% | **0%** | — |
| all | latency/call | 2.4–10s | **0.7–2.0s** | — |

## What's real after the fix
1. **Format + latency were the dominant Pass-1 confounds**, both caused by uncaptured thinking.
   With `think:false`+schema: 0 format failures everywhere, 0.7–2.0s/call.
2. **Regime labeling: weak partial reasoning.** Direction 53% (above 33% chance — net_move sign is
   an easy tell); variation 47% (at coin-flip — can't separate smooth/choppy). Reads the easy axis only.
3. **Vision: weak but non-zero.** 30% archetype (above ~14% chance), and 100% now reference a
   direction — gemma4:e2b *does* perceive something; it just classifies poorly. Not unusable, just weak.
4. **Apex decider: chooses HOLD on 100% of bars.** 0 trades, $0. This is now a real BEHAVIORAL
   finding, not a format artifact: given valid outputs, the model defaults to total inaction — it will
   not commit to a directional bet from this prompt. The decider doesn't decide.

## Verdict
- **Code boilerplate**: usable (3/4 compile, obeys house rules).
- **Numeric labeling**: only the easy axis beats chance; not reliable.
- **Chart vision**: weak (30%), but it does engage — usable only as a coarse triage signal at best.
- **Causal decider**: defaults to HOLD-everything; combined with latency/non-determinism, stays
  GRAVEYARD for live. Open question for a finale: force it to take a position each bar and measure
  whether it churns-and-loses (expected, since bar-level direction << day-level 53%).
