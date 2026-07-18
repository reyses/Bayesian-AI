---
name: distilled-llm_capability
description: Local LLM (gemma4 via Ollama) probed on a 4-rung capability ladder — code-gen usable, labeling/vision weak, causal decider defaults to HOLD-everything; confirms graveyard verdict on LLM-as-decider.
metadata: {type: distilled, topic: llm_capability, status: concluded}
---
## Verdict
Asked what the local LLM (gemma4:latest 8B / gemma4:e2b 5.1B, Ollama, temp=0) can
do for this project: label day regimes, write boilerplate code, classify trade
charts, and trade a real day bar-by-bar with zero lookahead. Pass 1 (free-form
JSON) was confounded by format failures and uncaptured chain-of-thought; Pass 2
(constrained decoding + `think:false`) isolated reasoning from formatting.
Result: code-gen is the one real competency; labeling/vision beat chance but
weakly; the causal decider chooses HOLD on 100% of bars — a real behavioral
finding, not an artifact. Concluded — confirms existing graveyard verdict.

## Key numbers (with CIs where they exist)
- Rung 2 (code-gen): 3/4 compile, 3/3 of those obey no-magic-numbers, latency 10.1s/call (`reports/rung2_repetitive_code.md`).
- Rung 1 (regime label, Pass 2, n=30 OOS): direction 53% (16/30, chance ~33%), variation 47% (14/30, chance ~50%), joint 27% (8/30), latency 0.7s/call (`reports/rung1_label_regimes.md`).
- Rung 3 (vision, Pass 2, n=10): archetype accuracy 30% (3/10, chance ~14%), "reads the chart" (mentions a direction) 10/10=100%, latency 2.0s/call (`reports/rung3_vision_chart.md`).
- Apex (causal trader, 2024_02_20, 90 decision bars, warmup 20): $+0.00 PnL, 0 trades, PF-based Trade WR +0.000, action counts {LONG:0, SHORT:0, CLOSE:0, HOLD:90, _BAD:0}, BAD/unparseable 0/90=0%, latency 0.99s/bar mean (`reports/apex_causal_trading_2024_02_20.md`).
- Pass 1 vs Pass 2 summary (`reports/LADDER_FINDINGS.md`): JSON-format failure 47-100% → 0%; latency 2.4-10s → 0.7-2.0s/call; apex BAD rate 100% → 0%.
- `apex_why_diagnostic.md`: on the largest trailing-10min-move bars (clear trends, position FLAT), the model still leans HOLD/hesitate in its own chain-of-thought — inaction is not chop-specific.

## Graveyard / never-retry
- **Causal LLM decider stays GRAVEYARD for live.** Given valid constrained outputs (format confound removed), it defaults to HOLD on 100% of bars (0 trades, $0) — "the decider doesn't decide." Combined with latency (0.7-2.0s/call, `reports/LADDER_FINDINGS.md`) and non-determinism, confirms the pre-existing MEMORY.md graveyard entry ("LLM-as-decider REJECTED").
- Free-form JSON prompting (Pass 1) is not a viable protocol for this model — 47-100% unparseable output; must use constrained decoding + `think:false`.

## Reusable assets
- `tools/ollama_client.py` — shared Ollama HTTP client (temp=0, timing, JSON-extraction, measured failure rate).
- `tools/rung1_label_regimes.py`, `tools/rung2_repetitive_code.py`, `tools/rung3_vision_chart.py` — the three perception/reasoning rungs, runnable standalone.
- `pipeline/causal_trader.py` — zero-lookahead causal decider harness (`_build_prompt` asserts the window ends at the decision bar; fills happen next-bar-open); reusable scaffold for any future LLM-in-the-loop probe.
- `tools/why_diagnostic.py` — chain-of-thought diagnostic on picked trending bars.

## Data locations
- `DATA/ATLAS/regime_labels_2d.csv` (OOS split) — ground truth for rung 1.
- `research/edge_case_triage/reports/` — chart PNGs + filename labels, ground truth for rung 3.
- `research/llm_capability/reports/causal_decisions_2024_02_20.jsonl` — full apex decision log.

## Open threads
- Finale idea floated in `LADDER_FINDINGS.md`: force the apex decider to take a position every bar and measure whether it churns-and-loses (expected, since bar-level direction accuracy is far below the day-level 53%) — not yet run.
- Vision may need a non-`e2b` multimodal model to move past 30% (untested).

## Sources
- research/llm_capability/README.md
- research/llm_capability/reports/LADDER_FINDINGS.md
- research/llm_capability/reports/apex_causal_trading_2024_02_20.md
- research/llm_capability/reports/apex_why_diagnostic.md
- research/llm_capability/reports/rung1_label_regimes.md
- research/llm_capability/reports/rung2_repetitive_code.md
- research/llm_capability/reports/rung3_vision_chart.md

## Archive recommendation
ARCHIVE (concluded limitation study; standing finding closes the loop — LLM-as-decider confirmed dead, code-gen/labeling/vision findings are reference-only, no further runs planned besides an optional untaken finale).
