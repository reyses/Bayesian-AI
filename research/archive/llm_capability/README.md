# research/llm_capability — Local-LLM Capability Ladder

**Question:** what can the local LLM (gemma4 via Ollama — the engine OpenClaw wraps) actually
do *for this project*, and where does it break? We probe it on a **progressive complexity
ladder**, scored against ground truth, all outputs written to `reports/`.

**Why not drive OpenClaw directly?** The OpenClaw gateway (`:18789`) wasn't running, there's no
CLI on PATH, and its config points at an un-pulled `llama3.3`. OpenClaw is an orchestration shell
— *its capability ceiling is the model's ceiling*. So we probe the model directly via the Ollama
HTTP API. Swap in OpenClaw later by pointing it at `gemma4:latest` (see `openclaw/HOW_TO_COMMAND_OPENCLAW.md`).

**Scope rail:** everything is confined to this folder; the model only reads data we hand it and
writes nothing — *we* write the outputs. Working tree is committed+pushed before each run (recovery point).

## The ladder (easy → hard)
| rung | file | task | ground truth | metric |
|---|---|---|---|---|
| 1 | `tools/rung1_label_regimes.py` | label day regime (2D taxonomy) from stats | `DATA/ATLAS/regime_labels_2d.csv` (OOS split) | direction/variation/joint accuracy |
| 2 | `tools/rung2_repetitive_code.py` | generate boilerplate under no-magic-numbers rule | `compile()` + magic-number heuristic | compiles % / rule-obeyed % |
| 3 | `tools/rung3_vision_chart.py` | classify a trade chart image | filename labels in `research/edge_case_triage/reports/` | archetype accuracy |
| **apex** | `pipeline/causal_trader.py` | **trade a real day bar-by-bar, ZERO lookahead** | realized PnL + behavior | PnL, PF-WR, BAD%, latency/bar |

## Layout
- `tools/ollama_client.py` — shared client (temp=0, timing, JSON-extraction with measured failure rate).
- `tools/rung{1,2,3}_*.py` — the three perception/reasoning rungs.
- `pipeline/causal_trader.py` — the apex causal decider. **Zero-lookahead firewall is asserted in `_build_prompt`** (window ends at the decision bar; fills happen next-bar-open).
- `reports/` — one `.md` per rung + `causal_decisions_*.jsonl` (full decision log).

## Run (from repo root)
```bash
python research/llm_capability/tools/rung1_label_regimes.py --n 50
python research/llm_capability/tools/rung2_repetitive_code.py
python research/llm_capability/tools/rung3_vision_chart.py --n 12
python research/llm_capability/pipeline/causal_trader.py --day 2024_02_20 --max_bars 120
```

## Standing finding (fill in as runs land)
LLM-as-decider is already graveyard-flagged (latency / non-determinism / no CI discipline). The
apex run quantifies *how* it fails at zero lookahead — it is a limitation study, not a strategy.
