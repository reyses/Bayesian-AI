# tools/ — Inventory

Total: 106 Python tools. **Physical files live in this directory** (flat, for clean imports).
Category folders (`_<name>/`) contain **README indexes only** — browse there for categorized lists.

## Categories (each has a README)

| Folder | Topic | Approximate count |
|---|---|---:|
| [`_data/`](_data/README.md) | Atlas build, NT8/Databento imports, validation | 13 |
| [`_labeling/`](_labeling/README.md) | Human + auto trade/peak/level labeling | 15 |
| [`_rm_pivot/`](_rm_pivot/README.md) | Current session — RM pivot research | 17 |
| [`_pivot/`](_pivot/README.md) | Pivot forward-pass variants (physics/residual/regression) | 11 |
| [`_physics/`](_physics/README.md) | Nightmare + physics engines | 12 |
| [`_engines/`](_engines/README.md) | Blended / KillShot / Overshoot / Saturation | 9 |
| [`_tier/`](_tier/README.md) | Tier-specific EDA, runners, classifiers | 15 |
| [`_regret/`](_regret/README.md) | Post-run PnL + regret + pathology | 15 |
| [`_peak/`](_peak/README.md) | Peak research, prediction, templates | 10 |
| [`_levels/`](_levels/README.md) | Level / zone analysis | 5 |
| [`_mtf/`](_mtf/README.md) | Multi-TF & resonance | 4 |
| [`_eda/`](_eda/README.md) | Feature / physics EDA | 18 |
| [`_charts/`](_charts/README.md) | Visualizations | 8 |
| [`_nn/`](_nn/README.md) | Neural network training | 7 |
| [`_parity/`](_parity/README.md) | Lookahead / parity validation | 6 |
| [`_util/`](_util/README.md) | Checkpoints, golden path, risk, discovery | 12 |

## Also

- **Full cross-referenced inventory**: [`../research/TOOLS_INDEX.md`](../research/TOOLS_INDEX.md) — all 106 tools in one searchable document
- **Per-tool docstrings**: every `.py` file has a docstring at top explaining purpose + usage; use `head -N tools/<name>.py` to read

## Rule

When adding a new tool: also add it to (a) `research/TOOLS_INDEX.md`, and (b) the appropriate category README here.
