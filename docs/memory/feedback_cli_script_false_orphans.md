---
name: feedback-cli-script-false-orphans
description: Standalone CLI scripts ("python path/to/x.py …") do NOT show up in Python-import grep but ARE active dependencies. Always grep for the bare filename in docs/, Jules_instructions/, config/, and as `subprocess.run([…])` literals before deleting.
metadata:
  type: feedback
---

**Rule**: Before deleting any `.py` file flagged as "no Python imports," verify it isn't a standalone CLI script by checking ALL of these:

1. `python <path/file.py>` or `python -m <module>` invocations in `docs/`, `Jules_instructions/`, `config/`, and tool docstrings.
2. `subprocess.run([…, '<path>', …])` or `[sys.executable, '<path>', …]` patterns in live code.
3. Inside-function imports (`from X.Y import Z` inside `def …`) — grep tools don't find them in cross-file dependency maps.
4. `--help` or `--example` references in README/spec docs.

**Why:** This failure mode hit four times in one cleanup session (2026-05-24):
- `training/ticker.py` — used by `run.py:158` (inside-function `from training.ticker import FileTicker`).
- `training/report.py` — used by `run.py:708,1149` (inside-function imports).
- `training/build_dataset_v2.py` — invoked via `python …` from `docs/JULES_standalone_research_v2.md:14` and other docs; no Python importers.
- `training/train_pivot_cnn_v2.py` — invoked via `python …`; sole "caller" was the docstring `Usage:` block.

Each restoration cost ~5 minutes. Doing the four-check audit BEFORE deletion would have cost ~2 minutes total. Net loss: ~13 minutes plus the user's confidence.

**How to apply:**
- After running a dependency-map agent, run a SECOND pass that greps the bare filename (without path) across `docs/`, `config/`, `Jules_instructions/`, and live code's `subprocess` calls.
- For any file you're about to delete, do the same grep yourself — don't rely solely on the agent.
- When a file has a `Usage:` block in its docstring with `python <path>` lines, that's a tell: it's a CLI script and likely has docs/config callers the import-grep missed.
- The `_v2`/`_v1` suffix is ALSO a tell: these are usually parallel CLI scripts targeting different feature schemas, not unused junk.

**Related**: see `feedback_v2_only_hard_rule.md` (the V2-only rule that made over-zealous deletion appealing in the first place) and `docs/daily/2026-05-24.md` section 9 (the build_dataset_v2 restore record).
