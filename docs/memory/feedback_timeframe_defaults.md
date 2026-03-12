---
name: Stop defaulting to 15s bars
description: Use the right ATLAS timeframe for the job — 15s is not the default for everything. This caused the 4x mismatch bug.
type: feedback
---

## Don't default to 15s bars

ATLAS has 14 timeframes: 1s, 5s, 15s, 30s, 1m, 3m, 5m, 15m, 30m, 1h (and more).
The forward pass iterates 15s bars, but that does NOT mean everything is 15s.

**Bug caused**: Oracle stats (avg_mfe_bar, p75_mfe_bar) are computed from 1m data
in fractal_discovery_agent.py, but the system consumed them as 15s bar counts
everywhere — a 4x mismatch that made anchor patience expire too early, pace
run 4x too fast, envelope decay 4x too aggressive.

**Rule**: Match the timeframe to the task:
- Session-level analysis → aggregate from 1h bars
- Trade overlay on price → use 1m or 5m, not 15s
- Oracle/template stats → computed from 1m (discovery TF)
- Forward pass iteration → 15s (execution TF)
- New tools/analysis → ask "what TF makes sense?" don't assume 15s
