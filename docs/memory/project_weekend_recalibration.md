---
name: weekend recalibration loop
description: Future concept — weekly oracle recalibration from live trades (export → oracle → retune → deploy Monday)
type: project
---

Weekend recalibration loop (NOT yet implemented — future roadmap item):
1. Friday close: export week's live trades via nt8_to_parquet
2. Saturday: run oracle on completed week (forward-looking labels now available)
3. Oracle reveals: which templates worked, direction accuracy, exit quality
4. Recalibrate: brain updates, gate thresholds, giveback/trail params
5. Monday open: deploy recalibrated system

**Why:** Walk-forward validation applied to live. Train on days 1→N-1, trade day N.
Weekly cycle gives oracle access to completed data without lookahead.

**How to apply:** Don't build this until live trading is stable. Prerequisite:
unified BarProcessor (OOS = live code), stable quantum scoring, proven OOS edge.
When ready, create `tools/weekend_recalibration.py` pipeline script.
