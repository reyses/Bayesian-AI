---
name: Telescoping TF entry scope
description: Research line - use 1m for entry primitive decisions, 15s/5s/1s as price ticker for exits, macro TFs via 192D context
type: project
---

Entry primitive TF architecture — telescoping scope concept:

1. **Macro scan** (1h/15m/5m): ZigZag identifies developing structure. Captured via 192D context vector.
2. **Setup recognition** (1m): 10-bar lookback matches entry primitive. 1m has enough bars for real swing geometry.
3. **Entry confirmation** (15s): micro price action confirms the 1m setup is playing out.
4. **Exit management** (15s/5s/1s): fast TFs as price ticker for giveback/envelope.

**Why:** 10 bars at 15s = 2.5 min (too short for setup recognition). 10 bars at 1m = 10 min (meaningful behavioral context). The 192D context already carries multi-TF info, so macro structure doesn't need to be in the lookback geometry.

**Implication:** Entry primitives could cluster exclusively at 1m (single pool, not per-TF), with 192D context differentiating macro setups. This would produce denser, more meaningful clusters.

**Status:** Research line — deferred. Current implementation uses per-TF entry clustering. Revisit after first full build + validation.

**How to apply:** When validating entry primitives, check if per-TF clusters at 15s/30s are meaningful or just noise. If they collapse, this is the fix.
