---
name: project-atlas-keep-policy
description: DATA/ATLAS retention rule set by Moises 2026-07-16 — keep only OHLCV price parquets + SFE feature stores; everything else regenerate-on-need
metadata: 
  node_type: memory
  type: project
  originSessionId: 49f1ab8b-f170-41ec-955f-86beb538417f
---

**ATLAS keep-policy (Moises, 2026-07-16, after the C:-drive 0-bytes-free incident):**
`DATA/ATLAS` holds ONLY (a) OHLCV price parquets, all timeframes 1s→1D, and (b) the
SFE-output feature stores `FEATURES_1s_v2` and `FEATURES_5s_v2`. Everything else
(experiment feature builds, DOE extraction outputs, derived baselines) is
regenerate-on-need and must NOT accumulate in ATLAS.

**Why:** the disk hit 0 bytes free mid-run on 2026-07-15 and crashed an overnight
pipeline; the sweep found ~27 GB of stale regenerable artifacts (RL experience .h5
buffers, artifacts/*.pt caches, 14× FEATURES_RUN_* builds, ML_CHECKPOINTS — which was
misnamed fspace_ml DOE output, not model checkpoints).

**How to apply:** when a pipeline writes derived data, put it in the research
project's own gitignored area, not ATLAS; when sweeping disk, anything in ATLAS
outside the two keepers is deletable after a consumer-grep (kept exception:
`order_flow_delta_5s.parquet`, 40 MB, 3 active research consumers). Deletion details
journaled in docs/daily/2026-07-16.md. Related: [[organize-research-folders]].
