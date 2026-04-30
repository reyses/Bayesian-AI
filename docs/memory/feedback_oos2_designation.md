# OOS-2 designation — ATLAS_NT8 as second OOS gate

**Designated: 2026-04-27.** User decision to elevate ATLAS_NT8 (NT8-feed
dumps, 32 days Mar 20 → Apr 26 2026) to a formal **second OOS validation
layer**, complementary to the existing Databento ATLAS_OOS.

## What changed in the validation ladder

```
OLD (5 gates):
  IS (ATLAS, Databento 2025)
  → OOS (ATLAS_OOS, Databento Jan-Feb 2026)
  → Phase 7 Replay → Live Sim → Live Real

NEW (6 gates, 2026-04-27):
  IS (ATLAS, Databento Jan-Dec 2025)
  → OOS (ATLAS_OOS, Databento Jan-Feb 2026)        — temporal shift, same feed
  → OOS-2 (ATLAS_NT8, NT8-feed Mar-Apr 2026)       — temporal AND feed shift
  → Phase 7 Replay → Live Sim → Live Real
```

## Why two OOS gates instead of one

ATLAS and ATLAS_OOS are both Databento data. They cross-check temporal
stability — does a strategy trained on 2025 still work on Jan-Feb 2026? — but
they share the same data feed. They do NOT cross-check:

- **Tick-by-tick fill semantics** (NT8 fills can differ from Databento prints
  due to broker matching, contract continuity, weekend handling).
- **Contract-roll behavior** (the way each feed handles MNQ rollover affects
  daily returns near contract change).
- **Volume reporting conventions** (NT8 volume ≠ Databento volume in some
  edge cases — single-print vs aggregated).
- **Session-boundary cuts** (Databento splits Globex sessions; NT8 dumps are
  calendar-day. Aggregation logic that assumes one will misbehave on the other).

A strategy that wins on IS+OOS but fails on OOS-2 has a feed-dependency
that the Databento-only chain cannot detect. **OOS-2 catches it before live.**

This matters specifically because the project's history includes:
- **Phantom spikes were fake edge** (`memory/feedback_phantom_spikes.md`):
  $4,350 of "edge" was NT8-feed phantom-spike artifact, vanished on clean
  Databento data. The reverse is also possible — Databento-found edge that
  vanishes on NT8 fills.
- **Frozen SFE cache bug 2026-04-16** (`project_frozen_sfe_cache.md`): live
  trading drifted from training because of feed-data invariant violation.
- **Lookahead audit 2026-04-17** (`feedback_lookahead_audit.md`): a +$740/day
  baseline collapsed to -$164/day after lookahead fix. Cross-feed validation
  is exactly the kind of check that catches subtle invariant violations.

## Decision rule — when to use OOS-2

| Strategy state | OOS-2 role |
|---|---|
| RC awaiting promotion to release | **Must validate on OOS-2** before live |
| Tier modification with claimed lift | Claim must replicate on OOS-2 |
| New filter / regime gate | Walk-forward on IS/OOS AND OOS-2 |
| Bug fix / refactor | OOS-2 used as parity check (numbers within tolerance) |
| Pure data-pipeline change | OOS-2 sanity check only |

**Promotion rule**: if a finding holds on IS + OOS but fails on OOS-2:
- DO NOT ship to live.
- Investigate which axis breaks: feed (data-source-specific bug, slippage
  difference, contract-roll handling) or temporal regime (Mar-Apr 2026 has
  characteristics Jan-Feb didn't).
- A strategy that legitimately works should hold on both within statistical
  noise; large divergence is a red flag.

## Constraints to remember

- **32 days is small.** Stat power on small-N tier work is weak. Treat OOS-2
  as a SANITY check ("does this generalize at all?"), not as the primary
  statistical gate. The primary gate stays ATLAS_OOS until OOS-2 has 60+ days.
- **Single contract (MNQ_06-26).** Roll boundaries between contracts not yet
  represented in OOS-2.
- **Holiday-truncated days are over-represented.** Of the 32 days, several
  are Sunday-evening short sessions (~6,500 rows) and Friday-close-shortened
  (~10,000 rows). Full sessions only ~16,500 rows.
- **2026-04-26 is truncated** (3.2h instead of full session) — exclude from
  any tail-of-window analysis until re-dumped.

## Expansion path

To make OOS-2 a primary gate (not just sanity check):

1. Enable `BayesianHistoryDumper.cs` v2.0.0 on a chart with **180+ days of
   load history** (NT8 supports this; just set "load N days" high). Single
   chart now produces 1s/1m/1h/1D simultaneously.
2. Re-run `python tools/atlas_nt8_rebuild.py` — incremental, only adds new days.
3. Re-run `python training/build_dataset_v2.py --atlas DATA/ATLAS_NT8` — also
   incremental.
4. Once OOS-2 has 60+ days, stat power supports primary-gate use.

## Tools that need updating to know about OOS-2

Going forward, analysis tools should default to running ALL THREE gates
(IS, OOS, OOS-2) when a parity claim is made:

- `tools/v15_filter_apply.py` — currently only runs on the NT8 trade-export CSV
  for ZigzagRunner. Could be extended to run the v1.5-RC bleed filter on
  ATLAS_NT8 daily 1D data for a parallel cross-check.
- `tools/tier9_bleed_filter.py` — already supports `--atlas` flag. Run with
  three different atlases for IS / OOS / OOS-2.
- `tools/v15_calibration_drift.py` — has a HARDCODED path to `DATA/ATLAS/1D`.
  Should be updated to take `--atlas` arg so it can run against OOS-2.
- Any new validation tool: bake in the three-atlas convention from the start.

## Related memory

- `memory/feedback_phantom_spikes.md` — historical case of feed-dependent
  fake edge
- `memory/feedback_lookahead_audit.md` — invariants must hold across data
  sources
- `memory/feedback_data_validation_first.md` — data integrity before analysis
- `memory/feedback_cnn_fragility.md` — small-sample overfit traps (relevant
  given OOS-2's 32-day size)

## Build provenance

OOS-2 dataset was built 2026-04-27 in this session:
- Raw bars: `DATA/ATLAS_NT8/{1s,5s,...,1D}/*.parquet` (via
  `tools/atlas_nt8_rebuild.py`)
- 139D features: `DATA/ATLAS_NT8/FEATURES_5s_v2/{25 families}/*.parquet`
  (via `training/build_dataset_v2.py --atlas DATA/ATLAS_NT8`)
- Schema parity vs canonical `DATA/ATLAS/FEATURES_5s_v2/` verified
  column-by-column.
- Provenance report: `reports/findings/2026-04-27_atlas_nt8_features_built.md`.
