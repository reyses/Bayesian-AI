# Event Library

Owner architecture (2026-08-03): **identify specific events, read the fuzzy
events.** Every owner-named tape state becomes (a) a strictly causal detector
and (b) its own cohort outcome table. This is the substrate for the
teacher-student ML target — *event classification + table lookup*, not price
prediction.

Run everything from the repo root with
`/home/moi/miniforge3/envs/bayesian/bin/python`.

**Live-day guard**: `2024_09_16` is the pocket-dojo live-sim day and is
hindsight-contaminated. It is excluded from every parquet, table and fit
(`pipeline/common.EXCLUDED_DAYS`). It appears ONLY in `tools/anchor_fire.py`,
where each detector is checked against the owner's calibration anchors.

## The six events

| # | event | owner phrasing | timeframe | N (v0) |
|---|---|---|---|---|
| 1 | `ULTRA_CHOP` | high flip rate, small net range over ~60s | 1s | 18,601 |
| 2 | `LEG_DESCENT` | stair-down: >=2 lower-high pushes, each ending in an impulse defense | 5s | 58,480 |
| 3 | `FAKEOUT_POKE` | poke <=2pt beyond a recent extreme, back inside within 60s | 5s | 78,731 RETURN |
| 4 | `STALL` | leg peak holds >=10min inside 30% giveback | 5s | 461 of 41,180 candidates |
| 5 | `DEFENDED_POKE_AT_SHELF` | >=5pt defended poke at any high-dwell shelf | 1m | 1,585 |
| 6 | `FLUSH_V_DAY` | open flush + V-recovery day class | 1m | 136 of 540 days |

## Layout

| Path | What it is |
|---|---|
| `pipeline/common.py` | Day loading + ET clock, the canonical 8.0pt close-based streaming `ZigZag` (same logic as `research/reversal_gauge`), Wilson / bootstrap / quartile helpers, `EXCLUDED_DAYS`, and the bounded-window rules that keep prior-evening bars out of every mask. |
| `pipeline/detectors.py` | The six detectors. `detect_ultra_chop` (1s), `scan_5s` (one shared bar loop producing LEG_DESCENT + FAKEOUT_POKE + STALL and their control cohorts), `scan_1m` (DEFENDED_POKE_AT_SHELF + FLUSH_V_DAY). Every threshold is a named constant with an origin comment. `_flush_confirm_ts` is **imported** from `research/reversal_gauge/builders/extract_freeze_events.py`, never copied, so the audited flushV detector cannot drift. |
| `pipeline/outcomes.py` | Forward-looking outcome measurement, deliberately a separate module: the causality boundary is a module boundary. Also holds `_sym_race`, the distance-symmetric ±10pt race whose null is 50%. All forward scans are hard-clipped at 16:00 ET. |
| `builders/build_event_library.py` | Corpus sweep -> one parquet per event type in `events/`. Multiprocess (`--workers`), ~10s for 603 day files. |
| `tools/build_tables.py` | Reads `events/*.parquet`, writes the master report `reports/event_library_v0.md` — per event: definition, prevalence, cohort table with CIs, causality self-audit, sharp/fuzzy verdict. |
| `tools/causality_audit.py` | **Truncation replay.** Re-runs every detector on days cut at 11:00 / 13:00 / 14:30 ET and requires every event stamped at or before the cut to reappear identically. Writes `reports/causality_audit.md`. This caught a real lookahead in the shelf detector. |
| `tools/anchor_fire.py` | Fires all six detectors on the calibration day `2024_09_16` and prints detection timestamps against the owner's anchors. Writes `reports/anchor_fire.md`. |
| `events/*.parquet` | Generated event tables (one row per event: causal fields + measured outcome). Reproducible from the builder; not intended for commit. |

## How to run

```bash
P=/home/moi/miniforge3/envs/bayesian/bin/python
# 1. materialise the library (603 day files -> events/*.parquet)
$P research/event_library/builders/build_event_library.py --workers 10
# 2. master report
$P research/event_library/tools/build_tables.py
# 3. causality audit (truncation replay)
$P research/event_library/tools/causality_audit.py --days 40
# 4. anchor sanity check on the excluded live day
$P research/event_library/tools/anchor_fire.py
```

## Data

`DATA/ATLAS/{1s,5s,1m}/YYYY_MM_DD.parquet` — per-day files, columns
`timestamp` (epoch s, UTC), `open`, `high`, `low`, `close`, `volume`. ET =
`tz_convert('America/New_York')`. MNQ tick 0.25.

**Day files start at 18:00 ET of the PREVIOUS calendar day.** Any unbounded
`minute_of_day >= X` mask therefore matches prior-evening bars first — the
audit bug that mislabelled flushV on 167/600 days in the reversal_gauge
builder. Every window in this package is bounded on both sides.

The 3-timeframe intersection is 603 day files; **540 carry RTH tape** (the
other 64 are Sunday-evening / holiday files whose only bars are the
18:00-19:00 prior-evening session).

## v0 findings (one line each)

Full numbers, CIs and definitions in `reports/event_library_v0.md`.

1. **ULTRA_CHOP — FUZZY.** Escape direction 50.9% up; post-escape drift median
   ~0pt at 5/15/30min against unconditional |moves| of 10/17/24pt. Also: the
   owner's anchor does **not** fire, and no useful absolute threshold makes it
   — at 60s resolution that moment has an ABOVE-median box (the window
   swallows an 11pt one-second flush). "Small net range" is implemented
   relative to the trailing ambient minute box, which is also the only
   era-robust choice.
2. **LEG_DESCENT — FUZZY.** Symmetric continuation is 50% flat at every chain
   depth N; the N>=2 vs N=1 delta is +0.1% [-0.7%, +0.9%]. Stair count carries
   no information. Impulse defenses hold only ~15% of the time for 5 minutes.
3. **FAKEOUT_POKE — SHARP on the level, FUZZY on direction.** A poke that
   snaps back clears the level (before a 10pt adverse move) 66.6% of the time
   vs 90.5% for a poke that sticks — a −23.9pp [−24.3, −23.5] significant
   split. The ±10pt direction race is still 50/50. The `~78.5% never exceed`
   reference could not be reproduced and traces to a different measurement
   (`oscillation_harvest.md` traverse completion).
4. **STALL — FUZZY.** 85.6% new-extreme vs 9.5% control looks enormous but is
   positional mechanics; the symmetric race is 50/50.
5. **DEFENDED_POKE_AT_SHELF — FUZZY across day-class.** ~37% crack on any
   high-dwell shelf, flushV vs other delta +1.9% [-4.9%, +9.0%]. The vshape
   V-floor's 1.4% does **not** generalise (nearest matched sub-cohort: 30.8%
   [18.6%, 46.4%], n=39): sharpness lived in the specificity of that
   construct. This detector also carried the one real lookahead the
   truncation audit caught — reading a 3-bar poke minimum at a stamp that can
   fire on bar +1. Fixed; the crack rate moved 28% -> 37%.
6. **FLUSH_V_DAY — SHARP as a day-class label.** Peak reclaimed later 86.8%
   vs 73.8% matched control, delta +13.0% [+5.6%, +20.1%].
