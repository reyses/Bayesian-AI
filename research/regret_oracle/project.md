# Project: Regret Oracle — continuous opportunity capture & selection

DMAIC frame. Started 2026-05-14.

## Translation table
- **regret oracle** = upper-bound opportunity set: every M_1m local extremum is
  marked as an entry, forward 60-min MFE measured. It is *greedy by design* —
  it takes every opportunity. It is NOT a strategy; it is the label universe a
  selector must learn to subset.
- **CRM** = Close Regression Mean (rolling-window mean of close at a timeframe).
- **session** = Globex maintenance-halt to maintenance-halt. The halt is 5–6 PM
  ET (21:00–22:00 UTC in EDT, 22:00–23:00 UTC in EST — DST-shifting, so detected
  empirically from 61-min gaps in the 1m stream, never hardcoded).
- **1-hour lapse premise** = we have a 1-hour window to identify and take the
  single best entry. The hour is a *budget*; velocity ($/min) measures $ per
  budget-minute spent.
- **MFE velocity** = mfe_dollars / time_to_mfe_min.

## Define
The v9c selector (`P_entry × MFE_pred`) produces +$32/day OOS at best but with
two structural failures: (1) calibration inversion — high P selects the most
repetitive/chop-like opportunities, which have the weakest forward edge;
(2) the MFE regressor predicts the mean (~$55), can't separate an $11 chop
scalp from a $40 structural move.

Root cause hypothesis: the selector has been trained/swept on **fragmented,
calendar-day-chunked** oracle captures (July only, single days) with **no
session awareness** and **no duration axis**. The seed-level mode is $11 — the
opportunity set is dominated by short chop scalps (57% of oracle entries are
SLOW 25–60 min trades, but the $-mode is dragged down by the FLASH/FAST pile).

Premise correction (user, 2026-05-14): the oracle taking all opportunities is
*correct* — the open problem is the **selector**, and the selector must respect
the **1-hour-lapse** premise (argmax-per-window, not threshold-crossing) and
**categorize on duration**.

## Measure  ← CURRENT PHASE
Rebuild the regret capture as a single continuous, session-aware pass over the
full IS.

- **IS** = 2025-01-01 → 2025-12-31 (full calendar year).
- **OOS** = 2026-01-01 forward. Clean calendar transition.
- **Sessions** = empirical halt-to-halt (61-min gap detection, DST-proof).
- **Base TF = 5s** (default) — for entry-pricing + forward-MFE-timing
  precision (1m quantizes time-to-MFE to whole minutes and its bar H/L misses
  the true excursion peak).
- **Extrema on RAW PRICE** — peaks on bar `high`, troughs on bar `low`;
  `entry_price` = the raw extreme. The opportunity (label) comes from price
  truth; the CRMs/anchors are the **analysis layer** (state vector / features),
  NOT the detection series. This also keeps the 5s-is-noise hard-rule intact —
  no 5s CRM is used as a predictor.
- Anchors computed continuously across halts (slow 1h rails intentionally span
  sessions; 15s/1m tactical anchors re-warm within their window — no reset).
- Extrema-detection window and forward-MFE window must not cross a halt or
  weekend gap. End-of-session extrema are **kept and tagged** (`full_window`,
  `available_fwd_min`), never dropped — blender-first.
- Output: `reports/findings/regret_oracle/oracle_entries_IS_full.csv` with
  added columns: `session_id`, `session_date`, `tod_minutes`, `full_window`,
  `available_fwd_min`, `mfe_velocity`, `volume`, `bar_range`.

Tool: `tools/regret_1m_oracle.py` (reworked in place).

## Analyze

### Phase 2a — Distribution EDA on raw oracle entries (no fusion, no filter)
Per user 2026-05-14 sequence: *understand the distribution before clustering.*
Per-cell **$/trade** is the metric (CLAUDE.md mandated form: mode + mean with
95% bootstrap CI on $/trade, histogram bin $2).

**Conditioning cells** (single-axis + key pair joints):
- `tod_bucket` — session-phase since post-halt open (4-hour blocks)
- `regime_2d` — joined via `session_date` → `regime_labels_2d.csv`
- `direction` — LONG vs SHORT (often asymmetric)
- `duration_bucket` — FLASH (0–3m) / FAST (3–10m) / MEDIUM (10–25m) / SLOW (25–60m)
- `liq_bucket` — volume (or bar_range) quartile within the dataset

**Per-cell two-test gate**:
1. `pass_mode` : `mode_$ > noise_floor_$`   → real trades, not measurement artifact
2. `pass_ci`   : `ci_lo_$ > 0`              → significant edge (CI excludes 0)
3. `tradeable` = both pass

**Noise floor (random-walk benchmark)** — per entry, in dollars:
    noise_floor_$ = bar_range_points · sqrt(N_fwd_bars)
Derivation: for Brownian motion E[one-sided MFE over N bars] = σ·sqrt(2N/π);
range-to-σ ratio σ ≈ bar_range / 1.6; MNQ $/point = 2 → the constants collapse
to that clean form. Per-cell noise floor = median across constituent entries.
(First-pass approximation — uses entry-bar range as proxy for forward-window
volatility; can be refined to a session-median or session-percentile later.)

**Decision the EDA must answer**:
- If stratification gives clean separation (some cells clearly tradeable, most
  clearly noise) → **selector gates on cells, fusion is unnecessary**.
- If real-trade cells still carry many noise-magnitude entries → **fusion
  becomes the next lever** (Phase 2.5).

Tool: `tools/regret_distribution_eda.py`. Outputs under
`reports/findings/regret_oracle/`:
- `cell_stats_<name>_<axis>.csv`              (one per single axis)
- `cell_stats_<name>_<a>_x_<b>.csv`           (one per pair joint)
- `per_cell_per_trade_stats_<name>.csv`       (combined)
- `tradeable_cells_<name>.csv`                (passed both gates)
Run: `python tools/regret_distribution_eda.py --input <oracle_csv> --tf-min 0.0833 --name IS_full`

### Phase 2.5 — Trade fusion (ONLY IF Phase 2a says we need it)
Consecutive same-direction oracle extrema fused into one trade:
- entry_price = best price across constituents
- entry_ts = first constituent's timestamp (or constituent holding best price)
- cluster_end_ts = min(last constituent's ts + 60min, session end, next-opposite-extremum ts)
- cluster_mfe_dollars = max excursion from entry_price over the cluster window
- viability gate applied at the *cluster* level, threshold derived from Phase 2a's noise floor

Skip if Phase 2a already gives a clean cell-gate solution.

### Phase 2d — Classification on (clusters OR raw entries, depending on 2a/2.5)
- Duration discriminator: is FLASH-vs-SLOW separable from the entry-time state
  vector? If AUC ≈ 0.5, duration is a post-hoc label only and the selector
  cannot use it.
- Cross categoricals: stack × rail × duration × tod.
- **Liquidity guard**: `volume` / `bar_range` already in the schema — Phase 2a
  uses these for the liq quartile axis; the selector inherits.

## Improve
- Fine-tune the trading window from the tod × velocity surface.
- Rebuild the selector as **argmax-per-1h-window** (v9d), not threshold-crossing
  — directly implements the 1-hour-lapse premise; ~6–7 trades/day not ~47.
- Make the selector/exit duration-conditional IF duration is separable at entry.

## Control
- Validate every classification finding and the selector on 2026 OOS.
- Per MEMORY hard rule: OOS sign-stability per categorical cell before quoting.
- Report $/trade as mode + mean w/ 95% bootstrap CI; never quote $/day without CI.

## Cycle log
- `cycle_01.md` — (pending) Measure phase: continuous session-aware capture.
