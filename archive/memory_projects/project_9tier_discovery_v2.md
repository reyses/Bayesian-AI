---
name: 9-tier discovery exercise (V2-native)
description: 2026-05-04 — recreate the legacy 9 ExNMP tiers in V2 by running NMP-only and finding signals that distinguish FADE_BETTER from FLIP_BETTER cohorts. Single-column V2 EDA found NO discriminating signal.
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## What this is

The user requested recreating the legacy 9-tier discovery process in V2-native
form. Legacy methodology (per `docs/daily/2026-04-06.md` through `2026-04-18`):

1. Run NMP entry: `|z_1m| > 2.0 + variance_ratio < 1.0`
2. Find the splitter axis (legacy found: velocity at entry, CNN flip,
   1h alignment, wick rejection)
3. Sub-classify entries → measure $/trade per sub-tier
4. Add new entry types when EDA reveals signals NMP misses

The V2 equivalent of step 1 is REVERSION: `|z_se_w| ≥ 1.8 + reversion_prob ≥ 0.55`.

## Methodology used (training_v2/tier_discovery.py)

For each NMP-only IS trade, compute via regret labels:
- `fade_peak` = peak realized in original direction
- `flip_peak` = -mae_pnl (peak realized if direction had been opposite)

Classify each trade:
- **FADE_BETTER**: fade_peak > flip_peak + $15 AND fade_peak > $15
- **FLIP_BETTER**: flip_peak > fade_peak + $15 AND flip_peak > $15
- **CHOP_SKIP**: neither direction had a clear peak

For each of 185 V2 columns at entry, compute Cohen's d between
FADE_BETTER and FLIP_BETTER cohorts. Walk-forward validate on 70/30 split.

## Findings (2026-05-04, 19,106 NMP IS trades)

### The opportunity is real and large

| cohort | n | actual $/trade | fade peak | flip peak |
|---|---:|---:|---:|---:|
| FADE_BETTER | 8,243 | +$9.29 | $139 | $32 |
| FLIP_BETTER | 8,122 | -$9.20 | $33 | **$146** |
| CHOP_SKIP | 2,741 | +$0.56 | $42 | $42 |

Roughly 50/50 fade/flip split. A perfect oracle that flipped the
FLIP_BETTER cohort would convert -$74,710 → +~$120K (peak-side estimate),
turning the IS total from $3K → ~$152K.

### But single-column V2 features carry NO signal

- **Highest Cohen's d across all 185 V2 columns: 0.040** (negligible — small
  effect threshold is 0.2, medium is 0.5)
- **0 of 25** top IS features survive walk-forward validation (70/30 split)
- Top features are all volume-related: `L2_15m_vol_accel_12`,
  `L2_1m_vol_mean_15`, `L1_15s_price_accel_1b` — but signs flip on validation

This is the same OOS-overfit pattern as the 2026-05-03 quantile-cell lesson
(75% IS-best rules collapsed on OOS, 25.8% survival rate).

## Why this differs from the legacy CNN flip's 70.6% accuracy

The legacy CNN flip predictor saw a 6×13 V1 grid AND learned **cross-feature
patterns** (distillation showed it combined `15m_wick_ratio × 1h_z_align ×
5m_velocity`). Single-column linear EDA cannot see those.

Plus we are missing key features the legacy used:

| Legacy discriminator | In V2 entry vector? |
|---|---|
| `wick_ratio` (multi-TF) | NO — wick rejection requires OHLCV math, not in V2 |
| `1h_z_align` (sign agreement) | Yes-ish (`sign(L2_1h_price_velocity_w)`) but encoded as a magnitude (positive and negative cancel out) |
| `dmi_diff` extreme | NO — DMI was a V1 synthesis concept |
| Directional wicks (upper/lower) | NO — needs 1m/5m/15m OHLCV |

User noted that directional wicks WERE calculated during the run in the
legacy path (`directional_wicks_batch` in `core_v2/v1_compat.py`) but are
NOT yet in the V2-native entry feature vector. Adding them is path A
described below.

## Three honest paths forward

| Path | What | OOS-overfit risk |
|---|---|---|
| **A** | Add directional wicks per-TF to entry features and rerun discovery | Low — recovers known-good signal |
| **B** | Run chord-style joint-quantile search on 185D pairs (~17k pairs) | Medium-High — pair-quantile rules collapsed on OOS in 2026-05-03 |
| **C** | Train V2DirectionCNN on the FADE/FLIP/CHOP labels — let it learn cross-feature patterns | Medium — known to overfit; needs walk-forward validation |

The user's framing was "what signal can we use to TURN BAD INTO GOOD" — the
goal is direction-flipping FLIP_BETTER cohort, not skipping it.

## Code artifacts

- `training_v2/tier_discovery.py` — the discovery tool with Cohen's d + walk-forward
- `training_v2/output/regret_nmp.pkl` — regret labels for 19,106 NMP-only trades
- `training_v2/output/nmp_only.pkl` — the trade pickle
- `reports/findings/v2_tier_discovery.md` — markdown report

## Conclusion

V2-native single-column EDA at NMP entry produces no actionable splitter.
The 9-tier legacy methodology relied on cross-feature patterns the CNN
implicitly learned, OR features (directional wicks, DMI synthesis) we don't
currently have in the V2 entry vector. Plan: add directional wicks to the
entry vector first (path A), then escalate to CNN if single-column with
wicks still fails.
