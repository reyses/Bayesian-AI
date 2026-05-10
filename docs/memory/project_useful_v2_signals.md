---
name: Useful V2 signals (chart-validated 2026-05-08)
description: Signals confirmed visually/empirically as informative on 2026_02_12; signals confirmed as redundant or noise
type: project
---

Validated by visual inspection on 2026_02_12 (OOS best #5 — clean morning rally + afternoon crash). Confirmed by SPC/I-MR analysis across 277 IS days where applicable.

**REVERSION / 3-BODY FRAMEWORK (locked 2026-05-09)**

Terminology:
- pivot ≡ inflection ≡ bar where direction changes
- reversion = the new leg that begins at the pivot (NOT mean-reversion in the OU sense)
- trend-follow / ride = trade WITH the move
- Each leg is a reversion (relative to the prior leg's direction)

3-body envelope:
- M_close (blue) = center / target / close-volatility
- M_high (green) = upper bar-extreme regression mean
- M_low (red) = lower bar-extreme regression mean
- Each anchor has its own σ envelope (±1, ±2, ±3)
- All 3 acts as elastic anchors — no force-free equilibrium

Goldilocks operating levels:
- ±1σ: too common (~32%) — no edge
- ±2σ: ★ PRIMARY trigger (~5% of bars) — real reversion edge
- ±3σ: extreme (~1.5%) — often regime-shift signal
- M_close ±3σ_close: outer-wall confirmation (rarest)

Mixed-regime asymmetric TF design:
- HL anchors at SLOW TF (15m, 1h, 4h) — rare extreme triggers
- CRM at FAST TF (1m, 5m) — responsive target
- At 1h-HL ±2σ, P(continuation) ≈ 0 (reversion structurally certain)

**MACRO-EVENT PROBLEM** (2026-05-09 finding):
The reversion framework FAILS during 2-3 hour impulse phases. Real-time
distinction between "in macro-event impulse" and "calm regime" is the
missing operational gate. 2026_03_03 lost $1,324 with reversion fires into
a 3hr crash. OOS forward-pass-honest = −$40/day — not because strategies
are bad, but because no impulse-phase suppressor exists.

CRM flatten-then-pivot detector (designed): state machine NORMAL →
DIRECTIONAL → FLATTENED → PIVOT → IMPULSE → STABILIZING → NORMAL,
walked bar-by-bar with no lookahead. Needs 5-min monitor window after
pivot to filter wiggles vs real impulses.

Compression-precedes-expansion validated empirically (1h, 50 IS days):
σ-rank ≈ 0.42 at T-60min before HIGH+ extremes (compressed),
rising to 0.63 by T-15min. Tradable early-warning lead time.

**STRUCTURAL REQUIREMENTS — every probability table must condition on**:
1. Time-of-day (`L0_time_of_day` available; reversion edge varies wildly
   across session: lunch highest, opens lowest)
2. Day-of-week (Friday vs Monday have different regimes)
3. Calendar event (FOMC/NFP/CPI = scheduled-impulse days; very different
   distributions; need external flag)
4. Current-day-state-so-far (already had impulse? quiet morning? Multiple
   impulses?). Day-state machine needed across the session.

These conditioning axes MUST be in the design from the start — adding
them later changes all bin populations and invalidates earlier calibrations.

**TODO (deferred work)**

- **Empirical breach probability lookup** (deferred): build a per-bar `P(price breaches +k·σ band in next N bars)` lookup. Walk IS data, group by `(z_se_bucket × regime × N × tod × dow × cal_event)`, count the fraction of forward windows where `max(z_se) ≥ k`. Replaces the broken `reversion_prob_w` (which is saturated by NMP entry-gate selection bias).
- **Macro-event detector v2** with 5-min monitor window for impulse confirmation
- **9-layer probabilistic stack** — fuse single-layer probabilities into compound conviction
- **Replay of 2026_02_12 + 2026_03_03** with macro-gate active to quantify expected impulse-day damage avoided

**USEFUL signals — keep / wire**

- `L2_1m_price_mean_15` — 1m regression mean. Hugs price tightly. Tradable as the "immediate fair value" anchor.
- `L2_5m_price_mean_9` — 5m regression mean. Smooth, less jitter. Reference for "where 1m wants to revert to."
- `L2_15m_price_mean_12` — 15m regression mean. Strategic direction line — its slope (over 1h lookback, Q75 magnitude) is the day-scale regime signal.
- `L2_5m_price_sigma_9` — 5m SE bands at ±1σ/±2σ/±3σ/±4σ. Cone width is itself a regime signal: tight cone = calm (fade-friendly), expanding cone = transitioning, wide cone = volatile.
- **1m-5m mean divergence** (`mean_1m − mean_5m`): per-bar continuous tradable signal. Cross above +Q75 = SHORT entry; cross below −Q75 = LONG entry; snapback through zero = exit.
- **15m mean slope (1h lookback)**: strategic gate. Sign of the slope at Q75 magnitude defines trade direction bias for the period.
- **15m mean curvature** (slope-of-slope): magnitude marks sharpness of pivot; combined with slope sign-change pinpoints inflection points. Q75 of |curvature| filters drift-throughs from real pivots.
- `L2_*_vol_mean_w` + `L2_*_vol_sigma_w` — **regime-change detector ONLY, NOT a direction signal**. Vol spikes look identical at rally peaks and crash legs (both = "something is happening"). Use as CONVICTION multiplier on directional signals or as a REGIME gate ("event day vs normal day"), never as a buy/sell trigger by itself.
- `L2_*_vol_velocity_w` + `L2_*_vol_accel_w` — **LEADING pre-pivot timing signal (direction-agnostic)**. Visually validated on 2026_02_12: at the 14:00-14:30 macro pivot all four volume-derived features (vol_mean, vol_sigma, vol_velocity, vol_accel) spike together BEFORE price breaks. Not a "which way" signal — a "WHEN" signal: tells you a pivot is imminent. Combine with σ-rank compression (T-30 to T-60min) for high-confidence early warning. **Multi-signal pre-pivot pattern**:
  1. σ-rank compression (T-30 to T-60min): bands narrow
  2. vol_velocity rising (T-N min): activity ramping
  3. vol_accel spiking (T-1 min): activity acceleration
  4. CRM slope flattening (T-K to 0): direction losing conviction
  5. CRM slope flips (T = 0): pivot fires
  6. vol_mean / σ explode (T+): impulse confirms

  Refined volume rule: **volume is direction-agnostic for WHICH WAY price will go — but vol_velocity/vol_accel are direction-agnostic LEADING indicators of WHEN a pivot is about to happen.**
- `L3_1m_swing_noise_15` — chop/calm gauge. Spikes 4-5x baseline at vol regime transitions. Matches SE band expansion visually.

**REDUNDANT — drop**

- `L2_*_vwap_w` at all TFs — visually identical to `price_mean_w` at the same TF. Pick one, drop vwap.
- `L2_4h_price_mean_18` — barely moves intraday (-9.5 pts on a -408 pt day). Useless at intraday timescale.
- `L2_1h_price_mean_12` line itself — step function at TF cadence, lagged. Drop the line, but its computed slope is useful.

**NOISE — don't use as primary signals**

- `L1_1m_price_velocity_1b` and L2 velocity at 1m / shorter — too noisy on its own. Sign agreement with longer TFs is required.
- `L3_1m_hurst_15` — jitters around 0.5 boundary. Not stable enough alone; would need smoothing.
- `L3_1m_reversion_prob_15` — saturated near 1.0 at NMP-qualifying bars (selection bias). Selection bias: rprob is high at entries (because NMP gates on it) and stays high through MFE. Can't use to detect "thesis dying."

**STRUCTURAL FACT — TF-anchored features**

V2 features at TF=1h, 4h, 1D update at TF cadence and hold flat between updates. At 5s sampling, 90-97% of signed bar-to-bar MR values are zero. **For velocity/slope/MR analysis on TF-anchored features, sample at TF cadence or compute slopes over rolling lookbacks at 5s cadence (not bar-to-bar diffs).**

**THREE-ROLE COMPOSITE FRAMEWORK** (validated visually on 2026_02_12):

```
Strategic gate (15m mean slope, 1h LB, Q75)
  green-shaded → LONG bias allowed
  red-shaded   → SHORT bias allowed
  un-shaded    → either / no-trade

Tactical entry (1m − 5m divergence)
  cross +Q75 → SHORT signal (allowed only in red regime)
  cross −Q75 → LONG signal  (allowed only in green regime)

Tactical exit
  divergence crosses back through zero (snapback)
```

Additional gates available but not yet wired:
- volume regime (vol_sigma expansion = step aside / wider stops)
- swing_noise (chop in calm regime → fade ok; chop in volatile → pass)

**Charts of record**:
- `chart/2026_02_12_regression_means.png` — price + 1m/5m/15m means + 5m SE bands + slope panel + 1m-5m divergence panel + strategic shading + inflection markers
- `chart/2026_02_12_other_features.png` — V2 feature panorama (vol, swing, z, hurst, accel)
