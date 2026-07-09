# AG TASK — Churn (oscillation) structure study

**For: Antigravity.** Self-contained; you have no conversation context. Read
fully. MNQ futures research repo; run via `.venv_wsl/bin/python` (WSL), paths
repo-root-relative. Reports → `research/leg_clock/reports/` (prefix `AG_churn_`).

## Context (hard-won, do not relearn)
The market here is ~91% oscillation ("churn") / ~9% trend. A prior study
(recovery_dynamics, 2026-06-30) found price behaves as a fixed-period clock
(mode 2–3 min swing periods, 71% revert <15 min) with a "breathing" amplitude
regime. The user (a discretionary MNQ trader) scalps this ebb-and-flow by eye
at the 5–20 point scale and wants the churn itself systematized.

## Goal
Characterize the oscillation as a CAUSAL, real-time state — and find where (if
anywhere) it pays after costs. Four questions, in order:

1. **State estimation**: build a causal oscillator state per bar — current
   swing phase (time & price position within the ongoing swing), amplitude
   estimate, period estimate. Simple is fine (zigzag-based or bandpass);
   judge it by forward usefulness, not elegance.
2. **Phase → next swing**: given the state says "near the top of a swing,
   amplitude A, period P", what does the NEXT swing do OOS? Does phase
   predict turn timing better than the unconditional clock?
3. **Ebb-flow economics**: a mechanical oscillation scalp (fade swing
   extremes, exit mid/opposite extreme) at 5–20 pt amplitudes: $/day after
   4 ticks/round-trip costs, by amplitude regime and time of day. The
   known-good hours are 9–13 CT (measured 2026-07-08); confirm/deny that the
   churn edge concentrates there.
4. **Churn→trend transition, conditioned**: ONLY within confirmed-oscillation
   state (do NOT lump regimes — measured lesson 2026-07-08), does anything
   causal precede the oscillation breaking into a trend? (Volume-rate,
   candle shape, and volume-at-price walls are already DEAD/weak — see the
   dead list.)

## Data
- `DATA/ATLAS/{5s,1m}/YYYY_MM_DD.parquet` (OHLCV; timestamp unix s).
  2024 = 259 days (train/develop), 2025 = 277 days (OOS — touch once per
  finding, not per tweak). MNQ tick 0.25, $0.50/tick.
- `DATA/ATLAS_NT8/` is SEALED. Do not touch it — it is the program's final
  gate, spent deliberately by the user only.

## Mandatory methodology (each has burned this repo before)
1. **All feature extraction through the ForwardPassSystem**
   (`core_v2/FPS/forward_pass_system.py`) or raw parquet with strictly
   trailing windows. Your previous task's 0.87 AUC collapsed to 0.55 when
   re-extracted leakproof — the entry bar itself had leaked. Never index a
   feature at or after the event bar; always strictly before.
2. **Both-years rule**: any config/threshold chosen on one year must hold on
   the other, or it is rejected as noise. (Four candidate "improvements"
   were rejected by this rule on 2026-07-08 alone.)
3. **Null-anchored**: every effect vs a matched null (same day/hour), plus
   shuffle nulls for classifiers. Signal bar: gap ≥0.10 REAL, 0.05–0.10
   CONDITIONAL, <0.05 NOISE.
4. **Economics, not AUC**: any claimed edge must show $/day with day-block
   bootstrap 95% CI and PF, after 4t costs. State "NOT significant" when the
   CI includes 0. AUC alone proves nothing here (measured, repeatedly).
5. **Sigma-relative distances** everywhere (never fixed tick tolerances) —
   user-corrected methodology, 2026-07-07.
6. **This task is LABEL-FREE.** Do NOT use `DATA/ai_cusp_picks/` as targets,
   training data, or evaluation truth anywhere in Q1–Q4. Reasons: they are
   hindsight-optimal (entries snapped to 1s extremes, 0-MAE by construction),
   they encode one particular swing-carving that would bias your oscillator
   state, and scoring against their timestamps re-imports a known
   unlearnable-timing trap. Judge everything on FORWARD PRICE OUTCOMES and
   dollars — those truths are causal and sufficient for churn. (At most: an
   optional read-only cross-check at the very end — "where do labels sit in
   my oscillator phase?" — reported as curiosity, never as validation.)

## Dead list (do NOT retest as-is)
Volume-rate buildup; candle wick/body shapes; band-level first-touch bounces;
zone touch-count; bar-to-bar slope persistence; confirm-then-ride at leg scale
(loses -$60..-200/day); APZ re-entry confirmation (PF 0.97/0.83 — the fade
edge IS entering into violence); VP wall gate (flips sign across years). Also
known: turns sit at LOW-volume nodes / outside the value area (weak but 2-yr
stable); leg length is fat-tailed momentum (rising MRL, OOS-stable) but
untradeable alone.

## Deliverables
- `AG_churn_findings.md`: the oscillator-state definition + its forward value
  (Q2), the ebb-flow economics table (Q3), the conditioned transition result
  (Q4) — each with nulls, both-years, CIs, and an explicit verdict line
  (REAL / CONDITIONAL / NOISE).
- Reusable tools in `research/leg_clock/tools/` (prefix `ag_churn_`).
- A `fable_5_review_packet.md`-style summary is welcome; it WILL be
  peer-reviewed the same way the last one was.
