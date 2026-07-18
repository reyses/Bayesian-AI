# Mamba production-run spec (DRAFT v1 — 2026-07-18, autonomous night)
Status: DRAFT — §6 (wrong-dir cut) fills when the wrong-direction dojo verdict
(doc 100) lands; the RUN itself is user-launched (Moises runs training).
Assembled back-to-front per Moises' order: corrected dossier (097) → exit dojo
(098) → wrong-dir dojo (100, pending) → this spec.

## 1. Objective — what this run is ALLOWED to promise
Per the blind exit-dojo verdict (doc 098): exit alpha comes from **riding
winners longer**, NOT from timing turns (turn-timing is a wash-to-loss blind;
winners +19.5 pts vs 5m-hold, the only significant regime). The run optimizes:
(a) signal-conditioned entry (COLD proved the reward teaches it, +0.21 gap),
(b) ride-length on winners (capture ratio vs the label's remaining move),
(c) fast cut of wrong-direction entries (§6, pending verdict).
Do NOT evaluate the run on turn-timing metrics.

## 2. Wiring (all landed, commits 46a8328f + 4abeb820)
- FIXED2 reward wiring: scorecard-only (dollars OUT of the gradient), capture
  live (entry-time remaining extent via label exit_price — labels TEACH, never
  observe), decaying cut bonus (t_hold + env MAE), selective wiggle, per-fire
  windowed regret via phit_feed, oracle-true actual_dir, real swing_id.
- `--compile_act` ON (gates PASSED: fp32 parity 2.4e-07, bitwise determinism,
  bf16 1.47×; 431 bars/s → full-curriculum epoch ~10.3h).
- Output hygiene: all artifacts → reports/runs/ + checkpoints/ (no root litter).

## 3. Units ruling (Moises, 2026-07-17 — supersedes σ-normalization)
- **Policy reads price directly in TICKS** (integers; MNQ 0.25pt=1 tick).
  Ledger observation floats: convert dollars→ticks. ONE canonical internal
  unit — assert at env boundaries (unit dead-wires burned us twice).
- **Noise profiling = separate channels, NOT input division**: the V2 z/std/vr
  families already carry it; add trail_vol (ticks) as an explicit ledger-side
  channel. Rationale receipt: mean-based σ estimators broke on fat tails
  (graveyard §4 vol-adaptive exits); dividing by a fallible σ corrupts price.
- **Reward**: capture stays LEG-RELATIVE (ratio to remaining move — unit-free,
  survives the ruling). Cost + MAE terms in raw ticks vs named constants
  (replace the σ-divisions; e.g. COST_TICKS=3 enters as a plain tick penalty
  scaled by a named REF_TICKS constant, documented).
- **Reporting**: points, mode-first, at every human boundary.

## 4. Observation additions (state vector deltas, all causal)
1. Leg-geometry block (doc 095 promotion): leg_age, amplitude, ER, giveback,
   stall — REQUIRED.
2. Dojo grammar channels (doc 098 citation audit): ER10 + 1m delta + rolling
   floor/peak; per-family fresh-fire AGES (KMDR/CLIMAX/HA/PROPP with/against);
   giveback VELOCITY + band-breakout flag; bar anatomy (close-in-range,
   range z); vol(5m) delta signed by favorable/adverse.
3. **NMP9 tier one-hot at entry** (9 + none): the Shainin strata are validated
   state (2026-07-18: threshold-robust, ride family aligned 0.76-0.85). The
   ladder tells the net WHICH KIND of trade it's managing.
4. phit_feed c_t + direction (already wired).

## 5. Training config
- Warm-start from supervised checkpoint (direction knowledge) + LONG curriculum;
  watch P(enter|sig)−P(enter|nosig) from epoch ~5 — if the gap hasn't emerged,
  bump w_s/w_w one step and restart (the warm prior fights selectivity early;
  COLD proves the reward teaches it).
- Overtrade watch: trades/day must fall below ~50 by mid-curriculum, else raise
  cost term (smoke trajectory: 3854→517 in 2 epochs — pressure works).
- Anti-freeze criteria stay: pct_flat must never trend to 1.0 (never appeared
  in any smoke arm — regression-watch only).
- Seed + [SMOKE] metrics every epoch; parity re-run of gates after ANY env edit.

## 6. Wrong-direction cut (PENDING — doc 100 verdict fills this)
Placeholder pending the 100-good/100-bad blind Red X result:
- If agents/conditions BEAT the dumb-stop ROC: encode the winning conditions as
  reward shaping (credit early bails scaled by damage-avoided) + state emphasis.
- If the dumb stop wins: the cut-head target is the dumb stop itself (bail at
  the ROC-optimal X ticks), and the net's job reduces to not-bailing GOOD dips.

## 7. Run gates (before Moises launches)
1. reward_env synthetic tests green after the ticks conversion (§3).
2. One 4-day smoke arm (same days/seed as the A/B/C/FIXED2 chain) confirming:
   capture>0 on right trades, no freeze, [SMOKE] json sane in ticks.
3. Speed gates re-PASS post-edits.
4. Reviewer sign-off, then "run with --fresh when ready" — Moises launches.
