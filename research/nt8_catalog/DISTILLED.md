---
name: distilled-nt8_catalog
description: The program's spine — signal league + calibrated combiner + λ validation + the turn/exit/cut closures that killed damage-control and left "ride the winner" as the whole edge
metadata: {type: distilled, topic: nt8_catalog, status: live}
---
## Verdict
The catalog is the program's central research spine. It ran the funnel Moises stated:
extract causal signals aligned with the AI labels → mix into ONE calibrated combiner P(right)
→ hand the completed signal to the Mamba for trade management. Along the way it (1) built a
55-stream signal league scored on direction-agreement with AI labels, (2) confirmed the
λ-completion thesis on the recovered 9-tier NMP ladder, (3) closed the turn-timing problem
(no static detector beats the ±2m chance line → sequential/Mamba lane), and (4) at full power
closed the entire CUT side of trade management — no stop/veto/LLM-bail beats never-bail; the
harvestable edge is entirely "let winners ride." All numbers below are direction-agreement AUC
or forward-drift points (train 2024 / test 2025+26, day-block bootstrap CIs) — NOT $/day, and
NOT deployed. KEEP-LIVE.

## Key numbers (with CIs where they exist)
Signal league (`reports/dossier_signal_league.md`; OOS AUC, base 0.50):
- Genuine separators (thin-N, actionable tails): PIVOT-16 0.939 (base 0.05, N=324, invertible),
  OHLC-01 **0.841** (N=619, both tails act; the star), VP-01 0.732, VWMA-10 0.714, ADX08 0.660.
- Dense workhorses: TMPL0 0.631 (N=157k), ROUND-05 0.623 (N=44k), SAR-23 0.618, RENKO-24 0.611
  (N=198k), PTRNENGULF 0.616, DOW-19 0.610, VWAP-03 0.604, CURVE 0.606, TURNHA 0.615.
- **PROP-TURN-P 0.689** (N=131k test) — strongest single dense feature, matches the whole pool.
- Dead/tautology: ATR-09 0.500 (pure inverter), ORB-02 0.436, RSI-06 0.515 (base 0.04).
Combiner (`reports/combiner_preview.md`): pooled OOS AUC **0.676** across 55 streams, N=1.07M
(0.687 on the earlier 12-stream pool); honest monotone calibration OOS — bottom decile observed
0.20 [0.20,0.21] (invert = 80% right), top decile 0.75 [0.73,0.77].
Economic conversion (`reports/econ_conversion.md`): P(right) converts to POINTS monotonically,
correct sign. Clean cell = top-decile @5m: mode +1.0, median +3.25, mean **+3.86 pts** ($7.72)
CI[+2.48,+5.06], net-of-0.6-friction +3.26. Bottom-decile inverted @5m +1.33 CI[+1.00,+1.65].
Tradeable window SHORT (1–5m); CIs blow out past 15m. Gate: Mamba handoff JUSTIFIED for
short-horizon management of the tails.
λ / NMP9 (`reports/nmp9_ladder.md`): ride family aligned 0.76–0.85 (FREIGHT 0.854, RIDEAGAINST
0.789, RIDEMOM 0.810, RIDECALM 0.781, FADEAGAINST 0.758); pure-fade family anti-aligned
0.17–0.30 (KILLSHOT 0.172, FADEMOM 0.206, FADECALM 0.289 — the naive fade is 71–83% WRONG on
direction, invertible; = doc-084's 0.26 pure-fade result). λ̂ head (trailing OLS slope k=21 of
log(|z_se|+0.1)) flips anti-aligned fade → aligned ride; λ̂'s SELECTIVE edge is +0.070 in the
calm regime, negligible +0.016 in momentum (most alignment comes from the flip, not λ̂).
NMP9 retune (`reports/nmp9_retune.md`): quantile-matched thresholds re-center occupancy but
change no verdict; combiner delta **+0.0006** — structure already captured, threshold-robust.
Turn detection (`reports/turn_detection_audit.md`, chance precision@2m 0.43): 46 detectors +
409-dim snapshot + static & dynamic proportional geometry ALL fail. Best real precision
TURN-CLIMAX 0.31; best coverage RENKO24 recall 0.30 precision 0.17; EXIT-KMDR leads −0.2m. The
turn is a SEQUENTIAL/path object.
Overfit decay (`reports/overfit_decay.md`): combiner shelf-life — 8wk windows median 37wk
(mode 7); 16wk windows median 57wk; ~half right-censored → true life LONGER. Edge is durable.

## Graveyard / never-retry (this program generated these)
- **All 24 NT8 catalog concepts = closed honest null** (doc 045): no realizable edge, no
  post-event drift both years, canonical engine. ORB-02 was the last survivor → LOOKAHEAD
  (index-space bug: 09:00 vs 08:30 slice → 30-min offset; %>0=1.00 tell). 6 artifact classes
  documented (index-space ×3, stored-excursion, unrealizable-peak, label-leakage).
- **CUT side CLOSED at full power** (doc 107, N=23,378, 282 test days): every dumb stop
  X=8..48 nets −6.9…−3.4 t/ep (all CIs ≤0); re-entry −4.50 [−9.55,+0.86]; path-veto −0.72
  (≈never-bail); blind LLM bail loses. Doc 100's "+17.7 dumb-stop edge" was a CLASS-BALANCE
  ARTIFACT (1:1 set; natural mix 49.7/43.2/7.1). Law: on top-decile entries, losers cut
  themselves — every overlay pays its toll on the 43% goods and can't earn it back on wrongs.
- **Proportional-turn family CLOSED** (doc 095): PROP-TURN-P PASSED the literal kill-rule but
  by fire-rate saturation (425/day firehose, modulation inert, precision below 0.43 chance,
  capture −0.88 pt/trade WORSE than static). Promoted as a FEATURE only.
- **Bracket SL/TP dead** (`reports/bracket_grid.md`): only positive cells are fat-right-tail
  (sealed pop-A cell mode −21 / median −20 / mean +2.06 — the typical trade LOSES); + 3
  external replications. No management beats the no-stop baseline.
- **P_hold F-space exit** (`reports/phold_exit_model.md`): full 409-dim V2 vector AUC 0.638 vs
  0.685 context-only baseline (delta −0.047, BELOW the 0.05 house bar); no P_hold policy beats
  fixed-5m median capture (oracle ratio 0.23). F-space adds nothing over trivial during-trade
  state.
- **Exit dojo blind** (doc 098, 200 eps, nonce-audited): LLM exit does NOT beat dumb 5m-hold
  overall (delta +3.9 [−1.0,+9.3] ns, beat 40%); only edge = winners +19.5 [+8.3,+32.1] sig;
  wrong-side = sig LOSS −9.6. The confluence grammar marks the turn REGION, not a tradeable edge.
- Volume-over-time / wick-body shapes / band-bounce / order-flow delta / touch-count / APZ
  re-entry — all DEAD (`reports/TESTED_VS_UNTESTED.md`).

## Reusable assets
- `tools/dossier_signal_pipeline.py` — the 55-stream causal-signal league engine (all generators).
- `tools/nmp9_league.py`, `tools/nmp9_quantile_match.py`, `tools/nmp9_probe_2024.py` — NMP9 ladder.
- `tools/propturn_p_tune.py` — P-modulated proportional-turn tuner (feature block for the Mamba).
- `tools/combiner_preview.py` — pooled P(right) calibrator; `tools/league_merge_from_rows.py`.
- FPS event-dataset tooling (128–156k bars/s) + the audited per-dossier event datasets.
- Exit/wrongdir dojo harness: `builders/telescope_packet_builder.py`, `tools/{dojo_gate,dojo_fleet,
  score_full_run,synthesize_full_run,wrongdir_fleet,score_wrongdir}.py` (scoped gate-only allowlist).

## Data locations
- `reports/signal_rows_*.parquet` (55 streams, gitignored — regenerate via dossier_signal_pipeline.py).
- AI labels cover 604 5s days (576 labeled); IS=2024, OOS=2025+26.
- Frozen params (no pickle): `reports/propturn_p_frozen.json`, `reports/nmp9_retuned_constants.json`.
- `reports/overfit_decay_rows.parquet`, `reports/{econ_drift,bracket_fills}.parquet`.
- Dojo transcripts: `research/exit_dojo/reports/full_run/`, `reports/wrongdir/`.

## Open threads
- **Mamba handoff** — the whole program funnels here: state = per-stream time-since-fire ×
  direction × P + PROP-TURN-P leg geometry (age/A/ER/giveback); exit head trains on RIDE-LENGTH
  ONLY (cut-head = never-bail per doc 107). Ceiling is honest: alpha from riding winners, not
  timing turns. You cannot label "winner" at entry → the real-time "is-this-running" (λ) read.
- **Genuinely untested** (`TESTED_VS_UNTESTED.md`, volume AT price ≠ volume over time): Volume
  Profile POC/Value-Area, prior-day OHLC / floor pivots, RSI/MACD divergence, squeeze→break as a
  SEQUENCE, VWAP-z as explicit gate, seasonality/day-of-week.
- Disk-space fragility flagged during the overnight league run (machine hit 0 bytes free).

## Sources
- `reports/dossier_signal_league.md`, `reports/combiner_preview.md`, `reports/econ_conversion.md`
- `reports/nmp9_ladder.md`, `reports/nmp9_retune.md`, `reports/turn_detection_audit.md`
- `reports/bracket_grid.md`, `reports/phold_exit_model.md`, `reports/overfit_decay.md`, `reports/TESTED_VS_UNTESTED.md`
- comms: 081 (night league), 045 (ORB lookahead / catalog null), 095 (propturn), 098 (exit dojo),
  100 (wrongdir) → superseded by 107 (SYNTHESIS: cut closed, ride is the program)

## Archive recommendation
**KEEP-LIVE — by definition.** This is the program's active spine (the Mamba handoff, the
combiner, and every standing graveyard rule flow from here). The card's value IS the consolidated
verdict ledger above; the folder stays put regardless of any per-report deadness.
