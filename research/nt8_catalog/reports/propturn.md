# PROP-TURN — proportional leg-turn confirmation, stop-and-reverse
_Moises 2026-07-16 design. Tuned on 2024 ONLY (sealed); all read-outs below are TEST 2025+26. Generated 2026-07-16 21:08._

## TL;DR — verdict
- **Turn bar: FAIL.** Frozen-cell dir-recall@±2m 0.042 [0.039,0.046], precision@2m 0.102 (both far below the 0.35 / 0.43 bar; lead is NEGATIVE -0.58m).
- **Capture: FAIL the 0.5–0.8 budget decisively.** Capture-ratio median -0.05 (typical leg-trade goes slightly the WRONG way vs the label it sits in); only 2% of trades land in the budget. Gross ≈ coin-flip (PF Trade WR -0.021); net -0.80 pt/trade after 0.6-pt friction (CI [-1.21,-0.38]).
- **League: the one positive.** As a COMBINER feature the fires carry weak but real direction signal: OOS AUC 0.636, monotonic terciles.
- **Two structural findings drove the result** — see §0.

## 0. Structural findings (why the sealed cell is what it is)
**(a) The literal spec BREAKS stop-and-reverse (verification-driven; DECLARED DEVIATION).** After every flip the new leg starts with amplitude A = the triggering retrace (< A_min). If price then reverses hard, A stays frozen < A_min, the "A ≥ A_min to fire" gate permanently blocks the opposite turn, and the opposite branch is disabled by the leg direction — the tracker holds a LOSING position for the rest of the day. On the raw spec this zeroed 82 of 318 test days and *partially* stuck most others (8 sample 2024 days: 314 fires literal vs 783 de-stuck, 2.49×). FIX (in the shared `_propturn_core`, flagged for review): keep the proportional confirm EXACTLY as specified for real legs (A ≥ A_min), and add a sub-minimal ESCAPE — a leg whose amplitude never reached A_min is re-designated when a full A_min counter-move occurs. Escape fires are ~3% of fires directly but unlock the mechanic's true ~100/day rate. Without it, every number is a bug artifact.

**(b) The fires/day ≤ 60 cap forces a DEGENERATE cell.** With sticking cured, dir-recall scales with fire rate, and there is a sharp cliff in direction-correctness at S=3 (share of near-turn fires with the RIGHT direction, 2024, A_min=5): S=0 → 1.00, S=1 → 0.99, S=2 → 0.85, **S=3 → 0.28**, S=5 → 0.18. At S≥3 the long stall delays the confirm so far that the assigned "new leg" inverts vs the actual turn and the lead goes negative. EVERY feasible cell (fires ≤ 60) is S=3 or S=5 — the usable regime (S≤2, dir-recall 0.22–0.32) all fires 94–705/day. So the sealed winner is degenerate by construction, and even the un-capped "good" regime is largely fire-rate saturation (precision ~0.16). PROP-TURN is a firehose, not a ~45/day turn detector.

## Mechanic
Causal leg tracker on the continuous 5s close stream (tail+day, doc-073). Leg runs from pivot P0 to running extreme E; amplitude A=|E-P0|. TURN fires (stop-AND-reverse) when close retraces from E by >= r*A, subject to A>=A_min and a STALL gate (>= S min since E last improved). On fire: pivot->E, leg flips, fire direction = the NEW leg. State runs continuously (incl. overnight); emission RTH-gated.

## 1. Tuning (2024 SEALED) — top-5 feasible cells
Objective: max dir-recall@±2m on 2024 interior label turns s.t. lead-median ≤ +1.0 min AND fires/day ≤ 60. (90-cell grid: r×S×A_min.)

| rank | r | S (min) | A_min (pt) | dir-recall@2m | recall@2m | precision@2m | lead-med (min) | fires/day |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.05 | 3 | 15 | 0.038 | 0.119 | 0.096 | -0.55 | 56.9 |
| 2 | 0.08 | 3 | 15 | 0.036 | 0.116 | 0.095 | -0.53 | 56.4 |
| 3 | 0.1 | 3 | 15 | 0.035 | 0.114 | 0.094 | -0.53 | 55.9 |
| 4 | 0.15 | 3 | 15 | 0.029 | 0.104 | 0.090 | -0.50 | 53.9 |
| 5 | 0.2 | 3 | 5 | 0.027 | 0.107 | 0.086 | -0.35 | 59.5 |

**FROZEN winner:** r=0.05, S=3 min, A_min=15 pt

### Frozen cell — 2024 selection stats (the numbers it was chosen on)
- dir-recall@2m **0.038**, recall@2m 0.119, precision@2m 0.096
- lead-median -0.55 min, fires/day 56.9, on 11545 interior turns / 258 days

## 2. TEST turn scorecard (2025+26) — frozen cell
| metric | value |
|---|---|
| dir-recall@±1m | 0.016 |
| **dir-recall@±2m [CI]** | **0.042** [0.039, 0.046] |
| dir-recall@±3m | 0.236 |
| dir-recall@±5m | 0.305 |
| recall@±1m / ±2m | 0.063 / 0.119 |
| precision@±2m (chance 0.43) | 0.102 |
| lead@2m median / mode | -0.58 / -0.50 min |
| lead@2m p25 / p75 | -1.28 / +0.23 min |
| fires/day (test) | 53.4 |

**Standing-bar verdict: FAIL** — bar = precision > 0.43 OR (dir-recall@2m ≥ 0.35 with lead ≤ +1 min). Best prior stream (RENKO24) sits at dir-recall 0.30 / precision 0.17.

## 3. League line (full 604-day pipeline; direction-agreement with AI labels)
- N=30928 (train 14386 / test 16542), OOS **AUC 0.636**, test base 0.57
- P-terciles: low: 0.46 [0.45,0.47] N=5514 | mid: 0.54 [0.52,0.55] N=5514 | high: 0.71 [0.69,0.72] N=5514
- coefs: {'pivot_age_min': -0.021, 'sig_with_leg': 0.318, 'value': -0.299, 'tod': -0.031, 'inter': -0.031}

## 4. CAPTURE — pure stop-and-reverse (TEST; the 50–80% budget headline)
Position flips at each fire (close fills); flat outside RTH (force-close 15:15, re-open at next fire). Per completed leg-trade: captured points (signed). Friction line = 0.6 pt/round-trip (MNQ 1 pt = $2).

| pop | trades/day | captured mode | median | mean [CI] (pt) | PF Trade WR | net mean [CI] (pt) |
|---|---|---|---|---|---|---|
| 2025 | 58.2 | -14.50 | -3.25 | -0.26 [-0.75, +0.19] | -0.027 | -0.86 [-1.35, -0.41] |
| 2026 | 61.7 | -0.50 | -3.25 | +0.05 [-0.92, +1.08] | +0.006 | -0.55 [-1.52, +0.48] |
| POOLED | 58.9 | -14.50 | -3.25 | -0.20 [-0.61, +0.22] | -0.021 | -0.80 [-1.21, -0.38] |

### Capture ratio — captured / single-overlap label displacement
Reference points (from prior turn work): fixed-5m top-decile ≈ +2.00 pt median (deduped); oracle exit ≈ +27.5 pt median, ratio ≈ 0.23; user budget = 0.5–0.8.

| pop | N (1-overlap) | ratio mode | ratio median | frac in [0.5,0.8] | frac > 0 |
|---|---|---|---|---|---|
| 2025 | 9950 | -0.05 | -0.05 | 0.02 | 0.38 |
| 2026 | 2509 | -0.05 | -0.05 | 0.02 | 0.37 |
| POOLED | 12459 | -0.05 | -0.05 | 0.02 | 0.38 |

**Capture-ratio vs the 0.5–0.8 budget: median -0.05 → BELOW budget.**

## 5. Honesty guards
- **Pseudo-replication:** fires within a day share ONE leg-tracker state → serially dependent; capture legs within a day are a stop-and-reverse chain. All CIs are day-block bootstraps (unit of independence = the day), never per-trade/per-fire.
- **No post-hoc test selection:** the cell was frozen on 2024 ALONE before any test number was computed, on the stated objective (max dir-recall@2m s.t. constraints). Other grid cells' TEST numbers appear ONLY in the clearly-labeled EXPLORATION table (appendix) and are never quoted as results — they exist to answer the design question "does any regime capture the budget" (answer: no), NOT to reselect a better cell.
- **Turn-bar vs capture read independently:** the standing turn-bar verdict and the capture/ratio read-out are reported separately; one passing does not carry the other.
- **Friction is real:** 0.6 pt/round-trip; at the observed trades/day the net line is what matters, not gross.

## Appendix (EXPLORATION — 2024 tuning grid, NOT results)
Top-15 of the 90 cells by 2024 dir-recall@2m (feasible flag shown). These are SELECTION-YEAR numbers; do not read as test performance.

| r | S | A_min | dir-recall@2m | recall@2m | precision@2m | lead-med | fires/day | feasible |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0 | 5 | 0.323 | 0.324 | 0.160 | +0.00 | 792.7 | False |
| 0.08 | 0 | 5 | 0.323 | 0.324 | 0.160 | +0.00 | 776.1 | False |
| 0.1 | 0 | 5 | 0.323 | 0.324 | 0.159 | +0.00 | 760.5 | False |
| 0.15 | 0 | 5 | 0.322 | 0.323 | 0.158 | +0.02 | 711.8 | False |
| 0.05 | 1 | 5 | 0.321 | 0.324 | 0.150 | +0.52 | 193.0 | False |
| 0.08 | 1 | 5 | 0.321 | 0.324 | 0.151 | +0.65 | 190.8 | False |
| 0.2 | 0 | 5 | 0.320 | 0.321 | 0.156 | +0.05 | 643.4 | False |
| 0.1 | 1 | 5 | 0.320 | 0.323 | 0.151 | +0.72 | 188.7 | False |
| 0.15 | 1 | 5 | 0.314 | 0.318 | 0.150 | +0.87 | 181.4 | False |
| 0.25 | 0 | 5 | 0.304 | 0.305 | 0.149 | +0.12 | 543.9 | False |
| 0.2 | 1 | 5 | 0.301 | 0.306 | 0.149 | +0.88 | 168.7 | False |
| 0.08 | 1 | 10 | 0.301 | 0.306 | 0.163 | +0.38 | 150.1 | False |
| 0.05 | 1 | 10 | 0.301 | 0.305 | 0.163 | +0.24 | 151.5 | False |
| 0.1 | 1 | 10 | 0.301 | 0.305 | 0.164 | +0.45 | 148.7 | False |
| 0.15 | 1 | 10 | 0.297 | 0.302 | 0.163 | +0.80 | 143.8 | False |

### EXPLORATION — does ANY stall regime hit the capture budget? (TEST; NOT a result)
Capture sim run on TEST for the frozen cell + the best NON-degenerate cells, to answer the design question directly. Shown per the honesty guard as exploration, not a result (these cells were NOT the sealed selection). Answer: **no regime captures the budget** — every one whipsaws to a slightly-negative capture ratio and coin-flip gross PF.

| cell | trades/day | capture median (pt) | PF Trade WR | capture-ratio median | frac in [0.5,0.8] | frac > 0 |
|---|---|---|---|---|---|---|
| FROZEN r.05/S3/A15 | 59 | -3.25 | -0.021 | -0.049 | 0.02 | 0.38 |
| r.10/S2/A5 | 98 | -2.50 | -0.031 | -0.034 | 0.01 | 0.38 |
| r.10/S1/A5 | 189 | -2.00 | -0.021 | -0.022 | 0.01 | 0.39 |
| r.10/S0/A5 | 976 | -1.00 | -0.055 | -0.009 | 0.00 | 0.41 |


### Declared choices
- value emitted per fire = the completed leg amplitude A (pts) — a natural strength scalar.
- Shared feature basis for the league logistic is the canonical DayCtx zigzag (pivot_age/sig_with_leg/tod), identical to every other stream — PROP-TURN supplies only the trigger times + directions + value.
- Capture fills use the fire bar CLOSE (causal; no intrabar peeking). Final leg each day force-closes at the RTH-close bar (≤15:15).
- lead uses the nearest fire (any direction) to the turn, matching turn_detection_audit.
- Run via the tool (import-driven), not `python dossier_signal_pipeline.py PROP-TURN`: the generator is appended after the module `__main__` block, so it registers on import (the tool's path) but not on direct script execution.