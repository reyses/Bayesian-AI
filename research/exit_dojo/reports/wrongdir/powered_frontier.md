# The POWERED cut frontier -- FULL 2025-26 test population (Task 106)

PURE EVALUATION. Nothing is retuned: every policy is frozen (plain-stop grid + re-entry 48/4/1 + veto 24t p*=0.45 loaded verbatim from veto_frozen.json). The 198-episode dojo set was 1:1-balanced and one-per-day (sized for LLM-fleet cost); doc 105 showed the plain stop's +17.7 t/ep there was CI[-12.4,+46.7] -- NOT significant. Here the SAME frozen policies run on the WHOLE test tape (every engagement, natural class mix, dead-band included) so the cut question gets real statistical power.

## Population + natural class mix
- **N = 23378 engagements** over **282 distinct test days** (2025-26; select_wrongdir.engagements(): P>=p90(train)=0.76023 frozen, 60s/day/dir de-dup, MIN_WINDOW=15m). NO one-per-day dedup, NO 50/50 balance -- this is the deployment tape.
- BAND=4pts (WRONG terminal<=-4, GOOD terminal>=+4), DIP=4pts (dipped = min drift <= -4).

| class | N | share |
|---|---|---|
| wrong | 11615 | 49.7% |
| good_dipped | 5912 | 25.3% |
| good_clean | 4188 | 17.9% |
| dead_band | 1663 | 7.1% |
| **total** | **23378** | 100% |

The 198 dojo set forced 50/50 wrong/good with distinct days. The natural tape is 50% wrong, 43% good (25% dipped / 18% clean), 7% dead-band. Dipped-goods -- the knife the re-entry repairs -- are 25.3% of the tape here, vs 25% in the balanced set.

## Friction convention (charged consistently across ALL policies)
net-vs-never-bail is friction-FREE: every single-round-trip policy trades exactly once, so the 2.4t/RT cancels against never-bail's one RT (doc 100/105 convention). stop+re-entry's EXTRA legs pay incremental friction inside the sim. The ABSOLUTE column re-adds it honestly via abs = net + terminal_ticks - 2.4, which charges exactly one RT for single-leg policies and n_legs RTs for re-entry. So 2.4t/RT is charged identically everywhere and is a constant offset in every delta.

## THE FRONTIER (net ticks/ep vs never-bail; day-block 95% CI, 4000 resamples; * = CI excl 0)
| policy | mean net | 95% day-block CI | median | mode | mean ABS w/friction |
|---|---|---|---|---|---|
| never-bail | +0.00 | [+0.00, +0.00] | +0.0 | +2.0 | +9.36 |
| stop X=8 | -6.85 | [-13.77, +0.14] | +0.0 | +2.0 | +2.51 |
| stop X=16 | -6.51 | [-13.31, +0.19] | +0.0 | +2.0 | +2.85 |
| stop X=24 | -5.36 | [-11.98, +1.14] | +0.0 | +2.0 | +3.99 |
| stop X=32 | -4.26 | [-10.38, +1.96] | +0.0 | +2.0 | +5.10 |
| stop X=48 | -3.39 | [-9.30, +2.48] | +0.0 | +2.0 | +5.97 |
| stop+re-entry (X=48,M=4,B=1) | -4.50 | [-9.55, +0.86] | +0.0 | +2.0 | +4.86 |
| stop+veto (24t, p*=0.45) | -0.72 | [-1.53, +0.08] | +0.0 | +2.0 | +8.64 |

Best plain stop on this full population: **X=48** (mean net -3.39 t/ep).

## Delta columns (day-block 95% CI; * = CI excludes 0)
| policy | delta vs never-bail | delta vs best stop (X=48) |
|---|---|---|
| stop X=8 | -6.85 [-13.77, +0.14] | -3.46 [-5.73, -1.26] * |
| stop X=16 | -6.51 [-13.31, +0.19] | -3.12 [-4.96, -1.37] * |
| stop X=24 | -5.36 [-11.98, +1.14] | -1.97 [-3.54, -0.52] * |
| stop X=32 | -4.26 [-10.38, +1.96] | -0.87 [-1.96, +0.18] |
| stop X=48 | -3.39 [-9.30, +2.48] | +0.00 [+0.00, +0.00] |
| stop+re-entry (X=48,M=4,B=1) | -4.50 [-9.55, +0.86] | -1.11 [-3.19, +0.96] |
| stop+veto (24t, p*=0.45) | -0.72 [-1.53, +0.08] | +2.67 [-3.08, +8.50] |

(delta-vs-never-bail equals the policy's own net since never-bail net==0 by construction; shown for completeness. delta-vs-best-stop isolates whether re-entry / veto add anything over the best plain stop.)

## Per-class decomposition (mean net vs never-bail, ticks/ep; CI = day-block 95%)
| policy | wrong (N=11615) | good_dipped (N=5912) | good_clean (N=4188) | dead_band (N=1663) |
|---|---|---|---|---|
| never-bail | +0.0 | +0.0 | +0.0 | +0.0 |
| stop X=8 | +139.8* | -273.5* | -27.0* | -32.2* |
| stop X=16 | +133.5* | -277.8* | +0.0 | -36.3* |
| stop X=24 | +126.8* | -259.2* | +0.0 | -39.3* |
| stop X=32 | +120.2* | -241.4* | +0.0 | -41.1* |
| stop X=48 | +107.3* | -212.4* | +0.0 | -41.8* |
| stop+re-entry (X=48,M=4,B=1) | +70.7* | -141.5* | +0.0 | -53.7* |
| stop+veto (24t, p*=0.45) | +10.1* | -20.6* | +0.0 | -7.4* |

(* = class-level CI excludes 0. WRONG: bail = money saved, net>0 expected. GOOD-dipped: bail = knifing a temporary dip, net<0 expected -- the trap. GOOD-clean: a stop rarely triggers, net~0. DEAD-BAND: near-scratch; whichever side the stop happens to catch.)

## The three questions (plain answers)

**(a) Does ANY cut policy beat never-bail with CI excluding 0 at scale?**
- **NO.** No cut policy's net-vs-never-bail CI excludes 0 on the positive side.

**(b) Does the doc-100 +17.7 (24t stop) edge survive the natural mix + the power increase?**
- 24t plain stop on the full tape: mean net **-5.36** t/ep, CI [-11.98, +1.14] (CI includes 0). The +17.7 does NOT survive the natural mix: the balanced 1:1 set over-weighted WRONG (where a stop pays), so the headline shrinks toward the natural-mix value.

**(c) Does re-entry's dipped-knife repair change sign at natural mix (dipped goods rarer than 1:1)?**
- stop+re-entry (48/4/1) full-tape net: mean **-4.50** t/ep, CI [-9.55, +0.86]. On GOOD-dipped (N=5912, 25.3% of the tape): re-entry -141.5 vs best-stop -212.4 t/ep. Delta vs best stop overall = -1.11 [-3.19, +0.96].

## Caveats (printed)
- **1m granularity**: drift is per-minute; intrabar stop/trigger crossings are invisible -- a real stop fires earlier and often deeper, a real re-entry trigger can fire-and-reverse within a minute. All numbers are 1m-resolution estimates, OPTIMISTIC about clean fills.
- **overlapping windows within a day**: the full tape has multiple engagements per day (23378 engagements / 282 days) with overlapping forward windows -- their P&L is NOT independent. The day-block bootstrap resamples DISTINCT days precisely to cover this dependence (the CI is wider and more honest than an i.i.d. per-episode bootstrap).
- **frozen-on-2024 params evaluated on 2025-26**: the re-entry 48/4/1 winner and the veto (coefs, scaler, p*=0.45) were sealed on 2024 train; the plain-stop grid is pre-registered (doc 103). Transfer risk was already demonstrated in docs 103/105 -- THIS full-tape frontier is the test of record for it. The veto AUC was 0.53 in-sample / 0.53 CV (below the 0.05 signal floor); its coefficients are noise-level, so stop+veto is expected to track either the plain 24t stop or near-never-bail depending on how often p*=0.45 vetoes.

_Path-sim frontier on the sealed test tape. A dojo/path number is a hypothesis, not a live result: any retained rule still graduates through the sealed harness (graduation firewall)._