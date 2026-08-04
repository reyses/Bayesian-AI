# Reversal Gauge v0 — combiner fit report

- events total: 118015
- resolved (label_resolved==1): 118009
- dropped non-finite feature rows: 0
- used: n=118009 events across 539 days
- base rate p(resume): 0.3363

## Cross-validated AUC (GroupKFold 5 by day, label_resume)

- fold AUCs: 0.6546, 0.6501, 0.6519, 0.6457, 0.6517
- mean +- std: 0.6508 +- 0.0029

## Baselines

- giveback_frac alone (same CV protocol): 0.6510 +- 0.0034 (folds: 0.6555, 0.6503, 0.6519, 0.6450, 0.6521)
- constant predictor: 0.5000 (0.5 by construction)
- combiner delta over giveback-only: -0.0002

## Reliability (10-bin, out-of-fold predictions)

| bin | n | mean predicted | observed resume rate |
|---|---|---|---|
| [0.0, 0.1) | 6369 | 0.052 | 0.080 |
| [0.1, 0.2) | 10370 | 0.154 | 0.150 |
| [0.2, 0.3) | 19114 | 0.255 | 0.230 |
| [0.3, 0.4) | 37634 | 0.356 | 0.348 |
| [0.4, 0.5) | 43960 | 0.436 | 0.452 |
| [0.5, 0.6) | 562 | 0.514 | 0.486 |
| [0.6, 0.7) | 0 | - | - |
| [0.7, 0.8) | 0 | - | - |
| [0.8, 0.9) | 0 | - | - |
| [0.9, 1.0] | 0 | - | - |

## Coefficients (standardized features, sorted by |coef|)

| feature | coef |
|---|---|
| giveback_frac | -0.8802 |
| spike_score | +0.0555 |
| pace_pts_s | +0.0290 |
| clock_sin | -0.0173 |
| is_flushV | +0.0123 |
| clock_cos | +0.0122 |
| worn_touches | -0.0080 |
| repoke | +0.0022 |
| (intercept) | -0.7920 |

VERDICT: cv AUC 0.651 +- 0.003 clears the program's 0.57 ceiling by more than one fold-std — a real break, pending replication.

## POST-AUDIT VERDICT (2026-08-04, after 3-agent adversarial review + day_class fix)

Three auditors: causality (2 flaws found in day_class — prior-evening open
and a dead window-close guard, BOTH FIXED and re-run: AUC unchanged), label
(all PASS, 3 events hand-verified from raw bars), stats (all arithmetic
reproduced exactly; one interpretive flaw, confirmed on fixed data):

1. **The pooled 0.651 AUC is mostly mechanical label coupling** — giveback
   at event time is the starting position of the race that defines the
   label. On the honest cut (giveback < 0.45): **AUC 0.5755 ± 0.0057**
   (n=82,577) — exactly AT the program's 0.57 oscillator/runaway wall.
   The wall is real, now measured at 118k-event scale.
2. **The combiner is giveback_frac alone** (delta −0.0001). With day_class
   FIXED, flushV still contributes nothing pooled (base rate 0.329 vs
   0.339). Day-shape conditioning pays at the SPECIFIC-event level (the
   defended V-floor poke: 1.4% crack vs 21% unconditional) but not as a
   linear flag over generic giveback events — conditioning must be
   event-specific cohort lookup, i.e. the Bayesian-table shape, not a
   pooled model.
3. **What ships as v0**: a CALIBRATED giveback→p(resume) readout
   (reliability table is clean: predicted matches observed within ~2-3pp
   per bin). It is a cockpit gauge, not an edge — it tells the owner what
   the base rate says at the current giveback depth.
