# Six turn/exit ports scored — none clear the bar; two are best-in-class; the turn
# problem is now formally resistant to static detection
**Doc:** 092 · **Date:** 2026-07-17 · **Author:** Claude (reviewer) · **Status:** FINAL
**Executor:** Opus worker (ladder trial #8). Sources: TURN_CATALOG (10/07/06) +
EXIT_CATALOG (05/06/04). Reviewer verified the audit rows against the regenerated
turn_detection_audit.md — exact match.

## 1. Results (test 2025+26; kill bar: precision@2m > 0.43 chance OR dir-recall@2m
## ≥ 0.35 with lead ≤ +1 min)
| stream | fires/day | league AUC (base) | dir-recall@2m | prec@2m | lead med | verdict |
|---|---|---|---|---|---|---|
| TURN-HA | ~98 | 0.615 (0.57) | **0.16** | 0.13 | +1.1m | fail — joint-2nd-best detector, lagging confirmer |
| EXIT-KMDR | ~42 | 0.576 (0.13 inv) | **0.16** | 0.20 | **−0.2m** | fail — joint-2nd-best AND the only strong LEADING stream |
| CTX-ER | ~43 | 0.561 (0.41) | 0.13 | 0.18 | +1.0m | fail — ties SAR-23 |
| TURN-CLIMAX | ~5 | 0.556 (0.33 inv) | 0.03 | **0.31** | +0.7m | fail — HIGHEST precision in the entire catalog, still < null |
| TURN-SWEEP | ~5 | **0.639** (0.53) | 0.01 | 0.19 | +0.3m | fail — sharpest league AUC of the six; coverage-starved |
| EXIT-TIMESTOP | ~5 | 0.533 (0.51) | 0.01 | 0.13 | −0.2m | fail — near-chance (flagged weakest pre-build) |

## 2. What survived the failure (three keepers for the combiner/state)
1. **TURN-HA** — 2nd-best turn coverage in the catalog (0.16 vs RENKO's 0.30
   firehose) + a solid league stream (AUC 0.615 on 52k fires). The article's own
   caveat ("HA recolors late") measured: +1.1 min lag.
2. **EXIT-KMDR** — 2nd-best coverage AND the only strong stream that **leads the
   turn** (median −0.2 min). Another inverter at the league level (base 0.13 —
   reversal-at-band fires mid-leg), but its TIMING is the most exit-shaped signal
   we own. The WPI thesis parameters used verbatim.
3. **TURN-CLIMAX** — the purest fires anywhere (precision 0.31); rare (~5/day).
   A high-conviction confirmation mark, not a coverage tool.

## 3. Structural verdicts
- **The ±2m precision clause is unreachable in principle for this corpus**: with
  ~24 turns/day, RANDOM placement scores 0.43; the best real stream scores 0.31.
  No event-detector's fires concentrate near turns. The operative bar is
  dir-recall, and nothing reaches 0.35 (RENKO 0.30 is the ceiling, via density).
- **46 static detectors have now been scored on the turn problem; none solve it.**
  Combined with doc 089 (409-feature static snapshot loses to path-trivial state),
  the evidence says: at the ±1-2 min scale, turns are visible only in PATHS.
  The sequential lane (Mamba) is not one option among several anymore — it is
  the remaining lane, now equipped with better inputs: TURN-HA (coverage),
  EXIT-KMDR (lead), TURN-CLIMAX (purity), CTX-ER (chop context), plus the
  free 4-feature P_hold state and fire-freshness.
- Declared choices all documented in the pipeline docstrings; EXIT-TIMESTOP's
  20-min window remains illustrative (not fit to our MFE distribution) —
  refit before drawing any conclusion from its null.
- League md currently scoped to the 6 new streams (run artifact); the full
  46-stream consolidated table = git history (0ad35951 lineage) + a
  league_merge_from_rows re-run when next needed.

## 4. Ladder trial #8: PASS
Synchronous run honored, kill bar applied without tuning, declared choices
explicit, 0-of-6 reported plainly. Eight trials, eight disciplined executions.
