# Execution Report — Phase-5 Entry Discriminator (executed by CLAUDE)
**Doc:** 035 · **Date:** 2026-07-11 · **Author:** Claude (reviewer, acting as EXECUTOR — AG session out of usage) · **Status:** FINAL

Moises directed me to execute the pending work myself. I did the highest-value,
lowest-risk slice: a **leakage-free entry discriminator** answering the doc-027
question directly (can entry F-space separate good ATR-09 fades from bad?). The
heavier queue items (B1 depth leakage ×15, B2 OHLC-01, B3 RSI-06, full multi-TF
ladder) are NOT done — left for AG's return or a follow-up (see §4).

## 1. Bug I found and fixed (the reason the prior Phase-5 was meaningless)
`tools/ag_phase5_final.py` concatenated **PhE + PhXit + PhPost** into one feature
vector to predict `hit`. PhXit is at the resolution bar and PhPost is AFTER it —
`hit` is DETERMINED at resolution. The model was predicting the outcome from
features measured at/after the outcome = lookahead contamination. Any separation
it showed was circular. (Evidence: `tools/ag_phase5_final.py` lines 43, 80-84.)

New script `tools/ag_phase5_entry_discriminator.py`:
- **PhE only** (entry anchor) — causal, live-usable.
- **Day-block bootstrap** (4000, resample days not events) — the prior code
  resampled events (pseudo-replication).
- Thresholds p_lo/p_hi (15/85 pctile) frozen on 2024, evaluated once on 2025.
- Validity gate = branch N≥30 AND ≥20 days AND day-block CI excludes 0 AND
  |mode| ≥ 2 pts. (First pass lacked the N floor and falsely marked a VA-13
  N=3 branch VALID — fixed; that is exactly the noise-as-signal trap.)

## 2. Result (honest; artifact: reports/AG_cat_00_PHASE5_ENTRY.md)
| Dossier | branch | N (days) | count-WR | EV pts | day-block CI | mode | verdict |
|---|---|---|---|---|---|---|---|
| ATR-09 | ACT | 64 (59) | 0.11 | +2.71 | [−7.65, +15.23] | −10 | not sig |
| ATR-09 | INVERT | 71 (59) | 0.90 | +2.36 | [−7.25, +10.00] | +11 | not sig |
| FIB-17 | ACT/INV | 3 / 35 | — | — | — | — | underpowered |
| VA-13 | ACT/INV | 5 / 3 | — | — | — | — | underpowered |

**Conclusion: no statistically significant entry discriminator on any of the
three, single-year OOS.** The honest state is a NULL, not an edge.

**The one lead worth pursuing — ATR-09 INVERT.** When entry F-space predicts the
fade will FAIL, riding the extension instead wins 90% by count with a +11-pt
mode (above friction) and a positive point EV. This is the doc-027 thesis
(λ>0 → ride) showing its face — but the day-block CI crosses 0 on 59 days, so it
is a hypothesis, not a finding. Two caveats stack: (a) underpowered (one year),
(b) INVERT EV is a MIRROR APPROXIMATION (−magnitude of the article-side trade),
not a simulated opposite trade with its own exits.

## 3. Caveats on what this is
- Features = **5s-timeframe V2 snapshot at entry** (L0-L5_5s, one bar/tier, 52
  dims incl. the NMP/λ family: z_se, hurst, lambda_hat). This is NOT yet the
  full multi-TF telescoping ladder (doc 017). The λ discriminators ARE present,
  so the negative result is informative: the 5s entry snapshot alone is not
  enough.
- FIB-17 (32 train events, 1 positive) and VA-13 (single feature survived) are
  too thin to model; reported only for completeness.

## 4. Still open (for AG / next session)
B1 depth-leakage re-derivation (15 dossiers, doc-034 mods), B2 OHLC-01 anchor,
B3 RSI-06 1948-pt trace, and the FULL multi-TF ladder + PhXit/PhPost DESCRIPTIVE
(non-predictive) conversion analysis. The ATR-09 INVERT lead argues for building
the fuller ladder and pooling both years before calling it dead.

Artifacts: `tools/ag_phase5_entry_discriminator.py`,
`reports/AG_cat_00_PHASE5_ENTRY.md`. Committed + pushed this turn.
