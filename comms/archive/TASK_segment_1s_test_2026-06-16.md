# TASK → Gemini — Small-scale segment regression on the CLEAN single-resolution F-space (2026-06-16)

Small, falsifiable test of a thesis we worked out tonight. NOT a core change — one day, one comparison.

## The thesis (why this test exists)
The V2 F-space mixes **8 timeframes treated as an ordinal/categorical variable** (TF #1..#8),
when scale is really a **continuous cardinal quantity** (window length in seconds). And every TF
feature is a **half-measurement** of ONE underlying signal: sample-starved, clock-phase-contaminated,
OHLC-lossy. Stacking 8 collinear slices of one stream and regressing them as if independent →
**multicollinearity** → the mess we kept seeing: SMEP betas exploding to |β|≈471, coin-flip beta
signs, ~25% IS→OOS cell survival, "fittability subsumed by vol", inconclusive DOE/segments.
**Hypothesis:** the segment regression was inconclusive partly because the *instrument* was a pile
of collinear half-measurements — not (only) because there's no signal. This test isolates ONE clean
resolution to see if the regression gets *cleaner / more stable*.

## The task — ONE day
Run the **segment regression** on a single active day (e.g. `2025_02_20`, your pick) **twice** and compare:
- **Arm A (CLEAN single-resolution):** features = the **base 5s family only** — `L1_5s_*`, `L2_5s_*`,
  `L3_5s_*` (optionally `L5_5s_*` distribution). **EXCLUDE `L2_5s_vwap_9`** (it's a session-aggregation,
  not a clean point-in-time feature). Consider also dropping `L2_5s_price_mean_9`/`vol_mean_9` (also
  window-means) — note whether it matters.
- **Arm B (full TF-mixed, the current mess):** all 8 TF families (the normal F-space), vwap excluded.

Data: `DATA/ATLAS/FEATURES_1s_v2/` (1s-anchored; NOTE the fastest computed family is **5s**, there is
no literal 1s-TF — so "1s features" = the 5s base family). Target = the segment's price-delta fit
(same as the existing stage1 pipeline), or a faithful minimal reproduction if the GPU pipeline is heavy.

## Report — the metrics that actually test the thesis (compare A vs B)
1. **Collinearity:** condition number of X (and/or max VIF) — is A much lower than B?
2. **Beta stability:** refit on first-half vs second-half of the day → correlation of the coefficients.
   (B had coin-flip signs; does A hold its signs?)
3. **Selection stability:** which features survive, and are they consistent across the halves?
4. **Fit quality / segments:** R², and the segment structure (#PRISTINE / RECOVERED / CHAOS, tier mix).
**Verdict gate:** A "wins" only if it is **less collinear AND more stable** at comparable fit. If A is
just as messy, the instrument wasn't the (only) problem — say so.

## DISCIPLINES (per `comms/CONTEXT_FOR_GEMINI.md`)
- **Causal / no lookahead.** **SEGMENT FIREWALL** — segments are non-causal, label-side only, DIAGNOSTIC.
- **Report DISTRIBUTIONS + MODE, never averages** (user, 2026-06-16: "average does not make sense to me") —
  if you show any $ or per-trade metric, lead with the histogram/mode; mean only with its CI, never the headline.
- No magic numbers. Write results to `reports/findings/segment_1s_test.md`; update `docs/daily/INDEX.md`.
- **PUSH BACK** if the A-vs-B comparison isn't apples-to-apples (same target, same N, same method) or the
  test design is flawed — don't just run it. Catching a flaw is the high-value move.
- **Close with a SUMMARY + the exact file LOCATION(s)** of what you wrote.

## Honest framing (don't over-claim)
This is *necessary-not-sufficient*: a cleaner single-resolution ruler removes the collinearity confound,
but it does NOT conjure signal — the vol-ceiling and base-rate kills still stand. The real prize is
settling "**was it our collinear instrument, or no signal?**" If A is dramatically cleaner, it justifies
prototyping the full window/scale-space redesign next; if not, the redesign is off the table. The
IS/OOS picture is the ultimate judge, but this one-day collinearity/stability cut is the cheap first read.
