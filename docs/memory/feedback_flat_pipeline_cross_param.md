---
name: flat-pipeline-cross-param
description: The FLAT hardened-leg forward pass is a hindsight-clean partition — its $/day is monotonic in zigzag subdivision and CANNOT be used to compare across ATR / pivot-density parameters.
metadata:
  type: feedback
---

The FLAT (no-ML) hardened-leg forward pass — enter every offline-zigzag leg at
its R-trigger, exit at the next pivot — must NOT be used to compare across the
ATR multiplier, or any parameter that changes the zigzag's pivot density.

**Why:** the offline zigzag's legs are a SEQUENTIAL PARTITION of the price path
(see [[midleg-entry-research]]: 99.8% of consecutive-leg gaps = 0). Every leg is
a genuine pivot-to-pivot swing — zero whipsaws by construction, because the
offline detector places pivots with knowledge of the whole day. A causal engine
flips at confirmations, many of them false, and whipsaws. As the ATR multiplier
(→ r_price) shrinks, the offline pass finds ever more clean swings while the
causal whipsaw cost it never pays explodes. The 2026-05-21 sweep
(`tools/atr_multiplier_sweep.py`) showed FLAT OOS $/day rising MONOTONICALLY
from −$167 (ATR×10) to $3,480 (ATR×1 — 100% winning days, PF 5.25), no interior
peak. That is the fingerprint of a metric scaling with zigzag SUBDIVISION, not
tradeability.

The X=4 FLAT baseline ($690/day IS, $454/day OOS) is ~trustworthy ONLY because
at a coarse threshold offline ≈ streaming (the validated ~98% match). The whole
L5 stack is validated at a fixed X=4, where the oracle inflation is a constant
that cancels inside deltas.

**How to apply:** FLAT hardened-leg $/day is valid as (a) a fixed-X baseline and
(b) the denominator of a fixed-X delta (B7 / B9 / B10 lift). It is NOT valid for
cross-X or any cross-pivot-density comparison. To compare ATR multipliers (or
any pivot-density parameter) you must use a CAUSAL streaming forward pass that
actually pays the whipsaw cost — only that has a real interior optimum.
`tools/atr_multiplier_sweep.py` is fine for fixed-X use; its cross-X verdict is
annotated invalid in both the script and the report.
