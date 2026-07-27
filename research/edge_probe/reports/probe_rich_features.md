# Rich-feature probe — does the F-space carry edge raw price does not?
4,182 frames, 25 episode-days, 42 rich features (abs-price-level fields excluded). Fwd 5-frame px target, walk-forward by day, OOS.

- **LINEAR OOS IC: +0.0267**, 95% day-block CI [-0.0782, +0.1241] incl 0
- **NONLINEAR (GBM) OOS IC: +0.0497**, 95% CI [-0.0353, +0.1268] incl 0
- baseline (raw-price probe): IC ~0.005 (not sig)
- test days: 16

## Verdict
Even the engineered F-space shows no OOS forward-px edge at this horizon on the packet data. The hand-built wrong-direction/outcome signals may be the ceiling; a heavy learned loop is not yet justified. Caveat: 150-episode packets, forward-PX target (not the trade-OUTCOME target the wrong-dir edge used) — retest at scale + on the outcome target before concluding.

## CORRECTED SYNTHESIS (not a null — underpowered-suggestive)
The auto-verdict fired the "not sig" branch, but the honest read is more
interesting:
- Point estimates: linear IC +0.027, nonlinear GBM **+0.050** — vs the
  raw-price null of ~0.005. The rich F-space looks ~5-10x more predictive than
  raw price at the point estimate. That is NOT the flat ~0 raw price gave.
- CIs include 0 ONLY because there are 16 test days (packet data: 25 days,
  expanding window). Underpowered, exactly as pre-flagged.
- Nonlinear > linear (+0.050 vs +0.027) hints at interaction structure a
  flexible model captures — consistent with the blackboard premise.

VERDICT: suggestive-but-underpowered. Do NOT conclude "no edge" (point
estimates argue against that) and do NOT claim edge (not significant). This is
precisely the case that MOTIVATES the at-scale test: materialize the 185D
F-space across ATLAS history (build_dataset — user-run, heavy/GPU) and re-probe
with real power, on both the forward-px AND the trade-OUTCOME target. That
powered test is the true blackboard go/no-go.
