# Forward-return probe — does causal edge exist? (linear, walk-forward)
ATLAS 1m: 576 days, 723,995 bars. K=5-bar target. 60d train / 20d test, step 20 = 25 folds.

- **OOS Information Coefficient (per-test-day mean): +0.0053**, 95% day-block CI [-0.0026, +0.0133] — not distinguishable from 0 (efficient at this scale)
- directional accuracy: 49.3% (50% = chance)
- **NONLINEAR (gradient boosting) OOS IC: +0.0008** over 491 days
- IC drift slope across folds: +0.00030/fold (stable)
- test days: 500

## Verdict for the blackboard
No linear causal edge at this horizon/featureset. Before shelving: a nonlinear/sequence model MIGHT find structure a linear probe cannot (interactions, temporal) — but the linear null is a strong prior that the sequence is near-efficient. The blackboard would be chasing a faint or absent signal.

Honest scale note: an IC of ~0.005 is tiny — even if real, tradeability depends on turnover vs the ~3.6-tick round-trip cost. Predictability != profitability.

## SYNTHESIS (2026-07-26) — negative on raw return, but it aims the blackboard
Linear IC +0.005 (CI incl 0) AND nonlinear GBM IC +0.001 — BOTH ~0. The 1m
price sequence is EFFICIENT for naive forward-return prediction. This is
reassuring discipline, not failure: if a gradient-boost could predict 1m
returns from price, the edge would already be arbitraged away.

BUT this probed the WRONG target + POOR features:
- target = raw forward RETURN (price direction). Our validated edge is
  TRADE-OUTCOME given an entry (a conditional) — a different question.
- features = simple OHLCV. Our edge lives in the RICH F-space (reversion_prob,
  ldist, lambda, leg-pure instruments) — NOT in raw ATLAS OHLCV.

We ALREADY have a POSITIVE probe on the RIGHT target+features: the
wrong-direction study (rich F-space + trade-outcome) passed tune/holdout at
73% precision. So the blackboard's mamba should target TRADE-OUTCOME / REGIME
from rich derived state, NOT next-tick return. This probe's payoff = ruling out
the naive framing and aiming the loop at the framing that demonstrably works.
NEXT probe: rich F-space features + trade-outcome target, nonlinear/sequence,
walk-forward.
