# DECISION PACKAGE — Interim NT8: Architecture A vs B (for Moises)
**Doc:** 126 · **Date:** 2026-07-18 · **Author:** Claude Fable · **Status:** DECISION-PENDING (owner)
A = python sensor -> thin NT8 executor (bridge pattern). B = native NinjaScript port.
The system being deployed just got maximally simple (docs 107/125): combiner P
entry + R-trigger exit + B9 sizing. NO cut logic. NO turn detection.

## Evidence table
| criterion | A (sensor+thin executor) | B (native port) |
|---|---|---|
| decider fidelity | THE tested python decider runs live unchanged | full C# re-implementation of 185D features + combiner + R-trigger; the NMP9 retune proved thresholds do NOT survive re-implementation (wick pair was 3x off python->python) |
| parity burden | one boundary to verify (bridge I/O) | every feature, every threshold, forever, on every change |
| retune cadence (monthly per shelf-life) | swap a coefficients file | recompile + redeploy .cs each time (deploy-gate friction) |
| operator failure modes | external corpus: NT8 in-strategy stop/trailing logic is where retail bots break -> keep NinjaScript surface MINIMAL | that failure surface IS the product |
| robustness | python process can die (this week: 3 restarts killed daemons) -> needs watchdog | runs standalone |
| latency | decisions at 1m cadence; bridge round-trip irrelevant at this horizon | instant, but nothing here needs it |
| precedent | BayesianBridge already deployed; engine_v2 is the live path | v1.0 ZigzagRunner native (released); BaseNmpRunner port exists |

## Recommendation: **A, with B''s one real virtue bolted on**
Thin NT8 executor that ALSO carries native safety rails: a catastrophic stop
+ flatten-on-disconnect (if the bridge/python dies, NT8 protects the position
natively). Python keeps the brain; NT8 keeps the parachute. This neutralizes
B''s only strong argument (process-death risk — which this week demonstrated
three times) without buying its parity burden.

## Phased rollout (if A approved)
1. SIM parity: bridge decider vs offline forward-pass on the same days
   (the known Python-vs-NT8 gap ~$680/day is exactly what this phase measures
   under the multi-gap-assumption rule — never a single doom number).
2. Live SIM with halt-after-N + drawdown caps (intervention rules on).
3. Micro live, one contract, deploy-gate per revision (versioning policy).

## What I need from Moises
- A or B (or A-with-rails as recommended).
- If A: green light to spec the thin executor .cs (v1 = entry orders from
  bridge signals + R-trigger mirror + catastrophic stop + disconnect-flatten)
  — spec first, NO deploy without per-revision approval per the gate.
