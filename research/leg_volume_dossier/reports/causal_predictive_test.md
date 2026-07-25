# Causal predictive test — is the climax+flat signal actionable?
signal = leg-pure vol climax (z>=2.0) fired >=2 bars ago AND 1m velocity <= 0. Prediction: leg peak within 3 bars.
N = 4497 frame-observations, 297 signal-on, 25 days (day-block bootstrap, 4000 resamples).

| | signal ON | signal OFF |
|---|---|---|
| P(peak within 3 bars) | 19.2% | 13.0% |
| mean fwd px over 3 bars | -6.16 pts | +4.42 pts |

**LIFT** = +6.1%, 95% CI [-0.2%, +12.7%] — **NOT significant** (CI includes 0)
**FWD-PX delta** = -10.58 pts, 95% CI [-17.23, -3.27]

Decision rule (pre-stated): CI(lift) > 0 => ship as (1) knowledge-pack v2 line, (2) student feature spec, (3) control-plane strike input. Else => descriptive-only; do not ship.
