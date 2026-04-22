# Residual direction signal stratified by S/R context

**Current zigzag**: $15.0. **S/R zigzag**: $30.0 on prior 5 days.

**Question**: does residual signal strength depend on distance-to-nearest-S/R?

IS pivot events: 26,760 | OOS: 7,593

## Baseline (all pivots, unstratified)

| Dataset | N | d (res_10_norm) | Direction accuracy |
|---|---:|---:|---:|
| IS  | 26,760 | -2.460 | 86.8% |
| OOS | 7,593 | -2.400 | 86.0% |

Decision rule: predict UP if res_10_norm < 0, DOWN if > 0.

## Per-stratum results

Context strata by nearest-level-distance (points, $=pts×2):

- `AT_LEVEL`: 0.0–2.5 pts ($0–$5)
- `NEAR_LEVEL`: 2.5–7.5 pts ($5–$15)
- `MEDIUM_DIST`: 7.5–15.0 pts ($15–$30)
- `FAR_FROM_SR`: 15.0–50.0 pts ($30–$100)
- `VERY_FAR`: 50.0–9999.0 pts ($100–$19998)

Each row further splits by whether nearest level is ABOVE (resistance) or BELOW (support).

| Context | Side | IS N | IS d | IS acc | OOS N | OOS d | OOS acc | vs baseline |
|---|---|---:|---:|---:|---:|---:|---:|---|
| AT_LEVEL | ABOVE | 7,823 | -2.449 | 86.5% | 2,705 | -2.441 | 86.4% | -0.3pp IS, +0.4pp OOS |
| AT_LEVEL | BELOW | 7,324 | -2.509 | 87.5% | 2,531 | -2.387 | 86.1% | +0.6pp IS, +0.0pp OOS |
| NEAR_LEVEL | ABOVE | 1,849 | -2.411 | 86.4% | 416 | -2.099 | 82.5% | -0.5pp IS, -3.6pp OOS |
| NEAR_LEVEL | BELOW | 1,896 | -2.423 | 86.3% | 417 | -2.313 | 84.9% | -0.5pp IS, -1.1pp OOS |
| MEDIUM_DIST | ABOVE | 474 | -2.651 | 87.8% | 121 | -2.687 | 89.3% | +0.9pp IS, +3.2pp OOS |
| MEDIUM_DIST | BELOW | 626 | -2.385 | 84.2% | 125 | -3.247 | 92.8% | -2.7pp IS, +6.8pp OOS |
| FAR_FROM_SR | ABOVE | 560 | -2.329 | 87.1% | 235 | -2.441 | 87.2% | +0.3pp IS, +1.2pp OOS |
| FAR_FROM_SR | BELOW | 1,090 | -2.718 | 89.1% | 158 | -2.626 | 86.7% | +2.2pp IS, +0.7pp OOS |
| VERY_FAR | ABOVE | 2,862 | -2.262 | 84.7% | 548 | -2.211 | 84.3% | -2.2pp IS, -1.7pp OOS |
| VERY_FAR | BELOW | 2,256 | -2.653 | 88.9% | 337 | -2.610 | 86.6% | +2.0pp IS, +0.6pp OOS |

## Interpretation

- If S/R **amplifies** residual: AT_LEVEL accuracy > baseline accuracy.
- If S/R **weakens** residual: FAR_FROM_SR accuracy > AT_LEVEL.
- If S/R **irrelevant**: all strata ≈ baseline.

**Best stratum**: `MEDIUM_DIST ABOVE` — IS 87.8% / OOS 89.3%
**Worst stratum**: `NEAR_LEVEL ABOVE` — IS 86.4% / OOS 82.5%
