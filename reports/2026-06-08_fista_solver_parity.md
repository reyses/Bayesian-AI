# FISTA vs sklearn solver parity
_device: cuda | generated 2026-06-08_

## 1. Synthetic ElasticNet parity (FISTA-CV vs sklearn-CV)
| seed | sklearn alpha | FISTA alpha | |sk| | |fista| | Jaccard |
|---|---|---|---|---|---|
| 0 | 0.01584 | 0.01584 | 31 | 31 | 0.938 |
| 1 | 0.02780 | 0.02780 | 27 | 27 | 1.000 |
| 2 | 0.01588 | 0.01588 | 32 | 33 | 0.970 |
| 3 | 0.01268 | 0.01268 | 27 | 27 | 1.000 |
| 4 | 0.03491 | 0.03491 | 37 | 37 | 1.000 |

-> mean Jaccard 0.981, min 0.938 — **PASS (solver-equivalent)** (threshold 0.9)

## 2. Synthetic Group Lasso parity (FISTA-CV vs sklearn)
| seed | |sklearn| | |fista| | Jaccard |
|---|---|---|---|
| 0 | 6 | 6 | 1.000 |
| 1 | 6 | 6 | 1.000 |
| 2 | 6 | 6 | 1.000 |
| 3 | 6 | 6 | 1.000 |
| 4 | 6 | 6 | 1.000 |

-> mean Jaccard 1.000, min 1.000 — **PASS (solver-equivalent)** (threshold 0.9)

## 3. Real segment-data parity (day 2025_07_29, 8 blocks x 120 bars)
- Blocks screened with >=1 active cell: 8/8
- Mean term-selection Jaccard (FISTA vs sklearn): **0.762** (min 0.500)
- **Tier disagreement rate: 0/8 (0.0%)**  <- the decision number

| block start | tier(FISTA) | tier(sklearn) | match |
|---|---|---|---|
| 0 | 2 | 2 | Y |
| 1933 | 3 | 3 | Y |
| 3866 | 1 | 1 | Y |
| 5799 | 3 | 3 | Y |
| 7732 | 3 | 3 | Y |
| 9665 | 2 | 2 | Y |
| 11598 | 3 | 3 | Y |
| 13531 | 2 | 2 | Y |

**Interpretation:** if the tier-disagreement rate is near 0%, the stage1/stage2 solver split is cosmetic and unifying stage2 to CPU-FISTA is safe & low-value. A high rate means the current cross-stage tiers are NOT comparable and unification is worth doing.

_total run time: 74.5s_
