# Pivot NN filter — post-hoc uplift on TEST set

Generated: 2026-04-22T17:20:28

Test trades: 3457 over 35 days
Thresholds: TAKE ≥ 0.55, FLIP ≤ 0.45

## Regime counts

- TAKE: 1475 trades (43%)
- FLIP: 899 trades (26%)
- SKIP: 1083 trades (31%)

## Aggregate comparison

| Variant | Trades used | Net $ | $/day | Trade WR | Day WR |
|---|---:|---:|---:|---:|---:|
| original | 3457 | $-23,228 | $-664 | -0.49 | 6% |
| nn-filtered | 2374 | $+17,575 | $+502 | +1.08 | 94% |
