# Zero-day diagnosis — pivot_physics_chains chains=1

Total days: 345 (IS 277 / OOS 68)

## Classification

| Class | Description | IS | OOS | All |
|---|---|---:|---:|---:|
| A_no_trades | No trades fired | 45 | 12 | 57 |
| B_wash_small_positive | Trades fired, net ∈ [0,25), gross_loss/gross_win > 50% | 3 | 0 | 3 |
| B_wash_exact | Trades fired, |net| < $5 | 0 | 0 | 0 |
| C_small_positive_few_trades | Trades fired, net ∈ [0,25), minimal offsets | 3 | 0 | 3 |
| Z_non_mode | Days outside mode bucket | 226 | 56 | 282 |

## A_no_trades: 57 days

Why no trades? Check data availability + signal strength.

- Mean day range: $188
- Median day range: $158
- Mean n_strong_res_bars: 65
- Mean n_1m_bars: 92

Compare to days with trades:
- Mean day range (traded): $831
- Mean n_strong_res_bars: 964
- Mean n_1m_bars: 1341

### Top-10 no-trade days by range (puzzling — why no entry?)

| Day | DoW | Range | n_1m | n_strong_res | pct_strong |
|---|---:|---:|---:|---:|---:|
| 2025_04_06 | Sun | $646 | 120 | 81 | 68% |
| 2025_05_25 | Sun | $481 | 120 | 93 | 78% |
| 2026_03_15 | Sun | $394 | 120 | 85 | 71% |
| 2026_03_08 | Sun | $338 | 120 | 74 | 62% |
| 2026_02_01 | Sun | $334 | 60 | 39 | 65% |
| 2025_03_09 | Sun | $317 | 120 | 82 | 68% |
| 2025_11_16 | Sun | $311 | 60 | 50 | 83% |
| 2025_04_13 | Sun | $309 | 120 | 84 | 70% |
| 2025_02_02 | Sun | $288 | 60 | 45 | 75% |
| 2025_02_09 | Sun | $287 | 60 | 36 | 60% |

### No-trade day distribution by day-of-week

| DoW | Count | % of no-trade |
|---|---:|---:|
| Mon | 0 | 0% |
| Tue | 0 | 0% |
| Wed | 1 | 2% |
| Thu | 1 | 2% |
| Fri | 0 | 0% |
| Sat | 0 | 0% |
| Sun | 55 | 96% |

## B_wash: 3 days (trades fired, net near zero)

- Mean n_trades: 9.7
- Mean gross_win: $+196
- Mean gross_loss: $-191
- Mean net: $+6

## C_small_positive: 3 days

- Mean n_trades: 1.0
- Mean gross_win: $+9
- Mean gross_loss: $+0

## Day range by class

| Class | N | Mean range | Median range |
|---|---:|---:|---:|
| A_no_trades | 57 | $188 | $158 |
| Z_non_mode | 282 | $842 | $734 |
| C_small_positive_few_trades | 3 | $289 | $286 |
| B_wash_small_positive | 3 | $336 | $361 |
