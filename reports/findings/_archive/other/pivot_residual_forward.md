# Pivot-residual FORWARD PASS

Forward pass with **NO LOOKAHEAD**. Pivot confirmed when price retraces $r_confirm from running extreme; entry at the confirmation bar (not the true pivot).

**Config**: r_confirm=$10.0, TP=$50.0, SL=$3.0, min\_res=0.5, inverse\_thr=2.0

## Results

| Dataset | Days | Pivots | Trades | $/day | $/trade | WR | $WR | Total $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 277 | 61,673 | 28,949 | $+657 | $+6.28 | 23.7% | +276% | $+181,868 |
| OOS | 68 | 18,155 | 8,288 | $+798 | $+6.55 | 22.6% | +283% | $+54,287 |

## Exit breakdown

| Reason | IS N | IS % | OOS N | OOS % |
|---|---:|---:|---:|---:|
| SL | 21,832 | 75.4% | 6,364 | 76.8% |
| TP | 3,717 | 12.8% | 1,164 | 14.0% |
| eod | 98 | 0.3% | 35 | 0.4% |
| inverse | 3,302 | 11.4% | 725 | 8.7% |

## Vs oracle

| | Oracle (pivot lookahead) | Forward pass (realistic) |
|---|---:|---:|
| IS $/day  | +$1,358 | $+657 |
| OOS $/day | +$1,664 | $+798 |
