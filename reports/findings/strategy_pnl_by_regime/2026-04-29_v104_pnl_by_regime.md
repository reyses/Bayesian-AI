# v1.0.4 PnL by 1D regime - cross-tab

Generated: 2026-04-29 22:36

## Setup

- Atlas: `DATA/ATLAS`
- Strategy: v1.0.4 (R=45.0, counter-trend, no SL/MFE/trail)
- Segments: `reports/findings/macro_segments/1D_segments.csv`

## Headline (full 14-month sim)

| Metric | Value |
|---|---:|
| Total PnL | $-15,220.80 |
| Trades | 5817 |
| Win rate | 36.2% |
| $/trade | $-2.62 |
| Trades with matching 1D segment | 5760/5817 |

## By Direction

seg_direction  n_trades    wr_pct  pnl_total  pnl_mean  pnl_median regime_dimension
           UP      2974 36.886348    -1195.1 -0.401849       -38.4        Direction
         DOWN      2786 35.391242   -15452.4 -5.546447       -38.9        Direction

## By Sub-pattern

seg_sub_pattern  n_trades    wr_pct  pnl_total  pnl_mean  pnl_median regime_dimension
            L_H      2974 36.886348    -1195.1 -0.401849       -38.4      Sub-pattern
            H_L      2786 35.391242   -15452.4 -5.546447       -38.9      Sub-pattern

## By Slope intensity (33/66 quantiles)

seg_slope_bucket  n_trades    wr_pct  pnl_total  pnl_mean  pnl_median                  regime_dimension
          medium      1841 36.230310     -343.4 -0.186529       -35.9 Slope intensity (33/66 quantiles)
             low      2030 37.339901    -4982.5 -2.454433       -37.9 Slope intensity (33/66 quantiles)
            high      1889 34.833245   -11321.6 -5.993436       -41.4 Slope intensity (33/66 quantiles)

## By Zone behavior

  zone_behavior  n_trades    wr_pct  pnl_total  pnl_mean  pnl_median regime_dimension
     cross_well       407 37.346437    -2474.3 -6.079361       -37.9    Zone behavior
escape_velocity      1312 36.128049    -2759.8 -2.103506       -38.4    Zone behavior
       captured       805 36.149068    -4309.5 -5.353416       -36.4    Zone behavior
  between_zones      3236 36.032138    -7103.9 -2.195272       -39.9    Zone behavior

## Direction asymmetry (UP vs DOWN — the key hypothesis)

| Side | Trades | $/trade | WR | Total PnL |
|---|---:|---:|---:|---:|
| UP regimes | 2974 | $-0.40 | 36.9% | $-1,195.10 |
| DOWN regimes | 2786 | $-5.55 | 35.4% | $-15,452.40 |
| **Delta** | | **$+5.14/trade** | **+1.5pp** | **$+14,257.30** |

**v1.0.4 makes $+5.14 more per trade in UP regimes vs DOWN.** Confirms the asymmetric-regime hypothesis from cross-TF nesting.

