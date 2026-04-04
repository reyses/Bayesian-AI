# Daily Journal Index

| Date | Summary |
|------|---------|
| 2026-04-04 | nn_v2 pipeline built (ticker/aggregator/sfe_ticker/nightmare/tree). NMP baseline: $10/day IS, $65/day OOS. Decision tree 11x'd PnL by filtering 60% of bad trades. Top features: 5m/15m/1m velocity. |
| 2026-04-03 | Clean data = -$2,427 (phantom spikes were fake edge). Template autopsy: 0 template matches, 100% peak reversal coin flip. Designed 79D unified feature vector (10 features x 6 TFs). NN architecture: direction + hold duration (signal half-life). 5s = atomic unit, TFs = aggregation windows. Cord length ceiling: 5% capture at 5s = $2,350/day. |
| 2026-03-29 | Levels + EDA (z_se=levels, 1m leads direction, 1h leads speed) + shapes (4 types distinct) + archive cleanup |
| 2026-03-28/29 | Oscillation research, calibration, level drawing tool, hand-drawn levels validated across month |
| 2026-03-27 | 29D pipeline built, 3-layer CNN ($602/day IS), pivoted to oscillation-aware proposal |
| 2026-03-26 | 18 live bug fixes, TradeCNN StatePredictor $1,609/day OOS, trailing stop |
| 2026-03-25 | Base measurements framework, CNN $736/day OOS, DMI flipper $208-400/day |
