# v0.2-RC backtest trade export — honest metrics + anomaly flags
Source: research\nt8_port\reports\backtest_v03_trades_2026-06-23_07-18.csv
Window: 2026-06-22 -> 2026-07-17 (20 active days), N=38 trades. Config note: catastrophic stop ON at 50 pts (Moises).

## Headline (per canonical metric definitions)
- Net: $220.80
- **Trade WR (PF-based)**: +0.04  (PF 1.04; 14W/24L by count)
- **Day WR**: 45% (9/20)
- **$/trade**: mode $-51, mean $5.81 [95% CI $-106.57, $118.53]
- **$/day**: mode $-50, mean $11.04 [95% CI $-149.39, $169.86]
- Significance: $/day CI INCLUDES 0 - NOT significant. N=20 days is small; treat as directional.

## Exit-name breakdown
- X_RTriggerReversal: 19 trades, net $-2,936.10
- X_SessionFlatten: 18 trades, net $3,223.30
- Exit on session close: 1 trades, net $-66.40

## Day P&L (worst -> best)
- 2026-07-17: $-687.10
- 2026-06-26: $-555.80
- 2026-07-07: $-515.30
- 2026-06-22: $-346.80
- 2026-06-23: $-273.30
- 2026-06-29: $-214.80
- 2026-07-03: $-66.40
- 2026-07-13: $-47.70
- 2026-07-06: $-39.80
- 2026-07-14: $-15.40
- 2026-07-08: $-6.80
- 2026-07-01: $48.70
- 2026-07-09: $170.10
- 2026-07-02: $197.20
- 2026-06-25: $216.20
- 2026-06-24: $255.80
- 2026-07-16: $461.10
- 2026-07-10: $512.60
- 2026-06-30: $554.60
- 2026-07-15: $573.70

## Anomaly flags
1. **RTriggerReversal exits: 0 of 38** - the designed exit NEVER fired. Winners exited on session close instead. The strategy tested is effectively 'ensemble entry + ride to close + disaster stop', NOT the designed R-trigger ride.
2. **Stop slip**: 0 cat-stop exits lost >10% beyond the 50-pt setting: 
3. **Entry clustering**: 27/38 entries before 09:00 local - first-qualifying-minute-of-day pattern; verify against harness selectivity in the P3 diff.
4. **Session semantics**: 'Exit on session close' fired at 14:00 local on most days while X_SessionFlatten fired once at 15:56 - the data-series session template and the strategy's 15:55-CT flatten disagree (TODO P2-5/P2-11).
