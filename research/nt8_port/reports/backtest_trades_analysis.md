# v0.2-RC backtest trade export — honest metrics + anomaly flags
Source: research\nt8_port\reports\backtest_v04_trades_5s.csv
Window: 2026-06-22 -> 2026-07-17 (20 active days), N=182 trades. Config note: catastrophic stop ON at 50 pts (Moises).

## Headline (per canonical metric definitions)
- Net: $5,923.70
- **Trade WR (PF-based)**: +0.54  (PF 1.54; 87W/95L by count)
- **Day WR**: 70% (14/20)
- **$/trade**: mode $-201, mean $32.55 [95% CI $1.31, $68.60]
- **$/day**: mode $24, mean $296.19 [95% CI $23.54, $592.38]
- Significance: $/day CI EXCLUDES 0 - significant at 95%. N=20 days is small; treat as directional.

## Exit-name breakdown
- X_RTriggerReversal: 137 trades, net $8,852.20
- Stop loss: 26 trades, net $-5,249.40
- X_SessionFlatten: 17 trades, net $2,367.70
- Exit on session close: 2 trades, net $-46.80

## Day P&L (worst -> best)
- 2026-06-26: $-863.30
- 2026-07-01: $-502.70
- 2026-07-16: $-461.00
- 2026-07-07: $-130.70
- 2026-07-09: $-68.40
- 2026-07-03: $-17.30
- 2026-06-25: $27.80
- 2026-07-08: $29.70
- 2026-07-06: $106.20
- 2026-07-13: $173.80
- 2026-07-10: $190.30
- 2026-07-14: $358.80
- 2026-06-22: $370.60
- 2026-06-29: $436.00
- 2026-06-30: $485.60
- 2026-06-23: $634.30
- 2026-07-17: $675.80
- 2026-07-02: $1,159.40
- 2026-07-15: $1,186.40
- 2026-06-24: $2,132.40

## Anomaly flags
1. **RTriggerReversal exits: 0 of 182** - the designed exit NEVER fired. Winners exited on session close instead. The strategy tested is effectively 'ensemble entry + ride to close + disaster stop', NOT the designed R-trigger ride.
2. **Stop slip**: 0 cat-stop exits lost >10% beyond the 50-pt setting: 
3. **Entry clustering**: 118/182 entries before 09:00 local - first-qualifying-minute-of-day pattern; verify against harness selectivity in the P3 diff.
4. **Session semantics**: 'Exit on session close' fired at 14:00 local on most days while X_SessionFlatten fired once at 15:56 - the data-series session template and the strategy's 15:55-CT flatten disagree (TODO P2-5/P2-11).
