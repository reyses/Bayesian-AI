# v0.2-RC backtest trade export — honest metrics + anomaly flags
Source: examples\NinjaTrader Grid 2026-07-18 06-27 PM.csv
Window: 2026-06-22 -> 2026-07-17 (20 active days), N=44 trades. Config note: catastrophic stop ON at 50 pts (Moises).

## Headline (per canonical metric definitions)
- Net: $816.50
- **Trade WR (PF-based)**: +0.29  (PF 1.29; 17W/27L by count)
- **Day WR**: 65% (13/20)
- **$/trade**: mode $-96, mean $18.56 [95% CI $-35.44, $76.07]
- **$/day**: mode $32, mean $40.83 [95% CI $-95.58, $167.38]
- Significance: $/day CI INCLUDES 0 - NOT significant. N=20 days is small; treat as directional.

## Exit-name breakdown
- X_CatastrophicStop: 25 trades, net $-2,696.50
- Exit on session close: 18 trades, net $3,467.00
- X_SessionFlatten: 1 trades, net $46.00

## Day P&L (worst -> best)
- 2026-06-26: $-831.00
- 2026-07-07: $-325.50
- 2026-06-23: $-312.00
- 2026-06-22: $-184.00
- 2026-06-25: $-162.50
- 2026-07-15: $-85.50
- 2026-07-10: $-13.50
- 2026-07-03: $19.50
- 2026-07-16: $21.00
- 2026-07-17: $35.50
- 2026-07-09: $95.50
- 2026-07-14: $129.00
- 2026-06-29: $157.50
- 2026-06-30: $192.50
- 2026-07-02: $206.50
- 2026-07-06: $233.50
- 2026-06-24: $253.50
- 2026-07-08: $402.50
- 2026-07-01: $479.00
- 2026-07-13: $505.00

## Anomaly flags
1. **RTriggerReversal exits: 0 of 44** - the designed exit NEVER fired. Winners exited on session close instead. The strategy tested is effectively 'ensemble entry + ride to close + disaster stop', NOT the designed R-trigger ride.
2. **Stop slip**: 4 cat-stop exits lost >10% beyond the 50-pt setting: #4 ($-143, MAE $145), #5 ($-128, MAE $131), #12 ($-150, MAE $188), #16 ($-328, MAE $365)
3. **Entry clustering**: 21/44 entries before 09:00 local - first-qualifying-minute-of-day pattern; verify against harness selectivity in the P3 diff.
4. **Session semantics**: 'Exit on session close' fired at 14:00 local on most days while X_SessionFlatten fired once at 15:56 - the data-series session template and the strategy's 15:55-CT flatten disagree (TODO P2-5/P2-11).
