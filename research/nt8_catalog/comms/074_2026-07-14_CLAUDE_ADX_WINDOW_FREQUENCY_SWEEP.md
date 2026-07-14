# ADX window FREQUENCY sweep — design map (Claude-executed for Moises)
**Doc:** 074 · **Date:** 2026-07-14 · **Author:** Claude (executor) · **Status:** RESULTS
Tool: `tools/adx_window_sweep.py`. FREQUENCY ONLY — never looks at returns, cannot overfit
an edge. Continuous window (no cold start). Sample: 2024_01..2024_03, 63 RTH days.

## Map (trig/day = raw fires per session; days% = fraction of sessions with >=1 fire)
```
smooth  N_adx  thr  trig/day  days%   | smooth  N_adx  thr  trig/day  days%
SMA        84   25     2.00    54%    | WILDER     84   15     2.75    71%
SMA        84   20     8.57    94%    | WILDER    168   15     2.60    38%
SMA       168   25     0.30    13%    | WILDER     84   20     0.06     3%
SMA       168   20     3.19    60%    | WILDER    168   20     0.00     0%
SMA       336   20     1.05    16%    | WILDER   >=25 anything     0.00  0%
SMA       720   15     0.95     8%    |
```
(N_sma cross fixed at 240 = 20 min, the legacy value.)

## Reading
1. **THRESHOLD dominates, not smoothing.** The `ADX>25` gate is what starves the signal.
2. **Legacy ADX-08 (SMA 168 / thr 25) = 0.30/day, 13% of days.** That is WHY it looked dead —
   it was correctly implemented but tuned to almost never fire on 5s bars.
3. **Wilder cannot reach 25 on 5s bars at any window** (confirms doc 073). Wilder ADX-08 is
   only "actionable" at thr<=15 — but ADX 15 is "no trend", which DEFEATS a trend gate.
4. **A trend GATE SHOULD be selective.** days% < 100% is a FEATURE (fire in trends, quiet in
   chop). Chasing "fires every day" by dropping thr to 15 destroys the concept.

## Recommendation (frequency only — edge NOT yet tested)
Keep the canonical `ADX>25` (a real trend), make it responsive by SHORTENING the window:
- **SMA, N_adx=84 (7 min), thr=25 → ~2 fires/day on 54% of sessions.** Actionable AND the
  threshold still means "trend". This is the candidate.
- (More breadth if wanted: SMA 168/thr20 = 60% of days, but thr 20 is a weaker trend claim.)

## GUARDRAIL (binding)
This chose FREQUENCY. It says NOTHING about edge. Next: FREEZE one setting, confirm the
frequency is stable on 2025 + full 2024, THEN measure edge OUT-OF-SAMPLE with the no-stops
horizon method. Never tune N against P&L. Frequency is the knob; edge is the read-out.

## Open
Sweep is 63 days of one quarter — a bursty setting here could be calm elsewhere. Confirm
frequency stability across 2024+2025 before freezing. N_sma (the cross MA) not yet swept.
FPS core untouched (this is a raw-stream design sweep, not the detector path).
