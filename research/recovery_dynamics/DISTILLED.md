---
name: distilled-recovery_dynamics
description: Recovery/oscillation is a stable structural clock (period) with amplitude as the real regime; "hold for the kicker" edge (~52%) is gated on an unbuilt oscillator-vs-runaway read.
metadata: {type: distilled, topic: recovery_dynamics, status: live}
---
## Verdict
Moises' question: what does holding a wrong trade to breakeven cost (time/drawdown/opportunity),
not just "does it recover." Found: oscillation period is a stable structural constant (fixed clock,
~7% no-return/trend rate, invariant to year and drift); amplitude (not period) is the true regime
variable (~4x swings 2024->2025). A held-to-breakeven trade's 2nd leg is favorable 52% of the time
(symmetric, A2/A1~1.13), but 9% never return (death). Death prediction at entry is only CONDITIONAL
(OOS AUC 0.572). Reduces to one unsolved discrimination: oscillator (hold) vs runaway (cut).

## Key numbers (with CIs where they exist)
- 536d 2024+2025, ~21k wrong trades: period MODE ~5min, ~2 trades foregone/trade, depth ~12-14pt
  ($25-28). `recovery_2024_2025.md`
- Anchor-every-bar period (644k anchors): mode ~2m, median 5m, mean 20m; no-return 7.2%(2024) vs
  7.3%(2025), stable <0.3%; amplitude 30-50% wider in 2025 at every bucket. `anchor_period.md`
- Amplitude regime (21d window, 518d): 10-min swing ranges ~4x, 5.5pt(calm Jun-2024)->~20pt(Apr-2025),
  trend share stable ~6-8%. `amplitude_evolution.md`. Drift-conditional return rate 93.7%->91.1%
  across drift spectrum; period=5min at every drift level. `anchor_drift_conditional.md`
- Death OOS (train2024 n=10268 death30% -> test2025 n=9584 death25%): AUC 0.572, gap +0.072 ->
  CONDITIONAL; VOL AUC-drop 0.0481 (higher vol=LESS death); TREND ~0.0008. `event_features.md`
- Cut-in-time (21383 trades, ~6.1 swings/hr baseline): cut@60min->26% stuck, unlocks med11/mean29.2/
  90th84 (15min: 51% stuck med4; 30min: 37% stuck med6). `cut_in_time.md`
- Oscillator vs runaway (n=42855, |PnL|>=5pt): OSCILLATORS 91.4%, RUNAWAYS 8.6% (49/51 win/loss),
  runaway terminal |PnL| median 94pt($188), 90th 335pt. `oscillator_vs_trend.md`
- Kicker (n=21383): 91% return to zero/9% death; of returns KICKER47%+JACKPOT5%=52% favorable, A2
  med18pt($35), A2/A1 med1.13. `oscillation_kicker.md`

## Graveyard / never-retry
- Fixed "cut at N min" rule — unsafe blanket; clock must be regime-adaptive.
- time_of_day death predictor — faked AUC 0.642 via EOD-censoring artifact; fixed by 60min horizon.
- Cubic/inflection-at-trend-start (Moises hypothesis) — tested (31k anchors): no-return |d3| 0.61 vs
  return 0.69 (weaker, not stronger) — NOT supported. `cubic_inflection.md`
- "27min vs 15min" period non-stationarity — CORRECTED: artifact of a >=5pt drawdown gate leaking
  amplitude into period; unconditional period is stable across years.

## Reusable assets
- `tools/opportunity_cost.py` — THE exercise; defines shared `ROOT`/`ONE_M` data paths.
- `tools/anchor_period.py` — anchor-every-bar first-return measurement (cached npz, `--fresh`).
- `tools/{cut_in_time,oscillation_kicker,oscillator_vs_trend,amplitude_evolution,anchor_drift_
  conditional,cubic_inflection,event_features}.py` — one script per same-named report.

## Data locations
1-min OHLC parquets under `ONE_M` (ATLAS-derived, per-year glob); cache `artifacts/anchor_period_cache_v2.npz`.

## Open threads
- Oscillator-vs-runaway discriminator — unbuilt ("next build"); causal trailing estimator of the
  amplitude "volume knob" + envelope-breach->runaway validation is its unbuilt prerequisite.
- Refined cubic test isolating the FIRST abandoned level of a run; condition recovery on causal env
  read (regime/lambda/z/vol); threshold sensitivity + multi-day generalization.

## Sources
`README.md`, `reports/{recovery_2024_2025,anchor_period,amplitude_evolution,anchor_drift_conditional,
event_features,cut_in_time,oscillation_kicker,oscillator_vs_trend,cubic_inflection}.md` (all under
`research/recovery_dynamics/`)

## Archive recommendation
KEEP-LIVE — explicit unbuilt next step (oscillator-vs-runaway discriminator); not concluded.
