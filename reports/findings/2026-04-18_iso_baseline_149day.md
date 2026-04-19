# ISO Baseline — 2026-04-18 (Post tier-fixes)

**Headline: $149.86/day, 277 days, 14,983 trades, 57% win days.**
Defaults: Phase 1 only, no regret, chains=1, 9 isolated per-tier engines.

Run command: `python training_iso/run_iso.py`

## Baseline snapshot

```
RESULTS: 277 days | 14,983 trades | $41,510
  $/day: $149.86
  Winning days: 157/277 (57%)
```

## Per-tier breakdown

| Tier | N | WR | Total | $/trade | Status |
|---|---:|---:|---:|---:|---|
| TREND_FOLLOWER | 780 | 68% | +$638 | +$0.82 | fixed (flip+peak rule+60m timeout) |
| CASCADE | 113 | 55% | +$4,841 | +$42.84 | **untouched — don't break** |
| KILL_SHOT | 301 | 61% | **−$345** | **−$1.15** | **anomaly — winners capped, losers run** |
| RIDE_AGAINST | 4,204 | 65% | +$2,944 | +$0.70 | fixed (flip+phantom+relaxation), exit too conservative |
| FADE_AGAINST | 326 | 47% | +$7,561 | +$23.19 | **untouched — asymmetric winners, don't break** |
| MTF_EXHAUSTION | 125 | 47% | +$2,890 | +$23.12 | **untouched — asymmetric winners, don't break** |
| MTF_BREAKOUT | 661 | 60% | −$96 | −$0.15 | flipped+phantom — same asymmetry as KILL_SHOT |
| NMP_FADE | 8,174 | 55% | +$11,116 | +$1.36 | workhorse volume, untouched |
| NMP_RIDE | 299 | 51% | +$11,962 | **+$40.01** | **hidden gem — relaxation candidate** |

## Progression (same-session)

| Baseline | After RIDE_AGAINST | After KILL_SHOT phantom | **Current** |
|---:|---:|---:|---:|
| +$76/day | +$86/day | +$109/day | **+$149.86/day** |

Session lift: **+$74/day** on the same 277-day IS window, same defaults.

## Anomaly flagged: KILL_SHOT

**61% WR + −$1.15/trade is mathematically "broken exits."**

Math: 301 trades × 61% = 184 winners, 117 losers, total −$345
→ avg winner ≈ $8, avg loser ≈ $16 → loser size is ~2× winner size.

Our three-question method so far has focused on **winner peak-arrival** rules
(exit at feature signature of a peak). That caps winners at ~$8 but does
nothing to cut losers early. Losers keep running and overwhelm the 61% WR.

The hole: **no loser-cut mechanism.** Q2 hold-cliff was applied to tail-loser
population on earlier tiers, but for KILL_SHOT the timeout is 30m — plenty
of time for a loser to bleed to full MAE.

## Next-round target

**KILL_SHOT loser asymmetry (Q2 applied to loser subset).**

EDA plan:
1. Split trades by `pnl` sign → 184 winners / 117 losers
2. For losers: `peak_pnl` distribution (how many NEVER had $5 of green?)
3. For losers: bar-N where MAE median lands (natural "dead" timescale)
4. Candidate rule: `bars_held > N AND peak_pnl < $3 → exit (no-progress cut)`

Same pattern likely applies to MTF_BREAKOUT (60% WR, −$0.15/trade — same
asymmetry, just smaller magnitude).

## Path to $300/day target

Need +$150/day more. Ranked by expected leverage:

1. **KILL_SHOT loser cut** → expect to move from −$345 to +$500–$1,500
   (+$2–5/day). Same rule on MTF_BREAKOUT adds similar.
2. **RIDE_AGAINST exit tuning** — 65% WR at $0.70/trade means exits are
   too early. Relax or refine → could double total (+$10/day).
3. **NMP_RIDE relaxation** — 299 trades × $40/trade. Doubling trigger rate
   (safely, via phantom-confirmed relaxation) → +$40/day.
4. **NMP_FADE micro-lift** — 8,174 trades. $0.50/trade improvement = +$15/day.

Total est upside: **+$75/day from current lineup.** Doesn't close the
$300/day gap alone — chain multiplier (chains=4) historically adds +$100/day
but that was on a smaller baseline. Worth measuring post-tier-stabilization.

## Isolation proof

User verified by running `--tier RIDE_AGAINST` and then full engine —
RIDE_AGAINST row identical in both, confirming no cross-tier interference.

## Artifacts
- Trade dump: `training_iso/output/trades/iso_is.pkl` (stripped features due to size)
- Flat CSV: `training_iso/output/trades/iso_is.csv`
- Engine: `training_iso/nightmare_iso.py`
- Pipeline: `training_iso/run_iso.py`
