---
name: Architectural distinction — context filters vs tiers (locked 2026-05-09)
description: Robustness filters and macro-pivot detectors are context layers conditioning when tiers fire, not tiers themselves
type: project
---

**Locked 2026-05-09 evening** after the FadeAtBand rejection.

## The distinction

```
TIER             = a strategy that fires ENTRIES with a direction
                   (FADE_CALM, KILL_SHOT, RIDE_AGAINST, FreightTrain etc.)

CONTEXT FILTER   = a gate/conditioner that decides WHEN tiers should fire
                   or WHEN their entries are valid
                   (4 robustness filters, CRM macro detector, regime label,
                    hurst, swing_noise, time-of-day, calendar event)
```

Tiers PRODUCE trade signals. Context filters CONDITION which signals are
allowed through to execution.

## Why the distinction matters

When I tested FadeAtBand "with its filters" as a single tier, I conflated
two things:
1. The entry rule (5s touches 15m ±2σ → fade to 5m mean) — this had no edge
2. The 4 robustness filters — these DID gate macro-event blowups

Treating them as one bundle made the verdict "FadeAtBand is rejected"
which is misleading. The correct verdict is:
- ENTRY RULE rejected
- FILTERS survive as reusable context components

The CRM macro-pivot detector v2 is the same: it's not a strategy, it's a
context filter that wraps any reversion strategy to suppress macro-impulse
days.

## Implications for system design

**Next-iteration entries** should be built as: entry rule + library of
context filters that can be turned on/off independently.

```
Strategy = Entry rule  ⊗  Context filter stack
         = (5s touches band)  ⊗  (hurst<0.60)  ⊗  (require_divergence)
                              ⊗  (CRM not in macro-impulse)
                              ⊗  (regime ≠ DOWN_SMOOTH)  ...
```

This lets us:
1. Test entry rules in isolation (no filter contamination)
2. Test filters in isolation (apply same filter to multiple entry rules,
   measure incremental contribution)
3. Build a context-filter LIBRARY — each filter is a reusable bool function

## What this means for the 9-layer probability stack

The 9 ExNMP tiers were tier-level strategies, but the empirical first-passage
probability table that's pending should be ORGANIZED AS A FILTER GRID:

```
P(reversion succeeds | state) = base table
                              + state filter   (current 3-body state)
                              + regime filter  (UP/DOWN/FLAT × SMOOTH/CHOPPY)
                              + macro filter   (CRM not in impulse)
                              + tod filter     (US morning vs lunch vs close)
                              + cal filter     (FOMC/NFP/CPI off)
                              + hurst filter
                              + sn filter
```

Each filter is a conditional on the table. The fully-conditioned cell tells
us "P(this entry succeeds given THIS exact context)". A tier becomes a
named selection of filter values.

## What we keep from FadeAtBand experiment

As context filters (proven on 2026-05-09):
- `hurst_5m < 0.60` — gates trend regimes
- `max_counter_trend_vel = 25.0` — gates against-strong-momentum
- `require_divergence` between 1m and 5m means — confirms reversion structure
- `confirm_bars = 6` — confirms persistent band-touch

The macro-gate IS the 4-filter stack here. CRM detector is the next-level
filter to add (impulse-day suppression). These compose multiplicatively in
the probability stack.

## Naming convention

When adding to `training_iso_v2/` or any future framework:
- Tiers go under `strategies/`
- Context filters should live under a new `filters/` directory
- Filters expose a single signature: `def __call__(state) -> bool`
- Tests apply filters as `entry_signal AND f1(state) AND f2(state) AND ...`

Nothing in the codebase enforces this yet — but starting today, every new
"check" we add must be classified as either an entry-rule component or a
context filter, and lived in the right place.
