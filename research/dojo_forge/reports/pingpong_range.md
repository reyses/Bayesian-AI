# Ping-pong between FIXED churn extremes (owner design)

Levels are taken from the observed churn window and do NOT move. This is the correction to the moving-σ-band version, whose "traverse completes" rate was inflated by the cubic chasing price.

Range from the prior 30min (bars strictly BEFORE the touch), width 8–60pt, K≥5 churn regime, entry at one extreme, target the OPPOSITE extreme, stop 8pt outside the range, max hold 60min, friction 0.89pt.
Excluded: 2024_09_16.

**N = 2093 trades across 169 sessions.** Median range width 43.0pt.

## Outcome

- target hit **19.5%** · stopped 72.1% · timeout 8.3%
- mean net **-0.50pt** ($-1.00) 95% CI `[-1.30, +0.35]` → NOT significant
- median net -10.14pt

## Head-to-head vs the moving band, same entries (PAIRED)

- fixed-extreme exit: **-0.50pt** `[-1.30, +0.35]`
- moving-band exit:   **-0.54pt** `[-0.92, -0.15]`
- paired Δ (fixed − band): **+0.03pt** 95% CI `[-0.71, +0.83]` → NOT significant

The paired comparison is the point: identical entries, identical regime filter, only the exit definition differs.

