# level_hold — do frozen band levels act as forward support/resistance?

**Origin (2026-07-07)**: user observation from live NT8 trading (see
`examples/Figure_1.png`): freezing the current regression-band values as
horizontal lines, price appears to respect them repeatedly later in the
session. Known objection: drawn-in-hindsight + eye counts hits and forgets
misses. This study makes it causal and counted.

## Design

- **Freeze**: every 1 min, record the causal band values (computed from data
  ≤ t only): FAST bands = OLS endpoint ± 2σ on 5s closes, 120-bar (10 min)
  window (mirror of NT8 1a-StatCloseRegressionBands); SLOW bands = same on
  1m closes, 60-bar (60 min) window (mirror of 1b-StatHIRegressionBands);
  plus rolling 60-min EXTREMES of those band lines ("levels are built from
  the extremes").
- **Touch**: within 30 min of the freeze, price range comes within 2 ticks
  of the frozen level (approach side = where price was at freeze).
- **Outcome** (symmetric barriers, R = C = 8 ticks = 2.0 pts): HOLD if price
  reverses 8 ticks off the level before penetrating 8 ticks through; BREAK
  otherwise; unresolved after 15 min or same-bar-both → dropped (counted).
  **Symmetric barriers ⇒ a random walk scores exactly 50%** — the primary
  null is built in.
- **Second null**: phantom levels — same freeze time/side/distance, value
  jittered ±(4–16) ticks. Separates "this level is special" from "everything
  near price bounces" (mean-reversion base rate).
- Dedup by (level value, touch bar) so the same level frozen repeatedly
  isn't double-counted.

## Read the results

`reports/level_hold_<date>.md` — P(hold|touch) with Wilson 95% CI per level
family, real vs phantom, per-day spread, N, and time-to-resolution
distribution ("explains within X minutes").

## Run

```
.venv_wsl/bin/python research/level_hold/tools/level_hold_study.py \
    --days 2024_02_20,...  [--r-ticks 8 --tol-ticks 2]
```
Data: DATA/ATLAS/{5s,1m} parquet (repo root, gitignored).
