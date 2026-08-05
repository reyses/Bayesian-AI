# After a vertical spike that immediately fades — next 20 min

Analog: ≥20pt inside 15s, directional (net ≥ 70% of range), then ≥30% giveback within 60s. Direction **up**, ET **09:45–10:15**. Anchor = the giveback point.
Excluded: 2024_09_16. One sample per 30min per day.

**N = 291 analogs across 291 sessions.**

## Forward return from the giveback point (spike direction = +)

- mean **+4.96pt** ($+9.91), 95% CI **[-3.94, +14.14]** → NOT significant (CI includes 0)
- median **+7.25pt** · share continuing: **52.6%**
- quantiles p10 `-80.8` p25 `-40.5` p50 `+7.2` p75 `+51.4` p90 `+89.5`

## Excursions (what the path does, not just where it ends)

- MFE median **50.8pt** (p75 `84.0`)
- MAE median **45.5pt** (p75 `80.0`)
- both-touch: 73.9% of paths reach ±10pt in BOTH directions inside the window

## Race: which is hit first from the anchor

| ±N pt | reached favorable first | adverse first | neither |
|---|---|---|---|
| ±5 | 7.2% | 4.1% | 0.0% (both: 88.7%) |
| ±10 | 16.2% | 10.0% | 0.0% (both: 73.9%) |
| ±15 | 20.6% | 13.7% | 0.0% (both: 65.6%) |
| ±20 | 27.5% | 19.2% | 0.3% (both: 52.9%) |

Note: "both" cannot be resolved into a true race without bar-order replay — treat those rows as ambiguous, not as wins.

