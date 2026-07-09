# Catalog concepts vs Bayesian-AI evidence (honest mapping, 2026-07-08)

Correction from Moises: "similar but not the same" — several catalog concepts
were wrongly lumped in with graveyard results. This is the audited mapping.

## Tested here and DEAD (do not re-run as-is)
| concept | our evidence |
|---|---|
| Volume-RATE buildup / expansion | pretrend_footprint: +2pp, non-monotonic (2026-07-08) |
| Candle wick/body shapes | two independent nulls (pretrend + AG replication) |
| Band-level bounce (first touch) | phantom==real, 63d (level_hold) |
| Order-flow delta (directional absorption) | DEAD 2026-06-27/28, tick purchase rejected |
| Touch-count of a zone | dead once sigma-relative (2026-07-07) |
| Cut-loser overlays / fixed stops | graveyard + fail-fast neutral (2026-07-08) |
| APZ re-entry confirmation | PF 0.97/0.83 vs 1.26/1.25 immediate — REJECTED both years (2026-07-08); the fade's edge IS entering into the violence |

## NOT tested here (wrongly dismissed — open)
| concept | why it's genuinely different |
|---|---|
| **Volume Profile: POC / Value Area / shapes** | volume AT PRICE. Everything killed was volume over TIME. Directly implements the wall/absorption thesis. |
| **Prior-day OHLC / floor pivots** | the classic level family; our level tests used band-derived levels only |
| RSI/MACD divergence at extremes | never measured here |
| Volatility squeeze → break (bandwidth as temporal pattern) | z_se tested as snapshot, never as squeeze-then-release sequence |
| 30-min opening range break | never measured |
| VWAP z-score as explicit gate/target | vwap_30 is a classifier input, never a rule |
| Seasonality / day-of-week | never measured |

## Partially blocked
| concept | blocker |
|---|---|
| Footprint imbalances, trapped buyers, cumulative delta divergence | needs tick/bid-ask data; purchase rejected 2026-06-28; DOM exists live (BayesianBridge) but not in history |

## Context these plug into
Active niche: fade + 9-13 CT (PF 1.25/1.26 both years, not yet significant).
Untested concepts should be evaluated (a) as gates on this niche, and (b)
independently against the labels — via research/leg_clock/tools/dev_loop_2025.py.
