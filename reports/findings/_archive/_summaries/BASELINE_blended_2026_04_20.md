# BASELINE — blended engine, CNN off, live parity

Frozen 2026-04-20 19:36. Any future change to the blended engine must be
compared to these exact numbers on these exact pickles.

## Data fingerprint

| Path | Size | mtime |
|---|---:|---|
| `training_iso/output/trades/blended_is.pkl` | 1,607 MB | 2026-04-20 19:36 |
| `training_iso/output/trades/blended_oos.pkl` | 420 MB | 2026-04-20 19:36 |

Engine configuration: CNN off, live parity mode.

## Headline numbers

| Metric | IS | OOS | Combined |
|---|---:|---:|---:|
| Trades | 88,011 | 22,488 | 110,499 |
| Days | 277 | 68 | 345 |
| **Total $** | **-$14,447** | **+$11,712** | **-$2,735** |
| **$/day** | **-$52.16** | **+$172.24** | **-$7.93** |
| Winning days | 135 (49%) | 41 (60%) | 176 (51%) |
| Best day | +$4,288 | +$3,211 | — |
| Worst day | -$6,421 | -$2,005 | — |

## Per-tier contribution

### IS (277 days, -$14,447 total)

| Tier | Trades | $ | % trades | W/L |
|---|---:|---:|---:|---:|
| RIDE_AGAINST | 46,375 | -$16,468 | 53% | 21,934/23,764 |
| FADE_CALM | 29,663 | -$3,068 | 34% | 14,639/14,700 |
| KILL_SHOT | 3,895 | -$1,422 | 4% | 2,013/1,820 |
| CASCADE | 893 | -$272 | 1% | 470/414 |
| FREIGHT_TRAIN | 16 | -$232 | <1% | 8/8 |
| MTF_EXHAUSTION | 16 | +$142 | <1% | 8/8 |
| MTF_BREAKOUT | 1,794 | +$2,980 | 2% | 839/908 |
| FADE_AGAINST | 5,359 | +$3,893 | 6% | 2,685/2,613 |

### OOS (68 days, +$11,712 total)

| Tier | Trades | $ | % trades | W/L |
|---|---:|---:|---:|---:|
| MTF_BREAKOUT | 480 | -$822 | 2% | 207/267 |
| FREIGHT_TRAIN | 1 | -$52 | <1% | 0/1 |
| MTF_EXHAUSTION | 4 | +$292 | <1% | 2/2 |
| FADE_CALM | 7,663 | +$530 | 34% | 3,835/3,754 |
| FADE_AGAINST | 1,120 | +$1,250 | 5% | 584/522 |
| CASCADE | 203 | +$1,438 | 1% | 115/87 |
| KILL_SHOT | 984 | +$2,599 | 4% | 536/435 |
| RIDE_AGAINST | 12,033 | +$6,476 | 54% | 5,843/6,054 |

## Structural observations

1. **Net-negative across the full 345 days** (-$2,735 = -$7.93/day combined).
   IS drags, OOS lifts. Combined is close to zero.
2. **IS and OOS disagree sharply on tier ranking.** RIDE_AGAINST is the
   biggest bleeder on IS (-$16,468) and the biggest earner on OOS (+$6,476).
   MTF_BREAKOUT flips the other way. Same engine, same rules — opposite
   outcomes. Regime-sensitive.
3. **Winning-day rate**: IS 49%, OOS 60%. The OOS period is structurally
   more favorable; can't count on that holding.
4. **Daily distribution is bimodal.** IS mode is "<-$500" (78 days = 28%).
   Best day +$4,288 vs worst -$6,421. Tails dominate the mean.

## What an improvement must do

To be called a win, a change must **not break either side**:

- ✓ IS total ≥ -$14,447 (don't make IS worse)
- ✓ OOS total ≥ +$11,712 (don't break OOS)
- ✓ Combined ≥ -$2,735

**Bonus** (the real goal): move IS toward zero or positive without
reducing OOS below +$11,712.

## What a bust looks like

- Combined drops below -$2,735.
- OOS loses more than IS gains (pattern we already saw with the z_range
  filter: +$2,952 IS, -$6,772 OOS, net -$3,820).
- Either side drops by more than ~$3K with no compensating lift on the
  other side.

## Improvement candidates currently on the table

| Candidate | IS Δ est | OOS Δ est | Status |
|---|---:|---:|---|
| z_range entry filter | +$2,952 | **-$6,772** | REJECTED (net -$3,820) |
| Day-rule filter on RIDE_AGAINST (Rule A, 30% skip) | +$18,514 | +$3,045 | **Promising, not wired** |

---

_File generated 2026-04-20. Pipeline regeneration will require this file
to be updated. Historical prior baselines: `reports/findings/baseline_740.md`
(superseded by the 2026-04-17 lookahead fix)._
