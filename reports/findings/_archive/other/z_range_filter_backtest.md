**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# z_range filter backtest — per-tier decile view

Thresholds: reject >= 2.5, size1 >= 2.0, size2 >= 1.5

**Bucket scheme**: per-tier percentile deciles. For each tier, baseline PnL is split into 10 equal-count buckets:
- **D1** = worst 10% of trades for THAT tier
- **D10** = best 10% of trades for THAT tier
- Baseline uniformly 10% per decile by construction
- Filter runs reuse the baseline thresholds so shifts are comparable

Tiers with fewer than 40 baseline trades are skipped.

## IS

| Strategy | Kept | Rejected | PnL | d vs Baseline | WR |
|---|---:|---:|---:|---:|---:|
| Baseline | 88,011 | 0 | $-14,447 | $+0 | 48% |
| Filter >= 2.5 | 76,201 | 11,810 | $-11,496 | $+2,952 | 48% |
| Sizing only | 88,011 | 0 | $-14,447 | $+0 | 48% |
| Filter + Sizing | 76,201 | 11,810 | $-11,496 | $+2,952 | 48% |

Skipped (N < 40): MTF_EXHAUSTION, FREIGHT_TRAIN

### IS — per-tier decile $ boundaries

Per-tier $ boundaries for deciles D1 (worst 10%) .. D10 (best 10%), computed from that tier's BASELINE PnL distribution.

| Tier | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CASCADE | <=$-31.3 | $-31.3..$-14.5 | $-14.5..$-6.5 | $-6.5..$-2.0 | $-2.0..$+1.0 | $+1.0..$+4.5 | $+4.5..$+9.5 | $+9.5..$+17.0 | $+17.0..$+30.4 | >$+30.4 |
| FADE_AGAINST | <=$-39.5 | $-39.5..$-20.5 | $-20.5..$-10.5 | $-10.5..$-4.5 | $-4.5..$+0.5 | $+0.5..$+5.5 | $+5.5..$+12.0 | $+12.0..$+21.5 | $+21.5..$+41.0 | >$+41.0 |
| FADE_CALM | <=$-43.5 | $-43.5..$-21.5 | $-21.5..$-11.5 | $-11.5..$-5.0 | $-5.0..$+0.0 | $+0.0..$+5.0 | $+5.0..$+11.0 | $+11.0..$+21.0 | $+21.0..$+43.0 | >$+43.0 |
| KILL_SHOT | <=$-27.5 | $-27.5..$-13.1 | $-13.1..$-7.0 | $-7.0..$-2.5 | $-2.5..$+1.0 | $+1.0..$+4.0 | $+4.0..$+8.0 | $+8.0..$+14.5 | $+14.5..$+26.5 | >$+26.5 |
| MTF_BREAKOUT | <=$-24.5 | $-24.5..$-12.5 | $-12.5..$-7.5 | $-7.5..$-3.5 | $-3.5..$-0.5 | $-0.5..$+2.5 | $+2.5..$+6.5 | $+6.5..$+13.0 | $+13.0..$+27.0 | >$+27.0 |
| RIDE_AGAINST | <=$-31.0 | $-31.0..$-16.5 | $-16.5..$-9.0 | $-9.0..$-4.5 | $-4.5..$-0.5 | $-0.5..$+3.0 | $+3.0..$+7.5 | $+7.5..$+15.0 | $+15.0..$+30.5 | >$+30.5 |

### IS — baseline decile share (sanity check: ~10% each)

Percent of kept trades in each decile (using the BASELINE thresholds). Baseline is 10% per decile by construction.

| Tier | N | $ | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CASCADE | 893 | $-272 | 10% | 10% | 11% | 10% | 10% | 10% | 10% | 10% | 10% | 10% |
| FADE_AGAINST | 5,359 | $+3,893 | 10% | 10% | 10% | 10% | 10% | 10% | 10% | 10% | 10% | 10% |
| FADE_CALM | 29,663 | $-3,068 | 10% | 10% | 10% | 10% | 10% | 10% | 9% | 10% | 10% | 10% |
| KILL_SHOT | 3,895 | $-1,422 | 10% | 10% | 11% | 10% | 10% | 10% | 9% | 10% | 9% | 10% |
| MTF_BREAKOUT | 1,794 | $+2,980 | 10% | 11% | 10% | 11% | 9% | 10% | 10% | 9% | 10% | 10% |
| RIDE_AGAINST | 46,375 | $-16,468 | 10% | 10% | 11% | 10% | 11% | 9% | 10% | 10% | 10% | 10% |

### IS — filter+sizing decile share

Percent of kept trades in each decile (using the BASELINE thresholds). Baseline is 10% per decile by construction.

| Tier | N | $ | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CASCADE | 709 | $-214 | 9% | 11% | 9% | 10% | 9% | 12% | 10% | 10% | 9% | 9% |
| FADE_AGAINST | 3,984 | $+1,875 | 9% | 10% | 11% | 10% | 11% | 10% | 11% | 10% | 9% | 9% |
| FADE_CALM | 26,094 | $-4,776 | 9% | 10% | 10% | 10% | 11% | 11% | 10% | 10% | 10% | 9% |
| KILL_SHOT | 3,424 | $-1,289 | 9% | 10% | 11% | 11% | 11% | 10% | 10% | 11% | 9% | 9% |
| MTF_BREAKOUT | 1,623 | $+2,521 | 9% | 10% | 10% | 11% | 9% | 11% | 10% | 10% | 10% | 8% |
| RIDE_AGAINST | 40,338 | $-9,544 | 9% | 9% | 11% | 10% | 11% | 10% | 10% | 10% | 9% | 9% |

### IS — filter+sizing decile $ contribution

Dollar contribution per decile.

| Tier | N | $ | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CASCADE | 709 | $-214 | $-4,511 | $-1,652 | $-633 | $-276 | $-17 | $+256 | $+534 | $+946 | $+1,496 | $+3,642 |
| FADE_AGAINST | 3,984 | $+1,875 | $-28,442 | $-11,024 | $-6,275 | $-2,952 | $-791 | $+1,339 | $+3,808 | $+6,376 | $+10,986 | $+28,850 |
| FADE_CALM | 26,094 | $-4,776 | $-220,754 | $-81,376 | $-42,126 | $-20,842 | $-6,352 | $+7,485 | $+20,258 | $+41,552 | $+77,490 | $+219,888 |
| KILL_SHOT | 3,424 | $-1,289 | $-20,502 | $-6,406 | $-3,702 | $-1,614 | $-174 | $+956 | $+2,038 | $+4,082 | $+6,145 | $+17,888 |
| MTF_BREAKOUT | 1,623 | $+2,521 | $-7,490 | $-2,922 | $-1,628 | $-931 | $-271 | $+193 | $+755 | $+1,476 | $+3,142 | $+10,200 |
| RIDE_AGAINST | 40,338 | $-9,544 | $-242,438 | $-85,322 | $-53,719 | $-26,478 | $-9,978 | $+5,861 | $+22,146 | $+45,766 | $+83,355 | $+251,262 |

### IS — per-tier decile SHIFT (filter+sizing vs baseline)

Decile-share shift in pp (kept trades) and $ shift. Negative pp on D1 = filter removed bottom-10% trades (good). Negative pp on D10 = filter removed top-10% trades (bad). A sign-asymmetric row is a real signal; a symmetric row (all small shifts, or D1 and D10 both down) is a variance cutter, not a loss filter.

| Tier | dN | d$ | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CASCADE | -184 | $+58 | -1pp | +1pp | -1pp | . | . | +2pp | +1pp | +1pp | . | -1pp |
| FADE_AGAINST | -1,375 | $-2,018 | -1pp | . | . | +1pp | +1pp | +1pp | +1pp | . | . | -1pp |
| FADE_CALM | -3,569 | $-1,708 | -1pp | . | . | . | +1pp | +1pp | . | . | . | -1pp |
| KILL_SHOT | -471 | $+133 | -1pp | . | . | . | . | +1pp | . | . | . | -1pp |
| MTF_BREAKOUT | -171 | $-460 | -1pp | . | . | . | . | +1pp | . | . | . | -2pp |
| RIDE_AGAINST | -6,037 | $+6,924 | -1pp | . | . | . | +1pp | +1pp | . | . | . | -1pp |

## OOS

| Strategy | Kept | Rejected | PnL | d vs Baseline | WR |
|---|---:|---:|---:|---:|---:|
| Baseline | 22,488 | 0 | $+11,712 | $+0 | 49% |
| Filter >= 2.5 | 19,673 | 2,815 | $+4,940 | $-6,772 | 49% |
| Sizing only | 22,488 | 0 | $+11,712 | $+0 | 49% |
| Filter + Sizing | 19,673 | 2,815 | $+4,940 | $-6,772 | 49% |

Skipped (N < 40): MTF_EXHAUSTION, FREIGHT_TRAIN

### OOS — per-tier decile $ boundaries

Per-tier $ boundaries for deciles D1 (worst 10%) .. D10 (best 10%), computed from that tier's BASELINE PnL distribution.

| Tier | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CASCADE | <=$-29.8 | $-29.8..$-15.0 | $-15.0..$-6.9 | $-6.9..$-1.5 | $-1.5..$+2.5 | $+2.5..$+7.5 | $+7.5..$+14.0 | $+14.0..$+23.0 | $+23.0..$+48.0 | >$+48.0 |
| FADE_AGAINST | <=$-44.5 | $-44.5..$-23.0 | $-23.0..$-11.2 | $-11.2..$-4.0 | $-4.0..$+1.5 | $+1.5..$+6.5 | $+6.5..$+15.5 | $+15.5..$+28.5 | $+28.5..$+50.0 | >$+50.0 |
| FADE_CALM | <=$-50.5 | $-50.5..$-26.0 | $-26.0..$-14.0 | $-14.0..$-6.0 | $-6.0..$+0.5 | $+0.5..$+6.0 | $+6.0..$+14.5 | $+14.5..$+26.8 | $+26.8..$+52.5 | >$+52.5 |
| KILL_SHOT | <=$-32.0 | $-32.0..$-13.5 | $-13.5..$-7.5 | $-7.5..$-2.5 | $-2.5..$+2.0 | $+2.0..$+5.5 | $+5.5..$+11.5 | $+11.5..$+21.0 | $+21.0..$+37.5 | >$+37.5 |
| MTF_BREAKOUT | <=$-33.6 | $-33.6..$-18.0 | $-18.0..$-11.0 | $-11.0..$-7.0 | $-7.0..$-2.5 | $-2.5..$+2.0 | $+2.0..$+7.5 | $+7.5..$+14.6 | $+14.6..$+29.1 | >$+29.1 |
| RIDE_AGAINST | <=$-35.0 | $-35.0..$-19.5 | $-19.5..$-11.0 | $-11.0..$-5.5 | $-5.5..$-0.5 | $-0.5..$+4.5 | $+4.5..$+11.0 | $+11.0..$+20.5 | $+20.5..$+38.0 | >$+38.0 |

### OOS — baseline decile share (sanity check: ~10% each)

Percent of kept trades in each decile (using the BASELINE thresholds). Baseline is 10% per decile by construction.

| Tier | N | $ | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CASCADE | 203 | $+1,438 | 10% | 10% | 9% | 11% | 11% | 10% | 8% | 10% | 9% | 10% |
| FADE_AGAINST | 1,120 | $+1,250 | 10% | 10% | 10% | 10% | 10% | 10% | 10% | 10% | 10% | 10% |
| FADE_CALM | 7,663 | $+530 | 10% | 10% | 10% | 10% | 11% | 9% | 10% | 10% | 10% | 10% |
| KILL_SHOT | 984 | $+2,599 | 11% | 10% | 10% | 10% | 10% | 9% | 11% | 9% | 10% | 10% |
| MTF_BREAKOUT | 480 | $-822 | 10% | 10% | 10% | 10% | 10% | 11% | 9% | 10% | 10% | 10% |
| RIDE_AGAINST | 12,033 | $+6,476 | 10% | 10% | 10% | 10% | 10% | 10% | 10% | 9% | 10% | 10% |

### OOS — filter+sizing decile share

Percent of kept trades in each decile (using the BASELINE thresholds). Baseline is 10% per decile by construction.

| Tier | N | $ | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CASCADE | 169 | $+515 | 9% | 11% | 11% | 12% | 12% | 12% | 8% | 8% | 9% | 8% |
| FADE_AGAINST | 861 | $+2,650 | 9% | 9% | 11% | 11% | 11% | 10% | 11% | 10% | 10% | 9% |
| FADE_CALM | 6,825 | $-1,892 | 9% | 10% | 10% | 10% | 11% | 10% | 11% | 10% | 10% | 9% |
| KILL_SHOT | 875 | $+1,587 | 10% | 9% | 10% | 11% | 10% | 9% | 11% | 10% | 9% | 9% |
| MTF_BREAKOUT | 436 | $-368 | 7% | 10% | 10% | 10% | 11% | 12% | 9% | 10% | 10% | 10% |
| RIDE_AGAINST | 10,503 | $+2,156 | 9% | 10% | 10% | 10% | 11% | 11% | 11% | 10% | 10% | 9% |

### OOS — filter+sizing decile $ contribution

Dollar contribution per decile.

| Tier | N | $ | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CASCADE | 169 | $+515 | $-894 | $-378 | $-184 | $-78 | $+26 | $+108 | $+138 | $+262 | $+506 | $+1,008 |
| FADE_AGAINST | 861 | $+2,650 | $-6,948 | $-2,386 | $-1,515 | $-656 | $-88 | $+316 | $+1,082 | $+1,864 | $+3,278 | $+7,703 |
| FADE_CALM | 6,825 | $-1,892 | $-61,227 | $-24,716 | $-13,570 | $-6,794 | $-1,836 | $+2,290 | $+7,502 | $+13,854 | $+25,182 | $+57,422 |
| KILL_SHOT | 875 | $+1,587 | $-5,922 | $-1,722 | $-886 | $-446 | $+22 | $+314 | $+804 | $+1,368 | $+2,316 | $+5,739 |
| MTF_BREAKOUT | 436 | $-368 | $-2,536 | $-1,048 | $-612 | $-380 | $-218 | $-3 | $+206 | $+470 | $+964 | $+2,788 |
| RIDE_AGAINST | 10,503 | $+2,156 | $-69,364 | $-27,316 | $-16,071 | $-8,135 | $-3,098 | $+2,420 | $+8,750 | $+15,810 | $+29,066 | $+70,093 |

### OOS — per-tier decile SHIFT (filter+sizing vs baseline)

Decile-share shift in pp (kept trades) and $ shift. Negative pp on D1 = filter removed bottom-10% trades (good). Negative pp on D10 = filter removed top-10% trades (bad). A sign-asymmetric row is a real signal; a symmetric row (all small shifts, or D1 and D10 both down) is a variance cutter, not a loss filter.

| Tier | dN | d$ | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CASCADE | -34 | $-924 | -1pp | . | +1pp | +2pp | +2pp | +1pp | -1pp | -2pp | . | -3pp |
| FADE_AGAINST | -259 | $+1,400 | -1pp | -1pp | +1pp | . | +1pp | . | +1pp | . | . | -1pp |
| FADE_CALM | -838 | $-2,422 | -1pp | . | . | . | +1pp | . | . | . | . | -1pp |
| KILL_SHOT | -109 | $-1,012 | . | . | . | +1pp | . | . | . | . | -1pp | . |
| MTF_BREAKOUT | -44 | $+454 | -3pp | . | . | . | +1pp | +1pp | . | . | . | . |
| RIDE_AGAINST | -1,530 | $-4,320 | -1pp | . | . | . | +1pp | . | . | . | . | -1pp |
