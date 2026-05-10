# Tier segment diagnostic — IS chronological stability

IS split into 2 chronological segments; same tier measured in each. Overfit rules should show divergence within IS (decay pattern, sign flips, or high variance).

Legend: **DECAY** = monotonic decline. **FLIPS(N)** = N sign changes across segments. **stable** = consistent.

## IS (2 segments) — $/trade across segments

Header shows segment index and day range.

| Tier | Overall | S1 (2025_01_01..2025_06_20) | S2 (2025_07_01..2025_12_19) | std | range | pattern |
|---|---:|---:|---:|---:|---:|---:|
| CASCADE | $+23.20 (n=66) | $-11.01 (n=36) | $+64.25 (n=30) | $37.63 | $75.26 | FLIPS (1) |
| FADE_AGAINST | $+51.14 (n=487) | $+67.30 (n=227) | $+37.03 (n=260) | $15.13 | $30.26 | DECAY |
| KILL_SHOT_ACTIVE | $+24.36 (n=36) | $+21.63 (n=27) | $+32.56 (n=9) | $5.46 | $10.93 | IMPROVING |
| KILL_SHOT_CALM | $+2.22 (n=267) | $+0.03 (n=130) | $+4.29 (n=137) | $2.13 | $4.25 | IMPROVING |
| MTF_BREAKOUT | $+3.81 (n=26) | $+27.75 (n=4) | $-0.55 (n=22) | $14.15 | $28.30 | FLIPS (1) |
| MTF_EXHAUSTION | $-14.35 (n=20) | $-62.23 (n=11) | $+44.17 (n=9) | $53.20 | $106.39 | FLIPS (1) |
| NMP_FADE | $+5.21 (n=3,538) | $+6.30 (n=1858) | $+3.99 (n=1680) | $1.15 | $2.31 | DECAY |
| NMP_RIDE | $+60.37 (n=619) | $+98.65 (n=351) | $+10.24 (n=268) | $44.21 | $88.41 | DECAY |
| RIDE_AGAINST | $+3.76 (n=70) | $+17.90 (n=25) | $-4.10 (n=45) | $11.00 | $22.00 | FLIPS (1) |
| TREND_FOLLOWER | $+17.77 (n=139) | $+26.74 (n=68) | $+9.18 (n=71) | $8.78 | $17.56 | DECAY |

## IS (2 segments) — total $ across segments

| Tier | Overall $ | S1 | S2 | sum check |
|---|---:|---:|---:|---:|
| CASCADE | $+1,531 | $-396 | $+1,928 | ✓ |
| FADE_AGAINST | $+24,905 | $+15,276 | $+9,629 | ✓ |
| KILL_SHOT_ACTIVE | $+877 | $+584 | $+293 | ✓ |
| KILL_SHOT_CALM | $+592 | $+4 | $+588 | ✓ |
| MTF_BREAKOUT | $+99 | $+111 | $-12 | ✓ |
| MTF_EXHAUSTION | $-287 | $-684 | $+398 | ✓ |
| NMP_FADE | $+18,418 | $+11,710 | $+6,708 | ✓ |
| NMP_RIDE | $+37,368 | $+34,626 | $+2,743 | ✓ |
| RIDE_AGAINST | $+263 | $+448 | $-184 | ✓ |
| TREND_FOLLOWER | $+2,470 | $+1,818 | $+652 | ✓ |
| **TOTAL** | **$+86,238** | **$+63,496** | **$+22,742** |  |

## IS (2 segments) — WR across segments

| Tier | Overall WR | S1 | S2 |
|---|---:|---:|---:|
| CASCADE | 50% | 44% | 57% |
| FADE_AGAINST | 48% | 46% | 49% |
| KILL_SHOT_ACTIVE | 83% | 85% | 78% |
| KILL_SHOT_CALM | 62% | 61% | 62% |
| MTF_BREAKOUT | 65% | 75% | 64% |
| MTF_EXHAUSTION | 25% | 27% | 22% |
| NMP_FADE | 60% | 61% | 59% |
| NMP_RIDE | 54% | 55% | 52% |
| RIDE_AGAINST | 51% | 57% | 49% |
| TREND_FOLLOWER | 74% | 74% | 75% |

---

## IS (4 quarters) — $/trade across segments

Header shows segment index and day range.

| Tier | Overall | S1 (2025_01_01..2025_03_21) | S2 (2025_04_01..2025_06_20) | S3 (2025_07_01..2025_09_18) | S4 (2025_09_19..2025_12_19) | std | range | pattern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CASCADE | $+23.20 (n=66) | $-92.06 (n=16) | $+53.83 (n=20) | $+24.21 (n=17) | $+116.62 (n=13) | $75.71 | $208.68 | FLIPS (1) |
| FADE_AGAINST | $+51.14 (n=487) | $+27.01 (n=105) | $+101.97 (n=122) | $+5.64 (n=136) | $+71.46 (n=124) | $37.58 | $96.32 | stable |
| KILL_SHOT_ACTIVE | $+24.36 (n=36) | $+19.80 (n=15) | $+23.92 (n=12) | $+64.00 (n=2) | $+23.57 (n=7) | $18.07 | $44.20 | stable |
| KILL_SHOT_CALM | $+2.22 (n=267) | $+1.67 (n=78) | $-2.42 (n=52) | $+2.62 (n=72) | $+6.14 (n=65) | $3.05 | $8.56 | FLIPS (2) |
| MTF_BREAKOUT | $+3.81 (n=26) | — | $+27.75 (n=4) | $-0.03 (n=15) | $-1.64 (n=7) | $13.49 | $29.39 | FLIPS (1) |
| MTF_EXHAUSTION | $-14.35 (n=20) | $-112.88 (n=4) | $-33.29 (n=7) | $-304.50 (n=1) | $+87.75 (n=8) | $142.60 | $392.25 | FLIPS (1) |
| NMP_FADE | $+5.21 (n=3,538) | $+5.94 (n=950) | $+6.68 (n=908) | $+0.44 (n=831) | $+7.47 (n=849) | $2.76 | $7.03 | stable |
| NMP_RIDE | $+60.37 (n=619) | $+45.36 (n=193) | $+163.74 (n=158) | $+5.47 (n=138) | $+15.29 (n=130) | $63.09 | $158.27 | stable |
| RIDE_AGAINST | $+3.76 (n=70) | $+25.44 (n=17) | $+1.88 (n=8) | $-2.23 (n=28) | $-7.18 (n=17) | $12.52 | $32.62 | FLIPS (1) |
| TREND_FOLLOWER | $+17.77 (n=139) | $+46.20 (n=28) | $+13.12 (n=40) | $+8.96 (n=28) | $+9.33 (n=43) | $15.55 | $37.23 | stable |

## IS (4 quarters) — total $ across segments

| Tier | Overall $ | S1 | S2 | S3 | S4 | sum check |
|---|---:|---:|---:|---:|---:|---:|
| CASCADE | $+1,531 | $-1,473 | $+1,076 | $+412 | $+1,516 | ✓ |
| FADE_AGAINST | $+24,905 | $+2,836 | $+12,440 | $+768 | $+8,862 | ✓ |
| KILL_SHOT_ACTIVE | $+877 | $+297 | $+287 | $+128 | $+165 | ✓ |
| KILL_SHOT_CALM | $+592 | $+130 | $-126 | $+188 | $+399 | ✓ |
| MTF_BREAKOUT | $+99 | — | $+111 | $-0 | $-12 | ✓ |
| MTF_EXHAUSTION | $-287 | $-452 | $-233 | $-304 | $+702 | ✓ |
| NMP_FADE | $+18,418 | $+5,642 | $+6,068 | $+365 | $+6,344 | ✓ |
| NMP_RIDE | $+37,368 | $+8,754 | $+25,871 | $+755 | $+1,988 | ✓ |
| RIDE_AGAINST | $+263 | $+432 | $+15 | $-62 | $-122 | ✓ |
| TREND_FOLLOWER | $+2,470 | $+1,294 | $+525 | $+251 | $+401 | ✓ |
| **TOTAL** | **$+86,238** | **$+17,461** | **$+46,035** | **$+2,499** | **$+20,242** |  |

---

## OOS (reference, no segmentation — too few days)

**4,521 trades · $+4,755 total**

| Tier | N | $ | $/trade | WR |
|---|---:|---:|---:|---:|
| NMP_RIDE | 65 | $+3,486 | $+53.64 | 51% |
| CASCADE | 28 | $+3,303 | $+117.96 | 71% |
| RIDE_AGAINST | 1,065 | $+2,640 | $+2.48 | 69% |
| MTF_BREAKOUT | 185 | $+971 | $+5.25 | 60% |
| MTF_EXHAUSTION | 33 | $+626 | $+18.95 | 48% |
| KILL_SHOT_ACTIVE | 46 | $+474 | $+10.29 | 65% |
| KILL_SHOT_CALM | 73 | $-364 | $-4.98 | 53% |
| TREND_FOLLOWER | 235 | $-378 | $-1.61 | 68% |
| NMP_FADE | 2,713 | $-2,616 | $-0.96 | 56% |
| FADE_AGAINST | 78 | $-3,388 | $-43.43 | 44% |
