# Max-Fill Tier EDA -- 2026-04-11

## Tier Summary

| Tier | Trades | WR | $/trade | $/day | Total |
|------|--------|-----|---------|-------|-------|
| MTF_BREAKOUT | 22,445 | 56% | $12.8 | $1,037 | $287,244 |
| FADE_CALM | 11,135 | 66% | $7.3 | $293 | $81,126 |
| RIDE_AGAINST | 9,950 | 52% | $8.0 | $288 | $79,838 |
| KILL_SHOT | 6,574 | 72% | $7.3 | $173 | $48,006 |
| MTF_EXHAUSTION | 924 | 76% | $51.8 | $173 | $47,826 |
| FADE_AGAINST | 796 | 77% | $12.6 | $36 | $10,004 |
| CASCADE | 1,601 | 71% | $4.0 | $23 | $6,448 |
| FREIGHT_TRAIN | 14 | 86% | $103.2 | $5 | $1,446 |
| FADE_MOMENTUM | 177 | 61% | $-1.3 | $-1 | $-228 |
| ABSORPTION | 16,058 | 21% | $-0.0 | $-2 | $-666 |
| REGIME_FLIP | 2,432 | 22% | $-0.4 | $-4 | $-1,012 |
| EXHAUSTION_BAR | 213 | 46% | $-11.2 | $-9 | $-2,390 |

Days: 277

## Direction

| Tier | Dir | Trades | WR | Avg |
|------|-----|--------|-----|-----|
| ABSORPTION | long | 8129 | 22% | $0.0 |
| ABSORPTION | short | 7929 | 20% | $-0.1 |
| CASCADE | long | 883 | 73% | $4.3 |
| CASCADE | short | 718 | 69% | $3.7 |
| EXHAUSTION_BAR | long | 132 | 48% | $-2.2 |
| EXHAUSTION_BAR | short | 81 | 42% | $-25.9 |
| FADE_AGAINST | long | 425 | 76% | $12.1 |
| FADE_AGAINST | short | 371 | 78% | $13.1 |
| FADE_CALM | long | 6075 | 67% | $7.3 |
| FADE_CALM | short | 5060 | 66% | $7.3 |
| FADE_MOMENTUM | long | 93 | 62% | $14.4 |
| FADE_MOMENTUM | short | 84 | 60% | $-18.7 |
| FREIGHT_TRAIN | long | 8 | 75% | $61.3 |
| FREIGHT_TRAIN | short | 6 | 100% | $159.2 |
| KILL_SHOT | long | 3537 | 72% | $7.4 |
| KILL_SHOT | short | 3037 | 71% | $7.2 |
| MTF_BREAKOUT | long | 10126 | 57% | $11.3 |
| MTF_BREAKOUT | short | 12319 | 55% | $14.0 |
| MTF_EXHAUSTION | long | 82 | 76% | $43.0 |
| MTF_EXHAUSTION | short | 842 | 76% | $52.6 |
| REGIME_FLIP | long | 1221 | 21% | $-0.9 |
| REGIME_FLIP | short | 1211 | 24% | $0.1 |
| RIDE_AGAINST | long | 4788 | 52% | $8.0 |
| RIDE_AGAINST | short | 5162 | 52% | $8.0 |

## Exit Reasons

### ABSORPTION

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| absorb_vr_rising | 7533 | 13% | $1.1 |
| absorb_ride_failing | 3915 | 27% | $-2.4 |
| absorb_z_collapsed | 3286 | 14% | $-1.9 |
| absorb_vol_persistent | 1290 | 66% | $4.7 |
| max_bars | 28 | 21% | $-0.6 |
| giveback | 6 | 100% | $15.2 |

### CASCADE

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| cascade_center | 1344 | 84% | $7.3 |
| cascade_no_conviction | 252 | 3% | $-13.7 |
| giveback | 4 | 100% | $17.6 |
| max_bars | 1 | 0% | $0.0 |

### EXHAUSTION_BAR

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| exhaust_mean | 92 | 73% | $37.0 |
| exhaust_no_conviction | 63 | 0% | $-51.4 |
| giveback | 46 | 65% | $3.4 |
| hard_stop | 12 | 0% | $-225.6 |

### FADE_AGAINST

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| mean_reached | 535 | 89% | $17.9 |
| z_reversal | 245 | 51% | $2.6 |
| giveback | 11 | 82% | $9.2 |
| max_bars | 3 | 0% | $-2.0 |
| hard_stop | 2 | 0% | $-157.8 |

### FADE_CALM

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| mean_reached | 7457 | 78% | $11.3 |
| z_reversal | 3424 | 42% | $0.7 |
| giveback | 187 | 72% | $6.5 |
| hard_stop | 39 | 0% | $-175.1 |
| max_bars | 28 | 25% | $-3.6 |

### FADE_MOMENTUM

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| mean_reached | 99 | 75% | $45.5 |
| giveback | 33 | 70% | $5.1 |
| hard_stop | 24 | 0% | $-225.9 |
| z_reversal | 21 | 52% | $24.9 |

### FREIGHT_TRAIN

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| freight_decel | 10 | 100% | $161.2 |
| giveback | 3 | 67% | $11.2 |
| hard_stop | 1 | 0% | $-199.5 |

### KILL_SHOT

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| kill_shot_center | 5469 | 85% | $12.3 |
| kill_shot_no_conviction | 1062 | 4% | $-16.3 |
| giveback | 29 | 62% | $-5.7 |
| max_bars | 7 | 29% | $1.0 |
| hard_stop | 7 | 0% | $-219.9 |

### MTF_BREAKOUT

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| breakout_overshot | 16050 | 51% | $13.5 |
| breakout_alignment_lost | 3828 | 60% | $12.5 |
| giveback | 2445 | 83% | $14.8 |
| hard_stop | 63 | 0% | $-223.1 |
| max_bars | 59 | 68% | $9.3 |

### MTF_EXHAUSTION

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| mtf_mean | 456 | 65% | $46.6 |
| giveback | 251 | 83% | $22.1 |
| mtf_5m_reaccel | 133 | 95% | $130.5 |
| mtf_1m_exhausted | 82 | 82% | $48.5 |
| hard_stop | 2 | 0% | $-155.5 |

### REGIME_FLIP

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| regime_vr_rising | 1035 | 35% | $3.4 |
| regime_z_collapsed | 926 | 11% | $-2.3 |
| regime_ride_failing | 427 | 12% | $-5.5 |
| giveback | 43 | 63% | $3.0 |
| hard_stop | 1 | 0% | $-160.5 |

### RIDE_AGAINST

| Exit | Count | WR | Avg |
|------|-------|-----|-----|
| mean_reached | 8294 | 48% | $7.5 |
| giveback | 1049 | 85% | $19.9 |
| ride_1h_flipped | 497 | 53% | $6.7 |
| max_bars | 67 | 48% | $9.1 |
| hard_stop | 43 | 0% | $-175.7 |

