# Regime-aware EDA cross-tab — ZigzagRunner R=45.0

Generated: 2026-04-29 07:08

## Setup

- Atlas: `DATA/ATLAS` window 2025-01-01 -> 2026-03-21
- Bars: 391,514 1m
- Regime labels from `DATA/ATLAS/regime_labels.csv`
- LinReg period: 30
- R: 45.0 pts

## PnL ($) per regime × direction-mode

```
              counter  skip_T0.5  skip_T1.0  adaptive_T0.5  adaptive_T1.0  stay_T0.5  stay_T1.0      BEST_MODE  BEST_PNL
UP             +3,138     +2,834     +7,834             +6         +2,832     +2,495     +7,849      stay_T1.0    +7,849
DOWN           -1,637     +4,019     +5,847         +5,321         +8,770     +5,024     +6,347  adaptive_T1.0    +8,770
CHOP          -14,902    -12,834    -13,672        -15,329        -14,184    -12,879    -13,498      skip_T0.5   -12,834
QUIET          -4,047     -2,911     -2,687         -3,388         -2,241     -2,468     -2,730  adaptive_T1.0    -2,241
TRANSITIONAL   +1,059       -771     +1,461         -5,899         -2,974       +387     +1,937      stay_T1.0    +1,937
UNKNOWN        +1,168       +475       -103           +185           -384       +472       -109        counter    +1,168
ALL           -15,221     -9,187     -1,319        -19,104         -8,181     -6,969       -204      stay_T1.0      -204
```

`BEST_MODE` = mode with highest PnL on that regime row.
Use this to decide regime-specific direction policy.

## Trade count per regime × direction-mode

```
              counter  skip_T0.5  skip_T1.0  adaptive_T0.5  adaptive_T1.0  stay_T0.5  stay_T1.0
UP                730        438        486            730            730        401        427
DOWN              884        482        556            884            884        458        532
CHOP            2,438      1,484      1,665          2,438          2,438      1,469      1,664
QUIET             269        200        221            269            269        216        226
TRANSITIONAL    1,451        928      1,040          1,451          1,451        960      1,053
UNKNOWN            45         30         33             45             45         31         35
ALL             5,817      3,562      4,001          5,817          5,817      3,535      3,937
```

## Win rate (%) per regime × direction-mode

```
              counter  skip_T0.5  skip_T1.0  adaptive_T0.5  adaptive_T1.0  stay_T0.5  stay_T1.0
UP               40.1       44.3       45.7           46.8           47.0       37.9       41.2
DOWN             35.5       44.8       42.8           50.5           49.0       37.1       39.3
CHOP             35.6       40.8       39.2           45.7           44.2       36.7       36.7
QUIET            33.1       32.5       33.5           37.2           37.5       33.8       32.7
TRANSITIONAL     36.0       38.0       37.7           44.5           43.0       36.8       36.7
UNKNOWN          42.2       40.0       36.4           48.9           42.2       38.7       34.3
ALL              36.2       40.6       39.7           45.9           44.7       36.7       37.3
```

## Reading guide

- **counter**: v1.0.4 baseline (HighPivot->Short, LowPivot->Long).
- **skip_T0.5 / skip_T1.0**: counter-trend default but SKIP entry+exit if slope opposes would-be direction by > T.
- **adaptive_T0.5 / adaptive_T1.0**: counter-trend default but FLIP to with-slope direction when |slope| > T.
- **stay_T0.5 / stay_T1.0**: counter-trend default but DO NOT EXIT existing position when slope still favors it (skip the would-be exit + new entry).

## Decision matrix

| Regime | Best mode | If best is counter | If best is skip/stay/adaptive |
|---|---|---|---|
| UP | (see table) | Counter survives in UP | Apply that mode in UP days |
| DOWN | (see table) | Counter survives in DOWN | Apply that mode in DOWN days |
| CHOP | counter (likely) | Counter is the sweet spot | Investigate why |
| QUIET | (see table) | Low-volume, low PnL all modes | Maybe skip entirely |

Next step: build a regime-aware NT8 strategy that picks the BEST_MODE per regime detected in real-time.
