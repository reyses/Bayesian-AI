# Regime label verification

Generated: 2026-04-29 09:33

## Refined v2 split: UP/DOWN by efficiency_ratio (>= 0.5 = pure, < 0.5 = trending chop)

Tests whether the original UP/DOWN labels conflate two distinct archetypes:
  - PURE: smooth directional move (high efficiency)
  - TRENDING_CHOP: directional but jagged (low efficiency)

If many UP days are TRENDING_CHOP, the original label is mushy.

## v2 distribution

| Regime v2 | N | % |
|---|---:|---:|
| CHOP | 130 | 37.4% |
| TRANSITIONAL | 98 | 28.2% |
| QUIET | 47 | 13.5% |
| UP_TRENDING_CHOP | 36 | 10.3% |
| DOWN_TRENDING_CHOP | 33 | 9.5% |
| UNKNOWN | 4 | 1.1% |

## Per-regime metric averages (v2)

```
         regime_v2   n  avg_dir  avg_eff  avg_range     avg_net
              CHOP 130 0.216950 0.013702   0.947232   -3.578846
DOWN_TRENDING_CHOP  33 0.767173 0.071361   1.776387 -507.848485
             QUIET  47 0.353027 0.030948   0.368481   11.148936
      TRANSITIONAL  98 0.594464 0.054135   1.005196   19.150510
           UNKNOWN   4 0.390988 0.038169        NaN   62.437500
  UP_TRENDING_CHOP  36 0.761869 0.073182   1.506176  448.055556
```

## Interpretation guide

| Metric pattern | Archetype |
|---|---|
| dir>0.6, eff>0.5, exp>1.0 | Pure UP/DOWN trend |
| dir>0.6, eff<0.5, exp>1.0 | Trending chop (directional but messy) |
| dir<0.4, exp>1.0 | Pure chop (non-directional, high vol) |
| dir<0.4, exp<0.5 | Quiet/flat (no direction, low vol) |
