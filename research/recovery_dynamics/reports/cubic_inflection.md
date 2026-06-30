# Cubic / inflection at no-return anchors — Moises' hypothesis (31092 sampled anchors)
centered +/-20-min cubic fit. |d3| = cubic strength (normalized); inflection = y''=0 in window.

## NO-RETURN (trend start) vs RETURN anchors
- cubic strength |d3|:      no-return **0.609** vs return 0.691 (ratio 0.88x)
- inflection inside window: no-return **82%** vs return 85%
- inflection distance to anchor (if in window): no-return 0.21 vs return 0.20 (0=at anchor)

## Cubic strength |d3| by PERIOD bucket (large-period zones carry stronger cubic?)
```
        period | mean |d3| | infl-in-window | n
          2-5m |    0.662 |           84% | 14472
         5-15m |    0.673 |           84% | 7523
        15-30m |    0.914 |           90% | 2721
        30-60m |    0.713 |           87% | 1728
       60-360m |    0.653 |           85% | 2204
     NO-RETURN |    0.609 |           82% | 2444
```

## Read
No-return anchors carry a ~equal cubic signature (0.88x). If the cubic strength also rises with period, the large-period/trend zones ARE
the cubic/inflection zones (hypothesis supported). Descriptive only (centered window + forward
no-return label); the payoff is a CAUSAL trailing-cubic launch detector — the next build.
