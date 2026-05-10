# 1h band window sweep

Generated: 2026-04-23T02:39:14
1h bars used: 6524 total
5s files scanned: 345

## Gaussian target (for reference)
- 68% within ±1σ
- 95% within ±2σ
- 99.7% within ±3σ

## Per-window results

| Window (h) | Days | Median SE ($) | %within±1σ | %within±2σ | %within±3σ | max\|z\| p50 | max\|z\| p90 | SE CoV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 344 | $87.2 | 60.9% | 93.4% | 99.3% | 3.2 | 5.0 | 0.62 |
| 24 | 344 | $130.6 | 55.5% | 89.4% | 97.2% | 3.1 | 4.3 | 0.63 |
| 48 | 341 | $175.8 | 52.2% | 86.4% | 96.9% | 2.7 | 4.3 | 0.69 |
| 60 | 341 | $193.5 | 51.4% | 87.5% | 97.3% | 2.5 | 4.2 | 0.69 |
| 120 | 338 | $284.5 | 49.3% | 84.1% | 96.1% | 2.2 | 3.8 | 0.57 |
| 168 | 335 | $347.2 | 47.5% | 83.8% | 96.2% | 2.1 | 3.5 | 0.56 |
| 336 | 326 | $486.6 | 52.1% | 86.2% | 96.0% | 1.7 | 3.2 | 0.45 |

## Interpretation

Read the %within±1σ column. The **closer to 68%**, the more gaussian the price distribution relative to that window. If %within±1σ is **much higher than 68%** (e.g. 85-95%), the bands are TOO WIDE — the window absorbs more volatility than is locally active. If **much lower** (e.g. 40-50%), bands are TOO NARROW — window over-reacts to recent noise and bands whip around.
