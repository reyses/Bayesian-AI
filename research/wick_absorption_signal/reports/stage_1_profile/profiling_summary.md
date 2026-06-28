# Wick Absorption Signal - Stage 1 Profiling

**Date/Time:** Generated automatically.

## Execution Complete
The script `stage_1_profiling.py` has successfully mapped the candle anatomies and cubic oscillations across all timeframes: ['5s', '15s', '1m', '5m', '15m', '1h'].

## Deliverables Generated in `reports/stage_1_profile/`:
- **Univariate Statistics:** `<tf>_summary_stats.csv` (includes skew, kurtosis, and tails)
- **Histograms:** `<tf>_histograms.png`
- **Correlations:** `<tf>_correlations.png`
- **Joint Distributions:** `<tf>_joint_wick_vs_wick.png` and `<tf>_joint_range_vs_volume.png`
- **Temporal Clustering (ACF):** `<tf>_autocorr.png`
- **Oscillation Map:** `oscillation_map.csv` (cubic regression turning points per TF)

### Oscillation Map Overview:
| tf | half_cycle_bars | half_cycle_minutes | turns |
| --- | --- | --- | --- |
| 5s | 4.917875633615208 | 0.40982296946793395 | 331827 |
| 15s | 4.849192537687543 | 1.2122981344218857 | 139636 |
| 1m | 4.597077568785948 | 4.597077568785948 | 38599 |
| 5m | 4.634630451815095 | 23.173152259075476 | 7659 |
| 15m | 4.642464678178964 | 69.63697017268446 | 2549 |
| 1h | 4.246043165467626 | 254.76258992805754 | 696 |

The profiles have been fully built without crossing the layers or testing hypotheses. Ready for collaborative review to determine where the structural signals live.

## Visual Profiles: 5s
### Histograms
![5s Histograms](./5s_histograms.png)
### Correlations
![5s Correlations](./5s_correlations.png)
### Joint: Wick vs Wick
![5s Wick vs Wick](./5s_joint_wick_vs_wick.png)
### Joint: Range vs Volume
![5s Range vs Vol](./5s_joint_range_vs_volume.png)
### Temporal (ACF)
![5s ACF](./5s_autocorr.png)

## Visual Profiles: 15s
### Histograms
![15s Histograms](./15s_histograms.png)
### Correlations
![15s Correlations](./15s_correlations.png)
### Joint: Wick vs Wick
![15s Wick vs Wick](./15s_joint_wick_vs_wick.png)
### Joint: Range vs Volume
![15s Range vs Vol](./15s_joint_range_vs_volume.png)
### Temporal (ACF)
![15s ACF](./15s_autocorr.png)

## Visual Profiles: 1m
### Histograms
![1m Histograms](./1m_histograms.png)
### Correlations
![1m Correlations](./1m_correlations.png)
### Joint: Wick vs Wick
![1m Wick vs Wick](./1m_joint_wick_vs_wick.png)
### Joint: Range vs Volume
![1m Range vs Vol](./1m_joint_range_vs_volume.png)
### Temporal (ACF)
![1m ACF](./1m_autocorr.png)

## Visual Profiles: 5m
### Histograms
![5m Histograms](./5m_histograms.png)
### Correlations
![5m Correlations](./5m_correlations.png)
### Joint: Wick vs Wick
![5m Wick vs Wick](./5m_joint_wick_vs_wick.png)
### Joint: Range vs Volume
![5m Range vs Vol](./5m_joint_range_vs_volume.png)
### Temporal (ACF)
![5m ACF](./5m_autocorr.png)

## Visual Profiles: 15m
### Histograms
![15m Histograms](./15m_histograms.png)
### Correlations
![15m Correlations](./15m_correlations.png)
### Joint: Wick vs Wick
![15m Wick vs Wick](./15m_joint_wick_vs_wick.png)
### Joint: Range vs Volume
![15m Range vs Vol](./15m_joint_range_vs_volume.png)
### Temporal (ACF)
![15m ACF](./15m_autocorr.png)

## Visual Profiles: 1h
### Histograms
![1h Histograms](./1h_histograms.png)
### Correlations
![1h Correlations](./1h_correlations.png)
### Joint: Wick vs Wick
![1h Wick vs Wick](./1h_joint_wick_vs_wick.png)
### Joint: Range vs Volume
![1h Range vs Vol](./1h_joint_range_vs_volume.png)
### Temporal (ACF)
![1h ACF](./1h_autocorr.png)
