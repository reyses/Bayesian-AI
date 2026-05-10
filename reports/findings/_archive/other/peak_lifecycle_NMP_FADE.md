# Peak Bucket Lifecycle — NMP_FADE

**8174 trades** (4486 winners / 3644 losers)

## Part 1 — Bar-by-bar peak bucket heatmap

At each bar N of the trade life, what % of still-open trades sit in each peak bucket? Row = cohort × bar. Cell = % of trades in that bucket at that bar.

Buckets (ticks): NOISE 0-4 · FAKE 5-9 · MARGINAL 10-19 · REAL 20-39 · STRONG 40-79 · DOMINANT 80+

### WINNERS

| bar | n_open | NOISE | FAKE | MARGINAL | REAL | STRONG | DOMINANT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4486 | 56% | 11% | 14% | 11% | 6% | 2% |
| 2 | 4480 | 40% | 11% | 17% | 16% | 11% | 5% |
| 3 | 4435 | 32% | 10% | 18% | 18% | 14% | 8% |
| 5 | 4310 | 23% | 9% | 16% | 21% | 18% | 13% |
| 7 | 4112 | 18% | 8% | 16% | 21% | 20% | 17% |
| 10 | 3701 | 14% | 7% | 13% | 22% | 23% | 21% |
| 15 | 2933 | 10% | 5% | 12% | 19% | 25% | 29% |
| 20 | 2529 | 7% | 4% | 10% | 18% | 26% | 36% |
| 25 | 2168 | 6% | 3% | 8% | 16% | 25% | 42% |
| 30 | 1865 | 5% | 3% | 6% | 15% | 24% | 47% |
| 45 | 1175 | 4% | 2% | 4% | 11% | 21% | 57% |

### LOSERS

| bar | n_open | NOISE | FAKE | MARGINAL | REAL | STRONG | DOMINANT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3644 | 67% | 10% | 10% | 8% | 4% | 1% |
| 2 | 3635 | 56% | 10% | 13% | 12% | 6% | 3% |
| 3 | 3631 | 50% | 10% | 14% | 14% | 8% | 4% |
| 5 | 3623 | 45% | 9% | 14% | 16% | 10% | 6% |
| 7 | 3612 | 42% | 8% | 14% | 16% | 12% | 8% |
| 10 | 3568 | 39% | 9% | 14% | 17% | 13% | 9% |
| 15 | 3210 | 34% | 8% | 14% | 17% | 15% | 12% |
| 20 | 2758 | 29% | 7% | 13% | 19% | 17% | 15% |
| 25 | 2416 | 26% | 7% | 13% | 18% | 18% | 17% |
| 30 | 2110 | 24% | 7% | 13% | 18% | 19% | 19% |
| 45 | 1337 | 21% | 7% | 11% | 18% | 19% | 25% |

## Part 2 — Per-bucket peak signature clustering

Clusters the 91D feature vector AT the peak bar for WINNERS in each bucket. Each cluster = distinct exit signature. Top 3 distinctive features per cluster -> candidate exit rule.

### Bucket: REAL

- Winners in bucket: 733
- PCA: 10 components (47% var)
- BIC: K2=30121, K3=29869, K4=30023  ->  **K=3** selected

#### Cluster 0  —  N=295

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 1m_z_se | 1.384 | 0.020 | 0.77 | HIGH |
| 2 | 15s_dmi_diff | 13.835 | 0.206 | 0.74 | HIGH |
| 3 | 1m_dir_vol | 1.130 | 0.033 | 0.71 | HIGH |
| 4 | 1m_z_low | -0.031 | -0.959 | 0.66 | HIGH |
| 5 | 15s_z_low | 0.279 | -0.528 | 0.65 | HIGH |

**Candidate exit rule:**
```
if 1m_z_se > 1.38 AND 15s_dmi_diff > 13.84 AND 1m_dir_vol > 1.13:
    return 'real_cluster_0'
```

#### Cluster 1  —  N=337

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 1m_z_se | -1.552 | 0.020 | 0.89 | LOW |
| 2 | 1m_dir_vol | -1.180 | 0.033 | 0.78 | LOW |
| 3 | 15s_dmi_diff | -14.004 | 0.206 | 0.78 | LOW |
| 4 | 1m_z_low | -2.030 | -0.959 | 0.77 | LOW |
| 5 | 15s_z_se | -1.042 | 0.024 | 0.76 | LOW |

**Candidate exit rule:**
```
if 1m_z_se < -1.55 AND 1m_dir_vol < -1.18 AND 15s_dmi_diff < -14.00:
    return 'real_cluster_1'
```

#### Cluster 2  —  N=101

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 5m_bar_range | 82.446 | 46.839 | 1.06 | HIGH |
| 2 | 5m_z_low | -1.852 | -0.811 | 1.01 | LOW |
| 3 | 1m_velocity | 5.371 | 0.005 | 0.92 | HIGH |
| 4 | 15m_bar_range | 143.485 | 82.438 | 0.91 | HIGH |
| 5 | 15s_bar_range | 17.178 | 9.468 | 0.87 | HIGH |

**Candidate exit rule:**
```
if 5m_bar_range > 82.45 AND 5m_z_low < -1.85 AND 1m_velocity > 5.37:
    return 'real_cluster_2'
```

### Bucket: STRONG

- Winners in bucket: 1174
- PCA: 10 components (47% var)
- BIC: K2=48103, K3=47061, K4=46793  ->  **K=4** selected

#### Cluster 0  —  N=118

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 15m_bar_range | 202.864 | 96.563 | 1.49 | HIGH |
| 2 | 15m_vol_rel | 2.019 | 0.871 | 1.41 | HIGH |
| 3 | 5m_bar_range | 114.051 | 57.810 | 1.39 | HIGH |
| 4 | 1m_bar_range | 65.432 | 34.534 | 1.21 | HIGH |
| 5 | 1h_vol_rel | 1.754 | 0.703 | 1.20 | HIGH |

**Candidate exit rule:**
```
if 15m_bar_range > 202.86 AND 15m_vol_rel > 2.02 AND 5m_bar_range > 114.05:
    return 'strong_cluster_0'
```

#### Cluster 1  —  N=243

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 1m_z_se | 1.600 | -0.005 | 0.90 | HIGH |
| 2 | 15s_z_se | 1.126 | -0.009 | 0.81 | HIGH |
| 3 | 1m_velocity | 5.994 | -0.096 | 0.81 | HIGH |
| 4 | 1m_dir_vol | 1.397 | -0.022 | 0.77 | HIGH |
| 5 | 15s_dmi_diff | 14.746 | -0.591 | 0.75 | HIGH |

**Candidate exit rule:**
```
if 1m_z_se > 1.60 AND 15s_z_se > 1.13 AND 1m_velocity > 5.99:
    return 'strong_cluster_1'
```

#### Cluster 2  —  N=490

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 1m_z_se | -1.532 | -0.005 | 0.86 | LOW |
| 2 | 15s_dmi_diff | -17.862 | -0.591 | 0.84 | LOW |
| 3 | 15s_z_se | -1.089 | -0.009 | 0.77 | LOW |
| 4 | 1m_dir_vol | -1.436 | -0.022 | 0.77 | LOW |
| 5 | 1m_velocity | -5.202 | -0.096 | 0.68 | LOW |

**Candidate exit rule:**
```
if 1m_z_se < -1.53 AND 15s_dmi_diff < -17.86 AND 15s_z_se < -1.09:
    return 'strong_cluster_2'
```

#### Cluster 3  —  N=323

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 1m_dmi_diff | 9.310 | -0.420 | 0.82 | HIGH |
| 2 | 15s_dmi_diff | 16.102 | -0.591 | 0.81 | HIGH |
| 3 | 1m_z_se | 1.291 | -0.005 | 0.73 | HIGH |
| 4 | 1m_dir_vol | 1.267 | -0.022 | 0.70 | HIGH |
| 5 | 15s_z_se | 0.897 | -0.009 | 0.65 | HIGH |

**Candidate exit rule:**
```
if 1m_dmi_diff > 9.31 AND 15s_dmi_diff > 16.10 AND 1m_z_se > 1.29:
    return 'strong_cluster_3'
```

### Bucket: DOMINANT

- Winners in bucket: 2153
- PCA: 10 components (49% var)
- BIC: K2=86470, K3=84197, K4=83574  ->  **K=4** selected

#### Cluster 0  —  N=672

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 15s_dmi_diff | -20.416 | 0.079 | 0.94 | LOW |
| 2 | 1m_z_se | -1.577 | 0.024 | 0.93 | LOW |
| 3 | 1m_dir_vol | -1.641 | 0.052 | 0.83 | LOW |
| 4 | 15s_z_se | -1.091 | 0.022 | 0.82 | LOW |
| 5 | 1m_dmi_diff | -13.454 | -0.874 | 0.77 | LOW |

**Candidate exit rule:**
```
if 15s_dmi_diff < -20.42 AND 1m_z_se < -1.58 AND 1m_dir_vol < -1.64:
    return 'dominant_cluster_0'
```

#### Cluster 1  —  N=907

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 15s_dmi_diff | 17.194 | 0.079 | 0.79 | HIGH |
| 2 | 1m_dmi_diff | 11.489 | -0.874 | 0.76 | HIGH |
| 3 | 1m_z_se | 1.256 | 0.024 | 0.71 | HIGH |
| 4 | 1m_dir_vol | 1.342 | 0.052 | 0.63 | HIGH |
| 5 | 15s_z_se | 0.824 | 0.022 | 0.59 | HIGH |

**Candidate exit rule:**
```
if 15s_dmi_diff > 17.19 AND 1m_dmi_diff > 11.49 AND 1m_z_se > 1.26:
    return 'dominant_cluster_1'
```

#### Cluster 2  —  N=188

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 15s_bar_range | 66.447 | 25.623 | 1.40 | HIGH |
| 2 | 1m_bar_range | 156.005 | 66.653 | 1.37 | HIGH |
| 3 | 5m_bar_range | 232.239 | 111.392 | 1.30 | HIGH |
| 4 | 15m_bar_range | 392.707 | 184.742 | 1.18 | HIGH |
| 5 | 1h_bar_range | 712.697 | 397.855 | 0.95 | HIGH |

**Candidate exit rule:**
```
if 15s_bar_range > 66.45 AND 1m_bar_range > 156.01 AND 5m_bar_range > 232.24:
    return 'dominant_cluster_2'
```

#### Cluster 3  —  N=386

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 5m_dmi_diff | -17.202 | -1.470 | 1.13 | LOW |
| 2 | 15m_z_se | -1.319 | -0.047 | 1.00 | LOW |
| 3 | 15m_dmi_diff | -15.888 | -2.173 | 0.94 | LOW |
| 4 | 1h_velocity | -77.012 | -0.071 | 0.93 | LOW |
| 5 | 15m_z_low | -1.868 | -0.747 | 0.92 | LOW |

**Candidate exit rule:**
```
if 5m_dmi_diff < -17.20 AND 15m_z_se < -1.32 AND 15m_dmi_diff < -15.89:
    return 'dominant_cluster_3'
```

---
_Generated by `tools/peak_bucket_lifecycle.py --tier NMP_FADE`_