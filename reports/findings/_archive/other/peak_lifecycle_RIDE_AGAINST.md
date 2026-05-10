# Peak Bucket Lifecycle — RIDE_AGAINST

**4204 trades** (2719 winners / 1463 losers)

## Part 1 — Bar-by-bar peak bucket heatmap

At each bar N of the trade life, what % of still-open trades sit in each peak bucket? Row = cohort × bar. Cell = % of trades in that bucket at that bar.

Buckets (ticks): NOISE 0-4 · FAKE 5-9 · MARGINAL 10-19 · REAL 20-39 · STRONG 40-79 · DOMINANT 80+

### WINNERS

| bar | n_open | NOISE | FAKE | MARGINAL | REAL | STRONG | DOMINANT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2711 | 51% | 11% | 13% | 14% | 8% | 3% |
| 2 | 2244 | 42% | 12% | 18% | 16% | 9% | 3% |
| 3 | 1834 | 37% | 13% | 22% | 16% | 8% | 5% |
| 5 | 1346 | 34% | 13% | 24% | 13% | 10% | 6% |
| 7 | 1019 | 29% | 14% | 27% | 13% | 10% | 7% |
| 10 | 723 | 23% | 12% | 28% | 16% | 11% | 10% |
| 15 | 382 | 6% | 9% | 36% | 21% | 15% | 12% |
| 20 | 21 | 24% | 10% | 29% | 14% | 14% | 10% |
| 25 | 16 | 31% | 12% | 31% | 12% | 12% | 0% |
| 30 | 15 | 33% | 13% | 33% | 13% | 7% | 0% |
| 45 | 15 | 33% | 13% | 33% | 13% | 7% | 0% |

### LOSERS

| bar | n_open | NOISE | FAKE | MARGINAL | REAL | STRONG | DOMINANT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1463 | 81% | 7% | 8% | 2% | 1% | 0% |
| 2 | 1451 | 74% | 9% | 12% | 3% | 2% | 0% |
| 3 | 1434 | 69% | 10% | 15% | 3% | 2% | 0% |
| 5 | 1390 | 66% | 11% | 18% | 2% | 2% | 1% |
| 7 | 1350 | 64% | 12% | 20% | 1% | 1% | 1% |
| 10 | 1280 | 62% | 12% | 22% | 1% | 1% | 1% |
| 15 | 1185 | 61% | 13% | 24% | 1% | 0% | 0% |
| 20 | 12 | 75% | 0% | 17% | 8% | 0% | 0% |
| 25 | 11 | 73% | 0% | 9% | 18% | 0% | 0% |
| 30 | 10 | 80% | 0% | 10% | 10% | 0% | 0% |
| 45 | 10 | 80% | 0% | 10% | 10% | 0% | 0% |

## Part 2 — Per-bucket peak signature clustering

Clusters the 91D feature vector AT the peak bar for WINNERS in each bucket. Each cluster = distinct exit signature. Top 3 distinctive features per cluster -> candidate exit rule.

### Bucket: REAL

- Winners in bucket: 1478
- PCA: 10 components (45% var)
- BIC: K2=59647, K3=58802, K4=58610  ->  **K=4** selected

#### Cluster 0  —  N=377

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 5m_dmi_diff | 11.029 | -1.380 | 0.88 | HIGH |
| 2 | 15s_z_se | -1.002 | 0.018 | 0.87 | LOW |
| 3 | 1m_dir_vol | -0.905 | 0.030 | 0.84 | LOW |
| 4 | 15s_z_low | -1.404 | -0.457 | 0.84 | LOW |
| 5 | 1m_velocity | -4.415 | 0.174 | 0.80 | LOW |

**Candidate exit rule:**
```
if 5m_dmi_diff > 11.03 AND 15s_z_se < -1.00 AND 1m_dir_vol < -0.905:
    return 'real_cluster_0'
```

#### Cluster 1  —  N=282

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 1h_bar_range | 469.532 | 284.348 | 0.88 | HIGH |
| 2 | 1h_dir_vol | -1.236 | -0.092 | 0.83 | LOW |
| 3 | 1h_velocity | -45.997 | -1.504 | 0.81 | LOW |
| 4 | 15s_bar_range | 23.082 | 13.244 | 0.75 | HIGH |
| 5 | 15s_z_high | 1.298 | 0.478 | 0.73 | HIGH |

**Candidate exit rule:**
```
if 1h_bar_range > 469.53 AND 1h_dir_vol < -1.24 AND 1h_velocity < -46.00:
    return 'real_cluster_1'
```

#### Cluster 2  —  N=289

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 15m_z_se | -1.355 | -0.109 | 1.06 | LOW |
| 2 | 5m_dmi_diff | -14.893 | -1.380 | 0.96 | LOW |
| 3 | 1m_dmi_diff | -12.438 | -1.478 | 0.95 | LOW |
| 4 | 5m_z_se | -1.076 | -0.048 | 0.93 | LOW |
| 5 | 5m_z_low | -1.935 | -0.881 | 0.91 | LOW |

**Candidate exit rule:**
```
if 15m_z_se < -1.35 AND 5m_dmi_diff < -14.89 AND 1m_dmi_diff < -12.44:
    return 'real_cluster_2'
```

#### Cluster 3  —  N=530

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 1m_bar_range | 20.025 | 33.417 | 0.50 | LOW |
| 2 | 15s_bar_range | 7.089 | 13.244 | 0.47 | LOW |
| 3 | 5m_bar_range | 55.377 | 84.371 | 0.44 | LOW |
| 4 | 1h_bar_range | 194.372 | 284.348 | 0.43 | LOW |
| 5 | 15m_bar_range | 88.957 | 129.913 | 0.41 | LOW |

_Signal too weak for rule (top features |Δ/σ| < 0.5)._

### Bucket: STRONG

- Winners in bucket: 721
- PCA: 10 components (48% var)
- BIC: K2=29643, K3=29577, K4=29548  ->  **K=4** selected

#### Cluster 0  —  N=127

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 15s_dmi_diff | -13.642 | -1.028 | 0.99 | LOW |
| 2 | 1m_z_se | -1.035 | 0.010 | 0.93 | LOW |
| 3 | 1m_z_high | -0.113 | 0.639 | 0.70 | LOW |
| 4 | 1m_z_low | -1.502 | -0.723 | 0.65 | LOW |
| 5 | 5m_bar_range | 67.126 | 127.821 | 0.65 | LOW |

**Candidate exit rule:**
```
if 15s_dmi_diff < -13.64 AND 1m_z_se < -1.04 AND 1m_z_high < -0.113:
    return 'strong_cluster_0'
```

#### Cluster 1  —  N=177

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 5m_dmi_diff | -19.854 | -3.591 | 1.04 | LOW |
| 2 | 5m_z_se | -1.229 | -0.160 | 0.89 | LOW |
| 3 | 1m_dmi_diff | -14.042 | -2.996 | 0.87 | LOW |
| 4 | 15m_velocity | -35.894 | -3.828 | 0.85 | LOW |
| 5 | 5m_z_low | -2.017 | -1.028 | 0.82 | LOW |

**Candidate exit rule:**
```
if 5m_dmi_diff < -19.85 AND 5m_z_se < -1.23 AND 1m_dmi_diff < -14.04:
    return 'strong_cluster_1'
```

#### Cluster 2  —  N=193

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 5m_dmi_diff | 9.320 | -3.591 | 0.83 | HIGH |
| 2 | 15s_z_se | -0.902 | 0.109 | 0.80 | LOW |
| 3 | 1h_dir_vol | 1.170 | -0.211 | 0.78 | HIGH |
| 4 | 15s_z_high | -0.302 | 0.618 | 0.78 | LOW |
| 5 | 1m_dmi_diff | 6.776 | -2.996 | 0.77 | HIGH |

**Candidate exit rule:**
```
if 5m_dmi_diff > 9.32 AND 15s_z_se < -0.902 AND 1h_dir_vol > 1.17:
    return 'strong_cluster_2'
```

#### Cluster 3  —  N=224

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 15s_z_se | 0.961 | 0.109 | 0.67 | HIGH |
| 2 | 1m_dir_vol | 0.936 | 0.080 | 0.65 | HIGH |
| 3 | 5m_z_low | -0.259 | -1.028 | 0.64 | HIGH |
| 4 | 1m_velocity | 7.042 | 1.167 | 0.62 | HIGH |
| 5 | 15s_z_high | 1.343 | 0.618 | 0.61 | HIGH |

**Candidate exit rule:**
```
if 15s_z_se > 0.961 AND 1m_dir_vol > 0.936 AND 5m_z_low > -0.259:
    return 'strong_cluster_3'
```

### Bucket: DOMINANT

- Winners in bucket: 294
- PCA: 10 components (49% var)
- BIC: K2=12759, K3=12786, K4=12859  ->  **K=2** selected

#### Cluster 0  —  N=152

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 1m_z_se | 0.843 | 0.147 | 0.55 | HIGH |
| 2 | 5m_dmi_diff | -12.977 | -3.634 | 0.52 | LOW |
| 3 | 15m_z_se | -0.755 | -0.066 | 0.49 | LOW |
| 4 | 1h_dir_vol | -1.407 | -0.376 | 0.49 | LOW |
| 5 | 1h_velocity | -46.903 | -3.528 | 0.48 | LOW |

_Signal too weak for rule (top features |Δ/σ| < 0.5)._

#### Cluster 1  —  N=142

| rank | feature | cluster mean | global mean | |Δ/σ| | side |
|---:|---|---:|---:|---:|---|
| 1 | 1m_z_se | -0.599 | 0.147 | 0.58 | LOW |
| 2 | 5m_dmi_diff | 6.366 | -3.634 | 0.56 | HIGH |
| 3 | 15m_z_se | 0.673 | -0.066 | 0.53 | HIGH |
| 4 | 1h_dir_vol | 0.728 | -0.376 | 0.52 | HIGH |
| 5 | 1h_velocity | 42.901 | -3.528 | 0.51 | HIGH |

**Candidate exit rule:**
```
if 1m_z_se < -0.599 AND 5m_dmi_diff > 6.37 AND 15m_z_se > 0.673:
    return 'dominant_cluster_1'
```

---
_Generated by `tools/peak_bucket_lifecycle.py --tier RIDE_AGAINST`_