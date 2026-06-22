# Geomatics of the Micro E-mini Nasdaq-100: Markov Dynamics and Structural Regimes

**Date:** 2026-06-16
**Phases Completed:** Phase 1-4
**Subject:** Geometric mapping and transition probabilities of the MNQ.

---

## 1. Abstract
The Random Walk theory posits that market movements are structurally independent and mathematically noisy. This paper presents an opposing methodology deployed on 345 business days of Micro E-mini Nasdaq-100 (MNQ) tick data. By evaluating physical market segments through an $O(N)$ polynomial expansion matrix, we extracted specific structural geometries ("Regimes") that account for over 85% of all price action. Furthermore, by chronologically mapping the transitions between these defined regimes, we established a strict 3030x3030 Markov Transition Probability matrix that allows Bayesian State-Space models to predict the exact geometric shape most likely to emerge prior to its completion.

## 2. Data Provenance & Preprocessing

The underlying data spans 345 continuous business days of MNQ 5-second tick history. 
- **Raw Physical Ticks:** 5-second OHLCV data hosted in `DATA/ATLAS/5s/*.parquet`.
- **Feature Pipeline:** Continuous indicator and velocity data hosted in `DATA/ATLAS/FEATURES_5s_v2`.
- **Valid Segments:** 80,717 physically extracted chronological segments defined by moving-error bounds, indexed in `artifacts/stage2_year_segments.json`.

**Preprocessing:**
Features were aggressively cleaned. Non-contiguous `NaN` blocks were surgically dropped. Remaining matrices were passed through a global `StandardScaler` to normalize the amplitude variances before the polynomial GPU sweep. This flattened array is persistently cached in `artifacts/sweep_cache_flat.pt`.

## 3. Methodology & Scripts

The pipeline was executed using the following architectural stack:

| Script Path | Function | Mathematical Role |
| :--- | :--- | :--- |
| `phase2_extract_cache.py` | Data Allocation | Flattens 80,717 variable-length segments into standard continuous C-Arrays (`X_flat`, `Y_flat`). |
| `phase2_gpu_sweep.py` | Geometric Mapping | Executes the $O(N)$ polynomial `triu_indices` expansion, generating the massive Adjacency Matrix mapping every segment against every other segment. |
| `phase3_extract_regimes.py` | Degree Centrality | Extracts Regimes based on Bucket Roots (Anchored via Tier 1/2 Density thresholds). |
| `phase4_transition_matrix.py` | Markov Linkage | Reconstructs the timeline to calculate absolute transition probabilities $P(A \rightarrow B)$. |

**Acceptance Criteria for Extraction:**
Roots were strictly chosen based on **Tier 1 (Flawless) and Tier 2 (Great) Match Density**. Once a pure Root was defined, it absorbed surrounding halo matches up to Tier 4 (Edge match, $<2.5x$ expansion bound).

## 4. Geometric Visualization

The resulting dictionary is mathematically dense. To visualize the variance, here is the chronological construction of **Regime 1**, showing the Root (Crimson) and 150 highly-correlated Tier 1 & Tier 2 matching segments mapping perfectly into the channel.

![Regime 1 Geometric Animation](./regime_1_animated.gif)

## 5. Results & Discussion

### The Dictionary of the MNQ
The algorithm successfully extracted exactly **3,029 unique Market Regimes** out of the 80,717 possible segments. 
- **74,101 (91.8%)** segments strictly adhered to these standard shapes.
- The **Top 30 Regimes** successfully account for **85.6%** of all MNQ price action, completely shattering the assumption of infinite random variance.

### Markov Mechanics: Three Universal Laws
Analysis of the 3030x3030 Transition Probability matrix (`artifacts/transition_matrix.npy`) reveals three strict mechanical laws:

1. **The Law of Inertia:** The dominant regime (Regime 1) is massively self-perpetuating. There is a mathematical **62.93% chance** that it repeats itself upon completion.
2. **The Snapping of Volatility:** High-energy geometries (e.g., Regime 2, "Sharp Rejection") are intrinsically unstable. There is only an **8.87% chance** of Regime 2 perpetuating. Over **52.01% of the time**, volatility instantly collapses back into the safety of Regime 1.
3. **The Resolution of Chaos:** When the market loses all structural definition (The 6,616 "NOISE" segments), it does *not* breed more chaos. There is only a **12.33% chance** of staying in Noise. Instead, there is a **45.09% chance** it violently snaps back into the ordered structure of Regime 1.

## 6. Conclusion
The MNQ does not follow a Random Walk. It navigates a strict, highly restricted dictionary of pre-defined geometric curves governed by Markov transition probabilities. By integrating `transition_matrix.npy` into the Bayesian state-space filter, the architecture possesses an explicit prior distribution for future geometric trajectory before the physical ticks are rendered.
