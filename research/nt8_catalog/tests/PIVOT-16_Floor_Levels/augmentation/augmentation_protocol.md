# ML Feature Augmentation Protocol: PIVOT-16_Floor_Levels

## 1. Goal
To systematically augment the core PIVOT-16_Floor_Levels edge using Binary Logistic Regression across our 416D Feature Space (F-Space). We aim to characterize the statistical response (Hit vs Miss) and discover hidden non-linear relationships.

## 2. Hypothesis & Assumptions (Temporal Mapping)

We segment the F-Space evolution into 4 distinct phases for PIVOT-16_Floor_Levels:

### Phase 1: Pre-Entry (Leading into the Setup)
* **Objective:** What structural conditions must exist in the 1D, 4H, and 1H timeframes before this setup even triggers?
* **Assumptions:** We hypothesize that macro trend alignment (e.g., L3 Hurst Exponents and L4 NMP Regimes) heavily dictates the win rate of PIVOT-16_Floor_Levels. A scramble or misalignment here is likely a filter condition.

### Phase 2: During Trade (Holding Period)
* **Objective:** How does the F-Space mutate while we are actively holding the position?
* **Assumptions:** We expect micro-timeframe (5s, 1m) volatility and orderflow features (L2/L3) to spike. Rapid decay of structural momentum in this phase leads to stop-outs.

### Phase 3: Nearing Exit (Leading to Exit Conditions)
* **Objective:** Which F-Space features predictably precede our mechanical exit conditions?
* **Assumptions:** Local exhaustion features (L1 `dist_to_sma`, L5 distributions) should show extreme Z-scores right before the baseline exit logic triggers, allowing us to build an early-warning ML exit model.

### Phase 4: Post-Exit (Aftermath)
* **Objective:** Did we exit optimally, or did the market continue without us?
* **Assumptions:** Post-exit structural integrity will indicate if PIVOT-16_Floor_Levels is leaving money on the table in high-trend regimes.
