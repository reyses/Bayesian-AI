import os
import glob

tests_dir = r"c:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
folders = [f.path for f in os.scandir(tests_dir) if f.is_dir() and not f.name.startswith('.')]

protocol_template = """# ML Feature Augmentation Protocol: {strategy_name}

## 1. Goal
To systematically augment the core {strategy_name} edge using Binary Logistic Regression across our 416D Feature Space (F-Space). We aim to characterize the statistical response (Hit vs Miss) and discover hidden non-linear relationships.

## 2. Hypothesis & Assumptions (Temporal Mapping)

We segment the F-Space evolution into 4 distinct phases for {strategy_name}:

### Phase 1: Pre-Entry (Leading into the Setup)
* **Objective:** What structural conditions must exist in the 1D, 4H, and 1H timeframes before this setup even triggers?
* **Assumptions:** We hypothesize that macro trend alignment (e.g., L3 Hurst Exponents and L4 NMP Regimes) heavily dictates the win rate of {strategy_name}. A scramble or misalignment here is likely a filter condition.

### Phase 2: During Trade (Holding Period)
* **Objective:** How does the F-Space mutate while we are actively holding the position?
* **Assumptions:** We expect micro-timeframe (5s, 1m) volatility and orderflow features (L2/L3) to spike. Rapid decay of structural momentum in this phase leads to stop-outs.

### Phase 3: Nearing Exit (Leading to Exit Conditions)
* **Objective:** Which F-Space features predictably precede our mechanical exit conditions?
* **Assumptions:** Local exhaustion features (L1 `dist_to_sma`, L5 distributions) should show extreme Z-scores right before the baseline exit logic triggers, allowing us to build an early-warning ML exit model.

### Phase 4: Post-Exit (Aftermath)
* **Objective:** Did we exit optimally, or did the market continue without us?
* **Assumptions:** Post-exit structural integrity will indicate if {strategy_name} is leaving money on the table in high-trend regimes.
"""

report_template = """# F-Space DOE Statistical Report: {strategy_name}

> **Status:** Pending Data Pipeline Execution

## 1. Stepwise Elimination Impact

### Pre-Stepwise (All 416 Features)
* **AIC:** [Pending]
* **BIC:** [Pending]
* **Pseudo R-Squared:** [Pending]
* **Baseline Win Rate:** [Pending]

### Post-Stepwise (Vital Few Features)
* **AIC:** [Pending]
* **BIC:** [Pending]
* **Pseudo R-Squared:** [Pending]
* **Augmented Predictive Win Rate:** [Pending]

## 2. Standardized Effects (Pareto)
*(Visuals pending script generation)*
* **Top Linear Effects:** [Pending]
* **Top Quadratic Effects:** [Pending]
* **Top Cubic Effects:** [Pending]

## 3. Interaction Plot Highlights
*(Visuals pending script generation)*
* **Interaction 1:** [Pending]
* **Interaction 2:** [Pending]
"""

followup_template = """# Follow-up Proposals: {strategy_name}

## 1. CNN Handoff
If the baseline DOE logistic regression fails to capture the complexity of the Phase 2 (During Trade) evolution, we propose routing the 416D sub-grid for {strategy_name} into a 2D CNN (treating the TF hierarchy as spatial channels). 

## 2. Feature Swaps
If our Phase 1 macro assumptions fail to improve the Win Rate, we propose substituting L4 NMP Regimes with raw Orderflow Imbalance aggregations (if available).

## 3. Execution Integration
Proposals for deploying the augmented {strategy_name} model back into NinjaTrader 8 as an optimized bloodhound/strategy logic.
"""

for folder in folders:
    strategy_name = os.path.basename(folder)
    aug_dir = os.path.join(folder, "augmentation")
    
    if not os.path.exists(aug_dir):
        os.makedirs(aug_dir)
        
    with open(os.path.join(aug_dir, "augmentation_protocol.md"), "w", encoding='utf-8') as f:
        f.write(protocol_template.format(strategy_name=strategy_name))
        
    with open(os.path.join(aug_dir, "fspace_doe_report.md"), "w", encoding='utf-8') as f:
        f.write(report_template.format(strategy_name=strategy_name))
        
    with open(os.path.join(aug_dir, "followup_proposals.md"), "w", encoding='utf-8') as f:
        f.write(followup_template.format(strategy_name=strategy_name))
        
print("Successfully generated augmentation dossiers across 18 strategies.")
