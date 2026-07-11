# AG Execution Report: Phase 5 Policy Evaluation

## Overview
Phase 5 focused on building the 6-tier "telescope" feature vectors (PhE, PhXit, PhPost) for the 4 priority dossiers (`ATR-09`, `FIB-17`, `VA-13`, `ORDERFLOW-14`) and evaluating the Doc-027 3-way policy (ACT / SKIP / INVERT) out-of-sample.

## 1. F-Space Extraction Completion
- **Claim:** F-Space extraction successfully ran and constructed the 6-tier feature matrices (L0-L5) across the events, resolving path navigation errors and bounds check issues. 
- **Artifact Path:** [ag_phase5_fspace.py](file:///c:/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/nt8_catalog/tools/ag_phase5_fspace.py)
- **Raw Check Output:**
```text
========================================
Extraction for ATR-09_Statistical_Fade:
PhE Shape: (799, 52)
PhXit Shape: (799, 52)
PhPost Shape: (799, 52)
Saved 799 samples for ATR-09_Statistical_Fade
========================================
Extraction for FIB-17_Confluence:
PhE Shape: (74, 52)
PhXit Shape: (74, 52)
PhPost Shape: (74, 52)
Saved 74 samples for FIB-17_Confluence
========================================
Extraction for VA-13_Rotation:
PhE Shape: (132, 52)
PhXit Shape: (132, 52)
PhPost Shape: (132, 52)
Saved 132 samples for VA-13_Rotation
========================================
Extraction for ORDERFLOW-14:
PhE Shape: (7367, 49)
PhXit Shape: (7367, 49)
PhPost Shape: (7367, 49)
Saved 7367 samples for ORDERFLOW-14
```

## 2. Doc-027 Policy Validation
- **Claim:** The 3-way out-of-sample policy evaluation (ACT/SKIP/INVERT) completed successfully. Only `ORDERFLOW-14` yielded a valid out-of-sample edge. The remaining technical-indicator dossiers either lacked sufficient data or produced an invalid edge.
- **Artifact Path:** [ag_phase5_final.py](file:///c:/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/nt8_catalog/tools/ag_phase5_final.py)
- **Report Path:** [AG_cat_00_PHASE5.md](file:///c:/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/nt8_catalog/reports/AG_cat_00_PHASE5.md)
- **Raw Check Output:**
```text
========================================
Processing ATR-09...
Lengths: X=799, Y=799, Years=799
Training on 2024, Testing on 2025
Selected 1 features out of 156

2025 Evaluation:
ACT Branch  (P >= 0.169): N=100, WR=0.02, EV=-10.70 pts (CI_lo: -15.61), Mode=-12.0 pts | Valid=False
INV Branch  (P <= 0.046): N=0, WR=0.00, EV=0.00 pts (CI_lo: 0.00), Mode=0 pts | Valid=False

========================================
Processing FIB-17...
Lengths: X=74, Y=74, Years=74
Training on 2024, Testing on 2025
L1 selection dropped all features. Using top 5 by correlation.
Selected 5 features out of 156

2025 Evaluation:
ACT Branch  (P >= 0.000): N=9, WR=0.00, EV=-13.00 pts (CI_lo: -15.08), Mode=-12.0 pts | Valid=False
INV Branch  (P <= 0.000): N=5, WR=0.80, EV=-14.35 pts (CI_lo: -66.70), Mode=12.0 pts | Valid=False

========================================
Processing VA-13...
Lengths: X=132, Y=132, Years=132
Training on 2024, Testing on 2025
L1 selection dropped all features. Using top 5 by correlation.
Selected 5 features out of 156

2025 Evaluation:
ACT Branch  (P >= 0.172): N=22, WR=0.00, EV=-7.44 pts (CI_lo: -9.86), Mode=-10.0 pts | Valid=False
INV Branch  (P <= 0.009): N=1, WR=0.00, EV=-4.75 pts (CI_lo: -4.75), Mode=-5.0 pts | Valid=False

========================================
Processing ORDERFLOW-14...
Lengths: X=7367, Y=7367, Years=7367
Training on 2025, Testing on 2026
Selected 39 features out of 147

2025 Evaluation:
ACT Branch  (P >= 0.559): N=192, WR=0.63, EV=1.74 pts (CI_lo: 0.62), Mode=4.0 pts | Valid=True
INV Branch  (P <= 0.459): N=177, WR=0.34, EV=-2.23 pts (CI_lo: -3.37), Mode=-6.0 pts | Valid=False

Phase 5 Evaluation Complete. Report saved to reports/AG_cat_00_PHASE5.md
```
