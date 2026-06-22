# Causal chaos-precursor — fittability vs the vol-persistence null

W=60 trailing, H=60 forward (5s bars); chaos = forward-vol >= IS p75 (1.3913).
Question: does causal fittability (1-R2) beat / add to trailing vol for predicting forward chaos?

### IS 2024-03  (n=316611, days=20, base chaos=25.0%)
- AUC trailing-vol (null, B) : 0.959
- unfit     : solo AUC 0.521 | vol+unfit AUC 0.959 (increment +0.000, coef -0.01)
- decel     : solo AUC 0.501 | vol+decel AUC 0.959 (increment -0.000, coef +0.00)
- resid_norm: solo AUC 0.564 | vol+resid_norm AUC 0.959 (increment -0.000, coef -0.03)
- ALL fittability + vol AUC : 0.959 (increment over vol -0.000)
- chaos rate by fittability quartile: Q1 fittable 27.5%, Q2 24.8%, Q3 23.8%, Q4 choppy 23.9%

### OOS 2025-03  (n=205190, days=18, base chaos=82.3%)
- AUC trailing-vol (null, B) : 0.922
- unfit     : solo AUC 0.528 | vol+unfit AUC 0.922 (increment +0.000, coef -0.06)
- decel     : solo AUC 0.504 | vol+decel AUC 0.922 (increment +0.000, coef -0.04)
- resid_norm: solo AUC 0.557 | vol+resid_norm AUC 0.922 (increment +0.000, coef -0.10)
- ALL fittability + vol AUC : 0.922 (increment over vol +0.000)
- chaos rate by fittability quartile: Q1 fittable 84.9%, Q2 81.8%, Q3 81.3%, Q4 choppy 81.2%

## Verdict (IS): FITTABILITY = vol in disguise (no independent precursor)
Note: AUC ~0.5 = no signal; the honest test is the A+B increment over vol-only, not AUC(A) alone (vol persistence inflates everything).