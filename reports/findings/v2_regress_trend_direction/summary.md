**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# V2 Trend Direction Regression — 2026-05-02 06:09 UTC

**Base TF:** `5m`
**Target:** `net_move (signed)`
**Ridge alpha:** 0.0
**Split:** day-level IS/VAL/OOS from `regime_labels_2d.csv`

## Test scores

- R²: 0.0323
- Bar-level direction accuracy: 63.2% (baseline 51.3%, lift +11.9%)
- Day-level direction accuracy: 74.6% on 71 days (baseline 52.1%, lift +22.5%)

## Per-regime accuracy (bar-level, test split)

  regime_2d  n_bars  n_days  accuracy_bar   mean_pred  mean_actual
DOWN_CHOPPY    1332       5      0.439189   27.844570  -375.103604
DOWN_SMOOTH    1782      10      0.726712 -130.492067  -456.250842
FLAT_CHOPPY    7740      33      0.581395   51.266099     4.453876
FLAT_SMOOTH    1350       7      0.422963   92.536150   -54.314444
  UP_CHOPPY    1104       4      0.745471  137.233935   318.562500
  UP_SMOOTH    2592      12      0.876157  193.137364   375.172454

