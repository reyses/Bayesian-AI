---
name: Multiscale chop → higher-TF velocity/range regularity
description: User-identified pattern from 2026-05-03 regime-stratified TF sweep. Chop at lower TF reliably WEAKENS signed velocity at higher TF for directional regimes (UP/DOWN). Range behavior is regime-asymmetric.
type: feedback
date: 2026-05-03
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## The pattern

User: "a lot of variation in one TF means that the higher TF was weaker velocity and more bar range".

**Why:** The user's reading of the multiscale-character table from `v2_features_regime_stratified_tf_sweep.py`. The pattern is real but regime-asymmetric.

## How to apply

When characterizing a TF's signal in the context of higher TFs, do NOT
assume "chop at low TF = trend at high TF". The empirical relationship is:

- **Chop at lower TF reliably WEAKENS signed velocity at the higher TF for directional regimes** (UP and DOWN). The intraday noise dilutes the directional component into the macro window.
- **Range expansion under chop is regime-dependent**:
  - Non-directional underlying (UP_*, FLAT_*) → chop EXPANDS range (intraday two-way action mechanically widens bars)
  - Directional clean sell (DOWN_SMOOTH) → SMOOTH version has WIDER range than CHOPPY version. Clean sells expand range further than chopped sells.

## Empirical numbers (1h cohort, IS-only, full year)

| Regime pair | 1h vel SMOOTH → CHOPPY | 1h range SMOOTH → CHOPPY |
|---|---|---|
| UP | 16.9 → 14.7 (−13%) | 70.7 → 80.1 (+13%) |
| DOWN | 20.4 → 12.1 (**−41%**) | 89.6 → 80.2 (−10%) |
| FLAT | 9.6 → 10.6 (+10%) | 59.5 → 81.0 (**+36%**) |

The strongest velocity-weakening effect: DOWN_SMOOTH→DOWN_CHOPPY at 1h (−41%). The strongest range expansion: FLAT_SMOOTH→FLAT_CHOPPY (+36%).

## Implications for signal design

- A high lower-TF chop signal is a hint that the higher-TF velocity will be diluted (especially for sells). Don't expect macro-trend follow-through in days that started choppy at the bar level.
- Range at the higher TF is NOT a reliable indicator of underlying regime: large 1h range can be either DOWN_SMOOTH (clean sell) or FLAT_CHOPPY (chop with no direction). Need to combine with velocity sign or direction_axis.
- The regime-conditional composite framework already established (target sign depends on modifier quantile) gets supporting evidence here: bar_range alone gives correlation with forward return ranging from -0.157 to +0.235 across regimes (range 0.39).

## Connection to other findings

- The 9-layer EDA stack already established that contextualization is real and that compositions need to be conditional. This pattern adds a specific multiscale regularity that informs which composite to choose.
- The chord finder identified 4h-TF chord cells with 100% regime purity. Combined with this regularity, it suggests a layered signal:
  1. Identify daily regime via 4h chord cells
  2. Within each regime, use the conditional rules (modifier quantile flips target sign)
  3. Calibrate trade horizon to the velocity-weakening pattern (don't over-extend into 1h hold when the lower TF is choppy in DOWN regime)

## Source

- Tool: `tools/v2_features_regime_stratified_tf_sweep.py`
- Output: `reports/findings/v2_features_regime_tf/multiscale_character.csv`
- Commit: `314e1072` (2026-05-03)
- Journal: `docs/daily/2026-05-03.md` (regime-stratified TF sweep section)
