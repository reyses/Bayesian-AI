# Triplet OOS-Surviving Rules — Dispatch Table

Generated 2026-05-03 from `reports/findings/v2_features_triplet_oos/surviving_cells.csv`.

**31 surviving triplet cells** out of 120 top-K-per-regime IS candidates (25.8% survival rate).

## Survivor distribution

| Regime | Count |
|---|---:|
| FLAT_SMOOTH | 12 |
| DOWN_SMOOTH | 9 |
| FLAT_CHOPPY | 5 |
| DOWN_CHOPPY | 3 |
| UP_SMOOTH | 2 |
| UP_CHOPPY | 0 |

| OOS direction | Count |
|---|---:|
| SHORT (lift < 0) | 19 |
| LONG (lift > 0) | 12 |

**Asymmetry**: most survivors are in non-trending regimes (FLAT/DOWN). UP_CHOPPY had 0 survivors — its IS top cells all collapsed on OOS. UP_SMOOTH only kept 2 — meaning bullish-trend triplet signals are largely overfit. The SHORT skew suggests bearish setups generalize better than bullish ones in this dataset (likely tied to market regime over Jan-Sep 2025 IS span).

## Rule schema

Each rule is a tuple:
```
(regime_2d, anchor_concept, anchor_tf, x_concept, x_tf, y_concept, y_tf,
 q_anchor, q_x, q_y) -> direction (LONG/SHORT), expected_lift_ticks, confidence_ci
```

Direction = sign of OOS mean_fwd. Hold horizon = 12 base bars (1h at 5m base).

## Top survivors by |OOS lift|

| Regime | Anchor | X | Y | Cell | OOS n | OOS lift | OOS WR | Direction |
|---|---|---|---|---|---:|---:|---:|---|
| DOWN_SMOOTH | vol_mean_w_1h | swing_noise_w_1m | z_se_w_15m | (0,2,1) | 37 | −46.9 | 27% | SHORT |
| DOWN_SMOOTH | vol_mean_w_1h | swing_noise_w_1m | reversion_prob_w_15m | (0,2,2) | 37 | −46.9 | 27% | SHORT |
| UP_SMOOTH | hurst_w_1h | bar_range_5m | z_se_w_15m | (2,2,2) | 145 | +20.5 | 70% | LONG |
| UP_SMOOTH | hurst_w_1h | swing_noise_w_1m | z_se_w_15m | (2,2,2) | 155 | +19.6 | 66% | LONG |

Full table: `surviving_cells.csv` (31 rows).

## Common structural pattern in survivors

All 31 survivors share this composition:
- **ANCHOR**: 1h-window structure feature — `vol_mean_w_1h`, `hurst_w_1h`, `bar_range_1h`
- **X (amplitude)**: short-TF microstructure — `swing_noise_w_1m`, `bar_range_5m`, `price_velocity_w_5m`
- **Y (context)**: 15m reversion-context — `z_se_w_15m`, `reversion_prob_w_15m`, `price_velocity_1b_15m`

**No surviving cell uses pure velocity/sigma combinations**. The reversion-context companion is what differentiates real signal from selection-bias noise.

## Operational deployment notes

1. **Direction is set by sign of OOS lift, not IS lift.** A cell whose IS lift was +50 but OOS lift is +20 is a LONG rule. A cell whose IS lift was −60 and OOS lift is −45 is a SHORT rule.
2. **Quantile boundaries must be IS-derived per regime.** The validation tool computed regime-local quantile edges from IS data; live deployment must use the same IS edges, not recompute on rolling data (otherwise you are re-fitting).
3. **Multiple cells may match the same bar.** If they all point same direction, fire. If they conflict, skip.
4. **Hold horizon ≈ 1h (12 base bars at 5m).** Match exit logic to that horizon.
5. **OOS lift is the EXPECTED PnL, not realized.** Bootstrap CI is reported in `surviving_cells.csv` columns `oos_ci_lo` / `oos_ci_hi`. Cells whose CI straddles 0 are weak; prefer cells where both CI bounds have the same sign.

## What to do next

1. **Tighten the rule set**: filter to cells where both OOS CI bounds have same sign. This removes the borderline survivors.
2. **Trade-replay simulation** on OOS: apply the rules bar-by-bar and compute realized $/day with proper position sizing and exit handling.
3. **Translate to NinjaTrader**: ~31 rules in a regime-gated dispatch table. Implementation = ~50 lines of NT8 lookup + entry trigger.
4. **Do NOT extend to 4-feature combinations** — Layer C1 OOS collapse rate (75%) implies 4-feature would be even worse.
