# LinReg slope filter — IS/OOS robustness check

Generated: 2026-04-29 23:20

## Method

For each tier:
1. Pick best slope_skip_threshold on IS-only data
2. Apply that EXACT threshold to OOS data
3. Compare per-trade improvement IS vs OOS

## Robustness verdict

- 0/1 tiers' filters GENERALIZE to OOS (filtered PnL > baseline)
- **Filter is overfit on IS for some tiers.** Be cautious — only ship tiers where OOS improvement is positive.

## Detailed table

```
     tier  is_baseline_n  is_baseline_pnl  is_baseline_per_trade  T_picked_on_is  is_kept_n  is_filtered_pnl  is_filtered_per_trade  is_improvement  is_improvement_per_kept_trade  oos_baseline_n  oos_baseline_pnl  oos_baseline_per_trade  oos_kept_n  oos_filtered_pnl  oos_filtered_per_trade  oos_improvement  oos_improvement_per_kept_trade  generalizes
KILL_SHOT             70           1876.5              26.807143             5.0         70           1876.5              26.807143             0.0                            0.0              36             695.0               19.305556          36             695.0               19.305556              0.0                             0.0        False
```

## Interpretation

- `is_improvement_per_kept_trade` — per-trade gain from filter, on IS
- `oos_improvement_per_kept_trade` — per-trade gain from same threshold, on OOS
- If OOS per-trade gain is similar magnitude to IS gain → filter is real
- If OOS per-trade gain is zero or negative → filter is IS-overfit
- `generalizes` flag = filter improves OOS PnL (modest test)
