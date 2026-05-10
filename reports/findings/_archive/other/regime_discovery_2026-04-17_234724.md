# Regime discovery from corrected trades — 2026-04-17_234724

Dataset: 9,882 corrected iso trades.
Baseline counter share: 48.7% (direction flip would've won this often).

Day-stratified split: 222 train / 55 valid days.
Tree max_depth=4, min_samples_leaf=100.

**Train acc: 0.516**  |  **Valid acc: 0.465**  |  Majority baseline: 0.513

## Leaf table (sorted by counter share on valid)

| Leaf | N_train | N_valid | Counter%_train | Counter%_valid | D$_train | D$_valid | Rule | Flag |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 11 | 196 | 39 | 39.3% | 53.8% | $+98.02 | $+75.69 | `15s_dir_vol > -3.3089 AND 15s_variance_ratio <= 1.6953 AND 1h_z_low > 1.3014` |  |
| 4 | 165 | 50 | 39.4% | 52.0% | $+51.60 | $+67.09 | `15s_dir_vol <= -3.3089 AND 5m_wick_ratio <= 0.6394 AND 15s_acceleration > -3.6250` |  |
| 12 | 147 | 36 | 36.1% | 47.2% | $+91.91 | $+104.25 | `15s_dir_vol > -3.3089 AND 15s_variance_ratio > 1.6953` |  |
| 10 | 6,786 | 1,674 | 50.2% | 46.4% | $+93.13 | $+87.35 | `15s_dir_vol > -3.3089 AND 15s_variance_ratio <= 1.6953 AND 1h_z_low <= 1.3014 AND 15s_acceleration > -13.8750` |  |
| 3 | 104 | 27 | 21.2% | 40.7% | $+63.14 | $+84.74 | `15s_dir_vol <= -3.3089 AND 5m_wick_ratio <= 0.6394 AND 15s_acceleration <= -3.6250` |  |
| 9 | 114 | 27 | 65.8% | 40.7% | $+236.14 | $+219.30 | `15s_dir_vol > -3.3089 AND 15s_variance_ratio <= 1.6953 AND 1h_z_low <= 1.3014 AND 15s_acceleration <= -13.8750` |  |
| 5 | 184 | 47 | 51.1% | 38.3% | $+76.12 | $+53.91 | `15s_dir_vol <= -3.3089 AND 5m_wick_ratio > 0.6394` |  |

## Verdict

**No candidate regime survives.** No leaf achieves >58% counter on both train and valid with >$10 delta. Direction at entry is confirmed random on this feature set. Stop chasing flip tiers; pivot effort to exits or entry filtering.
