# Regime discovery from corrected trades — 2026-04-17_234220

Dataset: 9,882 corrected iso trades.
Baseline counter share: 48.7% (direction flip would've won this often).

Day-stratified split: 222 train / 55 valid days.
Tree max_depth=4, min_samples_leaf=100.

**Train acc: 0.756**  |  **Valid acc: 0.725**  |  Majority baseline: 0.513

## Leaf table (sorted by counter share on valid)

| Leaf | N_train | N_valid | Counter%_train | Counter%_valid | D$_train | D$_valid | Rule | Flag |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 16 | 343 | 89 | 58.3% | 67.4% | $+123.49 | $+114.24 | `1m_reversion_prob > 0.8663 AND 15s_variance_ratio > 0.2828 AND 1m_variance_ratio <= 0.7252 AND 1m_dmi_diff <= -17.7353` | CAND |
| 17 | 4,198 | 993 | 70.7% | 66.4% | $+100.18 | $+96.15 | `1m_reversion_prob > 0.8663 AND 15s_variance_ratio > 0.2828 AND 1m_variance_ratio <= 0.7252 AND 1m_dmi_diff > -17.7353` | CAND |
| 7 | 117 | 29 | 79.5% | 65.5% | $+137.97 | $+175.26 | `1m_reversion_prob <= 0.8663 AND 1m_variance_ratio > 0.9987` |  |
| 10 | 123 | 29 | 67.5% | 65.5% | $+92.84 | $+56.57 | `1m_reversion_prob > 0.8663 AND 15s_variance_ratio <= 0.2828 AND 1m_variance_ratio <= 0.2067` |  |
| 19 | 125 | 27 | 36.8% | 55.6% | $+76.55 | $+85.31 | `1m_reversion_prob > 0.8663 AND 15s_variance_ratio > 0.2828 AND 1m_variance_ratio > 0.7252 AND 5m_hurst <= 0.6581` |  |
| 13 | 460 | 128 | 44.8% | 50.0% | $+80.38 | $+78.23 | `1m_reversion_prob > 0.8663 AND 15s_variance_ratio <= 0.2828 AND 1m_variance_ratio > 0.2067 AND 15m_hurst > 0.6101` |  |
| 20 | 163 | 43 | 58.9% | 48.8% | $+104.71 | $+83.66 | `1m_reversion_prob > 0.8663 AND 15s_variance_ratio > 0.2828 AND 1m_variance_ratio > 0.7252 AND 5m_hurst > 0.6581` |  |
| 12 | 138 | 29 | 61.6% | 48.3% | $+81.06 | $+88.24 | `1m_reversion_prob > 0.8663 AND 15s_variance_ratio <= 0.2828 AND 1m_variance_ratio > 0.2067 AND 15m_hurst <= 0.6101` |  |
| 3 | 100 | 22 | 25.0% | 18.2% | $+62.78 | $+47.25 | `1m_reversion_prob <= 0.8663 AND 1m_variance_ratio <= 0.9987 AND time_of_day <= 0.0212` |  |
| 6 | 1,770 | 449 | 3.6% | 6.0% | $+76.03 | $+72.43 | `1m_reversion_prob <= 0.8663 AND 1m_variance_ratio <= 0.9987 AND time_of_day > 0.0212 AND 15s_dmi_gap > 13.1500` |  |
| 5 | 401 | 106 | 9.0% | 4.7% | $+86.06 | $+80.81 | `1m_reversion_prob <= 0.8663 AND 1m_variance_ratio <= 0.9987 AND time_of_day > 0.0212 AND 15s_dmi_gap <= 13.1500` |  |

## Verdict

**2 candidate regime(s) found.** Listed above with rules. These pass the bar of >58% counter on both train and valid, >$10 mean oracle delta on valid, and >30 valid samples. Worth porting as tier rules in training_iso/nightmare_iso.py for end-to-end validation.
