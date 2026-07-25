# Tune/holdout validation (owner protocol)
25 days -> tune ['2025_01_21', '2025_03_13', '2025_04_16', '2025_04_25', '2025_11_27', '2026_03_05'] (29 eps) / test 127 eps on 19 days. Seed 42.

**Tuned knobs**: {'Z_SICK': 2.0, 'Z_FADE': 0.5, 'SICK_ARM': 3, 'LAG': 2} (tune objective +9.21 pts/ep)

## HOLDOUT (knobs locked)
- exit-at-armed vs never-bail: **-1.01 pts/ep**, 95% day-block CI [-10.06, +7.81] — does NOT generalize (CI includes 0)
- fired on 62/127 episodes; timing vs ground truth: median lead +8 min (negative = fired AFTER the peak), IQR [-2, +18]

NOTE: still within the 25 burned days; lockbox-conveyor fresh days remain the final confirmation tier.
