# GA-Kalman REGRET v2 — magnitude of wrongness + over-wait (entry-lag) test

Lookback 600s, forward 900s. Tests whether '51% wrong' was direction or LATE entry.

## OOS  (N=5987)
### (A) MAGNITUDE of direction-rightness (not just sign)
- effectively TIED (|chosen−flip MFE| ≤ 3pt): **6%** of trades
- chosen CLEARLY bigger (>+3pt): 46%   |   flip clearly bigger (wrong): 47%
- mean (chosen−flip) MFE: **+1.7pt**  (median -0.5pt)
### (B) OVER-WAIT / entry-lag test
- price ALREADY moving our way at entry (pre_move>0): **73%** of entries
- mean pre-entry move (chosen dir, 10m lookback): **+26.2pt** (median +22.0pt)
- mean pre-entry MFE (size of move before entry): 49.3pt vs forward chosen MFE 39.7pt
- → if pre-move ≫0 and chosen≈flip forward, the entry is RIGHT-direction but LATE (over-waited), not directionless.

## IS (ref)  (N=711)
### (A) MAGNITUDE of direction-rightness (not just sign)
- effectively TIED (|chosen−flip MFE| ≤ 3pt): **13%** of trades
- chosen CLEARLY bigger (>+3pt): 44%   |   flip clearly bigger (wrong): 44%
- mean (chosen−flip) MFE: **+2.8pt**  (median -0.5pt)
### (B) OVER-WAIT / entry-lag test
- price ALREADY moving our way at entry (pre_move>0): **85%** of entries
- mean pre-entry move (chosen dir, 10m lookback): **+22.3pt** (median +16.0pt)
- mean pre-entry MFE (size of move before entry): 32.8pt vs forward chosen MFE 21.4pt
- → if pre-move ≫0 and chosen≈flip forward, the entry is RIGHT-direction but LATE (over-waited), not directionless.
