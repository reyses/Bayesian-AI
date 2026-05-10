# Small-shape absorption (family-priority, 15m level only)

_Generated 2026-05-10T01:25:28.598135_

Split: IS
Min shape n threshold: 30
Large shapes (kept as primitive labels): ['EXPONENTIAL_DOWN', 'EXPONENTIAL_UP', 'FLATLINE', 'LINEAR_DOWN', 'LINEAR_UP', 'LOGARITHMIC_DOWN', 'LOGARITHMIC_UP', 'NOISE']
Small shapes (absorbed): ['STEP_DOWN']

## Absorption rules

No chord or feature-structure used. Each small shape is absorbed into the
first member of its family-preference list that is itself a large shape.
Family logic:

- **UP-direction families**: BACK_SKEWED_UP -> EXPONENTIAL_UP (accelerating
  curve neighbor); FRONT_SKEWED_UP -> LOGARITHMIC_UP (exhausting curve);
  STEP_UP -> LINEAR_UP if STEP_UP itself too small; etc.
- **DOWN-direction families**: mirror of UP.
- **Reversal families**: SYMMETRIC_V <-> ROUNDED_U (both pivot patterns).
- **Oscillators** (SINE_WAVE, DAMPED, EXPAND): -> NOISE (no clean primitive home).

## Per-shape absorption mapping

```
small_shape  n_small                              preference_list absorbed_to
  STEP_DOWN        1 STEP_DOWN -> LINEAR_DOWN -> EXPONENTIAL_DOWN LINEAR_DOWN
```

## Resulting absorbed shape distribution

```
absorbed_to
FLATLINE            1286
LINEAR_UP            241
LINEAR_DOWN          218
NOISE                162
EXPONENTIAL_UP       142
EXPONENTIAL_DOWN     142
LOGARITHMIC_UP        84
LOGARITHMIC_DOWN      54
```

## Per-original-shape destination crosstab

```
absorbed_to       EXPONENTIAL_DOWN  EXPONENTIAL_UP  FLATLINE  LINEAR_DOWN  LINEAR_UP  LOGARITHMIC_DOWN  LOGARITHMIC_UP  NOISE
original_shape                                                                                                               
EXPONENTIAL_DOWN               142               0         0            0          0                 0               0      0
EXPONENTIAL_UP                   0             142         0            0          0                 0               0      0
FLATLINE                         0               0      1286            0          0                 0               0      0
LINEAR_DOWN                      0               0         0          217          0                 0               0      0
LINEAR_UP                        0               0         0            0        241                 0               0      0
LOGARITHMIC_DOWN                 0               0         0            0          0                54               0      0
LOGARITHMIC_UP                   0               0         0            0          0                 0              84      0
NOISE                            0               0         0            0          0                 0               0    162
STEP_DOWN                        0               0         0            1          0                 0               0      0
```

## Implication

After this absorption, only the large shapes remain as primitive labels.
Small-shape phrases inherit the prior of the nearest large-shape neighbor
in the family tree (no chord/feature-structure used). The Bayesian-table
cell count reduces accordingly: each small shape no longer needs its own
cell. Per-shape HDBSCAN now runs on this consolidated label set.
