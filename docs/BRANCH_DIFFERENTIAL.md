# Branch Differential: main vs pre-snowflake
Generated 2026-02-23

## Summary

This diagnostic compares the current `main` branch (Snowflake architecture) against the `pre-snowflake` branch (Legacy architecture). The analysis confirms that `main` represents the evolved, optimized system, while `pre-snowflake` contains older, unoptimized, or experimental code paths.

*   **Files Identical:** `config/oracle_config.py`, `config/settings.py`, `config/workflow_manifest.json`
*   **Files with additive differences (easy merge):** 0 (Most differences are conflicting architectural choices)
*   **Files with conflicting logic (needs decision):** 16
*   **Dominant Strategy:** Keep `main`. The `main` branch contains critical performance optimizations (bisect, direct attribute access), the "Snowflake" Long/Short routing architecture, and refined logic (Time Exhaustion vs Physics Decay).

## Per-File Analysis

### training/orchestrator.py
| Feature | main | pre-SF | Conflict? |
| :--- | :--- | :--- | :--- |
| **Snowflake Routing** | **Yes** (Loads `_long` / `_short` libraries) | No (Unified library only) | **Yes** — Major architectural change |
| **Fractal DNA Tree** | **Yes** (Loads `fractal_dna_tree.pkl`) | No | **Yes** — Feature missing in pre-SF |
| **Dashboard Launch** | Optimized check | Standard check | No — Minor logic refinement |

### training/fractal_clustering.py
| Feature | main | pre-SF | Conflict? |
| :--- | :--- | :--- | :--- |
| **Optimization** | **Fast Path** (`try-except` attribute access) | Slow Path (`getattr` only) | **Yes** — Performance critical |
| **Directional Logic** | **Yes** (`direction` field in template) | No | **Yes** — Snowflake requirement |
| **Semantic Naming** | No | **Yes** (`generate_semantic_name`) | **Yes** — Feature dropped in main? |

### training/doe_parameter_generator.py
| Feature | main | pre-SF | Conflict? |
| :--- | :--- | :--- | :--- |
| **Strategy** | **Optuna TPE** (Standard, Efficient) | Custom Multi-Stage (LHS, Mutation, Crossover) | **Yes** — Complete rewrite |
| **Complexity** | Low | High | **Yes** |

### core/quantum_field_engine.py
| Feature | main | pre-SF | Conflict? |
| :--- | :--- | :--- | :--- |
| **Analytical OU** | No (Explicit `0.0`) | **Yes** (Uses `scipy.special.erfi`) | **Yes** — Optimization vs Feature |
| **Dependencies** | **Numba, NumPy** | Numba, NumPy, **SciPy** | **Yes** — `main` is lighter |

### training/timeframe_belief_network.py
| Feature | main | pre-SF | Conflict? |
| :--- | :--- | :--- | :--- |
| **Exit Logic** | **Time Exhaustion** (MFE stats) | **Physics Decay** (Real-time trajectory) | **Yes** — Logic divergence |
| **Complexity** | Low | High (Cascading decay calculation) | **Yes** |

### core/bayesian_brain.py
| Feature | main | pre-SF | Conflict? |
| :--- | :--- | :--- | :--- |
| **Logging** | **Logger** (`logger.info`) | Print (`print`) | **Yes** — Standard practice |

### core/dynamic_binner.py
| Feature | main | pre-SF | Conflict? |
| :--- | :--- | :--- | :--- |
| **Optimization** | **Bisect** (Python list, O(log n)) | NumPy (`searchsorted`, slower for scalars) | **Yes** — Performance optimization |

### training/trade_analytics.py
| Feature | main | pre-SF | Conflict? |
| :--- | :--- | :--- | :--- |
| **Part 8** | **Yes** (Best/Worst Deep Dive) | No | **Yes** — Feature added in main |
| **Console Output** | Minimal | Verbose | **Yes** |

### config/oracle_config.py
| Feature | main | pre-SF | Conflict? |
| :--- | :--- | :--- | :--- |
| **Thresholds** | **Identical** | **Identical** | No |

## Merge Strategy Recommendations

The `main` branch is consistently superior in terms of:
1.  **Architecture:** Implements "Snowflake" (Long/Short split).
2.  **Performance:** Uses `bisect` for binning, `try-except` for attributes, avoids `scipy` in hot paths.
3.  **Standards:** Uses `logging` instead of `print`, `Optuna` instead of custom genetic algorithms.

**Recommendation:**
*   **Discard `pre-snowflake` changes.** Do not merge.
*   The features present in `pre-snowflake` (Semantic Naming, Physics Decay, Analytical OU) appear to be experimental paths that were either optimized away or rejected in favor of the current "Snowflake" architecture.
*   **Key Action:** Ensure `training/fractal_dna_tree.py` is preserved (it is present in `main`, absent in `pre-snowflake`).

### Key Files Recap
*   `training/fractal_clustering.py`: **Keep main** (Optimized).
*   `training/orchestrator.py`: **Keep main** (Snowflake).
*   `training/timeframe_belief_network.py`: **Keep main** (Time Exhaustion).
*   `training/doe_parameter_generator.py`: **Keep main** (Optuna).
*   `core/quantum_field_engine.py`: **Keep main** (Bolt optimizations).
*   `core/bayesian_brain.py`: **Keep main** (Logging).
*   `config/oracle_config.py`: **Identical**.
