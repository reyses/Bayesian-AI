# Evaluation of Legacy CUDA Pattern Detector

## Overview
This document evaluates the usefulness of restoring `archive/cuda_modules/pattern_detector.py`, a legacy module designed for GPU-accelerated pattern recognition.

## Comparison

| Feature | Legacy (`cuda_modules/pattern_detector.py`) | Current (`core/pattern_utils.py`) |
| :--- | :--- | :--- |
| **Algorithm** | Detects Compression, Wedge, Breakdown. | Detects Compression, Wedge, Breakdown. |
| **Logic** | Compression: `recent_range < 0.7 * prev_range`<br>Wedge: `low > prev_low` & `high < prev_high`<br>Breakdown: `low < min(prev_4_lows)` | Identical logic implemented via NumPy/Pandas. |
| **Execution** | Numba CUDA Kernel (JIT compiled). Requires GPU. | NumPy/Pandas Vectorization (C-optimized). Runs on CPU. |
| **Overhead** | High (Host-to-Device memory transfer, Kernel launch). | Low (Zero-copy operations on existing arrays). |
| **Confidence** | Hardcoded (e.g., 0.85, 0.75). | **Bayesian Learning**: Confidence is derived from historical success rates of the state. |
| **Dependencies** | Requires NVIDIA Driver + CUDA Toolkit + Numba. | Standard Python Data Science stack (Pandas, NumPy). |

## Analysis

1.  **Functional Equivalence**: The core logic for pattern detection is identical in both implementations. No unique patterns were lost.
2.  **Performance**: For the scale of data typically processed (OHLCV bars, thousands to millions), CPU vectorization is often faster than CUDA due to the significant overhead of moving data to/from the GPU. The complexity of these patterns ($O(N)$ with small constants) does not saturate the CPU enough to justify GPU offloading.
3.  **Architecture**: The legacy module returns hardcoded "confidence" values (e.g., 0.85 for compression). This contradicts the project's **Bayesian Brain** philosophy, where confidence is dynamically learned from trade outcomes. Hardcoding these values would introduce bias.

## Conclusion

**Status: DO NOT RESTORE**

The module `archive/cuda_modules/pattern_detector.py` is **obsolete**. Its functionality is fully covered by `core/pattern_utils.py` in a way that is:
*   More efficient (no PCI-E transfer overhead).
*   Easier to maintain (no CUDA dependencies).
*   Better aligned with the Bayesian learning architecture.

Restoring this module would introduce unnecessary complexity and technical debt without providing any functional or performance benefit.
