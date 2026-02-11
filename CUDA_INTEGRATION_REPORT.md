# CUDA Modules Integration Report

## Executive Summary

The `cuda_modules` directory contains legacy components (Layer 7-9 Hierarchy) built with Numba for CUDA acceleration. While the architecture has shifted to the "Fractal Three-Body Quantum" engine (PyTorch/NumPy vectorized), several core algorithms within `cuda_modules` offer superior logic to their current replacements in `QuantumFieldEngine`.

This report details the specific logic that should be ported and integrated into the main `QuantumFieldEngine` to enhance the system's robustness without reverting to the deprecated architecture.

---

## 1. Pattern Detector (`cuda_modules/pattern_detector.py`)

**Current Status:** Deprecated Numba kernel.
**Main Logic:** Detects geometric patterns: `COMPRESSION`, `WEDGE`, and `BREAKDOWN`.
**Current Replacement:** `QuantumFieldEngine` uses `pattern_maturity` (Z-score based) but lacks geometric shape recognition.

### Integration Strategy
The geometric logic is valuable for context. We should port the sliding window logic to vectorized PyTorch/NumPy operations in `QuantumFieldEngine.batch_compute_states`.

**Algorithm to Port:**
1.  **Compression (High Priority):**
    - Logic: Recent range (5 bars) < 70% of previous range (5 bars).
    - *Benefit:* Identifies "coiling" before expansion, which complements the `entropy` metric in the Quantum state.
2.  **Wedge:**
    - Logic: Higher lows and lower highs over 5 bars.
    - *Benefit:* confirming contraction.
3.  **Breakdown:**
    - Logic: Current low < minimum of previous 4 lows.
    - *Benefit:* Instant confirmation of support failure.

**Action Items:**
- Add `pattern_type` (Enum/String) field to `ThreeBodyQuantumState`.
- Implement vectorized `_detect_geometric_patterns` in `QuantumFieldEngine`.

---

## 2. Velocity Gate (`cuda_modules/velocity_gate.py`)

**Current Status:** Deprecated Numba kernel.
**Main Logic:** Detects "Cascades" (large move in short time) using a rolling window (e.g., 10 points in 0.5s).
**Current Replacement:** `QuantumFieldEngine` checks `abs(velocity) > 1.0`. This is a simple threshold and misses the *temporal density* of the move.

### Integration Strategy
The rolling window "max-min" logic is far more robust than a simple velocity threshold, as it captures *sustained* pressure rather than single-tick noise.

**Algorithm to Port:**
- **Rolling Window Cascade:**
    - Scan last N ticks (or 1s bars).
    - Calculate `(Max High - Min Low)` within a time window (e.g., 5 seconds).
    - Trigger `cascade_detected` if range > Threshold AND time_delta < Time_Window.

**Action Items:**
- Enhance `cascade_detected` in `ThreeBodyQuantumState` to use this logic.
- In `QuantumFieldEngine`, implement a rolling max/min check over the `micro_df` (or 1s bars) to derive this metric.

---

## 3. Confirmation Engine (`cuda_modules/confirmation.py`)

**Current Status:** Deprecated Numba kernel.
**Main Logic:** `Volume[i] > Mean(Volume[i-3:i]) * 1.2`.
**Current Replacement:** `QuantumFieldEngine` uses `Volume[i] > Mean(Volume[i-20:i]) * 1.2`.

### Integration Strategy
The new logic (20-bar mean) is statistically more stable than the legacy 3-bar mean. However, the "flash" nature of the 3-bar check is useful for high-frequency scalping.

**Recommendation:**
- **Merge Logic:** Keep the 20-bar mean for the "Structure" check.
- **Optional:** Add a `flash_volume` flag to `ThreeBodyQuantumState` using the 3-bar logic for "urgent" entry signals.

---

## 4. Hardened Verification (`cuda_modules/hardened_verification.py`)

**Current Status:** Standalone script.
**Main Logic:**
1.  **Handshake:** Verify GPU availability (RTX 3060 specific checks).
2.  **Injection:** CPU -> GPU data transfer verification.
3.  **Handoff:** GPU kernel execution verification.

### Integration Strategy
This is a critical "Health Check" tool. It should be adapted to verify the *PyTorch/CUDA* stack instead of Numba.

**Action Items:**
- Create `scripts/gpu_health_check.py` (or similar).
- Port the "Injection" and "Handoff" tests to use `torch.tensor` operations on the GPU.
- Use this script in the `pre-commit` or CI/CD pipeline to ensure the production environment is GPU-accelerated and stable.

---

## Summary of Recommendations

| Component | Status | Recommendation | Priority |
| :--- | :--- | :--- | :--- |
| **Pattern Detector** | Deprecated | **Port Logic** to `QuantumFieldEngine` (Vectorized). Add `pattern_type` to State. | **High** |
| **Velocity Gate** | Deprecated | **Upgrade** `cascade_detected` in `QuantumFieldEngine` with rolling window logic. | **High** |
| **Confirmation** | Deprecated | **Discard** (New logic is superior), optionally keep as "Flash" signal. | Low |
| **Verification** | Script | **Adapt** into `scripts/gpu_health_check.py` for PyTorch stack. | Medium |
