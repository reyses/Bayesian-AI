"""
CUDA-Accelerated Pattern Detector
Modernized implementation of legacy pattern recognition logic using Numba CUDA kernels.
Designed for high-throughput batch processing in QuantumFieldEngine.
"""
import numpy as np
from core.pattern_utils import PATTERN_NONE, PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN

# Constants for kernel (must be int)
K_NONE = 0
K_COMPRESSION = 1
K_WEDGE = 2
K_BREAKDOWN = 3

# Thresholds (must match core/pattern_utils.py)
COMPRESSION_RATIO = 0.7

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @cuda.jit
    def detect_geometric_pattern_kernel(highs, lows, out_type):
        """
        CUDA Kernel to detect geometric patterns for every bar in parallel.

        Logic matches core.pattern_utils.detect_geometric_patterns_vectorized:
        - Compression: Recent range < 0.7 * Prev range
        - Wedge: Higher Lows AND Lower Highs (5 bars)
        - Breakdown: Low < Min(Prev 4 Lows)
        """
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        n = highs.shape[0]

        for idx in range(start, n, stride):
            # Initialize
            out_type[idx] = K_NONE

            # Needs at least 10 bars for full lookback (Compression needs 10)
            if idx < 9:
                continue

            # 1. COMPRESSION (Priority 1)
            # Recent range: idx-4 to idx (5 bars)
            # Prev range: idx-9 to idx-5 (5 bars)

            # Compute recent range
            rec_max = -1e30
            rec_min = 1e30
            for k in range(idx - 4, idx + 1):
                h = highs[k]
                l = lows[k]
                if h > rec_max: rec_max = h
                if l < rec_min: rec_min = l
            rec_range = rec_max - rec_min

            # Compute prev range
            prev_max = -1e30
            prev_min = 1e30
            for k in range(idx - 9, idx - 4): # indices idx-9, -8, -7, -6, -5
                h = highs[k]
                l = lows[k]
                if h > prev_max: prev_max = h
                if l < prev_min: prev_min = l
            prev_range = prev_max - prev_min

            if prev_range > 0 and rec_range < prev_range * COMPRESSION_RATIO:
                out_type[idx] = K_COMPRESSION
                # Continue to next priority?
                # CPU version uses masks:
                # patterns[compression_mask] = PATTERN_COMPRESSION
                # patterns[wedge_mask] = PATTERN_WEDGE
                # patterns[breakdown_mask] = PATTERN_BREAKDOWN
                # So Breakdown overwrites Wedge overwrites Compression?
                # Let's check priority in pattern_utils.py:
                # 1. patterns[...] = PATTERN_COMPRESSION
                # 2. patterns[wedge_mask] = PATTERN_WEDGE (Overwrites!)
                # 3. patterns[breakdown_mask] = PATTERN_BREAKDOWN (Overwrites!)
                # So Priority is Breakdown > Wedge > Compression.
                # We should check in reverse order or use a variable.

                # Let's mimic the CPU overwrite logic.

            # 2. WEDGE (Priority 2)
            # lows[idx] > lows[idx-4] and highs[idx] < highs[idx-4]
            if lows[idx] > lows[idx-4] and highs[idx] < highs[idx-4]:
                out_type[idx] = K_WEDGE

            # 3. BREAKDOWN (Priority 3)
            # lows[idx] < min(lows[idx-4 : idx])
            # Min of previous 4 bars (idx-4, -3, -2, -1)
            min_prev_4 = 1e30
            for k in range(idx - 4, idx):
                if lows[k] < min_prev_4:
                    min_prev_4 = lows[k]

            if lows[idx] < min_prev_4:
                out_type[idx] = K_BREAKDOWN

else:
    detect_geometric_pattern_kernel = None


def detect_geometric_patterns_cuda(highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    """
    Execute CUDA kernel for geometric pattern detection.
    Returns array of pattern strings (matching CPU implementation).
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba not installed/available for CUDA patterns")

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    n = len(highs)
    highs = np.ascontiguousarray(highs.astype(np.float32))
    lows = np.ascontiguousarray(lows.astype(np.float32))

    # Alloc GPU memory
    d_highs = cuda.to_device(highs)
    d_lows = cuda.to_device(lows)
    d_out_type = cuda.device_array(n, dtype=np.int32)

    # Config
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    # Launch
    detect_geometric_pattern_kernel[blocks_per_grid, threads_per_block](
        d_highs, d_lows, d_out_type
    )

    # Copy back
    out_types = d_out_type.copy_to_host()

    # Map int to string (Vectorized map)
    # Default 'NONE'
    # We can use np.choose or a lookup array

    # Create lookup table
    # 0: NONE, 1: COMPRESSION, 2: WEDGE, 3: BREAKDOWN
    # Note: PATTERN_* are strings.

    # Fast mapping using object array
    lookup = np.array([PATTERN_NONE, PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN], dtype=object)

    # Safety clamp
    out_types = np.clip(out_types, 0, 3)

    return lookup[out_types]
