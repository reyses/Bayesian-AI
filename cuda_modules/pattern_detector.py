"""
Bayesian-AI - Pattern Detector
CUDA-accelerated pattern recognition (L7)
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple

PATTERN_NONE = 0
PATTERN_COMPRESSION = 1
PATTERN_WEDGE = 2
PATTERN_BREAKDOWN = 3

try:
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
except (ImportError, Exception):
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @cuda.jit
    def detect_pattern_kernel(highs, lows, out_type, out_conf):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)

        for idx in range(start, highs.shape[0], stride):
            # Initialize
            out_type[idx] = PATTERN_NONE
            out_conf[idx] = 0.0

            # Check Compression (Priority 1)
            # Needs at least 10 bars (indices idx-9 to idx)
            if idx >= 9:
                # Recent range: idx-4 to idx
                recent_max = -1e9
                recent_min = 1e9
                for k in range(idx - 4, idx + 1):
                    val_h = highs[k]
                    val_l = lows[k]
                    if val_h > recent_max: recent_max = val_h
                    if val_l < recent_min: recent_min = val_l
                recent_range = recent_max - recent_min

                # Previous range: idx-9 to idx-5
                prev_max = -1e9
                prev_min = 1e9
                for k in range(idx - 9, idx - 4):
                    val_h = highs[k]
                    val_l = lows[k]
                    if val_h > prev_max: prev_max = val_h
                    if val_l < prev_min: prev_min = val_l
                prev_range = prev_max - prev_min

                if prev_range > 0 and recent_range < prev_range * 0.7:
                     out_type[idx] = PATTERN_COMPRESSION
                     out_conf[idx] = 0.85
                     continue

            # Check Wedge and Breakdown (Priority 2 & 3)
            # Needs at least 5 bars (indices idx-4 to idx)
            if idx >= 4:
                # Wedge
                if lows[idx] > lows[idx-4] and highs[idx] < highs[idx-4]:
                     out_type[idx] = PATTERN_WEDGE
                     out_conf[idx] = 0.75
                     continue

                # Breakdown
                # lows[idx] < min(lows[idx-4 : idx]) (strictly previous 4 bars)
                min_prev = 1e9
                for k in range(idx - 4, idx):
                    if lows[k] < min_prev:
                        min_prev = lows[k]

                if lows[idx] < min_prev:
                    out_type[idx] = PATTERN_BREAKDOWN
                    out_conf[idx] = 0.90
                    continue
else:
    detect_pattern_kernel = None

class CUDAPatternDetector:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and NUMBA_AVAILABLE

        if self.use_gpu:
            try:
                self.use_gpu = cuda.is_available()
            except Exception:
                self.use_gpu = False

        if not self.use_gpu and use_gpu:
             logging.warning("CUDA requested for PatternDetector but not available. Falling back to CPU.")

    def detect(self, bars: pd.DataFrame, window_size: int = 20) -> Tuple[str, float]:
        if self.use_gpu:
             return self._detect_gpu(bars)
        else:
             return self._detect_cpu(bars)

    def _detect_gpu(self, bars: pd.DataFrame) -> Tuple[str, float]:
        # Convert to numpy arrays
        highs = bars['high'].values.astype(np.float32)
        lows = bars['low'].values.astype(np.float32)
        n = len(highs)

        # We need at least 5 bars for any pattern
        if n < 5:
            return ('none', 0.0)

        # Optimization: process only a relevant window on GPU if data is huge
        # We need at least 10 bars for compression check at the end.
        # So we can send the last 20 bars (or window_size, if larger).
        LOOKBACK = 200
        if n > LOOKBACK:
            highs = highs[-LOOKBACK:]
            lows = lows[-LOOKBACK:]
            n = LOOKBACK

        # Let's handle the data transfer
        d_highs = cuda.to_device(highs)
        d_lows = cuda.to_device(lows)
        d_out_type = cuda.device_array(n, dtype=np.int32)
        d_out_conf = cuda.device_array(n, dtype=np.float32)

        # Launch kernel
        threads_per_block = 256
        blocks = (n + threads_per_block - 1) // threads_per_block

        detect_pattern_kernel[blocks, threads_per_block](d_highs, d_lows, d_out_type, d_out_conf)

        # We only care about the last element
        # Copy back only the last element? Or copy all?
        # Copying one element is tricky without a dedicated kernel or array slice copy.
        # Numba cuda allows copying slices back.

        last_type_arr = d_out_type[n-1:n].copy_to_host()
        last_conf_arr = d_out_conf[n-1:n].copy_to_host()

        last_type = last_type_arr[0]
        last_conf = last_conf_arr[0]

        pattern_name = 'none'
        if last_type == PATTERN_COMPRESSION:
            pattern_name = 'compression'
        elif last_type == PATTERN_WEDGE:
            pattern_name = 'wedge'
        elif last_type == PATTERN_BREAKDOWN:
            pattern_name = 'breakdown'

        return (pattern_name, float(last_conf))

    def _detect_cpu(self, bars: pd.DataFrame) -> Tuple[str, float]:
        # Logic from layer_engine_cuda.py _compute_L7_15m_CPU
        highs = bars['high'].values
        lows = bars['low'].values

        if len(highs) >= 5:
            # Note: This compression check implicitly requires at least 10 bars
            # for prev_range calculation to be safe/meaningful, otherwise slicing
            # like [-10:-5] on small array yields empty and .max() raises error.
            # We add a check for safety.
            if len(highs) >= 10:
                recent_range = highs[-5:].max() - lows[-5:].min()
                prev_range = highs[-10:-5].max() - lows[-10:-5].min()

                if prev_range > 0 and recent_range < prev_range * 0.7:
                    return ('compression', 0.85)

        # Need to ensure we have enough data for these checks
        if len(lows) >= 5:
            if lows[-1] > lows[-5] and highs[-1] < highs[-5]:
                return ('wedge', 0.75)

            if lows[-1] < lows[-5:-1].min():
                return ('breakdown', 0.90)

        return ('none', 0.0)

_pattern_detector = None
def get_pattern_detector(use_gpu: bool = True) -> CUDAPatternDetector:
    global _pattern_detector
    if _pattern_detector is None:
        _pattern_detector = CUDAPatternDetector(use_gpu)
    return _pattern_detector
