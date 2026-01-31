import pandas as pd
import numpy as np
from typing import Tuple

class CUDAPatternDetector:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        if not self.use_gpu:
             # print("[CUDA] GPU not available for pattern detector, using CPU")
             pass

    def detect(self, bars: pd.DataFrame, window_size: int = 20) -> Tuple[str, float]:
        if self.use_gpu:
             # TODO: Implement CUDA kernel
             # For now fallback to CPU or return placeholder
             return self._detect_cpu(bars)
        else:
             return self._detect_cpu(bars)

    def _detect_cpu(self, bars: pd.DataFrame) -> Tuple[str, float]:
        # Logic from layer_engine_cuda.py _compute_L7_15m_CPU
        highs = bars['high'].values
        lows = bars['low'].values

        if len(highs) >= 10:
            recent_range = highs[-5:].max() - lows[-5:].min()
            prev_range = highs[-10:-5].max() - lows[-10:-5].min()

            if prev_range > 0 and recent_range < prev_range * 0.7:
                return ('compression', 0.85)

        # Need to ensure we have enough data for these checks
        if len(lows) >= 5:
            if lows[-1] > lows[-5] and highs[-1] < highs[-5]:
                return ('wedge', 0.75)

            if lows[-1] < lows[-5:].min():
                return ('breakdown', 0.90)

        return ('none', 0.0)

_pattern_detector = None
def get_pattern_detector(use_gpu: bool = True) -> CUDAPatternDetector:
    global _pattern_detector
    if _pattern_detector is None:
        _pattern_detector = CUDAPatternDetector(use_gpu)
    return _pattern_detector
