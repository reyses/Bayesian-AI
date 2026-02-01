"""
Algorithm V2 - Confirmation Engine
CUDA-accelerated trade confirmation (L8)
"""
import pandas as pd
import numpy as np

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class CUDAConfirmationEngine:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and NUMBA_AVAILABLE

        if self.use_gpu:
            self.use_gpu = cuda.is_available()

        if not self.use_gpu and use_gpu:
             # GPU requested but not available
             pass

    def confirm(self, bars: pd.DataFrame, L7_pattern_active: bool) -> bool:
        if not L7_pattern_active:
            return False

        if self.use_gpu:
             # TODO: Implement CUDA kernel
             return self._confirm_cpu(bars, L7_pattern_active)
        else:
             return self._confirm_cpu(bars, L7_pattern_active)

    def _confirm_cpu(self, bars: pd.DataFrame, L7_pattern_active: bool) -> bool:
        # Logic from layer_engine_cuda.py _compute_L8_5m_CPU
        if not L7_pattern_active:
            return False

        volumes = bars['volume'].values
        if len(volumes) >= 3 and volumes[-1] > volumes[-3:].mean() * 1.2:
            return True

        return False

_confirmation_engine = None
def get_confirmation_engine(use_gpu: bool = True) -> CUDAConfirmationEngine:
    global _confirmation_engine
    if _confirmation_engine is None:
        _confirmation_engine = CUDAConfirmationEngine(use_gpu)
    return _confirmation_engine
