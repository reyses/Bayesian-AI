import pandas as pd
import numpy as np

class CUDAConfirmationEngine:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        if not self.use_gpu:
             # print("[CUDA] GPU not available for confirmation engine, using CPU")
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
