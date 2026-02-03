"""
Bayesian-AI - Confirmation Engine
CUDA-accelerated trade confirmation (L8)
"""
import pandas as pd
import numpy as np

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except (ImportError, Exception):
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @cuda.jit
    def confirm_kernel(volumes, results):
        idx = cuda.grid(1)

        # We need at least 3 elements to compute the mean of the last 3
        if idx < 2 or idx >= volumes.shape[0]:
            return

        v3 = volumes[idx]
        v2 = volumes[idx-1]
        v1 = volumes[idx-2]

        mean_3 = (v1 + v2 + v3) / 3.0

        # Condition: current volume > mean of last 3 * 1.2
        if v3 > mean_3 * 1.2:
            results[idx] = 1
        else:
            results[idx] = 0
else:
    confirm_kernel = None

class CUDAConfirmationEngine:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and NUMBA_AVAILABLE

        if self.use_gpu:
            try:
                self.use_gpu = cuda.is_available()
            except Exception:
                self.use_gpu = False

        if not self.use_gpu and use_gpu:
             raise RuntimeError("CUDA requested for ConfirmationEngine but not available. CPU fallback disabled by configuration.")

    def confirm(self, bars: pd.DataFrame, L7_pattern_active: bool) -> bool:
        if not L7_pattern_active:
            return False

        if self.use_gpu and NUMBA_AVAILABLE:
             # CUDA Implementation
             if 'volume' not in bars.columns:
                 return False

             volumes = bars['volume'].values
             if len(volumes) < 3:
                 return False

             # Optimization: Only process recent history to minimize transfer overhead
             LOOKBACK = 100
             if len(volumes) > LOOKBACK:
                 volumes = volumes[-LOOKBACK:]

             # Prepare data for GPU
             # Numba needs contiguous arrays, astype returns a copy which is usually contiguous
             volumes_gpu = volumes.astype(np.float32)

             d_volumes = cuda.to_device(volumes_gpu)
             results = np.zeros(len(volumes_gpu), dtype=np.int32)
             d_results = cuda.to_device(results)

             # Launch kernel
             threads_per_block = 256
             blocks = (len(volumes_gpu) + threads_per_block - 1) // threads_per_block

             confirm_kernel[blocks, threads_per_block](d_volumes, d_results)

             results = d_results.copy_to_host()

             # We only care about the confirmation status of the latest bar
             return bool(results[-1] == 1)
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
