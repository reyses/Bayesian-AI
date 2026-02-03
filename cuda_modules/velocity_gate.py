"""
Bayesian-AI - CUDA Velocity Gate (Layer 9)
Numba-accelerated tick-level cascade detection: 10+ points in <0.5sec
"""
import numpy as np
import pandas as pd
import time

try:
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
except (ImportError, Exception):
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @cuda.jit
    def detect_cascade_kernel(tick_prices, tick_times, cascade_threshold, time_window, results):
        """
        CUDA kernel: Detect velocity cascades in tick stream
        """
        idx = cuda.grid(1)

        if idx >= tick_prices.shape[0] - 50:  # Need at least 50 ticks
            return

        # Scan last 50 ticks for cascade
        window_start = max(0, idx)
        window_end = min(tick_prices.shape[0], idx + 50)

        max_price = tick_prices[window_start]
        min_price = tick_prices[window_start]
        start_time = tick_times[window_start]
        end_time = tick_times[window_end - 1]

        # Find max/min in window
        for i in range(window_start, window_end):
            if tick_prices[i] > max_price:
                max_price = tick_prices[i]
            if tick_prices[i] < min_price:
                min_price = tick_prices[i]

        # Calculate move and time elapsed
        price_move = abs(max_price - min_price)
        time_elapsed = end_time - start_time

        # Cascade condition: Large move in short time
        if price_move >= cascade_threshold and time_elapsed <= time_window:
            results[idx] = 1
        else:
            results[idx] = 0
else:
    detect_cascade_kernel = None

class CUDAVelocityGate:
    """High-level interface for CUDA velocity cascade detection"""
    
    def __init__(self, cascade_threshold=10.0, time_window=0.5, use_gpu=True):
        self.cascade_threshold = cascade_threshold  # Points
        self.time_window = time_window  # Seconds
        self.use_gpu = use_gpu and NUMBA_AVAILABLE
        
        if self.use_gpu:
            try:
                self.use_gpu = cuda.is_available()
            except Exception:
                # Catch dynamic lib errors (missing drivers)
                self.use_gpu = False

        if not self.use_gpu and use_gpu: # User requested GPU but not available
            raise RuntimeError("CUDA requested for VelocityGate but not available. CPU fallback disabled by configuration.")
    
    def detect_cascade(self, tick_data):
        """
        Detect if velocity cascade is occurring
        
        Args:
            tick_data: Array or DataFrame with ['price', 'timestamp'] for last 100+ ticks
            
        Returns:
            bool: True if cascade detected
        """
        # Optimization: Only process recent history
        # We only need the last 50 ticks for the algorithm + some buffer
        # This prevents O(N) transfer/processing on every tick
        LOOKBACK = 200

        if isinstance(tick_data, np.ndarray):
            if len(tick_data) < 50:
                return False

            # Slice to relevant window
            relevant_data = tick_data[-LOOKBACK:]

            prices = relevant_data.astype(np.float32)
            times = np.arange(len(prices), dtype=np.float32) * 0.01  # Assume 10ms between ticks
        elif isinstance(tick_data, pd.DataFrame):
            # DataFrame from DataAggregator
            if len(tick_data) < 50:
                return False

            # Slice to relevant window
            relevant_data = tick_data.iloc[-LOOKBACK:] if len(tick_data) > LOOKBACK else tick_data

            prices = relevant_data['price'].values.astype(np.float32)
            
            # Handle timestamp conversion
            if 'timestamp' in relevant_data.columns:
                timestamps = relevant_data['timestamp'].values
                # Check if it's datetime64
                if pd.api.types.is_datetime64_any_dtype(timestamps):
                    # Convert to float (seconds)
                    times = timestamps.astype('int64') // 10**9 # ns to s
                    times = times.astype(np.float32)
                elif hasattr(timestamps[0], 'timestamp'): # List of objects?
                    times = np.array([t.timestamp() for t in timestamps], dtype=np.float32)
                else:
                    # Already float/int
                    times = timestamps.astype(np.float32)
            else:
                times = np.arange(len(prices), dtype=np.float32) * 0.01
        else:
             return False # Unknown format
        
        if self.use_gpu and NUMBA_AVAILABLE:
            # GPU execution
            d_prices = cuda.to_device(prices)
            d_times = cuda.to_device(times)
            
            results = np.zeros(len(prices), dtype=np.int32)
            d_results = cuda.to_device(results)
            
            # Launch kernel
            threads_per_block = 256
            blocks = (len(prices) + threads_per_block - 1) // threads_per_block
            
            detect_cascade_kernel[blocks, threads_per_block](
                d_prices, d_times, self.cascade_threshold, self.time_window, d_results
            )
            
            results = d_results.copy_to_host()
            
            # Any cascade in recent window?
            return bool(results[-10:].sum() > 0)
            
        else:
            # CPU fallback
            return self._cpu_detect(prices, times)
    
    def _cpu_detect(self, prices, times):
        """CPU fallback implementation"""
        if len(prices) < 50:
            return False
        
        # Check last 50 ticks
        window_prices = prices[-50:]
        window_times = times[-50:]
        
        price_move = abs(window_prices.max() - window_prices.min())
        time_elapsed = window_times[-1] - window_times[0]
        
        return bool(price_move >= self.cascade_threshold and time_elapsed <= self.time_window)

# Singleton
_velocity_gate = None

def get_velocity_gate(cascade_threshold=10.0, time_window=0.5, use_gpu=True):
    """Get singleton velocity gate"""
    global _velocity_gate
    if _velocity_gate is None:
        _velocity_gate = CUDAVelocityGate(cascade_threshold, time_window, use_gpu)
    return _velocity_gate
