#!/usr/bin/env python3
"""
GPU-Accelerated Fractal Atlas Builder
Generates a persistent, multi-resolution copy of the entire dataset.
Uses CUDA for high-performance resampling and VRAM-aware batching.
"""

import os
import sys
import argparse
import math
import gc
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# --- GPU SETUP ---
GPU_AVAILABLE = False
try:
    import torch
    import numba
    from numba import cuda
    if cuda.is_available():
        GPU_AVAILABLE = True
except ImportError:
    pass

# Dummy decorator and mock classes if GPU not available
if not GPU_AVAILABLE:
    def jit_device(func): return func
    def jit_kernel(func): return func

    class MockDeviceArray:
        def __init__(self, ary):
            self.ary = ary
        def copy_to_host(self):
            return self.ary

    class MockCuda:
        def jit(self, *args, **kwargs):
            if 'device' in kwargs and kwargs['device']:
                return jit_device
            return jit_kernel
        def grid(self, n): return 0
        def to_device(self, ary): return MockDeviceArray(ary)
        def device_array(self, shape, dtype): return MockDeviceArray(np.zeros(shape, dtype=dtype))
        def empty_cache(self): pass
        def is_available(self): return False

    cuda = MockCuda()
else:
    # Use real decorators
    pass

# Constants
RESOLUTIONS = {
    '1s': 1,
    '5s': 5,
    '15s': 15,
    '30s': 30,
    '1m': 60,
    '2m': 120,
    '3m': 180,
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '4h': 14400,
    '1D': 86400,
    '1W': 604800,
}

# --- CUDA KERNELS ---

@cuda.jit(device=True)
def _device_binary_search(arr, val, n):
    """
    Finds the insertion point (left) for val in sorted arr.
    Returns index i such that arr[i-1] < val <= arr[i].
    """
    l = 0
    r = n
    while l < r:
        mid = (l + r) // 2
        if arr[mid] < val:
            l = mid + 1
        else:
            r = mid
    return l

@cuda.jit
def _resample_kernel(in_times, in_opens, in_highs, in_lows, in_closes, in_vols,
                     out_times, out_opens, out_highs, out_lows, out_closes, out_vols, out_counts,
                     interval, n_input, n_output):
    """
    Resamples OHLCV data to a new interval.
    Each thread handles one output bar.
    """
    idx = cuda.grid(1)
    if idx < n_output:
        start_t = out_times[idx]
        end_t = start_t + interval

        start_idx = _device_binary_search(in_times, start_t, n_input)
        end_idx = _device_binary_search(in_times, end_t, n_input)

        count = end_idx - start_idx
        out_counts[idx] = count

        if count > 0:
            agg_open = in_opens[start_idx]
            agg_close = in_closes[end_idx - 1]
            agg_vol = 0.0

            # Initialize min/max
            agg_high = in_highs[start_idx]
            agg_low = in_lows[start_idx]

            for i in range(start_idx, end_idx):
                h = in_highs[i]
                l = in_lows[i]
                v = in_vols[i]

                if h > agg_high:
                    agg_high = h
                if l < agg_low:
                    agg_low = l
                agg_vol += v

            out_opens[idx] = agg_open
            out_highs[idx] = agg_high
            out_lows[idx] = agg_low
            out_closes[idx] = agg_close
            out_vols[idx] = agg_vol

class ParquetWriterManager:
    """Manages open ParquetWriters to avoid repeated file open/close."""
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        # Key: (resolution, year, month), Value: ParquetWriter
        self.writers = {}
        # Keep track of schema for each resolution
        self.schemas = {}

    def write(self, resolution, df):
        if df.empty:
            return

        # Ensure timestamp is datetime for period extraction
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        else:
            return

        # Group by Day (YYYYMMDD) so discovery agent and forward pass can
        # filter by date using simple lexicographic string comparison.
        periods = ts.dt.to_period('D').unique()

        for period in periods:
            mask = (ts.dt.to_period('D') == period)
            chunk = df[mask].copy()

            key = (resolution, period.year, period.month, period.day)

            if key not in self.writers:
                self._open_writer(key, chunk)

            # Write
            if key in self.writers:
                try:
                    table = pa.Table.from_pandas(chunk, schema=self.schemas.get(key))
                    self.writers[key].write_table(table)
                except Exception as e:
                    print(f"Error writing table for {key}: {e}")

    def _open_writer(self, key, sample_df):
        resolution, year, month, day = key
        res_dir = self.base_path / resolution
        res_dir.mkdir(parents=True, exist_ok=True)

        # YYYYMMDD.parquet — matches discovery agent and forward pass expectations
        filename = f"{year}{month:02d}{day:02d}.parquet"
        file_path = res_dir / filename

        try:
            schema = pa.Table.from_pandas(sample_df).schema
            schema = schema.remove_metadata()
            self.schemas[key] = schema
            self.writers[key] = pq.ParquetWriter(
                file_path, schema, compression='snappy', use_dictionary=False
            )
        except Exception as e:
            print(f"Error opening writer for {key}: {e}")

    def close_all(self):
        for w in self.writers.values():
            try:
                w.close()
            except Exception as e:
                print(f"Error closing writer: {e}")
        self.writers.clear()

class AtlasBuilder:
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.writer_manager = ParquetWriterManager(output_path)

    def calculate_batch_size(self):
        if not GPU_AVAILABLE:
            return 100000
        try:
            free_mem, _ = torch.cuda.mem_get_info()
            # 1 row ~ 48 bytes. Output expansion ~ 2x (allocations).
            # Safe limit: 40% of VRAM.
            batch_size = int((free_mem * 0.4) / 64)
            return max(min(batch_size, 10_000_000), 100_000)
        except Exception:
            return 500_000

    def process_gpu(self, df, interval):
        if df.empty: return pd.DataFrame()

        # Input transfer
        in_times = df['timestamp'].values.astype(np.int64)
        in_opens = df['open'].values.astype(np.float64)
        in_highs = df['high'].values.astype(np.float64)
        in_lows = df['low'].values.astype(np.float64)
        in_closes = df['close'].values.astype(np.float64)
        in_vols = df['volume'].values.astype(np.float64)

        n_input = len(in_times)

        # Output setup
        min_t = in_times[0]
        max_t = in_times[-1]
        start_t = (min_t // interval) * interval
        end_t = ((max_t // interval) + 1) * interval

        # Safety check for empty range
        if start_t >= end_t: return pd.DataFrame()

        out_times_np = np.arange(start_t, end_t, interval, dtype=np.int64)
        n_output = len(out_times_np)

        if n_output == 0: return pd.DataFrame()

        # Device Alloc
        d_in_times = cuda.to_device(in_times)
        d_in_opens = cuda.to_device(in_opens)
        d_in_highs = cuda.to_device(in_highs)
        d_in_lows = cuda.to_device(in_lows)
        d_in_closes = cuda.to_device(in_closes)
        d_in_vols = cuda.to_device(in_vols)

        d_out_times = cuda.to_device(out_times_np)
        d_out_opens = cuda.device_array(n_output, dtype=np.float64)
        d_out_highs = cuda.device_array(n_output, dtype=np.float64)
        d_out_lows = cuda.device_array(n_output, dtype=np.float64)
        d_out_closes = cuda.device_array(n_output, dtype=np.float64)
        d_out_vols = cuda.device_array(n_output, dtype=np.float64)
        d_out_counts = cuda.device_array(n_output, dtype=np.int32)

        # Kernel
        tpb = 256
        blocks = (n_output + tpb - 1) // tpb
        _resample_kernel[blocks, tpb](
            d_in_times, d_in_opens, d_in_highs, d_in_lows, d_in_closes, d_in_vols,
            d_out_times, d_out_opens, d_out_highs, d_out_lows, d_out_closes, d_out_vols, d_out_counts,
            interval, n_input, n_output
        )

        # Copy Back
        out_counts = d_out_counts.copy_to_host()
        mask = out_counts > 0

        if not np.any(mask):
            return pd.DataFrame()

        return pd.DataFrame({
            'timestamp': d_out_times.copy_to_host()[mask],
            'open': d_out_opens.copy_to_host()[mask],
            'high': d_out_highs.copy_to_host()[mask],
            'low': d_out_lows.copy_to_host()[mask],
            'close': d_out_closes.copy_to_host()[mask],
            'volume': d_out_vols.copy_to_host()[mask]
        })

    def process_cpu(self, df, interval):
        # Fallback
        if df.empty: return pd.DataFrame()
        temp = df.copy()
        temp['datetime'] = pd.to_datetime(temp['timestamp'], unit='s')
        temp = temp.set_index('datetime')
        rule = f"{interval}s"
        try:
            res = temp.resample(rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()

            if res.empty:
                return pd.DataFrame()

            res = res.reset_index()
            res['timestamp'] = res['datetime'].astype('int64') // 10**9
            return res.drop(columns=['datetime'])
        except Exception as e:
            print(f"CPU Resample Error: {e}")
            return pd.DataFrame()

    def run(self):
        batch_size = self.calculate_batch_size()
        print(f"Batch Size: {batch_size}")

        if not self.input_path.exists():
             print(f"Input file {self.input_path} does not exist.")
             return

        pq_file = pq.ParquetFile(self.input_path)
        pbar = tqdm(total=pq_file.metadata.num_rows, unit='rows')

        try:
            for batch in pq_file.iter_batches(batch_size=batch_size):
                df = batch.to_pandas()

                # Normalize columns
                if 'price' in df.columns:
                    if 'close' not in df.columns: df['close'] = df['price']
                    if 'open' not in df.columns: df['open'] = df['price']
                    if 'high' not in df.columns: df['high'] = df['price']
                    if 'low' not in df.columns: df['low'] = df['price']

                if 'timestamp' in df.columns:
                    # Handle datetime64 explicitly
                    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                         df['timestamp'] = df['timestamp'].astype('int64') // 10**9
                    elif not pd.api.types.is_numeric_dtype(df['timestamp']):
                         # Attempt numeric conversion (e.g. strings)
                         df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

                    # Handle float (e.g. 1735689600.0)
                    if pd.api.types.is_float_dtype(df['timestamp']):
                        if df['timestamp'].isnull().any():
                            raise ValueError("Critical column 'timestamp' contains missing values.")
                        df['timestamp'] = df['timestamp'].astype('int64')

                df = df.sort_values('timestamp')

                # Write 1s
                self.writer_manager.write('1s', df)

                # Resample
                for res_name, interval in RESOLUTIONS.items():
                    if GPU_AVAILABLE:
                        try:
                            out_df = self.process_gpu(df, interval)
                        except Exception as e:
                            print(f"GPU Error ({res_name}): {e}")
                            out_df = self.process_cpu(df, interval)
                    else:
                        out_df = self.process_cpu(df, interval)

                    self.writer_manager.write(res_name, out_df)

                pbar.update(len(df))

                if GPU_AVAILABLE:
                    torch.cuda.empty_cache()

        finally:
            self.writer_manager.close_all()
            pbar.close()
            self.print_summary()

    def print_summary(self):
        print("\n" + "="*40)
        print("FRACTAL ATLAS GENERATION SUMMARY")
        print("="*40)
        print(f"{'Resolution':<12} | {'Files':<8} | {'Size (MB)':<12}")
        print("-" * 40)

        # Include 1s in summary
        all_res = ['1s'] + list(RESOLUTIONS.keys())

        for res in all_res:
            res_dir = self.output_path / res
            if res_dir.exists():
                files = list(res_dir.glob("*.parquet"))
                total_size = sum(f.stat().st_size for f in files) / (1024*1024)
                print(f"{res:<12} | {len(files):<8} | {total_size:.2f}")
            else:
                print(f"{res:<12} | {'0':<8} | {'0.00'}")
        print("="*40 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="DATA/glbx-mdp3-2025.parquet")
    parser.add_argument("--output", default="DATA/ATLAS")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    builder = AtlasBuilder(args.input, args.output)
    builder.run()

if __name__ == "__main__":
    main()
