"""
Fractal Discovery Agent (The Parallel Scanner)
Scans the Atlas (Historical Data) for Physics Archetypes using GPU acceleration.
Uses multiprocessing for I/O and CPU work, with serialized GPU compute.
"""
import os
import glob
import time
import asyncio
import pandas as pd
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, AsyncGenerator, Dict, Any, Optional

from core.quantum_field_engine import QuantumFieldEngine
from core.three_body_state import ThreeBodyQuantumState


@dataclass
class PatternEvent:
    pattern_type: str  # 'ROCHE_SNAP' or 'STRUCTURAL_DRIVE'
    timestamp: float
    price: float
    z_score: float
    velocity: float
    momentum: float
    coherence: float
    file_source: str
    idx: int
    state: ThreeBodyQuantumState
    window_data: Optional[pd.DataFrame] = None  # Slice of future data for simulation


# --- Standalone worker function (must be top-level for pickling) ---

def _scan_file_worker(file_path: str) -> List[PatternEvent]:
    """
    Worker: loads one parquet file, runs GPU batch compute, extracts patterns.
    Each worker creates its own QuantumFieldEngine instance.
    """
    try:
        t0 = time.perf_counter()

        # Load parquet
        df = pd.read_parquet(file_path)
        if df.empty:
            return []

        t_load = time.perf_counter()

        # Create engine per-worker (each gets its own CUDA context)
        engine = QuantumFieldEngine()

        # Batch compute states on GPU
        results = engine.batch_compute_states(df, use_cuda=True)

        t_compute = time.perf_counter()

        # Extract patterns
        detected = []
        n_bars = len(df)

        for res in results:
            state = res['state']
            bar_idx = res['bar_idx']

            # Lookahead window (~4 hours at 15s = 1000 bars)
            window_end = min(n_bars, bar_idx + 1000)
            window_slice = df.iloc[bar_idx:window_end].copy()

            if state.cascade_detected:
                detected.append(PatternEvent(
                    pattern_type='ROCHE_SNAP',
                    timestamp=state.timestamp,
                    price=state.particle_position,
                    z_score=state.z_score,
                    velocity=state.particle_velocity,
                    momentum=state.momentum_strength,
                    coherence=state.coherence,
                    file_source=file_path,
                    idx=bar_idx,
                    state=state,
                    window_data=window_slice
                ))

            if state.structure_confirmed:
                detected.append(PatternEvent(
                    pattern_type='STRUCTURAL_DRIVE',
                    timestamp=state.timestamp,
                    price=state.particle_position,
                    z_score=state.z_score,
                    velocity=state.particle_velocity,
                    momentum=state.momentum_strength,
                    coherence=state.coherence,
                    file_source=file_path,
                    idx=bar_idx,
                    state=state,
                    window_data=window_slice
                ))

        t_extract = time.perf_counter()

        fname = os.path.basename(file_path)
        roche = sum(1 for p in detected if p.pattern_type == 'ROCHE_SNAP')
        struct = sum(1 for p in detected if p.pattern_type == 'STRUCTURAL_DRIVE')
        print(
            f"    {fname}: {n_bars:,} bars | "
            f"{len(detected)} patterns (ROCHE: {roche}, STRUCT: {struct}) | "
            f"load={t_load - t0:.1f}s compute={t_compute - t_load:.1f}s extract={t_extract - t_compute:.1f}s"
        )

        return detected

    except Exception as e:
        print(f"    {os.path.basename(file_path)}: ERROR - {e}")
        return []


class FractalDiscoveryAgent:
    def __init__(self):
        # Engine kept for non-parallel use (stream_to_gpu, etc.)
        self.engine = QuantumFieldEngine()

    def scan_atlas_parallel(self, data_path: str, max_workers: int = None) -> List[PatternEvent]:
        """
        Scans data files for physics archetypes using multiprocessing.
        Each worker loads a file and runs GPU batch compute independently.
        """
        files = self._find_files(data_path)
        if not files:
            return []

        if max_workers is None:
            max_workers = min(len(files), max(1, multiprocessing.cpu_count() - 2))

        print(f"FractalDiscoveryAgent: Scanning {len(files)} files with {max_workers} workers...")
        t_start = time.perf_counter()

        all_patterns = []

        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_scan_file_worker, f): f for f in files}

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    patterns = future.result()
                    all_patterns.extend(patterns)
                except Exception as e:
                    print(f"    {os.path.basename(file_path)}: FAILED - {e}")

        elapsed = time.perf_counter() - t_start
        roche = sum(1 for p in all_patterns if p.pattern_type == 'ROCHE_SNAP')
        struct = sum(1 for p in all_patterns if p.pattern_type == 'STRUCTURAL_DRIVE')
        print(
            f"  Scan complete: {len(all_patterns)} patterns "
            f"(ROCHE: {roche}, STRUCT: {struct}) in {elapsed:.1f}s"
        )

        return all_patterns

    # --- Legacy async interface (kept for backward compat) ---

    async def scan_atlas_async(self, data_path: str) -> AsyncGenerator[PatternEvent, None]:
        """
        Async fallback — delegates to parallel scanner.
        """
        patterns = self.scan_atlas_parallel(data_path)
        for p in patterns:
            yield p

    def _find_files(self, data_path: str) -> List[str]:
        if os.path.isfile(data_path):
            print(f"  Source: single file '{data_path}'")
            return [data_path]
        elif os.path.isdir(data_path):
            files = glob.glob(os.path.join(data_path, "*.parquet"))
            files.sort()
            print(f"  Source: directory '{data_path}' -> {len(files)} parquet files")
            return files
        print(f"  WARNING: '{data_path}' not found (not a file or directory)")
        return []

    def detect_patterns(self, df: pd.DataFrame, file_path: str) -> List[PatternEvent]:
        """
        Runs QuantumFieldEngine and filters for Archetypes.
        Single-threaded version for direct use.
        """
        results = self.engine.batch_compute_states(df, use_cuda=True)

        detected = []
        n_bars = len(df)

        for res in results:
            state = res['state']
            bar_idx = res['bar_idx']

            window_end = min(n_bars, bar_idx + 1000)
            window_slice = df.iloc[bar_idx:window_end].copy()

            if state.cascade_detected:
                detected.append(PatternEvent(
                    pattern_type='ROCHE_SNAP',
                    timestamp=state.timestamp,
                    price=state.particle_position,
                    z_score=state.z_score,
                    velocity=state.particle_velocity,
                    momentum=state.momentum_strength,
                    coherence=state.coherence,
                    file_source=file_path,
                    idx=bar_idx,
                    state=state,
                    window_data=window_slice
                ))

            if state.structure_confirmed:
                detected.append(PatternEvent(
                    pattern_type='STRUCTURAL_DRIVE',
                    timestamp=state.timestamp,
                    price=state.particle_position,
                    z_score=state.z_score,
                    velocity=state.particle_velocity,
                    momentum=state.momentum_strength,
                    coherence=state.coherence,
                    file_source=file_path,
                    idx=bar_idx,
                    state=state,
                    window_data=window_slice
                ))

        return detected

    def stream_to_gpu(self, df: pd.DataFrame):
        """
        Direct interface to engine batch compute (helper)
        """
        return self.engine.batch_compute_states(df, use_cuda=True)
