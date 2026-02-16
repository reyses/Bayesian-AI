"""
Fractal Discovery Agent (The Async Scanner)
Scans the Atlas (Historical Data) for Physics Archetypes using GPU acceleration.
"""
import os
import glob
import asyncio
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, AsyncGenerator, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

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
    window_data: Optional[pd.DataFrame] = None # Slice of future data for simulation

class FractalDiscoveryAgent:
    def __init__(self):
        # Initialize Engine (asserts CUDA availability)
        self.engine = QuantumFieldEngine()
        self.executor = ThreadPoolExecutor(max_workers=4) # For IO operations

    async def scan_atlas_async(self, data_path: str) -> AsyncGenerator[PatternEvent, None]:
        """
        Asynchronously scans data files for physics archetypes.
        """
        files = self._find_files(data_path)
        print(f"FractalDiscoveryAgent: Found {len(files)} files to scan in '{data_path}'")
        total_patterns = 0

        for file_idx, file_path in enumerate(files):
            fname = os.path.basename(file_path)
            # Offload IO to thread
            try:
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(self.executor, pd.read_parquet, file_path)

                if df.empty:
                    print(f"  [{file_idx+1}/{len(files)}] {fname}: EMPTY - skipped")
                    continue

                print(f"  [{file_idx+1}/{len(files)}] {fname}: {len(df):,} bars loaded, computing states...", end="", flush=True)

                # Process on GPU
                patterns = self.detect_patterns(df, file_path)
                total_patterns += len(patterns)
                roche = sum(1 for p in patterns if p.pattern_type == 'ROCHE_SNAP')
                struct = sum(1 for p in patterns if p.pattern_type == 'STRUCTURAL_DRIVE')
                print(f" -> {len(patterns)} patterns (ROCHE: {roche}, STRUCT: {struct}) [total: {total_patterns}]")

                for p in patterns:
                    yield p

            except Exception as e:
                print(f"  [{file_idx+1}/{len(files)}] {fname}: ERROR - {e}")

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
        """
        # Batch compute states (Runs on GPU)
        results = self.engine.batch_compute_states(df, use_cuda=True)

        detected = []
        n_bars = len(df)

        for res in results:
            state = res['state']
            bar_idx = res['bar_idx']

            # Extract Window (e.g. 1000 bars lookahead for simulation)
            # We copy it to ensure it persists if df is released (though in generator df persists until loop moves)
            # Slicing keeps reference.
            # Using 1000 bars ~ 4 hours at 15s.
            window_end = min(n_bars, bar_idx + 1000)
            window_slice = df.iloc[bar_idx : window_end].copy()

            if state.cascade_detected:
                # Roche Snap
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
                # Structural Drive
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
