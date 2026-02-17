"""
Fractal Discovery Agent (Top-Down Hierarchical Scanner)
Scans the Atlas using a FRACTAL TOP-DOWN approach:
  1. Start at the largest timeframe (macro patterns)
  2. Each macro pattern's time window defines WHERE to look in the next smaller TF
  3. Drill down recursively until reaching the finest resolution (1s)

GPU Strategy: Single-engine batch compute.
  - Load files with ThreadPool (I/O parallelism)
  - Concatenate into one DataFrame
  - Run ONE batch_compute_states on the GPU (full utilization)
  - Split results back by file origin
"""
import os
import glob
import time
import pandas as pd
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from core.quantum_field_engine import QuantumFieldEngine
from core.three_body_state import ThreeBodyQuantumState


# Timeframe hierarchy: largest to smallest (top-down scan order)
TIMEFRAME_HIERARCHY = ['1D', '4h', '1h', '15m', '5m', '1m', '15s', '5s', '1s']

# Maps timeframe labels to seconds
TIMEFRAME_SECONDS = {
    '1s': 1, '5s': 5, '15s': 15, '1m': 60, '5m': 300,
    '15m': 900, '1h': 3600, '4h': 14400, '1D': 86400, '1W': 604800
}

MIN_BARS_FOR_CHILD_REGRESSION = 30

@dataclass
class PatternEvent:
    pattern_type: str       # 'ROCHE_SNAP' or 'STRUCTURAL_DRIVE'
    timestamp: float
    price: float
    z_score: float
    velocity: float
    momentum: float
    coherence: float
    file_source: str
    idx: int
    state: ThreeBodyQuantumState
    timeframe: str = '15s'
    depth: int = 0          # 0 = macro (top TF), 1 = next level down, etc.
    parent_type: str = ''   # Parent pattern type (e.g. 'ROCHE_SNAP' or 'STRUCTURAL_DRIVE')
    parent_tf: str = ''     # Parent's timeframe (e.g. '1D')
    window_data: Optional[pd.DataFrame] = None
    parent_chain: Optional[List[Dict]] = None  # Full ancestry chain


# ---------------------------------------------------------------------------
# I/O helper (top-level for ThreadPool pickling)
# ---------------------------------------------------------------------------

def _load_parquet(file_path: str) -> Tuple[str, pd.DataFrame]:
    """Load a parquet file. Returns (path, df)."""
    try:
        df = pd.read_parquet(file_path)
        return (file_path, df)
    except Exception as e:
        print(f"    WARNING: Failed to load {os.path.basename(file_path)}: {e}")
        return (file_path, pd.DataFrame())


# ---------------------------------------------------------------------------
# Main Agent
# ---------------------------------------------------------------------------

class FractalDiscoveryAgent:
    def __init__(self):
        # Single GPU engine — all compute goes through this
        self.engine = QuantumFieldEngine()

    def scan_atlas_topdown(self, atlas_root: str, timeframes: List[str] = None,
                           max_workers: int = None,
                           on_level_complete=None,
                           resume_manifest: List[PatternEvent] = None,
                           resume_levels: List[str] = None) -> List[PatternEvent]:
        """
        TOP-DOWN FRACTAL DISCOVERY (Single-Engine GPU)

        Starts at the largest available timeframe, finds macro patterns,
        then drills into each pattern's time window at the next smaller TF.
        Repeats until reaching the finest TF.

        GPU compute is serialized through a single QuantumFieldEngine to
        maximize GPU utilization (one large batch instead of many tiny ones).
        """
        # Determine available timeframes (sorted large -> small)
        if timeframes is None:
            timeframes = []
            for tf in TIMEFRAME_HIERARCHY:
                tf_path = os.path.join(atlas_root, tf)
                if os.path.isdir(tf_path):
                    timeframes.append(tf)

        if not timeframes:
            print(f"WARNING: No valid timeframe directories found in {atlas_root}")
            return []

        # I/O thread count (not GPU workers)
        io_workers = max(1, multiprocessing.cpu_count() - 2) if max_workers is None else max_workers

        # Resume support
        completed_levels = set(resume_levels or [])
        all_patterns = list(resume_manifest or [])

        if completed_levels:
            print(f"\n  RESUMING: {len(completed_levels)} levels already done, "
                  f"{len(all_patterns)} patterns cached")

        print(f"\n{'='*70}")
        print(f"  FRACTAL TOP-DOWN DISCOVERY")
        print(f"  ATLAS: {atlas_root}")
        print(f"  Hierarchy: {' -> '.join(timeframes)} ({len(timeframes)} levels)")
        print(f"  I/O threads: {io_workers} | GPU: single engine")
        if completed_levels:
            remaining = [tf for tf in timeframes if tf not in completed_levels]
            print(f"  Remaining: {' -> '.join(remaining) if remaining else 'ALL DONE'}")
        print(f"{'='*70}")

        t_total = time.perf_counter()
        current_windows = None  # None = scan everything (Level 0)

        for level, tf in enumerate(timeframes):
            # Skip already-completed levels but rebuild windows for continuity
            if tf in completed_levels:
                level_patterns = [p for p in all_patterns if p.timeframe == tf]
                if level_patterns:
                    tf_secs = TIMEFRAME_SECONDS.get(tf, 15)
                    drilldown_secs = tf_secs
                    current_windows = []
                    for p in level_patterns:
                        # Rebuild chain entry for resumed patterns
                        full_chain = self._build_parent_chain(p)

                        current_windows.append((p.timestamp, p.timestamp + drilldown_secs,
                                                p.pattern_type, tf, full_chain))
                    current_windows = self._merge_windows(current_windows)
                    print(f"  Level {level} [{tf}] SKIPPED (cached: {len(level_patterns)} patterns, "
                          f"{len(current_windows)} windows)")
                continue

            tf_path = os.path.join(atlas_root, tf)
            files = self._find_files(tf_path)
            if not files:
                print(f"\n  Level {level} [{tf}]: no files, skipping")
                continue

            t_level = time.perf_counter()
            tf_secs = TIMEFRAME_SECONDS.get(tf, 15)

            if current_windows is None:
                # --- LEVEL 0: Full scan (macro patterns) ---
                print(f"\n  Level {level} [{tf}] MACRO: scanning {len(files)} files (full scan)...")
                level_patterns = self._batch_scan_full(files, tf, level, io_workers)
            else:
                # --- LEVEL 1+: Windowed scan (drill into parent bodies) ---
                n_windows = len(current_windows)
                total_window_secs = sum(w[1] - w[0] for w in current_windows)
                print(
                    f"\n  Level {level} [{tf}] DRILL-DOWN: "
                    f"scanning {len(files)} files within {n_windows} parent windows "
                    f"({total_window_secs/3600:.1f}h total coverage)..."
                )
                level_patterns = self._batch_scan_windowed(
                    files, tf, level, current_windows, io_workers
                )

            all_patterns.extend(level_patterns)
            completed_levels.add(tf)

            level_elapsed = time.perf_counter() - t_level
            roche = sum(1 for p in level_patterns if p.pattern_type == 'ROCHE_SNAP')
            struct = sum(1 for p in level_patterns if p.pattern_type == 'STRUCTURAL_DRIVE')
            print(
                f"  Level {level} [{tf}] complete: {len(level_patterns)} patterns "
                f"(R:{roche} S:{struct}) in {level_elapsed:.1f}s"
            )

            # Checkpoint callback
            if on_level_complete:
                on_level_complete(level, tf, all_patterns, list(completed_levels))

            if not level_patterns:
                print(f"  No patterns at [{tf}] — stopping drill-down.")
                break

            # Build time windows for next level
            child_tf_idx = TIMEFRAME_HIERARCHY.index(tf) + 1
            if child_tf_idx < len(timeframes):
                child_tf = timeframes[child_tf_idx]
                child_tf_secs = TIMEFRAME_SECONDS.get(child_tf, 15)
                min_bars_needed = MIN_BARS_FOR_CHILD_REGRESSION  # regression_period(21) + some margin
                min_window_secs = child_tf_secs * min_bars_needed
                drilldown_secs = max(tf_secs, min_window_secs)
            else:
                drilldown_secs = tf_secs

            current_windows = []
            for p in level_patterns:
                w_start = p.timestamp
                w_end = p.timestamp + drilldown_secs

                # Build star schema parent chain
                full_chain = self._build_parent_chain(p)

                current_windows.append((w_start, w_end, p.pattern_type, tf, full_chain))

            current_windows = self._merge_windows(current_windows)
            print(f"  -> {len(current_windows)} merged windows for next level")

        elapsed = time.perf_counter() - t_total

        # Final summary
        from collections import Counter
        print(f"\n{'='*70}")
        print(f"  FRACTAL DISCOVERY COMPLETE: {len(all_patterns)} total patterns in {elapsed:.1f}s")
        tf_counts = Counter(p.timeframe for p in all_patterns)
        depth_counts = Counter(p.depth for p in all_patterns)
        print(f"  By timeframe:")
        for tf in timeframes:
            count = tf_counts.get(tf, 0)
            if count > 0:
                print(f"    [{tf:>4s}] {count:>7,} patterns")
        print(f"  By depth:")
        for d in sorted(depth_counts.keys()):
            print(f"    depth {d}: {depth_counts[d]:>7,} patterns")
        print(f"{'='*70}")

        return all_patterns

    def scan_day_cascade(self, atlas_root: str, date_str: str) -> List[PatternEvent]:
        """
        Scans a single day top-down (4H -> 15s) to find actionable patterns.
        Used for Forward Pass execution.
        """
        # We start at 4H.
        current_windows = None

        # Define hierarchy subset
        hierarchy = ['4h', '1h', '15m', '5m', '1m', '15s']

        for level, tf in enumerate(hierarchy):
            tf_path = os.path.join(atlas_root, tf)
            # Find file for this date
            files = sorted(glob.glob(os.path.join(tf_path, '*.parquet')))
            day_file = None
            for f in files:
                if date_str in os.path.basename(f):
                    day_file = f
                    break

            if not day_file:
                # If macro data missing, can't start cascade
                return []

            # Scan
            if current_windows is None:
                 # Level 0 (4H) - scan full file
                 patterns = self._batch_scan_full([day_file], tf, level, 1)
            else:
                 # Drill down
                 patterns = self._batch_scan_windowed([day_file], tf, level, current_windows, 1)

            if not patterns:
                return []

            # If we are at 15s, return these
            if tf == '15s':
                return patterns

            # Otherwise build windows for next level
            tf_secs = TIMEFRAME_SECONDS.get(tf, 15)
            child_tf = hierarchy[level+1]
            child_tf_secs = TIMEFRAME_SECONDS.get(child_tf, 15)
            min_window = child_tf_secs * 30
            drilldown = max(tf_secs, min_window)

            next_windows = []
            for p in patterns:
                full_chain = self._build_parent_chain(p)
                next_windows.append((p.timestamp, p.timestamp + drilldown, p.pattern_type, tf, full_chain))

            current_windows = self._merge_windows(next_windows)

        return []

    # ------------------------------------------------------------------
    # Batch scan methods (single GPU engine, threaded I/O)
    # ------------------------------------------------------------------

    def _batch_scan_full(self, files: List[str], timeframe: str, depth: int,
                         io_workers: int) -> List[PatternEvent]:
        """
        Load all files (threaded I/O), concatenate, run ONE GPU batch,
        then extract patterns with per-file tracking.
        """
        t0 = time.perf_counter()

        # 1. Parallel I/O load
        file_dfs = self._load_files_threaded(files, io_workers)
        if not file_dfs:
            return []

        t_io = time.perf_counter()

        # 2. Concatenate with file origin tracking
        dfs = []
        file_boundaries = []  # (start_idx, end_idx, file_path)
        offset = 0
        for fpath, df in file_dfs:
            if df.empty:
                continue
            n = len(df)
            dfs.append(df)
            file_boundaries.append((offset, offset + n, fpath))
            offset += n

        if not dfs:
            return []

        combined = pd.concat(dfs, ignore_index=True)
        total_bars = len(combined)
        print(f"    Loaded {total_bars:,} bars from {len(file_boundaries)} files in {t_io - t0:.1f}s")

        # 3. Single GPU batch compute
        t_gpu = time.perf_counter()
        results = self.engine.batch_compute_states(combined, use_cuda=True)
        t_gpu_done = time.perf_counter()
        print(f"    GPU compute: {len(results)} states in {t_gpu_done - t_gpu:.1f}s")

        # 4. Extract patterns
        tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 15)
        lookahead_bars = max(50, int(4 * 3600 / tf_seconds))
        detected = []

        for res in results:
            state = res['state']
            bar_idx = res['bar_idx']

            # Find which file this bar belongs to
            file_path = ''
            for (start, end, fpath) in file_boundaries:
                if start <= bar_idx < end:
                    file_path = fpath
                    break

            window_end = min(total_bars, bar_idx + lookahead_bars)
            window_slice = combined.iloc[bar_idx:window_end].copy()

            if state.cascade_detected:
                detected.append(PatternEvent(
                    pattern_type='ROCHE_SNAP', timestamp=state.timestamp,
                    price=state.particle_position, z_score=state.z_score,
                    velocity=state.particle_velocity, momentum=state.momentum_strength,
                    coherence=state.coherence, file_source=file_path, idx=bar_idx,
                    state=state, timeframe=timeframe, depth=depth,
                    parent_type='', parent_tf='', window_data=window_slice
                ))

            if state.structure_confirmed:
                detected.append(PatternEvent(
                    pattern_type='STRUCTURAL_DRIVE', timestamp=state.timestamp,
                    price=state.particle_position, z_score=state.z_score,
                    velocity=state.particle_velocity, momentum=state.momentum_strength,
                    coherence=state.coherence, file_source=file_path, idx=bar_idx,
                    state=state, timeframe=timeframe, depth=depth,
                    parent_type='', parent_tf='', window_data=window_slice
                ))

        # Per-file summary
        from collections import Counter
        file_counts = Counter(os.path.basename(p.file_source) for p in detected)
        for fname, count in sorted(file_counts.items()):
            r = sum(1 for p in detected if os.path.basename(p.file_source) == fname and p.pattern_type == 'ROCHE_SNAP')
            s = count - r
            print(f"    [{timeframe}] {fname}: R:{r} S:{s}")

        return detected

    def _batch_scan_windowed(self, files: List[str], timeframe: str, depth: int,
                              windows: List[Tuple], io_workers: int) -> List[PatternEvent]:
        """
        Load all files (threaded I/O), filter to parent windows,
        concatenate filtered bars, run ONE GPU batch.
        """
        t0 = time.perf_counter()

        # 1. Parallel I/O load
        file_dfs = self._load_files_threaded(files, io_workers)
        if not file_dfs:
            return []

        t_io = time.perf_counter()

        # 2. Filter each file to parent windows and concatenate
        filtered_dfs = []
        filtered_parent_types = []
        filtered_parent_tfs = []
        filtered_parent_chains = []
        filtered_file_sources = []
        total_raw = 0
        total_filtered = 0

        for fpath, df in file_dfs:
            if df.empty:
                continue

            total_raw += len(df)

            # Get timestamps as float seconds
            if 'timestamp' not in df.columns:
                continue
            ts_col = df['timestamp']
            if hasattr(ts_col.iloc[0], 'timestamp'):
                ts_seconds = ts_col.apply(lambda x: x.timestamp()).values
            else:
                ts_seconds = ts_col.values.astype(float)

            # Build mask: bar inside ANY parent window
            mask = np.zeros(len(df), dtype=bool)
            bar_parent_type = np.full(len(df), '', dtype=object)
            bar_parent_tf = np.full(len(df), '', dtype=object)
            bar_parent_chain = np.empty(len(df), dtype=object)

            for (w_start, w_end, p_type, p_tf, p_chain) in windows:
                in_window = (ts_seconds >= w_start) & (ts_seconds <= w_end)
                new_bars = in_window & ~mask
                bar_parent_type[new_bars] = p_type
                bar_parent_tf[new_bars] = p_tf
                # Assign the list object to each matching index without broadcasting
                bar_parent_chain[new_bars] = [p_chain] * int(np.sum(new_bars))
                mask |= in_window

            n_in = int(np.sum(mask))
            if n_in == 0:
                continue

            total_filtered += n_in
            df_filtered = df.loc[mask].reset_index(drop=True)
            filtered_dfs.append(df_filtered)
            filtered_parent_types.extend(bar_parent_type[mask].tolist())
            filtered_parent_tfs.extend(bar_parent_tf[mask].tolist())
            filtered_parent_chains.extend(bar_parent_chain[mask].tolist())
            filtered_file_sources.extend([fpath] * n_in)

        if not filtered_dfs:
            print(f"    No bars in window ({total_raw:,} raw bars checked)")
            return []

        combined = pd.concat(filtered_dfs, ignore_index=True)
        print(f"    Filtered {total_filtered:,}/{total_raw:,} bars in window from {len(filtered_dfs)} files ({t_io - t0:.1f}s I/O)")

        # 3. Single GPU batch compute
        t_gpu = time.perf_counter()
        results = self.engine.batch_compute_states(combined, use_cuda=True)
        t_gpu_done = time.perf_counter()
        print(f"    GPU compute: {len(results)} states in {t_gpu_done - t_gpu:.1f}s")

        # 4. Extract patterns
        tf_seconds_val = TIMEFRAME_SECONDS.get(timeframe, 15)
        lookahead_bars = max(50, int(4 * 3600 / tf_seconds_val))
        detected = []
        n_bars = len(combined)

        for res in results:
            state = res['state']
            bar_idx = res['bar_idx']

            p_type = filtered_parent_types[bar_idx] if bar_idx < len(filtered_parent_types) else ''
            p_tf = filtered_parent_tfs[bar_idx] if bar_idx < len(filtered_parent_tfs) else ''
            p_chain = filtered_parent_chains[bar_idx] if bar_idx < len(filtered_parent_chains) else []
            file_path = filtered_file_sources[bar_idx] if bar_idx < len(filtered_file_sources) else ''

            window_end = min(n_bars, bar_idx + lookahead_bars)
            window_slice = combined.iloc[bar_idx:window_end].copy()

            if state.cascade_detected:
                detected.append(PatternEvent(
                    pattern_type='ROCHE_SNAP', timestamp=state.timestamp,
                    price=state.particle_position, z_score=state.z_score,
                    velocity=state.particle_velocity, momentum=state.momentum_strength,
                    coherence=state.coherence, file_source=file_path, idx=bar_idx,
                    state=state, timeframe=timeframe, depth=depth,
                    parent_type=p_type, parent_tf=p_tf, window_data=window_slice,
                    parent_chain=p_chain
                ))

            if state.structure_confirmed:
                detected.append(PatternEvent(
                    pattern_type='STRUCTURAL_DRIVE', timestamp=state.timestamp,
                    price=state.particle_position, z_score=state.z_score,
                    velocity=state.particle_velocity, momentum=state.momentum_strength,
                    coherence=state.coherence, file_source=file_path, idx=bar_idx,
                    state=state, timeframe=timeframe, depth=depth,
                    parent_type=p_type, parent_tf=p_tf, window_data=window_slice,
                    parent_chain=p_chain
                ))

        roche = sum(1 for p in detected if p.pattern_type == 'ROCHE_SNAP')
        struct = sum(1 for p in detected if p.pattern_type == 'STRUCTURAL_DRIVE')
        print(f"    Extracted {len(detected)} patterns (R:{roche} S:{struct})")

        return detected

    def _build_parent_chain(self, p: PatternEvent) -> List[Dict]:
        """Builds the full parent chain for a given pattern."""
        chain_entry = {
            'tf': p.timeframe,
            'type': p.pattern_type,
            'z': p.z_score,
            'mom': p.momentum,
            'coh': p.coherence,
            'timestamp': p.timestamp
        }
        return [chain_entry] + (p.parent_chain or [])

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_files_threaded(self, files: List[str], max_workers: int) -> List[Tuple[str, pd.DataFrame]]:
        """Load multiple parquet files in parallel using threads (I/O bound)."""
        results = []
        n_workers = min(len(files), max_workers)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_load_parquet, f): f for f in files}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    fpath = futures[future]
                    print(f"    {os.path.basename(fpath)}: LOAD FAILED - {e}")

        # Sort by filename for deterministic ordering
        results.sort(key=lambda x: x[0])
        return results

    def _find_files(self, data_path: str) -> List[str]:
        if os.path.isfile(data_path):
            return [data_path]
        elif os.path.isdir(data_path):
            files = glob.glob(os.path.join(data_path, "*.parquet"))
            files.sort()
            return files
        return []

    @staticmethod
    def _merge_windows(windows: List[Tuple]) -> List[Tuple]:
        """
        Merge overlapping time windows. Keeps the dominant parent_type.
        Input:  [(start, end, parent_type, parent_tf, parent_chain), ...]
        Output: [(start, end, parent_type, parent_tf, parent_chain), ...] merged
        """
        if not windows:
            return []

        sorted_wins = sorted(windows, key=lambda w: w[0])
        merged = [sorted_wins[0]]

        for w in sorted_wins[1:]:
            prev = merged[-1]
            if w[0] <= prev[1]:
                new_end = max(prev[1], w[1])

                # Check tuple length to support both 4-element (legacy) and 5-element (star schema)
                # But since we updated scan_atlas_topdown, we expect 5.
                # Be safe: check if w has chain
                w_has_chain = len(w) > 4
                prev_has_chain = len(prev) > 4

                if (w[1] - w[0]) > (prev[1] - prev[0]):
                    # Dominant is W
                    chain = w[4] if w_has_chain else (prev[4] if prev_has_chain else [])
                    merged[-1] = (prev[0], new_end, w[2], w[3], chain)
                else:
                    # Dominant is Prev
                    chain = prev[4] if prev_has_chain else (w[4] if w_has_chain else [])
                    merged[-1] = (prev[0], new_end, prev[2], prev[3], chain)
            else:
                merged.append(w)

        return merged

    # ------------------------------------------------------------------
    # Direct-use methods (kept for backward compat)
    # ------------------------------------------------------------------

    def scan_atlas_parallel(self, data_path: str, timeframe: str = '15s',
                            max_workers: int = None) -> List[PatternEvent]:
        """Single-timeframe batch scan (flat, no hierarchy)."""
        files = self._find_files(data_path)
        if not files:
            return []

        io_workers = min(len(files), max(1, multiprocessing.cpu_count() - 2)) if max_workers is None else max_workers
        print(f"  Scanning {len(files)} [{timeframe}] files...")

        patterns = self._batch_scan_full(files, timeframe, depth=0, io_workers=io_workers)
        print(f"  [{timeframe}] {len(patterns)} patterns")
        return patterns

    def detect_patterns(self, df: pd.DataFrame, file_path: str,
                        timeframe: str = '15s') -> List[PatternEvent]:
        """Single-threaded pattern detection (for direct use)."""
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
                    pattern_type='ROCHE_SNAP', timestamp=state.timestamp,
                    price=state.particle_position, z_score=state.z_score,
                    velocity=state.particle_velocity, momentum=state.momentum_strength,
                    coherence=state.coherence, file_source=file_path, idx=bar_idx,
                    state=state, timeframe=timeframe, window_data=window_slice
                ))

            if state.structure_confirmed:
                detected.append(PatternEvent(
                    pattern_type='STRUCTURAL_DRIVE', timestamp=state.timestamp,
                    price=state.particle_position, z_score=state.z_score,
                    velocity=state.particle_velocity, momentum=state.momentum_strength,
                    coherence=state.coherence, file_source=file_path, idx=bar_idx,
                    state=state, timeframe=timeframe, window_data=window_slice
                ))

        return detected

    def stream_to_gpu(self, df: pd.DataFrame):
        """Direct interface to engine batch compute."""
        return self.engine.batch_compute_states(df, use_cuda=True)
