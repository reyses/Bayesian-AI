"""
Data Loading Performance Profiler
Diagnoses bottlenecks in full learn cycle data loading
"""
import time
import os
from typing import Dict, List
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    """Context manager to time code blocks"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[TIMER] {label}: {elapsed:.2f}s")


class DataLoadingProfiler:
    """Profile data loading performance"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
    
    def profile_full_load(self, data_directory: str, file_pattern: str = "*.dbn.zst"):
        """
        Profile complete data loading pipeline
        
        Args:
            data_directory: Path to data files
            file_pattern: Glob pattern for files
        """
        import glob
        
        print("\n" + "="*80)
        print("DATA LOADING PERFORMANCE PROFILE")
        print("="*80)
        
        # Find files
        with timer("File Discovery"):
            file_paths = glob.glob(os.path.join(data_directory, file_pattern))
            print(f"  Found {len(file_paths)} files")
        
        if not file_paths:
            print(f"⚠️  No files found matching {file_pattern}")
            return
        
        # Profile first file (representative)
        test_file = file_paths[0]
        print(f"\nProfiling: {os.path.basename(test_file)}")
        print("-" * 80)
        
        # Step 1: File I/O
        with timer("1. File Read (I/O)"):
            file_size_mb = os.path.getsize(test_file) / (1024 * 1024)
            print(f"  File size: {file_size_mb:.2f} MB")
            
            with open(test_file, 'rb') as f:
                raw_data = f.read()
        
        # Step 2: Decompression (if .zst)
        if test_file.endswith('.zst'):
            with timer("2. Decompression (.zst)"):
                try:
                    import zstandard as zstd
                    dctx = zstd.ZstdDecompressor()
                    decompressed = dctx.decompress(raw_data)
                    decompressed_mb = len(decompressed) / (1024 * 1024)
                    print(f"  Decompressed: {decompressed_mb:.2f} MB")
                    print(f"  Compression ratio: {decompressed_mb/file_size_mb:.1f}x")
                except ImportError:
                    print("  ⚠️  zstandard not installed, skipping")
        
        # Step 3: Databento parsing
        with timer("3. Databento Parsing"):
            try:
                import databento as db
                store = db.DBNStore.from_file(test_file)
                df = store.to_df()
                print(f"  Rows: {len(df):,}")
                print(f"  Columns: {list(df.columns)}")
            except Exception as e:
                print(f"  ⚠️  Databento parse failed: {e}")
        
        # Step 4: DataFrame operations
        with timer("4. DataFrame Processing"):
            # Typical operations
            df['timestamp'] = df.index.astype('int64') / 1e9  # Convert to seconds
            df = df.reset_index(drop=True)
            print(f"  Processed rows: {len(df):,}")
        
        print("\n" + "="*80)
        print("BOTTLENECK ANALYSIS")
        print("="*80)
        
        # Estimate full load time
        total_time_per_file = sum([
            self._get_last_timing("File Read"),
            self._get_last_timing("Decompression"),
            self._get_last_timing("Databento Parsing"),
            self._get_last_timing("DataFrame Processing")
        ])
        
        estimated_total = total_time_per_file * len(file_paths)
        
        print(f"Time per file: {total_time_per_file:.2f}s")
        print(f"Total files: {len(file_paths)}")
        print(f"Estimated full load: {estimated_total:.1f}s ({estimated_total/60:.1f} minutes)")
        
        # Identify bottleneck
        slowest_step = max([
            ("File I/O", self._get_last_timing("File Read")),
            ("Decompression", self._get_last_timing("Decompression")),
            ("Databento Parse", self._get_last_timing("Databento Parsing")),
            ("DataFrame Ops", self._get_last_timing("DataFrame Processing"))
        ], key=lambda x: x[1])
        
        print(f"\n🔴 BOTTLENECK: {slowest_step[0]} ({slowest_step[1]:.2f}s per file)")
        
        print("\n" + "="*80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*80)
        
        self._print_recommendations(slowest_step[0], len(file_paths), file_size_mb)
    
    def _get_last_timing(self, label: str) -> float:
        """Get last recorded timing for a label"""
        if label not in self.timings:
            return 0.0
        return self.timings[label][-1] if self.timings[label] else 0.0
    
    def _print_recommendations(self, bottleneck: str, num_files: int, file_size_mb: float):
        """Print optimization recommendations"""
        
        if bottleneck == "File I/O":
            print("File I/O is the bottleneck:")
            print("  ✓ Use SSD instead of HDD")
            print("  ✓ Load files in parallel (multiprocessing)")
            print("  ✓ Pre-cache frequently used files")
            print("  ✓ Consider storing decompressed data")
        
        elif bottleneck == "Decompression":
            print("Decompression is the bottleneck:")
            print("  ✓ Decompress files once, cache as .parquet")
            print("  ✓ Use parallel decompression (multiple cores)")
            print("  ✓ Consider storing uncompressed data")
        
        elif bottleneck == "Databento Parse":
            print("Databento parsing is the bottleneck:")
            print("  ✓ Convert to .parquet once (much faster to load)")
            print("  ✓ Use databento's native methods (avoid conversion)")
            print("  ✓ Filter data during parse (reduce DataFrame size)")
        
        elif bottleneck == "DataFrame Ops":
            print("DataFrame operations are the bottleneck:")
            print("  ✓ Optimize timestamp conversion")
            print("  ✓ Use categorical types for repeated strings")
            print("  ✓ Reduce unnecessary operations")
        
        print(f"\n💡 QUICK WIN: If loading {num_files} files repeatedly:")
        print(f"   1. Convert to .parquet once (fast load)")
        print(f"   2. Cache in memory if total < 16GB")
        print(f"   3. Use memory-mapped files for large datasets")


# Optimization Solutions

def convert_dbn_to_parquet(dbn_directory: str, output_directory: str):
    """
    One-time conversion: .dbn.zst → .parquet
    Parquet loads 10-100x faster
    """
    import glob
    import databento as db
    import pandas as pd
    from pathlib import Path
    
    os.makedirs(output_directory, exist_ok=True)
    
    dbn_files = glob.glob(os.path.join(dbn_directory, "*.dbn.zst"))
    
    print(f"\nConverting {len(dbn_files)} files to Parquet...")
    
    for i, dbn_file in enumerate(dbn_files, 1):
        print(f"[{i}/{len(dbn_files)}] {os.path.basename(dbn_file)}", end=" ")
        
        with timer(""):
            # Load DBN
            store = db.DBNStore.from_file(dbn_file)
            df = store.to_df()
            
            # Convert timestamp
            df['timestamp'] = df.index.astype('int64') / 1e9
            df = df.reset_index(drop=True)
            
            # Save as parquet
            output_file = os.path.join(
                output_directory,
                os.path.basename(dbn_file).replace('.dbn.zst', '.parquet')
            )
            
            df.to_parquet(output_file, compression='snappy')
            
            parquet_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"→ {parquet_size:.1f} MB")
    
    print(f"\n✓ Conversion complete. Parquet files in: {output_directory}")
    print(f"  Next time, load from .parquet (10-100x faster)")


def load_parquet_cached(parquet_files: List[str], cache_memory: bool = True):
    """
    Load parquet with optional memory caching
    
    Args:
        parquet_files: List of .parquet file paths
        cache_memory: If True, cache in RAM (faster for repeated loads)
    """
    import pandas as pd
    
    _cache = {}
    
    def load_file(filepath: str) -> pd.DataFrame:
        if cache_memory and filepath in _cache:
            print(f"[CACHE HIT] {os.path.basename(filepath)}")
            return _cache[filepath].copy()
        
        print(f"[LOADING] {os.path.basename(filepath)}", end=" ")
        with timer(""):
            df = pd.read_parquet(filepath)
        
        if cache_memory:
            _cache[filepath] = df
        
        return df
    
    # Load all files
    dataframes = []
    for filepath in parquet_files:
        dataframes.append(load_file(filepath))
    
    # Concatenate
    print("\n[CONCAT] Combining DataFrames...", end=" ")
    with timer(""):
        combined = pd.concat(dataframes, ignore_index=True)
    
    print(f"Total rows: {len(combined):,}")
    return combined


def parallel_load_dbn(dbn_files: List[str], max_workers: int = 4):
    """
    Load multiple DBN files in parallel
    
    Args:
        dbn_files: List of .dbn.zst file paths
        max_workers: Number of parallel workers
    """
    from concurrent.futures import ProcessPoolExecutor
    import databento as db
    import pandas as pd
    
    def load_single_file(filepath: str) -> pd.DataFrame:
        """Load single DBN file"""
        store = db.DBNStore.from_file(filepath)
        df = store.to_df()
        df['timestamp'] = df.index.astype('int64') / 1e9
        df = df.reset_index(drop=True)
        return df
    
    print(f"\n[PARALLEL] Loading {len(dbn_files)} files with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with timer("Parallel Load"):
            dataframes = list(executor.map(load_single_file, dbn_files))
    
    print("[CONCAT] Combining DataFrames...", end=" ")
    with timer(""):
        combined = pd.concat(dataframes, ignore_index=True)
    
    print(f"Total rows: {len(combined):,}")
    return combined


# Usage Examples
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python data_loading_optimizer.py profile /path/to/data")
        print("  python data_loading_optimizer.py convert /path/to/dbn /path/to/parquet")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "profile":
        data_dir = sys.argv[2]
        profiler = DataLoadingProfiler()
        profiler.profile_full_load(data_dir)
    
    elif command == "convert":
        dbn_dir = sys.argv[2]
        parquet_dir = sys.argv[3]
        convert_dbn_to_parquet(dbn_dir, parquet_dir)
    
    else:
        print(f"Unknown command: {command}")
