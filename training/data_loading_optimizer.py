"""
Data Loading Optimizer - FIXED for Windows
Issue: Windows multiprocessing requires picklable top-level functions
"""
import os
import time
import glob
import pandas as pd
import databento as db
from typing import List
from concurrent.futures import ProcessPoolExecutor


# CRITICAL: This must be at MODULE LEVEL for Windows multiprocessing
def _load_single_dbn_file(filepath: str) -> pd.DataFrame:
    """
    Load single DBN file (must be top-level function for pickle)
    
    Args:
        filepath: Path to .dbn or .dbn.zst file
        
    Returns:
        DataFrame with data
    """
    try:
        store = db.DBNStore.from_file(filepath)
        df = store.to_df()
        
        # Convert timestamp
        df['timestamp'] = df.index.astype('int64') / 1e9
        df = df.reset_index(drop=True)
        
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


def parallel_load_dbn(dbn_files: List[str], max_workers: int = 4) -> pd.DataFrame:
    """
    Load multiple DBN files in parallel (WINDOWS COMPATIBLE)
    
    Args:
        dbn_files: List of .dbn.zst file paths
        max_workers: Number of parallel workers
        
    Returns:
        Combined DataFrame
    """
    if not dbn_files:
        raise ValueError("No files provided")
    
    print(f"\n[PARALLEL] Loading {len(dbn_files)} files with {max_workers} workers...")
    
    start = time.time()
    
    # Use ProcessPoolExecutor with top-level function
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        dataframes = list(executor.map(_load_single_dbn_file, dbn_files))
    
    load_time = time.time() - start
    print(f"[PARALLEL] Loaded in {load_time:.1f}s")
    
    # Filter out empty DataFrames (from errors)
    dataframes = [df for df in dataframes if not df.empty]
    
    if not dataframes:
        raise ValueError("No valid data loaded from files")
    
    print("[CONCAT] Combining DataFrames...", end=" ")
    start = time.time()
    combined = pd.concat(dataframes, ignore_index=True)
    concat_time = time.time() - start
    print(f"{concat_time:.1f}s")
    
    print(f"[DONE] Total: {len(combined):,} rows")
    return combined


def load_data_from_directory(directory: str, use_parallel: bool = True, 
                             max_workers: int = 4) -> pd.DataFrame:
    """
    Load all DBN files from directory
    
    Args:
        directory: Directory containing .dbn or .dbn.zst files
        use_parallel: Use parallel loading (default True)
        max_workers: Number of workers for parallel loading
        
    Returns:
        Combined DataFrame
    """
    # Find all DBN files
    dbn_files = glob.glob(os.path.join(directory, "*.dbn")) + \
                glob.glob(os.path.join(directory, "*.dbn.zst"))
    
    if not dbn_files:
        raise ValueError(f"No .dbn or .dbn.zst files found in {directory}")
    
    print(f"\nFound {len(dbn_files)} DBN files")
    
    if use_parallel and len(dbn_files) > 1:
        # Parallel loading
        return parallel_load_dbn(dbn_files, max_workers=max_workers)
    else:
        # Sequential loading (fallback)
        print("[SEQUENTIAL] Loading files one by one...")
        dataframes = []
        
        for i, filepath in enumerate(dbn_files, 1):
            print(f"[{i}/{len(dbn_files)}] {os.path.basename(filepath)}")
            df = _load_single_dbn_file(filepath)
            if not df.empty:
                dataframes.append(df)
        
        print("[CONCAT] Combining DataFrames...")
        return pd.concat(dataframes, ignore_index=True)


def convert_dbn_to_parquet(dbn_directory: str, output_directory: str):
    """
    One-time conversion: .dbn.zst → .parquet
    """
    os.makedirs(output_directory, exist_ok=True)
    
    dbn_files = glob.glob(os.path.join(dbn_directory, "*.dbn")) + \
                glob.glob(os.path.join(dbn_directory, "*.dbn.zst"))
    
    if not dbn_files:
        print(f"No DBN files found in {dbn_directory}")
        return
    
    print(f"\nConverting {len(dbn_files)} files to Parquet...")
    
    for i, dbn_file in enumerate(dbn_files, 1):
        print(f"[{i}/{len(dbn_files)}] {os.path.basename(dbn_file)}", end=" ")
        
        start = time.time()
        
        # Load DBN
        df = _load_single_dbn_file(dbn_file)
        
        if df.empty:
            print("SKIPPED (error)")
            continue
        
        # Save as parquet
        output_file = os.path.join(
            output_directory,
            os.path.basename(dbn_file).replace('.dbn.zst', '.parquet').replace('.dbn', '.parquet')
        )
        
        df.to_parquet(output_file, compression='snappy')
        
        elapsed = time.time() - start
        parquet_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"→ {parquet_size:.1f} MB ({elapsed:.1f}s)")
    
    print(f"\n✓ Conversion complete. Parquet files in: {output_directory}")


def load_parquet_from_directory(directory: str) -> pd.DataFrame:
    """
    Load all parquet files from directory (FAST)
    
    Args:
        directory: Directory containing .parquet files
        
    Returns:
        Combined DataFrame
    """
    parquet_files = glob.glob(os.path.join(directory, "*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No .parquet files found in {directory}")
    
    print(f"\nLoading {len(parquet_files)} parquet files...")
    
    start = time.time()
    dataframes = [pd.read_parquet(f) for f in parquet_files]
    load_time = time.time() - start
    
    print(f"Loaded in {load_time:.1f}s")
    
    print("Concatenating...", end=" ")
    start = time.time()
    combined = pd.concat(dataframes, ignore_index=True)
    concat_time = time.time() - start
    print(f"{concat_time:.1f}s")
    
    print(f"Total: {len(combined):,} rows")
    return combined


# Usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python data_loading_optimizer.py load <directory>")
        print("  python data_loading_optimizer.py convert <dbn_dir> <parquet_dir>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "load":
        directory = sys.argv[2]
        data = load_data_from_directory(directory, use_parallel=True, max_workers=4)
        print(f"\nLoaded {len(data):,} rows")
    
    elif command == "convert":
        dbn_dir = sys.argv[2]
        parquet_dir = sys.argv[3]
        convert_dbn_to_parquet(dbn_dir, parquet_dir)
    
    else:
        print(f"Unknown command: {command}")