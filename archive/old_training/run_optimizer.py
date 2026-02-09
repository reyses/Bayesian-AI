"""
Standalone Runner for Data Loading Optimizer
Quick commands to run preprocessing tasks
"""
import sys
import os

# Add project to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level
sys.path.insert(0, project_root)

# Try to import from training directory or current directory
try:
    from training.data_loading_optimizer import (
        load_data_from_directory,
        convert_dbn_to_parquet,
        load_parquet_from_directory
    )
except ModuleNotFoundError:
    # Running from inside training directory
    from data_loading_optimizer import (
        load_data_from_directory,
        convert_dbn_to_parquet,
        load_parquet_from_directory
    )


def main():
    print("\n" + "="*80)
    print("DATA LOADING OPTIMIZER - Standalone Runner")
    print("="*80 + "\n")
    
    if len(sys.argv) < 2:
        print("USAGE:\n")
        print("1. Load DBN files (parallel):")
        print("   python run_optimizer.py load DATA/RAW\n")
        
        print("2. Load DBN files (sequential, no parallel):")
        print("   python run_optimizer.py load DATA/RAW --no-parallel\n")
        
        print("3. Convert DBN to Parquet (one-time):")
        print("   python run_optimizer.py convert DATA/RAW DATA/PARQUET\n")
        
        print("4. Load Parquet files (fast):")
        print("   python run_optimizer.py load-parquet DATA/PARQUET\n")
        
        print("5. Convert then load:")
        print("   python run_optimizer.py convert-and-load DATA/RAW DATA/PARQUET\n")
        
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    try:
        if command == "load":
            # Load DBN files
            if len(sys.argv) < 3:
                print("ERROR: Missing directory argument")
                print("Usage: python run_optimizer.py load DATA/RAW")
                sys.exit(1)
            
            directory = sys.argv[2]
            use_parallel = "--no-parallel" not in sys.argv
            max_workers = 4
            
            # Check for custom worker count
            if "--workers" in sys.argv:
                idx = sys.argv.index("--workers")
                max_workers = int(sys.argv[idx + 1])
            
            print(f"Loading from: {directory}")
            print(f"Parallel: {use_parallel}")
            if use_parallel:
                print(f"Workers: {max_workers}")
            
            data = load_data_from_directory(
                directory, 
                use_parallel=use_parallel,
                max_workers=max_workers
            )
            
            print(f"\n✓ Successfully loaded {len(data):,} rows")
            print(f"  Columns: {list(data.columns)}")
            print(f"  Memory: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Optional: Save to parquet for next time
            if "--save-parquet" in sys.argv:
                output_file = sys.argv[sys.argv.index("--save-parquet") + 1]
                print(f"\nSaving to {output_file}...")
                data.to_parquet(output_file, compression='snappy')
                print("✓ Saved")
        
        elif command == "convert":
            # Convert DBN to Parquet
            if len(sys.argv) < 4:
                print("ERROR: Missing directories")
                print("Usage: python run_optimizer.py convert DATA/RAW DATA/PARQUET")
                sys.exit(1)
            
            dbn_dir = sys.argv[2]
            parquet_dir = sys.argv[3]
            
            print(f"Converting:")
            print(f"  From: {dbn_dir}")
            print(f"  To:   {parquet_dir}")
            
            convert_dbn_to_parquet(dbn_dir, parquet_dir)
            
            print("\n✓ Conversion complete")
            print(f"\nNext time, use:")
            print(f"  python run_optimizer.py load-parquet {parquet_dir}")
        
        elif command == "load-parquet":
            # Load Parquet files
            if len(sys.argv) < 3:
                print("ERROR: Missing directory argument")
                print("Usage: python run_optimizer.py load-parquet DATA/PARQUET")
                sys.exit(1)
            
            directory = sys.argv[2]
            
            print(f"Loading parquet from: {directory}")
            
            data = load_parquet_from_directory(directory)
            
            print(f"\n✓ Successfully loaded {len(data):,} rows")
            print(f"  Columns: {list(data.columns)}")
            print(f"  Memory: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        elif command == "convert-and-load":
            # Convert then load
            if len(sys.argv) < 4:
                print("ERROR: Missing directories")
                print("Usage: python run_optimizer.py convert-and-load DATA/RAW DATA/PARQUET")
                sys.exit(1)
            
            dbn_dir = sys.argv[2]
            parquet_dir = sys.argv[3]
            
            # Step 1: Convert
            print("STEP 1: Converting to Parquet...")
            convert_dbn_to_parquet(dbn_dir, parquet_dir)
            
            # Step 2: Load
            print("\nSTEP 2: Loading Parquet...")
            data = load_parquet_from_directory(parquet_dir)
            
            print(f"\n✓ Complete: {len(data):,} rows loaded")
        
        else:
            print(f"ERROR: Unknown command '{command}'")
            print("Run without arguments to see usage")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()