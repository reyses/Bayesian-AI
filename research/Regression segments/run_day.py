import os
import sys
import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description="Wrapper to run Stage 1 and Stage 2 for a single day.")
    parser.add_argument('--day', type=str, required=True, help="The day to process, e.g. 2026_01_02")
    parser.add_argument('--hours', type=int, default=24, help="Hours to process in Stage 1")
    args = parser.parse_args()
    
    day = args.day
    print(f"========================================")
    print(f"🚀 Starting Pipeline for Day: {day}")
    print(f"========================================")
    
    # Run Stage 1
    t0 = time.time()
    print(f"\n[1/2] Running Stage 1 (Speed Pass)...")
    try:
        subprocess.run([sys.executable, "research/Regression segments/stage1_speed_pass.py", "--day", day, "--hours", str(args.hours)], check=True)
        print(f"✅ Stage 1 completed in {(time.time() - t0)/60:.1f} mins.")
    except subprocess.CalledProcessError as e:
        print(f"❌ [FATAL] Stage 1 failed for {day}. Exiting.")
        sys.exit(1)
        
    # Run Stage 2
    t1 = time.time()
    print(f"\n[2/2] Running Stage 2 (Parallel Chaos Resolution)...")
    try:
        subprocess.run([sys.executable, "research/Regression segments/stage2_parallel_chaos.py", "--day", day], check=True)
        print(f"✅ Stage 2 completed in {(time.time() - t1)/60:.1f} mins.")
    except subprocess.CalledProcessError as e:
        print(f"❌ [FATAL] Stage 2 failed for {day}. Exiting.")
        sys.exit(1)
        
    print(f"\n========================================")
    print(f"🎉 Pipeline successfully finished for {day} in {(time.time() - t0)/60:.1f} total mins.")
    print(f"Final output: artifacts/stage2_segments_{day}.json")
    print(f"========================================")

if __name__ == "__main__":
    main()
