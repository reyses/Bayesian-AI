
import os
import sys
import shutil
import time
import subprocess

def main():
    print("=== HYPERVOLUME CLUSTERING VERIFICATION ===")

    # 1. Setup paths
    DATA_DIR = "DATA/ATLAS_1MONTH"
    CHECKPOINT_DIR = "checkpoints_verify"
    REPORTS_DIR = "reports/is"

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: {DATA_DIR} not found. Cannot verify.")
        sys.exit(1)

    # 2. Clean previous run
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
    if os.path.exists(REPORTS_DIR):
        shutil.rmtree(REPORTS_DIR)

    # 3. Run Orchestrator (Fresh Training + Forward Pass)
    cmd = [
        sys.executable, "-m", "training.orchestrator",
        "--data", DATA_DIR,
        "--checkpoint-dir", CHECKPOINT_DIR,
        "--fresh",
        "--iterations", "50", # Low iterations for speed
        "--no-dashboard"
    ]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.time()

    # Run with timeout
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("ERROR: Process timed out after 600s!")
        sys.exit(1)

    duration = time.time() - t0
    print(f"Execution finished in {duration:.1f}s")

    if proc.returncode != 0:
        print("ERROR: Orchestrator failed!")
        print("STDERR:\n", proc.stderr[-2000:])
        print("STDOUT:\n", proc.stdout[-2000:])
        sys.exit(1)

    # 4. Checks
    print("\n--- Verifying Outputs ---")

    # Check Tree Checkpoint
    tree_pkl = os.path.join(CHECKPOINT_DIR, "hypervolume_tree.pkl")
    if os.path.exists(tree_pkl):
        print(f"[PASS] hypervolume_tree.pkl exists ({os.path.getsize(tree_pkl)/1024:.1f} KB)")
    else:
        print("[FAIL] hypervolume_tree.pkl MISSING")

    # Check Trade Log
    log_csv = os.path.join(REPORTS_DIR, "oracle_trade_log.csv")
    if os.path.exists(log_csv):
        print(f"[PASS] oracle_trade_log.csv exists")
        with open(log_csv, 'r') as f:
            lines = f.readlines()
            print(f"       Trade count: {len(lines)-1}")
            if len(lines) > 1:
                print("       Sample row:", lines[1].strip())
    else:
        print("[FAIL] oracle_trade_log.csv MISSING (No trades taken?)")
        print("STDOUT tail:\n", proc.stdout[-1000:])

    # Check for Gate 0.5 (should be gone)
    if "Gate 0.5" in proc.stdout:
        print("[FAIL] Gate 0.5 detected in output logs (should be removed)")
    else:
        print("[PASS] Gate 0.5 not seen in logs")

    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
