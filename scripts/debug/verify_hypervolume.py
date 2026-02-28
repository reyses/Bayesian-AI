
import os
import sys
import shutil
import time
import subprocess
import argparse
import logging

def main():
    parser = argparse.ArgumentParser(description="Verify hypervolume clustering.")
    parser.add_argument("--data-dir", default="DATA/ATLAS_1MONTH", help="Path to data directory")
    parser.add_argument("--checkpoint-dir", default="checkpoints_verify", help="Path to checkpoint directory")
    parser.add_argument("--reports-dir", default="reports/is", help="Path to reports directory")
    parser.add_argument("--iterations", default="50", help="Number of iterations for orchestrator")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("=== HYPERVOLUME CLUSTERING VERIFICATION ===")

    # 1. Setup paths
    DATA_DIR = args.data_dir
    CHECKPOINT_DIR = args.checkpoint_dir
    REPORTS_DIR = args.reports_dir

    if not os.path.exists(DATA_DIR):
        logger.error(f"{DATA_DIR} not found. Cannot verify.")
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
        "--iterations", args.iterations, # Low iterations for speed
        "--no-dashboard"
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    t0 = time.time()

    # Run with timeout
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        logger.error(f"Process timed out after {args.timeout}s!")
        sys.exit(1)

    duration = time.time() - t0
    logger.info(f"Execution finished in {duration:.1f}s")

    if proc.returncode != 0:
        logger.error("Orchestrator failed!")
        logger.error(f"STDERR:\n{proc.stderr[-2000:]}")
        logger.error(f"STDOUT:\n{proc.stdout[-2000:]}")
        sys.exit(1)

    # 4. Checks
    logger.info("--- Verifying Outputs ---")

    # Check Tree Checkpoint
    tree_pkl = os.path.join(CHECKPOINT_DIR, "hypervolume_tree.pkl")
    if os.path.exists(tree_pkl):
        logger.info(f"[PASS] hypervolume_tree.pkl exists ({os.path.getsize(tree_pkl)/1024:.1f} KB)")
    else:
        logger.error("[FAIL] hypervolume_tree.pkl MISSING")

    # Check Trade Log
    log_csv = os.path.join(REPORTS_DIR, "oracle_trade_log.csv")
    if os.path.exists(log_csv):
        logger.info("[PASS] oracle_trade_log.csv exists")
        with open(log_csv, 'r') as f:
            lines = f.readlines()
            logger.info(f"Trade count: {len(lines)-1}")
            if len(lines) > 1:
                logger.info(f"Sample row: {lines[1].strip()}")
    else:
        logger.error("[FAIL] oracle_trade_log.csv MISSING (No trades taken?)")
        logger.error(f"STDOUT tail:\n{proc.stdout[-1000:]}")

    # Check for Gate 0.5 (should be gone)
    if "Gate 0.5" in proc.stdout:
        logger.error("[FAIL] Gate 0.5 detected in output logs (should be removed)")
    else:
        logger.info("[PASS] Gate 0.5 not seen in logs")

    logger.info("Verification Complete.")

if __name__ == "__main__":
    main()
