
import os
import sys
import shutil
import time
import subprocess
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Verify Hypervolume Clustering.")
    parser.add_argument("--data-dir", default="DATA/ATLAS_1MONTH", help="Path to data directory")
    parser.add_argument("--checkpoint-dir", default="checkpoints_verify", help="Path to checkpoint directory")
    parser.add_argument("--reports-dir", default="reports/is", help="Path to reports directory")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")

    args = parser.parse_args()

    logger.info("=== HYPERVOLUME CLUSTERING VERIFICATION ===")

    if not os.path.exists(args.data_dir):
        logger.error(f"{args.data_dir} not found. Cannot verify.")
        sys.exit(1)

    # 2. Clean previous run
    if os.path.exists(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)
    if os.path.exists(args.reports_dir):
        shutil.rmtree(args.reports_dir)

    # 3. Run Orchestrator (Fresh Training + Forward Pass)
    cmd = [
        sys.executable, "-m", "training.orchestrator",
        "--data", args.data_dir,
        "--checkpoint-dir", args.checkpoint_dir,
        "--fresh",
        "--iterations", "50", # Low iterations for speed
        "--no-dashboard"
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    t0 = time.time()

    # Run with timeout
    try:
        # Merge stderr into stdout so we can grep the logs regardless of where they are written
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        logger.error(f"Process timed out after {args.timeout}s!")
        sys.exit(1)

    duration = time.time() - t0
    logger.info(f"Execution finished in {duration:.1f}s")

    if proc.returncode != 0:
        logger.error("Orchestrator failed!")
        logger.error(f"OUTPUT:\n{proc.stdout[-4000:]}")
        sys.exit(1)

    # 4. Checks
    logger.info("--- Verifying Outputs ---")

    # Check Tree Checkpoint
    tree_pkl = os.path.join(args.checkpoint_dir, "hypervolume_tree.pkl")
    if os.path.exists(tree_pkl):
        logger.info(f"[PASS] hypervolume_tree.pkl exists ({os.path.getsize(tree_pkl)/1024:.1f} KB)")
    else:
        logger.error("[FAIL] hypervolume_tree.pkl MISSING")

    # Check Trade Log
    log_csv = os.path.join(args.reports_dir, "oracle_trade_log.csv")
    if os.path.exists(log_csv):
        logger.info("[PASS] oracle_trade_log.csv exists")
        with open(log_csv, 'r') as f:
            lines = f.readlines()
            logger.info(f"       Trade count: {len(lines)-1}")
            if len(lines) > 1:
                logger.info(f"       Sample row: {lines[1].strip()}")
    else:
        logger.error("[FAIL] oracle_trade_log.csv MISSING (No trades taken?)")
        logger.info(f"STDOUT tail:\n{proc.stdout[-1000:]}")

    # Check for Gate 0.5 (should be gone)
    if "Gate 0.5" in proc.stdout:
        logger.error("[FAIL] Gate 0.5 detected in output logs (should be removed)")
    else:
        logger.info("[PASS] Gate 0.5 not seen in logs")

    logger.info("Verification Complete.")

if __name__ == "__main__":
    main()
