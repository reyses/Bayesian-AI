"""
Bayesian-AI - Sentinel Bridge
The Local Feedback Loop
"""
import logging
import subprocess
import os
import shutil
import datetime

# We avoid configuring basicConfig to write to CUDA_Debug.log here
# to prevent the sentinel from modifying the file it is monitoring.

def trigger_jules_repair(fault_details):
    """Logic -> Cloud Correction -> Constraint: Async API Pull"""
    # Sentinel detects CRITICAL error in CUDA_Debug.log
    # Invokes Jules API: jules fix --context=CUDA_Debug.log --target=cuda/
    print(f"Triggering Jules Repair for: {fault_details}")
    try:
        if shutil.which("jules") is None:
            # Fallback: Comment for Jules
            print(f"WARNING: 'jules' command not found.")
            print(f"PROTOCOL_OVERRIDE: @Jules, please manually inspect and fix: {fault_details}")
            # Try gh cli if available
            if shutil.which("gh"):
                subprocess.run(["gh", "pr", "comment", "--body", f"@Jules {fault_details}"], check=False)
        else:
            subprocess.run(["jules", "fix", "--context", "CUDA_Debug.log", "--target=cuda/"], check=True)
            subprocess.run(["git", "pull", "origin", "jules-fix-branch"], check=True)

        # Log rotation to prevent re-triggering
        log_file = 'CUDA_Debug.log'
        if os.path.exists(log_file):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_log = f"{log_file}.processed_{timestamp}"
            shutil.move(log_file, processed_log)
            print(f"Rotated {log_file} to {processed_log}")

    except Exception as e:
        print(f"Failed to trigger repair: {e}")

def main():
    log_file = 'CUDA_Debug.log'
    if not os.path.exists(log_file):
        print(f"{log_file} not found.")
        return

    try:
        with open(log_file, 'r') as f:
            content = f.read()
            if 'CRITICAL' in content:
                print("CRITICAL error found in logs. Initiating repair...")
                trigger_jules_repair("CRITICAL Error Detected in CUDA_Debug.log. Please repair.")
            else:
                print("No critical errors found in logs.")
    except Exception as e:
        print(f"Error reading log: {e}")

if __name__ == "__main__":
    main()
