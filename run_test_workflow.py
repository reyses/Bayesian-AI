#!/usr/bin/env python3
import subprocess
import sys
import time
import argparse
import shlex

def install_dependencies():
    """Installs dependencies from requirements.txt."""
    print(f"\n{'='*60}")
    print("Installing Dependencies...")
    print(f"{'='*60}\n")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--prefer-binary", "-r", "requirements.txt"])
        print("\n[SUCCESS] Dependencies installed.")
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILURE] Failed to install dependencies: {e}")
        sys.exit(1)

def run_command(command, description, critical=True):
    """
    Runs a shell command.
    Returns True if successful, False otherwise.
    """
    # Create a display string for the command
    if isinstance(command, list):
        cmd_str = shlex.join(command)
    else:
        cmd_str = command

    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd_str}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        # Run the command and let it print to stdout/stderr
        # If command is a list, subprocess runs it directly (safer, no shell injection)
        # If command is a string, we might need shell=True, but we're moving to lists.
        use_shell = isinstance(command, str)

        process = subprocess.run(
            command,
            shell=use_shell,
            check=False  # Don't raise exception on non-zero exit code
        )

        return_code = process.returncode
        duration = time.time() - start_time

        if return_code == 0:
            print(f"\n[SUCCESS] {description} passed in {duration:.2f}s")
            return True
        else:
            print(f"\n[FAILURE] {description} failed with return code {return_code} in {duration:.2f}s")
            if critical:
                print(f"Critical test failed: {description}. Aborting workflow.")
                sys.exit(return_code)
            return False

    except Exception as e:
        print(f"\n[ERROR] An error occurred while running {description}: {e}")
        if critical:
            sys.exit(1)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the local test workflow.")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies from requirements.txt before running tests.")
    args = parser.parse_args()

    if args.install_deps:
        install_dependencies()

    print("Starting Local Test Workflow...")
    start_time = time.time()

    # Use sys.executable to ensure we use the same python interpreter
    python_cmd = sys.executable

    steps = [
        {
            "command": [python_cmd, "tests/topic_build.py"],
            "description": "Integrity Test (Topic Build)",
            "critical": True
        },
        {
            "command": [python_cmd, "-m", "pytest", "tests/topic_math.py"],
            "description": "Math & Logic Test",
            "critical": True
        },
        {
            "command": [python_cmd, "tests/topic_diagnostics.py"],
            "description": "Diagnostics Test",
            "critical": True
        },
        {
            "command": [python_cmd, "tests/test_bayesian_brain.py"],
            "description": "Bayesian Brain Test",
            "critical": True
        },
        {
            "command": [python_cmd, "tests/test_state_vector.py"],
            "description": "State Vector Test",
            "critical": True
        },
        {
            "command": [python_cmd, "tests/test_legacy_layer_engine.py"],
            "description": "Legacy Layer Engine Test",
            "critical": True
        },
        {
            "command": [python_cmd, "scripts/gpu_health_check.py"],
            "description": "GPU Health Check",
            "critical": False  # Allow failure if no GPU (continue-on-error: true)
        },
        {
            "command": [python_cmd, "-m", "pytest", "tests/test_training_validation.py"],
            "description": "Training Validation",
            "critical": True
        }
    ]

    failed_steps = []

    for step in steps:
        # Pass the command list directly
        success = run_command(step["command"], step["description"], step["critical"])
        if not success:
            failed_steps.append(step["description"])

    total_duration = time.time() - start_time

    print(f"\n{'='*60}")
    if not failed_steps:
        print(f"Workflow Completed Successfully in {total_duration:.2f}s")
        sys.exit(0)
    else:
        print(f"Workflow Completed with Non-Critical Failures in {total_duration:.2f}s")
        print("Failed Steps (Non-Critical):")
        for step in failed_steps:
            print(f" - {step}")
        # Exit with success because failed steps were non-critical
        sys.exit(0)

if __name__ == "__main__":
    main()
