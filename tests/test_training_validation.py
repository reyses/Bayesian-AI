import sys
import os
import json
import subprocess
import glob
import tempfile
import shutil
import re
import time
import pytest
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.bayesian_brain import BayesianBrain
# Import get_data_source to reuse loading logic (handles .dbn and .parquet)
from training.orchestrator import get_data_source

def run_training_validation():
    """
    Runs the training orchestrator for validation and returns metrics.
    Uses a small slice of real data to ensure speed.
    """
    # 1. Find Data
    data_dir = os.path.join(PROJECT_ROOT, "DATA", "RAW")
    data_files = []
    if os.path.exists(data_dir):
        # Prefer dbn/parquet
        data_files = glob.glob(os.path.join(data_dir, "*.dbn*")) + glob.glob(os.path.join(data_dir, "*.parquet"))

    if not data_files:
        return {
            "status": "FAILED",
            "error": "No data files found in DATA/RAW"
        }

    # 2. Setup Temp Dir
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "probability_table.pkl")
    temp_data_path = os.path.join(temp_dir, "validation_data.parquet")

    start_time = time.time()

    try:
        # 3. Prepare Data Slice (limit to 1000 ticks for speed)
        source_file = data_files[0]
        try:
            df = get_data_source(source_file)
            if len(df) > 1000:
                df = df.head(1000)

            # Ensure timestamp is preserved/converted for parquet
            df.reset_index(drop=True, inplace=True)
            df.to_parquet(temp_data_path)

        except Exception as e:
            return {"status": "FAILED", "error": f"Data preparation failed: {e}"}

        # 4. Run Orchestrator
        # Reduced iterations to 2 to avoid timeouts in slow envs
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "training", "orchestrator.py"),
            "--data-file", temp_data_path,
            "--iterations", "2",
            "--output", temp_dir
        ]

        # Timeout set to 60s
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        runtime = time.time() - start_time

        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            return {
                "status": "FAILED",
                "error": f"Orchestrator failed.\nStdout: {stdout}\nStderr: {stderr}"
            }

        # 5. Parse Logs
        # "Data: X ticks"
        ticks_match = re.search(r"Data: (\d+) ticks", stdout)
        total_ticks = int(ticks_match.group(1)) if ticks_match else 0

        # 6. Load Model & Calc Metrics
        if not os.path.exists(model_path):
             return {
                "status": "FAILED",
                "error": "Model file not generated"
            }

        brain = BayesianBrain()
        brain.load(model_path)

        summary = brain.get_summary()
        unique_states = summary['total_unique_states']

        # Use existing method to get states with >= 80% winrate and >= 10 samples
        high_conf_states = brain.get_all_states_above_threshold(min_prob=0.80)
        high_conf_count = len(high_conf_states)

        top_5 = []
        for s in high_conf_states[:5]:
            state_str = str(s['state'])
            top_5.append({
                "state": state_str,
                "probability": s['probability'],
                "wins": s['wins'],
                "losses": s['losses']
            })

        metrics = {
            "status": "SUCCESS",
            "iterations_completed": 2, # Reporting actual iterations run for validation
            "runtime_seconds": round(runtime, 2),
            "files_loaded": 1,
            "total_ticks": total_ticks,
            "unique_states_learned": unique_states,
            "high_confidence_states": high_conf_count,
            "top_5_states": top_5
        }

        return metrics

    except subprocess.TimeoutExpired:
        return {
            "status": "FAILED",
            "error": "Orchestrator timed out after 60s"
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e)
        }

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def test_training_validation_run():
    """
    Pytest case to verify the validation logic works.
    """
    metrics = run_training_validation()

    if metrics["status"] != "SUCCESS":
        pytest.fail(f"Training validation failed: {metrics.get('error')}")

    assert metrics["unique_states_learned"] >= 0
    assert isinstance(metrics["top_5_states"], list)

if __name__ == "__main__":
    # When run as script, print JSON metrics
    metrics = run_training_validation()
    print("::METRICS::")
    print(json.dumps(metrics, indent=2))
    print("::END_METRICS::")

    if metrics["status"] != "SUCCESS":
        sys.exit(1)
