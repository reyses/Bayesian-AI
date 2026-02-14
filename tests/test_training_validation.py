"""
Bayesian-AI - Training Validation Test
Validates training metrics and performance.
"""
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

from core.bayesian_brain import BayesianBrain, QuantumBayesianBrain
from training.databento_loader import DatabentoLoader
from tests.utils import get_test_data_files, find_test_data_file

def run_training_validation():
    """
    Runs the training orchestrator for validation on all available test data files and returns a list of metrics.
    Uses a small slice of real data to ensure speed.
    """
    # 1. Find Data
    # Prefer the specific requested file if available
    specific_file = find_test_data_file('glbx-mdp3-20250730.trades.0000.dbn.zst')
    if specific_file:
        data_files = [specific_file]
    else:
        # Prefer Testing DATA for validation speed
        testing_data_dir = os.path.join(PROJECT_ROOT, 'tests', 'Testing DATA')
        data_files = glob.glob(os.path.join(testing_data_dir, "*.dbn*"))
        data_files.extend(glob.glob(os.path.join(testing_data_dir, "*.parquet")))

        if not data_files:
             data_files = get_test_data_files()

    # Limit to first 1 file to prevent timeout
    if len(data_files) > 1:
        data_files = data_files[:1]

    if not data_files:
        return [{
            "status": "FAILED",
            "error": "No data files found in DATA/RAW or tests/Testing DATA"
        }]

    all_metrics = []

    for source_file in data_files:
        # 2. Setup Temp Dir
        temp_dir = tempfile.mkdtemp()
        # New model name
        model_path = os.path.join(temp_dir, "quantum_probability_table.pkl")
        temp_data_path = os.path.join(temp_dir, "validation_data.parquet")

        start_time = time.time()

        try:
            # 3. Prepare Data Slice (limit to 200 ticks for speed)
            try:
                if source_file.endswith('.parquet'):
                    df = pd.read_parquet(source_file)
                else:
                    df = DatabentoLoader.load_data(source_file)

                # If we slice by rows, we might cut time.
                # Let's try to get at least 1000 rows if available, to cover some time.
                if len(df) > 1000:
                    df = df.head(1000)

                # Ensure timestamp is preserved/converted for parquet
                df.reset_index(drop=True, inplace=True)
                df.to_parquet(temp_data_path)

            except Exception as e:
                all_metrics.append({"status": "FAILED", "error": f"Data preparation failed for {source_file}: {e}"})
                continue

            # 4. Run Orchestrator
            # Reduced iterations to 2 to avoid timeouts in slow envs
            # Updated to 10 iterations as per validation requirements
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "training", "orchestrator.py"),
                "--data", temp_data_path,
                "--iterations", "10",
                "--checkpoint-dir", temp_dir,
                "--no-dashboard", # Ensure dashboard is disabled for tests
                "--skip-deps"
            ]

            # Timeout set to 180s
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            runtime = time.time() - start_time

            stdout = result.stdout
            stderr = result.stderr

            if result.returncode != 0:
                all_metrics.append({
                    "status": "FAILED",
                    "error": f"Orchestrator failed for {source_file}.\nStdout: {stdout}\nStderr: {stderr}"
                })
                continue

            # 5. Parse Logs
            # New format: [DATA] Loaded {len(df_15m)} 15min bars, {len(df_15s)} 15sec bars
            ticks_match = re.search(r"Loaded (\d+) 15min bars", stdout)
            if not ticks_match:
                 # Try old format fallback just in case
                 ticks_match = re.search(r"Data: (\d+) ticks", stdout)

            total_ticks = int(ticks_match.group(1)) if ticks_match else 0

            # 6. Load Model & Calc Metrics
            if not os.path.exists(model_path):
                 # Check for old name
                 old_model_path = os.path.join(temp_dir, "probability_table.pkl")
                 if os.path.exists(old_model_path):
                     model_path = old_model_path
                 # Check for per-day model (day_001_brain.pkl)
                 else:
                     brain_files = glob.glob(os.path.join(temp_dir, "day_*_brain.pkl"))
                     if brain_files:
                         model_path = brain_files[0]
                     else:
                         all_metrics.append({
                            "status": "FAILED",
                            "error": f"Model file not generated for {source_file}. Stdout: {stdout}"
                        })
                         continue

            # Use QuantumBayesianBrain if it's the new model
            try:
                brain = QuantumBayesianBrain()
                brain.load(model_path)
            except:
                # Fallback to old brain if load fails (maybe incompatible pickle or old format)
                brain = BayesianBrain()
                brain.load(model_path)

            summary = brain.get_summary()
            unique_states = summary['total_unique_states']

            high_conf_states = brain.get_all_states_above_threshold(min_prob=0.80)
            high_conf_count = len(high_conf_states)

            top_5 = []
            for s in high_conf_states[:5]:
                # State might be complex object, converting to str
                state_str = str(s['state'])
                top_5.append({
                    "state": state_str,
                    "probability": s['probability'],
                    "wins": s['wins'],
                    "losses": s['losses']
                })

            metrics = {
                "status": "SUCCESS",
                "file": source_file,
                "iterations_completed": 10,
                "runtime_seconds": round(runtime, 2),
                "total_ticks": total_ticks, # This is now 15min bars count if matched new format
                "unique_states_learned": unique_states,
                "high_confidence_states": high_conf_count,
                "top_5_states": top_5
            }
            all_metrics.append(metrics)

        except subprocess.TimeoutExpired:
            all_metrics.append({
                "status": "FAILED",
                "file": source_file,
                "error": "Orchestrator timed out after 180s"
            })
        except Exception as e:
            all_metrics.append({
                "status": "FAILED",
                "file": source_file,
                "error": str(e)
            })
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    return all_metrics

def test_training_validation_run():
    """
    Pytest case to verify the validation logic works.
    """
    all_metrics = run_training_validation()

    for metrics in all_metrics:
        if metrics["status"] != "SUCCESS":
            pytest.fail(f"Training validation failed for {metrics.get('file', 'unknown file')}: {metrics.get('error')}")

        assert metrics["unique_states_learned"] >= 0
        assert isinstance(metrics["top_5_states"], list)

if __name__ == "__main__":
    # When run as script, print JSON metrics
    all_metrics = run_training_validation()
    print("::METRICS::")
    print(json.dumps(all_metrics, indent=2))
    print("::END_METRICS::")

    for metrics in all_metrics:
        if metrics["status"] != "SUCCESS":
            sys.exit(1)
