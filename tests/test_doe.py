"""
Bayesian-AI - DOE Features Verification
Tests Grid Search, Walk-Forward, and Monte Carlo modules in Orchestrator
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.orchestrator import TrainingOrchestrator

@pytest.fixture
def synthetic_data():
    """Generates synthetic tick data for testing"""
    # Generate 5000 ticks
    timestamps = pd.date_range(start='2024-01-01', periods=5000, freq='1s')
    prices = np.cumsum(np.random.randn(5000)) + 10000
    volumes = np.random.randint(1, 10, 5000)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes,
        'type': 'trade'
    })
    return df

def test_grid_search(synthetic_data):
    """Verify Grid Search logic"""
    # Use temporary directory for output
    output_dir = 'temp_test_doe_grid'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    orchestrator = TrainingOrchestrator("MNQ", data=synthetic_data, use_gpu=False, output_dir=output_dir)

    # Minimal grid
    param_grid = {
        'min_prob': [0.7],
        'min_conf': [0.1, 0.5]
    }

    result = orchestrator.run_grid_search(param_grid)

    assert 'params' in result
    assert 'metrics' in result
    assert result['params']['min_prob'] == 0.7

    # Cleanup
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def test_walk_forward(synthetic_data):
    """Verify Walk Forward logic"""
    output_dir = 'temp_test_doe_wf'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    orchestrator = TrainingOrchestrator("MNQ", data=synthetic_data, use_gpu=False, output_dir=output_dir)

    # 1 Window only
    results = orchestrator.run_walk_forward(train_window=2000, test_window=1000, step=5000)

    assert len(results) > 0

    # Cleanup
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def test_monte_carlo(synthetic_data):
    """Verify Monte Carlo logic"""
    output_dir = 'temp_test_doe_mc'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    orchestrator = TrainingOrchestrator("MNQ", data=synthetic_data, use_gpu=False, output_dir=output_dir)

    # Run 2 iterations
    stats = orchestrator.run_monte_carlo(iterations=2, sample_fraction=0.5)

    assert 'mean' in stats

    # Cleanup
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
