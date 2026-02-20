
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from training.timeframe_belief_network import TimeframeBeliefNetwork, TimeframeWorker

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    # Return dummy states for batch_compute_states
    engine.batch_compute_states.return_value = [{'bar_idx': i, 'state': MagicMock()} for i in range(100)]
    return engine

@pytest.fixture
def mock_scaler():
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 16)) # 16D feature vector
    return scaler

@pytest.fixture
def mock_library():
    return {
        1: {'dir_coeff': [0.1]*16, 'dir_intercept': 0.0, 'mfe_coeff': [0.1]*16, 'mfe_intercept': 0.0},
        2: {} # Unfitted
    }

def test_initialization_15s_default(mock_library, mock_scaler, mock_engine):
    """Test legacy initialization with 15s default."""
    valid_tids = [1, 2]
    centroids = np.zeros((2, 16))

    tbn = TimeframeBeliefNetwork(mock_library, mock_scaler, mock_engine, valid_tids, centroids)

    # Check workers
    assert 15 in tbn.workers
    assert 3600 in tbn.workers
    assert 5 not in tbn.workers # 5s not active by default (15s base)
    assert 1 not in tbn.workers # 1s not active by default

    # Check leaf
    assert tbn.workers[15].is_leaf
    assert not tbn.workers[3600].is_leaf

    # Check bars_per_update
    assert tbn.workers[15].bars_per_update == 1
    assert tbn.workers[3600].bars_per_update == 240 # 3600 // 15

def test_initialization_1s(mock_library, mock_scaler, mock_engine):
    """Test initialization with 1s base resolution."""
    valid_tids = [1, 2]
    centroids = np.zeros((2, 16))

    tbn = TimeframeBeliefNetwork(mock_library, mock_scaler, mock_engine, valid_tids, centroids,
                                 base_resolution_seconds=1)

    # Check workers
    assert 15 in tbn.workers
    assert 5 in tbn.workers
    assert 1 in tbn.workers

    # Check leaf
    assert tbn.workers[1].is_leaf
    assert not tbn.workers[5].is_leaf
    assert not tbn.workers[15].is_leaf

    # Check bars_per_update
    assert tbn.workers[1].bars_per_update == 1
    assert tbn.workers[5].bars_per_update == 5 # 5 // 1
    assert tbn.workers[15].bars_per_update == 15 # 15 // 1
    assert tbn.workers[3600].bars_per_update == 3600 # 3600 // 1

def test_initialization_5s(mock_library, mock_scaler, mock_engine):
    """Test initialization with 5s base resolution."""
    valid_tids = [1, 2]
    centroids = np.zeros((2, 16))

    tbn = TimeframeBeliefNetwork(mock_library, mock_scaler, mock_engine, valid_tids, centroids,
                                 base_resolution_seconds=5)

    # Check workers
    assert 15 in tbn.workers
    assert 5 in tbn.workers
    assert 1 not in tbn.workers # 1s is below base

    # Check leaf
    assert tbn.workers[5].is_leaf

    # Check bars_per_update
    assert tbn.workers[5].bars_per_update == 1
    assert tbn.workers[15].bars_per_update == 3 # 15 // 5

def test_prepare_day_1s(mock_library, mock_scaler, mock_engine):
    """Test prepare_day with 1s data."""
    tbn = TimeframeBeliefNetwork(mock_library, mock_scaler, mock_engine, [1], np.zeros((1, 16)),
                                 base_resolution_seconds=1)

    # Create dummy 1s DataFrame - enough for 5 bars at 1h (5 * 3600 = 18000 bars)
    dates = pd.date_range('2025-01-01', periods=18000, freq='1s')
    df_1s = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 100
    }, index=dates)
    df_1s['timestamp'] = dates.astype(np.int64) // 10**9

    # Mock engine to return dummy states
    mock_engine.batch_compute_states.return_value = [{'bar_idx': i, 'state': MagicMock()} for i in range(100)]

    tbn.prepare_day(df_1s)

    # Check if workers received states
    # 1s worker should have used base states
    assert len(tbn.workers[1]._states) > 0
    # 5s worker should have resampled
    assert len(tbn.workers[5]._states) > 0
    # 1h worker should have resampled (len=1 for 3600s data)
    assert len(tbn.workers[3600]._states) > 0

def test_tick_all(mock_library, mock_scaler, mock_engine):
    """Test tick_all with 1s base."""
    tbn = TimeframeBeliefNetwork(mock_library, mock_scaler, mock_engine, [1], np.zeros((1, 16)),
                                 base_resolution_seconds=1)

    # Prepare dummy states with numeric attributes
    mock_states = []
    for i in range(3600):
        s = MagicMock()
        s.z_score = 0.0
        s.pattern_maturity = 0.0
        s.tunnel_probability = 0.0
        s.particle_velocity = 0.0
        s.momentum_strength = 0.0
        s.coherence = 0.0
        s.adx_strength = 0.0
        s.hurst_exponent = 0.5
        s.dmi_plus = 0.0
        s.dmi_minus = 0.0
        s.term_pid = 0.0
        s.oscillation_coherence = 0.0
        mock_states.append({'bar_idx': i, 'state': s})

    for w in tbn.workers.values():
        w.prepare(mock_states)

    # Tick at bar 0
    updated = tbn.tick_all(0)
    # All workers should update on bar 0 (first bar)
    assert updated == len(tbn.workers)

    # Tick at bar 1 (1s elapsed)
    # 1s worker updates (bars_per_update=1)
    # 5s worker (bars_per_update=5) does NOT update (1 // 5 = 0, same as prev)
    updated = tbn.tick_all(1)
    assert updated == 1 # Only 1s worker

    # Tick at bar 5 (5s elapsed)
    # 1s worker updates
    # 5s worker updates (5 // 5 = 1, changed from 0)
    updated = tbn.tick_all(5)
    assert updated >= 2 # 1s and 5s at least
