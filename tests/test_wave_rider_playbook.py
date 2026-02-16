import pytest
import pickle
import os
import numpy as np
from dataclasses import replace
from training.wave_rider import WaveRider, Position
from core.three_body_state import ThreeBodyQuantumState
from core.bayesian_brain import TradeOutcome
from config.symbols import MNQ

# Mock Data
TEMPLATE_ID = 420
TEMPLATE_PARAMS = {
    'stop_loss_ticks': 15,
    'take_profit_ticks': 40,
    'trailing_stop_ticks': 12
}

# Centroid: Z, V, M, C
CENTROID = np.array([1.5, 2.0, 3.0, 0.8])

def create_mock_playbook(path):
    data = {
        TEMPLATE_ID: {
            'centroid': CENTROID,
            'params': TEMPLATE_PARAMS,
            'member_count': 100
        }
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)

@pytest.fixture
def playbook_path(tmp_path):
    path = tmp_path / "pattern_playbook.pkl"
    create_mock_playbook(path)
    return str(path)

def test_load_playbook(playbook_path):
    """Test that playbook loads correctly"""
    rider = WaveRider(MNQ, playbook_path=playbook_path)
    assert rider.centroids is not None
    assert len(rider.template_ids) == 1
    assert rider.template_ids[0] == TEMPLATE_ID
    assert TEMPLATE_ID in rider.playbook_data

def test_match_state(playbook_path):
    """Test vector matching logic"""
    rider = WaveRider(MNQ, playbook_path=playbook_path)

    # State close to centroid
    # Vector: [abs(z), abs(v), abs(m), c]
    # Centroid: [1.5, 2.0, 3.0, 0.8]
    state_match = ThreeBodyQuantumState.null_state()
    state_match = replace(state_match,
        z_score=1.55,           # diff 0.05
        particle_velocity=2.05, # diff 0.05
        momentum_strength=3.0,  # diff 0.0
        coherence=0.8           # diff 0.0
    )
    # Dist ~ sqrt(0.05^2 + 0.05^2) = sqrt(0.005) ~ 0.07 < 0.5

    tid = rider.match_current_state(state_match)
    assert tid == TEMPLATE_ID

    # State far from centroid
    state_far = replace(state_match, z_score=10.0)
    tid_far = rider.match_current_state(state_far)
    assert tid_far is None

def test_configure_and_open_position(playbook_path):
    """Test applying template configuration to position"""
    rider = WaveRider(MNQ, playbook_path=playbook_path)
    state = ThreeBodyQuantumState.null_state()

    # Configure explicitly
    rider.configure_for_template(TEMPLATE_ID)
    assert rider.active_template_params == TEMPLATE_PARAMS

    # Open position with template ID implicitly using active params
    entry_price = 100.0
    rider.open_position(entry_price, 'long', state, template_id=TEMPLATE_ID)

    assert rider.position is not None
    assert rider.position.template_id == TEMPLATE_ID

    # Check Stop Loss (15 ticks * 0.25 = 3.75)
    expected_sl = 100.0 - (15 * 0.25)
    assert rider.position.stop_loss == expected_sl

    # Check Take Profit (40 ticks * 0.25 = 10.0)
    expected_tp = 100.0 + (40 * 0.25)
    assert rider.position.take_profit == expected_tp

    # Check Trailing Config
    expected_trail = {'tight': 12, 'medium': 12, 'wide': 12}
    assert rider.position.trailing_config == expected_trail

def test_update_trail_with_template_config(playbook_path):
    """Test trail updates with fixed template config"""
    rider = WaveRider(MNQ, playbook_path=playbook_path)
    state = ThreeBodyQuantumState.null_state()

    rider.configure_for_template(TEMPLATE_ID)
    rider.open_position(100.0, 'long', state, template_id=TEMPLATE_ID)

    # Move price up by 5 points ($10 profit for MNQ usually)
    # Trailing Stop should be 12 ticks (3.0 pts) below HWM
    current_price = 105.0
    decision = rider.update_trail(current_price, state)

    assert not decision['should_exit']
    # HWM is 105.0. Stop should be 105.0 - 3.0 = 102.0
    assert rider.position.stop_loss == 102.0

    # Test Take Profit Exit
    # TP is at 110.0 (entry + 10)
    decision_tp = rider.update_trail(110.0, state)
    assert decision_tp['should_exit']
    assert decision_tp['exit_reason'] == 'take_profit'

def test_bayesian_brain_outcome_template_id():
    """Verify TradeOutcome can hold template_id"""
    outcome = TradeOutcome(
        state=ThreeBodyQuantumState.null_state(),
        entry_price=100.0,
        exit_price=110.0,
        pnl=20.0,
        result='WIN',
        timestamp=1000.0,
        exit_reason='tp',
        template_id=TEMPLATE_ID
    )
    assert outcome.template_id == TEMPLATE_ID
