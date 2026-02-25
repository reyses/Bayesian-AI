"""
Tests for new WaveRider features (Adaptive Trail, Runner Mode).
"""
import pytest
import sys
import os
import time

# Add project root to path
sys.path.append(os.getcwd())

from training.wave_rider import WaveRider, Position
from config.symbols import MNQ
from core.state_vector import StateVector
from dataclasses import replace

def get_stable_state():
    """Return a state that won't trigger immediate structure break."""
    # L8_confirm=True prevents L8 structure break
    return replace(StateVector.null_state(), L8_confirm=True)

def test_separate_sl_and_trail_stop():
    """Test that stop loss and trail stop are distinguished."""
    rider = WaveRider(MNQ)
    state = get_stable_state()

    # Open a short position at 100.0 with SL at 105.0
    # PT=10 (40 ticks), Trail=20 ticks (5pts).
    # Activation = max(3, 40*0.15) = 6 ticks (1.5 pts).
    rider.open_position(100.0, 'short', state,
                       stop_distance_ticks=20,
                       profit_target_ticks=40,
                       trailing_stop_ticks=20,
                       trail_activation_ticks=6)

    assert rider.position.trail_activation_ticks == 6
    assert not rider.position.is_trail_active

    # 1. Hit initial SL (Price goes to 106.0) -> Should be 'stop_loss'
    decision = rider.update_trail(106.0, state)
    assert decision['should_exit']
    assert decision['exit_reason'] == 'stop_loss'

    # Reset
    rider.open_position(100.0, 'short', state,
                       stop_distance_ticks=20,
                       profit_target_ticks=40,
                       trailing_stop_ticks=20,
                       trail_activation_ticks=6)

    # 2. Move to profit (Price to 98.0, profit 2.0 pts = 8 ticks > 6) -> Trail activates
    decision = rider.update_trail(98.0, state)
    assert not decision['should_exit']
    assert rider.position.is_trail_active

    # 3. Hit trail stop (Price back to 103.0). High water mark 98.0. Trail 5.0 pts -> Stop at 103.0.
    decision = rider.update_trail(103.0, state)
    assert decision['should_exit']
    assert decision['exit_reason'] == 'trail_stop'

def test_trail_activation_threshold_logic():
    """Test the new activation logic: max(3, PT*0.15)."""
    rider = WaveRider(MNQ)
    state = get_stable_state()

    # Case 1: PT=100 ticks. Activation should be max(3, 15) = 15 ticks.
    rider.open_position(100.0, 'long', state, profit_target_ticks=100)
    assert rider.position.trail_activation_ticks == 15

    # Case 2: PT=10 ticks. Activation should be max(3, 1.5) = 3 ticks.
    rider.open_position(100.0, 'long', state, profit_target_ticks=10)
    assert rider.position.trail_activation_ticks == 3

def test_adaptive_trail_distance():
    """Test adaptive trail distance based on wave maturity."""
    rider = WaveRider(MNQ)
    state = get_stable_state()

    # Open long at 100. Trail=20 ticks (5pts).
    rider.open_position(100.0, 'long', state, trailing_stop_ticks=20)

    # Activate trail first (profit > activation)
    # Let's say activation is 3 ticks. Move price to 101 (4 ticks).
    decision = rider.update_trail(101.0, state)
    assert not decision['should_exit']
    assert rider.position is not None
    assert rider.position.is_trail_active

    # Initial trail: 20 ticks
    assert rider.position.trailing_stop_ticks == 20

    # 1. Early Wave (maturity 0.2) -> Trail should widen to 1.5x (30 ticks)
    exit_signal = {'wave_maturity': 0.2}
    rider.update_trail(101.0, state, exit_signal=exit_signal)
    assert rider.position.trailing_stop_ticks == 30

    # 2. Mid Wave (maturity 0.5) -> Trail should return to 1.0x (20 ticks)
    exit_signal = {'wave_maturity': 0.5}
    rider.update_trail(101.0, state, exit_signal=exit_signal)
    assert rider.position.trailing_stop_ticks == 20

    # 3. Late Wave (maturity 0.8) -> Trail should tighten to 0.5x (10 ticks)
    exit_signal = {'wave_maturity': 0.8}
    rider.update_trail(101.0, state, exit_signal=exit_signal)
    assert rider.position.trailing_stop_ticks == 10

def test_runner_mode():
    """Test dynamic profit target extension (Phase C)."""
    rider = WaveRider(MNQ)
    state = get_stable_state()

    # Open long at 100. PT=10pts (40 ticks). Trail=20 ticks.
    rider.open_position(100.0, 'long', state,
                        profit_target_ticks=40,
                        trailing_stop_ticks=20)

    initial_pt = rider.position.profit_target # 110.0
    initial_trail = rider.position.trailing_stop_ticks

    # Move price to near TP (109.0) -> Not hit yet
    rider.update_trail(109.0, state)
    assert rider.position.profit_target == initial_pt

    # Move price to HIT TP (110.0) BUT with high conviction
    # We simulate hitting TP logic inside update_trail.
    # Logic: if unrealized_pnl >= tp_target and conviction >= 0.6 -> extend

    # We must provide wave_maturity to control adaptive logic (avoid widening to 1.5x)
    exit_signal = {'conviction': 0.7, 'wave_maturity': 0.5}
    # This call should TRIGGER runner mode instead of exiting
    decision = rider.update_trail(110.0, state, exit_signal=exit_signal)

    # Should NOT exit
    assert not decision['should_exit']

    # PT should extend (1.5x of 40 ticks = 60 ticks = 15pts -> 115.0)
    assert rider.position.profit_target == 115.0

    # Trail should tighten (0.6x of 20 = 12 ticks)
    assert rider.position.trailing_stop_ticks == 12
    assert rider.position.last_adjustment_reason == 'runner_mode'
