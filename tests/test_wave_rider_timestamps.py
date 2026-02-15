
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

def test_wave_rider_timestamp_backtesting():
    """Test that WaveRider uses correct timestamps for backtesting (not wall-clock time)."""
    rider = WaveRider(MNQ)

    # State with explicit timestamp (simulated past time)
    # Using float timestamp: 1000.0
    sim_time = 1000.0
    state = replace(StateVector.null_state(), timestamp=sim_time)

    # Open position
    rider.open_position(100.0, 'long', state)

    # Verify entry time is sim_time, NOT current wall clock
    assert rider.position.entry_time == sim_time
    assert rider.price_history[0][0] == sim_time
    assert abs(rider.position.entry_time - time.time()) > 100  # Ensure it's not "now"

    # Update trail at sim_time + 60s
    next_time = sim_time + 60.0
    next_state = replace(StateVector.null_state(), timestamp=next_time)

    # Use update_trail but verify it doesn't exit yet to check history
    # To prevent exit, ensure price movement doesn't hit stop or break structure
    # StateVector.null_state might have defaults that trigger structure break if not handled?
    # Actually, L8_confirm=False by default might be okay or trigger break.
    # Let's set L8_confirm=True and L7 same to avoid structure break
    next_state = replace(next_state, L8_confirm=True)
    rider.position.entry_layer_state = replace(rider.position.entry_layer_state, L7_pattern='none')
    next_state = replace(next_state, L7_pattern='none')

    decision = rider.update_trail(101.0, next_state)

    if decision['should_exit']:
        print("Exited early:", decision['exit_reason'])
    else:
        # Verify price history update used next_time ONLY if not exited
        assert len(rider.price_history) == 2
        assert rider.price_history[1][0] == next_time

    # Trigger exit
    exit_time = sim_time + 120.0
    exit_state = replace(StateVector.null_state(), timestamp=exit_time)
    # Force stop hit (long at 100, price drops to 90)
    decision = rider.update_trail(90.0, exit_state)

    assert decision['should_exit']
    markers = decision['regret_markers']

    # Verify exit markers use simulation time
    assert markers.entry_time == sim_time
    assert markers.exit_time == exit_time
    # duration is not a field in RegretMarkers, so calculate it
    assert markers.exit_time - markers.entry_time == 120.0

def test_wave_rider_explicit_timestamp_arg():
    """Test that WaveRider prioritizes explicit timestamp argument."""
    rider = WaveRider(MNQ)
    state = StateVector.null_state() # timestamp=0 by default or something

    explicit_time = 5000.0
    rider.open_position(100.0, 'long', state, timestamp=explicit_time)

    assert rider.position.entry_time == explicit_time

    # Ensure no structure break
    state = replace(state, L8_confirm=True)
    rider.position.entry_layer_state = replace(rider.position.entry_layer_state, L7_pattern='none')
    state = replace(state, L7_pattern='none')

    update_time = 5060.0
    decision = rider.update_trail(101.0, state, timestamp=update_time)

    if not decision['should_exit']:
        assert rider.price_history[-1][0] == update_time
