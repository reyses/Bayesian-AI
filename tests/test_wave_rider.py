"""
Tests for WaveRider position management.
Addresses ISSUE-003: Missing test coverage for execution logic.
"""
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from training.wave_rider import WaveRider, Position
from config.symbols import MNQ
from core.state_vector import StateVector
from dataclasses import replace

def test_open_position():
    """Test that a position is correctly opened with stop loss calculated."""
    rider = WaveRider(MNQ)
    state = StateVector.null_state()
    
    # Open a short position at 100.0
    rider.open_position(100.0, 'short', state)
    
    assert rider.position is not None
    assert rider.position.entry_price == 100.0
    assert rider.position.side == 'short'
    # Assuming default stop loss of 20 ticks (5.0 points for NQ/MNQ usually, but checking logic)
    # If 1 tick = 0.25, 20 ticks = 5.0. Short from 100 -> Stop at 105.
    assert rider.position.stop_loss == 105.0

def test_adaptive_trail():
    """Test that the trailing stop tightens as profit increases."""
    rider = WaveRider(MNQ)
    # Use L8_confirm=True to prevent immediate structure break exit
    state = replace(StateVector.null_state(), L8_confirm=True)
    
    rider.open_position(100.0, 'short', state)
    
    # Scenario 1: Small profit (< $50), loose trail
    decision = rider.update_trail(98.0, state) # 2 points profit
    assert not decision['should_exit']
    
    # Scenario 2: Small profit (< $50), tight trail (10 ticks = 2.5 pts)
    # Manually set high water mark to simulate price having gone down to 90.0
    rider.position.high_water_mark = 90.0 
    
    # Price snaps back to 93.0 (3.0 points pullback from low)
    # Since trail is tight (2.5 pts), this SHOULD trigger exit
    decision = rider.update_trail(93.0, state)
    assert decision['should_exit']