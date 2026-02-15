"""
Tests for WaveRider position management.
Addresses ISSUE-003: Missing test coverage for execution logic.
"""
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from training.wave_rider import WaveRider, Position, RegretAnalyzer
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

def test_regret_analysis_loss_no_peak():
    """
    Test that a trade with a loss and no potential profit (peak == entry)
    is NOT classified as optimal and has 0% efficiency.
    """
    analyzer = RegretAnalyzer()

    # User scenario:
    # Side: LONG
    # Entry: 21864.75
    # Exit:  21862.25
    # Peak:  21864.75 (Same as Entry)
    # Actual PnL: -5.00

    entry_price = 21864.75
    exit_price = 21862.25
    entry_time = 1000
    exit_time = 1005
    side = 'long'
    exit_reason = 'trail_stop'

    # Price history shows immediate drop or staying flat then drop
    price_history = [
        (1000, 21864.75),
        (1001, 21863.00),
        (1002, 21862.50),
        (1005, 21862.25)
    ]
    tick_value = 0.5 # Derived from user report (-5.00 PnL for -2.5 pts / 0.25 tick size = 10 ticks)

    markers = analyzer.analyze_exit(
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=entry_time,
        exit_time=exit_time,
        side=side,
        exit_reason=exit_reason,
        price_history=price_history,
        tick_value=tick_value
    )

    # The bug was that efficiency was 1.0 (100%) and regret type was 'optimal'
    # The fix should make efficiency 0.0 (0%) and regret type NOT 'optimal'

    assert markers.exit_efficiency == 0.0, f"Expected 0.0 efficiency for loss, got {markers.exit_efficiency}"
    assert markers.regret_type != 'optimal', f"Expected non-optimal regret type for loss, got {markers.regret_type}"
    assert markers.potential_max_pnl == 0.0
    assert markers.actual_pnl < 0

def test_regret_analysis_breakeven_no_peak():
    """
    Test that a trade with 0 PnL and no potential profit (peak == entry)
    is considered optimal (100% efficiency).
    """
    analyzer = RegretAnalyzer()

    entry_price = 100.0
    exit_price = 100.0
    price_history = [(1000, 100.0), (1005, 100.0)]

    markers = analyzer.analyze_exit(
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=1000,
        exit_time=1005,
        side='long',
        exit_reason='flat',
        price_history=price_history,
        tick_value=1.0
    )

    assert markers.exit_efficiency == 1.0
    assert markers.regret_type == 'optimal'
