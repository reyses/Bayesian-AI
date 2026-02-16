"""
Tests for WaveRider position management.
Addresses ISSUE-003: Missing test coverage for execution logic.
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

def test_delayed_regret_analysis_early_exit():
    """Test 'closed_too_early' detection with delayed lookahead."""
    rider = WaveRider(MNQ)
    rider.review_wait_time = 60.0 # 60 seconds wait
    state = replace(StateVector.null_state(), L8_confirm=True)

    start_time = time.time()
    rider.open_position(100.0, 'long', state)

    # Price moves up to 110 (Small Peak)
    rider.process_pending_reviews(start_time + 10, 110.0)
    rider.update_trail(110.0, state)

    # Price drops to 105 (Exit)
    exit_time = start_time + 20
    decision = rider.update_trail(105.0, state, timestamp=exit_time)
    assert decision['should_exit']
    assert len(rider.pending_reviews) == 1

    # Lookahead: Price goes to 120 (True Peak)
    rider.process_pending_reviews(start_time + 40, 120.0)

    # Trigger review
    rider.process_pending_reviews(start_time + 90, 115.0)

    assert len(rider.pending_reviews) == 0
    assert len(rider.regret_analyzer.regret_history) == 1

    markers = rider.regret_analyzer.regret_history[0]
    assert markers.peak_favorable == pytest.approx(120.0)
    assert markers.regret_type == 'closed_too_early'
    assert markers.peak_favorable_time > markers.exit_time

def test_delayed_regret_analysis_late_exit():
    """Test 'closed_too_late' detection."""
    rider = WaveRider(MNQ)
    rider.review_wait_time = 60.0
    state = replace(StateVector.null_state(), L8_confirm=True)

    start_time = time.time()
    rider.open_position(100.0, 'long', state)

    # Price moves up to 120 (Major Peak)
    rider.process_pending_reviews(start_time + 10, 120.0)
    rider.update_trail(120.0, state)

    # Price drops significantly to 105 (Exit)
    # Profit was 20 pts ($40). Exit profit 5 pts ($10).
    # Gave back 15 pts ($30). Potential 20 pts ($40).
    # Gave back > 20% of Potential (30 > 8).

    exit_time = start_time + 30
    decision = rider.update_trail(105.0, state, timestamp=exit_time) # Exit due to trail stop hit

    # Lookahead: Price stays low
    rider.process_pending_reviews(start_time + 50, 100.0)

    # Trigger review
    rider.process_pending_reviews(start_time + 100, 100.0)

    markers = rider.regret_analyzer.regret_history[0]
    assert markers.peak_favorable == pytest.approx(120.0)
    assert markers.regret_type == 'closed_too_late'
    # Peak was before exit
    assert markers.peak_favorable_time < markers.exit_time

def test_regret_analysis_wrong_direction():
    """Test 'wrong_direction' detection for losses."""
    rider = WaveRider(MNQ)
    rider.review_wait_time = 60.0
    state = replace(StateVector.null_state(), L8_confirm=True)

    start_time = time.time()
    rider.open_position(100.0, 'long', state)

    # Price goes DOWN immediately to 95.0
    rider.process_pending_reviews(start_time + 10, 98.0)
    rider.update_trail(98.0, state)

    # Hit stop loss (assume 20 ticks = 5 pts) -> 95.0
    exit_time = start_time + 20
    decision = rider.update_trail(95.0, state, timestamp=exit_time)

    if not decision['should_exit']:
        rider.position.stop_loss = 96.0
        decision = rider.update_trail(95.0, state, timestamp=exit_time)

    assert decision['should_exit']

    # Trigger review
    rider.process_pending_reviews(start_time + 100, 95.0)

    markers = rider.regret_analyzer.regret_history[0]
    assert markers.actual_pnl < 0
    assert markers.regret_type == 'wrong_direction'

def test_efficiency_zero_potential_loss():
    """Test efficiency is 0.0 when potential is 0 and outcome is loss."""
    rider = WaveRider(MNQ)
    start_time = time.time()

    # Short at 100. Price goes UP immediately (Bad).
    # Entry 100. Exit 105 (Loss 5 pts).
    # Peak favorable = 100 (Entry). Potential = 0.

    history = [
        (start_time, 100.0),
        (start_time + 10, 102.0),
        (start_time + 20, 105.0)
    ]

    markers = rider.regret_analyzer.analyze_exit(
        entry_price=100.0,
        exit_price=105.0, # Loss for Short
        entry_time=start_time,
        exit_time=start_time+20,
        side='short',
        exit_reason='stop',
        price_history=history,
        tick_value=rider.asset.tick_value
    )

    assert markers.potential_max_pnl == 0.0
    assert markers.actual_pnl < 0
    assert markers.exit_efficiency == 0.0
    assert markers.regret_type == 'wrong_direction'

def test_update_trail_returns_trade_outcome():
    """Test that update_trail returns a TradeOutcome object upon exit."""
    rider = WaveRider(MNQ)
    state = replace(StateVector.null_state(), L8_confirm=True)
    template_id = "test_template_123"

    # Open position with template_id
    rider.open_position(100.0, 'long', state, template_id=template_id)

    # Force exit
    exit_time = time.time() + 10
    decision = rider.update_trail(90.0, state, timestamp=exit_time) # Stop loss hit (20 ticks = 5 pts)

    assert decision['should_exit']
    assert 'trade_outcome' in decision, "Return value should contain 'trade_outcome'"

    outcome = decision['trade_outcome']
    # Check fields
    assert outcome.entry_price == 100.0
    assert outcome.exit_price == 90.0
    assert outcome.result == 'LOSS'
    assert outcome.exit_reason == 'trail_stop' # or stop_loss
    assert outcome.state == state

    # Verify template_id persistence
    assert outcome.template_id == template_id
