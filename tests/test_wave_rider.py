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
    is classified as 'wrong_direction' and has 0% efficiency.
    """
    analyzer = RegretAnalyzer()

    entry_price = 21864.75
    exit_price = 21862.25
    entry_time = 1000
    exit_time = 1005
    side = 'long'
    exit_reason = 'trail_stop'

    # Price history shows immediate drop
    price_history = [
        (1000, 21864.75),
        (1001, 21863.00),
        (1002, 21862.50),
        (1005, 21862.25)
    ]
    tick_value = 0.5

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

    assert markers.exit_efficiency == 0.0
    assert markers.regret_type == 'wrong_direction'
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

def test_delayed_regret_analysis_workflow():
    """
    Test the full workflow:
    1. Trade Exit -> Pending Review created
    2. Time advances -> Review processes
    3. Lookahead logic works (detects Early Exit)
    """
    # Use '15s' timeframe -> Wait = 60s * 5 = 300s
    rider = WaveRider(MNQ, timeframe='15s')
    # Force calibration every 1 trade
    rider.calibration_interval = 1

    state = StateVector.null_state()
    start_time = 1000.0

    # 1. Open Long Trade at 100
    rider.open_position(100.0, 'long', state)
    rider.position.entry_time = start_time # Mock time

    # 2. Exit Trade at 105 (Profit 5 pts)
    # Mock current time = 1060 (1 min later)
    # exit efficiency without lookahead: 100% (peak=105, exit=105)
    # But wait, update_trail relies on time.time() internally?
    # No, open_position calls time.time(). update_trail uses time.time().
    # We must patch time.time() or just rely on relative logic if we can control it.
    # WaveRider uses time.time() in update_trail.
    # Let's mock time.time

    with pytest.MonkeyPatch.context() as m:
        m.setattr(time, 'time', lambda: 1060.0)

        # Trigger exit (structure break for simplicity)
        state_break = replace(StateVector.null_state(), L7_pattern='changed')
        # update_trail clears history. So we must populate history first.
        rider.price_history = [(1000.0, 100.0), (1030.0, 102.0), (1060.0, 105.0)]

        result = rider.update_trail(105.0, state_break)

        assert result['should_exit']
        assert len(rider.pending_reviews) == 1
        review = rider.pending_reviews[0]
        assert review.exit_price == 105.0
        # Wait duration for 15s TF is 300s. Due time = 1060 + 300 = 1360.
        assert review.review_due_time == 1360.0

    # 3. Simulate Post-Trade Data (Price goes to 110 at 1200)
    # This means we exited Early (at 105), peak is 110.
    # We feed data via process_pending_reviews

    # T=1100, Price=106
    rider.process_pending_reviews(1100.0, 106.0)
    assert len(rider.pending_reviews) == 1 # Still pending

    # T=1200, Price=110 (True Peak)
    rider.process_pending_reviews(1200.0, 110.0)

    # T=1300, Price=108
    rider.process_pending_reviews(1300.0, 108.0)

    # T=1360 (Due Time), Price=107
    # This should trigger analysis and remove from pending

    # Capture print output or check calibration stats
    # We set calibration_interval=1, so it should calibrate immediately after this review
    trades_before = rider.trades_since_calibration # Should be 1 (incremented at exit)
    # Wait, in new code:
    # update_trail: total_trades += 1. calibration happens in process_pending_reviews.
    # So trades_since_calibration starts at 0?
    # Let's check code:
    # update_trail: total_trades += 1. (No trades_since_calibration update there anymore?)
    # process_pending: trades_since_calibration += 1. If >= interval, calibrate.

    # So before process completes, trades_since_calibration is 0 (assuming fresh instance)

    rider.process_pending_reviews(1360.0, 107.0)

    assert len(rider.pending_reviews) == 0
    # Should have calibrated
    # The analyzer history should have the Delayed marker
    # The last marker in history should have peak=110 (from post-trade data)
    last_marker = rider.regret_analyzer.regret_history[-1] # The delayed one
    # Note: update_trail calls analyze_exit (preliminary) -> adds to history
    # process_pending_reviews calls analyze_exit (delayed) -> adds to history
    # So history has 2 entries per trade.

    prelim_marker = rider.regret_analyzer.regret_history[-2]
    delayed_marker = rider.regret_analyzer.regret_history[-1]

    assert prelim_marker.peak_favorable == 105.0 # At exit time
    assert delayed_marker.peak_favorable == 110.0 # From lookahead

    assert delayed_marker.regret_type == 'closed_too_early'
    assert prelim_marker.regret_type == 'optimal' # 105 exit / 105 peak = 1.0 efficiency
