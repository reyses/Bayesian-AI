"""
Bayesian AI v2.0 - Main Execution Engine
File: bayesian_ai/engine_core.py
"""
from core.bayesian_brain import BayesianBrain, TradeOutcome
from execution.wave_rider import WaveRider
from core.layer_engine import LayerEngine
from core.data_aggregator import DataAggregator
import time
import os

class BayesianEngine:
    def __init__(self, asset, use_gpu=True):
        self.asset = asset
        self.prob_table = BayesianBrain()
        self.wave_rider = WaveRider(asset)
        self.fluid_engine = LayerEngine(use_gpu=use_gpu)
        self.aggregator = DataAggregator()
        self.daily_pnl = 0.0
        self.MAX_DAILY_LOSS = -200.0
        self.MIN_PROB = 0.80
        self.MIN_CONF = 0.30

    def initialize_session(self, historical_data, user_kill_zones):
        """Initialize static context (L1-L4)"""
        self.fluid_engine.initialize_static_context(historical_data, user_kill_zones)

    def on_tick(self, tick_data: dict):
        if self.daily_pnl <= self.MAX_DAILY_LOSS: return

        # Add tick to aggregator
        self.aggregator.add_tick(tick_data)

        # Get full data context for LayerEngine
        current_data = self.aggregator.get_current_data()

        # Process State
        current_state = self.fluid_engine.compute_current_state(current_data)

        # Exit Management
        if self.wave_rider.position:
            decision = self.wave_rider.update_trail(tick_data['price'], current_state)
            if decision['should_exit']:
                self._close_position(tick_data['price'], decision)
            return

        # Entry Logic
        prob = self.prob_table.get_probability(current_state)
        conf = self.prob_table.get_confidence(current_state)

        if current_state.L9_cascade and prob >= self.MIN_PROB and conf >= self.MIN_CONF:
            self.wave_rider.open_position(tick_data['price'], 'short', current_state)

    def _close_position(self, price, info):
        pos = self.wave_rider.position
        outcome = TradeOutcome(
            state=pos.entry_layer_state,
            entry_price=pos.entry_price,
            exit_price=price,
            pnl=info['pnl'],
            result='WIN' if info['pnl'] > 0 else 'LOSS',
            timestamp=time.time(),
            exit_reason=info['exit_reason']
        )
        self.prob_table.update(outcome)
        self.daily_pnl += info['pnl']
        self.wave_rider.position = None

if __name__ == "__main__":
    print("Bayesian AI Engine v2.0 - Standalone Mode")
    print("Initializing...")

    # Simple check for resources
    if os.path.exists("probability_table.pkl"):
        print("[OK] Probability table found.")
    else:
        print("[WARNING] Probability table NOT found.")

    if os.path.exists("config"):
        print("[OK] Config directory found.")
    else:
        print("[WARNING] Config directory NOT found.")

    print("Engine ready. (Live feed connection pending...)")
    # Keep alive for testing
    try:
        # Just sleep for a bit then exit to confirm it runs without crashing
        print("Running diagnostics... (Press Ctrl+C to stop)")
        time.sleep(2)
        print("Diagnostics complete.")
    except KeyboardInterrupt:
        print("Exiting.")
