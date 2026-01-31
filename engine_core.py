"""
ProjectX v2.0 - Main Execution Engine
File: projectx/engine_core.py
"""
from core.bayesian_brain import BayesianBrain, TradeOutcome
from execution.wave_rider import WaveRider
from core.layer_engine import LayerEngine
import time

class ProjectXEngine:
    def __init__(self, asset, use_gpu=True):
        self.asset = asset
        self.prob_table = BayesianBrain()
        self.wave_rider = WaveRider(asset)
        self.fluid_engine = LayerEngine(use_gpu=use_gpu)
        self.daily_pnl = 0.0
        self.MAX_DAILY_LOSS = -200.0
        self.MIN_PROB = 0.80
        self.MIN_CONF = 0.30

    def initialize_session(self, historical_data, user_kill_zones):
        """Initialize static context (L1-L4)"""
        self.fluid_engine.initialize_static_context(historical_data, user_kill_zones)

    def on_tick(self, tick_data: dict):
        if self.daily_pnl <= self.MAX_DAILY_LOSS: return

        # Process State
        current_state = self.fluid_engine.compute_current_state(tick_data)

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
            state=pos.entry_layer_state, # Fixed attribute name (was pos.entry_state)
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
