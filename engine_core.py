"""
ProjectX v2.0 - Main Execution Engine
File: projectx/engine_core.py
"""
from core.bayesian_brain import BayesianBrain, TradeOutcome
from execution.wave_rider import WaveRider
from core.layer_engine_cuda import LayerEngine
import time

class ProjectXEngine:
    def __init__(self, asset):
        self.asset = asset
        self.prob_table = BayesianBrain() [cite: 38]
        self.wave_rider = WaveRider(asset)
        self.fluid_engine = LayerEngine(use_gpu=True)
        self.daily_pnl = 0.0 [cite: 38]
        self.MAX_DAILY_LOSS = -200.0 [cite: 39]
        self.MIN_PROB = 0.80 [cite: 38]
        self.MIN_CONF = 0.30 [cite: 38]

    def on_tick(self, tick_data: dict):
        if self.daily_pnl <= self.MAX_DAILY_LOSS: return [cite: 40]

        # Process State [cite: 41, 42]
        current_state = self.fluid_engine.compute_current_state(tick_data)
        
        # Exit Management [cite: 43]
        if self.wave_rider.position:
            decision = self.wave_rider.update_trail(tick_data['price'], current_state)
            if decision['should_exit']:
                self._close_position(tick_data['price'], decision)
            return

        # Entry Logic [cite: 44, 45, 46]
        prob = self.prob_table.get_probability(current_state)
        conf = self.prob_table.get_confidence(current_state)
        
        if current_state.L9_cascade and prob >= self.MIN_PROB and conf >= self.MIN_CONF:
            self.wave_rider.open_position(tick_data['price'], 'short', current_state) [cite: 47]

    def _close_position(self, price, info):
        pos = self.wave_rider.position
        outcome = TradeOutcome(
            state=pos.entry_state,
            entry_price=pos.entry_price,
            exit_price=price,
            pnl=info['pnl'],
            result='WIN' if info['pnl'] > 0 else 'LOSS',
            timestamp=time.time(),
            exit_reason=info['exit_reason']
        )
        self.prob_table.update(outcome) [cite: 48, 49]
        self.daily_pnl += info['pnl']
        self.wave_rider.position = None