"""
Bayesian AI v2.0 - Wave Rider Exit System
File: bayesian_ai/execution/wave_rider.py
"""
import time
from dataclasses import dataclass
from typing import Optional, Dict
from core.state_vector import StateVector

@dataclass
class Position:
    entry_price: float
    entry_time: float
    side: str  # 'long' or 'short'
    stop_loss: float
    high_water_mark: float
    entry_layer_state: StateVector

class WaveRider:
    def __init__(self, asset_profile):
        self.asset = asset_profile
        self.position: Optional[Position] = None

    def open_position(self, entry_price: float, side: str, state: StateVector):
        stop_dist = 20 * self.asset.tick_size
        stop_loss = entry_price + stop_dist if side == 'short' else entry_price - stop_dist
        self.position = Position(entry_price, time.time(), side, stop_loss, entry_price, state)

    def update_trail(self, current_price: float, current_state: StateVector) -> Dict:
        if not self.position: return {'should_exit': False}

        # Update High Water Mark
        if self.position.side == 'short':
            profit = self.position.entry_price - current_price
            self.position.high_water_mark = min(self.position.high_water_mark, current_price)
        else:
            profit = current_price - self.position.entry_price
            self.position.high_water_mark = max(self.position.high_water_mark, current_price)

        profit_usd = profit * self.asset.point_value

        # Adaptive Trail logic
        if profit_usd < 50: trail_ticks = 10
        elif profit_usd < 150: trail_ticks = 20
        else: trail_ticks = 30

        trail_dist = trail_ticks * self.asset.tick_size
        new_stop = self.position.high_water_mark + trail_dist if self.position.side == 'short' else self.position.high_water_mark - trail_dist

        # Check Stop Hit or Structure Break
        stop_hit = (self.position.side == 'short' and current_price >= new_stop) or \
                   (self.position.side == 'long' and current_price <= new_stop)

        structure_broken = self._check_layer_breaks(current_state)

        if stop_hit or structure_broken:
            return {
                'should_exit': True,
                'exit_price': current_price,
                'exit_reason': 'structure_break' if structure_broken else 'trail_stop',
                'pnl': profit_usd
            }

        self.position.stop_loss = new_stop
        return {'should_exit': False, 'current_stop': new_stop}

    def _check_layer_breaks(self, current: StateVector) -> bool:
        entry = self.position.entry_layer_state
        if entry.L7_pattern != current.L7_pattern: return True
        if not current.L8_confirm: return True
        return False
