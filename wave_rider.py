"""
ProjectX v2.0 - Wave Rider Exit System
File: projectx/execution/wave_rider.py
"""
import time
from dataclasses import dataclass
from typing import Optional, Dict
from core.state_vector import StateVector # Using existing StateVector

@dataclass
class Position:
    entry_price: float
    entry_time: float
    side: str  # 'long' or 'short'
    stop_loss: float
    high_water_mark: float
    entry_state: StateVector

class WaveRider:
    def __init__(self, asset_profile):
        self.asset = asset_profile
        self.position: Optional[Position] = None
    
    def open_position(self, entry_price: float, side: str, state: StateVector):
        stop_dist = 20 * self.asset.tick_size [cite: 27]
        stop_loss = entry_price + stop_dist if side == 'short' else entry_price - stop_dist [cite: 27, 28]
        self.position = Position(entry_price, time.time(), side, stop_loss, entry_price, state)
    
    def update_trail(self, current_price: float, current_state: StateVector) -> Dict:
        if not self.position: return {'should_exit': False}
        
        # Update High Water Mark [cite: 29, 30]
        if self.position.side == 'short':
            profit = self.position.entry_price - current_price
            self.position.high_water_mark = min(self.position.high_water_mark, current_price)
        else:
            profit = current_price - self.position.entry_price
            self.position.high_water_mark = max(self.position.high_water_mark, current_price)
            
        profit_usd = profit * self.asset.point_value [cite: 30]
        
        # Adaptive Trail logic [cite: 31, 32]
        if profit_usd < 50: trail_ticks = 10
        elif profit_usd < 150: trail_ticks = 20
        else: trail_ticks = 30
        
        trail_dist = trail_ticks * self.asset.tick_size
        new_stop = self.position.high_water_mark + trail_dist if self.position.side == 'short' else self.position.high_water_mark - trail_dist [cite: 32]
        
        # Check Stop Hit or Structure Break [cite: 33, 34, 36, 37]
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
        
        self.position.stop_loss = new_stop [cite: 35]
        return {'should_exit': False, 'current_stop': new_stop}

    def _check_layer_breaks(self, current: StateVector) -> bool:
        entry = self.position.entry_state
        if entry.L7_pattern != current.L7_pattern: return True [cite: 36]
        if not current.L8_confirm: return True [cite: 37]
        return False