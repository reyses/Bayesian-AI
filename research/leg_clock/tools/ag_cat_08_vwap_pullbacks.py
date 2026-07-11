import os
import sys

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from research.leg_clock.tools.ag_cat_harness import ConceptHarness, run_sweep
from core_v2.FPS.forward_pass_system import BarState

class VWAPPullbackConcept(ConceptHarness):
    """
    Concept 08: VWAP pullbacks (from the Blacklist).
    
    Hypothesis: 
    Price pulls back to the daily VWAP and bounces.
    
    Definition:
    Calculate Intraday VWAP (resets daily).
    If Prev Close > VWAP and Curr Low <= VWAP and Curr Close > VWAP -> Long
    If Prev Close < VWAP and Curr High >= VWAP and Curr Close < VWAP -> Short
    """
    
    def reset(self):
        self.cum_pv = 0.0
        self.cum_v = 0.0
        self.prev_vwap = None
        self.prev_close = None
        
    def eval_state(self, state: BarState) -> float:
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0.0
            
        h, l, c, v = ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']
        
        tp = (h + l + c) / 3.0
        self.cum_pv += tp * v
        self.cum_v += v
        
        if self.cum_v == 0:
            return 0.0
            
        vwap = self.cum_pv / self.cum_v
        
        signal = 0.0
        
        if self.prev_vwap is not None and self.prev_close is not None:
            # Pullback to VWAP and bounce Long
            if self.prev_close > self.prev_vwap and l <= vwap and c > vwap:
                signal = 1.0
            # Pullback to VWAP and bounce Short
            elif self.prev_close < self.prev_vwap and h >= vwap and c < vwap:
                signal = -1.0
                
        self.prev_vwap = vwap
        self.prev_close = c
        
        return signal

if __name__ == '__main__':
    run_sweep(VWAPPullbackConcept, "08_VWAP_Pullbacks", years=['2024', '2025'])
