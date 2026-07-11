import os
import sys
import numpy as np
from collections import defaultdict

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from research.leg_clock.tools.ag_cat_harness import ConceptHarness, run_sweep
from core_v2.FPS.forward_pass_system import BarState

class VolProfilePOCConcept(ConceptHarness):
    """
    Concept 11: Volume Profile POC tests.
    
    Hypothesis: 
    Price acts as a magnet to the intraday Point of Control (POC). If price is 
    far from the POC and starts reversing, it will revert to the POC.
    
    Definition:
    Track daily Volume Profile (price binned to integers).
    If Close > POC + 20 and Close < Prev Close -> Short (revert to POC)
    If Close < POC - 20 and Close > Prev Close -> Long (revert to POC)
    """
    
    def reset(self):
        self.vp = defaultdict(float)
        self.poc_price = None
        self.poc_vol = 0.0
        self.prev_close = None
        
    def eval_state(self, state: BarState) -> float:
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0.0
            
        c = ohlcv['close']
        v = ohlcv['volume']
        
        # Bin to integer price
        p_bin = int(c)
        self.vp[p_bin] += v
        
        if self.vp[p_bin] > self.poc_vol:
            self.poc_vol = self.vp[p_bin]
            self.poc_price = p_bin
            
        signal = 0.0
        
        if self.poc_price is not None and self.prev_close is not None:
            # Revert from above
            if c > self.poc_price + 20 and c < self.prev_close:
                signal = -1.0
            # Revert from below
            elif c < self.poc_price - 20 and c > self.prev_close:
                signal = 1.0
                
        self.prev_close = c
        return signal

if __name__ == '__main__':
    run_sweep(VolProfilePOCConcept, "11_VolProfile_POC", years=['2024', '2025'])
