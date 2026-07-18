import os
import sys
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from research.leg_clock.tools.ag_cat_harness import ConceptHarness, run_sweep
from core_v2.FPS.forward_pass_system import BarState

class BandBounceConcept(ConceptHarness):
    """
    Concept 03: Band-level first-touch bounce (from the Blacklist).
    
    Hypothesis: 
    When price escapes a 20-period, 2-std Bollinger Band for the first time
    after being inside, it will mean-revert.
    
    Definition:
    Lookback = 20 bars (1m).
    Upper = MA + 2*STD
    Lower = MA - 2*STD
    If Prev Close < Prev Upper AND Curr Close > Curr Upper -> Go Short
    If Prev Close > Prev Lower AND Curr Close < Curr Lower -> Go Long
    """
    
    def reset(self):
        self.closes = []
        self.prev_upper = None
        self.prev_lower = None
        self.prev_close = None
    
    def eval_state(self, state: BarState) -> float:
        # We only evaluate on the 1-minute close using the 1m OHLCV data
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0.0
            
        c = ohlcv['close']
        
        self.closes.append(c)
        if len(self.closes) < 20:
            self.prev_close = c
            return 0.0
            
        if len(self.closes) > 20:
            self.closes.pop(0)
            
        ma = np.mean(self.closes)
        std = np.std(self.closes)
        upper = ma + 2 * std
        lower = ma - 2 * std
        
        signal = 0.0
        
        if self.prev_upper is not None and self.prev_close is not None:
            # First touch upper -> short
            if self.prev_close <= self.prev_upper and c > upper:
                signal = -1.0
            # First touch lower -> long
            elif self.prev_close >= self.prev_lower and c < lower:
                signal = 1.0
                
        self.prev_upper = upper
        self.prev_lower = lower
        self.prev_close = c
        
        return signal

if __name__ == '__main__':
    run_sweep(BandBounceConcept, "03_Band_Bounce", years=['2024', '2025'])
