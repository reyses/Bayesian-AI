import os
import sys

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from research.leg_clock.tools.ag_cat_harness import ConceptHarness, run_sweep
from core_v2.FPS.forward_pass_system import BarState

class SNRBreakoutConcept(ConceptHarness):
    """
    Concept 06: Support/Resistance dynamic breaks (from the Blacklist).
    
    Hypothesis: 
    Breaking out of a rolling N-bar high or low triggers momentum in the direction
    of the breakout.
    
    Definition:
    Lookback = 60 bars (1 hour of 1-minute bars).
    If Prev Close < Prev 60-bar High AND Curr Close > Curr 60-bar High -> Go Long
    If Prev Close > Prev 60-bar Low AND Curr Close < Curr 60-bar Low -> Go Short
    """
    
    def reset(self):
        self.highs = []
        self.lows = []
        self.prev_close = None
        self.prev_max_high = None
        self.prev_min_low = None
        
    def eval_state(self, state: BarState) -> float:
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0.0
            
        h, l, c = ohlcv['high'], ohlcv['low'], ohlcv['close']
        
        self.highs.append(h)
        self.lows.append(l)
        
        if len(self.highs) < 60:
            self.prev_close = c
            return 0.0
            
        if len(self.highs) > 60:
            self.highs.pop(0)
            self.lows.pop(0)
            
        max_high = max(self.highs)
        min_low = min(self.lows)
        
        signal = 0.0
        
        if self.prev_max_high is not None and self.prev_close is not None:
            # Breakout Long
            if self.prev_close < self.prev_max_high and c > max_high:
                signal = 1.0
            # Breakdown Short
            elif self.prev_close > self.prev_min_low and c < min_low:
                signal = -1.0
                
        self.prev_max_high = max_high
        self.prev_min_low = min_low
        self.prev_close = c
        
        return signal

if __name__ == '__main__':
    run_sweep(SNRBreakoutConcept, "06_SNR_Breaks", years=['2024', '2025'])
