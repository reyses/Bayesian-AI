import os
import sys

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from research.leg_clock.tools.ag_cat_harness import ConceptHarness, run_sweep
from core_v2.FPS.forward_pass_system import BarState

class MACrossoverConcept(ConceptHarness):
    """
    Concept 07: Multi-timeframe moving average crossover (from the Blacklist).
    
    Hypothesis: 
    A fast EMA crossing a slow EMA signals a change in trend.
    
    Definition:
    Fast EMA = 20 bars (20 minutes).
    Slow EMA = 60 bars (60 minutes).
    If Prev Fast < Prev Slow AND Curr Fast > Curr Slow -> Go Long
    If Prev Fast > Prev Slow AND Curr Fast < Curr Slow -> Go Short
    """
    
    def reset(self):
        self.ema_fast = None
        self.ema_slow = None
        self.alpha_fast = 2.0 / (20 + 1)
        self.alpha_slow = 2.0 / (60 + 1)
        self.prev_fast = None
        self.prev_slow = None
        
    def eval_state(self, state: BarState) -> float:
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0.0
            
        c = ohlcv['close']
        
        if self.ema_fast is None:
            self.ema_fast = c
            self.ema_slow = c
        else:
            self.ema_fast = (c - self.ema_fast) * self.alpha_fast + self.ema_fast
            self.ema_slow = (c - self.ema_slow) * self.alpha_slow + self.ema_slow
            
        signal = 0.0
        
        if self.prev_fast is not None and self.prev_slow is not None:
            # Fast crosses above Slow
            if self.prev_fast <= self.prev_slow and self.ema_fast > self.ema_slow:
                signal = 1.0
            # Fast crosses below Slow
            elif self.prev_fast >= self.prev_slow and self.ema_fast < self.ema_slow:
                signal = -1.0
                
        self.prev_fast = self.ema_fast
        self.prev_slow = self.ema_slow
        
        return signal

if __name__ == '__main__':
    run_sweep(MACrossoverConcept, "07_MA_Crossover", years=['2024', '2025'])
