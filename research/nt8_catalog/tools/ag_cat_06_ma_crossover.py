import os
import sys

# Ensure we can import the harness
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ag_bayes_harness import BayesianHarness

class MACrossoverConcept:
    def __init__(self):
        self.ema_fast = None
        self.ema_slow = None
        self.alpha_fast = 2.0 / (20 + 1)
        self.alpha_slow = 2.0 / (60 + 1)
        self.prev_fast = None
        self.prev_slow = None

    def eval_state(self, state):
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0
            
        c = ohlcv['close']
        
        if self.ema_fast is None:
            self.ema_fast = c
            self.ema_slow = c
        else:
            self.ema_fast = (c - self.ema_fast) * self.alpha_fast + self.ema_fast
            self.ema_slow = (c - self.ema_slow) * self.alpha_slow + self.ema_slow
            
        event = 0
        
        if self.prev_fast is not None and self.prev_slow is not None:
            # Fast crosses above Slow -> Expect UP
            if self.prev_fast <= self.prev_slow and self.ema_fast > self.ema_slow:
                event = 1
            # Fast crosses below Slow -> Expect DOWN
            elif self.prev_fast >= self.prev_slow and self.ema_fast < self.ema_slow:
                event = -1
                
        self.prev_fast = self.ema_fast
        self.prev_slow = self.ema_slow
        return event

if __name__ == '__main__':
    harness = BayesianHarness(
        concept_name='MA_Crossover',
        article_citation='how-to-choose-your-technical-indicators.md',
        registered_response='Trend Continuation (+2 sigma)',
        horizon_bars=60
    )
    
    harness.run_sweep(MACrossoverConcept, years=['2024'])
