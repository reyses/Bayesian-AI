import os
import sys
import numpy as np
from collections import deque

# Ensure we can import the harness
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ag_bayes_harness import BayesianHarness

class SqueezeConcept:
    def __init__(self):
        self.period = 20
        self.prices = deque(maxlen=self.period)
        self.bandwidths = deque(maxlen=60)  # Track last 60 bandwidths
        
    def eval_state(self, state):
        ohlcv = state.ohlcv_1m
        if ohlcv is None:
            return 0
            
        c = ohlcv['close']
        self.prices.append(c)
        
        event = 0
        if len(self.prices) == self.period:
            sma = np.mean(self.prices)
            std = np.std(self.prices)
            
            if sma > 0:
                bw = (4 * std) / sma
                self.bandwidths.append(bw)
                
                # Check for squeeze: current bandwidth is the lowest in the last 60 periods
                if len(self.bandwidths) == 60:
                    if bw == min(self.bandwidths):
                        # Squeeze detected
                        event = 1  # Volatility event
                        
        return event

if __name__ == '__main__':
    harness = BayesianHarness(
        concept_name='Squeeze_State',
        article_citation='bollinger-bands-explained-a-futures-traders-guide.md',
        registered_response='Volatility Expansion (Direction-Free Breakout)',
        horizon_bars=60,
        event_mode='volatility'
    )
    
    harness.run_sweep(SqueezeConcept, years=['2024'])
