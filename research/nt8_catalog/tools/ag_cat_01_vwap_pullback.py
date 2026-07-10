import os
import sys

# Ensure we can import the harness
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ag_bayes_harness import BayesianHarness

class VWAPPullbackConcept:
    def __init__(self):
        self.sum_pv = 0.0
        self.sum_v = 0.0
        self.prev_price = None

    def eval_state(self, state):
        if not hasattr(self, 'current_day'):
            self.current_day = state.day
        if state.day != self.current_day:
            self.current_day = state.day
            self.sum_pv = 0.0
            self.sum_v = 0.0
        ohlcv = state.ohlcv_1m
        if ohlcv is None:
            return 0
        vol = ohlcv['volume']
            
        self.sum_pv += state.price * vol
        self.sum_v += vol
        
        vwap = self.sum_pv / self.sum_v if self.sum_v > 0 else state.price
        
        event = 0
        
        if self.prev_price is not None:
            # Did we cross or touch the VWAP?
            # From above (pullback to support) -> Expect Bounce UP (event = 1)
            if self.prev_price > vwap and state.price <= vwap:
                event = 1
            # From below (rally to resistance) -> Expect Bounce DOWN (event = -1)
            elif self.prev_price < vwap and state.price >= vwap:
                event = -1
                
        self.prev_price = state.price
        return event

if __name__ == '__main__':
    harness = BayesianHarness(
        concept_name='VWAP_Pullbacks',
        article_citation='what-is-volume-weighted-average-price-vwap.md',
        registered_response='Directional first-touch bounce (+2 sigma)',
        horizon_bars=60
    )
    
    harness.run_sweep(VWAPPullbackConcept, years=['2024']) # Only 2024 as training set
