import os
import sys
import numpy as np

# Ensure we can import the harness
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ag_bayes_harness import BayesianHarness

class APZTouchesConcept:
    def __init__(self):
        self.closes = []
        self.trs = []
        self.prev_close = None
        self.alpha = 2.0 / (20 + 1)
        
        self.ema_c = None
        self.ema_tr = None

    def eval_state(self, state):
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0
            
        c = ohlcv['close']
        h = ohlcv['high']
        l = ohlcv['low']
        
        if self.prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
            
        self.closes.append(c)
        self.trs.append(tr)
        
        if len(self.closes) < 20:
            self.prev_close = c
            return 0
            
        if self.ema_c is None:
            self.ema_c = np.mean(self.closes)
            self.ema_tr = np.mean(self.trs)
        else:
            self.ema_c = (c - self.ema_c) * self.alpha + self.ema_c
            self.ema_tr = (tr - self.ema_tr) * self.alpha + self.ema_tr
            
        apz_upper = self.ema_c + 2.0 * self.ema_tr
        apz_lower = self.ema_c - 2.0 * self.ema_tr
        
        event = 0
        
        if self.prev_close is not None:
            # Touch of upper band -> expect mean reversion DOWN (-1)
            if self.prev_close < apz_upper and c >= apz_upper:
                event = -1
            # Touch of lower band -> expect mean reversion UP (+1)
            elif self.prev_close > apz_lower and c <= apz_lower:
                event = 1
                
        self.prev_close = c
        return event

if __name__ == '__main__':
    harness = BayesianHarness(
        concept_name='APZ_Touches',
        article_citation='adaptive-price-zones-indicator.md',
        registered_response='Directional first-touch bounce (Mean Reversion)',
        horizon_bars=60
    )
    
    harness.run_sweep(APZTouchesConcept, years=['2024'])
