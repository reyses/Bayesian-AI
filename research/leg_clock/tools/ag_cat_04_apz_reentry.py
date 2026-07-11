import os
import sys
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from research.leg_clock.tools.ag_cat_harness import ConceptHarness, run_sweep
from core_v2.FPS.forward_pass_system import BarState

class APZReentryConcept(ConceptHarness):
    """
    Concept 04: APZ re-entry confirmation (from the Blacklist).
    
    Hypothesis: 
    Adaptive Price Zones (APZ) use True Range for bands. When price leaves the band
    and then crosses back INSIDE the band, it confirms a reversal.
    
    Definition:
    EMA period = 20
    Band deviation = 2.0 * EMA(True Range)
    If Prev Close > Prev APZ Upper AND Curr Close < Curr APZ Upper -> Go Short
    If Prev Close < Prev APZ Lower AND Curr Close > Curr APZ Lower -> Go Long
    """
    
    def reset(self):
        self.closes = []
        self.trs = []
        self.prev_close = None
        self.prev_apz_upper = None
        self.prev_apz_lower = None
        self.alpha = 2.0 / (20 + 1)
        
        self.ema_c = None
        self.ema_tr = None
    
    def eval_state(self, state: BarState) -> float:
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0.0
            
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
            return 0.0
            
        if self.ema_c is None:
            self.ema_c = np.mean(self.closes)
            self.ema_tr = np.mean(self.trs)
        else:
            self.ema_c = (c - self.ema_c) * self.alpha + self.ema_c
            self.ema_tr = (tr - self.ema_tr) * self.alpha + self.ema_tr
            
        apz_upper = self.ema_c + 2.0 * self.ema_tr
        apz_lower = self.ema_c - 2.0 * self.ema_tr
        
        signal = 0.0
        
        if self.prev_apz_upper is not None:
            # Re-entry inside upper band -> Short
            if self.prev_close > self.prev_apz_upper and c < apz_upper:
                signal = -1.0
            # Re-entry inside lower band -> Long
            elif self.prev_close < self.prev_apz_lower and c > apz_lower:
                signal = 1.0
                
        self.prev_apz_upper = apz_upper
        self.prev_apz_lower = apz_lower
        self.prev_close = c
        
        return signal

if __name__ == '__main__':
    run_sweep(APZReentryConcept, "04_APZ_Reentry", years=['2024', '2025'])
