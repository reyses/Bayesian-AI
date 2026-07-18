import os
import sys
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from research.leg_clock.tools.ag_cat_harness import ConceptHarness, run_sweep
from core_v2.FPS.forward_pass_system import BarState

class PivotPointsConcept(ConceptHarness):
    """
    Concept 05: Pivot Points (from the Blacklist).
    
    Hypothesis: 
    Price mean-reverts at standard daily pivot support and resistance levels.
    
    Definition:
    Using a rolling 24-hour window (1440 1-minute bars) as a proxy for Daily HLC.
    Pivot (P) = (High + Low + Close) / 3
    R1 = 2 * P - Low
    S1 = 2 * P - High
    
    If price touches R1 from below -> Short (Reversion)
    If price touches S1 from above -> Long (Reversion)
    """
    
    def reset(self):
        self.highs = []
        self.lows = []
        self.closes = []
        self.prev_close = None
        self.prev_R1 = None
        self.prev_S1 = None
        
    def eval_state(self, state: BarState) -> float:
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0.0
            
        h, l, c = ohlcv['high'], ohlcv['low'], ohlcv['close']
        
        self.highs.append(h)
        self.lows.append(l)
        self.closes.append(c)
        
        if len(self.closes) < 1440:
            self.prev_close = c
            return 0.0
            
        if len(self.closes) > 1440:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
            
        rolling_H = max(self.highs)
        rolling_L = min(self.lows)
        rolling_C = self.closes[-1]
        
        P = (rolling_H + rolling_L + rolling_C) / 3.0
        R1 = 2 * P - rolling_L
        S1 = 2 * P - rolling_H
        
        signal = 0.0
        
        if self.prev_R1 is not None and self.prev_close is not None:
            # Reversion from R1
            if self.prev_close < self.prev_R1 and c >= R1:
                signal = -1.0
            # Reversion from S1
            elif self.prev_close > self.prev_S1 and c <= S1:
                signal = 1.0
                
        self.prev_R1 = R1
        self.prev_S1 = S1
        self.prev_close = c
        
        return signal

if __name__ == '__main__':
    run_sweep(PivotPointsConcept, "05_Pivot_Points", years=['2024', '2025'])
