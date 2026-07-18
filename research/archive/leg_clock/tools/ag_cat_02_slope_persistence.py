import os
import sys

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from research.leg_clock.tools.ag_cat_harness import ConceptHarness, run_sweep
from core_v2.FPS.forward_pass_system import BarState

class SlopePersistenceConcept(ConceptHarness):
    """
    Concept 02: Bar-to-bar slope persistence (from the Blacklist).
    
    Hypothesis: 
    Momentum persists. If N consecutive bars close in the same direction, 
    the trend will continue.
    
    Definition:
    Lookback = 3 bars.
    If 3 consecutive bars are bullish (Close > Open) -> go long.
    If 3 consecutive bars are bearish (Close < Open) -> go short.
    """
    
    def reset(self):
        self.history = []
    
    def eval_state(self, state: BarState) -> float:
        # We only evaluate on the 1-minute close using the 1m OHLCV data
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0.0
            
        o, c = ohlcv['open'], ohlcv['close']
        
        # 1 = Bullish, -1 = Bearish, 0 = Doji
        if c > o:
            direction = 1
        elif c < o:
            direction = -1
        else:
            direction = 0
            
        self.history.append(direction)
        
        if len(self.history) < 3:
            return 0.0
            
        # Keep only the last 3
        if len(self.history) > 3:
            self.history.pop(0)
            
        if sum(self.history) == 3:
            return 1.0  # 3 consecutive bullish bars
            
        if sum(self.history) == -3:
            return -1.0 # 3 consecutive bearish bars
            
        return 0.0

if __name__ == '__main__':
    run_sweep(SlopePersistenceConcept, "02_Slope_Persistence", years=['2024', '2025'])
