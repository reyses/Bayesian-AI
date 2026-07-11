import os
import sys

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from research.leg_clock.tools.ag_cat_harness import ConceptHarness, run_sweep
from core_v2.FPS.forward_pass_system import BarState

class CandleShapeConcept(ConceptHarness):
    """
    Concept 01: Candle wick/body shapes (from the Blacklist).
    
    Hypothesis: 
    Large lower wicks (pin bars) indicate buying pressure -> go long.
    Large upper wicks indicate selling pressure -> go short.
    
    Definition:
    Wick size must be > 2x the body size.
    Lower wick = min(Open, Close) - Low
    Upper wick = High - max(Open, Close)
    Body = abs(Open - Close)
    """
    
    def eval_state(self, state: BarState) -> float:
        # We only evaluate on the 1-minute close using the 1m OHLCV data
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0.0
            
        o, h, l, c = ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close']
        
        body = abs(o - c)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        # Avoid division by zero
        if body == 0:
            body = 0.25 # 1 tick minimum for ratio math
            
        # Pin bar thresholds
        if lower_wick > (2 * body) and upper_wick < body:
            return 1.0  # Bullish pin bar
            
        if upper_wick > (2 * body) and lower_wick < body:
            return -1.0 # Bearish pin bar
            
        return 0.0

if __name__ == '__main__':
    run_sweep(CandleShapeConcept, "01_Candle_Shapes", years=['2024', '2025'])
