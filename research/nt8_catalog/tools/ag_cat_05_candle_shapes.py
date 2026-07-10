import os
import sys

# Ensure we can import the harness
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ag_bayes_harness import BayesianHarness

class CandleShapeConcept:
    def eval_state(self, state):
        ohlcv = state.ohlcv_1m
        if ohlcv is None or ohlcv['high'] == ohlcv['low']:
            return 0
            
        o, h, l, c = ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close']
        
        body = abs(o - c)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        # Avoid division by zero
        if body == 0:
            body = 0.25 # 1 tick minimum for ratio math
            
        # Pin bar thresholds
        if lower_wick > (2 * body) and upper_wick < body:
            return 1  # Bullish pin bar -> Expect UP continuation
            
        if upper_wick > (2 * body) and lower_wick < body:
            return -1 # Bearish pin bar -> Expect DOWN continuation
            
        return 0

if __name__ == '__main__':
    harness = BayesianHarness(
        concept_name='Candle_Shapes',
        article_citation='5-key-indicators-for-day-trading-futures.md',  # Or whatever article referenced candles
        registered_response='Directional Continuation (+2 sigma)',
        horizon_bars=60
    )
    
    harness.run_sweep(CandleShapeConcept, years=['2024'])
