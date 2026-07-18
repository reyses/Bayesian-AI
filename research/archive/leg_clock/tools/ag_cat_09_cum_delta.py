import os
import sys
import pandas as pd
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from research.leg_clock.tools.ag_cat_harness import ConceptHarness, run_sweep
from core_v2.FPS.forward_pass_system import BarState

# Load Delta globally to avoid repeating it per day
delta_df = pd.read_parquet('DATA/ATLAS/order_flow_delta_5s.parquet')
delta_df['ts_sec'] = delta_df.index.astype(np.int64) // 10**9
# Map timestamp integer to cumulative delta
cum_delta_map = dict(zip(delta_df['ts_sec'], delta_df['cum_delta']))

class CumDeltaDivergenceConcept(ConceptHarness):
    """
    Concept 09: Cumulative Delta Divergence (Order Flow).
    
    Hypothesis: 
    When price breaks to a new local high but Cumulative Delta fails to break 
    its local high, it's a bearish divergence (Short). 
    When price breaks to a new low but Cum Delta fails to break its low, 
    it's a bullish divergence (Long).
    
    Definition:
    Track rolling 60-bar (1h) High/Low for Price and Cum Delta.
    If Price breaks 60-bar High AND Cum Delta < 60-bar Cum Delta High -> Short
    If Price breaks 60-bar Low AND Cum Delta > 60-bar Cum Delta Low -> Long
    """
    
    def reset(self):
        self.closes = []
        self.cum_deltas = []
        self.prev_close = None
        self.prev_max_close = None
        self.prev_min_close = None
        self.prev_max_cd = None
        self.prev_min_cd = None
        
    def eval_state(self, state: BarState) -> float:
        # Evaluate on the 1-minute boundary
        if not state.is_1m_close:
            return 0.0
            
        c = state.price
        ts = int(state.timestamp)
        cd = cum_delta_map.get(ts, 0.0)
        
        self.closes.append(c)
        self.cum_deltas.append(cd)
        
        if len(self.closes) < 60:
            self.prev_close = c
            return 0.0
            
        if len(self.closes) > 60:
            self.closes.pop(0)
            self.cum_deltas.pop(0)
            
        max_close = max(self.closes)
        min_close = min(self.closes)
        max_cd = max(self.cum_deltas)
        min_cd = min(self.cum_deltas)
        
        signal = 0.0
        
        if self.prev_max_close is not None and self.prev_close is not None:
            # Bearish Divergence
            if self.prev_close < self.prev_max_close and c > max_close:
                if cd < max_cd:
                    signal = -1.0
            # Bullish Divergence
            elif self.prev_close > self.prev_min_close and c < min_close:
                if cd > min_cd:
                    signal = 1.0
                    
        self.prev_max_close = max_close
        self.prev_min_close = min_close
        self.prev_max_cd = max_cd
        self.prev_min_cd = min_cd
        self.prev_close = c
        
        return signal

if __name__ == '__main__':
    run_sweep(CumDeltaDivergenceConcept, "09_CumDelta_Divergence", years=['2024', '2025'])
