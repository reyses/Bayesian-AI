import os
import sys
import pandas as pd
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from research.leg_clock.tools.ag_cat_harness import ConceptHarness, run_sweep
from core_v2.FPS.forward_pass_system import BarState

# Load Delta globally
delta_df = pd.read_parquet('DATA/ATLAS/order_flow_delta_5s.parquet')
delta_df['ts_sec'] = delta_df.index.astype(np.int64) // 10**9
# Map timestamp integer to delta
delta_map = dict(zip(delta_df['ts_sec'], delta_df['delta']))

class FootprintImbalanceConcept(ConceptHarness):
    """
    Concept 10: Footprint Imbalances (Proxy via Extreme Delta).
    
    Hypothesis: 
    Large localized imbalances (extreme delta on a 5-second bar) indicate 
    aggressive institutional participation and lead to short-term momentum.
    
    Definition:
    If 5s delta > 200 contracts -> Long
    If 5s delta < -200 contracts -> Short
    """
    
    def reset(self):
        pass
        
    def eval_state(self, state: BarState) -> float:
        ts = int(state.timestamp)
        d = delta_map.get(ts, 0.0)
        
        if d > 200:
            return 1.0
        elif d < -200:
            return -1.0
            
        return 0.0

if __name__ == '__main__':
    run_sweep(FootprintImbalanceConcept, "10_Footprint_Imbalance", years=['2024', '2025'])
