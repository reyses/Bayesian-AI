import os
import sys
import numpy as np

# Ensure we can import the harness
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ag_bayes_harness import BayesianHarness

class PivotTouchesConcept:
    def __init__(self):
        self.daily_pivots = {}
        self.prev_price = None

    def eval_state(self, state):
        event = 0
        
        # Manually track Daily High/Low/Close
        if not hasattr(self, 'current_day'):
            self.current_day = state.day
            self.day_high = state.price
            self.day_low = state.price
            self.day_close = state.price
            self.yesterday_hlc = None
            
        if state.day != self.current_day:
            self.yesterday_hlc = (self.day_high, self.day_low, self.day_close)
            self.current_day = state.day
            self.day_high = state.price
            self.day_low = state.price
            
            # Compute Pivots
            H, L, C = self.yesterday_hlc
            P = (H + L + C) / 3.0
            R1 = (2 * P) - L
            S1 = (2 * P) - H
            R2 = P + (H - L)
            S2 = P - (H - L)
            R3 = R1 + (H - L)
            S3 = S1 - (H - L)
            
            self.daily_pivots = {'P': P, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}

        # Update running H/L/C
        if state.price > self.day_high:
            self.day_high = state.price
        if state.price < self.day_low:
            self.day_low = state.price
        self.day_close = state.price

        if self.yesterday_hlc is not None and self.prev_price is not None:
            # Check for touch of any pivot level
            # Touch from above -> Bounce UP (1)
            # Touch from below -> Bounce DOWN (-1)
            
            # We add a small tolerance for "touch" (e.g. 1 tick = 0.25)
            # Actually, crossing it is a definitive touch.
            for level_name, level_price in self.daily_pivots.items():
                if self.prev_price > level_price and state.price <= level_price:
                    event = 1  # Touch from above -> expect bounce up
                    break
                elif self.prev_price < level_price and state.price >= level_price:
                    event = -1 # Touch from below -> expect bounce down
                    break
                    
        self.prev_price = state.price
        return event

if __name__ == '__main__':
    harness = BayesianHarness(
        concept_name='Pivot_Touches',
        article_citation='how-pivot-points-can-help-guide-your-futures-trades.md',
        registered_response='Directional first-touch bounce (+2 sigma)',
        horizon_bars=60
    )
    
    harness.run_sweep(PivotTouchesConcept, years=['2024'])
