import unittest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.doe_parameter_generator import DOEParameterGenerator
from training.integrated_statistical_system import IntegratedStatisticalEngine, TradeRecord

class TestDOEFeatures(unittest.TestCase):
    def setUp(self):
        self.detector = MagicMock()
        self.generator = DOEParameterGenerator(self.detector)

    def test_lhs_stratification(self):
        """Verify LHS produces stratified samples"""
        print("\nTesting LHS Stratification...")
        day = 1
        n_samples = 100
        # LHS is generated in batch of 500 by default.
        # We need to collect enough samples to check stratification.

        # Collect samples for one parameter
        param_name = 'confidence_threshold'
        min_val, max_val, _ = self.generator.param_ranges[param_name]

        values = []
        for i in range(10, 10 + n_samples):
            pset = self.generator.generate_latin_hypercube_set(i, day, 'CORE', n_samples=n_samples)
            values.append(pset.parameters[param_name])

        values = np.array(values)

        # Check range
        self.assertTrue(np.all(values >= min_val))
        self.assertTrue(np.all(values <= max_val))

        # Check stratification
        # Divide range into n_bins. Each bin should have roughly equal count.
        # Ideally exactly 1 per bin if n_samples matches LHS size.
        # But we use n_samples=100.

        # Normalize to [0, 1]
        norm_values = (values - min_val) / (max_val - min_val)

        # With Stratified LHS, if we requested 100 samples, and collected 100 samples,
        # we should have 1 sample in [0, 0.01), 1 in [0.01, 0.02), etc.
        # Check empty bins count
        n_bins = n_samples
        hist, _ = np.histogram(norm_values, bins=n_bins, range=(0, 1))

        # In perfect LHS, every bin has 1 sample.
        # However, due to floating point and scaling, it might not be perfect 1.
        # But it should be much better than random.
        # For random, many bins would be empty (approx 36% empty for Poisson).
        empty_bins = np.sum(hist == 0)
        print(f"LHS Empty Bins (out of {n_bins}): {empty_bins}")

        # Allow a small margin of error due to scaling artifacts, but it should be very low.
        self.assertLess(empty_bins, 10, "LHS stratification failed: too many empty bins")

    def test_response_surface_optimization(self):
        """Verify Response Surface logic suggests better parameters"""
        print("\nTesting Response Surface Optimization...")

        # Define a simple objective function: Maximize -(x-50)^2 (Parabola peak at 50)
        # Parameter: 'take_profit_ticks' (Range 30-60)
        target = 50

        # Populate history with some points around the peak
        # Points: 30, 35, 40, 55, 60
        test_points = [30, 35, 40, 55, 60]

        for p_val in test_points:
            params = self.generator.generate_baseline_set(0, 1, 'CORE').parameters.copy()
            params['take_profit_ticks'] = p_val

            # Simple quadratic performance metric
            perf = -((p_val - target) ** 2)

            self.generator.update_best_params(params, performance=perf)

        # Ask RSM to suggest next point (Iteration 800+)
        pset = self.generator.generate_response_surface_set(800, day=1, context='CORE')
        suggested_val = pset.parameters['take_profit_ticks']

        print(f"Target: {target}, Suggested: {suggested_val}")

        # Should be closer to 50 than the worst historical points
        # Allowed error margin
        self.assertTrue(45 <= suggested_val <= 55, f"RSM failed to find peak at 50, got {suggested_val}")

    def test_integrated_statistical_engine_components(self):
        """Smoke test for ANOVA and Monte Carlo components"""
        print("\nTesting Integrated Statistical Engine Components...")

        # Setup dummy asset profile
        class MockAsset:
            point_value = 2.0
            ticker = "MNQ"

        engine = IntegratedStatisticalEngine(MockAsset())

        # 1. Test Trade Recording
        trade = TradeRecord(
            state_hash=123, entry_price=100, exit_price=110, entry_time=0, exit_time=60,
            side='LONG', pnl=20, result='WIN', exit_reason='TP',
            peak_favorable=110, potential_max_pnl=20, pnl_left_on_table=0, gave_back_pnl=0,
            exit_efficiency=1.0, regret_type='optimal'
        )
        engine.record_trade(trade)

        # 2. Test Decision Logic (Insufficient Data)
        decision = engine.should_fire(123)
        self.assertFalse(decision['should_fire'])
        self.assertIn('Insufficient data', decision['reason'])

        # 3. Test Monte Carlo (Risk Analyzer)
        # Direct call to component
        mc = engine.monte_carlo
        results = mc.simulate_drawdown(win_rate=0.5, avg_win=20, avg_loss=-20, n_trades=50)
        self.assertIn('expected_max_dd', results)
        self.assertIn('prob_profit', results)

        # 4. Test DOE (ANOVA)
        doe = engine.doe
        # Add dummy experiments
        for i in range(10):
            doe.record_experiment_result(i, win_rate=0.5, sharpe=1.0+i/10, max_dd=100, total_pnl=500)
            doe.experiments.append({
                'experiment_id': i,
                'stop_loss': 10 if i<5 else 20,
                'trail_tight': 10 if i%2==0 else 15,
                'trail_medium': 20 if i%2==0 else 25,
                'trail_wide': 30 if i%2==0 else 35,
                'min_samples': 30 if i%2==0 else 40
            })

        effects = doe.analyze_factor_importance(response_var='sharpe')
        # Should have results
        if not effects.empty:
            print("ANOVA Effects found")
            self.assertIn('factor', effects.columns)
            self.assertIn('p_value', effects.columns)

if __name__ == '__main__':
    unittest.main()
