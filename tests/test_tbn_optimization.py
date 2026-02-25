
import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from training.timeframe_belief_network import TimeframeBeliefNetwork, TimeframeWorker

class MockState:
    def __init__(self):
        self.z_score = 1.0
        self.particle_velocity = 0.5
        self.momentum_strength = 0.2
        self.coherence = 0.8
        self.adx_strength = 25.0
        self.hurst_exponent = 0.6
        self.dmi_plus = 30.0
        self.dmi_minus = 10.0
        self.term_pid = 0.1
        self.oscillation_coherence = 0.9

class TestTBNOptimization(unittest.TestCase):
    def test_scaler_optimization(self):
        # Setup Scaler
        scaler = StandardScaler()
        # 16 features
        X_train = np.random.randn(100, 16)
        scaler.fit(X_train)

        # Setup TBN (mocking pattern_library, engine, valid_tids, centroids_scaled)
        centroids_scaled = np.random.randn(5, 16)
        tbn = TimeframeBeliefNetwork(
            pattern_library={},
            scaler=scaler,
            engine=None,
            valid_tids=[],
            centroids_scaled=centroids_scaled
        )

        # Verify optimization params extracted
        self.assertTrue(hasattr(tbn, 'scaler_mean'))
        self.assertTrue(hasattr(tbn, 'scaler_scale'))
        self.assertIsNotNone(tbn.scaler_mean)
        self.assertIsNotNone(tbn.scaler_scale)
        np.testing.assert_array_equal(tbn.scaler_mean, scaler.mean_)
        np.testing.assert_array_equal(tbn.scaler_scale, scaler.scale_)

        # Setup Worker
        worker = TimeframeWorker(tf_seconds=15)
        worker._states = [{'state': MockState(), 'bar_idx': 0}]

        # We need to access the worker's logic directly or inspect the result.
        # Since tick modifies internal state based on computation, let's verify computation directly
        # by calling the optimized logic and comparing with sklearn logic

        state = MockState()
        feat = TimeframeBeliefNetwork.state_to_features(state, 15)

        # Optimized path
        feat_opt = (np.array(feat) - tbn.scaler_mean) / tbn.scaler_scale

        # Standard path
        feat_std = scaler.transform([feat])[0]

        # Verify identical
        np.testing.assert_allclose(feat_opt, feat_std, rtol=1e-10, atol=1e-10)

        print("Optimization numerical verification passed!")

if __name__ == '__main__':
    unittest.main()
