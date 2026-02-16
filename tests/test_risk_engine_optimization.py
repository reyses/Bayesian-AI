import sys
import os
import numpy as np
import pytest

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.risk_engine import QuantumRiskEngine

class TestQuantumRiskEngine:
    def test_instantiation(self):
        engine = QuantumRiskEngine(theta=0.1, horizon_seconds=600, num_paths=100)
        assert engine.theta == 0.1
        assert engine.horizon_seconds == 600
        assert engine.num_paths == 100

    def test_calculate_probabilities_sigma_zero(self):
        engine = QuantumRiskEngine()
        p_tunnel, p_escape = engine.calculate_probabilities(price=100.0, center=100.0, sigma=0.0)
        assert p_tunnel == 0.5
        assert p_escape == 0.0

    def test_calculate_probabilities_tunnel(self):
        # Price is far from stop, close to center. Should tunnel (revert) with high probability.
        # But OU process pushes to center.
        # L2: price > center. Stop = center + 3sigma.
        # Price = center + 0.1 sigma.
        # Very close to center. Should hit center easily.
        engine = QuantumRiskEngine(theta=0.1, horizon_seconds=1000, num_paths=1000)
        p_tunnel, p_escape = engine.calculate_probabilities(price=100.1, center=100.0, sigma=1.0)
        assert p_tunnel > 0.9 # High probability to hit center
        assert p_escape < 0.1

    def test_calculate_probabilities_escape(self):
        # Price is close to stop.
        # L2: price > center. Stop = center + 3sigma.
        # Price = center + 2.9 sigma.
        # Very close to stop. Should escape with high probability?
        # Reversion force pushes back to center.
        # But random noise pushes around.
        # Sigma step = sigma * sqrt(1 - exp(-2*theta*dt)).
        # If sigma=1, theta=0.1, dt=1.
        # sigma_step = sqrt(1 - exp(-0.2)) = sqrt(1 - 0.818) = sqrt(0.18) = 0.42.
        # Distance to stop is 0.1.
        # One step can cross it.
        # Probability of crossing 0.1 with sigma 0.42 is high.
        engine = QuantumRiskEngine(theta=0.1, horizon_seconds=1000, num_paths=1000)
        p_tunnel, p_escape = engine.calculate_probabilities(price=102.9, center=100.0, sigma=1.0)
        # Escape should be significant.
        assert p_escape > 0.3 # At least some paths should hit stop

    def test_probabilities_sum_le_one(self):
        engine = QuantumRiskEngine(num_paths=100)
        p_tunnel, p_escape = engine.calculate_probabilities(price=101.0, center=100.0, sigma=1.0)
        assert 0.0 <= p_tunnel <= 1.0
        assert 0.0 <= p_escape <= 1.0
        assert p_tunnel + p_escape <= 1.0 + 1e-9

    def test_determinism(self):
        # Should produce same result twice
        engine = QuantumRiskEngine(num_paths=500)
        p1 = engine.calculate_probabilities(price=101.0, center=100.0, sigma=1.0)
        p2 = engine.calculate_probabilities(price=101.0, center=100.0, sigma=1.0)
        assert p1 == p2

    def test_boundary_logic_L2(self):
        # Test explicit boundary conditions for L2 (Price > Center)
        # If price is EXACTLY stop_level, should be 100% escape.
        engine = QuantumRiskEngine(num_paths=100)
        price = 103.0
        center = 100.0
        sigma = 1.0
        # Stop level = 100 + 3*1 = 103.
        # Price == Stop Level.
        p_tunnel, p_escape = engine.calculate_probabilities(price=price, center=center, sigma=sigma)
        assert p_escape == 1.0
        assert p_tunnel == 0.0

        # If price is slightly BELOW stop_level, should act normally.
        p_tunnel, p_escape = engine.calculate_probabilities(price=102.99, center=center, sigma=sigma)
        assert p_escape < 1.0 # Due to random noise some might revert or stay within bounds

        # If price is slightly ABOVE stop_level, should be 100% escape.
        p_tunnel, p_escape = engine.calculate_probabilities(price=103.01, center=center, sigma=sigma)
        assert p_escape == 1.0
        assert p_tunnel == 0.0

    def test_boundary_logic_L3(self):
        # Test explicit boundary conditions for L3 (Price < Center)
        # If price is EXACTLY stop_level, should be 100% escape.
        engine = QuantumRiskEngine(num_paths=100)
        price = 97.0
        center = 100.0
        sigma = 1.0
        # Stop level = 100 - 3*1 = 97.
        # Price == Stop Level.
        p_tunnel, p_escape = engine.calculate_probabilities(price=price, center=center, sigma=sigma)
        assert p_escape == 1.0
        assert p_tunnel == 0.0

        # If price is slightly ABOVE stop_level (closer to center), normal.
        p_tunnel, p_escape = engine.calculate_probabilities(price=97.01, center=center, sigma=sigma)
        assert p_escape < 1.0

        # If price is slightly BELOW stop_level (further away), 100% escape.
        p_tunnel, p_escape = engine.calculate_probabilities(price=96.99, center=center, sigma=sigma)
        assert p_escape == 1.0
        assert p_tunnel == 0.0

if __name__ == "__main__":
    pytest.main([__file__])
