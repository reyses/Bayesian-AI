import sys
import os
import unittest
import numpy as np
from dataclasses import dataclass

# Add project root to sys.path
sys.path.append(os.getcwd())

from training.pid_oscillation_analyzer import PIDOscillationAnalyzer, PIDSignal

@dataclass
class MockState:
    timestamp: float
    particle_position: float
    particle_velocity: float
    z_score: float
    term_pid: float
    oscillation_coherence: float
    coherence: float
    adx_strength: float
    escape_probability: float = 0.0

class TestPIDOscillationAnalyzer(unittest.TestCase):
    def test_regime_detection(self):
        analyzer = PIDOscillationAnalyzer(sigma_per_bar=10.0)

        # Create a state in PID regime
        # |term_pid| >= 0.3, osc_coh >= 0.5, base_coh >= 0.4, adx <= 30.0, |z| < 2.0
        state_pid = MockState(
            timestamp=1000.0,
            particle_position=100.0,
            particle_velocity=0.0,
            z_score=1.5, # near outer band
            term_pid=0.5,
            oscillation_coherence=0.8,
            coherence=0.9,
            adx_strength=15.0
        )

        # Create a state NOT in PID regime (low osc_coh)
        state_noise = MockState(
            timestamp=1015.0,
            particle_position=100.0,
            particle_velocity=0.0,
            z_score=0.5,
            term_pid=0.1,
            oscillation_coherence=0.2, # too low
            coherence=0.9,
            adx_strength=15.0
        )

        # 1. Feed noise state -> no signal, regime count 0
        sig = analyzer.tick(state_noise)
        self.assertIsNone(sig)
        self.assertEqual(analyzer._regime_n, 0)

        # 2. Feed PID state -> regime count 1, no signal yet (need 3)
        sig = analyzer.tick(state_pid)
        self.assertIsNone(sig)
        self.assertEqual(analyzer._regime_n, 1)

        # 3. Feed 2 more PID states -> regime count 3
        analyzer.tick(state_pid)
        sig = analyzer.tick(state_pid) # Now regime_n = 3

        # Still None because direction logic depends on band touch?
        # z=1.5 -> band_touched='1sig' (1.0 <= |z| < 2.0)
        # Logic:
        # if z <= -1.0: LONG
        # elif z >= 1.0: SHORT
        # else: None
        # So z=1.5 should trigger SHORT if regime is established.

        self.assertIsNotNone(sig)
        self.assertEqual(sig.direction, 'SHORT')
        self.assertEqual(sig.band_touched, '1sig')
        self.assertEqual(sig.regime_bars, 3)
        self.assertEqual(sig.pid_class, 'TENSION') # outer_roche if z>=1.5, wait, z=1.5 IS >= 1.5

        # Test STABLE signal
        state_stable = MockState(
            timestamp=1060.0,
            particle_position=100.0,
            particle_velocity=0.0,
            z_score=1.2, # < 1.5, so not outer_roche tension
            term_pid=0.5,
            oscillation_coherence=0.8,
            coherence=0.9,
            adx_strength=15.0
        )
        # Continue regime
        sig = analyzer.tick(state_stable)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.pid_class, 'STABLE')
        self.assertEqual(sig.z_score, 1.2)

if __name__ == '__main__':
    unittest.main()
