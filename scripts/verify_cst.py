
import sys
import os
import numpy as np
from dataclasses import dataclass

# Add root to path
sys.path.append(os.getcwd())

from core.quantum_field_engine import QuantumFieldEngine
from core.three_body_state import ThreeBodyQuantumState
from training.wave_rider import WaveRider, Position

def test_build_vector():
    print("Testing build_16d_vector...")
    # Mock state
    state = ThreeBodyQuantumState.null_state()
    # Populate some fields
    # state is frozen, so we might need to bypass or create a new one properly if possible,
    # but null_state is a factory.
    # Actually ThreeBodyQuantumState is a dataclass, but frozen=True?
    # Let's check ThreeBodyQuantumState definition.
    # It is usually frozen.
    # I can use object() and setattr if I am mocking, or just use null_state.

    # Ancestry
    ancestry = {
        'timeframe': '15s',
        'depth': 5,
        'parent_type': 'ROCHE_SNAP',
        'parent_chain': [{'z': 3.0, 'dmi_plus': 20, 'dmi_minus': 10, 'type': 'ROCHE_SNAP'}]
    }

    vec = QuantumFieldEngine.build_16d_vector(state, ancestry)
    print(f"Vector: {vec}")
    assert len(vec) == 16
    print("build_16d_vector OK")

def test_wave_rider_check():
    print("Testing WaveRider.check_structural_integrity...")
    wr = WaveRider(asset_profile=None) # Mock asset profile?
    # WaveRider init needs asset_profile.
    class MockAsset:
        tick_size = 0.25
        tick_value = 5.0
        point_value = 20.0

    wr = WaveRider(MockAsset())

    # Setup position
    cst_centroid = np.zeros(16)
    cst_ancestry = {
        'timeframe': '15s',
        'depth': 5,
        'parent_type': 'ROCHE_SNAP',
        'parent_chain': []
    }

    wr.open_position(
        entry_price=100.0, side='long', state=None,
        cst_centroid=cst_centroid,
        cst_basin_mean=1.0,
        cst_basin_std=0.1,
        cst_ancestry=cst_ancestry
    )

    # Current state (perfect match)
    state_ok = ThreeBodyQuantumState.null_state()
    # build_16d_vector uses getattr, so null_state (zeros) should match zeros centroid if we ignore ancestry features scaling
    # Wait, build_16d_vector puts ancestry into vector.
    # So vector will NOT be zero.
    # We need to set centroid to what build_16d_vector produces for the "entry" state.

    vec_ok = QuantumFieldEngine.build_16d_vector(state_ok, cst_ancestry)
    wr.position.cst_centroid = np.array(vec_ok)

    is_ok = wr.check_structural_integrity(state_ok)
    print(f"Check match: {is_ok}")
    assert is_ok == True

    # Broken state (modify state to be far)
    # Since ThreeBodyQuantumState is frozen, we can't modify it easily.
    # We can create a mock class.
    @dataclass
    class MockState:
        z_score: float = 100.0 # Huge Z
        particle_velocity: float = 0.0
        momentum_strength: float = 0.0
        coherence: float = 0.0
        adx_strength: float = 0.0
        hurst_exponent: float = 0.0
        dmi_plus: float = 0.0
        dmi_minus: float = 0.0
        term_pid: float = 0.0
        oscillation_coherence: float = 0.0

    state_broken = MockState()
    is_broken = wr.check_structural_integrity(state_broken)
    print(f"Check broken: {is_broken}")
    # Threshold is 1.0 + 3*0.1 = 1.3
    # Distance should be large due to z_score=100
    assert is_broken == False
    print("check_structural_integrity OK")

if __name__ == "__main__":
    test_build_vector()
    test_wave_rider_check()
