
import pytest
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.quantum_field_engine import QuantumFieldEngine

def test_calculate_wave_function_numerical_stability():
    """Test that _calculate_wave_function handles large z_scores without division by zero"""
    engine = QuantumFieldEngine()
    
    # Case 1: z_score = 0 (Center)
    res = engine._calculate_wave_function(0.0, 1.0, 1.0)
    assert not np.isnan(res['P0'])
    assert not np.isnan(res['P1'])
    assert not np.isnan(res['P2'])
    assert abs((res['P0'] + res['P1'] + res['P2']) - 1.0) < 1e-6
    # P0 should be dominant
    assert res['P0'] > res['P1']
    assert res['P0'] > res['P2']

    # Case 2: z_score = 100 (Far outlier)
    # This previously caused RuntimeWarning and division by zero
    res = engine._calculate_wave_function(100.0, 1.0, 1.0)
    assert not np.isnan(res['P0'])
    assert not np.isnan(res['P1'])
    assert not np.isnan(res['P2'])
    assert abs((res['P0'] + res['P1'] + res['P2']) - 1.0) < 1e-6
    
    # At z=100, it is closest to 2.0 (distance 98) compared to 0 (distance 100) or -2 (distance 102)
    # So P1 should be 1.0 (or very close)
    assert res['P1'] > 0.99
    assert res['P0'] < 1e-10
    assert res['P2'] < 1e-10
    
    # Case 3: z_score = -100 (Far outlier negative)
    res = engine._calculate_wave_function(-100.0, 1.0, 1.0)
    # Closest to -2.0
    assert res['P2'] > 0.99
    
    # Case 4: z_score = 2.0 (Upper Singularity)
    res = engine._calculate_wave_function(2.0, 1.0, 1.0)
    # P1 (exp(0)=1) should be max, but P0 (exp(-2)) is also significant
    # P1 = 1, P0 = 0.135, P2 = exp(-8) ~= 0
    # normalized P1 approx 1/1.135 = 0.88
    assert res['P1'] > res['P0']
    assert res['P1'] > res['P2']

if __name__ == "__main__":
    pytest.main([__file__])
