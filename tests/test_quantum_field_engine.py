
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.quantum_field_engine import QuantumFieldEngine
from core.three_body_state import ThreeBodyQuantumState

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

def test_batch_compute_states_cpu():
    """Test CPU fallback for batch_compute_states"""
    engine = QuantumFieldEngine()
    engine.use_gpu = False # Force CPU

    # Create dummy data
    n = 100
    dates = pd.date_range(start='2025-01-01', periods=n, freq='15s')
    data = pd.DataFrame({
        'timestamp': dates.astype('int64') // 10**9,
        'open': np.linspace(100, 110, n),
        'high': np.linspace(101, 111, n),
        'low': np.linspace(99, 109, n),
        'close': np.linspace(100.5, 110.5, n),
        'volume': np.random.rand(n) * 1000
    })

    # Add trend to verify physics
    # Linear trend should result in stable slope and small sigma

    results = engine.batch_compute_states(data, use_cuda=False)

    assert len(results) > 0
    # Should start from rp (21)
    rp = engine.regression_period
    assert len(results) == n - rp

    first_res = results[0]
    assert isinstance(first_res['state'], ThreeBodyQuantumState)
    assert first_res['bar_idx'] == rp

    # Check physics values are calculated (not all zeros)
    state = first_res['state']
    # With linear price 100->110 over 100 steps
    # Slope should be approx 0.1
    # Center should track price

    assert state.center_position != 0.0
    # sigma might be small but non-zero due to noise or perfect fit
    # In perfect linear fit, sigma -> 0. But we used linspace for open/high/low/close differently?
    # close is linspace. So sigma should be very small (epsilon).
    # Wait, code sets sigma = max(sigma, 1e-6)
    assert state.sigma_fractal >= 1e-6

if __name__ == "__main__":
    pytest.main([__file__])
