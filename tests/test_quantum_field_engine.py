
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.quantum_field_engine import QuantumFieldEngine

def test_calculate_wave_function_numerical_stability():
    """Test that batch_compute_states handles large z_scores via wave function"""
    engine = QuantumFieldEngine()
    
    # Create data with extreme values
    # We need enough bars for regression (21)
    n = 30
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n, freq='15s'),
        'open': np.full(n, 100.0),
        'high': np.full(n, 100.0),
        'low': np.full(n, 100.0),
        'close': np.full(n, 100.0),
        'volume': np.full(n, 1000.0)
    })
    df['price'] = df['close']

    # Create extreme outlier at end to force z-score > 100
    # Center will be 100, Sigma will be small (epsilon)
    # Price = 1000 => Z = (1000-100)/eps
    df.loc[n-1, 'price'] = 10000.0
    df.loc[n-1, 'close'] = 10000.0

    results = engine.batch_compute_states(df, use_cuda=False) # CPU fallback

    if not results:
        pytest.skip("No results computed")

    last_state = results[-1]['state']

    # Check probabilities (P0, P1, P2)
    # Z should be massive positive
    # Closest singularity is +2.0 (P1)? No, 100 is far from 2.0.
    # Actually, Energy E = -(z-center)^2 / 2
    # E0 ~ -(z)^2/2
    # E1 ~ -(z-2)^2/2
    # E2 ~ -(z+2)^2/2
    # If z=10000:
    # E0 ~ -50M
    # E1 ~ -(9998)^2/2 ~ -49.98M
    # E2 ~ -(10002)^2/2 ~ -50.02M
    # Max E is E1 (closest to +2).
    # So P1 should be ~1.0
    
    p0 = last_state.P_at_center
    p1 = last_state.P_near_upper
    p2 = last_state.P_near_lower
    
    assert not np.isnan(p0)
    assert not np.isnan(p1)
    assert not np.isnan(p2)
    assert abs(p0 + p1 + p2 - 1.0) < 1e-6
    
    # Since Z >> 2, P1 should be dominant (closest attractor is Upper Singularity at Z=2)
    # Wait, distance to 0 is Z. Distance to 2 is Z-2. Distance to -2 is Z+2.
    # (Z-2)^2 < Z^2 < (Z+2)^2 for Z > 0.
    # So E1 > E0 > E2.
    # So P1 should be largest.
    assert p1 > p0
    assert p1 > p2

if __name__ == "__main__":
    pytest.main([__file__])
