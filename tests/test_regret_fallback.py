import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from training.batch_regret_analyzer import BatchRegretAnalyzer

# Mock classes for minimal dependency
class MockState:
    trend_direction_15m = 'UP'
    def __hash__(self): return 12345

class MockTradeOutcome:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'state'):
            self.state = MockState()

def test_regret_analysis_cpu_fallback():
    """
    Verify that BatchRegretAnalyzer falls back to CPU execution gracefully
    when GPU is unavailable or disabled, and produces valid results.
    """
    analyzer = BatchRegretAnalyzer()

    # Create dummy data (1 hour of 1s data)
    dates = pd.date_range(start='2024-01-01 10:00:00', periods=3600, freq='s')
    prices = np.linspace(100, 110, 3600)

    # Create DataFrame with explicit index
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'high': prices,
        'low': prices,
        'open': prices,
        'volume': 100
    })
    df['price'] = df['close']

    # Explicitly set index to be robust, though analyzer handles 'timestamp' column too
    df = df.set_index('timestamp')

    # Create a list of trades (enough to trigger potential GPU usage if it were available)
    trades = []
    for i in range(20): # > 10 trades to trigger potential GPU path
        t_entry = dates[i*60].timestamp()
        t_exit = dates[i*60 + 30].timestamp()

        trade = MockTradeOutcome(
            entry_price=100.0,
            exit_price=100.5,
            entry_time=t_entry,
            exit_time=t_exit,
            direction='LONG',
            result='WIN',
            exit_reason='TP'
        )
        trades.append(trade)

    # Force CPU by patching torch.cuda.is_available to return False
    # Also patch TORCH_AVAILABLE just in case logic uses that too (though the code checks torch.cuda.is_available())
    with patch('torch.cuda.is_available', return_value=False):
        # Also need to ensure TORCH_AVAILABLE in module doesn't bypass the check if we want to be strict,
        # but the code is: if use_gpu: ... (where use_gpu = TORCH_AVAILABLE and torch.cuda.is_available() ...)

        print("\nRunning analysis with forced CPU fallback...")
        results = analyzer.batch_analyze_day(trades, df, current_timeframe='15s')

        # Verifications
        assert results is not None, "Analysis returned None"
        assert results['total_trades'] == 20, f"Expected 20 trades, got {results['total_trades']}"
        assert results['analyzed_trades'] == 20, f"Expected 20 analyzed trades, got {results['analyzed_trades']}"

        markers = results['regret_markers']
        assert len(markers) == 20

        # Check a marker content to ensure valid calculation
        m = markers[0]
        assert m.entry_price == 100.0
        assert m.exit_price == 100.5
        # Since price is linear 100->110 over 3600s.
        # Entry at 0. Exit at 30s.
        # Peak lookahead is 5 bars of TF1 (60s). So lookahead 300s.
        # Peak should be price at roughly 30s + 300s = 330s.
        # Price increases. So peak > exit.
        assert m.peak_favorable >= m.exit_price

    print("CPU fallback verification passed.")

if __name__ == "__main__":
    test_regret_analysis_cpu_fallback()
