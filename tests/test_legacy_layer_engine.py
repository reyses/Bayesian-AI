"""
Bayesian AI v2.0 - Legacy Layer Engine Test
Validates deprecated LayerEngine logic and L2 thresholds.
Refactored from test_phase1.py and verify_phase1_fixes.py
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.state_vector import StateVector
from core.bayesian_brain import BayesianBrain, TradeOutcome
from archive.layer_engine import LayerEngine
from config.symbols import MNQ, calculate_pnl
from tests.utils import load_test_data, get_cuda_availability, find_test_data_file
from training.databento_loader import DatabentoLoader

class TestLegacyLayerEngine(unittest.TestCase):
    def load_ohlcv_data(self):
        """Helper to load specific OHLCV data for legacy tests"""
        # Prefer the OHLCV file in Testing DATA as legacy engine expects OHLCV
        ohlcv_file = 'glbx-mdp3-20251230-20260129.ohlcv-1s.dbn.zst'
        path = find_test_data_file(ohlcv_file)

        if path and os.path.exists(path):
            print(f"Loading OHLCV data from {path}...")
            df = DatabentoLoader.load_data(path)
            # Ensure index
            if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('datetime', inplace=True)
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('datetime', inplace=True)

            if 'price' in df.columns and 'close' not in df.columns:
                df['close'] = df['price']
            return df

        # Fallback to default if specific file not found (will likely fail resample if trades only)
        print("OHLCV file not found, falling back to default load_test_data...")
        return load_test_data()

    def test_layer_engine_computation(self):
        """Test LayerEngine computation (from test_phase1.py)"""
        print("\n=== TEST: LayerEngine Computation ===")

        # Load test data (specifically OHLCV)
        data = self.load_ohlcv_data()
        print(f"Loaded {len(data)} rows of data.")

        # Resample for different timeframes
        bars_4hr = data.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        bars_1hr = data.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        bars_15m = data.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        bars_5m = data.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

        # Initialize engine
        use_gpu = get_cuda_availability()
        engine = LayerEngine(use_gpu=use_gpu)
        engine.initialize_static_context(data, kill_zones=[21500, 21600, 21700])

        last_price = data.iloc[-1]['close']
        last_timestamp = data.index[-1].timestamp()

        # Create current snapshot
        current_data = {
            'price': last_price,
            'timestamp': last_timestamp,
            'bars_4hr': bars_4hr.tail(6),
            'bars_1hr': bars_1hr.tail(10),
            'bars_15m': bars_15m.tail(20),
            'bars_5m': bars_5m.tail(20),
            'ticks': data['close'].values[-50:]  # Use last 50 closes as ticks
        }

        # Compute state
        state = engine.compute_current_state(current_data)

        print(f"[OK] Computed state:")
        state_dict = state.to_dict()
        for key, value in state_dict.items():
            if key not in ['timestamp', 'price']:
                print(f"  {key}: {value}")

        self.assertIsInstance(state, StateVector)
        self.assertIsNotNone(state.L1_bias)

    def test_integration(self):
        """Test full integration: LayerEngine -> StateVector -> BayesianBrain (from test_phase1.py)"""
        print("\n=== TEST: Full Integration ===")

        # Setup
        data = self.load_ohlcv_data()
        bars_4hr = data.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        bars_1hr = data.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        bars_15m = data.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        bars_5m = data.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

        use_gpu = get_cuda_availability()
        engine = LayerEngine(use_gpu=use_gpu)
        engine.initialize_static_context(data, kill_zones=[21500, 21600])
        brain = BayesianBrain()

        # Simulate 5 trades
        # We will just iterate over the last 5 data points
        subset = data.tail(5)

        for i in range(5):
            row = subset.iloc[i]
            price = row['close']
            timestamp = subset.index[i].timestamp()

            current_data = {
                'price': price,
                'timestamp': timestamp,
                'bars_4hr': bars_4hr,
                'bars_1hr': bars_1hr,
                'bars_15m': bars_15m,
                'bars_5m': bars_5m,
                'ticks': data['close'].values[-(50+i):-i] if i > 0 else data['close'].values[-50:]
            }

            # Compute state
            state = engine.compute_current_state(current_data)

            # Simulate trade outcome
            entry_price = price
            exit_price = entry_price + np.random.choice([20, -10])  # Random win/loss
            pnl = calculate_pnl(MNQ, entry_price, exit_price, 'short')

            outcome = TradeOutcome(
                state=state,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                result='WIN' if pnl > 0 else 'LOSS',
                timestamp=float(i),
                exit_reason='test'
            )

            brain.update(outcome)
            print(f"Trade {i+1}: {outcome.result} | PnL: ${pnl:.2f}")

        # Summary
        summary = brain.get_summary()
        print(f"\n[OK] Learning Summary:")
        print(f"  Unique states learned: {summary['total_unique_states']}")
        print(f"  Total trades: {summary['total_trades']}")

        self.assertGreater(summary['total_trades'], 0)

    def test_layer_engine_L2_logic(self):
        """Test LayerEngine L2 threshold fix (Random Walk vs Trend) (from verify_phase1_fixes.py)"""
        print("\n=== TEST: LayerEngine L2 Logic ===")
        engine = LayerEngine(use_gpu=False)

        # Scenario 1: Random Walk (Vol 10)
        # Avg Range = 10.
        # 5-day Box should be ~22 (sqrt(5)*10).
        # Ratio ~2.2.
        # Should be 'chopping' (Threshold 3.0).

        # Generate random walk data
        np.random.seed(42)
        dates = pd.date_range(start='2025-01-01', periods=30, freq='D')

        # Create a "Chopping" scenario (High volatility but low displacement)
        # Day 1: 100-110 (up). Day 2: 110-100 (down). Day 3: 100-110...
        closes = []
        highs = []
        lows = []
        opens = []

        for i in range(30):
            base = 100
            o = base
            c = base
            h = base + 10
            l = base - 10
            # Wait, daily range is h-l = 20.
            # Avg Range = 20.
            # 5-day Box: max(h) - min(l) = 110 - 90 = 20.
            # Ratio 1.0.
            # 20 > 20 * 3.0 (60). False. 'chopping'. Correct.
            closes.append(c); highs.append(h); lows.append(l); opens.append(o)

        data_chop = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': 1000
        }, index=dates)

        engine.daily_data = data_chop
        res = engine._compute_L2_30d()
        self.assertEqual(res, 'chopping', "Perfect chop should be 'chopping'")

        # Scenario 2: Strong Trend
        # Daily Range = 10.
        # Move 10 per day.
        # 5 days: 0-10, 10-20, ... 40-50.
        # Box: 0-50 = 50.
        # Avg Range = 10.
        # Ratio 5.0.
        # 50 > 10 * 3.0 (30). True. 'trending'. Correct.

        closes = []
        highs = []
        lows = []
        opens = []
        price = 100
        for i in range(30):
            price += 10
            o = price
            c = price + 5
            h = price + 10
            l = price
            # Daily Range: 10.
            closes.append(c); highs.append(h); lows.append(l); opens.append(o)

        data_trend = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': 1000
        }, index=dates)

        engine.daily_data = data_trend
        res = engine._compute_L2_30d()
        self.assertEqual(res, 'trending', "Strong trend should be 'trending'")

        # Scenario 3: Reverting Move (The edge case)
        # Avg Range = 10.
        # We want Box ~ 25 (2.5x). Should be 'chopping' now (was 'trending' with 1.5x).

        closes = []
        highs = []
        lows = []
        opens = []

        for i in range(25): # Filler
            closes.append(100); highs.append(110); lows.append(100); opens.append(100)

        # Last 5 days
        # Day 26
        closes.append(110); highs.append(110); lows.append(100); opens.append(100)
        # Day 27
        closes.append(120); highs.append(120); lows.append(110); opens.append(110)
        # Day 28
        closes.append(125); highs.append(125); lows.append(115); opens.append(115)
        # Day 29
        closes.append(120); highs.append(120); lows.append(110); opens.append(110)
        # Day 30
        closes.append(110); highs.append(110); lows.append(100); opens.append(100)

        # Box: 125-100 = 25.
        # Avg Range: 10.
        # Ratio 2.5.

        data_revert = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': 1000
        }, index=dates)

        engine.daily_data = data_revert
        res = engine._compute_L2_30d()
        self.assertEqual(res, 'chopping', "Reverting move (2.5x range) should be 'chopping' with new threshold")

if __name__ == "__main__":
    unittest.main()
