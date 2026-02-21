"""
Tests for training/trade_analytics.py
"""
import os
import tempfile
import pandas as pd
import pytest
from training.trade_analytics import run_trade_analytics

@pytest.fixture
def dummy_trade_log():
    # Create a dummy CSV with necessary columns
    data = {
        'entry_time': [1700000000 + i*3600 for i in range(20)], # 20 trades
        'direction': ['LONG', 'SHORT'] * 10,
        'oracle_label': [1, -1, 0, 2, -2] * 4,
        'oracle_mfe': [10.0, 12.0, 2.0, 20.0, 15.0] * 4,
        'oracle_mae': [5.0, 4.0, 10.0, 3.0, 6.0] * 4,
        'actual_pnl': [50.0, -20.0, -10.0, 100.0, 80.0, 10.0, -50.0, 0.0, 200.0, -10.0] * 2,
        'capture_rate': [0.5, -0.2, -0.5, 0.8, 0.6] * 4,
        'hold_bars': [10, 20, 5, 30, 15] * 4,
        'entry_workers': ['{"1h":{"d":0.8,"c":0.9},"5m":{"d":0.7,"c":0.5}}'] * 20,
        'exit_workers': ['{"5m":{"d":0.2}}'] * 10 + ['{"5m":{"d":0.8}}'] * 10,
        'belief_conviction': [0.8] * 20,
        'wave_maturity': [0.1] * 20,
        'exit_reason': ['target', 'stop'] * 10,
        'entry_depth': [6] * 20,
        # Add columns for regression features
        'decision_wave_maturity': [0.5] * 20,
        'dmi_diff': [10.0] * 20,
        'long_bias': [0.6] * 20,
        'short_bias': [0.4] * 20,
        'exit_conviction': [0.7] * 20,
        'exit_wave_maturity': [0.9] * 20,
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        path = tmp.name

    yield path

    if os.path.exists(path):
        os.remove(path)

def test_run_trade_analytics(dummy_trade_log):
    report = run_trade_analytics(dummy_trade_log)

    print(report)

    assert "JULES TRADE ANALYTICS SUITE" in report
    assert "GOOD vs BAD TRADE COMPARISON" in report
    assert "ANOVA: Categorical Variables" in report
    assert "LINEAR REGRESSION: actual_pnl" in report
    assert "LOGISTIC REGRESSION: is_win" in report
    assert "CAPTURE RATE REGRESSION" in report
    assert "SESSION x DIRECTION" in report

def test_missing_file():
    report = run_trade_analytics("non_existent_file.csv")
    assert "skipped" in report

def test_empty_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        path = tmp.name

    report = run_trade_analytics(path)
    assert "failed to load log" in report or "skipped" in report

    os.remove(path)
