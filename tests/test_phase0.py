import pytest
import pandas as pd
import numpy as np
import os
import sys
from training.orchestrator import TrainingOrchestrator
from core.unconstrained_explorer import UnconstrainedExplorer

def test_phase0_exploration():
    """
    Test Phase 0 Unconstrained Exploration
    """
    # Create synthetic data
    # We need enough data for 15m and 15s bars.
    # QuantumFieldEngine needs len(df_macro) >= regression_period (21)
    # df_macro is 15m bars. So we need 21 * 15m = 315 minutes of data.
    # Use 15s interval to speed up test (1 tick per 15s)
    # 315 minutes * 4 = 1260 ticks.
    # Let's generate 2000 ticks.

    n_ticks = 2000
    start_time = pd.Timestamp('2024-01-01 09:30:00')
    # Use float timestamps
    timestamps = [(start_time + pd.Timedelta(seconds=i*15)).timestamp() for i in range(n_ticks)]

    # Generate random walk price
    np.random.seed(42)
    prices = 10000 + np.cumsum(np.random.normal(0, 1, n_ticks))
    volumes = np.random.randint(1, 10, n_ticks)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes,
        'type': 'trade'
    })

    # Initialize Orchestrator in PHASE0 mode
    orchestrator = TrainingOrchestrator(
        asset_ticker='MNQ',
        data=data,
        use_gpu=False, # Use CPU for test
        output_dir='debug_outputs',
        verbose=True,
        debug_file='debug_outputs/test_phase0.log',
        mode='PHASE0'
    )

    # Run training (exploration)
    # 1 iteration is enough as it loops over ticks
    orchestrator.run_training(iterations=1)

    # Check if trades were executed
    engine = orchestrator.engine
    assert isinstance(engine.explorer, UnconstrainedExplorer)

    print(f"Trades executed: {engine.explorer.trades_executed}")
    print(f"Unique states: {len(engine.explorer.unique_states_seen)}")

    assert engine.explorer.trades_executed > 0
    assert len(engine.explorer.unique_states_seen) > 0

    # Check if Phase 0 stopped correctly (either max trades or unique states)
    # Config default: 500 trades or 50 unique states
    # In 24000 ticks, it should trigger many trades if unconstrained.

    if engine.explorer.trades_executed >= 500:
        print("Stopped by max trades")
    elif len(engine.explorer.unique_states_seen) >= 50:
        print("Stopped by unique states")
    else:
        # Maybe data wasn't enough to reach limits?
        print(f"Finished with {engine.explorer.trades_executed} trades and {len(engine.explorer.unique_states_seen)} states")

if __name__ == "__main__":
    test_phase0_exploration()
