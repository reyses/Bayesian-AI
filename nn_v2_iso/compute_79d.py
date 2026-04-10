"""
Compute 79D features from Aggregator + SFE — single source of truth.

Used by:
  - live/live_engine.py (live trading)
  - nn_v2/build_dataset_v2.py (batch feature building)

Guarantees: SFE and OHLCV-derived features use the SAME windowed data.
"""
import numpy as np
from typing import Dict, Tuple, Optional

from core.statistical_field_engine import StatisticalFieldEngine
from core.features_79d import extract_79d, TF_ORDER, FEATURE_NAMES_79D

SFE_MIN_BARS = 21
SFE_WINDOW = 300  # max bars to feed SFE


def compute_79d_from_aggregator(
    agg,
    sfe: StatisticalFieldEngine,
    prev_velocities: dict,
    ts: float,
) -> Tuple[Optional[np.ndarray], dict, dict, dict]:
    """Compute 79D feature vector from aggregator state.

    Args:
        agg: nn_v2.Aggregator with accumulated bars
        sfe: StatisticalFieldEngine instance
        prev_velocities: velocity state from previous computation
        ts: current timestamp

    Returns:
        (feat_79d, prev_velocities, states_by_tf, ohlcv_by_tf)
        feat_79d is None if insufficient data.
    """
    states_by_tf = {}
    ohlcv_by_tf = {}

    for tf in TF_ORDER:
        df = agg.get_closed_bars_df(tf)
        if len(df) < SFE_MIN_BARS:
            continue

        # Window to SFE_WINDOW — SFE and features use the SAME data
        windowed = df.tail(SFE_WINDOW).reset_index(drop=True) if len(df) > SFE_WINDOW else df
        ohlcv_by_tf[tf] = windowed

        states = sfe.batch_compute_states(windowed)
        if states:
            states_by_tf[tf] = states[-1]

    if '1m' not in states_by_tf:
        return None, prev_velocities, states_by_tf, ohlcv_by_tf

    feat, prev_velocities = extract_79d(
        states_by_tf, ohlcv_by_tf, prev_velocities, ts)

    return feat, prev_velocities, states_by_tf, ohlcv_by_tf
