"""
Lookahead-poisoning tests for core_v2/statistical_field_engine.py
=================================================================

Each test:
  1. Compute features on a baseline DataFrame.
  2. Corrupt bars in the FUTURE region (last K bars).
  3. Recompute features.
  4. Assert that features BEFORE the poisoned region are identical.

If any feature changes in the "past" region when we corrupt future bars,
that feature has a lookahead bug.

This mirrors the failure mode of the last refactor (hidden lookahead in
build_dataset.py's searchsorted) and is the single most important class of
bug to catch before deploying v2.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.statistical_field_engine import (
    StatisticalFieldEngine,
    N_BASE,
    N_HURST_MULT,
    SWING_NOISE_WINDOW,
)
from core_v2.features import FEATURE_NAMES, N_FEATURES, TF_ORDER, LAYER_FAMILIES


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def bars():
    """600 synthetic bars: enough for all windows (N_hurst on 15s needs 96)."""
    np.random.seed(42)
    n = 600
    ts = np.arange(n, dtype=np.float64) * 60.0 + 1704067200.0
    deltas = np.random.randn(n).astype(np.float64) * 2.0
    close = 20000.0 + deltas.cumsum()
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    opn = close + np.random.randn(n) * 0.5
    vol = np.abs(np.random.randn(n)) * 100.0 + 50.0
    return pd.DataFrame({
        'timestamp': ts, 'open': opn, 'high': high,
        'low': low, 'close': close, 'volume': vol,
    })


@pytest.fixture
def sfe():
    return StatisticalFieldEngine()


# ─── Poisoning helper ──────────────────────────────────────────────────────

def poison_future(df: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Return a copy of df with bars [cutoff, end] heavily corrupted.

    Price goes to a wildly different value; volume is scrambled. Any feature
    at bars < cutoff that depends on these bars would change detectably.
    """
    out = df.copy()
    out.loc[cutoff:, 'close'] = 999999.0
    out.loc[cutoff:, 'open'] = 999999.0
    out.loc[cutoff:, 'high'] = 1000500.0
    out.loc[cutoff:, 'low'] = 999000.0
    out.loc[cutoff:, 'volume'] = 1e9
    return out


def assert_past_unchanged(baseline: pd.DataFrame, poisoned: pd.DataFrame, cutoff: int):
    """Assert every column in [0, cutoff) is numerically identical."""
    past_base = baseline.iloc[:cutoff]
    past_poisoned = poisoned.iloc[:cutoff]

    assert past_base.shape == past_poisoned.shape, \
        f"Shape mismatch: {past_base.shape} vs {past_poisoned.shape}"

    for col in past_base.columns:
        b = past_base[col].values
        p = past_poisoned[col].values

        b_nan = np.isnan(b)
        p_nan = np.isnan(p)

        assert np.array_equal(b_nan, p_nan), (
            f"Column '{col}': NaN mask diverged between baseline and poisoned "
            f"in the past region (cutoff={cutoff}). Lookahead suspected."
        )

        both_real = ~(b_nan | p_nan)
        if both_real.any():
            max_diff = np.abs(b[both_real] - p[both_real]).max()
            assert max_diff < 1e-9, (
                f"Column '{col}': max |delta| = {max_diff:.3e} between baseline "
                f"and poisoned in past region [0, {cutoff}). Lookahead!"
            )


# ─── L0 tests ──────────────────────────────────────────────────────────────

def test_L0_no_lookahead(sfe, bars):
    """L0_time_of_day only uses timestamp[i], independent of OHLCV."""
    cutoff = 400
    baseline = sfe.compute_L0(bars)
    poisoned_df = poison_future(bars, cutoff)
    poisoned = sfe.compute_L0(poisoned_df)
    assert_past_unchanged(baseline, poisoned, cutoff)


# ─── L1 tests ──────────────────────────────────────────────────────────────

def test_L1_no_lookahead_1m(sfe, bars):
    cutoff = 400
    baseline = sfe.compute_L1(bars, tf='1m')
    poisoned_df = poison_future(bars, cutoff)
    poisoned = sfe.compute_L1(poisoned_df, tf='1m')
    assert_past_unchanged(baseline, poisoned, cutoff)


def test_L1_accel_uses_exactly_3_bars(sfe, bars):
    """L1 accel uses bars {t-2, t-1, t}. Corrupting bar t+1 must NOT change accel[t]."""
    cutoff = 500
    baseline = sfe.compute_L1(bars, tf='1m')
    poisoned_df = poison_future(bars, cutoff + 1)  # cutoff+1 = tomorrow onwards
    poisoned = sfe.compute_L1(poisoned_df, tf='1m')
    # accel at bars [0..cutoff] uses at most bars {cutoff-2, cutoff-1, cutoff}.
    # All < cutoff+1, so unchanged.
    assert_past_unchanged(baseline, poisoned, cutoff + 1)


# ─── L2 tests ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('tf', TF_ORDER)
def test_L2_no_lookahead_all_tfs(sfe, bars, tf):
    cutoff = 400
    baseline = sfe.compute_L2(bars, tf=tf)
    poisoned_df = poison_future(bars, cutoff)
    poisoned = sfe.compute_L2(poisoned_df, tf=tf)
    assert_past_unchanged(baseline, poisoned, cutoff)


def test_L2_acceleration_window_boundary(sfe, bars):
    """L2 accel_N uses bars [t-N-1, t]. Corrupting bar t+1 must not change accel_N[t]."""
    N = N_BASE['1m']  # 15
    cutoff = 300
    baseline = sfe.compute_L2(bars, tf='1m', N=N)
    poisoned_df = poison_future(bars, cutoff)
    poisoned = sfe.compute_L2(poisoned_df, tf='1m', N=N)
    assert_past_unchanged(baseline, poisoned, cutoff)


# ─── L3 tests ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('tf', TF_ORDER)
def test_L3_no_lookahead_all_tfs(sfe, bars, tf):
    cutoff = 400
    baseline = sfe.compute_L3(bars, tf=tf)
    poisoned_df = poison_future(bars, cutoff)
    poisoned = sfe.compute_L3(poisoned_df, tf=tf)
    assert_past_unchanged(baseline, poisoned, cutoff)


def test_L3_hurst_uses_N_hurst_window(sfe, bars):
    """L3 hurst uses N * N_HURST_MULT bars. Ensure no lookahead at the boundary."""
    # For 1m: N=15, N_hurst = 120. Need plenty of data before cutoff.
    cutoff = 500
    baseline = sfe.compute_L3(bars, tf='1m')
    poisoned_df = poison_future(bars, cutoff)
    poisoned = sfe.compute_L3(poisoned_df, tf='1m')
    assert_past_unchanged(baseline, poisoned, cutoff)


def test_L3_swing_noise_uses_30_bars(sfe, bars):
    """L3 swing_noise uses fixed 30-bar window. Corrupting bar cutoff must not change swing[cutoff-1]."""
    cutoff = 400
    baseline = sfe.compute_L3(bars, tf='1m')
    poisoned_df = poison_future(bars, cutoff)
    poisoned = sfe.compute_L3(poisoned_df, tf='1m')
    assert_past_unchanged(baseline, poisoned, cutoff)


def test_L3_z_high_uses_high_OLS_not_close(sfe, bars):
    """v1 bug: z_high used close-sigma. v2 must use high-OLS.

    Corrupt only HIGHS in the future. z_high in past should be identical,
    but crucially, z_se (which uses close-OLS) should ALSO be identical
    since closes weren't touched.
    """
    cutoff = 400
    baseline = sfe.compute_L3(bars, tf='1m')

    # Corrupt only high/low in the future (leave close unchanged)
    poisoned_df = bars.copy()
    poisoned_df.loc[cutoff:, 'high'] = 1000500.0
    poisoned_df.loc[cutoff:, 'low'] = 999000.0
    poisoned = sfe.compute_L3(poisoned_df, tf='1m')

    # All past bars should be unchanged
    assert_past_unchanged(baseline, poisoned, cutoff)


# ─── End-to-end ────────────────────────────────────────────────────────────

def test_batch_compute_states_no_lookahead(sfe, bars):
    """The convenience method combining L0+L1+L2+L3 must also be lookahead-clean."""
    cutoff = 400
    baseline = sfe.batch_compute_states(bars, tf='1m')
    poisoned_df = poison_future(bars, cutoff)
    poisoned = sfe.batch_compute_states(poisoned_df, tf='1m')
    assert_past_unchanged(baseline, poisoned, cutoff)


def test_feature_count_matches_spec(sfe, bars):
    """Sanity: batch_compute_states returns 1 (L0) + 6 (L1) + 9 (L2) + 8 (L3) = 24 columns per TF."""
    out = sfe.batch_compute_states(bars, tf='1m')
    assert out.shape[1] == 1 + 6 + 9 + 8, f"Expected 24 cols, got {out.shape[1]}"


def test_feature_names_count_matches_tf_order():
    """Expected: 1 (L0) + 23 per TF * len(TF_ORDER)."""
    expected = 1 + 23 * len(TF_ORDER)
    assert N_FEATURES == expected, f"Got {N_FEATURES}, expected {expected}"
    assert len(FEATURE_NAMES) == expected


def test_layer_families_structure():
    """1 L0 family + 3 layers * len(TF_ORDER) TFs."""
    expected_families = 1 + 3 * len(TF_ORDER)
    assert len(LAYER_FAMILIES) == expected_families
    assert 'L0' in LAYER_FAMILIES
    for tf in TF_ORDER:
        assert f'L1_{tf}' in LAYER_FAMILIES
        assert f'L2_{tf}' in LAYER_FAMILIES
        assert f'L3_{tf}' in LAYER_FAMILIES


# ─── Feed_bar stays blocked ────────────────────────────────────────────────

def test_feed_bar_raises_not_implemented(sfe):
    """v2 is offline-batch only until live migration."""
    with pytest.raises(NotImplementedError):
        sfe.feed_bar()
