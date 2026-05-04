"""
core_v2/v1_compat.py — V1 feature-concept derivations from V2 features.

The V2 feature schema (research/feature_spec_v2.md, "Principle 7") deliberately
dropped these V1 concepts because they're algebraic compositions that the CNN
can learn from L1+L2 primitives:

  wick_ratio, p_at_center, variance_ratio, vol_rel, dir_vol, dmi_diff/dmi_gap

Principle 7 is the right move for ML — but the legacy 9-tier engine
(training/nightmare_blended.py) is rule-based and uses these as primary
signals with hardcoded thresholds. To migrate the tier engine to V2 features
without retuning every threshold, we compute the V1 concepts ON THE FLY
from V2 features (and a small OHLCV window where needed).

This shim is for the legacy rule-based tier engine ONLY. New ML-based
components should use V2 primitives directly per Principle 7.

Per-concept derivation:

  wick_ratio   = 1 - abs(L1_body) / max(L1_bar_range, eps)
                 (V1 formula: `1 - abs(close-open)/(high-low)`. body = close-open,
                  bar_range = high-low. Identical with V2 primitives.)

  p_at_center  = softmax over 3 Gaussian energies E0=-z²/2, E1=-(z-2)²/2,
                 E2=-(z+2)²/2. p_at_center = p0.
                 (V1 formula in core/cuda_statistics.py lines 148-167.)

  variance_ratio = std(close[-10:]) / std(close[-60:])
                 (V1 formula in core/features.py lines 305-309. Hardcoded
                  10/60 windows. Needs raw close history; V2 cache doesn't
                  have these specific windows.)

  vol_rel      = volume[-1] / mean(volume[-30:])
                 (V1 formula in core/features.py lines 323-327. Hardcoded
                  30-bar window. Needs raw volume history.)

  dir_vol      = sign(price_velocity_w) * vol_rel
                 (V1 formula in core/features.py lines 355-357. Uses
                  windowed velocity sign × vol_rel.)

  dmi_diff     = (substitute) sign(L2_price_velocity_w) * 5.0
                 V1 DMI is Wilder's directional movement; V2 doesn't have
                 it. The only tier consumer (MTF_BREAKOUT) uses it as a
                 sign filter — substitute velocity-sign × scale matches
                 the threshold check semantics.

All exposed functions accept either:
  - a single dict of V2 feature name -> value (single-bar inference path), OR
  - aligned numpy arrays + raw OHLCV pandas series (batch path).

Lookahead discipline: every formula uses ONLY current-bar values + past N
bars of OHLCV. The V2 features are already lookahead-clean (verified by
tests/test_core_v2_lookahead.py). The shim adds zero new alignment.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Optional


_EPS = 1e-8


# ─── Single-bar derivations (fast path for live engine) ──────────────────

def wick_ratio_from_v2(body: float, bar_range: float) -> float:
    """V1 combined wick: 1 - abs(body) / bar_range.

    Returns total wick fraction (upper + lower) of the bar's range.
    Returns 0.0 for zero-range bars (degenerate).
    """
    if bar_range <= _EPS:
        return 0.0
    return 1.0 - abs(body) / bar_range


def directional_wicks_from_ohlc(open_: float, high: float, low: float,
                                    close: float) -> tuple[float, float]:
    """Compute (upper_wick_ratio, lower_wick_ratio) from raw OHLC.

    upper_wick = high - max(open, close)   — rejection at the ceiling
    lower_wick = min(open, close) - low    — rejection at the support
    ratio = wick_size / bar_range          — fraction in [0, 1]

    Both summed with abs(body)/bar_range = 1.0 (decomposes the bar):
        upper_ratio + lower_ratio + abs(body)/bar_range = 1

    Direction encoding:
      Strong lower wick + small upper wick => long bias (bounced off support).
      Strong upper wick + small lower wick => short bias (rejected at ceiling).
      Both wicks small => trending bar (clean direction in body).
      Both wicks large + small body => indecision (doji).
    """
    bar_range = high - low
    if bar_range <= _EPS:
        return 0.0, 0.0
    body_top = max(open_, close)
    body_bot = min(open_, close)
    upper_wick = max(high - body_top, 0.0)
    lower_wick = max(body_bot - low, 0.0)
    return upper_wick / bar_range, lower_wick / bar_range


def directional_wicks_batch(open_: np.ndarray, high: np.ndarray,
                                low: np.ndarray, close: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized directional_wicks_from_ohlc."""
    bar_range = high - low
    safe_range = np.where(bar_range > _EPS, bar_range, 1.0)
    body_top = np.maximum(open_, close)
    body_bot = np.minimum(open_, close)
    upper_wick = np.maximum(high - body_top, 0.0)
    lower_wick = np.maximum(body_bot - low, 0.0)
    upper_ratio = upper_wick / safe_range
    lower_ratio = lower_wick / safe_range
    upper_ratio[bar_range <= _EPS] = 0.0
    lower_ratio[bar_range <= _EPS] = 0.0
    return upper_ratio, lower_ratio


def p_at_center_from_z(z_se: float) -> float:
    """V1 p_at_center: softmax(p0=N(0,1), p1=N(2,1), p2=N(-2,1)).

    Energies (negative log-likelihoods):
      E0 = -z²/2     (centered at z=0)
      E1 = -(z-2)²/2 (centered at z=+2σ)
      E2 = -(z+2)²/2 (centered at z=-2σ)

    Returns p0 = P(near regression mean).

    For numerical stability, subtract max energy before exp.
    """
    e0 = -0.5 * z_se * z_se
    e1 = -0.5 * (z_se - 2.0) * (z_se - 2.0)
    e2 = -0.5 * (z_se + 2.0) * (z_se + 2.0)
    m = max(e0, e1, e2)
    p0 = math.exp(e0 - m)
    p1 = math.exp(e1 - m)
    p2 = math.exp(e2 - m)
    return p0 / (p0 + p1 + p2)


def vol_rel_from_history(volume_now: float,
                           volume_history: np.ndarray,
                           window: int = 30) -> float:
    """V1: volume[-1] / mean(volume[-window:]).

    volume_history is the trailing volume series (len >= 1).
    """
    n = len(volume_history)
    if n == 0:
        return 1.0
    take = volume_history[-window:] if n >= window else volume_history
    avg = float(take.mean())
    avg = max(avg, 1.0)
    return float(volume_now) / avg


def variance_ratio_from_history(close_history: np.ndarray,
                                   short: int = 10, long: int = 60) -> float:
    """V1: std(close[-short:]) / std(close[-long:]).

    Falls back to short=5/long=full when n<long, returns 1.0 when n<short."""
    n = len(close_history)
    if n < short:
        return 1.0
    if n < long:
        s = float(np.std(close_history[-5:])) if n >= 10 else 1.0
        l = float(np.std(close_history))
        return s / l if l > _EPS else 1.0
    s = float(np.std(close_history[-short:]))
    l = float(np.std(close_history[-long:]))
    return s / l if l > _EPS else 1.0


def dir_vol_from_v2(price_velocity_w: float, vol_rel: float) -> float:
    """V1: sign(velocity) * vol_rel."""
    if price_velocity_w > 0:
        return vol_rel
    if price_velocity_w < 0:
        return -vol_rel
    return 0.0


def dmi_substitute_from_v2(price_velocity_w: float,
                              scale: float = 5.0) -> float:
    """V1 DMI substitute: sign(price_velocity_w) * scale.

    The only consumer in nightmare_blended.py (MTF_BREAKOUT line 785-787)
    checks `dmi > -5` for long and `dmi < 5` for short. Returning ±5
    preserves the filter semantics: any positive velocity yields dmi=+5,
    any negative yields dmi=-5, zero yields 0.

    DOWNGRADE: V1 DMI is a smoothed +DI−−DI difference with a real
    distribution shape. The substitute only carries direction, not
    magnitude. If MTF_BREAKOUT later wants to threshold magnitude
    (`abs(dmi) > X`), this substitute will misbehave.
    """
    if price_velocity_w > 0:
        return scale
    if price_velocity_w < 0:
        return -scale
    return 0.0


# ─── Batch derivations (parity tests, cache rebuilds) ────────────────────

def derive_v1_concepts_batch(v2_df: pd.DataFrame,
                                ohlcv_native: pd.DataFrame,
                                tf: str,
                                tf_period_seconds: int) -> pd.DataFrame:
    """Compute V1 concepts for all rows of v2_df at one TF.

    Args:
      v2_df: DataFrame with V2 columns L1_<tf>_body, L1_<tf>_bar_range,
             L2_<tf>_price_velocity_w, L3_<tf>_z_se_w. Indexed by 5s
             anchor timestamps (one row per 5s anchor bar).
      ohlcv_native: raw OHLCV at the NATIVE TF cadence (one row per
             TF bar, NOT step-filled to 5s anchor). Should include
             history before v2_df's first timestamp for the
             variance_ratio 60-bar window.
      tf: timeframe label (e.g. '1m')
      tf_period_seconds: TF period in seconds (60 for 1m, 300 for 5m, etc.)

    Returns DataFrame with same length as v2_df, columns:
      wick_ratio, p_at_center, variance_ratio, vol_rel, dir_vol, dmi_substitute.

    The variance_ratio and vol_rel are computed at TF cadence using V1's
    exact windows (std-10 / std-60 for VR, mean-30 for vol_rel) then
    step-filled to the 5s anchor via last-closed-bar lookup, matching
    the V1 cache's alignment semantics.
    """
    n = len(v2_df)
    out = pd.DataFrame(index=v2_df.index)

    body = v2_df[f'L1_{tf}_body'].values
    bar_range = v2_df[f'L1_{tf}_bar_range'].values
    price_vel_w = v2_df[f'L2_{tf}_price_velocity_w'].values
    z_se = v2_df[f'L3_{tf}_z_se_w'].values

    # ── wick_ratio (uses V2's L1 body/bar_range — already TF-aligned) ──
    safe_range = np.where(bar_range > _EPS, bar_range, 1.0)
    wr = 1.0 - np.abs(body) / safe_range
    wr[bar_range <= _EPS] = 0.0
    out['wick_ratio'] = wr

    # ── p_at_center (uses V2's L3 z_se — already TF-aligned) ──────────
    e0 = -0.5 * z_se * z_se
    e1 = -0.5 * (z_se - 2.0) ** 2
    e2 = -0.5 * (z_se + 2.0) ** 2
    m = np.maximum(np.maximum(e0, e1), e2)
    p0 = np.exp(e0 - m)
    p1 = np.exp(e1 - m)
    p2 = np.exp(e2 - m)
    out['p_at_center'] = p0 / (p0 + p1 + p2)

    # ── variance_ratio + vol_rel: native TF cadence, then step-fill ──
    if 'timestamp' not in ohlcv_native.columns:
        raise ValueError("ohlcv_native missing 'timestamp' column")
    tf_ts = ohlcv_native['timestamp'].values
    if pd.api.types.is_datetime64_any_dtype(tf_ts):
        tf_ts = (ohlcv_native['timestamp'].astype('int64') // 10**9).values
    tf_ts = tf_ts.astype(np.int64)
    closes = ohlcv_native['close'].values.astype(np.float64)
    volumes = (ohlcv_native['volume'].values if 'volume' in ohlcv_native.columns
                  else np.ones(len(ohlcv_native))).astype(np.float64)
    n_tf = len(closes)

    # Compute VR + vol_rel at native TF cadence using V1 windows
    vr_native = np.full(n_tf, 1.0)
    for j in range(n_tf):
        if j >= 60:
            s = float(np.std(closes[j-9:j+1]))
            l = float(np.std(closes[j-59:j+1]))
            vr_native[j] = s / l if l > _EPS else 1.0
        elif j >= 10:
            s = float(np.std(closes[max(j-4, 0):j+1]))
            l = float(np.std(closes[:j+1]))
            vr_native[j] = s / l if l > _EPS else 1.0
    vol_rel_native = np.full(n_tf, 1.0)
    for j in range(n_tf):
        start = max(j - 29, 0)
        win = volumes[start:j+1]
        avg = max(float(win.mean()), 1.0)
        vol_rel_native[j] = volumes[j] / avg

    # Step-fill to anchor: for each anchor ts, idx of last closed TF bar
    anchor_ts = v2_df['timestamp'].values.astype(np.int64) if 'timestamp' in v2_df.columns else \
                  np.arange(n, dtype=np.int64)
    if 'timestamp' not in v2_df.columns:
        raise ValueError("v2_df missing 'timestamp' column required for alignment")
    last_closed = np.searchsorted(tf_ts, anchor_ts - tf_period_seconds, side='right') - 1
    valid = (last_closed >= 0) & (last_closed < n_tf)
    safe_idx = np.clip(last_closed, 0, n_tf - 1)

    vr_aligned = np.where(valid, vr_native[safe_idx], 1.0)
    vol_rel_aligned = np.where(valid, vol_rel_native[safe_idx], 1.0)

    out['variance_ratio'] = vr_aligned
    out['vol_rel'] = vol_rel_aligned

    # ── dir_vol = sign(price_velocity_w) * vol_rel ───────────────────
    out['dir_vol'] = np.where(price_vel_w > 0, vol_rel_aligned,
                                  np.where(price_vel_w < 0, -vol_rel_aligned, 0.0))

    # ── DMI substitute ──────────────────────────────────────────────
    out['dmi_substitute'] = np.where(price_vel_w > 0, 5.0,
                                          np.where(price_vel_w < 0, -5.0, 0.0))

    return out


# ─── Sanity self-test ────────────────────────────────────────────────────

if __name__ == '__main__':
    # Quick sanity checks
    assert wick_ratio_from_v2(body=0.0, bar_range=10.0) == 1.0  # no body, all wick
    assert wick_ratio_from_v2(body=10.0, bar_range=10.0) == 0.0  # all body, no wick
    assert wick_ratio_from_v2(body=5.0, bar_range=10.0) == 0.5

    # p_at_center: at z=0, p0 should dominate
    assert p_at_center_from_z(0.0) > p_at_center_from_z(2.0)
    assert abs(p_at_center_from_z(0.0) - 1.0 / (1.0 + 2.0 * math.exp(-2.0))) < 1e-9

    # vol_rel: equal to volume_now / mean
    h = np.array([100.0] * 30)
    assert abs(vol_rel_from_history(150.0, h, 30) - 1.5) < 1e-9

    print("v1_compat sanity tests pass")
