"""Price-first I-MR computation: control charts, regime detection, and oracle MFE/MAE."""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  SPC constants for n=2 subgroup
# ---------------------------------------------------------------------------
D4 = 3.267
E2 = 2.660


def compute_price_imr(base_df, context_days=21, analysis_days=7):
    """Pure price-based I-MR chart from 15m close prices.

    I chart = raw close price per bar.
    MR = |close[t] - close[t-1]| (bar-to-bar price movement).
    Control limits calibrated from the warmup (context) period.

    Returns dict with all arrays and control limit values.
    """
    close = base_df['close'].values.astype(float)
    timestamps = base_df['timestamp'].values.astype(float)
    n = len(close)

    # MR: bar-to-bar signed price change (tracks direction)
    mr_signed = np.diff(close)
    mr_signed = np.concatenate([[0.0], mr_signed])  # pad first bar with 0
    mr_abs = np.abs(mr_signed)

    # Determine warmup boundary (in bar count)
    t_min, t_max = timestamps[0], timestamps[-1]
    data_span_days = (t_max - t_min) / 86400

    # Auto-adjust if data is shorter than context window
    if context_days > 0 and data_span_days < context_days + 1:
        old_ctx = context_days
        context_days = max(0, int(data_span_days * 0.3))
        print(f"  Auto-adjusted context: {old_ctx}d -> {context_days}d "
              f"(data span is only {data_span_days:.1f}d)")

    t_warmup_end = t_min + context_days * 86400

    warmup_mask = timestamps < t_warmup_end
    warmup_end_idx = int(warmup_mask.sum())

    if warmup_end_idx < 20:
        # Not enough warmup — use first 30% of data
        warmup_end_idx = max(20, int(n * 0.3))

    # Control limits from warmup period (use absolute MR for limits)
    warmup_close = close[:warmup_end_idx]
    warmup_mr_abs = mr_abs[1:warmup_end_idx]  # skip first MR (=0)

    center = float(np.mean(warmup_close))
    mr_bar = float(np.mean(warmup_mr_abs)) if len(warmup_mr_abs) > 0 else 1.0

    ucl_mr = D4 * mr_bar
    ucl_i = center + E2 * mr_bar
    lcl_i = center - E2 * mr_bar

    # Analysis window
    if analysis_days > 0:
        t_analysis_end = t_warmup_end + analysis_days * 86400
        analysis_mask = (timestamps >= t_warmup_end) & (timestamps < t_analysis_end)
    else:
        analysis_mask = timestamps >= t_warmup_end

    print(f"  Price I-MR: {n} bars, warmup={warmup_end_idx}, "
          f"analysis={int(analysis_mask.sum())}")
    print(f"  Center={center:.2f}, MR_bar={mr_bar:.2f}, "
          f"UCL_MR={ucl_mr:.2f}, UCL_I={ucl_i:.2f}, LCL_I={lcl_i:.2f}")

    return {
        'close': close,
        'mr': mr_signed,
        'mr_abs': mr_abs,
        'timestamps': timestamps,
        'center': center,
        'mr_bar': mr_bar,
        'ucl_mr': ucl_mr,
        'ucl_i': ucl_i,
        'lcl_i': lcl_i,
        'warmup_end_idx': warmup_end_idx,
        'analysis_mask': analysis_mask,
    }


def detect_regimes(price_imr, min_regime_bars=8):
    """Detect natural price regimes from MR UCL breaks.

    A new regime starts when MR > UCL_MR (price behavior changed character).
    Tiny regimes (< min_regime_bars) get merged into their larger neighbor.

    Returns:
        regime_ids: array of regime IDs per bar (full length, -1 for warmup)
        regime_meta: list of dicts with regime stats
    """
    close = price_imr['close']
    mr_abs = price_imr['mr_abs']
    ucl_mr = price_imr['ucl_mr']
    warmup_end = price_imr['warmup_end_idx']
    analysis_mask = price_imr['analysis_mask']
    n = len(close)

    # Initialize all bars to -1 (warmup/excluded)
    regime_ids = np.full(n, -1, dtype=int)

    # Find analysis bar indices
    analysis_indices = np.where(analysis_mask)[0]
    if len(analysis_indices) == 0:
        return regime_ids, []

    # Assign regime IDs: new regime at each |MR| > UCL break
    current_regime = 0
    for i, idx in enumerate(analysis_indices):
        if i > 0 and mr_abs[idx] > ucl_mr:
            current_regime += 1
        regime_ids[idx] = current_regime

    n_raw = current_regime + 1

    # Merge tiny regimes into neighbors. Vectorized via np.bincount: per-pass
    # cost is O(n + R) instead of O(n × R) for the dict-comprehension version.
    # Merges all tiny regimes in one pass (each into its larger neighbor),
    # repeats until no tiny regimes remain or pass cap hits.
    valid_mask = regime_ids >= 0
    n_valid_bars = int(valid_mask.sum())
    max_passes = max(64, int(np.log2(max(n_raw, 2))) + 4)
    for _pass in range(max_passes):
        active = regime_ids[valid_mask]
        if active.size == 0:
            break
        max_id = int(active.max())
        counts = np.bincount(active, minlength=max_id + 1)
        unique_ids = sorted({int(r) for r in np.unique(active)})
        # Skip already-empty IDs (left over after prior merges).
        live = [r for r in unique_ids if counts[r] > 0]
        tiny = [r for r in live if counts[r] < min_regime_bars]
        if not tiny:
            break
        # Build merge map: each tiny ID → its larger neighbor (in live order).
        live_idx = {r: i for i, r in enumerate(live)}
        merge_to = {}
        for r in tiny:
            i = live_idx[r]
            left = live[i - 1] if i - 1 >= 0 else None
            right = live[i + 1] if i + 1 < len(live) else None
            # Skip if neighbor is also tiny — let the next pass handle it,
            # so we don't chain into a tiny target.
            if left is not None and counts[left] >= min_regime_bars and \
               right is not None and counts[right] >= min_regime_bars:
                merge_to[r] = left if counts[left] >= counts[right] else right
            elif left is not None and counts[left] >= min_regime_bars:
                merge_to[r] = left
            elif right is not None and counts[right] >= min_regime_bars:
                merge_to[r] = right
            elif left is not None:
                merge_to[r] = left
            elif right is not None:
                merge_to[r] = right
        if not merge_to:
            break
        # Apply merges in one vectorized pass via remap LUT.
        lut = np.arange(max_id + 1, dtype=regime_ids.dtype)
        for src, dst in merge_to.items():
            lut[src] = dst
        regime_ids[valid_mask] = lut[regime_ids[valid_mask]]

    # Re-compact to 0-based contiguous (vectorized).
    unique_ids = sorted({int(r) for r in np.unique(regime_ids) if r >= 0})
    remap = np.full(int(np.max(regime_ids)) + 2, -1, dtype=regime_ids.dtype)
    for new_id, old_id in enumerate(unique_ids):
        remap[old_id] = new_id
    pos_mask = regime_ids >= 0
    regime_ids[pos_mask] = remap[regime_ids[pos_mask]]

    n_regimes = len(unique_ids)

    # Build regime metadata
    regime_meta = []
    for rid in range(n_regimes):
        mask = regime_ids == rid
        indices = np.where(mask)[0]
        r_close = close[mask]
        r_mr = mr_abs[mask]

        regime_meta.append({
            'regime_id': rid,
            'start_idx': int(indices[0]),
            'end_idx': int(indices[-1]),
            'n_bars': int(mask.sum()),
            'mean_price': float(np.mean(r_close)),
            'volatility': float(np.mean(r_mr)),
            'price_change': float(r_close[-1] - r_close[0]) if len(r_close) > 1 else 0.0,
            'direction': 'LONG' if (r_close[-1] > r_close[0]) else 'SHORT',
        })

    print(f"  Regimes: {n_raw} raw -> {n_regimes} after merge "
          f"(min_bars={min_regime_bars})")
    for rm in regime_meta:
        print(f"    R{rm['regime_id']}: {rm['n_bars']:>4} bars, "
              f"price={rm['mean_price']:.1f}, vol={rm['volatility']:.2f}, "
              f"dir={rm['direction']}, chg={rm['price_change']:+.1f}")

    return regime_ids, regime_meta


def compute_regime_oracle(base_df, regime_ids, regime_meta, lookahead=16):
    """Compute oracle MFE/MAE per analysis bar using regime-based direction.

    Direction comes from the regime's price trend (not z-score).

    Returns:
        bar_indices: array of bar indices in base_df
        mfes: array of MFE values
        maes: array of MAE values
        directions: array of 'LONG'/'SHORT' strings
    """
    close = base_df['close'].values.astype(float)
    high = base_df['high'].values.astype(float)
    low = base_df['low'].values.astype(float)
    n = len(base_df)

    # Build regime direction lookup
    regime_dir = {}
    for rm in regime_meta:
        regime_dir[rm['regime_id']] = rm['direction']

    analysis_indices = np.where(regime_ids >= 0)[0]

    bar_indices = []
    mfes = []
    maes = []
    directions = []

    for idx in analysis_indices:
        if idx + lookahead >= n:
            continue

        entry = close[idx]
        future_high = high[idx + 1: idx + 1 + lookahead]
        future_low = low[idx + 1: idx + 1 + lookahead]

        if len(future_high) == 0:
            continue

        max_up = float(future_high.max() - entry)
        max_down = float(entry - future_low.min())

        rid = regime_ids[idx]
        direction = regime_dir.get(rid, 'LONG')

        if direction == 'LONG':
            mfe_val = max_up
            mae_val = max_down
        else:
            mfe_val = max_down
            mae_val = max_up

        bar_indices.append(idx)
        mfes.append(mfe_val)
        maes.append(mae_val)
        directions.append(direction)

    print(f"  Oracle: {len(mfes)} bars with MFE/MAE "
          f"(lookahead={lookahead})")

    return (np.array(bar_indices), np.array(mfes),
            np.array(maes), np.array(directions))

