"""Mode-tuned exit thresholds for NMP_REGIME.

Production thresholds were derived from the MEAN of the per-tier peak
distribution (q_tp=0.30 quantile of peak, gb_min = peak_mean × 0.5).
That overshoots fat-tail distributions: in Q5 high-vol bins the mean
peak was $144 but the modal trade hit much smaller peaks.

This module derives MODE-TUNED thresholds:
    tp_pts(cell) = mode_peak / 2   # take profit at typical (modal) peak
    gb_min(cell) = mode_peak * 0.4 # arm giveback EARLY (at 40% of mode)
    gb_keep(cell) = 0.55            # KEEP 55% of peak — tighter trail
    time_stop_bars: same
    sl_pts: same (capped by SL_PTS_FLOOR=25 anyway)

Why these numbers:
- Mode peak ~ $44-64 depending on regime (vs mean $80-150)
- Arming giveback at 40% of mode = ~$18-26 (vs production $41)
  → giveback fires for the TYPICAL trade, not just outliers
- Keep 55% of peak (vs production 30%)
  → exits at 55% of peak (e.g., peak $50, exit at $27.50)
  → captures more of the typical $44-64 mode peak

Output: cells dict in same shape as threshold_bayesian.py — drop-in for
engine via `--thresholds`.

Usage:
    python -m training.calibration.threshold_mode_tuned --regret training_iso_v2/output/regret_nmp.pkl \
        --out training_iso_v2/output/thresholds_mode.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List

import numpy as np

from training.regret.regret import RegretLabel
from training.utils.state import REGIME_VOCAB
from training.calibration.threshold_optimizer import DEFAULT_THRESHOLDS


ALL_TIERS = ('MA_ALIGN', 'REVERSION', 'VEL_BODY_CHORD',
                  'NMP_KEEP', 'NMP_FLIP', 'CNN_ENTRY', 'BASE_NMP')


def histogram_mode(values: np.ndarray, bin_width: float) -> float:
    if len(values) == 0:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < bin_width:
        return float(np.median(v))
    n_bins = max(1, int(np.ceil((hi - lo) / bin_width)))
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(v, bins=edges)
    if counts.sum() == 0:
        return float(np.median(v))
    j = int(np.argmax(counts))
    return float((edges[j] + edges[j + 1]) / 2)


def derive_mode_thresholds(labels: List[RegretLabel],
                                    gb_keep: float = 0.55,
                                    gb_min_factor: float = 0.4,
                                    bin_width: float = 10.0,
                                    tp_clip: tuple = (5.0, 50.0),
                                    sl_clip: tuple = (5.0, 50.0),
                                    time_stop_bars: int = 480,
                                    ) -> Dict:
    """Derive per-tier mode-tuned thresholds.

    For each tier:
      - Compute mode of peak_pnl (winners + losers; the typical peak amount)
      - tp_pts = mode_peak / 2 (in points; clipped to tp_clip)
      - gb_min = mode_peak * gb_min_factor (arms early)
      - gb_keep = gb_keep (tighter trail than production)
      - sl_pts = median(|MAE|) / 2 (clipped) — basic risk management
      - time_stop_bars = fixed (no fat-tail issue here)
    """
    by_tier: Dict[str, List[RegretLabel]] = {}
    for l in labels:
        by_tier.setdefault(l.entry_tier, []).append(l)

    # Universal pool
    all_peaks = np.array([l.peak_pnl for l in labels], dtype=np.float64)
    all_maes = np.abs(np.array([l.mae_pnl for l in labels], dtype=np.float64))
    universal = _build_threshold(all_peaks, all_maes, gb_keep, gb_min_factor,
                                                bin_width, tp_clip, sl_clip, time_stop_bars)

    # Tier pools
    tier_pools = {}
    for tier in by_tier.keys():
        tier_labels = by_tier[tier]
        peaks = np.array([l.peak_pnl for l in tier_labels], dtype=np.float64)
        maes = np.abs(np.array([l.mae_pnl for l in tier_labels], dtype=np.float64))
        if len(peaks) < 100:
            tier_pools[tier] = dict(universal)
        else:
            tier_pools[tier] = _build_threshold(peaks, maes, gb_keep, gb_min_factor,
                                                            bin_width, tp_clip, sl_clip,
                                                            time_stop_bars)

    # Cells: replicate per-tier value for every regime so engine lookup works
    cells = {}
    seen_keys = set()
    for l in labels:
        key = f'{l.entry_regime_idx}|{l.entry_tier}'
        if key in seen_keys:
            continue
        seen_keys.add(key)
        cells[key] = dict(tier_pools.get(l.entry_tier, universal))
    # Also pre-populate for known additional tier names so engine doesn't
    # fall through to default for unrecognized tiers:
    for r in range(len(REGIME_VOCAB)):
        for t in ALL_TIERS:
            key = f'{r}|{t}'
            if key not in cells:
                cells[key] = dict(tier_pools.get(t, universal))

    return {
        'cells': cells,
        'tier_pools': tier_pools,
        'universal': universal,
        'meta': {
            'method': 'mode-tuned',
            'gb_keep': gb_keep,
            'gb_min_factor': gb_min_factor,
            'bin_width': bin_width,
            'time_stop_bars': time_stop_bars,
            'n_labels': len(labels),
        },
    }


def _build_threshold(peaks: np.ndarray, maes: np.ndarray,
                              gb_keep: float, gb_min_factor: float,
                              bin_width: float, tp_clip: tuple, sl_clip: tuple,
                              time_stop_bars: int) -> Dict:
    mode_peak = histogram_mode(peaks, bin_width)
    median_mae = float(np.median(maes)) if len(maes) else 50.0

    tp_pts = mode_peak / 2.0
    tp_pts = float(np.clip(tp_pts, tp_clip[0], tp_clip[1]))
    sl_pts = median_mae / 2.0
    sl_pts = float(np.clip(sl_pts, sl_clip[0], sl_clip[1]))
    gb_min = max(mode_peak * gb_min_factor, 5.0)

    return {
        'tp_pts': round(tp_pts, 2),
        'sl_pts': round(sl_pts, 2),
        'gb_min': round(gb_min, 2),
        'gb_keep': round(gb_keep, 3),
        'time_stop_bars': int(time_stop_bars),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--regret', default='training_iso_v2/output/regret_nmp.pkl')
    p.add_argument('--out', default='training_iso_v2/output/thresholds_mode.json')
    p.add_argument('--gb-keep', type=float, default=0.55)
    p.add_argument('--gb-min-factor', type=float, default=0.4)
    p.add_argument('--bin-width', type=float, default=10.0)
    p.add_argument('--time-stop-bars', type=int, default=480)
    args = p.parse_args()

    with open(args.regret, 'rb') as f:
        labels = pickle.load(f)
    print(f'Loaded {len(labels)} regret labels')

    thr_map = derive_mode_thresholds(
        labels, gb_keep=args.gb_keep,
        gb_min_factor=args.gb_min_factor,
        bin_width=args.bin_width,
        time_stop_bars=args.time_stop_bars,
    )

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(thr_map, f, indent=2)
    print(f'\nUniversal: {thr_map["universal"]}')
    print(f'\nPer-tier:')
    for tier, thr in thr_map['tier_pools'].items():
        print(f'  {tier:<18}: {thr}')
    print(f'\nSaved -> {args.out}')


if __name__ == '__main__':
    main()
