"""Bayesian regime-only exit thresholds.

Replaces threshold_optimizer.py's grid search with a *derivation*:
  thresholds(regime) = f(posterior_distribution_in_regime)

No combo search, no per-IS argmax. The thresholds are functions of the
cell's empirical peak / MAE / time-to-peak / capture distributions, with
hierarchical shrinkage toward a universal pool when the cell is thin.

Granularity: REGIME ONLY (6 cells: UP_SMOOTH, UP_CHOPPY, ..., FLAT_CHOPPY).
The same exit policy applies to ALL trades in a regime regardless of which
strategy (tier) fired the entry — exits respond to the MARKET's post-entry
behavior, which is a property of regime, not of the entry mechanism.

Three knobs (global, optional walk-forward CV):
    q_tp        : take-profit quantile of the peak distribution    (default 0.30)
    q_sl        : stop-loss quantile of the |MAE| distribution     (default 0.70)
    ttp_factor  : time-stop multiplier on median time-to-peak      (default 1.5)

Output JSON has the SAME shape as threshold_optimizer.py output (so the
engine's `lookup_thresholds` works unchanged): cells keyed `{regime}|{tier}`
with all tiers in a regime sharing one threshold dict, plus a universal pool.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from training_iso_v2.regret import RegretLabel
from training_iso_v2.state import REGIME_VOCAB
from training_iso_v2.threshold_optimizer import DEFAULT_THRESHOLDS


# ─── Defaults / floors ────────────────────────────────────────────────────

# Global formula knobs (can be overridden via CLI)
Q_TP_DEFAULT = 0.30          # 30th percentile of peak — small consistent TP
Q_SL_DEFAULT = 0.70          # 70th percentile of |MAE| — generous SL latitude
TTP_FACTOR_DEFAULT = 1.5     # 1.5× median TtP for time stop

# Hierarchical shrinkage: minimum cell n before we trust the cell's quantiles
MIN_N_REGIME = 200           # below this, blend toward universal

# Hard floors / ceilings (risk management)
TP_MIN_PTS = 5.0
TP_MAX_PTS = 200.0
SL_MIN_PTS = 5.0             # never wider than this would mean $-10 SL is fine
SL_MAX_PTS = 50.0            # cap at $-100 SL even on noisy quantiles
TIME_STOP_MIN_BARS = 60      # 5 min minimum
TIME_STOP_MAX_BARS = 720     # 60 min maximum (matches regret horizon)

# Tier list — each regime emits the same threshold for all tiers, but the
# JSON shape is per-cell so the engine's lookup_thresholds works unchanged.
ALL_TIERS = ('MA_ALIGN', 'REVERSION', 'VEL_BODY_CHORD',
                  'CNN_ENTRY', 'BASE_NMP')


# ─── Per-cell stats ────────────────────────────────────────────────────────

@dataclass
class RegimeStats:
    """Empirical distribution stats per regime."""
    n: int = 0
    peak_q_tp: float = 0.0
    peak_mu: float = 0.0
    mae_q_sl: float = 0.0
    mae_mu: float = 0.0
    ttp_median: float = 0.0
    ttp_mu: float = 0.0
    capture_mu: float = 0.0


def _compute_stats(labels: List[RegretLabel],
                        q_tp: float, q_sl: float) -> RegimeStats:
    """Compute the empirical stats needed to derive thresholds."""
    if not labels:
        return RegimeStats()
    peaks = np.array([l.peak_pnl for l in labels], dtype=np.float64)
    maes = np.array([l.mae_pnl for l in labels], dtype=np.float64)  # negative
    ttps = np.array([l.time_to_peak_s for l in labels], dtype=np.float64)
    caps = np.array([l.capture_ratio for l in labels
                          if l.capture_ratio == l.capture_ratio  # not NaN
                          and 0 <= l.capture_ratio <= 1], dtype=np.float64)

    return RegimeStats(
        n=len(labels),
        peak_q_tp=float(np.quantile(peaks, q_tp)) if len(peaks) else 0.0,
        peak_mu=float(peaks.mean()) if len(peaks) else 0.0,
        mae_q_sl=float(np.quantile(np.abs(maes), q_sl)) if len(maes) else 0.0,
        mae_mu=float(np.abs(maes).mean()) if len(maes) else 0.0,
        ttp_median=float(np.median(ttps)) if len(ttps) else 0.0,
        ttp_mu=float(ttps.mean()) if len(ttps) else 0.0,
        capture_mu=float(caps.mean()) if len(caps) else 0.05,
    )


def _shrink(cell: RegimeStats, universal: RegimeStats,
                shrinkage_n: float = MIN_N_REGIME) -> RegimeStats:
    """Empirical-Bayes shrinkage toward universal pool when n is thin."""
    if cell.n == 0:
        return universal
    w = cell.n / (cell.n + shrinkage_n)
    return RegimeStats(
        n=cell.n,
        peak_q_tp=w * cell.peak_q_tp + (1 - w) * universal.peak_q_tp,
        peak_mu=w * cell.peak_mu + (1 - w) * universal.peak_mu,
        mae_q_sl=w * cell.mae_q_sl + (1 - w) * universal.mae_q_sl,
        mae_mu=w * cell.mae_mu + (1 - w) * universal.mae_mu,
        ttp_median=w * cell.ttp_median + (1 - w) * universal.ttp_median,
        ttp_mu=w * cell.ttp_mu + (1 - w) * universal.ttp_mu,
        capture_mu=w * cell.capture_mu + (1 - w) * universal.capture_mu,
    )


def _stats_to_thresholds(s: RegimeStats, ttp_factor: float) -> Dict:
    """Apply the formulas: stats → exit thresholds.

    All formulas live HERE — single source of truth.
    """
    # tp_pts: peak quantile in $, convert to pts (1pt = $2)
    tp_pts = s.peak_q_tp / 2.0
    tp_pts = float(np.clip(tp_pts, TP_MIN_PTS, TP_MAX_PTS))

    # sl_pts: |MAE| quantile in $, convert to pts
    sl_pts = s.mae_q_sl / 2.0
    sl_pts = float(np.clip(sl_pts, SL_MIN_PTS, SL_MAX_PTS))

    # time_stop: median TtP × buffer, seconds → 5s bars
    time_stop_bars = int(s.ttp_median * ttp_factor / 5.0)
    time_stop_bars = int(np.clip(time_stop_bars, TIME_STOP_MIN_BARS,
                                            TIME_STOP_MAX_BARS))

    # giveback: arm at half typical peak, keep ratio derived from capture
    gb_min = max(s.peak_mu * 0.5, 5.0)
    # Aim for capture_mu × 1.5 (lift above historical capture), bounded
    gb_keep = float(np.clip(max(s.capture_mu * 1.5, 0.3), 0.3, 0.8))

    return {
        'tp_pts': round(tp_pts, 2),
        'sl_pts': round(sl_pts, 2),
        'gb_min': round(gb_min, 2),
        'gb_keep': round(gb_keep, 3),
        'time_stop_bars': int(time_stop_bars),
    }


# ─── Builder ──────────────────────────────────────────────────────────────

def derive_thresholds(labels: Iterable[RegretLabel],
                            q_tp: float = Q_TP_DEFAULT,
                            q_sl: float = Q_SL_DEFAULT,
                            ttp_factor: float = TTP_FACTOR_DEFAULT,
                            tiers: Tuple[str, ...] = ALL_TIERS,
                            group_by: str = 'tier',
                            ) -> Dict:
    """Derive exit thresholds via Bayesian formulas.

    group_by:
        'tier'         : 3 cells = MA_ALIGN, REVERSION, VEL_BODY_CHORD
                          (all regimes share a tier's thresholds)
        'regime'       : 6 cells = regime_2d codes
                          (all tiers share a regime's thresholds)
        'tier_regime'  : 18 cells = (tier × regime)
                          (each (tier, regime) pair has its own)

    Returns dict in the same shape threshold_optimizer.py emits, so the
    engine's lookup_thresholds works unchanged.
    """
    labels = list(labels)
    if not labels:
        return {
            'cells': {}, 'tier_pools': {},
            'universal': DEFAULT_THRESHOLDS.copy(), 'meta': {'n_labels': 0},
        }

    universal_stats = _compute_stats(labels, q_tp, q_sl)
    universal_thr = _stats_to_thresholds(universal_stats, ttp_factor)

    cells: Dict[str, Dict] = {}
    cell_stats_log: Dict[str, dict] = {}
    tier_pool_thr: Dict[str, Dict] = {}

    if group_by == 'tier':
        # 3 cells. Each tier's thresholds shared across all regimes.
        by_tier: Dict[str, List[RegretLabel]] = {}
        for l in labels:
            by_tier.setdefault(str(l.entry_tier), []).append(l)
        for tier, sub in by_tier.items():
            raw = _compute_stats(sub, q_tp, q_sl)
            shrunk = _shrink(raw, universal_stats)
            thr = _stats_to_thresholds(shrunk, ttp_factor)
            tier_pool_thr[tier] = thr
            cell_stats_log[tier] = asdict(shrunk)
            # Replicate this tier's thr across every regime key seen
            for regime in {int(l.entry_regime_idx) for l in sub}:
                cells[f'{regime}|{tier}'] = dict(thr)

    elif group_by == 'regime':
        # 6 cells. All tiers in a regime share that regime's thresholds.
        by_regime: Dict[int, List[RegretLabel]] = {}
        for l in labels:
            by_regime.setdefault(int(l.entry_regime_idx), []).append(l)
        for regime, sub in by_regime.items():
            raw = _compute_stats(sub, q_tp, q_sl)
            shrunk = _shrink(raw, universal_stats)
            thr = _stats_to_thresholds(shrunk, ttp_factor)
            cell_stats_log[str(regime)] = asdict(shrunk)
            for tier in tiers:
                cells[f'{regime}|{tier}'] = dict(thr)
        for tier in tiers:
            tier_pool_thr[tier] = dict(universal_thr)

    elif group_by == 'tier_regime':
        # 18 cells. Each (regime, tier) pair has its own.
        by_pair: Dict[Tuple[int, str], List[RegretLabel]] = {}
        for l in labels:
            key = (int(l.entry_regime_idx), str(l.entry_tier))
            by_pair.setdefault(key, []).append(l)
        # Build per-tier pools first for hierarchical fallback
        by_tier_only: Dict[str, List[RegretLabel]] = {}
        for l in labels:
            by_tier_only.setdefault(str(l.entry_tier), []).append(l)
        tier_stats = {}
        for tier, sub in by_tier_only.items():
            raw = _compute_stats(sub, q_tp, q_sl)
            shrunk = _shrink(raw, universal_stats)
            tier_stats[tier] = shrunk
            tier_pool_thr[tier] = _stats_to_thresholds(shrunk, ttp_factor)
        for (regime, tier), sub in by_pair.items():
            raw = _compute_stats(sub, q_tp, q_sl)
            # Shrink toward tier pool first, which itself shrinks toward universal
            parent = tier_stats.get(tier, universal_stats)
            shrunk = _shrink(raw, parent)
            thr = _stats_to_thresholds(shrunk, ttp_factor)
            cell_stats_log[f'{regime}|{tier}'] = asdict(shrunk)
            cells[f'{regime}|{tier}'] = thr

    else:
        raise ValueError(f"group_by must be 'tier' | 'regime' | 'tier_regime'; got {group_by!r}")

    return {
        'cells': cells,
        'tier_pools': tier_pool_thr,
        'universal': universal_thr,
        'meta': {
            'n_labels': len(labels),
            'q_tp': q_tp,
            'q_sl': q_sl,
            'ttp_factor': ttp_factor,
            'group_by': group_by,
            'method': f'bayesian-{group_by}',
            'cell_stats': cell_stats_log,
            'min_n_regime': MIN_N_REGIME,
        },
    }


# ─── CLI ────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Bayesian regime-only threshold deriver')
    p.add_argument('--regret', type=str,
                       default='training_iso_v2/output/regret_is.pkl')
    p.add_argument('--out', type=str,
                       default='training_iso_v2/output/thresholds_bayesian.json')
    p.add_argument('--q-tp', type=float, default=Q_TP_DEFAULT,
                       help='Quantile of peak distribution for TP (default 0.30)')
    p.add_argument('--q-sl', type=float, default=Q_SL_DEFAULT,
                       help='Quantile of |MAE| distribution for SL (default 0.70)')
    p.add_argument('--ttp-factor', type=float, default=TTP_FACTOR_DEFAULT,
                       help='Multiplier on median TtP for time stop (default 1.5)')
    p.add_argument('--group-by', type=str, default='tier',
                       choices=['tier', 'regime', 'tier_regime'],
                       help='Cell granularity (tier=3 cells, regime=6, tier_regime=18)')
    return p.parse_args()


def main():
    args = _parse_args()
    print(f'Loading regret labels from {args.regret}...')
    with open(args.regret, 'rb') as f:
        labels = pickle.load(f)
    print(f'  {len(labels)} labels')
    print(f'Knobs: q_tp={args.q_tp}, q_sl={args.q_sl}, ttp_factor={args.ttp_factor}')
    print(f'Method: Bayesian, group_by={args.group_by} (NO grid search)')

    thr_map = derive_thresholds(labels,
                                          q_tp=args.q_tp, q_sl=args.q_sl,
                                          ttp_factor=args.ttp_factor,
                                          group_by=args.group_by)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(thr_map, f, indent=2)
    print(f'Saved -> {args.out}')

    # Per-cell summary
    print(f'\nUniversal: {thr_map["universal"]}')
    cstats = thr_map['meta']['cell_stats']
    gb = args.group_by
    if gb == 'tier':
        print(f'\nPer-tier thresholds:')
        print(f'  {"tier":<18} {"n":>6} {"peak_qTP":>9} {"|MAE|_qSL":>10} '
                  f'{"TtP_med":>8} {"cap_mu":>7} | {"tp":>6} {"sl":>6} {"ts":>5} '
                  f'{"gb_min":>7} {"gb_kp":>6}')
        for tier, thr in thr_map['tier_pools'].items():
            s = cstats.get(tier, {})
            print(f'  {tier:<18} {s.get("n", 0):>6} '
                      f'${s.get("peak_q_tp", 0):>+8.1f} ${s.get("mae_q_sl", 0):>+9.1f} '
                      f'{s.get("ttp_median", 0):>7.0f}s {s.get("capture_mu", 0):>7.3f} | '
                      f'{thr["tp_pts"]:>6.1f} {thr["sl_pts"]:>6.1f} '
                      f'{thr["time_stop_bars"]:>5d} '
                      f'{thr["gb_min"]:>7.1f} {thr["gb_keep"]:>6.2f}')
    elif gb == 'regime':
        print(f'\nPer-regime thresholds (shared across tiers in regime):')
        print(f'  {"regime":<14} {"n":>6} {"peak_qTP":>9} {"|MAE|_qSL":>10} '
                  f'{"TtP_med":>8} | {"tp":>6} {"sl":>6} {"ts":>5} '
                  f'{"gb_min":>7} {"gb_kp":>6}')
        for r in sorted(cstats.keys(), key=int):
            rint = int(r)
            rname = REGIME_VOCAB[rint] if rint < len(REGIME_VOCAB) else f'R{rint}'
            s = cstats[r]
            thr = next((thr_map['cells'][k] for k in thr_map['cells']
                              if k.startswith(f'{rint}|')), None)
            if thr is None: continue
            print(f'  {rname:<14} {s["n"]:>6} '
                      f'${s["peak_q_tp"]:>+8.1f} ${s["mae_q_sl"]:>+9.1f} '
                      f'{s["ttp_median"]:>7.0f}s | '
                      f'{thr["tp_pts"]:>6.1f} {thr["sl_pts"]:>6.1f} '
                      f'{thr["time_stop_bars"]:>5d} '
                      f'{thr["gb_min"]:>7.1f} {thr["gb_keep"]:>6.2f}')
    else:  # tier_regime
        print(f'\n(regime, tier) cells:')
        for k in sorted(cstats.keys()):
            s = cstats[k]
            thr = thr_map['cells'].get(k, {})
            print(f'  {k}: n={s["n"]}  '
                      f'tp={thr.get("tp_pts"):.1f} sl={thr.get("sl_pts"):.1f} '
                      f'ts={thr.get("time_stop_bars")} '
                      f'gb_min={thr.get("gb_min"):.1f} gb_keep={thr.get("gb_keep"):.2f}')


if __name__ == '__main__':
    main()
