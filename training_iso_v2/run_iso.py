"""ISO forward pass — runs the 9 legacy V2-ported tiers in parallel,
each in its own engine. Each tier produces an independent trade list.

Usage:
    python -m training_iso_v2.run_iso --is
    python -m training_iso_v2.run_iso --oos
    python -m training_iso_v2.run_iso --is --thresholds path/to/thresholds.json
    python -m training_iso_v2.run_iso --tiers FADE_CALM,KILL_SHOT,CASCADE
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import List

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_iso_v2.ticker import V2Ticker, MultiDayV2Ticker
from training_iso_v2.exits import default_exit_suite
from training_iso_v2.iso_orchestrator import IsoOrchestrator
from training_iso_v2.v2_cols import swing_noise_w, reversion_prob_w, z_se_w
from training_iso_v2.regret_full import (label_trades_full, FullRegretLabel,
                                                          ACTIONS)
from training_iso_v2.strategies import (
    FadeCalm, FadeMomentum, RideCalm, RideMomentum,
    FadeAgainst, RideAgainst, KillShot, Cascade, FreightTrain,
    NMPFadeRaw, NMPRideRaw,
    FadeAtBand,
    MAAlignTrendFollow, CompressionBounceLong, CatHarvestRide,
    CrmCuspFade,
)
from training_iso_v2.strategies._nmp_base import NMPBaseStrategy


OUTPUT_DIR = 'training_iso_v2/output'

ALL_ISO_TIERS = {
    # ── 2026-05-10 PRIORITY SHIFT: TREND-FOLLOWING tiers (PRIMARY) ──────
    'MA_ALIGN':                MAAlignTrendFollow,    # 7-of-8 vwap → 70.5% IS
    'COMPRESSION_BOUNCE_LONG': CompressionBounceLong, # 15m vol-sigma crush → LONG bounce (P_oos=0.64)
    'CAT_HARVEST':             CatHarvestRide,        # Tue/Wed/Thu UTC 1 → SHORT (96% crash bias)
    # ── 2026-05-10 CRM cusp — |z| local-max fade (cusp_research validation) ─
    'CRM_CUSP_FADE':           CrmCuspFade,           # cusp at z in [1.5,1.8) → OOS +$0.28-0.82/t
    # ── Fade tiers (retuned 2026-05-10 — secondary edge) ─────────────────
    'FADE_CALM': FadeCalm,
    'FADE_MOMENTUM': FadeMomentum,
    'RIDE_CALM': RideCalm,
    'RIDE_MOMENTUM': RideMomentum,
    'FADE_AGAINST': FadeAgainst,
    'RIDE_AGAINST': RideAgainst,
    'KILL_SHOT': KillShot,
    'CASCADE': Cascade,
    'FREIGHT_TRAIN': FreightTrain,
    # Diagnostic baselines (no velocity/wick filter) to test filter contribution
    'NMP_FADE_RAW': NMPFadeRaw,
    'NMP_RIDE_RAW': NMPRideRaw,
    # Band-fade: 5s price hits 15m ±2σ → fade to 5m mean (chart-validated framework)
    'FADE_AT_BAND': FadeAtBand,
}


def _entry_extras(state):
    """Capture entry-time signal values exits depend on.

    OUReversionDecay reads `entry_reversion_prob` to detect when the
    OU mean-reversion thesis weakens during the trade.
    SwingNoiseSpike reads `entry_swing_noise` to detect chop expansion.
    ZSeRetracement reads `entry_z_se` to detect retracement to peak.
    """
    return {
        'entry_swing_noise': state.get(swing_noise_w('1m')),
        'entry_reversion_prob': state.get(reversion_prob_w('1m')),
        'entry_z_se': state.get(z_se_w('1m')),
    }


def _histogram_mode(values, bin_width: float = 2.0) -> float:
    """Mode of a per-trade $ distribution (CLAUDE.md: bin width $2 for $/trade)."""
    import numpy as np
    v = np.asarray([x for x in values if x == x], dtype=np.float64)
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


def _pf_trade_wr(pnls) -> float:
    """PF-based Trade WR per CLAUDE.md: (sum_profits / |sum_losses|) - 1.

    0 = break-even (gross profit equals gross loss)
    +1 = winners 2x loser size (PF=2)
    -0.5 = winners only half loser size (PF=0.5)
    Returns NaN if no losses (avoid /0); inf if PF infinite.
    """
    import math
    p = sum(x for x in pnls if x > 0)
    l = sum(x for x in pnls if x < 0)
    if l == 0:
        return float('inf') if p > 0 else 0.0
    pf = p / abs(l)
    return pf - 1.0


def _resolve_days(target='is', start=None, end=None, days_csv=None):
    if days_csv:
        return [d.strip().replace('-', '_') for d in days_csv.split(',')]
    import glob
    l0_dir = 'DATA/ATLAS/FEATURES_5s_v2/L0'
    files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in files]
    if start:
        days = [d for d in days if d.replace('_', '-') >= start]
    if end:
        days = [d for d in days if d.replace('_', '-') <= end]
    if target == 'is':
        return [d for d in days if d.startswith('2025_')]
    if target == 'oos':
        return [d for d in days if d.startswith('2026_')]
    return days


def _build_strategies(tier_names: List[str],
                              calibrated_thresholds: dict = None,
                              seed_per_regime: dict = None):
    """Build tier strategies. If calibrated_thresholds is provided, uses
    per-tier kwargs from the JSON; else uses hardcoded defaults.

    seed_per_regime: optional dict {regime_label: {z_thr, r_thr}} threaded
    into each NMP-based strategy via the per_regime kwarg, so the seed
    fires at regime-specific (z, r) thresholds instead of universal 1.8/0.55.
    """
    # Strip _meta from per-regime map; only regime keys remain
    per_regime_clean = None
    if seed_per_regime:
        per_regime_clean = {k: v for k, v in seed_per_regime.items()
                                  if not k.startswith('_')}

    out = []
    for n in tier_names:
        cls = ALL_ISO_TIERS.get(n)
        if cls is None:
            raise ValueError(f'Unknown tier: {n}. Choices: {list(ALL_ISO_TIERS)}')
        kwargs = {}
        if calibrated_thresholds and n in calibrated_thresholds:
            kwargs = {k: v for k, v in calibrated_thresholds[n].items()
                            if not k.startswith('_')}
        # Only NMP-derived tiers consume per_regime; FreightTrain has its
        # own gate (velocity + swing_noise + hurst) so it ignores the seed map.
        if per_regime_clean is not None and issubclass(cls, NMPBaseStrategy):
            kwargs['per_regime'] = per_regime_clean
        out.append(cls(**kwargs))
    return out


def _per_tier_regret_summary(trades: List, n_days: int) -> dict:
    """Run regret_full on a tier's trades; return summary stats for table.
    Returns {} when trades is empty (no labels to compute)."""
    if not trades:
        return {}
    labels = label_trades_full(trades, store_curves=False)
    if not labels:
        return {}
    import numpy as np
    regret_vals = np.array([l.regret for l in labels], dtype=np.float64)
    actions = [l.best_action for l in labels]
    n = len(labels)
    pct = lambda a: actions.count(a) / max(n, 1) * 100
    return {
        'mode_regret': _histogram_mode(regret_vals, bin_width=10.0),
        'mean_regret': float(regret_vals.mean()),
        'pct_same_extended': pct('same_extended'),
        'pct_counter_extended': pct('counter_extended'),
        'pct_same_early': pct('same_early'),
        'pct_counter_early': pct('counter_early'),
        'labels': labels,  # caller may save these
    }


def run_target(target: str, days_csv: str = None, start: str = None,
                  end: str = None, tier_names: List[str] = None,
                  threshold_map: dict = None, out_pkl_prefix: str = None,
                  smoke: bool = False,
                  calibrated_thresholds: dict = None,
                  seed_per_regime: dict = None,
                  mfe_targets: dict = None,
                  run_regret: bool = True):
    if smoke:
        days = ['2025_06_15']
    else:
        days = _resolve_days(target, start, end, days_csv)
    if not days:
        print(f'No days resolved')
        return {}

    if tier_names is None:
        tier_names = list(ALL_ISO_TIERS.keys())
    strategies = _build_strategies(tier_names, calibrated_thresholds,
                                                seed_per_regime=seed_per_regime)
    cal_str = ' (calibrated)' if calibrated_thresholds else ' (defaults)'
    seed_str = ' (per-regime seed)' if seed_per_regime else ''
    print(f'Days: {len(days)}, Tiers: {tier_names}{cal_str}{seed_str}')

    orch = IsoOrchestrator(
        strategies=strategies,
        exits=default_exit_suite(mfe_targets=mfe_targets),
        threshold_map=threshold_map,
        entry_extras_hook=_entry_extras,
    )

    multi = MultiDayV2Ticker(days=days)
    per_tier = orch.run(tqdm(multi, desc=f'iso {target}', total=10000))

    print(f'\n{"="*120}')
    print(f'ISO results [{target}]   {len(days)} days')
    print(f'  $/trade: MODE (histogram bin=$2) per CLAUDE.md')
    print(f'  WR     : PF-based Trade WR = (sum_profits / |sum_losses|) - 1; 0 = break-even')
    if run_regret:
        print(f'  regret : multi-axis counterfactual; mode_reg + best_action mix '
                  f'(sm_x = same_extended %, ct_x = counter_extended %)')
    print(f'{"="*120}')
    header = (f'{"tier":<16} {"n":>6} {"$_total":>10} {"mode/t":>8} {"mean/t":>8} '
                  f'{"$/day":>9} {"PF-WR":>7} {"day-WR":>7}')
    if run_regret:
        header += f' | {"mode_reg":>9} {"%sm_x":>6} {"%ct_x":>6}'
    print(header)
    n_days = len(days)
    total_all = 0.0
    regret_by_tier = {}  # for saving alongside trades
    for name, trades in per_tier.items():
        n = len(trades)
        if n == 0:
            print(f'{name:<16} {n:>6} {"":>10} {"":>8} {"":>8} {"":>9} {"":>7} {"":>7}')
            continue
        pnls = [t.pnl for t in trades]
        total = sum(pnls)
        mean_per_trade = total / n
        mode_per_trade = _histogram_mode(pnls, bin_width=2.0)
        per_day = total / max(n_days, 1)
        pf_wr = _pf_trade_wr(pnls)
        # Day WR — count-based (winning days / total active days)
        from collections import defaultdict
        day_pnl = defaultdict(float)
        for t in trades:
            day_pnl[t.entry_day] += t.pnl
        active_days = len(day_pnl)
        winning_days = sum(1 for d, p in day_pnl.items() if p > 0)
        day_wr = winning_days / max(active_days, 1)
        total_all += total
        pf_str = f'{pf_wr:>+6.2f}' if pf_wr != float('inf') else '   inf'
        line = (f'{name:<16} {n:>6} ${total:>+9.0f} ${mode_per_trade:>+7.2f} '
                    f'${mean_per_trade:>+7.2f} ${per_day:>+8.2f} {pf_str} '
                    f'{day_wr:>6.1%}')
        if run_regret:
            print(f'{line}', end='', flush=True)
            print(f' | (computing regret...)', end='\r', flush=True)
            r = _per_tier_regret_summary(trades, n_days)
            if r:
                regret_by_tier[name] = r['labels']
                line += (f' | ${r["mode_regret"]:>+8.2f} '
                            f'{r["pct_same_extended"]:>5.1f}% '
                            f'{r["pct_counter_extended"]:>5.1f}%')
            else:
                line += ' | (no regret data)'
        print(f'\r{line}')
    print(f'{"-"*120}')
    print(f'{"TOTAL (sum)":<16} {sum(len(t) for t in per_tier.values()):>6} '
              f'${total_all:>+9.0f}                          ${total_all/max(n_days,1):>+8.2f}')

    if out_pkl_prefix:
        os.makedirs(os.path.dirname(out_pkl_prefix) or '.', exist_ok=True)
        for name, trades in per_tier.items():
            with open(f'{out_pkl_prefix}_{name}.pkl', 'wb') as f:
                pickle.dump(trades, f)
        # Save regret labels alongside if computed
        if run_regret and regret_by_tier:
            for name, labels in regret_by_tier.items():
                with open(f'{out_pkl_prefix}_{name}_regret.pkl', 'wb') as f:
                    pickle.dump(labels, f)
            print(f'\nPer-tier pickles + regret labels saved with prefix '
                      f'{out_pkl_prefix}_<TIER>(_regret).pkl')
        else:
            print(f'\nPer-tier pickles saved with prefix {out_pkl_prefix}_<TIER>.pkl')
    return per_tier


def parse_args():
    p = argparse.ArgumentParser(description='V2-native iso forward pass')
    p.add_argument('--is', dest='run_is', action='store_true')
    p.add_argument('--oos', action='store_true')
    p.add_argument('--smoke', action='store_true', help='single test day')
    p.add_argument('--days', type=str, default=None,
                       help='comma-sep day list (overrides is/oos)')
    p.add_argument('--start', type=str, default=None)
    p.add_argument('--end', type=str, default=None)
    p.add_argument('--tiers', type=str, default=None,
                       help='comma-sep subset of: ' + ','.join(ALL_ISO_TIERS.keys()))
    p.add_argument('--thresholds', type=str, default=None,
                       help='Path to exit-thresholds JSON (per-tier exit policy)')
    p.add_argument('--calibrated', type=str, nargs='?',
                       const='training_iso_v2/output/tier_thresholds.json',
                       default=None,
                       help='Use calibrated tier ENTRY thresholds. Defaults to '
                              'training_iso_v2/output/tier_thresholds.json if path omitted.')
    p.add_argument('--no-regret', action='store_true',
                       help='Skip regret analysis (faster; use for quick smoke).')
    p.add_argument('--seed-per-regime', type=str, nargs='?',
                       const='training_iso_v2/output/seed_thresholds_per_regime.json',
                       default=None,
                       help='Use per-regime NMP seed thresholds (z_thr, r_thr by regime). '
                              'Defaults to training_iso_v2/output/seed_thresholds_per_regime.json '
                              'if path omitted.')
    p.add_argument('--mfe-targets', type=str, nargs='?',
                       const='training_iso_v2/output/mfe_targets_per_cell.json',
                       default=None,
                       help='Per-cell MFE price targets (modal peak PnL by cell). '
                              'Powers MFEPriceTarget exit. Defaults to '
                              'training_iso_v2/output/mfe_targets_per_cell.json.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    threshold_map = None
    if args.thresholds:
        with open(args.thresholds, 'r') as f:
            threshold_map = json.load(f)
        print(f'Loaded exit thresholds: {args.thresholds}')

    calibrated_thresholds = None
    if args.calibrated:
        with open(args.calibrated, 'r') as f:
            calibrated_thresholds = json.load(f)
        print(f'Loaded ENTRY calibrated thresholds: {args.calibrated}')

    seed_per_regime = None
    if args.seed_per_regime:
        with open(args.seed_per_regime, 'r') as f:
            seed_per_regime = json.load(f)
        print(f'Loaded PER-REGIME seed thresholds: {args.seed_per_regime}')

    mfe_targets = None
    if args.mfe_targets and os.path.exists(args.mfe_targets):
        with open(args.mfe_targets, 'r') as f:
            mfe_targets = json.load(f)
        print(f'Loaded PER-CELL MFE targets: {args.mfe_targets}')
    elif args.mfe_targets:
        print(f'(MFE targets path {args.mfe_targets} does not exist yet — '
                  f'price-target exit will be inactive this run; rerun after '
                  f'extract_mfe_targets.py builds it.)')

    tier_names = None
    if args.tiers:
        tier_names = [t.strip() for t in args.tiers.split(',') if t.strip()]

    common = dict(tier_names=tier_names, threshold_map=threshold_map,
                            calibrated_thresholds=calibrated_thresholds,
                            seed_per_regime=seed_per_regime,
                            mfe_targets=mfe_targets,
                            run_regret=not args.no_regret)

    if args.smoke:
        run_target('is', smoke=True,
                       out_pkl_prefix=os.path.join(OUTPUT_DIR, 'smoke'), **common)
        return

    if args.days or args.start or args.end:
        run_target('is', days_csv=args.days, start=args.start, end=args.end,
                       out_pkl_prefix=os.path.join(OUTPUT_DIR, 'custom'), **common)
        return

    if args.run_is or (not args.oos):
        run_target('is',
                       out_pkl_prefix=os.path.join(OUTPUT_DIR, 'is'), **common)
    if args.oos:
        run_target('oos',
                       out_pkl_prefix=os.path.join(OUTPUT_DIR, 'oos'), **common)


if __name__ == '__main__':
    main()
