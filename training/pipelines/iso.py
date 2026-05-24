"""End-to-end ISO pipeline orchestrator.

Runs the full V2-native iso research stack as a sequence of phases:

    1. Per-regime SEED calibration  (z_thr, r_thr per regime)
       -> training_iso_v2/output/seed_thresholds_per_regime.json
    2. Per-tier filter calibration  (velocity / wick quantiles)
       -> training_iso_v2/output/tier_thresholds.json
    3. ISO IS forward pass          (with regret, per-regime seed, tier filters)
       -> training_iso_v2/output/is_<TIER>(_regret).pkl
    4. Peak-signature mining on IS  (feature coalescence at MFE bars)
       -> reports/findings/peak_signatures/{trade_mfe_features.parquet,
                                                          per_cell_feature_signatures.csv,
                                                          top_features_per_cell.json}
    5. ISO OOS forward pass
       -> training_iso_v2/output/oos_<TIER>(_regret).pkl
    6. Peak-signature mining on OOS (validation: IS signatures should hold)
       -> reports/findings/peak_signatures_oos/...

Phases 1, 2 cache their JSON outputs and skip on rerun unless --fresh.
Phases 3-6 always rerun (research outputs change with each iso run).

Usage:
    python -m training.pipelines.iso                  # full pipeline (cached calibrations)
    python -m training.pipelines.iso --fresh          # rebuild all calibrations
    python -m training.pipelines.iso --from 3         # skip calibration, just run iso + analysis
    python -m training.pipelines.iso --is-only        # phases 1-4
    python -m training.pipelines.iso --oos-only       # phases 1-2 + 5-6

Per-phase tunables:
    --n-days-seed N   sample N IS days for seed calibration (default 80)
    --n-days-tier N   sample N IS days for tier calibration  (default 30)
    --no-regret       skip regret analysis in iso runs (faster)
    --skip-peak-mine  skip phases 4 & 6 (peak signature mining)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = 'training_iso_v2/output'
HOURLY_REGIME_CSV = 'DATA/ATLAS/regime_labels_hourly.csv'
VEL_REGIME_THR_JSON = f'{OUTPUT_DIR}/velocity_regime_thresholds.json'
SEED_JSON = f'{OUTPUT_DIR}/seed_thresholds_per_regime.json'
TIER_JSON = f'{OUTPUT_DIR}/tier_thresholds.json'

PEAK_MINE_DIR_IS = 'reports/findings/peak_signatures'
PEAK_MINE_DIR_OOS = 'reports/findings/peak_signatures_oos'
PEAK_TRAJ_DIR_IS = 'reports/findings/peak_trajectory'
PEAK_TRAJ_DIR_OOS = 'reports/findings/peak_trajectory_oos'
MFE_TARGETS_JSON = f'{OUTPUT_DIR}/mfe_targets_per_cell.json'


def run_phase(label: str, cmd: list, cache_path: str = None, fresh: bool = False):
    """Run a phase via subprocess. Skip if cache_path exists and not fresh.

    Hardens against silent failures: if cache_path is set, verifies the file
    exists AFTER the run. A subprocess that returns 0 but produces no output
    file is treated as a hard failure (the next phase would crash on a
    missing file otherwise).
    """
    if cache_path and Path(cache_path).exists() and not fresh:
        print(f'\n{">" * 6}  [{label}]  CACHED ({cache_path}); skipping. '
                  f'Use --fresh to regenerate.')
        return
    print(f'\n{">" * 6}  [{label}]')
    print(f'        cmd: {" ".join(cmd)}')
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f'\n!!! [{label}] failed with exit code {rc}')
        sys.exit(rc)
    if cache_path and not Path(cache_path).exists():
        print(f'\n!!! [{label}] returned 0 but expected output {cache_path} '
                  f'was not created. Treating as failure.')
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description='V2-native ISO pipeline orchestrator')
    ap.add_argument('--fresh', action='store_true',
                          help='Regenerate ALL calibrations even if cached')
    ap.add_argument('--from', type=int, default=0, dest='start_phase',
                          help='First phase to run (0-7)')
    ap.add_argument('--to', type=int, default=7, dest='end_phase',
                          help='Last phase to run (0-7)')
    ap.add_argument('--is-only', action='store_true',
                          help='Run phases 1-4 only (skip OOS)')
    ap.add_argument('--oos-only', action='store_true',
                          help='Run calibrations + OOS only (skip IS)')
    ap.add_argument('--n-days-seed', type=int, default=80,
                          help='IS days sampled for seed calibration')
    ap.add_argument('--n-days-tier', type=int, default=30,
                          help='IS days sampled for tier-filter calibration')
    ap.add_argument('--no-regret', action='store_true',
                          help='Skip regret analysis in iso runs')
    ap.add_argument('--skip-peak-mine', action='store_true',
                          help='Skip phases 4 & 6 (peak signature mining)')
    args = ap.parse_args()

    if args.is_only and args.oos_only:
        print('--is-only and --oos-only are mutually exclusive')
        sys.exit(2)

    py = sys.executable
    in_range = lambda n: args.start_phase <= n <= args.end_phase

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Phase 0: VELOCITY-regime thresholds (forward-pass-honest regime) ──
    # Calibrates vel_thr + sn_thr from IS feature distributions. Ticker
    # reads these per-bar to compute regime from L2 velocity + L3 swing_noise
    # — already-rolling features, so by construction past-only data.
    if 0 >= args.start_phase and 0 <= args.end_phase:
        run_phase(
            'Phase 0: VELOCITY-regime thresholds (per-bar, no lookahead)',
            [py, 'tools/calibrate_velocity_regime.py',
                  '--out', VEL_REGIME_THR_JSON],
            cache_path=VEL_REGIME_THR_JSON, fresh=args.fresh,
        )

    # ── Phase 1: per-regime SEED calibration ────────────────────────────
    if in_range(1):
        run_phase(
            'Phase 1: per-regime SEED calibration (z_thr, r_thr)',
            [py, 'tools/calibrate_seed_per_regime.py',
                  '--n-days', str(args.n_days_seed),
                  '--out-json', SEED_JSON],
            cache_path=SEED_JSON, fresh=args.fresh,
        )

    # ── Phase 2: per-tier filter calibration ────────────────────────────
    if in_range(2):
        run_phase(
            'Phase 2: tier filter calibration (velocity / wick quantiles)',
            [py, '-m', 'training.calibrate_tiers',
                  '--n-days', str(args.n_days_tier),
                  '--out', TIER_JSON],
            cache_path=TIER_JSON, fresh=args.fresh,
        )

    # ── Iso common args (used by both IS and OOS runs) ──────────────────
    # MFE targets only included if the JSON exists (it's built by Phase 4d
    # from a prior IS pass, then used by subsequent runs).
    iso_base = [py, '-m', 'training.pipelines.v2_native_iso',
                       '--calibrated', TIER_JSON,
                       '--seed-per-regime', SEED_JSON]
    if Path(MFE_TARGETS_JSON).exists():
        iso_base += ['--mfe-targets', MFE_TARGETS_JSON]
    if args.no_regret:
        iso_base.append('--no-regret')

    # ── Phase 3: ISO IS forward pass ────────────────────────────────────
    if in_range(3) and not args.oos_only:
        run_phase(
            'Phase 3: ISO IS forward pass',
            iso_base + ['--is'],
            cache_path=None, fresh=False,
        )

    # ── Phase 4: peak signature mining on IS ────────────────────────────
    if in_range(4) and not args.oos_only and not args.skip_peak_mine:
        run_phase(
            'Phase 4: peak signature mining on IS pickles',
            [py, 'tools/peak_signature_mining.py',
                  '--prefix', f'{OUTPUT_DIR}/is',
                  '--out-dir', PEAK_MINE_DIR_IS],
            cache_path=None, fresh=False,
        )
        # 4b: re-rank by movement signature (filter selection-bias features)
        run_phase(
            'Phase 4b: peak-signature re-rank (movement-weighted)',
            [py, 'tools/peak_signature_rerank.py',
                  '--parquet', f'{PEAK_MINE_DIR_IS}/trade_mfe_features.parquet',
                  '--out-dir', PEAK_MINE_DIR_IS],
            cache_path=None, fresh=False,
        )
        # 4c: trajectory mining around MFE — categorize features as
        #     forming / peaked / decaying / smooth across offsets
        run_phase(
            'Phase 4c: peak TRAJECTORY mining (lead/lag exit signatures)',
            [py, 'tools/peak_trajectory_mining.py',
                  '--prefix', f'{OUTPUT_DIR}/is',
                  '--out-dir', PEAK_TRAJ_DIR_IS],
            cache_path=None, fresh=False,
        )
        # 4d: extract per-cell modal MFE PnL ($) → mfe_targets_per_cell.json
        # Powers the MFEPriceTarget exit (price-target gate). Engine reads
        # this JSON via --mfe-targets (auto-included on subsequent runs).
        run_phase(
            'Phase 4d: extract per-cell MFE price targets ($)',
            [py, 'tools/extract_mfe_targets.py',
                  '--parquet', f'{PEAK_MINE_DIR_IS}/trade_mfe_features.parquet',
                  '--out', MFE_TARGETS_JSON],
            cache_path=None, fresh=False,
        )

    # ── Phase 5: ISO OOS forward pass ───────────────────────────────────
    if in_range(5) and not args.is_only:
        run_phase(
            'Phase 5: ISO OOS forward pass',
            iso_base + ['--oos'],
            cache_path=None, fresh=False,
        )

    # ── Phase 6: peak signature mining on OOS ───────────────────────────
    if in_range(6) and not args.is_only and not args.skip_peak_mine:
        run_phase(
            'Phase 6: peak signature mining on OOS pickles (IS-vs-OOS validation)',
            [py, 'tools/peak_signature_mining.py',
                  '--prefix', f'{OUTPUT_DIR}/oos',
                  '--out-dir', PEAK_MINE_DIR_OOS],
            cache_path=None, fresh=False,
        )
        run_phase(
            'Phase 6b: OOS peak-signature re-rank (movement-weighted)',
            [py, 'tools/peak_signature_rerank.py',
                  '--parquet', f'{PEAK_MINE_DIR_OOS}/trade_mfe_features.parquet',
                  '--out-dir', PEAK_MINE_DIR_OOS],
            cache_path=None, fresh=False,
        )
        run_phase(
            'Phase 6c: OOS peak TRAJECTORY mining',
            [py, 'tools/peak_trajectory_mining.py',
                  '--prefix', f'{OUTPUT_DIR}/oos',
                  '--out-dir', PEAK_TRAJ_DIR_OOS],
            cache_path=None, fresh=False,
        )

    # ── Phase 7: best/worst day visualization (IS + OOS) ────────────────
    if 7 >= args.start_phase and 7 <= args.end_phase:
        if not args.oos_only:
            run_phase(
                'Phase 7a: best/worst day plots (IS)',
                [py, 'tools/iso_best_worst_days.py',
                      '--prefix', f'{OUTPUT_DIR}/is',
                      '--out-dir', 'reports/findings/iso_best_worst_days_is'],
                cache_path=None, fresh=False,
            )
        if not args.is_only:
            run_phase(
                'Phase 7b: best/worst day plots (OOS)',
                [py, 'tools/iso_best_worst_days.py',
                      '--prefix', f'{OUTPUT_DIR}/oos',
                      '--out-dir', 'reports/findings/iso_best_worst_days_oos'],
                cache_path=None, fresh=False,
            )

    print(f'\n{">" * 6}  PIPELINE COMPLETE')
    print(f'        seed thresholds : {SEED_JSON}')
    print(f'        tier thresholds : {TIER_JSON}')
    print(f'        iso pickles     : {OUTPUT_DIR}/{{is,oos}}_<TIER>.pkl')
    print(f'        peak signatures : {PEAK_MINE_DIR_IS}/  +  {PEAK_MINE_DIR_OOS}/')


if __name__ == '__main__':
    main()
