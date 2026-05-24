"""Sequentially retrain the full B1-B10 GBM stack.

Fail-fast: stops on the first non-zero exit. Each `train_b*.py` is invoked as
a subprocess with its OWN default config — unless `--honest-is` is passed, in
which case the wrapper injects the per-stage flags that point each trainer at
the CAUSAL_FLAT (lookahead-free, GBM-less) IS data.

NOTE — per project policy, the USER runs training. This is a one-button
re-trigger; it does not modify any trainer's hyper-parameters.

Usage:
    python tools/retrain_b_stack.py                  # B1 -> B10, vanilla
    python tools/retrain_b_stack.py --honest-is      # B1 -> B10 on forward pass IS
    python tools/retrain_b_stack.py --from 7         # resume from B7
    python tools/retrain_b_stack.py --to 6           # B1..B6 only
    python tools/retrain_b_stack.py --only 9         # B9 alone
    python tools/retrain_b_stack.py --dry-run        # print plan; no execution

--honest-is auto-builds two derived artifacts if missing (via subprocess):
  - reports/findings/regret_oracle/zigzag_pivot_dataset_CAUSAL_IS_atr4.parquet
    (built by  tools/build_causal_truth_dataset.py)
  - reports/findings/regret_oracle/causal_flat_trade_trajectory_IS.parquet
    (built by  tools/build_trade_trajectory_dataset.py with --legs/--truth/--out
     pointing at the forward pass artifacts)

The only manual prereq is the forward pass IS legs CSV
(reports/findings/trade_outcome_table/causal_flat_zigzag_legs_IS.csv) — that
requires a heavy forward_zigzag run and is left to the user:
    python training_zigzag/forward_zigzag.py --is --flat
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Honest-IS data paths (relative to REPO). Each per-stage 'honest_args' list
# names which flag receives which path.
HONEST_PATHS = {
    'HONEST_IS_TRUTH': 'reports/findings/regret_oracle/zigzag_pivot_dataset_CAUSAL_IS_atr4.parquet',
    'HONEST_IS_TRAJ':  'reports/findings/regret_oracle/causal_flat_trade_trajectory_IS.parquet',
    'HONEST_IS_LEGS':  'reports/findings/trade_outcome_table/causal_flat_zigzag_legs_IS.csv',
}

# Auto-built prereqs for --honest-is: (output path, command-tokens). Run only
# if the output is missing. The forward pass IS legs CSV is NOT auto-built — it
# requires a heavy forward_zigzag run and is left to the user.
HONEST_BUILD_STEPS = [
    # 1. Retag the IS V2-per-1m truth with forward pass pivots.
    (HONEST_PATHS['HONEST_IS_TRUTH'],
     ['tools/build_causal_truth_dataset.py']),
    # 2. Build the forward pass trajectory (uses the new forward pass truth + forward pass legs).
    (HONEST_PATHS['HONEST_IS_TRAJ'],
     ['tools/build_trade_trajectory_dataset.py',
      '--legs', HONEST_PATHS['HONEST_IS_LEGS'],
      '--truth', HONEST_PATHS['HONEST_IS_TRUTH'],
      '--out', HONEST_PATHS['HONEST_IS_TRAJ']]),
]

# (stage number, script path, description, honest-IS args template)
# honest_args is a flat list of ['--flag', '<TOKEN>', '--flag2', '<TOKEN2>', ...]
# where <TOKEN> is a HONEST_PATHS key substituted in --honest-is mode.
STAGES = [
    (1,  'tools/train_b1_pivot_imminent.py',
         'B1  pivot-imminent classifier',
         ['--is-dataset', '<HONEST_IS_TRUTH>']),
    (2,  'tools/train_b2_fakeout.py',
         'B2  fakeout detector',
         ['--is-dataset', '<HONEST_IS_TRUTH>']),
    (3,  'tools/train_b3_ttp_regressor.py',
         'B3  time-to-pivot regressor',
         ['--is-dataset', '<HONEST_IS_TRUTH>']),
    (4,  'tools/train_b4_pivot_region.py',
         'B4  pivot-region classifier',
         ['--is-dataset', '<HONEST_IS_TRUTH>']),
    (5,  'tools/train_b5_leg_phase.py',
         'B5  leg-phase classifier',
         ['--is-dataset', '<HONEST_IS_TRUTH>']),
    (6,  'tools/train_b6_directional_pivot.py',
         'B6  directional-pivot classifier',
         ['--is-dataset', '<HONEST_IS_TRUTH>']),
    (7,  'tools/train_b7_leg_sizer.py',
         'B7  leg-amplitude regressor (entry sizing / skip)',
         ['--is-truth', '<HONEST_IS_TRUTH>']),
    (8,  'tools/train_b8_hour_risk.py',
         'B8  hour-of-day risk classifier',
         ['--is-truth', '<HONEST_IS_TRUTH>', '--is-legs', '<HONEST_IS_LEGS>']),
    (9,  'tools/train_b9_remaining_amplitude.py',
         'B9  during-trade remaining-amplitude regressor (K=5)',
         ['--traj', '<HONEST_IS_TRAJ>']),
    (10, 'tools/train_b10_vol_regime_sizer.py',
         'B10 vol-regime day-mode classifier',
         ['--is-legs', '<HONEST_IS_LEGS>']),
]


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f'{s:.0f}s'
    m, s = divmod(int(s), 60)
    if m < 60:
        return f'{m}m{s:02d}s'
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m'


def _resolve_honest(args_list: list[str]) -> list[str]:
    """Substitute <HONEST_*> tokens with concrete paths."""
    out = []
    for a in args_list:
        if a.startswith('<') and a.endswith('>'):
            key = a[1:-1]
            if key not in HONEST_PATHS:
                raise ValueError(f'unknown honest-IS token {a!r}')
            out.append(HONEST_PATHS[key])
        else:
            out.append(a)
    return out


def _ensure_honest_artifacts(dry_run: bool = False) -> bool:
    """Hard-check forward pass IS legs (user prereq) and auto-build the two derived
    artifacts if missing. Returns True if everything is ready; False on any
    hard failure (missing legs CSV, or a build subprocess failed).
    """
    legs_path = REPO / HONEST_PATHS['HONEST_IS_LEGS']
    if not legs_path.exists():
        print('--honest-is requires the forward pass IS legs CSV at:')
        print(f'  {legs_path.relative_to(REPO)}')
        print('Build it first (it requires a forward pass):')
        print('  python training_zigzag/forward_zigzag.py --is --flat')
        return False

    for out_rel, build_cmd in HONEST_BUILD_STEPS:
        out_path = REPO / out_rel
        if out_path.exists():
            continue
        bar = '=' * 78
        print(f'\n{bar}\n[honest-is prereq] BUILD missing: {out_rel}\n{bar}',
              flush=True)
        cmd_str = 'python ' + build_cmd[0] + ''.join(
            f' {a}' for a in build_cmd[1:])
        if dry_run:
            print(f'  DRY-RUN: would run: {cmd_str}')
            continue
        print(f'  $ {cmd_str}')
        cmd = [sys.executable, str(REPO / build_cmd[0])] + list(build_cmd[1:])
        t = time.perf_counter()
        rc = subprocess.call(cmd, cwd=str(REPO))
        dt = time.perf_counter() - t
        if rc != 0:
            print(f'\n!!! prereq build FAILED (exit {rc}) after {_fmt_secs(dt)}.')
            return False
        if not out_path.exists():
            print(f'\n!!! prereq build exited 0 but {out_rel} is still missing.')
            return False
        print(f'--- prereq built in {_fmt_secs(dt)} ---')
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--from', dest='from_b', type=int, default=1,
                    help='start at B<N> (inclusive, default 1)')
    ap.add_argument('--to', type=int, default=10,
                    help='stop after B<N> (inclusive, default 10)')
    ap.add_argument('--only', type=int, default=None,
                    help='run only B<N> (overrides --from/--to)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the plan; do not execute')
    ap.add_argument('--honest-is', action='store_true',
                    help='retrain on the CAUSAL_FLAT IS data (lookahead-free, no '
                         'GBMs upstream). Injects per-stage --is-dataset / '
                         '--is-truth / --is-legs / --traj flags pointing at the '
                         'forward pass artifacts. Preflight-checks the artifacts exist.')
    ap.add_argument('--extra-args', default='',
                    help='string of args to pass through to every trainer in '
                         'addition to per-stage honest-IS flags (quote it).')
    args = ap.parse_args()

    if args.only is not None:
        stages = [s for s in STAGES if s[0] == args.only]
    else:
        stages = [s for s in STAGES if args.from_b <= s[0] <= args.to]
    if not stages:
        print('No stages selected.')
        return 0

    extra = args.extra_args.split() if args.extra_args else []

    # ── Preflight ─────────────────────────────────────────────────────
    missing_scripts = [p for _, p, _, _ in stages if not (REPO / p).exists()]
    if missing_scripts:
        print(f'Missing {len(missing_scripts)} trainer script(s):')
        for p in missing_scripts:
            print(f'  - {p}')
        return 2

    if args.honest_is:
        if not _ensure_honest_artifacts(dry_run=args.dry_run):
            return 3

    # ── Print plan ────────────────────────────────────────────────────
    mode = 'HONEST-IS (causal_flat)' if args.honest_is else 'vanilla (hardened defaults)'
    print(f'Plan: {len(stages)} stage(s)   mode = {mode}'
          + (f'   extra-args = {extra}' if extra else '')
          + ('   DRY-RUN' if args.dry_run else ''))
    for n, path, desc, honest_args in stages:
        injected = _resolve_honest(honest_args) if args.honest_is else []
        suffix = f'   inject: {" ".join(injected)}' if injected else ''
        print(f'   B{n:<2}  {path:<46}  {desc}{suffix}')

    if args.dry_run:
        return 0

    # ── Execute ───────────────────────────────────────────────────────
    print()
    t0 = time.perf_counter()
    results: list[tuple[int, float, int]] = []  # (n, secs, rc)
    for n, path, desc, honest_args in stages:
        bar = '=' * 78
        print(f'\n{bar}\n=== B{n}: {desc}\n=== {path}\n{bar}', flush=True)
        cmd = [sys.executable, str(REPO / path)]
        if args.honest_is:
            cmd.extend(_resolve_honest(honest_args))
        cmd.extend(extra)
        t = time.perf_counter()
        rc = subprocess.call(cmd, cwd=str(REPO))
        dt = time.perf_counter() - t
        results.append((n, dt, rc))
        if rc != 0:
            print(f'\n!!! B{n} FAILED (exit {rc}) after {_fmt_secs(dt)}. '
                  'Stopping the chain.')
            _final(results, time.perf_counter() - t0, failed=True)
            return rc
        print(f'\n--- B{n} done in {_fmt_secs(dt)} ---')

    _final(results, time.perf_counter() - t0, failed=False)
    return 0


def _final(results, total_secs, failed: bool):
    print('\n' + '=' * 78)
    print('SUMMARY')
    print('=' * 78)
    for n, dt, rc in results:
        status = 'OK ' if rc == 0 else f'FAIL({rc})'
        print(f'  B{n:<2}   {status}    {_fmt_secs(dt):>8}')
    print('-' * 78)
    label = 'PARTIAL (failed)' if failed else 'COMPLETE'
    print(f'  TOTAL  {label:<10}  {_fmt_secs(total_secs):>8}')


if __name__ == '__main__':
    sys.exit(main())
