"""ATR-multiplier retrain sweep -- fully independent harness.

Retrains the pivot-coupled B-model stack (B1-B9) at each of several zigzag ATR
multipliers, each into its own folder, so a calibration can be hot-swapped as
the volatility regime shifts.

INDEPENDENCE GUARANTEE -- this harness modifies NO existing file:
  * All output goes under --root (default: checkpoints_atr_sweep/, which the
    existing `checkpoints_*/` .gitignore rule already excludes).
  * build_is_hardened_legs.py and train_b7_leg_sizer.py hold the ATR multiplier
    as a module constant with no CLI override. The harness does NOT edit them --
    per multiplier it writes a one-line-patched COPY into <root>/_patched/ and
    runs the copy. Originals stay byte-identical (verified by a line diff).
  * The other 11 scripts are the originals, invoked via CLI with every output
    flag redirected into the sweep folder -- no production path is ever written.
  * B10 is ATR-independent: not retrained; the production b10_vol_regime_*.pkl
    are copied (read-only) into each folder so each folder is a complete stack.

This harness only ORCHESTRATES. Run it yourself when ready -- it is a multi-hour
job (all sklearn GBM, CPU). Resumable: per-X `_done` markers + --from/--to.

    python tools/atr_sweep_retrain.py --dry-run         # print the plan, run nothing
    python tools/atr_sweep_retrain.py --multipliers 4.0 # single self-check pass
    python tools/atr_sweep_retrain.py --from 1 --to 3    # a sub-range
    python tools/atr_sweep_retrain.py                   # full sweep (overnight)

NOTE: this sweep only BUILDS per-multiplier candidates. Selecting which X to
deploy needs a separate CAUSAL streaming forward pass -- the FLAT $/day metric
is oracle-contaminated across ATR. Out of scope here.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / 'tools'
PROD = REPO / 'reports' / 'findings' / 'regret_oracle'   # production models (read-only)

DEFAULT_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]

# trap scripts: original filename -> the module constant that holds the ATR mult
TRAP_SCRIPTS = {
    'build_is_hardened_legs.py': 'ATR_MULT',
    'train_b7_leg_sizer.py': 'TRAIN_ATR_MULT',
}

# every pipeline script the sweep invokes (precheck must find them all)
PIPELINE_SCRIPTS = [
    'build_zigzag_pivot_dataset.py', 'build_is_hardened_legs.py',
    'build_trade_trajectory_dataset.py', 'train_b1_pivot_imminent.py',
    'train_b2_fakeout.py', 'train_b3_ttp_regressor.py',
    'train_b4_pivot_region.py', 'train_b5_leg_phase.py',
    'train_b6_directional_pivot.py', 'train_b7_leg_sizer.py',
    'train_b8_hour_risk.py', 'train_b9_remaining_amplitude.py',
]


def suffix(x: float) -> str:
    """2.5 -> 'atr_02p5', 10.0 -> 'atr_10p0' (zero-padded so folders sort right).
    Names to one decimal place; multipliers must have <= 1 decimal."""
    return f'atr_{x:04.1f}'.replace('.', 'p')


def patch_trap(name: str, const: str, x: float, dest_dir: Path) -> Path:
    """Write a copy of tools/<name> with `<const> = 4.0` rewritten to `= x`.
    The original is never touched. Aborts loudly unless exactly one line changes
    (i.e. the script's format drifted and the harness needs updating)."""
    orig = TOOLS / name
    src = orig.read_text(encoding='utf-8')
    new, n = re.subn(rf'(?m)^{const} = 4\.0[ \t]*$', f'{const} = {x}', src)
    if n != 1:
        raise RuntimeError(
            f'patch_trap: expected exactly one `{const} = 4.0` line in {name}, '
            f'found {n}. The script changed -- update this harness.')
    diffs = [(a, b) for a, b in zip(src.splitlines(), new.splitlines()) if a != b]
    if len(diffs) > 1 or (diffs and not diffs[0][0].startswith(const)):
        raise RuntimeError(f'patch_trap: unexpected diff in {name}: {diffs}')
    dest = dest_dir / f'{suffix(x)}__{name}'
    dest.write_text(new, encoding='utf-8')
    return dest


def build_chain(x: float, D: Path, patched: dict) -> list:
    """Ordered list of (label, argv) for one multiplier. All argv items are str.
    14 subprocess steps; b10_copy (step 15) is handled separately in do_pass."""
    py = sys.executable
    xs = str(x)
    P_IS = str(D / 'pivot_dataset_IS.parquet')
    P_OOS = str(D / 'pivot_dataset_OOS.parquet')
    L_IS = str(D / 'hardened_legs_IS.csv')
    L_OOS = str(D / 'hardened_legs_OOS.csv')
    TRAJ = str(D / 'trajectory_IS.parquet')
    B7_IS = str(D / 'b7_leg_sizer_IS.parquet')    # B7 IS leg cache -> B8 input
    B7_OOS = str(D / 'b7_leg_sizer_OOS.parquet')  # B7 OOS leg cache -> B8 input
    legs_py = str(patched['build_is_hardened_legs.py'])
    b7_py = str(patched['train_b7_leg_sizer.py'])

    def t(n):  # an unmodified original tool
        return str(TOOLS / n)

    chain = [
        ('pivot_IS', [py, t('build_zigzag_pivot_dataset.py'),
            '--target', 'is', '--atr-mult', xs,
            '--root', 'DATA/ATLAS', '--out', P_IS]),
        ('pivot_OOS', [py, t('build_zigzag_pivot_dataset.py'),
            '--target', 'oos', '--atr-mult', xs,
            '--root', 'DATA/ATLAS_NT8', '--out', P_OOS]),
        ('legs_IS', [py, legs_py,
            '--truth', P_IS,
            '--bars1m-dir', 'DATA/ATLAS/1m', '--bars5s-dir', 'DATA/ATLAS/5s',
            '--out', L_IS, '--report', str(D / 'hardened_legs_IS.txt')]),
        ('legs_OOS', [py, legs_py,
            '--truth', P_OOS,
            '--bars1m-dir', 'DATA/ATLAS_NT8/1m',
            '--bars5s-dir', 'DATA/ATLAS_NT8/5s',
            '--out', L_OOS, '--report', str(D / 'hardened_legs_OOS.txt')]),
        ('trajectory_IS', [py, t('build_trade_trajectory_dataset.py'),
            '--legs', L_IS, '--truth', P_IS, '--bars5s-dir', 'DATA/ATLAS/5s',
            '--out', TRAJ, '--report', str(D / 'trajectory_IS.txt')]),
    ]
    # B1-B6 -- pivot-dataset GBMs. B3-B6 also take --out-cache; B1/B2 do not.
    b16 = [
        ('b1', 'train_b1_pivot_imminent.py', 'b1_pivot_imminent', False),
        ('b2', 'train_b2_fakeout.py', 'b2_fakeout', False),
        ('b3', 'train_b3_ttp_regressor.py', 'b3_ttp_regressor', True),
        ('b4', 'train_b4_pivot_region.py', 'b4_pivot_region', True),
        ('b5', 'train_b5_leg_phase.py', 'b5_leg_phase', True),
        ('b6', 'train_b6_directional_pivot.py', 'b6_directional_pivot', True),
    ]
    for label, script, stem, has_cache in b16:
        argv = [py, t(script), '--is-dataset', P_IS, '--oos-dataset', P_OOS,
                '--out-pkl', str(D / f'{stem}.pkl'),
                '--out-report', str(D / f'{stem}.txt')]
        if has_cache:
            argv += ['--out-cache', str(D / f'{stem}_cache.parquet')]
        chain.append((label, argv))
    # B7 (patched copy) -- entry leg-amplitude sizer
    chain.append(('b7', [py, b7_py,
        '--is-truth', P_IS, '--oos-truth', P_OOS,
        '--out-pkl', str(D / 'b7_leg_sizer.pkl'),
        '--out-cache', B7_OOS, '--out-is-cache', B7_IS,
        '--out-report', str(D / 'b7_leg_sizer.txt')]))
    # B8 -- hour-risk; consumes the B7 leg caches (NOT the hardened-legs CSVs)
    chain.append(('b8', [py, t('train_b8_hour_risk.py'),
        '--is-truth', P_IS, '--oos-truth', P_OOS,
        '--is-legs', B7_IS, '--oos-legs', B7_OOS,
        '--out-pkl', str(D / 'b8_hour_risk.pkl'),
        '--out-oos-cache', str(D / 'b8_hour_risk_OOS.parquet'),
        '--out-report', str(D / 'b8_hour_risk.txt')]))
    # B9 -- during-trade remaining-amplitude (K=5/10/30/60/120) -> --out-dir
    chain.append(('b9', [py, t('train_b9_remaining_amplitude.py'),
        '--traj', TRAJ, '--out-dir', str(D)]))
    return chain


def run_step(label: str, argv: list, log_fh, env: dict) -> tuple:
    """Run one step; stdout+stderr -> log file. Returns (returncode, seconds)."""
    log_fh.write(f'\n{"=" * 72}\n[{label}]  {datetime.now():%Y-%m-%d %H:%M:%S}\n')
    log_fh.write('  ' + ' '.join(argv) + '\n' + '=' * 72 + '\n')
    log_fh.flush()
    t0 = time.time()
    rc = subprocess.run(argv, cwd=str(REPO), env=env,
                        stdout=log_fh, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    log_fh.write(f'\n[{label}] exit={rc}  {dt:.0f}s\n')
    log_fh.flush()
    return rc, dt


def do_pass(x: float, root: Path, env: dict, skip_done: bool) -> str:
    """Run the full retrain chain for one multiplier. Returns a status string."""
    D = root / suffix(x)
    if skip_done and (D / '_done').exists():
        print(f'  {suffix(x)}: _done present -- skip')
        return 'skipped'
    D.mkdir(parents=True, exist_ok=True)
    for marker in ('_done', '_failed'):
        (D / marker).unlink(missing_ok=True)

    patched_dir = root / '_patched'
    patched_dir.mkdir(parents=True, exist_ok=True)
    patched = {name: patch_trap(name, const, x, patched_dir)
               for name, const in TRAP_SCRIPTS.items()}

    chain = build_chain(x, D, patched)
    n_total = len(chain) + 1   # +1 for b10_copy
    steps_meta = []
    started = datetime.now(timezone.utc).isoformat()
    status = 'ok'

    with open(D / '_log.txt', 'a', encoding='utf-8') as log:
        log.write(f'\n\n{"#" * 72}\n# ATR x{x}  ({suffix(x)})  {started}\n'
                  f'{"#" * 72}\n')
        for i, (label, argv) in enumerate(chain, 1):
            print(f'  {suffix(x)}  [{i:2d}/{n_total}] {label} ...',
                  end='', flush=True)
            rc, dt = run_step(label, argv, log, env)
            steps_meta.append({'label': label, 'returncode': rc,
                               'seconds': round(dt, 1), 'cmd': ' '.join(argv)})
            print(f' {"ok" if rc == 0 else "FAIL"} ({dt:.0f}s)')
            if rc != 0:
                status = f'failed at {label}'
                break
        if status == 'ok':
            print(f'  {suffix(x)}  [{n_total:2d}/{n_total}] b10_copy ...',
                  end='', flush=True)
            b10_ok = True
            for f in ('b10_vol_regime_high.pkl', 'b10_vol_regime_low.pkl'):
                src = PROD / f
                if src.exists():
                    shutil.copy2(src, D / f)
                else:
                    b10_ok = False
                    log.write(f'\n[b10_copy] WARNING: {src} not found\n')
            steps_meta.append({'label': 'b10_copy',
                               'returncode': 0 if b10_ok else 1, 'seconds': 0.0})
            print(' ok' if b10_ok else ' WARN (production b10 not found)')

    finished = datetime.now(timezone.utc).isoformat()
    manifest = {
        'multiplier': x, 'suffix': suffix(x),
        'started_utc': started, 'finished_utc': finished, 'status': status,
        'steps': steps_meta,
        'outputs': sorted(p.name for p in D.iterdir() if p.is_file()),
    }
    (D / '_manifest.json').write_text(json.dumps(manifest, indent=2),
                                      encoding='utf-8')
    (D / ('_done' if status == 'ok' else '_failed')).write_text(
        finished + '\n', encoding='utf-8')
    return status


def precheck() -> list:
    """Return a list of blocking problems (empty == good to go)."""
    problems = []
    for s in PIPELINE_SCRIPTS:
        if not (TOOLS / s).exists():
            problems.append(f'missing pipeline script: tools/{s}')
    for label, p in (('IS V2 features', REPO / 'DATA/ATLAS/FEATURES_5s_v2/L1_5s'),
                      ('OOS V2 features',
                       REPO / 'DATA/ATLAS_NT8/FEATURES_5s_v2/L1_5s')):
        if not p.is_dir() or not any(p.glob('*.parquet')):
            problems.append(f'missing/empty {label}: {p}')
    for f in ('b10_vol_regime_high.pkl', 'b10_vol_regime_low.pkl'):
        if not (PROD / f).exists():
            problems.append(f'missing production B10 (needed for the per-X '
                             f'copy): {PROD / f}')
    return problems


def main():
    ap = argparse.ArgumentParser(
        description='ATR-multiplier retrain sweep (fully independent harness)')
    ap.add_argument('--multipliers', nargs='+', type=float,
                    default=DEFAULT_MULTIPLIERS,
                    help='ATR multipliers to retrain at (default: the 11-value '
                         'finer-in-1-4 set)')
    ap.add_argument('--from', dest='x_from', type=float, default=None,
                    help='restrict to multipliers >= this')
    ap.add_argument('--to', dest='x_to', type=float, default=None,
                    help='restrict to multipliers <= this')
    ap.add_argument('--root', default='checkpoints_atr_sweep',
                    help='sweep output root (default is auto-gitignored)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the per-multiplier command plan; run nothing')
    ap.add_argument('--no-skip-done', action='store_true',
                    help='re-run a multiplier even if its _done marker exists')
    args = ap.parse_args()

    mults = sorted(set(args.multipliers))
    bad = [m for m in mults if round(m, 1) != m]
    if bad:
        print(f'ERROR: multipliers must have <= 1 decimal place '
              f'(folder naming): {bad}')
        sys.exit(2)
    if args.x_from is not None:
        mults = [m for m in mults if m >= args.x_from]
    if args.x_to is not None:
        mults = [m for m in mults if m <= args.x_to]
    if not mults:
        print('No multipliers selected.')
        return

    root = Path(args.root)
    if not root.is_absolute():
        root = REPO / root

    problems = precheck()
    if problems:
        print('PRECHECK problems:')
        for p in problems:
            print('  -', p)
        if not args.dry_run:
            print('Aborting (use --dry-run to inspect the plan regardless).')
            sys.exit(1)

    print('=' * 72)
    print('ATR-MULTIPLIER RETRAIN SWEEP -- independent; modifies no existing file')
    print(f'  multipliers: {", ".join(str(m) for m in mults)}')
    print(f'  output root: {root}')
    print('=' * 72)

    if args.dry_run:
        for x in mults:
            D = root / suffix(x)
            patched = {n: root / '_patched' / f'{suffix(x)}__{n}'
                       for n in TRAP_SCRIPTS}
            print(f'\n===== {suffix(x)}  (ATR x{x})  ->  {D} =====')
            chain = build_chain(x, D, patched)
            for i, (label, argv) in enumerate(chain, 1):
                print(f'  [{i:2d}] {label}')
                print('       ' + ' '.join(argv))
            print(f'  [{len(chain) + 1:2d}] b10_copy  '
                  f'(copy production b10_vol_regime_*.pkl -> {D})')
        print('\nDRY RUN -- nothing executed, no folder created.')
        return

    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        p for p in (str(REPO), str(TOOLS), env.get('PYTHONPATH', '')) if p)
    root.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.time()
    for x in mults:
        print(f'\n----- ATR x{x}  ({suffix(x)}) -----')
        results[x] = do_pass(x, root, env, skip_done=not args.no_skip_done)

    lines = ['# ATR Retrain Sweep Summary', '',
             f'Generated {datetime.now(timezone.utc).isoformat()}',
             f'Root: {root}', '',
             '| Multiplier | Folder | Status |', '|--:|---|---|']
    for x in mults:
        lines.append(f'| {x} | {suffix(x)} | {results[x]} |')
    (root / 'SWEEP_SUMMARY.md').write_text('\n'.join(lines) + '\n',
                                           encoding='utf-8')

    mins = (time.time() - t0) / 60
    print(f'\nSweep finished in {mins:.0f} min  ->  {root / "SWEEP_SUMMARY.md"}')
    for x in mults:
        print(f'  x{x:<5}: {results[x]}')


if __name__ == '__main__':
    main()
