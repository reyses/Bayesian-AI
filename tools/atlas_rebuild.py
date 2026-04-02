"""
Atlas Rebuild — clean, validate, and rebuild features in one step.

1. Clean 1s tick spikes (interpolate bad ticks)
2. Validate all TFs against 1s ground truth (fix mismatches)
3. Rebuild ATLAS_FEATURES for all TFs

Usage:
  python tools/atlas_rebuild.py                  # full rebuild
  python tools/atlas_rebuild.py --dry-run        # report only
  python tools/atlas_rebuild.py --skip-clean     # skip spike cleaning
  python tools/atlas_rebuild.py --skip-validate  # skip validation
  python tools/atlas_rebuild.py --skip-features  # skip feature rebuild
"""
import subprocess
import sys
import os
import time

DRY_RUN = '--dry-run' in sys.argv
SKIP_CLEAN = '--skip-clean' in sys.argv
SKIP_VALIDATE = '--skip-validate' in sys.argv
SKIP_FEATURES = '--skip-features' in sys.argv


def run(cmd, desc):
    """Run a command, print status."""
    print(f'\n{"="*60}')
    print(f'STEP: {desc}')
    print(f'  CMD: {cmd}')
    print(f'{"="*60}')
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    status = 'OK' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
    print(f'\n  {status} ({elapsed:.0f}s)')
    return result.returncode == 0


def main():
    print(f'ATLAS REBUILD PIPELINE')
    print(f'  Mode: {"DRY RUN" if DRY_RUN else "FULL REBUILD"}')
    print()

    steps = []

    # Step 1: Clean tick spikes
    if not SKIP_CLEAN:
        dry = ' --dry-run' if DRY_RUN else ''
        steps.append((
            f'python tools/clean_tick_spikes.py{dry}',
            'Clean 1s tick spikes (interpolate bad ticks)'
        ))

    # Step 2: Validate all TFs against 1s
    if not SKIP_VALIDATE and not DRY_RUN:
        steps.append((
            'python tools/validate_data.py',
            'Validate all TFs against 1s ground truth'
        ))

    # Step 3: Rebuild features
    if not SKIP_FEATURES and not DRY_RUN:
        # Delete existing March features to force rebuild
        steps.append((
            'python -c "import glob,os; [os.remove(f) for tf in [\'1s\',\'5s\',\'15s\',\'30s\',\'1m\',\'3m\',\'5m\',\'15m\',\'30m\',\'1h\',\'4h\',\'1D\'] for f in glob.glob(f\'DATA/ATLAS_FEATURES/{tf}/2026_03.parquet\')]"',
            'Clear March feature cache'
        ))
        steps.append((
            'python tools/build_feature_atlas.py',
            'Rebuild ATLAS_FEATURES for all TFs'
        ))

    for cmd, desc in steps:
        ok = run(cmd, desc)
        if not ok:
            print(f'\n*** PIPELINE FAILED at: {desc} ***')
            return

    print(f'\n{"="*60}')
    print(f'ATLAS REBUILD COMPLETE')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
