"""
Clean + rebuild the FEATURES_5s folder for an atlas.

Deletes <atlas>/FEATURES_5s/, then runs build_dataset.py --atlas <atlas>
with warm-start auto-detect (picks up sibling atlas's checkpoint if any).

Usage:
    python tools/rebuild_features.py --atlas DATA/ATLAS_NT8
    python tools/rebuild_features.py --atlas DATA/ATLAS_NT8 --resolution 5s
    python tools/rebuild_features.py --atlas DATA/ATLAS --yes          # skip confirm
    python tools/rebuild_features.py --atlas DATA/ATLAS_NT8 --warm-from DATA/ATLAS/checkpoint.json
    python tools/rebuild_features.py --atlas DATA/ATLAS_NT8 --cold     # force cold start

Flags:
    --atlas        Atlas root (required)
    --resolution   Anchor TF (default 5s)
    --warm-from    Checkpoint path; passed through to build_dataset
    --cold         Equivalent to --warm-from ""
    --yes, -y      Skip "are you sure" prompt
    --start, --end, --days   Passed through to build_dataset
"""
import os
import sys
import shutil
import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas', required=True, help='Atlas root')
    ap.add_argument('--resolution', default='5s', choices=['5s', '15s', '1m'])
    ap.add_argument('--warm-from', default=None, help='Checkpoint path for warm-start')
    ap.add_argument('--cold', action='store_true', help='Force cold start')
    ap.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--days', default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.atlas):
        print(f'ERROR: atlas root {args.atlas} not found')
        return 1

    feat_dir = os.path.join(args.atlas, f'FEATURES_{args.resolution}')
    n_existing = 0
    if os.path.isdir(feat_dir):
        import glob
        n_existing = len(glob.glob(os.path.join(feat_dir, '*.parquet')))

    print(f'Atlas:        {args.atlas}')
    print(f'Resolution:   {args.resolution}')
    print(f'Feature dir:  {feat_dir}')
    print(f'Existing:     {n_existing} parquet files')
    warm_mode = 'COLD (forced)' if args.cold else (
        f'EXPLICIT: {args.warm_from}' if args.warm_from else 'AUTO-DETECT')
    print(f'Warm-start:   {warm_mode}')
    print()

    if n_existing > 0 and not args.yes:
        reply = input(f'Delete {n_existing} files in {feat_dir} and rebuild? [y/N] ')
        if reply.strip().lower() not in ('y', 'yes'):
            print('Aborted.')
            return 0

    # Clean — robust on Windows where the dir handle may be held by
    # Explorer / an editor. Delete files individually; ignore dir-rmdir
    # failure (empty dir is fine, build will write into it).
    if os.path.isdir(feat_dir):
        import glob as _glob
        deleted = 0
        errors = []
        for fp in _glob.glob(os.path.join(feat_dir, '*')):
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
                    deleted += 1
                elif os.path.isdir(fp):
                    shutil.rmtree(fp, ignore_errors=True)
            except Exception as e:
                errors.append((fp, str(e)))
        print(f'Deleted {deleted} files in {feat_dir}' +
              (f' ({len(errors)} errors)' if errors else ''))
        if errors:
            for fp, e in errors[:5]:
                print(f'  skip: {fp} ({e})')
        # Try to remove empty dir; OK if it fails (Windows lock)
        try:
            os.rmdir(feat_dir)
        except OSError:
            pass
    os.makedirs(feat_dir, exist_ok=True)

    # Build command
    cmd = [sys.executable, 'training/build_dataset.py',
           '--atlas', args.atlas,
           '--resolution', args.resolution]
    if args.cold:
        cmd += ['--warm-from', '']
    elif args.warm_from is not None:
        cmd += ['--warm-from', args.warm_from]
    if args.start: cmd += ['--start', args.start]
    if args.end:   cmd += ['--end', args.end]
    if args.days:  cmd += ['--days', str(args.days)]

    print(f'Running: {" ".join(cmd)}')
    print()
    return subprocess.call(cmd)


if __name__ == '__main__':
    sys.exit(main())
