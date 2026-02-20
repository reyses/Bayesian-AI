#!/usr/bin/env python3
"""
OOS ATLAS Setup
Reorganizes DATA/ATLAS into a clean train/test split:

  DATA/ATLAS/         -- monthly files Jan-Nov 2025 only (training)
  DATA/ATLAS_OOS/     -- daily files Dec 2025 - Feb 2026 (blind OOS test)

Actions taken:
  1. Move daily files >= OOS_START (20251201) from ATLAS/{tf}/ to ATLAS_OOS/{tf}/
  2. Delete training-period daily files from ATLAS/{tf}/ (monthly files cover same data)
  3. Delete training-period monthly files that are >= OOS month (2025_12, 2026_xx)
     from ATLAS/{tf}/ since that data now lives in ATLAS_OOS as daily files.

Usage:
    python scripts/setup_oos_atlas.py
    python scripts/setup_oos_atlas.py --dry-run
    python scripts/setup_oos_atlas.py --atlas DATA/ATLAS --oos DATA/ATLAS_OOS --oos-start 20251201
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path

# Default boundaries
_DEFAULT_OOS_START = '20251201'   # YYYYMMDD — first day of blind test window
_MONTHLY_RE = re.compile(r'^\d{4}_\d{2}$')   # YYYY_MM
_DAILY_RE   = re.compile(r'^\d{8}$')          # YYYYMMDD


def yyyymmdd_to_yyyymm(yyyymmdd: str) -> str:
    """'20251201' -> '2025_12'"""
    return f"{yyyymmdd[:4]}_{yyyymmdd[4:6]}"


def main():
    parser = argparse.ArgumentParser(
        description="Split ATLAS into training (monthly) and OOS (daily) directories."
    )
    parser.add_argument('--atlas',     default=os.path.join('DATA', 'ATLAS'))
    parser.add_argument('--oos',       default=os.path.join('DATA', 'ATLAS_OOS'))
    parser.add_argument('--oos-start', default=_DEFAULT_OOS_START, metavar='YYYYMMDD',
                        help="First day of OOS window (default: 20251201)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print actions without executing them")
    args = parser.parse_args()

    atlas_root = Path(args.atlas)
    oos_root   = Path(args.oos)
    oos_start  = args.oos_start           # YYYYMMDD e.g. '20251201'
    oos_start_ym = yyyymmdd_to_yyyymm(oos_start)  # YYYY_MM e.g. '2025_12'
    dry = args.dry_run

    print(f"OOS ATLAS Setup")
    print(f"  Training ATLAS:  {atlas_root}  (monthly, through {oos_start_ym} exclusive)")
    print(f"  OOS ATLAS:       {oos_root}    (daily, from {oos_start} onward)")
    print(f"  Dry run:         {dry}")
    print()

    if not atlas_root.exists():
        print(f"ERROR: {atlas_root} not found"); sys.exit(1)

    tf_dirs = sorted([d for d in atlas_root.iterdir() if d.is_dir()])
    if not tf_dirs:
        print(f"ERROR: No TF subdirectories in {atlas_root}"); sys.exit(1)

    moved = deleted_daily = deleted_monthly = 0

    for tf_dir in tf_dirs:
        tf = tf_dir.name
        oos_tf_dir = oos_root / tf

        all_parquets = sorted(tf_dir.glob('*.parquet'))
        if not all_parquets:
            continue

        daily_files   = [f for f in all_parquets if _DAILY_RE.match(f.stem)]
        monthly_files = [f for f in all_parquets if _MONTHLY_RE.match(f.stem)]

        tf_moved = tf_del_d = tf_del_m = 0

        # ── Daily files ────────────────────────────────────────────────────────
        for f in daily_files:
            day_str = f.stem   # YYYYMMDD

            if day_str >= oos_start:
                # Move to ATLAS_OOS
                dest = oos_tf_dir / f.name
                if not dry:
                    oos_tf_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))
                print(f"  MOVE  {tf}/{f.name}  ->  ATLAS_OOS/{tf}/")
                tf_moved += 1
            else:
                # Training period daily file — delete (monthly file covers same data)
                if not dry:
                    f.unlink()
                print(f"  DEL   {tf}/{f.name}  (duplicate of monthly)")
                tf_del_d += 1

        # ── Monthly files ──────────────────────────────────────────────────────
        for f in monthly_files:
            ym = f.stem   # YYYY_MM

            if ym >= oos_start_ym:
                # OOS period monthly file — delete (data now in ATLAS_OOS as daily)
                if not dry:
                    f.unlink()
                print(f"  DEL   {tf}/{f.name}  (OOS month, now in ATLAS_OOS as daily files)")
                tf_del_m += 1
            # else: training period monthly — keep as-is

        if tf_moved or tf_del_d or tf_del_m:
            print(f"    [{tf}]  moved={tf_moved}  del_daily={tf_del_d}  del_monthly={tf_del_m}")
        moved          += tf_moved
        deleted_daily  += tf_del_d
        deleted_monthly += tf_del_m

    print()
    print(f"Summary:")
    print(f"  Files moved to ATLAS_OOS:    {moved}")
    print(f"  Daily files deleted (train): {deleted_daily}")
    print(f"  Monthly files deleted (OOS): {deleted_monthly}")
    print()

    # Print resulting structure
    if not dry:
        print("Resulting structure:")
        for root, label in [(atlas_root, 'ATLAS (training)'), (oos_root, 'ATLAS_OOS (blind test)')]:
            if root.exists():
                tf_dirs_now = sorted([d for d in root.iterdir() if d.is_dir()])
                for td in tf_dirs_now:
                    files = sorted(td.glob('*.parquet'))
                    if files:
                        print(f"  {label}/{td.name}/  {len(files)} files  "
                              f"[{files[0].stem} .. {files[-1].stem}]")

    print("\nDone.")
    print("\nTo train:   python training/orchestrator.py --data DATA/ATLAS --fresh --no-dashboard")
    print("To OOS sim: python training/orchestrator.py --oos --data DATA/ATLAS_OOS "
          "--account-size 100.0 --no-dashboard")


if __name__ == '__main__':
    main()
