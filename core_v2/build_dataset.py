"""
Build v2 Feature Dataset — sequential per-TF, zero lookahead, decoupled by layer-family.

Writes per-layer-family parquets to:
    DATA/ATLAS/FEATURES_5s_v2/
        L0/YYYY_MM_DD.parquet                (1 col)
        L1_{TF}/YYYY_MM_DD.parquet           (6 cols) for tf in TF_ORDER
        L2_{TF}/YYYY_MM_DD.parquet           (9 cols)
        L3_{TF}/YYYY_MM_DD.parquet           (8 cols)

Every parquet is 5s-anchor aligned (timestamp column = 5s bar starts), so
layer-family files for the same day share identical timestamp indices and
can be joined trivially at load time.

================================================================
LOOKAHEAD DISCIPLINE — CRITICAL INVARIANT
================================================================

Higher-TF features are computed at the TF's NATIVE cadence, then step-filled
onto the 5s anchor timeline using:

    idx = searchsorted(tf_ts, anchor_ts - period, side='right') - 1

This is the lookahead-bias fix pattern (see commit 0c001c1f / 2026-04-17 audit).
The semantics:

    A bar labeled B at TF `tf` covers [B, B + period) — it closes at B + period.
    At anchor timestamp `ts`, we can only use bars where B + period <= ts,
    i.e. B <= ts - period. searchsorted(..., side='right') - 1 finds the
    last such bar.

Any "simplification" or "inlining" of this pattern risks reintroducing the
bug. It lives in ONE function (`_last_closed_idx`) with docstring + test.
================================================================

Usage:
    python core_v2/build_dataset.py --atlas DATA/ATLAS --fresh
    python core_v2/build_dataset.py --atlas DATA/ATLAS --days 5
    python core_v2/build_dataset.py --atlas DATA/ATLAS --start 2025-06-01 --end 2025-06-30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore', module='numba')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.statistical_field_engine import StatisticalFieldEngine
from core_v2.features import (
    TF_ORDER,
    TF_SECONDS,
    LAYER_FAMILIES,
    N_BASE,
    DEFAULT_FEATURES_ROOT,
)


# Subdirectory name inside the ATLAS root for v2 features
FEATURES_V2_SUBDIR = 'FEATURES_5s_v2'

# Schema version embedded in each layer-family parquet (for drift detection)
SCHEMA_VERSION = 1

# The anchor TF from which timestamps come
ANCHOR_TF = '5s'
ANCHOR_PERIOD = 5  # seconds

def set_anchor_globals(anchor_tf: str):
    global FEATURES_V2_SUBDIR, ANCHOR_TF, ANCHOR_PERIOD
    if anchor_tf == '1s':
        FEATURES_V2_SUBDIR = 'FEATURES_1s_v2'
        ANCHOR_TF = '1s'
        ANCHOR_PERIOD = 1
    elif anchor_tf == '5s':
        FEATURES_V2_SUBDIR = 'FEATURES_5s_v2'
        ANCHOR_TF = '5s'
        ANCHOR_PERIOD = 5
    else:
        raise ValueError(f"Unsupported anchor: {anchor_tf}")


# ─── Lookahead-safe alignment (THE critical invariant) ────────────────────

def _last_closed_idx(tf_ts: np.ndarray, anchor_ts: np.ndarray, period: int) -> np.ndarray:
    """For each anchor timestamp, return the index of the LAST closed bar in tf_ts.

    A bar labeled B covers [B, B + period) and closes at B + period.
    At anchor time ts, we can only use bars where B + period <= ts,
    i.e. B <= ts - period.

    Returns array of same length as anchor_ts. Values < 0 mean "no bar
    has closed yet" (warmup).

    Invariant: idx[i] < len(tf_ts) AND (idx[i] < 0 OR tf_ts[idx[i]] + period <= anchor_ts[i]).

    This is the verbatim pattern from training/build_dataset.py:284 - the
    fix for the lookahead bug from the prior refactor. Do NOT modify.
    """
    return np.searchsorted(tf_ts, anchor_ts - period, side='right') - 1


# ─── Bar loading ──────────────────────────────────────────────────────────

def _atlas_tf_dir(atlas_root: str, tf: str) -> str:
    return os.path.join(atlas_root, tf)


def _list_day_files(atlas_root: str, tf: str) -> list[str]:
    """Return sorted list of day-level parquet filenames for a TF."""
    pat = os.path.join(_atlas_tf_dir(atlas_root, tf), '*.parquet')
    files = sorted(glob.glob(pat))
    return files


def _day_from_path(path: str) -> str:
    """Extract 'YYYY_MM_DD' from a day parquet path."""
    return os.path.splitext(os.path.basename(path))[0]


def _load_tf_all_days(atlas_root: str, tf: str,
                     day_range: tuple[str, str] | None = None) -> pd.DataFrame:
    """Load ALL day-level bars for a TF (concatenated, deduplicated, sorted).

    If day_range is (start_day, end_day), only load days in that range.
    Returns DataFrame sorted by timestamp with OHLCV columns.
    """
    files = _list_day_files(atlas_root, tf)
    if not files:
        raise FileNotFoundError(
            f"No parquet files found under {_atlas_tf_dir(atlas_root, tf)}"
        )

    if day_range is not None:
        start_day, end_day = day_range
        files = [p for p in files
                 if start_day <= _day_from_path(p) <= end_day]

    dfs = [pd.read_parquet(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    df = (df.drop_duplicates(subset='timestamp', keep='last')
            .sort_values('timestamp')
            .reset_index(drop=True))
    return df


def _load_anchor_day(atlas_root: str, day: str) -> pd.DataFrame:
    """Load a single day's 5s anchor bars."""
    path = os.path.join(_atlas_tf_dir(atlas_root, ANCHOR_TF), f'{day}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing anchor file: {path}")
    return pd.read_parquet(path)


def _load_1s_day(atlas_root: str, day: str) -> pd.DataFrame:
    """Load a single day's 1s bars — the SOURCE for the L5 intra-bar distribution.

    L5 is within-bar-only (no cross-day trailing window), so per-day 1s is
    sufficient and correct (unlike L1-L4 which need all-history for warmup)."""
    path = os.path.join(_atlas_tf_dir(atlas_root, '1s'), f'{day}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing 1s file: {path}")
    return pd.read_parquet(path)


# ─── Per-day write helpers ────────────────────────────────────────────────

def _output_root(atlas_root: str) -> str:
    return os.path.join(atlas_root, FEATURES_V2_SUBDIR)


def _family_dir(atlas_root: str, family: str) -> str:
    return os.path.join(_output_root(atlas_root), family)


def _family_path(atlas_root: str, family: str, day: str) -> str:
    return os.path.join(_family_dir(atlas_root, family), f'{day}.parquet')


def _write_family(df: pd.DataFrame, path: str, schema_version: int):
    """Write a layer-family parquet with schema_version metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # pyarrow parquet writer supports kv-metadata via the `metadata` arg on Table.
    # For simplicity we use pandas' to_parquet (pyarrow engine by default).
    # Store schema_version as a column attribute via pd.ExcelWriter-style
    # metadata would require pyarrow direct. Instead, embed as a small
    # sidecar JSON (lighter than a custom pyarrow path).
    df.to_parquet(path, engine='pyarrow', index=False)
    sidecar = path + '.meta.json'
    with open(sidecar, 'w') as f:
        json.dump({
            'schema_version': schema_version,
            'family': os.path.basename(os.path.dirname(path)),
            'n_rows': len(df),
            'columns': list(df.columns),
            'written_at': datetime.utcnow().isoformat() + 'Z',
        }, f, indent=2)


# ─── Per-TF pipeline ──────────────────────────────────────────────────────

def _compute_tf_features_all_history(sfe: StatisticalFieldEngine,
                                      tf: str,
                                      tf_bars: pd.DataFrame) -> dict:
    """Compute L1/L2/L3/L4 for a TF over the FULL cross-day history.

    Returns dict with keys 'L1', 'L2', 'L3', 'L4', each a DataFrame aligned to
    tf_bars (same length, same row index).
    """
    l1 = sfe.compute_L1(tf_bars, tf=tf)
    l2 = sfe.compute_L2(tf_bars, tf=tf)
    l3 = sfe.compute_L3(tf_bars, tf=tf)
    N = sfe.windows.get(tf, 12)
    z_se = l3[f'L3_{tf}_z_se_{N}'].values
    l4 = sfe.compute_L4_NMP(tf_bars, tf=tf, z_se=z_se)
    return {'L1': l1, 'L2': l2, 'L3': l3, 'L4': l4}


def _align_to_anchor(tf_ts: np.ndarray,
                      tf_feature_df: pd.DataFrame,
                      anchor_ts: np.ndarray,
                      period: int) -> pd.DataFrame:
    """Step-fill a TF's feature DataFrame onto the 5s anchor timeline.

    Uses `_last_closed_idx` for lookahead-safe alignment.
    Rows where no TF bar has closed yet get NaN.
    """
    idx = _last_closed_idx(tf_ts, anchor_ts, period)
    valid = idx >= 0
    safe = np.where(valid, idx, 0)
    aligned = tf_feature_df.iloc[safe].reset_index(drop=True)
    # Wipe rows that were in warmup (idx < 0) — we don't want bar-0 values
    # leaking into the pre-warmup anchor timestamps.
    if (~valid).any():
        aligned.loc[~valid, :] = np.nan
    return aligned


def _build_layer_family_for_day(
    atlas_root: str,
    tf: str,
    layer: str,
    features_all_history: pd.DataFrame,
    tf_ts: np.ndarray,
    anchor_ts: np.ndarray,
    day: str,
):
    """Align TF features to 5s anchor, write one layer-family parquet for the day."""
    period = TF_SECONDS[tf]
    aligned = _align_to_anchor(tf_ts, features_all_history, anchor_ts, period)
    aligned.insert(0, 'timestamp', anchor_ts)
    family = f'{layer}_{tf}'
    path = _family_path(atlas_root, family, day)
    _write_family(aligned, path, SCHEMA_VERSION)


# ─── Main entry point ─────────────────────────────────────────────────────

def run(
    atlas_root: str,
    days: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    n_days: int | None = None,
    fresh: bool = False,
    skip_existing: bool = True,
    include_tfs: list[str] | None = None,
    anchor: str = '5s',
):
    """Build v2 feature parquets under {atlas_root}/FEATURES_5s_v2/.

    Args:
        atlas_root: e.g. 'DATA/ATLAS'.
        days: explicit list of day labels 'YYYY_MM_DD'. If None, derived from
              anchor parquets in {atlas_root}/5s/.
        start, end: 'YYYY-MM-DD' date bounds (inclusive); applied only if days is None.
        n_days: if set and days is None, take only the first n_days.
        fresh: if True, delete the FEATURES_5s_v2/ tree before writing.
        skip_existing: if True, skip days whose output parquets already exist
                       (useful for resume). Overridden to False by fresh.
        include_tfs: list of TF labels to process. Default: all TF_ORDER.
        anchor: '1s' or '5s'
    """
    set_anchor_globals(anchor)
    print(f"ATLAS root: {atlas_root}")
    print(f"Output:     {_output_root(atlas_root)}")

    if include_tfs is None:
        include_tfs = list(TF_ORDER)
    print(f"TFs:        {include_tfs}")

    # 1. Determine which days to process
    if days is None:
        anchor_files = _list_day_files(atlas_root, ANCHOR_TF)
        days = [_day_from_path(p) for p in anchor_files]

        if start is not None:
            start_day = start.replace('-', '_')
            days = [d for d in days if d >= start_day]
        if end is not None:
            end_day = end.replace('-', '_')
            days = [d for d in days if d <= end_day]
        if n_days is not None:
            days = days[:n_days]

    if not days:
        raise RuntimeError("No days to process after filtering.")

    print(f"Days:       {len(days)} ({days[0]} -> {days[-1]})")

    # 2. Fresh wipe if requested.
    # Windows holds handles on recently-written files (antivirus, indexer, etc.)
    # so shutil.rmtree can fail with PermissionError. We work around by:
    #   a) deleting files individually (ignoring locked ones)
    #   b) retrying rmdir with a short backoff
    #   c) not failing the whole run if a few files can't be removed -
    #      they'll just be overwritten later by new writes.
    if fresh:
        import time
        out_root = _output_root(atlas_root)
        if os.path.exists(out_root):
            print(f"FRESH: clearing {out_root}")
            unremoved = 0
            for root, dirs, files in os.walk(out_root, topdown=False):
                for f in files:
                    fp = os.path.join(root, f)
                    removed = False
                    for _attempt in range(3):
                        try:
                            os.remove(fp)
                            removed = True
                            break
                        except PermissionError:
                            time.sleep(0.2)
                        except FileNotFoundError:
                            removed = True
                            break
                    if not removed:
                        unremoved += 1
                for d in dirs:
                    dp = os.path.join(root, d)
                    for _attempt in range(3):
                        try:
                            os.rmdir(dp)
                            break
                        except (PermissionError, OSError):
                            time.sleep(0.2)
                        except FileNotFoundError:
                            break
            if unremoved:
                print(f"WARN: {unremoved} files could not be removed "
                      f"(likely held by another process). They will be "
                      f"overwritten on write.")
        skip_existing = False

    os.makedirs(_output_root(atlas_root), exist_ok=True)

    # 3. Instantiate SFE (stateless — one instance reused across all TFs and days)
    sfe = StatisticalFieldEngine()

    # 4. Per TF: load all-history bars ONCE, compute features ONCE, slice per day.
    #    This is efficient: for each TF, we touch disk once and CPU once per kernel,
    #    regardless of the number of days being built.
    for tf in include_tfs:
        print(f"\n--- TF {tf} ---")
        day_range = (min(days), max(days))

        # Load ALL TF bars in the day range (plus we want cross-day history
        # for warmup, so use the full file list up through end_day).
        # We load through end_day; anything before is also included for warmup.
        tf_files = _list_day_files(atlas_root, tf)
        tf_files = [p for p in tf_files
                    if _day_from_path(p) <= day_range[1]]
        if not tf_files:
            print(f"  no {tf} files through {day_range[1]}, skipping TF")
            continue

        dfs = [pd.read_parquet(p) for p in tf_files]
        tf_bars = (pd.concat(dfs, ignore_index=True)
                     .drop_duplicates(subset='timestamp', keep='last')
                     .sort_values('timestamp')
                     .reset_index(drop=True))
        tf_ts = tf_bars['timestamp'].values.astype(np.int64)
        print(f"  loaded {len(tf_bars):,} {tf} bars ({len(tf_files)} files)")

        # Compute features for the entire history
        print(f"  computing L1/L2/L3/L4 ...")
        features = _compute_tf_features_all_history(sfe, tf, tf_bars)

        # For each day, load that day's 5s anchor and step-fill the features onto it
        for day in tqdm(days, desc=f"  writing {tf}", ncols=80):
            # Skip if all 4 layer outputs already exist and we allow it
            if skip_existing and all(
                os.path.exists(_family_path(atlas_root, f'{lyr}_{tf}', day))
                for lyr in ('L1', 'L2', 'L3', 'L4')
            ):
                continue

            try:
                anchor = _load_anchor_day(atlas_root, day)
            except FileNotFoundError:
                continue
            anchor_ts = anchor['timestamp'].values.astype(np.int64)

            for layer in ('L1', 'L2', 'L3', 'L4'):
                _build_layer_family_for_day(
                    atlas_root=atlas_root,
                    tf=tf,
                    layer=layer,
                    features_all_history=features[layer],
                    tf_ts=tf_ts,
                    anchor_ts=anchor_ts,
                    day=day,
                )

        # Free memory before next TF
        del tf_bars, features

    # 5. L0 (global — computed from 5s anchor timestamps only)
    print(f"\n--- L0 (global) ---")
    for day in tqdm(days, desc="  writing L0", ncols=80):
        if skip_existing and os.path.exists(_family_path(atlas_root, 'L0', day)):
            continue
        try:
            anchor = _load_anchor_day(atlas_root, day)
        except FileNotFoundError:
            continue
        l0 = sfe.compute_L0(anchor)
        l0.insert(0, 'timestamp', anchor['timestamp'].values.astype(np.int64))
        _write_family(l0, _family_path(atlas_root, 'L0', day), SCHEMA_VERSION)

    # 6. L5 (intra-bar 1s distribution — sourced from raw 1s, grouped per TF)
    #    Load each day's 1s ONCE, compute all TFs from it. Within-bar-only, so no
    #    cross-day warmup needed. Step-fill uses period=TF_SECONDS[tf] on the L5
    #    bar_ts (last-closed rule), EXACTLY like L1-L4 — NOT period=5/1 (= lookahead).
    print(f"\n--- L5 (intra-bar 1s distribution) ---")
    for day in tqdm(days, desc="  writing L5", ncols=80):
        if skip_existing and all(
            os.path.exists(_family_path(atlas_root, f'L5_{tf}', day))
            for tf in include_tfs
        ):
            continue
        try:
            anchor = _load_anchor_day(atlas_root, day)
            df_1s = _load_1s_day(atlas_root, day)
        except FileNotFoundError:
            continue  # no anchor or no 1s for this day -> L1-L4 still built above
        anchor_ts = anchor['timestamp'].values.astype(np.int64)
        for tf in include_tfs:
            if skip_existing and os.path.exists(_family_path(atlas_root, f'L5_{tf}', day)):
                continue
            l5 = sfe.compute_L5_ldist(df_1s, tf)
            if len(l5) == 0:
                continue
            l5_ts = l5['bar_ts'].to_numpy(np.int64)
            l5_feat = l5.drop(columns=['bar_ts']).reset_index(drop=True)
            aligned = _align_to_anchor(l5_ts, l5_feat, anchor_ts, TF_SECONDS[tf])
            aligned.insert(0, 'timestamp', anchor_ts)
            _write_family(aligned, _family_path(atlas_root, f'L5_{tf}', day), SCHEMA_VERSION)

    print(f"\nDone. Wrote {len(days)} days to {_output_root(atlas_root)}")


# ─── CLI ──────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description='Build v2 139D feature parquets (layer-family decoupled)')
    ap.add_argument('--atlas', type=str, default='DATA/ATLAS',
                    help='ATLAS root dir (default: DATA/ATLAS)')
    ap.add_argument('--start', type=str, default=None,
                    help='YYYY-MM-DD lower bound (inclusive)')
    ap.add_argument('--end', type=str, default=None,
                    help='YYYY-MM-DD upper bound (inclusive)')
    ap.add_argument('--days', type=int, default=None,
                    help='Process only the first N days after filtering')
    ap.add_argument('--fresh', action='store_true',
                    help='Wipe FEATURES_5s_v2/ before writing')
    ap.add_argument('--no-skip-existing', action='store_true',
                    help='Overwrite existing parquets (default is skip)')
    ap.add_argument('--tfs', type=str, nargs='+', default=None,
                    help=f'Which TFs to build (default: all of {TF_ORDER})')
    ap.add_argument('--anchor', type=str, default='5s', choices=['1s', '5s'],
                    help='Anchor timeline (1s or 5s). Default: 5s')
    return ap.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(
        atlas_root=args.atlas,
        start=args.start,
        end=args.end,
        n_days=args.days,
        fresh=args.fresh,
        skip_existing=not args.no_skip_existing,
        include_tfs=args.tfs,
        anchor=args.anchor,
    )
