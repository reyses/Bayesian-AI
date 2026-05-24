"""
databento_to_nt8_ticks.py -- Convert DATA/ATLAS/1s OHLCV parquet to NT8
tick-format import files.

Strategy: each 1s OHLCV bar -> up to 4 fake ticks (Open/High/Low/Close) at
sub-second offsets. NT8 builds whatever bars (1s/5s/1m/etc.) it needs from
these ticks. This gives proper sub-minute SL/Trail evaluation in v1.0.6-RC.

NT8 tick import format (from NT8 docs):
  yyyyMMdd HHmmss fffffff;Last;Volume

Tick distribution within each second (preserves OHLC path):
  - If H == L (flat second): emit 1 tick at close
  - If close > open (up bar):   T+0.0=O, T+0.25=L, T+0.50=H, T+0.75=C
  - If close < open (down bar): T+0.0=O, T+0.25=H, T+0.50=L, T+0.75=C
  - If close == open (doji):    T+0.0=O, T+0.25=H, T+0.50=L, T+0.75=C
Volume distributed evenly across emitted ticks.

Output: per-month chunks (~120 MB each) for manageable NT8 import.
  examples/nt8_import_ticks/{INSTRUMENT}_2025-01.Last.txt
  ...

Usage:
    python tools/databento_to_nt8_ticks.py
    python tools/databento_to_nt8_ticks.py --instrument MNQ_CONT
    python tools/databento_to_nt8_ticks.py --tz America/New_York
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np


# Sub-second offsets for OHLC ticks (in milliseconds)
TICK_OFFSETS_MS = [0, 250, 500, 750]


def emit_ticks_for_bar(open_, high, low, close, volume) -> list[tuple[int, float, int]]:
    """Return list of (offset_ms, price, volume) tuples for one 1s OHLCV bar."""
    if high == low:
        # Flat second — single tick at close
        return [(0, close, int(volume))]

    # Determine OHLC path order
    if close > open_:
        prices = [open_, low, high, close]
    elif close < open_:
        prices = [open_, high, low, close]
    else:  # doji — pick a path
        prices = [open_, high, low, close]

    # Distribute volume across 4 ticks
    v_each = max(1, int(volume) // 4)
    v_remainder = int(volume) - v_each * 3
    vols = [v_each, v_each, v_each, v_remainder]

    # Dedupe consecutive duplicates while preserving OHLC info
    out = []
    last_p = None
    for off, p, v in zip(TICK_OFFSETS_MS, prices, vols):
        if last_p is not None and p == last_p:
            # Merge with prior tick: add volume
            out[-1] = (out[-1][0], out[-1][1], out[-1][2] + v)
        else:
            out.append((off, p, v))
            last_p = p
    return out


def convert_month(parquet_files: list[Path], out_path: str, tz: str, ym: str) -> int:
    """Convert one month's worth of 1s parquet files into a single NT8 tick .txt."""
    if not parquet_files:
        return 0
    parts = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df.empty:
            continue
        parts.append(df)
    if not parts:
        return 0

    bars = pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Convert UTC -> target timezone with DST awareness
    bars["dt_utc"] = pd.to_datetime(bars["timestamp"], unit="s", utc=True)
    bars["dt_local"] = bars["dt_utc"].dt.tz_convert(tz)

    n_bars = len(bars)
    n_ticks_written = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        # Vectorized base date string
        opens   = bars["open"].to_numpy()
        highs   = bars["high"].to_numpy()
        lows    = bars["low"].to_numpy()
        closes  = bars["close"].to_numpy()
        volumes = bars["volume"].to_numpy()
        dt_locals = bars["dt_local"].to_list()

        for i in range(n_bars):
            ticks = emit_ticks_for_bar(opens[i], highs[i], lows[i], closes[i], volumes[i])
            base_dt = dt_locals[i]
            base_str = base_dt.strftime("%Y%m%d %H%M%S")
            for off_ms, p, v in ticks:
                # NT8 format: yyyyMMdd HHmmss fffffff (7-digit fractional)
                # Convert ms offset to 7-digit string
                frac = f"{off_ms*10000:07d}"  # ms × 10,000 = nanoseconds-ish (7-digit precision)
                f.write(f"{base_str} {frac};{p:.2f};{v}\n")
                n_ticks_written += 1
    return n_ticks_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="DATA/ATLAS",
                    help="Source root (default: DATA/ATLAS)")
    ap.add_argument("--out-dir", default="examples/nt8_import_ticks",
                    help="Where to write monthly tick .txt files")
    ap.add_argument("--instrument", default="MNQ_CONT",
                    help="Instrument label for filename. Create matching custom " +
                         "instrument in NT8 first.")
    ap.add_argument("--tz", default="America/Los_Angeles",
                    help="Target timezone. Default America/Los_Angeles (DST-aware).")
    ap.add_argument("--start-ym", default=None, help="YYYY-MM lower bound (e.g., 2025-06)")
    ap.add_argument("--end-ym", default=None, help="YYYY-MM upper bound")
    args = ap.parse_args()

    print("=" * 80)
    print("DATABENTO 1s OHLCV -> NT8 TICKS CONVERTER")
    print("=" * 80)
    print(f"Source:     {args.atlas}/1s/")
    print(f"Output:     {args.out_dir}")
    print(f"Instrument: {args.instrument}")
    print(f"Timezone:   {args.tz}")
    print()

    src = Path(args.atlas) / "1s"
    if not src.exists():
        print(f"FATAL: missing {src}")
        return

    # Group parquet files by year-month (parquet filename format YYYY_MM_DD.parquet)
    files_by_ym: dict[str, list[Path]] = {}
    for f in sorted(src.glob("*.parquet")):
        # Extract YYYY-MM from "YYYY_MM_DD.parquet"
        stem = f.stem
        if len(stem) >= 7 and stem[4] == "_":
            ym = stem[:7].replace("_", "-")
            if args.start_ym and ym < args.start_ym: continue
            if args.end_ym and ym > args.end_ym: continue
            files_by_ym.setdefault(ym, []).append(f)

    print(f"Months found: {len(files_by_ym)}")
    print()

    summary = []
    total_ticks = 0
    for ym in sorted(files_by_ym.keys()):
        files = files_by_ym[ym]
        out_path = os.path.join(args.out_dir, f"{args.instrument}_{ym}.Last.txt")
        print(f"  {ym}: {len(files)} files -> {out_path}")
        n = convert_month(files, out_path, args.tz, ym)
        size_mb = os.path.getsize(out_path) / 1024 / 1024 if os.path.exists(out_path) else 0
        print(f"    {n:,} ticks  ({size_mb:.1f} MB)")
        summary.append((ym, len(files), n, size_mb))
        total_ticks += n

    print()
    print("=" * 80)
    print(f"SUMMARY  ({total_ticks:,} total ticks)")
    print("=" * 80)
    for ym, nf, nt, mb in summary:
        print(f"  {ym}: {nf:>3} files  {nt:>10,} ticks  {mb:>6.1f} MB")
    print()
    print("=" * 80)
    print("CUSTOM INSTRUMENT CREATION (do this in NT8 BEFORE importing)")
    print("=" * 80)
    print(f"1. NT8 -> Tools -> Instrument Manager (or Ctrl+I)")
    print(f"2. Click 'Add' (top-left). Fill in:")
    print(f"   - Master instrument: {args.instrument}")
    print(f"   - Description: 'MNQ continuous (Databento)' (or anything)")
    print(f"   - Exchange: GLOBEX")
    print(f"   - Currency: USD")
    print(f"   - Tick size: 0.25")
    print(f"   - Point value: 2 (= $2/point on MNQ)")
    print(f"   - Initial margin / maintenance: copy from MNQ JUN26")
    print(f"3. Click 'OK'. The instrument now appears in NT8's instrument list.")
    print()
    print("=" * 80)
    print("TICK IMPORT (per-month, after instrument exists)")
    print("=" * 80)
    print(f"For each .Last.txt file in {os.path.abspath(args.out_dir)}:")
    print(f"1. NT8 -> File -> Utilities -> Historical Data -> Import")
    print(f"2. Instrument: select '{args.instrument}'")
    print(f"3. Type: Tick")
    print(f"4. File: browse to the .Last.txt file for the month you're importing")
    print(f"5. Click Import. NT8 will load (~1-5 min per file).")
    print(f"6. Repeat for each month file.")
    print()
    print("After all months imported:")
    print(f"  - NT8 has full tick data on '{args.instrument}'")
    print(f"  - Strategy Analyzer + Playback can run any TF (1s/5s/1m/etc.)")
    print(f"  - v1.0.6-RC will get proper sub-second SL/Trail evaluation")


if __name__ == "__main__":
    main()
