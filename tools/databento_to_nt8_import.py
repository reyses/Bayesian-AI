"""
databento_to_nt8_import.py -- Convert DATA/ATLAS/{tf}/*.parquet to NT8
Historical Data Manager import format.

NT8 import format (per NinjaTrader 8 docs):
  - One text file per instrument + timeframe
  - Line format for minute bars: `yyyyMMdd HHmmss;Open;High;Low;Close;Volume`
  - Line format for daily bars:  `yyyyMMdd;Open;High;Low;Close;Volume`
  - Timestamp interpreted as the user's local timezone in NT8 settings

User local timezone: US Pacific (PDT/PST per earlier evidence in trades CSVs).
We convert UTC parquet timestamps -> US/Pacific to match.

Output:
  examples/nt8_import/MNQ_06-26_1m.txt
  examples/nt8_import/MNQ_06-26_1D.txt
  ...

Then in NT8:
  1. File -> Utilities -> Historical Data -> Import
  2. Instrument: MNQ JUN26 (or matching contract)
  3. Type: Minute (for 1m file), Day (for 1D), Tick/Second (for 1s — large)
  4. Browse to the .txt file
  5. NT8 imports. Repeat per timeframe.

Usage:
    python tools/databento_to_nt8_import.py --tfs 1m 1D
    python tools/databento_to_nt8_import.py --instrument "MNQ JUN26" --tz "America/Los_Angeles"
    python tools/databento_to_nt8_import.py --tfs 1s   # warning: 36M+ rows
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

# NT8 import format per timeframe — second-grain timestamp for sub-day,
# day-only for daily.
TF_FORMAT = {
    "1s":  "%Y%m%d %H%M%S",
    "5s":  "%Y%m%d %H%M%S",
    "15s": "%Y%m%d %H%M%S",
    "30s": "%Y%m%d %H%M%S",
    "1m":  "%Y%m%d %H%M%S",
    "5m":  "%Y%m%d %H%M%S",
    "15m": "%Y%m%d %H%M%S",
    "30m": "%Y%m%d %H%M%S",
    "1h":  "%Y%m%d %H%M%S",
    "4h":  "%Y%m%d %H%M%S",
    "1D":  "%Y%m%d",
}


def convert_tf(parquet_root: str, tf: str, out_path: str, tz: str,
               instrument_label: str) -> tuple:
    """Read DATA/ATLAS/{tf}/*.parquet, convert UTC -> local tz, write NT8 .txt."""
    src = Path(parquet_root) / tf
    if not src.exists():
        return (0, f"missing source dir: {src}")

    files = sorted(src.glob("*.parquet"))
    parts = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"  WARN read fail {f.name}: {e}")
            continue
        if df.empty:
            continue
        parts.append(df)

    if not parts:
        return (0, f"no rows in {src}")

    bars = pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    n = len(bars)
    print(f"    loaded {n:,} bars from {len(files)} files")

    # Convert UTC unix seconds -> tz-aware local timestamps (handles DST)
    bars["dt_utc"] = pd.to_datetime(bars["timestamp"], unit="s", utc=True)
    bars["dt_local"] = bars["dt_utc"].dt.tz_convert(tz)
    fmt = TF_FORMAT.get(tf, "%Y%m%d %H%M%S")
    bars["fmt_dt"] = bars["dt_local"].dt.strftime(fmt)

    # Write NT8 import format: yyyyMMdd HHmmss;O;H;L;C;V
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for _, r in bars.iterrows():
            f.write(f"{r['fmt_dt']};{r['open']:.2f};{r['high']:.2f};{r['low']:.2f};{r['close']:.2f};{int(r['volume'])}\n")

    return (n, f"OK -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="DATA/ATLAS",
                    help="Source root (default: DATA/ATLAS — Databento 12+ months)")
    ap.add_argument("--out-dir", default="examples/nt8_import",
                    help="Where to write NT8 import .txt files")
    ap.add_argument("--instrument", default="MNQ_06-26",
                    help="Instrument label for filename. NT8 dialog still asks for instrument; this is just naming.")
    ap.add_argument("--tz", default="America/Los_Angeles",
                    help="Target timezone for timestamps (NT8 reads in user's local tz). " +
                         "Default America/Los_Angeles (PDT/PST, handles DST).")
    ap.add_argument("--tfs", nargs="+",
                    default=["1m", "1D"],
                    help="Timeframes to export (subset of: 1s 5s 15s 30s 1m 5m 15m 30m 1h 4h 1D). " +
                         "Default: 1m and 1D. Add 1s only if needed (36M+ rows).")
    args = ap.parse_args()

    print("=" * 80)
    print("DATABENTO -> NT8 IMPORT CONVERTER")
    print("=" * 80)
    print(f"Source:     {args.atlas}")
    print(f"Output:     {args.out_dir}")
    print(f"Instrument: {args.instrument}")
    print(f"Timezone:   {args.tz}")
    print(f"TFs:        {args.tfs}")
    print()

    summary = []
    for tf in args.tfs:
        out_path = os.path.join(args.out_dir, f"{args.instrument}_{tf}.txt")
        print(f"Converting {tf}...")
        n, msg = convert_tf(args.atlas, tf, out_path, args.tz, args.instrument)
        print(f"  {msg}")
        summary.append((tf, n, msg))
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for tf, n, msg in summary:
        print(f"  {tf:>4}: {n:>10,} bars  {msg}")
    total = sum(n for _, n, _ in summary)
    print(f"  Total: {total:,} bars exported")
    print()
    print("=" * 80)
    print("NEXT STEPS — Manual NT8 import")
    print("=" * 80)
    print(f"1. NT8 -> File -> Utilities -> Historical Data -> Import")
    print(f"2. Instrument: select MNQ JUN26 (or matching front-month)")
    print(f"3. Type: select matching timeframe (Minute for 1m, Day for 1D)")
    print(f"4. File: browse to {os.path.abspath(args.out_dir)}")
    print(f"5. Choose the corresponding {{instrument}}_{{tf}}.txt file")
    print(f"6. Click Import. NT8 will load and merge with existing data.")
    print(f"7. Repeat for each timeframe.")
    print()
    print("CAVEATS:")
    print("  - DATA/ATLAS is back-adjusted continuous (verified — no roll gaps).")
    print("    NT8 will treat imports as the selected instrument's bars regardless.")
    print("  - Timezone conversion uses tz='{}' (DST-aware).".format(args.tz))
    print("  - 1s import is large (~36M rows for 14 months) — hours to import.")
    print("  - After import, NT8 SA can run on full 14-month range.")


if __name__ == "__main__":
    main()
