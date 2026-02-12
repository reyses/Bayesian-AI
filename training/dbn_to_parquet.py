"""
DBN-to-Parquet Converter
Converts Databento .dbn.zst files to pipeline-compatible .parquet files.

Handles multiple data scenarios:
  1. OHLCV-1s files  (single file, multi-month, may contain spreads)
  2. Trades files    (per-day files in a directory)
  3. Mixed symbols   (filters to front-month outrights, stitches continuous series)

Output schema (compatible with orchestrator.train()):
  timestamp  float64   Unix seconds
  open       float64   OHLC bar open
  high       float64   OHLC bar high
  low        float64   OHLC bar low
  close      float64   OHLC bar close
  price      float64   Alias for close
  volume     uint64    Bar volume
  symbol     object    Contract symbol (e.g. NQH5)

Usage:
  python -m training.dbn_to_parquet <input_path> [--output <output_path>] [--symbol-filter front-month]
"""
import os
import sys
import time
import argparse
import glob
import numpy as np
import pandas as pd
import databento as db
from typing import List, Optional, Dict, Tuple

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════
# NQ FUTURES CONTRACT CALENDAR
# ═══════════════════════════════════════════════════════════════════════

# Quarterly NQ/MNQ contracts: H (Mar), M (Jun), U (Sep), Z (Dec)
# Roll typically happens ~1 week before expiry on 3rd Friday of expiry month
NQ_ROLL_SCHEDULE = {
    # 2025 contracts
    'NQH5': {'expiry': '2025-03-21', 'roll_to': 'NQM5'},
    'NQM5': {'expiry': '2025-06-20', 'roll_to': 'NQU5'},
    'NQU5': {'expiry': '2025-09-19', 'roll_to': 'NQZ5'},
    'NQZ5': {'expiry': '2025-12-19', 'roll_to': 'NQH6'},
    # 2026 contracts
    'NQH6': {'expiry': '2026-03-20', 'roll_to': 'NQM6'},
    'NQM6': {'expiry': '2026-06-19', 'roll_to': 'NQU6'},
    'NQU6': {'expiry': '2026-09-18', 'roll_to': 'NQZ6'},
    'NQZ6': {'expiry': '2026-12-18', 'roll_to': 'NQH7'},
}


def is_outright(symbol: str) -> bool:
    """Check if symbol is an outright contract (not a spread)."""
    return '-' not in symbol


def is_front_month(symbol: str, timestamp: pd.Timestamp) -> bool:
    """
    Check if a symbol is the front-month contract at the given timestamp.
    Front-month = the nearest expiry contract that hasn't rolled yet.
    """
    if not is_outright(symbol):
        return False

    if symbol not in NQ_ROLL_SCHEDULE:
        return False

    expiry = pd.Timestamp(NQ_ROLL_SCHEDULE[symbol]['expiry'], tz='UTC')
    # Front month until ~8 days before expiry (typical roll window)
    roll_date = expiry - pd.Timedelta(days=8)
    return timestamp < roll_date


def detect_front_month_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a continuous front-month series from multi-symbol OHLCV data.

    For each timestamp, keeps only the front-month contract.
    At roll boundaries, stitches to the next contract.
    """
    if 'symbol' not in df.columns:
        return df  # Single-symbol data, nothing to filter

    symbols = df['symbol'].unique()
    outrights = [s for s in symbols if is_outright(s)]

    if len(outrights) <= 1:
        # Already single-symbol
        if outrights:
            return df[df['symbol'] == outrights[0]].copy()
        return df

    print(f"  Outright contracts found: {sorted(outrights)}")

    # Sort outrights by expiry
    known = [(s, NQ_ROLL_SCHEDULE[s]['expiry']) for s in outrights if s in NQ_ROLL_SCHEDULE]
    known.sort(key=lambda x: x[1])

    if not known:
        # Unknown contracts - just take the one with most data
        counts = df[df['symbol'].isin(outrights)].groupby('symbol').size()
        best = counts.idxmax()
        print(f"  No roll schedule for these symbols. Using {best} (most data).")
        return df[df['symbol'] == best].copy()

    # Build continuous series: for each time window, use the front-month contract
    pieces = []
    for i, (sym, expiry_str) in enumerate(known):
        expiry = pd.Timestamp(expiry_str, tz='UTC')
        roll_date = expiry - pd.Timedelta(days=8)

        sym_data = df[df['symbol'] == sym].copy()
        if sym_data.empty:
            continue

        # Ensure we have a datetime column for filtering
        if 'ts_dt' not in sym_data.columns:
            sym_data['ts_dt'] = pd.to_datetime(sym_data['timestamp'], unit='s', utc=True)

        # Keep data up to roll date
        sym_data = sym_data[sym_data['ts_dt'] < roll_date]

        # If not the first contract, start after the previous contract's roll
        if i > 0:
            prev_sym, prev_expiry_str = known[i - 1]
            prev_roll = pd.Timestamp(prev_expiry_str, tz='UTC') - pd.Timedelta(days=8)
            sym_data = sym_data[sym_data['ts_dt'] >= prev_roll]

        if not sym_data.empty:
            pieces.append(sym_data.drop(columns=['ts_dt']))
            start = sym_data['timestamp'].iloc[0]
            end = sym_data['timestamp'].iloc[-1]
            print(f"  {sym}: {len(sym_data):>10,} bars  "
                  f"({pd.Timestamp(start, unit='s').strftime('%Y-%m-%d')} to "
                  f"{pd.Timestamp(end, unit='s').strftime('%Y-%m-%d')})")

    # Handle the last contract (no roll yet)
    last_sym, last_expiry_str = known[-1]
    last_roll = pd.Timestamp(last_expiry_str, tz='UTC') - pd.Timedelta(days=8)
    last_data = df[df['symbol'] == last_sym].copy()
    if not last_data.empty:
        last_data['ts_dt'] = pd.to_datetime(last_data['timestamp'], unit='s', utc=True)
        # Keep data from roll date onwards (may overlap with above, dedup later)
        already_in = last_data['ts_dt'] >= last_roll
        extra = last_data[already_in].drop(columns=['ts_dt'])
        if not extra.empty and (not pieces or pieces[-1]['symbol'].iloc[0] != last_sym):
            pieces.append(extra)

    if not pieces:
        raise ValueError("No data after front-month filtering")

    result = pd.concat(pieces, ignore_index=True)
    result = result.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first')
    result = result.reset_index(drop=True)

    print(f"  Continuous series: {len(result):,} bars")
    return result


# ═══════════════════════════════════════════════════════════════════════
# CORE CONVERSION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def load_dbn_file(filepath: str) -> pd.DataFrame:
    """
    Load a single .dbn or .dbn.zst file and normalize to pipeline schema.

    Returns DataFrame with columns:
      timestamp, open, high, low, close, price, volume, symbol
    """
    print(f"Loading {os.path.basename(filepath)}...")
    t0 = time.time()

    store = db.DBNStore.from_file(filepath)
    df = store.to_df()

    t1 = time.time()
    print(f"  Raw: {len(df):,} rows in {t1 - t0:.1f}s")

    # Reset DatetimeIndex to column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # ── Normalize column names ──
    rename_map = {}

    # Timestamp
    if 'ts_event' in df.columns:
        rename_map['ts_event'] = 'timestamp'
    elif 'ts_recv' in df.columns:
        rename_map['ts_recv'] = 'timestamp'

    # Price columns (OHLCV data already has open/high/low/close)
    if 'size' in df.columns and 'volume' not in df.columns:
        rename_map['size'] = 'volume'

    df = df.rename(columns=rename_map)

    # Convert timestamp to float seconds
    if 'timestamp' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = df['timestamp'].astype('int64') / 1e9
        elif pd.api.types.is_integer_dtype(df['timestamp']):
            df['timestamp'] = df['timestamp'] / 1e9

    # Ensure 'price' column exists (alias for close)
    if 'price' not in df.columns and 'close' in df.columns:
        df['price'] = df['close']
    elif 'close' not in df.columns and 'price' in df.columns:
        df['close'] = df['price']

    # Ensure OHLC columns exist (derive from price/close if needed)
    if 'close' in df.columns:
        for col in ('open', 'high', 'low'):
            if col not in df.columns:
                df[col] = df['close']

    # Ensure volume
    if 'volume' not in df.columns:
        df['volume'] = 0

    # Ensure symbol
    if 'symbol' not in df.columns:
        df['symbol'] = 'UNKNOWN'

    # Filter for trade actions if present (trades data)
    if 'action' in df.columns:
        before = len(df)
        df = df[df['action'] == 'T']
        print(f"  Filtered trades: {before:,} -> {len(df):,}")

    # Select final columns
    keep = ['timestamp', 'open', 'high', 'low', 'close', 'price', 'volume', 'symbol']
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    # Sort and dedup
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def convert_file(input_path: str, output_path: str,
                 symbol_filter: str = 'front-month') -> str:
    """
    Convert a single .dbn.zst file to .parquet.

    Args:
        input_path: Path to .dbn.zst file
        output_path: Path for output .parquet file
        symbol_filter: 'front-month' (default), 'all-outrights', or 'none'

    Returns:
        Output file path
    """
    t_start = time.time()
    df = load_dbn_file(input_path)

    # Symbol filtering
    if symbol_filter == 'front-month':
        print("  Applying front-month filter...")
        df = detect_front_month_series(df)
    elif symbol_filter == 'all-outrights':
        print("  Filtering to outright contracts only...")
        df = df[df['symbol'].apply(is_outright)].reset_index(drop=True)

    # Report
    ts_min = pd.Timestamp(df['timestamp'].iloc[0], unit='s')
    ts_max = pd.Timestamp(df['timestamp'].iloc[-1], unit='s')
    n_days = (ts_max - ts_min).days
    symbols = df['symbol'].unique()

    print(f"  Final: {len(df):,} bars | {n_days} days "
          f"({ts_min.strftime('%Y-%m-%d')} to {ts_max.strftime('%Y-%m-%d')})")
    print(f"  Symbols: {list(symbols)}")
    print(f"  Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")

    # Save
    print(f"  Writing {output_path}...", end='', flush=True)
    df.to_parquet(output_path, compression='snappy')
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    elapsed = time.time() - t_start
    print(f" {size_mb:.1f} MB ({elapsed:.1f}s total)")

    return output_path


def convert_directory(input_dir: str, output_dir: str,
                      symbol_filter: str = 'none') -> List[str]:
    """
    Convert all .dbn.zst files in a directory to .parquet.

    Args:
        input_dir: Directory containing .dbn.zst files
        output_dir: Output directory for .parquet files
        symbol_filter: Filter mode

    Returns:
        List of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(input_dir, '*.dbn.zst')) +
        glob.glob(os.path.join(input_dir, '*.dbn'))
    )

    if not files:
        print(f"No .dbn/.dbn.zst files found in {input_dir}")
        return []

    print(f"Converting {len(files)} files from {input_dir} to {output_dir}\n")
    outputs = []

    for i, fp in enumerate(files, 1):
        base = os.path.basename(fp).replace('.dbn.zst', '.parquet').replace('.dbn', '.parquet')
        out_fp = os.path.join(output_dir, base)

        if os.path.exists(out_fp):
            print(f"[{i}/{len(files)}] {base} -- already exists, skipping")
            outputs.append(out_fp)
            continue

        print(f"[{i}/{len(files)}] {base}")
        try:
            convert_file(fp, out_fp, symbol_filter=symbol_filter)
            outputs.append(out_fp)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. {len(outputs)}/{len(files)} files converted.")
    return outputs


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Convert Databento .dbn.zst files to pipeline-compatible .parquet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single OHLCV file (auto front-month filter)
  python -m training.dbn_to_parquet DATA/glbx-mdp3-20250101-20260209.ohlcv-1s.dbn.zst

  # Convert with custom output path
  python -m training.dbn_to_parquet DATA/input.dbn.zst --output DATA/output.parquet

  # Convert directory of per-day trade files
  python -m training.dbn_to_parquet DATA/RAW --output DATA/Parquet

  # Keep all outrights (no front-month stitching)
  python -m training.dbn_to_parquet DATA/input.dbn.zst --symbol-filter all-outrights

  # No symbol filtering at all
  python -m training.dbn_to_parquet DATA/input.dbn.zst --symbol-filter none
        """
    )

    parser.add_argument('input', help='Input .dbn.zst file or directory')
    parser.add_argument('--output', '-o', default=None,
                        help='Output .parquet file or directory (default: same name/location)')
    parser.add_argument('--symbol-filter', choices=['front-month', 'all-outrights', 'none'],
                        default='front-month',
                        help='Symbol filtering mode (default: front-month)')

    args = parser.parse_args()
    input_path = args.input

    # Resolve relative paths
    if not os.path.isabs(input_path):
        input_path = os.path.join(PROJECT_ROOT, input_path)

    if not os.path.exists(input_path):
        print(f"ERROR: Input not found: {input_path}")
        return 1

    if os.path.isdir(input_path):
        # Directory mode
        output_dir = args.output or os.path.join(os.path.dirname(input_path), 'Parquet')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(PROJECT_ROOT, output_dir)
        convert_directory(input_path, output_dir, symbol_filter=args.symbol_filter)
    else:
        # Single file mode
        if args.output:
            output_path = args.output
            if not os.path.isabs(output_path):
                output_path = os.path.join(PROJECT_ROOT, output_path)
        else:
            output_path = input_path.replace('.dbn.zst', '.parquet').replace('.dbn', '.parquet')

        convert_file(input_path, output_path, symbol_filter=args.symbol_filter)

    return 0


if __name__ == '__main__':
    sys.exit(main())
