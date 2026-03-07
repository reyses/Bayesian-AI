"""
Convert NinjaTrader 8 exported tick data to ATLAS-compatible parquet files.

Generates ALL timeframes needed by the trainer:
  1s, 5s, 15s, 30s, 1m, 2m, 3m, 5m, 15m, 30m, 1h, 4h, 1D, 1W

Usage:
    python tools/nt8_to_parquet.py <file> [--out DATA/ATLAS_OOS]

Supports:
  - Tick/Last data (no header): YYYYMMDD HHMMSS microseconds;last;bid;ask;volume
  - OHLCV bar data (with header): Date;Time;Open;High;Low;Close;Volume
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# All timeframes the trainer expects, with their bar size in seconds
TIMEFRAMES = [
    ('1s',    1),
    ('5s',    5),
    ('15s',   15),
    ('30s',   30),
    ('1m',    60),
    ('2m',    120),
    ('3m',    180),
    ('5m',    300),
    ('15m',   900),
    ('30m',   1800),
    ('1h',    3600),
    ('4h',    14400),
    ('1D',    86400),
    ('1W',    604800),
]


def detect_format(file_path: str) -> str:
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    if first_line[:8].isdigit() and len(first_line.split(';')) >= 4:
        return 'tick'
    return 'bar'


def parse_ticks_to_1s(file_path: str, chunk_size: int = 5_000_000) -> pd.DataFrame:
    """Parse tick file in chunks, aggregate to 1s bars."""
    total_lines = 0
    with open(file_path, 'r') as f:
        for _ in f:
            total_lines += 1
    print(f"  Total ticks: {total_lines:,}")

    all_bars = []
    buffer_ticks = []

    with open(file_path, 'r') as f:
        pbar = tqdm(total=total_lines, desc="  Reading ticks", unit=" ticks",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        while True:
            lines = []
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())

            if not lines:
                break

            pbar.update(len(lines))

            records = buffer_ticks
            buffer_ticks = []

            for line in lines:
                parts = line.split(';')
                if len(parts) < 4:
                    continue
                dt_parts = parts[0].split(' ')
                if len(dt_parts) < 2:
                    continue

                date_str = dt_parts[0]
                time_str = dt_parts[1]
                price = float(parts[1])
                volume = int(parts[4]) if len(parts) > 4 else 1

                y = int(date_str[:4])
                mo = int(date_str[4:6])
                d = int(date_str[6:8])
                h = int(time_str[:2])
                mi = int(time_str[2:4])
                s = int(time_str[4:6])

                epoch = int(datetime(y, mo, d, h, mi, s).timestamp())
                records.append((epoch, price, volume))

            if not records:
                continue

            df_chunk = pd.DataFrame(records, columns=['bar_ts', 'price', 'volume'])

            # Buffer last second (might be incomplete across chunks)
            last_ts = df_chunk['bar_ts'].iloc[-1]
            if len(lines) == chunk_size:
                buffer_mask = df_chunk['bar_ts'] == last_ts
                buffer_ticks = list(df_chunk[buffer_mask].itertuples(index=False, name=None))
                df_chunk = df_chunk[~buffer_mask]

            if len(df_chunk) == 0:
                continue

            bars = df_chunk.groupby('bar_ts').agg(
                open=('price', 'first'),
                high=('price', 'max'),
                low=('price', 'min'),
                close=('price', 'last'),
                volume=('volume', 'sum')
            ).reset_index()
            all_bars.append(bars)

        pbar.close()

    # Remaining buffer
    if buffer_ticks:
        df_buf = pd.DataFrame(buffer_ticks, columns=['bar_ts', 'price', 'volume'])
        bars = df_buf.groupby('bar_ts').agg(
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('volume', 'sum')
        ).reset_index()
        all_bars.append(bars)

    print("  Concatenating bar chunks...")
    result = pd.concat(all_bars, ignore_index=True)

    # Merge bars spanning chunk boundaries
    result = result.groupby('bar_ts').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index()

    result = result.sort_values('bar_ts').reset_index(drop=True)
    print(f"  1s bars: {len(result):,}")
    return result


def resample_from_1s(df_1s: pd.DataFrame, bar_seconds: int) -> pd.DataFrame:
    """Resample 1s bars to coarser timeframe."""
    df = df_1s.copy()
    df['target_ts'] = df['bar_ts'] - (df['bar_ts'] % bar_seconds)

    result = df.groupby('target_ts').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index().rename(columns={'target_ts': 'bar_ts'})

    return result


def to_atlas_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to final ATLAS schema."""
    out = pd.DataFrame()
    out['timestamp'] = df['bar_ts'].astype(np.int64)
    out['open'] = df['open'].astype(np.float64)
    out['high'] = df['high'].astype(np.float64)
    out['low'] = df['low'].astype(np.float64)
    out['close'] = df['close'].astype(np.float64)
    out['volume'] = df['volume'].astype(np.float64)
    return out


def write_monthly_parquets(df: pd.DataFrame, out_dir: Path, tf_label: str):
    """Split into monthly files, merge with existing."""
    out_path = out_dir / tf_label
    out_path.mkdir(parents=True, exist_ok=True)

    timestamps = pd.to_datetime(df['timestamp'], unit='s')
    df = df.copy()
    df['_ym'] = timestamps.dt.strftime('%Y_%m')

    for ym, group in df.groupby('_ym'):
        fname = out_path / f"{ym}.parquet"
        new_data = group.drop(columns=['_ym'])

        if fname.exists():
            existing = pd.read_parquet(fname)
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
            n_new = len(combined) - len(existing)
            print(f"    {fname.name}: {len(existing):,} existing + {n_new:,} new -> {len(combined):,} merged")
        else:
            combined = new_data.sort_values('timestamp').reset_index(drop=True)
            print(f"    {fname.name}: {len(combined):,} bars (new file)")

        combined.to_parquet(fname, index=False)


def main():
    parser = argparse.ArgumentParser(description='Convert NT8 export to ATLAS parquet (all TFs)')
    parser.add_argument('file', help='Path to NT8 exported file')
    parser.add_argument('--out', default='DATA/ATLAS_OOS', help='Output directory (default: DATA/ATLAS_OOS)')
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    out_dir = Path(args.out)
    fmt = detect_format(str(file_path))
    print(f"Converting NT8 export: {file_path}")
    print(f"  Detected format: {fmt}")

    if fmt == 'tick':
        df_1s = parse_ticks_to_1s(str(file_path))
    else:
        # Bar data — read and treat as finest available
        print("  Bar format detected — reading as-is")
        raise NotImplementedError("Bar format: use tick export for all-TF generation")

    # Generate all timeframes from 1s
    print(f"\nGenerating {len(TIMEFRAMES)} timeframes...")
    for tf_label, tf_seconds in tqdm(TIMEFRAMES, desc="  Timeframes"):
        if tf_seconds == 1:
            atlas_df = to_atlas_df(df_1s)
        else:
            resampled = resample_from_1s(df_1s, tf_seconds)
            atlas_df = to_atlas_df(resampled)

        print(f"  {tf_label:>4s}: {len(atlas_df):>8,} bars")
        write_monthly_parquets(atlas_df, out_dir, tf_label)

    ts_min = pd.to_datetime(df_1s['bar_ts'].min(), unit='s')
    ts_max = pd.to_datetime(df_1s['bar_ts'].max(), unit='s')
    print(f"\nDone! All timeframes written to {out_dir}/")
    print(f"  Date range: {ts_min} -> {ts_max}")


if __name__ == '__main__':
    main()
