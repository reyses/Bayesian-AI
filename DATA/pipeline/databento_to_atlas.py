"""
Databento to ATLAS — reads any Databento download folder and creates
ATLAS parquet structure.

Auto-detects the schema (trades, ohlcv-1s, ohlcv-1m, ohlcv-1h, ohlcv-1d)
from filenames and processes accordingly.

Filters to front-month MNQ contract only.
Saves as DATA/ATLAS/{tf}/YYYY_MM_DD.parquet (one file per day).

Usage:
  python tools/databento_to_atlas.py "C:/Users/reyse/OneDrive/Desktop/RAW/GLBX-20260402-DD6HDFKMA9"
  python tools/databento_to_atlas.py all     # process all folders in RAW
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import os
import sys
import glob
import gc
import re
import calendar
import numpy as np
import pandas as pd
import databento as db
from datetime import datetime, date, timedelta, timezone
from tqdm import tqdm

# Ensure repo root is importable when run as a script (python DATA/pipeline/...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core_v2.sessions import session_day_array  # canonical session-day boundary (CME 17:00 CT)

RAW_ROOT = 'C:/Users/reyse/OneDrive/Desktop/RAW'
ATLAS_OUT = 'DATA/ATLAS'

# --- Front-month roll: NT8 convention --------------------------------------
# NT8 rolls the continuous contract on a VOLUME-based schedule whose published
# date is the Monday of expiry week (the 3rd-Friday quarterly expiry minus 4
# days). We reproduce that EXACT calendar so the IS (Databento) roll matches the
# OOS (ATLAS_NT8 = what live trades) roll — date-keyed, deterministic, no
# flip-flop. The earlier calendar-MONTH->contract map silently dropped the back
# half of every quarter-expiry month: the expiring contract stops trading at its
# 3rd-Friday expiry, so demanding it all month discarded every post-expiry day
# (confirmed: 2024-03 ATLAS died exactly at 2024_03_15).
#
# Roll dates VERIFIED against the user's NT8 install (Tools > Instruments >
# NQ/MNQ > Contract months), authoritative — the "Monday of expiry week" rule
# reproduces every one exactly:
#   03-24(H4) 2023-12-11 | 06-24(M4) 2024-03-11 | 09-24(U4) 2024-06-17 |
#   12-24(Z4) 2024-09-16  (resolves the bad "09-09" web source -> it is 09-16).
# 2025 (web-confirmed, same rule): M5->U5 06-16, U5->Z5 09-15, Z5->H6 12-15.

_QCODE = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}     # CME quarterly cycle (H/M/U/Z)
_ROLL_OFFSET_DAYS = 4                           # 3rd Friday -> Monday of expiry week
# Explicit per-(year, quarter_month) overrides if NT8's published date ever
# deviates from the rule. Verify against NT8 (Tools > Instruments > NQ/MNQ >
# Contract months) before adding, e.g. (2024, 9): date(2024, 9, 16).
_ROLL_OVERRIDES = {}

# An outright MNQ contract, e.g. 'MNQH4' (root + month code + single-digit year).
# Excludes calendar spreads like 'MNQM5-MNQU5'. MNQ-specific (repo trades MNQ only).
_OUTRIGHT_RE = re.compile(r'^MNQ[FGHJKMNQUVXZ]\d$')


def _third_friday(year, month):
    weeks = calendar.monthcalendar(year, month)
    fridays = [w[calendar.FRIDAY] for w in weeks if w[calendar.FRIDAY] != 0]
    return date(year, month, fridays[2])


def _roll_date(year, month):
    """NT8 front-month roll date for a quarterly expiry = Monday of expiry week."""
    if (year, month) in _ROLL_OVERRIDES:
        return _ROLL_OVERRIDES[(year, month)]
    return _third_friday(year, month) - timedelta(days=_ROLL_OFFSET_DAYS)


def _contract_symbol(year, month):
    return f'MNQ{_QCODE[month]}{year % 10}'


def _next_quarter(year, month):
    order = [3, 6, 9, 12]
    i = order.index(month)
    return (year + 1, 3) if i == 3 else (year, order[i + 1])


def _build_roll_calendar(start_year, end_year):
    """Sorted [(roll_date, from_symbol, to_symbol)]. On roll_date(Y, qm) the front
    month switches from the expiring quarter's contract to the next quarter's."""
    events = []
    for y in range(start_year, end_year + 1):
        for qm in (3, 6, 9, 12):
            ny, nm = _next_quarter(y, qm)
            events.append((_roll_date(y, qm),
                           _contract_symbol(y, qm), _contract_symbol(ny, nm)))
    events.sort(key=lambda e: e[0])
    return events


def _front_for_day(d, events):
    """Front-month symbol for date d. Before the first roll, the front is the
    earliest event's from_symbol; otherwise the to_symbol of the latest roll <= d."""
    front = events[0][1]
    for rd, _frm, to in events:
        if rd <= d:
            front = to
        else:
            break
    return front

# Schema to TF mapping
SCHEMA_TO_TF = {
    'ohlcv-1s': '1s',
    'ohlcv-1m': '1m',
    'ohlcv-1h': '1h',
    'ohlcv-1d': '1D',
    'trades': 'trades',  # raw trades, need aggregation
}


def detect_schema(folder):
    """Auto-detect the Databento schema from filenames."""
    files = glob.glob(os.path.join(folder, '*.dbn.zst'))
    files = [f for f in files if 'condition' not in os.path.basename(f)]
    if not files:
        return None, []

    # Check first filename for schema
    fname = os.path.basename(files[0])
    for schema_key in SCHEMA_TO_TF:
        if schema_key in fname:
            return schema_key, files

    return None, files


def read_dbn(fpath):
    """Read a .dbn.zst file into DataFrame."""
    data = db.DBNStore.from_file(fpath)
    df = data.to_df()
    if len(df) == 0:
        return pd.DataFrame()

    df = df.reset_index()

    # Convert timestamp
    if 'ts_event' in df.columns:
        df['timestamp'] = df['ts_event'].astype(np.int64) // 10**9
    elif 'ts_recv' in df.columns:
        df['timestamp'] = df['ts_recv'].astype(np.int64) // 10**9

    return df


class FrontMonthSelector:
    """Selects the front-month OUTRIGHT MNQ contract per UTC day from NT8's roll
    calendar (volume-based convention; roll date = Monday of expiry week). Date-
    keyed and deterministic, so the IS roll matches the OOS (ATLAS_NT8 = live)
    roll exactly, with no flip-flop. Calendar spreads are excluded. If the
    calendar's front contract is unexpectedly absent from a day's data, it falls
    back to the highest-volume outright and flags it. Every per-day decision is
    recorded in ``self.manifest`` so the feature pipeline can mask warmup bars at
    each roll seam — adjacent contracts trade at different absolute prices
    (contango), so a trailing window straddling the seam would read a fake jump.
    """

    def __init__(self):
        self.events = None      # roll calendar (lazy)
        self.prev_front = None  # for the per-day roll flag
        self.manifest = []      # list of per-day decision dicts

    def _pick_day(self, day_df, day_str, day_date):
        if 'symbol' not in day_df.columns:
            return day_df
        sym = day_df['symbol'].astype(str)
        outright = day_df[sym.map(lambda s: bool(_OUTRIGHT_RE.match(s)))]
        if len(outright) == 0:
            return day_df.iloc[0:0]
        present = set(outright['symbol'].unique())
        front = _front_for_day(day_date, self.events)
        fallback = front not in present
        if fallback:
            # calendar front missing (data gap / unexpected) -> highest-volume outright
            if 'volume' in outright.columns:
                front = outright.groupby('symbol')['volume'].sum().idxmax()
            elif 'size' in outright.columns:                 # raw trades schema
                front = outright.groupby('symbol')['size'].sum().idxmax()
            else:
                front = outright.groupby('symbol').size().idxmax()
        rolled = self.prev_front is not None and front != self.prev_front
        self.prev_front = front
        self.manifest.append({
            'day': day_str, 'chosen': front, 'rolled': bool(rolled),
            'calendar_fallback': bool(fallback), 'n_outrights': int(len(present)),
        })
        return day_df[day_df['symbol'] == front].copy()

    def filter(self, df):
        """Per-SESSION-day front-month selection on a (possibly multi-day) frame."""
        if 'symbol' not in df.columns or len(df) == 0:
            return df
        if self.events is None:
            self.events = _build_roll_calendar(2020, 2035)
        labels = pd.Series(session_day_array(df['timestamp'].to_numpy()), index=df.index)
        parts = []
        for ds, g in df.groupby(labels, sort=True):
            d0 = datetime.strptime(ds, '%Y_%m_%d').date()
            parts.append(self._pick_day(g, ds, d0))
        return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def normalize_ohlcv(df):
    """Normalize OHLCV columns — handle Databento fixed-point if needed."""
    if len(df) == 0:
        return df

    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns and len(df) > 0 and df[col].iloc[0] > 1e6:
            df[col] = df[col] / 1e9

    # Keep only standard columns
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def aggregate_trades_to_1s(df):
    """Aggregate raw trades to 1s OHLCV bars."""
    if 'price' not in df.columns:
        return pd.DataFrame()

    df['bar_ts'] = (df['timestamp'] // 1).astype(int)  # floor to second

    bars = df.groupby('bar_ts').agg(
        timestamp=('bar_ts', 'first'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('size', 'sum') if 'size' in df.columns else ('price', 'count'),
    ).reset_index(drop=True)

    return bars


def save_daily(df, tf_label):
    """Save DataFrame to daily parquet files."""
    tf_dir = os.path.join(ATLAS_OUT, tf_label)
    os.makedirs(tf_dir, exist_ok=True)

    df['_day'] = session_day_array(df['timestamp'].to_numpy())  # session-day partition (CME 17:00 CT)

    total = 0
    for day, group in df.groupby('_day'):
        day_path = os.path.join(tf_dir, f'{day}.parquet')
        out = group.drop(columns=['_day']).sort_values('timestamp').reset_index(drop=True)

        # Merge with existing
        if os.path.exists(day_path):
            old = pd.read_parquet(day_path)
            out = pd.concat([old, out]).drop_duplicates(
                subset='timestamp', keep='last').sort_values('timestamp').reset_index(drop=True)

        out.to_parquet(day_path, index=False)
        total += len(group)

    return total


def process_folder(folder, selector):
    """Process a single Databento download folder."""
    name = os.path.basename(folder)
    schema, files = detect_schema(folder)
    files = sorted(files)  # chronological (filenames carry YYYYMMDD) for monotonic roll

    if schema is None:
        print(f'  {name}: unknown schema, skipping')
        return

    if schema == 'trades':
        print(f'  {name}: raw trades — skipping (use ohlcv-1s instead)')
        return

    tf = SCHEMA_TO_TF[schema]
    print(f'  {name}: {schema} -> {tf} ({len(files)} files)')

    if len(files) == 1 and tf in ('1s', 'trades'):
        # Single large file — process in one go
        print(f'    Reading single file...')
        df = read_dbn(files[0])
        df = selector.filter(df)

        if tf == 'trades':
            df = aggregate_trades_to_1s(df)
            tf = '1s'
        else:
            df = normalize_ohlcv(df)

        n = save_daily(df, tf)
        print(f'    Saved {n:,} bars to {tf}/')
        del df; gc.collect()

    else:
        # Daily files — process one at a time
        all_bars = []
        for fpath in tqdm(files, desc=f'    {tf}'):
            df = read_dbn(fpath)
            if len(df) == 0:
                continue
            df = selector.filter(df)
            if len(df) == 0:
                continue

            if tf == 'trades':
                df = aggregate_trades_to_1s(df)
            else:
                df = normalize_ohlcv(df)

            all_bars.append(df)

        if all_bars:
            combined = pd.concat(all_bars, ignore_index=True)
            combined = combined.drop_duplicates(subset='timestamp', keep='last').sort_values('timestamp')
            actual_tf = '1s' if schema == 'trades' else tf
            n = save_daily(combined, actual_tf)
            print(f'    Saved {n:,} bars to {actual_tf}/')
            del combined; gc.collect()


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'all'

    print(f'DATABENTO TO ATLAS')
    print(f'  Output: {ATLAS_OUT}/{{tf}}/YYYY_MM_DD.parquet')
    print()

    if target == 'all':
        folders = sorted(glob.glob(os.path.join(RAW_ROOT, 'GLBX-*')))
        folders = [f for f in folders if os.path.isdir(f)]
    else:
        folders = [target]

    selector = FrontMonthSelector()
    for folder in folders:
        process_folder(folder, selector)
        gc.collect()

    # Roll manifest — per-day contract selection + roll seams (for downstream seam masking)
    if selector.manifest:
        man_path = os.path.join(ATLAS_OUT, 'roll_manifest.csv')
        man = pd.DataFrame(selector.manifest)
        if os.path.exists(man_path):
            old = pd.read_csv(man_path)
            man = pd.concat([old, man]).drop_duplicates(subset='day', keep='last')
        man = man.sort_values('day').reset_index(drop=True)
        man.to_csv(man_path, index=False)
        n_rolls = int(man['rolled'].sum()) if 'rolled' in man.columns else 0
        print(f'\nRoll manifest: {len(man)} days, {n_rolls} roll seams -> {man_path}')

    # Summary
    print(f'\nATLAS SUMMARY:')
    for tf in ['1s', '1m', '1h', '1D']:
        tf_dir = os.path.join(ATLAS_OUT, tf)
        if os.path.exists(tf_dir):
            files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
            if files:
                total = sum(len(pd.read_parquet(f)) for f in files[:3])  # sample first 3
                avg = total // min(3, len(files))
                est_total = avg * len(files)
                print(f'  {tf:>4}: {len(files)} days, ~{est_total:,} bars')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
