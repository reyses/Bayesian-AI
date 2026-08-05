#!/usr/bin/env python3
"""
QC validator for seeded_fakeout_b0.csv against ATLAS 5s parquet data.
Checks: arithmetic (poke_depth), extreme finding (poke_ext), timestamp validity, kind classification.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
CSV_PATH = Path("/media/moi/WindowsCode/Bayesian-AI/research/event_library/qc/seeded_fakeout_b0.csv")
ATLAS_DIR = Path("/media/moi/WindowsCode/Bayesian-AI/DATA/ATLAS/5s")
OUTPUT_PATH = Path("/media/moi/WindowsCode/Bayesian-AI/research/event_library/qc/seeded_result_HAIKU_v2.json")

TICK = 0.25
TOLERANCE = 0.25  # Allow within one tick

# Cache parquet files to avoid reloading
_parquet_cache = {}

def load_5s_file(day_str):
    """Load 5s parquet for a day. Format: YYYY_MM_DD -> YYYY_MM_DD.parquet"""
    if day_str in _parquet_cache:
        return _parquet_cache[day_str]

    # Files are per-day: 2024_01_04.parquet
    pq_path = ATLAS_DIR / f"{day_str}.parquet"

    if not pq_path.exists():
        return None

    df = pd.read_parquet(pq_path)
    _parquet_cache[day_str] = df
    return df

def check_2_poke_depth(row):
    """Check: poke_depth = |poke_ext - ref_px| within 0.25"""
    computed = abs(row['poke_ext'] - row['ref_px'])
    recorded = row['poke_depth']
    if abs(computed - recorded) > TOLERANCE:
        return False, f"poke_depth mismatch: computed {computed:.2f}, recorded {recorded:.2f}"
    return True, None

def check_3_poke_ext_extreme(row, df):
    """Check: poke_ext is the running extreme of CLOSES between arm_ts/ts-120s and ts"""
    if df is None or df.empty:
        return None, "No ATLAS data for day"

    # Determine start time
    start_ts = row['arm_ts'] if pd.notna(row['arm_ts']) else row['ts'] - 120
    end_ts = row['ts']
    dir_val = row['dir']

    # Get bars in the window
    window_df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]

    if window_df.empty:
        return None, "No bars in window"

    # Compute running extreme
    if dir_val == 1:
        computed_extreme = window_df['close'].max()
    else:
        computed_extreme = window_df['close'].min()

    recorded_extreme = row['poke_ext']

    if abs(computed_extreme - recorded_extreme) > TOLERANCE:
        return False, f"poke_ext mismatch: computed {computed_extreme:.2f}, recorded {recorded_extreme:.2f}"

    return True, None

def check_4_timestamp_validity(row, df):
    """Check: ts exists in day's 5s file, poke_ext's bar is within 60s before ts"""
    if df is None or df.empty:
        return None, "No ATLAS data for day"

    ts = row['ts']
    poke_ext = row['poke_ext']
    dir_val = row['dir']

    # Check if ts exists in the file
    if ts not in df['timestamp'].values:
        return False, f"ts {ts} not found in ATLAS 5s file"

    # Find bars where close matches poke_ext (within tolerance)
    matching_bars = df[(df['close'] >= poke_ext - TOLERANCE) & (df['close'] <= poke_ext + TOLERANCE)]

    if matching_bars.empty:
        return False, f"poke_ext {poke_ext:.2f} not found in file"

    # Check if any matching bar is within 60s before ts (but not at or after ts)
    matching_in_window = matching_bars[(matching_bars['timestamp'] >= ts - 60) & (matching_bars['timestamp'] < ts)]

    if matching_in_window.empty:
        return False, f"poke_ext {poke_ext:.2f} bar not found within 60s before ts"

    return True, None

def check_5_kind_classification(row, df):
    """Check: kind is RETURN if close crosses ref_px within 60s after ts, else BREAKOUT"""
    if df is None or df.empty:
        return None, "No ATLAS data for day"

    ts = row['ts']
    ref_px = row['ref_px']
    poke_ext = row['poke_ext']
    dir_val = row['dir']
    recorded_kind = row['kind']

    # Look ahead 60s after ts
    lookahead_end = ts + 60
    lookahead_df = df[(df['timestamp'] > ts) & (df['timestamp'] <= lookahead_end)]

    if lookahead_df.empty:
        # No bars in lookahead window; default to BREAKOUT
        expected_kind = "BREAKOUT"
    else:
        # Check if any close crosses ref_px in the opposite direction to poke
        crosses_back = False
        if dir_val == 1:
            # Poke is up; return if close goes back below ref_px
            crosses_back = (lookahead_df['close'] < ref_px).any()
        else:
            # Poke is down; return if close goes back above ref_px
            crosses_back = (lookahead_df['close'] > ref_px).any()

        expected_kind = "RETURN" if crosses_back else "BREAKOUT"

    if recorded_kind != expected_kind:
        return False, f"kind mismatch: expected {expected_kind}, recorded {recorded_kind}"

    return True, None

def main():
    # Load CSV
    df_csv = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df_csv)} rows from {CSV_PATH}")

    flagged = []

    for idx, row in df_csv.iterrows():
        ts = int(row['ts'])
        day = row['day']

        # Load ATLAS data for this day
        atlas_df = load_5s_file(day)

        # Check 2: arithmetic on poke_depth
        ok, msg = check_2_poke_depth(row)
        if ok is False:
            flagged.append({"ts": ts, "check": 2, "detail": msg})
            print(f"Row {idx} (ts={ts}): CHECK 2 FAIL - {msg}")

        # Check 3: poke_ext extreme
        ok, msg = check_3_poke_ext_extreme(row, atlas_df)
        if ok is False:
            flagged.append({"ts": ts, "check": 3, "detail": msg})
            print(f"Row {idx} (ts={ts}): CHECK 3 FAIL - {msg}")
        elif ok is None:
            print(f"Row {idx} (ts={ts}): CHECK 3 SKIP - {msg}")

        # Check 4: timestamp validity
        ok, msg = check_4_timestamp_validity(row, atlas_df)
        if ok is False:
            flagged.append({"ts": ts, "check": 4, "detail": msg})
            print(f"Row {idx} (ts={ts}): CHECK 4 FAIL - {msg}")
        elif ok is None:
            print(f"Row {idx} (ts={ts}): CHECK 4 SKIP - {msg}")

        # Check 5: kind classification
        ok, msg = check_5_kind_classification(row, atlas_df)
        if ok is False:
            flagged.append({"ts": ts, "check": 5, "detail": msg})
            print(f"Row {idx} (ts={ts}): CHECK 5 FAIL - {msg}")
        elif ok is None:
            print(f"Row {idx} (ts={ts}): CHECK 5 SKIP - {msg}")

    # Write results
    result = {
        "model": "haiku_tight_brief",
        "inspected": len(df_csv),
        "flagged": flagged
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults written to {OUTPUT_PATH}")
    print(f"Inspected: {len(df_csv)}, Flagged: {len(flagged)}")
    if flagged:
        print(f"Flagged ts: {[f['ts'] for f in flagged]}")

if __name__ == "__main__":
    main()
