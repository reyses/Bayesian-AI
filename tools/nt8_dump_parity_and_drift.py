"""
nt8_dump_parity_and_drift.py -- Validate ATLAS_NT8 CSV dump + measure daily drift.

Three jobs in one tool:
  1. Aggregate 1s -> synthetic 1m, compare to actual 1m (parity check).
  2. Aggregate 1m -> daily OHLC for the full date range.
  3. Compute the buy-and-hold daily drift (= close - open per day, $/contract).

Why: if always-long strategy makes +$X/day and buy-and-hold is also +$X/day,
the strategy has zero edge over the regime. Strategy must beat the drift to
be deployable.

Usage:
    python tools/nt8_dump_parity_and_drift.py
    python tools/nt8_dump_parity_and_drift.py --contract MNQ_06-26
    python tools/nt8_dump_parity_and_drift.py --start 2026-04-20 --end 2026-04-26
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ATLAS_NT8_ROOT = "DATA/ATLAS_NT8"
PT_VALUE_USD   = 2.0   # MNQ: $2 per point per contract


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # First column may have BOM (﻿timestamp); rename
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    return df


def _list_days(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".csv"):
            continue
        out.append(f[:-4])  # strip .csv
    return out


def _aggregate_1s_to_1m(df_1s: pd.DataFrame) -> pd.DataFrame:
    """Bucket 1s bars into 60-second buckets ending at the same boundaries
    NT8 uses for 1m bars. NT8 1m bar at timestamp T covers (T-60, T] in 1s."""
    df = df_1s.copy()
    # Bucket each 1s bar by 60-second ceil
    df["minute_end"] = ((df["timestamp"] // 60) + 1) * 60 - 1
    grouped = df.groupby("minute_end", sort=True).agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",     "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index().rename(columns={"minute_end": "timestamp"})
    return grouped


def _parity_check(syn_1m: pd.DataFrame, act_1m: pd.DataFrame, day: str) -> dict:
    """Compare synthetic vs actual 1m bars."""
    merged = pd.merge(
        syn_1m, act_1m,
        on="timestamp", how="outer",
        suffixes=("_syn", "_act"),
        indicator=True,
    )
    n_total       = len(merged)
    n_match       = (merged["_merge"] == "both").sum()
    n_syn_only    = (merged["_merge"] == "left_only").sum()
    n_act_only    = (merged["_merge"] == "right_only").sum()

    # On matched rows, check OHLC equality
    both = merged[merged["_merge"] == "both"]
    ohlc_diff = {}
    for col in ["open", "high", "low", "close"]:
        diff = (both[f"{col}_syn"] - both[f"{col}_act"]).abs()
        ohlc_diff[col] = {
            "max":   float(diff.max())   if len(diff) else 0.0,
            "mean":  float(diff.mean())  if len(diff) else 0.0,
            "n_eq": int((diff == 0).sum()) if len(diff) else 0,
        }
    return {
        "day":           day,
        "n_act_bars":    len(act_1m),
        "n_syn_bars":    len(syn_1m),
        "n_match":       int(n_match),
        "n_syn_only":    int(n_syn_only),
        "n_act_only":    int(n_act_only),
        "ohlc_diff":     ohlc_diff,
    }


def _aggregate_1m_to_1d(df_1m: pd.DataFrame, day_label: str) -> dict:
    """Single-day 1m -> 1D summary. Returns OHLC and drift metrics."""
    df = df_1m.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return None
    return {
        "day":          day_label,
        "open":         float(df["open"].iloc[0]),
        "high":         float(df["high"].max()),
        "low":          float(df["low"].min()),
        "close":        float(df["close"].iloc[-1]),
        "volume":       int(df["volume"].sum()),
        "n_bars":       len(df),
        "open_to_close_pts":  float(df["close"].iloc[-1] - df["open"].iloc[0]),
        "high_to_low_pts":    float(df["high"].max() - df["low"].min()),
        "first_bar_ts": int(df["timestamp"].iloc[0]),
        "last_bar_ts":  int(df["timestamp"].iloc[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="MNQ_06-26")
    ap.add_argument("--start",    default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end",      default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--atlas",    default=ATLAS_NT8_ROOT)
    args = ap.parse_args()

    folder_1s = os.path.join(args.atlas, "1s", args.contract)
    folder_1m = os.path.join(args.atlas, "1m", args.contract)

    days_1s = set(_list_days(folder_1s))
    days_1m = set(_list_days(folder_1m))
    common  = sorted(days_1s & days_1m)

    if args.start:
        s = args.start.replace("-", "_")
        common = [d for d in common if d >= s]
    if args.end:
        e = args.end.replace("-", "_")
        common = [d for d in common if d <= e]

    print(f"Contract: {args.contract}")
    print(f"1s days available: {len(days_1s)}")
    print(f"1m days available: {len(days_1m)}")
    print(f"Both available + filtered: {len(common)}")
    if not common:
        print("No days in scope — nothing to do.")
        return
    print(f"Range: {common[0]} -> {common[-1]}")
    print()

    parity_rows = []
    daily_rows  = []

    for day in common:
        try:
            df_1s = _load_csv(os.path.join(folder_1s, f"{day}.csv"))
        except (pd.errors.EmptyDataError, FileNotFoundError):
            df_1s = pd.DataFrame()
        try:
            df_1m_actual = _load_csv(os.path.join(folder_1m, f"{day}.csv"))
        except (pd.errors.EmptyDataError, FileNotFoundError):
            df_1m_actual = pd.DataFrame()

        if df_1s.empty:
            continue

        # Synthesize 1m from 1s
        syn_1m = _aggregate_1s_to_1m(df_1s)

        # 1) Parity (only if both populated)
        if not df_1m_actual.empty and len(df_1m_actual) > 1:
            parity = _parity_check(syn_1m, df_1m_actual, day)
            parity_rows.append(parity)
        else:
            parity_rows.append({
                "day":           day,
                "n_act_bars":    len(df_1m_actual),
                "n_syn_bars":    len(syn_1m),
                "n_match":       0,
                "n_syn_only":    len(syn_1m),
                "n_act_only":    0,
                "ohlc_diff":     {c: {"max": float("nan"), "mean": float("nan"), "n_eq": 0}
                                  for c in ["open","high","low","close"]},
            })

        # 2) Daily aggregation — use synthesized 1m (since actual 1m is broken)
        daily = _aggregate_1m_to_1d(syn_1m, day)
        if daily is not None:
            daily_rows.append(daily)

    # ── Parity report ──────────────────────────────────────────────────────
    print("=" * 72)
    print("PARITY:  1s aggregated -> 1m   vs   actual 1m")
    print("=" * 72)
    print(f"{'day':<14} {'act':>5} {'syn':>5} {'match':>6} {'O-eq':>5} {'H-eq':>5} {'L-eq':>5} {'C-eq':>5} {'maxClose-d':>11}")
    n_total_match = 0
    n_total_act   = 0
    for p in parity_rows:
        n_total_match += p["n_match"]
        n_total_act   += p["n_act_bars"]
        print(f"{p['day']:<14} {p['n_act_bars']:>5} {p['n_syn_bars']:>5} {p['n_match']:>6} "
              f"{p['ohlc_diff']['open']['n_eq']:>5} {p['ohlc_diff']['high']['n_eq']:>5} "
              f"{p['ohlc_diff']['low']['n_eq']:>5} {p['ohlc_diff']['close']['n_eq']:>5} "
              f"{p['ohlc_diff']['close']['max']:>11.4f}")
    pct = 100.0 * n_total_match / n_total_act if n_total_act else 0.0
    print(f"\nTotal matched 1m bars: {n_total_match} / {n_total_act}  ({pct:.2f}%)")
    print()

    # ── Daily drift report ─────────────────────────────────────────────────
    daily_df = pd.DataFrame(daily_rows)
    daily_df = daily_df.sort_values("day").reset_index(drop=True)

    daily_df["drift_usd"]      = daily_df["open_to_close_pts"] * PT_VALUE_USD
    daily_df["range_usd"]      = daily_df["high_to_low_pts"]   * PT_VALUE_USD
    daily_df["dow"]            = pd.to_datetime(daily_df["day"], format="%Y_%m_%d").dt.dayofweek

    print("=" * 72)
    print("DAILY DRIFT (buy-and-hold from 1st bar open to last bar close)")
    print("=" * 72)
    print(f"{'day':<14} {'open':>10} {'close':>10} {'drift_$':>10} {'range_$':>10} {'bars':>5}")
    for _, r in daily_df.iterrows():
        print(f"{r['day']:<14} {r['open']:>10.2f} {r['close']:>10.2f} "
              f"{r['drift_usd']:>+10.2f} {r['range_usd']:>10.2f} {r['n_bars']:>5}")
    print()

    n_days = len(daily_df)
    total_drift = daily_df["drift_usd"].sum()
    mean_drift  = daily_df["drift_usd"].mean()
    median      = daily_df["drift_usd"].median()
    pos_days    = (daily_df["drift_usd"] > 0).sum()
    print("=" * 72)
    print("REGIME BASELINE  (= what a perfect 'always long, hold all session' yields)")
    print("=" * 72)
    print(f"  N days          : {n_days}")
    print(f"  Total drift     : ${total_drift:+,.2f}  (across full window)")
    print(f"  Mean drift/day  : ${mean_drift:+,.2f}")
    print(f"  Median drift/day: ${median:+,.2f}")
    print(f"  Up days         : {pos_days}/{n_days}  ({100.0*pos_days/n_days:.1f}%)")
    print(f"  Best  day       : ${daily_df['drift_usd'].max():+,.2f}")
    print(f"  Worst day       : ${daily_df['drift_usd'].min():+,.2f}")
    print(f"  Mean day range  : ${daily_df['range_usd'].mean():,.2f}")
    print()

    # ── Save outputs ───────────────────────────────────────────────────────
    out_dir = "reports/findings/nt8_dump_validation"
    os.makedirs(out_dir, exist_ok=True)
    daily_df.to_csv(os.path.join(out_dir, f"daily_{args.contract}.csv"), index=False)
    pd.DataFrame([{
        "day": p["day"],
        "n_act": p["n_act_bars"], "n_syn": p["n_syn_bars"], "n_match": p["n_match"],
        "open_max_diff": p["ohlc_diff"]["open"]["max"],
        "close_max_diff": p["ohlc_diff"]["close"]["max"],
        "high_max_diff": p["ohlc_diff"]["high"]["max"],
        "low_max_diff":  p["ohlc_diff"]["low"]["max"],
    } for p in parity_rows]).to_csv(os.path.join(out_dir, f"parity_{args.contract}.csv"), index=False)
    print(f"Saved: {out_dir}/daily_{args.contract}.csv")
    print(f"Saved: {out_dir}/parity_{args.contract}.csv")


if __name__ == "__main__":
    main()
