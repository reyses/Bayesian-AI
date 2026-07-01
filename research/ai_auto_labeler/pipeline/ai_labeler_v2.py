"""AI Auto-Labeler v2 — structural launch labeling (Moises spec).

Fixes v1's fixed-60-bar prominence. Logic:
  1. cubic (N=20, centered) on 1m -> turns (tops/bottoms) + slope + curvature.
  2. ENTRY: flat-zone best bar. The flat zone = the contiguous span where the smoothed price stays
     within FLAT_BAND of the turn (curvature-defined broad hump). Snap the entry to the real 1s
     extreme in that span (lowest low for LONG / highest high for SHORT) = 0-MAE entry.
  3. CONFIRM: walk forward from the entry, NO 60-bar cap; skip sub-TREND_PTS wiggles; continue until a
     >= TREND_PTS favorable move confirms the trend. If price breaks back past the entry (adverse >
     REVERSAL_TOL) before +TREND_PTS -> FLAG the region for inspection (should not happen if the entry
     is the true extreme). If the session ends first -> no trade.
  4. EXIT: mirror of entry. The trend ends when it retraces >= TREND_PTS from its running peak (a real
     opposing move) or the session ends. Snap the exit to the 1s best bar in that peak's flat zone =
     best exit (max MFE). Bounded by the trend's real extent, not a fixed window.

Outputs: DATA/ai_cusp_picks/ai_picks_<date>_multi.json  (trades)
         DATA/ai_cusp_picks/flagged/<date>_flagged.json  (reversal regions for human inspection)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "viz", "core"))
from cubic_utils import find_raw_turns  # noqa: E402

ONE_M = os.path.join(ROOT, "DATA", "ATLAS", "1m")
ONE_S = os.path.join(ROOT, "DATA", "ATLAS", "1s")
OUT = os.path.join(ROOT, "DATA", "ai_cusp_picks")
FLAG = os.path.join(OUT, "flagged")

CUBIC_N = 20            # centered cubic window (matches the marker / Figure_4)
TREND_PTS = 7.0         # a launch must follow through >= this (structural; regime-adaptive later)
FLAT_BAND_PTS = 3.0     # flat zone = smoothed price stays within this of the turn extreme
REVERSAL_TOL_PTS = 1.0  # adverse past entry beyond this before +TREND_PTS -> flag for inspection
TICK = 0.25


def flat_span(smooth, i, n):
    """Curvature-defined flat zone: expand around turn i while smoothed price stays within FLAT_BAND."""
    lo = hi = i
    while lo > 0 and abs(smooth[lo - 1] - smooth[i]) <= FLAT_BAND_PTS:
        lo -= 1
    while hi < n - 1 and abs(smooth[hi + 1] - smooth[i]) <= FLAT_BAND_PTS:
        hi += 1
    return lo, hi


def best_bar_1s(df1s, ts0, ts1, direction):
    """Real extreme in [ts0, ts1] on 1s: lowest low (LONG) / highest high (SHORT)."""
    m = (df1s["timestamp"].values >= ts0) & (df1s["timestamp"].values <= ts1)
    if not m.any():
        return None, None
    sub = df1s.iloc[m]
    if direction == "LONG":
        k = sub["low"].idxmin(); return float(sub.loc[k, "low"]), float(sub.loc[k, "timestamp"])
    k = sub["high"].idxmax(); return float(sub.loc[k, "high"]), float(sub.loc[k, "timestamp"])


def walk_trend(hi, lo, entry_price, i, direction, end):
    """Forward walk. Returns (status, exit_i, mfe_pts, mae_pts). status: confirmed|reversal|no_trend."""
    confirmed = False
    run_max, run_i, max_adv = 0.0, i, 0.0
    for j in range(i + 1, end):
        if direction == "LONG":
            fav = hi[j] - entry_price; adv = entry_price - lo[j]
        else:
            fav = entry_price - lo[j]; adv = hi[j] - entry_price
        max_adv = max(max_adv, adv)
        if not confirmed:
            if adv > REVERSAL_TOL_PTS:
                return "reversal", j, 0.0, adv
            if fav >= TREND_PTS:
                confirmed = True
        if fav > run_max:
            run_max, run_i = fav, j
        if confirmed and run_max - fav >= TREND_PTS:      # retraced >= TREND_PTS from peak -> trend ended
            return "confirmed", run_i, run_max, max_adv
    return ("confirmed", run_i, run_max, max_adv) if confirmed else ("no_trend", i, 0.0, max_adv)


def process_day(date_key, cache):
    df1m = pd.read_parquet(os.path.join(ONE_M, f"{date_key}.parquet"))
    close = df1m["close"].values.astype(float)
    ts1m = df1m["timestamp"].values.astype(float)
    hi1m = df1m["high"].values.astype(float)
    lo1m = df1m["low"].values.astype(float)
    n = len(close)
    if n < 100:
        return [], []
    turns, smooth, slope, curv = find_raw_turns(close, CUBIC_N)

    # continuous 1s (prev/curr/next) for extreme snapping across sessions
    dt = pd.Timestamp(date_key.replace("_", "-"))
    dfs = []
    for k in [(dt - pd.Timedelta(days=1)), dt, (dt + pd.Timedelta(days=1))]:
        kk = k.strftime("%Y_%m_%d")
        if kk not in cache:
            p = os.path.join(ONE_S, f"{kk}.parquet")
            cache[kk] = pd.read_parquet(p) if os.path.exists(p) else None
        if cache[kk] is not None:
            dfs.append(cache[kk])
    if not dfs:
        return [], []
    df1s = pd.concat(dfs).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    trades, flags, consumed_ts = [], [], -1.0   # de-dup by EXIT TIME (no temporal overlap)
    for tn in turns:                             # turns are in index (time) order
        i = tn["index"]
        if i < CUBIC_N or i >= n - CUBIC_N or ts1m[i] < consumed_ts:
            continue
        direction = "SHORT" if tn["type"] == "top" else "LONG"
        # ENTRY: flat-zone best bar (1s)
        a, b = flat_span(smooth, i, n)
        entry_price, entry_ts = best_bar_1s(df1s, ts1m[a], ts1m[b] + 60, direction)
        if entry_price is None or entry_ts < consumed_ts:   # snapped-back entry would overlap prior trade
            continue
        # CONFIRM + EXIT region (1m structural walk)
        status, exit_i, mfe, mae = walk_trend(hi1m, lo1m, entry_price, i, direction, n)
        if status == "reversal":
            flags.append({"date": date_key, "turn_ts": float(ts1m[i]), "direction": direction,
                          "entry_price": entry_price, "adverse_pts": round(mae, 2),
                          "reason": "broke past entry before +TREND_PTS"})
            continue
        if status == "no_trend" or mfe < TREND_PTS:
            continue
        # EXIT: flat-zone best bar around the peak (1s)
        ea, eb = flat_span(smooth, exit_i, n)
        exit_price, exit_ts = best_bar_1s(df1s, ts1m[ea], ts1m[eb] + 60, "SHORT" if direction == "LONG" else "LONG")
        if exit_price is None:
            continue
        if exit_ts <= entry_ts:                  # safety: exit must be after entry
            continue
        pnl = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
        trades.append({"entry_ts": entry_ts, "exit_ts": exit_ts, "direction": direction,
                       "side": "Buy" if direction == "LONG" else "Sell",
                       "entry_price": entry_price, "exit_price": exit_price,
                       "pnl_dollars": round(pnl / TICK * 0.50, 2), "mae_dollars": round(mae / TICK * 0.50, 2),
                       "original_timestamp": float(ts1m[i])})
        consumed_ts = exit_ts                    # next trade cannot start before this exit
    return trades, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="YYYY_MM_DD (single day)")
    ap.add_argument("--month", help="YYYY_MM (batch)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(FLAG, exist_ok=True)
    days = ([a.day] if a.day else
            [os.path.basename(f)[:-8] for f in sorted(glob.glob(os.path.join(ONE_M, f"{a.month}_*.parquet")))])
    cache = {}
    tot_t = tot_f = 0
    for dk in days:
        try:
            trades, flags = process_day(dk, cache)
        except Exception as e:
            print(f"{dk}: ERROR {e}"); continue
        if trades:
            json.dump({"trades": trades}, open(os.path.join(OUT, f"ai_picks_{dk.replace('_','-')}_multi.json"), "w"), indent=2)
        if flags:
            json.dump({"flagged": flags}, open(os.path.join(FLAG, f"{dk}_flagged.json"), "w"), indent=2)
        tot_t += len(trades); tot_f += len(flags)
        print(f"{dk}: {len(trades)} trades, {len(flags)} flagged")
    print(f"\nTOTAL: {tot_t} trades, {tot_f} flagged for inspection")
    if tot_t:
        print("(v2: flat-zone entry/exit, uncapped forward walk, reversals -> flagged/)")


if __name__ == "__main__":
    main()
