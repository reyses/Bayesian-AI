"""AI Auto-Labeler v2 — structural launch labeling (Moises spec).

Fixes v1's fixed-60-bar prominence AND the retrace-exit. Logic (Moises spec):
  1. cubic (N=20, centered) on 1m -> turns (tops/bottoms) + slope + curvature.
  2. SEGMENT: zigzag_turns() collapses the cubic turns into SIGNIFICANT alternating pivots — a swing
     < TREND_PTS on the smoothed cubic is a WIGGLE (absorbed); a run continues until the cubic actually
     reverses >= TREND_PTS. Each leg (pivot -> next opposite pivot) = ONE trade. TREND_PTS only filters
     wiggles; the CUBIC decides where a run ends, NOT a price retrace.
  3. ENTRY: flat-zone best bar at the leg's START turn. Flat zone = the contiguous span where the
     smoothed price stays within FLAT_BAND of the turn (broad hump). Snap to the real 1s extreme in that
     span (lowest low for LONG / highest high for SHORT) = 0-MAE entry.
  4. EXIT: flat-zone best bar at the leg's END turn = the cubic's ACTUAL direction change (held through
     intra-leg wiggles). If price broke past the entry inside the leg -> FLAG for inspection (kept anyway).

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
from amplitude_scale import scale_for_day # noqa: E402

ONE_M = os.path.join(ROOT, "DATA", "ATLAS", "1m")
ONE_S = os.path.join(ROOT, "DATA", "ATLAS", "1s")
OUT = os.path.join(ROOT, "DATA", "ai_cusp_picks")
FLAG = os.path.join(OUT, "flagged")

CUBIC_N = 20            # centered cubic window (matches the marker / Figure_4)

# Fixed (baseline) defaults
TREND_PTS = 4.0         # fixed-T mode: cubic-swing wiggle filter
FLAT_BAND_PTS = 3.0     # fixed-T mode: flat zone = smoothed price stays within this of the turn extreme
REVERSAL_TOL_PTS = 1.0  # fixed-T mode: adverse past entry beyond this before +TREND_PTS -> flag for inspection

# Adaptive defaults
ADAPTIVE = True
K_TREND = 0.60          # multiplier for the local amplitude scale
AMP_MODE = "w60"        # mode for amplitude scale (w30, w60, day, day_c5, etc)
TREND_MIN_PTS = 2.5
TREND_MAX_PTS = 12.0
FLAT_BAND_RATIO = 0.75  # flat_band = RATIO * local_trend_pts
REVERSAL_TOL_RATIO = 0.25 # reversal_tol = RATIO * local_trend_pts
TICK = 0.25

def flat_span(smooth, i, n, band):
    """Curvature-defined flat zone: expand around turn i while smoothed price stays within band."""
    lo = hi = i
    while lo > 0 and abs(smooth[lo - 1] - smooth[i]) <= band:
        lo -= 1
    while hi < n - 1 and abs(smooth[hi + 1] - smooth[i]) <= band:
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


def zigzag_turns(smooth, turns, thr_arr):
    """Segment the cubic into SIGNIFICANT alternating pivots. A swing < thr on the smoothed cubic is a
    WIGGLE and is absorbed (does NOT end a run). The run continues until the cubic actually turns by
    >= thr in the opposite direction. Returns [[index, type, value], ...] alternating top/bottom.
    (Moises: the cubic decides continuation; 7pt only filters wiggles, it is NOT a retrace-exit.)"""
    if not turns:
        return []
    piv = []
    hi_i = lo_i = turns[0]["index"]
    hi_v = lo_v = float(smooth[turns[0]["index"]])
    direction = 0                                          # 0 unknown, +1 up-leg, -1 down-leg
    for tn in turns[1:]:
        i = tn["index"]; v = float(smooth[i])
        if v > hi_v:
            hi_i, hi_v = i, v
        if v < lo_v:
            lo_i, lo_v = i, v
        if direction >= 0 and hi_v - v >= thr_arr[i]:             # reversed DOWN >= thr from the high -> a top
            piv.append([hi_i, "top", hi_v]); direction = -1; lo_i, lo_v = i, v
        elif direction <= 0 and v - lo_v >= thr_arr[i]:           # reversed UP >= thr from the low -> a bottom
            piv.append([lo_i, "bottom", lo_v]); direction = 1; hi_i, hi_v = i, v
    return piv


def process_day(date_key, cache, fixed=False):
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

    # Set up threshold arrays
    if ADAPTIVE and not fixed:
        scale_arr = scale_for_day(date_key, close, AMP_MODE, ONE_M, cache)
        thr_arr = np.clip(K_TREND * scale_arr, TREND_MIN_PTS, TREND_MAX_PTS)
    else:
        thr_arr = np.full(n, TREND_PTS)

    # Segment the cubic into SIGNIFICANT legs; each leg (turn -> next opposite turn) = ONE trade.
    piv = zigzag_turns(smooth, turns, thr_arr)
    
    changed = True
    while changed and len(piv) >= 2:
        changed = False
        trades, flags = [], []
        consumed_ts = -1.0
        
        for k in range(len(piv) - 1):
            i0, ty0, _ = piv[k]
            i1, ty1, _ = piv[k + 1]
            if i0 < CUBIC_N // 2 or i1 >= n - CUBIC_N // 2:
                continue
                
            direction = "LONG" if ty0 == "bottom" else "SHORT"
            if fixed or not ADAPTIVE:
                entry_band = FLAT_BAND_PTS
                exit_band = FLAT_BAND_PTS
                min_pnl = TREND_PTS
                max_mae = REVERSAL_TOL_PTS
            else:
                entry_band = FLAT_BAND_RATIO * thr_arr[i0]
                exit_band = FLAT_BAND_RATIO * thr_arr[i1]
                min_pnl = thr_arr[i0]
                max_mae = REVERSAL_TOL_RATIO * thr_arr[i0]

            # ENTRY: flat-zone best bar at the leg's START turn (0-MAE)
            a, b = flat_span(smooth, i0, n, entry_band)
            entry_price, entry_ts = best_bar_1s(df1s, ts1m[a], ts1m[b] + 60, direction)
            if entry_price is None:
                piv.pop(k+1)
                piv.pop(k)
                changed = True
                break
                
            is_marginal = False
            
            # OPTION A: Force entry_ts to be at least consumed_ts + 1 if they overlap
            if entry_ts <= consumed_ts:
                m = (df1s["timestamp"].values > consumed_ts) & (df1s["timestamp"].values <= max(ts1m[b] + 60, consumed_ts + 300))
                if not m.any():
                    # Impossible to enter
                    piv.pop(k+1)
                    piv.pop(k)
                    changed = True
                    break
                sub = df1s.iloc[m]
                idx = sub.index[0]
                entry_price = float(sub.loc[idx, "low"] if direction == "LONG" else sub.loc[idx, "high"])
                entry_ts = float(sub.loc[idx, "timestamp"])

            # EXIT: flat-zone best bar at the leg's END turn = the cubic's ACTUAL direction change
            ea, eb = flat_span(smooth, i1, n, exit_band)
            exit_price, exit_ts = best_bar_1s(df1s, ts1m[ea], ts1m[eb] + 60,
                                              "SHORT" if direction == "LONG" else "LONG")
            if exit_price is None or exit_ts <= entry_ts:
                # Impossible to exit profitably or chronologically
                piv.pop(k+1)
                piv.pop(k)
                changed = True
                break
                
            # held through wiggles: MAE = worst adverse inside the leg; flag if price broke past the entry
            seg_hi = hi1m[i0:i1 + 1]; seg_lo = lo1m[i0:i1 + 1]
            mae = float(entry_price - seg_lo.min()) if direction == "LONG" else float(seg_hi.max() - entry_price)
            pnl = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
            
            if pnl < min_pnl:
                is_marginal = True
                
            if mae > max_mae:               # broke past entry inside the leg -> inspect (kept anyway)
                flags.append({"date": date_key, "turn_ts": float(ts1m[i0]), "direction": direction,
                              "entry_price": entry_price, "adverse_pts": round(mae, 2),
                              "reason": f"price broke past entry (mae > {max_mae:.2f})"})
            trades.append({"entry_ts": entry_ts, "exit_ts": exit_ts, "direction": direction,
                           "side": "Buy" if direction == "LONG" else "Sell",
                           "entry_price": entry_price, "exit_price": exit_price,
                           "pnl_dollars": round(pnl / TICK * 0.50, 2),
                           "mae_dollars": round(max(mae, 0.0) / TICK * 0.50, 2),
                           "is_marginal": is_marginal,
                           "original_timestamp": float(ts1m[i0])})
            consumed_ts = exit_ts

    return trades, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="YYYY_MM_DD (single day)")
    ap.add_argument("--month", help="YYYY_MM (batch)")
    ap.add_argument("--fixed", action="store_true", help="Force fixed-T mode even if ADAPTIVE is True")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(FLAG, exist_ok=True)
    days = ([a.day] if a.day else
            [os.path.basename(f)[:-8] for f in sorted(glob.glob(os.path.join(ONE_M, f"{a.month}_*.parquet")))])
    cache = {}
    tot_t = tot_f = 0
    for dk in days:
        try:
            trades, flags = process_day(dk, cache, fixed=a.fixed)
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
        print("(v2: cubic-leg segmentation, flat-zone best-bar entry/exit, exit at the cubic turn, reversals -> flagged/)")


if __name__ == "__main__":
    main()
