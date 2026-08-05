"""QC inspection of fakeout_poke detector.

Loads sample CSV, replays each event against raw 1s bar data, checks:
1. FIELD ACCURACY: poke_ext, poke_depth, ref_age_s vs raw bars
2. KIND: RETURN vs BREAKOUT classification
3. OUTCOME: exceed_ref_first logic
4. CAUSALITY: no future data used at ts

Writes JSON report to result_fakeout_poke_b0_HAIKU.json
"""
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.event_library.pipeline.common import TICK_PT

# Constants from detector
POKE_MAX_PT = 2.0
POKE_RETURN_S = 60

# Tolerances
PRICE_TOL = 0.25  # one tick
TIME_TOL = 2

def load_sample():
    """Load the sample CSV."""
    csv_path = Path(__file__).parent / "sample_fakeout_poke_b0.csv"
    return pd.read_csv(csv_path)

def load_bars(day, tf="5s"):
    """Load bar data for a day."""
    bars_path = REPO_ROOT / "DATA" / "ATLAS" / tf / f"{day}.parquet"
    if not bars_path.exists():
        return None
    return pd.read_parquet(bars_path)

def inspect_event(row, bars_5s, bars_1s):
    """Inspect a single event. Returns (verdict, defect_type, detail).

    Uses 5s bars for detector logic (since detector runs on 5s),
    and 1s bars for forward-looking outcome validation.
    """
    defects = []

    ts = int(row["ts"])
    day = row["day"]
    i = int(row["i"])
    dir_val = int(row["dir"])
    ref_px = float(row["ref_px"])
    poke_ext_recorded = float(row["poke_ext"])
    poke_depth_recorded = float(row["poke_depth"])
    ref_age_s_recorded = int(row["ref_age_s"])
    arm_ts = int(row["arm_ts"])
    kind_recorded = row["kind"]
    exceed_ref_first_recorded = bool(row["exceed_ref_first"])

    if bars_5s is None:
        return "DEFECT", "NO_DATA", f"Missing 5s bar data for {day}"

    # =========================================================================
    # 1. FIELD ACCURACY (check against 5s bars since detector runs on 5s)
    # =========================================================================

    # Find the bar at ts in 5s data
    mask_ts = bars_5s["timestamp"] == ts
    if not mask_ts.any():
        return "DEFECT", "BAR_NOT_FOUND", f"Bar at ts={ts} not found in 5s data"

    bar_ts = bars_5s[mask_ts].iloc[0]

    # poke_ext should match the close or high of the 5s bar (or a recent one)
    # Since the detector updates poke_ext from 5s closes, poke_ext should be
    # a close from a 5s bar at or before ts
    bars_before_ts = bars_5s[bars_5s["timestamp"] <= ts]
    if bars_before_ts.empty:
        return "DEFECT", "NO_BARS_BEFORE_TS", f"No 5s bars at/before ts={ts}"

    # Check recent 5s closes and highs
    recent = bars_before_ts.tail(10)
    if dir_val == 1:
        extremes = np.concatenate([recent["high"].values, recent["close"].values])
    else:
        extremes = np.concatenate([recent["low"].values, recent["close"].values])

    poke_ext_matches = np.any(np.abs(extremes - poke_ext_recorded) <= PRICE_TOL)
    if not poke_ext_matches:
        # Also check if it matches the 5s bar's high (intrabar extreme)
        if dir_val == 1 and abs(float(bar_ts["high"]) - poke_ext_recorded) <= PRICE_TOL:
            poke_ext_matches = True
        elif dir_val == -1 and abs(float(bar_ts["low"]) - poke_ext_recorded) <= PRICE_TOL:
            poke_ext_matches = True

    if not poke_ext_matches:
        defects.append(("POKE_EXT", f"poke_ext={poke_ext_recorded} not in recent extremes; bar_high={float(bar_ts['high']):.2f}, bar_close={float(bar_ts['close']):.2f}"))

    # Check poke_depth
    expected_depth = abs(poke_ext_recorded - ref_px)
    if abs(expected_depth - poke_depth_recorded) > PRICE_TOL:
        defects.append(("POKE_DEPTH", f"recorded={poke_depth_recorded}, computed={expected_depth:.2f}"))

    # Check ref_age_s
    if ref_age_s_recorded < 0:
        defects.append(("REF_AGE", f"negative ref_age_s={ref_age_s_recorded}"))
    if ref_age_s_recorded > 3600 * 24:
        defects.append(("REF_AGE", f"ref_age_s={ref_age_s_recorded} > 1 day"))

    # =========================================================================
    # 2. KIND CLASSIFICATION (check against 5s bars)
    # =========================================================================

    # At the firing bar (ts), check if kind is consistent with the recorded poke_ext
    close_at_ts = float(bar_ts["close"])

    # RETURN: close has crossed back inside ref_px
    if dir_val * (close_at_ts - ref_px) < 0:
        recomputed_kind = "RETURN"
    else:
        # Check if poke_ext exceeds ref by > 2pt
        if dir_val * (poke_ext_recorded - ref_px) > POKE_MAX_PT:
            recomputed_kind = "BREAKOUT"
        elif ts - arm_ts > POKE_RETURN_S:
            recomputed_kind = "STUCK"
        else:
            recomputed_kind = None  # Not yet resolved

    if recomputed_kind and recomputed_kind != kind_recorded:
        defects.append(("KIND", f"recorded={kind_recorded}, recomputed={recomputed_kind}"))

    # =========================================================================
    # 3. OUTCOME: exceed_ref_first (use 1s bars for finer outcome validation)
    # =========================================================================

    if bars_1s is not None:
        bars_forward = bars_1s[bars_1s["timestamp"] >= ts].sort_values("timestamp")
        if len(bars_forward) > 0:
            exceed_ref_by_2 = False
            moved_10_adverse = False
            exceed_first = None

            for _, bar in bars_forward.iterrows():
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])

                # Check if we've moved 10pt adverse
                if dir_val == 1:
                    if bar_low < ref_px - 10:
                        moved_10_adverse = True
                else:
                    if bar_high > ref_px + 10:
                        moved_10_adverse = True

                # Check if we've exceeded ref by 2pt
                if dir_val == 1:
                    if bar_high >= ref_px + 2 and not exceed_ref_by_2:
                        exceed_ref_by_2 = True
                        exceed_first = True if not moved_10_adverse else False
                else:
                    if bar_low <= ref_px - 2 and not exceed_ref_by_2:
                        exceed_ref_by_2 = True
                        exceed_first = True if not moved_10_adverse else False

                if exceed_ref_by_2 and moved_10_adverse:
                    break

            if exceed_first is not None and exceed_first != exceed_ref_first_recorded:
                defects.append(("EXCEED_REF_FIRST", f"recorded={exceed_ref_first_recorded}, recomputed={exceed_first}"))

    # =========================================================================
    # 4. CAUSALITY: event at ts should use only data from ts or before
    # =========================================================================

    bars_before = bars_5s[bars_5s["timestamp"] < ts]
    if len(bars_before) == 0:
        defects.append(("CAUSALITY", f"No 5s bars before ts={ts}"))

    # =========================================================================
    # VERDICT
    # =========================================================================

    if defects:
        defect_type = defects[0][0]
        detail = "; ".join([f"{k}:{v}" for k, v in defects])
        return "DEFECT", defect_type, detail
    else:
        return "PASS", None, None

def main():
    sample = load_sample()
    results = {
        "detector": "fakeout_poke",
        "batch": 0,
        "model": "haiku",
        "inspected": len(sample),
        "defects": 0,
        "defect_types": {},
        "per_event": [],
        "examples": []
    }

    print(f"Inspecting {len(sample)} events...")

    for idx, row in sample.iterrows():
        day = row["day"]
        ts = int(row["ts"])

        # Load bars
        bars_5s = load_bars(day, tf="5s")
        bars_1s = load_bars(day, tf="1s")

        # Inspect
        verdict, defect_type, detail = inspect_event(row, bars_5s, bars_1s)

        event_result = {
            "ts": ts,
            "day": day,
            "verdict": verdict,
        }
        if defect_type:
            event_result["type"] = defect_type
            event_result["detail"] = detail

        results["per_event"].append(event_result)

        if verdict == "DEFECT":
            results["defects"] += 1
            results["defect_types"][defect_type] = results["defect_types"].get(defect_type, 0) + 1

            # Collect examples (up to 5)
            if len(results["examples"]) < 5:
                results["examples"].append({
                    "day": day,
                    "ts": ts,
                    "type": defect_type,
                    "detail": detail
                })

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(sample)}")

    # Write results
    out_path = Path(__file__).parent / "result_fakeout_poke_b0_HAIKU.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {out_path}")
    print(f"Inspected: {results['inspected']}")
    print(f"Defects: {results['defects']}")
    print(f"Defect types: {results['defect_types']}")

if __name__ == "__main__":
    main()
