"""Sanity-fire every detector on the CALIBRATION day 2024_09_16 and print the
detection timestamps around the owner's anchors.

2024_09_16 is the pocket-dojo live-sim day. It is hindsight-contaminated and
is EXCLUDED from every table and fit in this package. It appears here, and
only here, so each detector can be checked against the tape state the owner
actually named.

Owner anchors:
  ULTRA_CHOP            10:23:50-10:24:31  (~24 flips / 42s inside 13.25pt)
  LEG_DESCENT           09:56-10:24        (stair 19697 -> 19633)
  FLUSH_V_DAY           open flush -173.5pt by 09:35, ~79% V-recovery by 09:56

Run from repo root:
  python research/event_library/tools/anchor_fire.py
Writes research/event_library/reports/anchor_fire.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline.common import (LIVE_DAY, REPORTS_DIR, load_day)      # noqa: E402
from pipeline import detectors as det                              # noqa: E402
from pipeline import outcomes as out                               # noqa: E402

ANCHORS = {
    "ULTRA_CHOP": ("10:23:50", "10:24:31"),
    "LEG_DESCENT": ("09:56:00", "10:24:00"),
    "FAKEOUT_POKE": ("09:30:00", "15:30:00"),
    "STALL": ("09:30:00", "15:30:00"),
    "DEFENDED_POKE_AT_SHELF": ("10:00:00", "12:30:00"),
    "FLUSH_V_DAY": ("09:30:00", "10:30:00"),
}


def _et(ts):
    return (pd.Timestamp(int(ts), unit="s", tz="UTC")
            .tz_convert("America/New_York").strftime("%H:%M:%S"))


def _in_anchor(ts, name):
    lo, hi = ANCHORS[name]
    t = _et(ts)
    return lo <= t <= hi


def main() -> None:
    lines = [f"# Anchor fire — detector sanity check on {LIVE_DAY}", "",
             "`2024_09_16` is the live-sim day: EXCLUDED from every table in "
             "this package. It is used here only to check that each detector "
             "fires on the tape state the owner named.", ""]

    d1s = load_day(LIVE_DAY, "1s")
    d5s = load_day(LIVE_DAY, "5s")
    d1m = load_day(LIVE_DAY, "1m")

    # ---- 1. ULTRA_CHOP ----------------------------------------------------
    chop = out.outcomes_ultra_chop(d1s, det.detect_ultra_chop(d1s))
    lines += ["## 1. ULTRA_CHOP", "",
              f"fires: {len(chop)}  |  in anchor window "
              f"{ANCHORS['ULTRA_CHOP'][0]}-{ANCHORS['ULTRA_CHOP'][1]}: "
              f"{sum(_in_anchor(r['ts'], 'ULTRA_CHOP') for r in chop)}", ""]
    for r in chop[:20]:
        lines.append(f"- {_et(r['ts'])}  box {r['box_pt']:.2f}pt  "
                     f"flips {r['flips']:.0f}  ambient {r['ambient_pt']:.2f}pt "
                     f"ratio {r['box_ambient_ratio']:.2f}  "
                     f"escape {r['escape_lag_s']}s dir {r['escape_dir']:+d}")
    # what the anchor window actually measures, for the report's honesty note
    idx = pd.DatetimeIndex(pd.to_datetime(d1s.ts, unit="s", utc=True))
    flip = det._flip_flags(d1s.close)
    fr = pd.DataFrame({"h": d1s.high, "l": d1s.low, "c": d1s.close, "f": flip},
                      index=idx).rolling(f"{det.CHOP_WIN_S}s")
    box = (fr["h"].max() - fr["l"].min()).to_numpy()
    flips = fr["f"].sum().to_numpy()
    rth = d1s.rth_mask()
    a = np.array([_in_anchor(t, "ULTRA_CHOP") for t in d1s.ts])
    lines += ["",
              f"anchor-window 60s stats: box "
              f"{np.nanmin(box[a]):.2f}-{np.nanmax(box[a]):.2f}pt, flips "
              f"{np.nanmin(flips[a]):.0f}-{np.nanmax(flips[a]):.0f}",
              f"day RTH reference:       box p50 {np.nanpercentile(box[rth], 50):.2f} "
              f"p90 {np.nanpercentile(box[rth], 90):.2f}pt, flips p50 "
              f"{np.nanpercentile(flips[rth], 50):.0f} p90 "
              f"{np.nanpercentile(flips[rth], 90):.0f}", ""]

    # ---- 2/3/4. 5s detectors ---------------------------------------------
    s5 = det.scan_5s(d5s)
    dsc = out.outcomes_leg_descent(d5s, s5["leg_descent"])
    lines += ["## 2. LEG_DESCENT", "",
              f"defended pushes: {len(dsc)}  |  chain_n>=2: "
              f"{sum(r['chain_n'] >= 2 for r in dsc)}  |  in anchor "
              f"09:56-10:24: {sum(_in_anchor(r['ts'], 'LEG_DESCENT') for r in dsc)}",
              ""]
    for r in dsc:
        if _in_anchor(r["ts"], "LEG_DESCENT") or r["chain_n"] >= 2:
            lines.append(f"- {_et(r['ts'])}  N={r['chain_n']}  step "
                         f"{r['step_high']:.2f}->{r['step_low']:.2f} "
                         f"({r['step_depth']:.2f}pt)  chain descent "
                         f"{r['chain_descent']:.2f}pt  race {r['race']}")
    lines.append("")

    pk = out.outcomes_fakeout_poke(d5s, s5["fakeout_poke"])
    ret = [r for r in pk if r["kind"] == "RETURN"]
    lines += ["## 3. FAKEOUT_POKE", "",
              f"armed pokes: {len(pk)}  |  RETURN (the event): {len(ret)}  |  "
              f"STUCK: {sum(r['kind'] == 'STUCK' for r in pk)}  |  BREAKOUT: "
              f"{sum(r['kind'] == 'BREAKOUT' for r in pk)}", ""]
    for r in ret[:15]:
        lines.append(f"- {_et(r['ts'])}  dir {r['dir']:+d}  ref "
                     f"{r['ref_px']:.2f}  poke +{r['poke_depth']:.2f}pt  "
                     f"race {r['race']}  exceed_ref {r['exceed_ref']}")
    lines.append("")

    st = out.outcomes_stall(d5s, s5["stall"])
    lines += ["## 4. STALL", "",
              f"candidates: {len(st)}  |  STALL (giveback<=30% for 10min): "
              f"{sum(r['stalled'] for r in st)}", ""]
    for r in st:
        if r["stalled"]:
            lines.append(f"- {_et(r['ts'])}  peak {_et(r['peak_ts'])} "
                         f"{r['peak_px']:.2f}  dir {r['dir']:+d}  mfe "
                         f"{r['mfe_pt']:.1f}pt  give {r['give_frac']:.0%}  "
                         f"race {r['race']}")
    lines.append("")

    # ---- 5/6. 1m detectors ------------------------------------------------
    s1 = det.scan_1m(d1m)
    sh = out.outcomes_shelf(d1m, s1["defended_poke_shelf"])
    lines += ["## 5. DEFENDED_POKE_AT_SHELF", "", f"events: {len(sh)}", ""]
    for r in sh:
        lines.append(f"- {_et(r['ts'])}  shelf {r['shelf_px']:.2f} "
                     f"(dwell {r['dwell_frac']:.0%})  poke {r['poke_px']:.2f}  "
                     f"bounce {r['bounce_pt']:.1f}pt  class {r['day_class']}  "
                     f"outcome {r['outcome']}")
    lines.append("")

    fv = out.outcomes_flush_v(d1m, s1["flush_v_day"])
    lines += ["## 6. FLUSH_V_DAY", ""]
    for r in fv:
        lines.append(f"- is_flush={r['is_flush']}  confirm {_et(r['ts'])}  "
                     f"flush {r['flush_pt']:.1f}pt  rec {r['rec_frac']:.0%}  "
                     f"v_low {r['v_low']:.2f} v_peak {r['v_peak']:.2f}  "
                     f"first {r['first']}  close_frac {r['close_frac']:.2f}")
    lines.append("")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "anchor_fire.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
