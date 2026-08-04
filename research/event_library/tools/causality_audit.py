"""Empirical lookahead audit: TRUNCATION-REPLAY every detector.

Prose claims of causality are cheap. This runs each detector twice — once on
the full day, once on the day truncated at a cut time — and checks that every
event stamped at or before the cut is IDENTICAL in both runs. If a detector
peeks at a future bar, the truncated run cannot reproduce the full run's rows.

Outcome fields are excluded from the comparison by construction: they are
forward-looking on purpose and live in `outcomes.py`.

Run from repo root:
  python research/event_library/tools/causality_audit.py [--days N]
Writes research/event_library/reports/causality_audit.md
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline.common import (Day, REPORTS_DIR, day_list, load_day)   # noqa: E402
from pipeline import detectors as det                                # noqa: E402

# ET minutes-of-day at which each day is truncated and replayed
CUT_MODS = (11 * 60, 13 * 60, 14 * 60 + 30)

# fields that define an event's IDENTITY (detector output only, no outcomes)
KEYS = {
    "ultra_chop": ["ts", "box_lo", "box_hi", "box_pt", "flips", "ambient_pt"],
    "leg_descent": ["ts", "chain_n", "step_high", "step_low", "defense_pt"],
    "fakeout_poke": ["ts", "dir", "ref_px", "poke_ext", "kind"],
    "stall": ["ts", "dir", "peak_ts", "peak_px", "mfe_pt", "stalled"],
    "defended_poke_shelf": ["ts", "shelf_px", "poke_px", "dwell_frac",
                            "day_class"],
    "flush_v_day": ["ts", "is_flush", "v_low", "v_peak"],
}


def truncate(d: Day, cut_mod: float) -> Day | None:
    """A Day containing only bars strictly before `cut_mod` ET on the named
    day. Prior-evening bars (mod >= 1080) are kept — they are genuinely in the
    past — which is also what a live engine would hold."""
    keep = ~((d.mod >= cut_mod) & (d.mod < 18 * 60))
    # everything from the first bar at/after the cut onward must go
    first_cut = np.flatnonzero((d.mod >= cut_mod) & (d.mod < 18 * 60))
    if first_cut.size == 0:
        return None
    j = int(first_cut[0])
    if j < 2:
        return None
    df = pd.DataFrame({"timestamp": d.ts[:j], "open": d.open[:j],
                       "high": d.high[:j], "low": d.low[:j],
                       "close": d.close[:j], "volume": d.volume[:j]})
    _ = keep
    return Day(d.day, d.tf, df)


def rows_to_frame(rows, keys):
    if not rows:
        return pd.DataFrame(columns=keys)
    return pd.DataFrame(rows)[keys].sort_values("ts").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=25)
    args = ap.parse_args()

    days = day_list()
    rng = np.random.default_rng(20260803)
    sample = sorted(rng.choice(days, size=min(args.days, len(days)),
                               replace=False).tolist())

    stats = {k: dict(compared=0, mismatch=0, extra=0, missing=0)
             for k in KEYS}

    for day in tqdm(sample, desc="days"):
        full = {}
        d1s, d5s, d1m = (load_day(day, tf) for tf in ("1s", "5s", "1m"))
        if d1s is None or d5s is None or d1m is None:
            continue
        full["ultra_chop"] = det.detect_ultra_chop(d1s)
        s5 = det.scan_5s(d5s)
        full.update(s5)
        s1 = det.scan_1m(d1m)
        full.update(s1)

        for cut in CUT_MODS:
            t1s, t5s, t1m = (truncate(x, cut) for x in (d1s, d5s, d1m))
            if t1s is None or t5s is None or t1m is None:
                continue
            trunc = {"ultra_chop": det.detect_ultra_chop(t1s)}
            trunc.update(det.scan_5s(t5s))
            trunc.update(det.scan_1m(t1m))

            cut_ts = int(t1m.ts[-1])
            for name, keys in KEYS.items():
                a = rows_to_frame(full.get(name, []), keys)
                b = rows_to_frame(trunc.get(name, []), keys)
                a = a[a["ts"] <= cut_ts].reset_index(drop=True)
                b = b[b["ts"] <= cut_ts].reset_index(drop=True)
                st = stats[name]
                st["compared"] += len(a)
                ta, tb = set(a["ts"]), set(b["ts"])
                st["extra"] += len(tb - ta)
                st["missing"] += len(ta - tb)
                common = sorted(ta & tb)
                if common:
                    aa = a[a["ts"].isin(common)].drop_duplicates("ts") \
                          .set_index("ts").sort_index()
                    bb = b[b["ts"].isin(common)].drop_duplicates("ts") \
                          .set_index("ts").sort_index()
                    diff = (aa != bb) & ~(aa.isna() & bb.isna())
                    st["mismatch"] += int(diff.any(axis=1).sum())

    lines = ["# Causality audit — truncation replay", "",
             f"Each of {len(sample)} randomly sampled days was replayed with "
             f"the tape cut at {', '.join(f'{c//60:02d}:{c%60:02d}' for c in CUT_MODS)} "
             "ET. Every event stamped at or before the cut must appear, "
             "identically, in the truncated run. Outcome fields are excluded "
             "(they are forward-looking by design).", "",
             "| detector | events compared | field mismatches | MISSING in "
             "truncated run | EXTRA in truncated run |", "|---|---|---|---|---|"]
    for name, st in stats.items():
        lines.append(f"| {name} | {st['compared']} | {st['mismatch']} | "
                     f"{st['missing']} | {st['extra']} |")
    lines += ["",
              "`MISSING` = the detector needed a FUTURE bar to emit an event "
              "the full run produced -> lookahead. `EXTRA` = the truncated "
              "run emitted an event the full run suppressed -> a forward-"
              "looking SAMPLING rule (de-dup / refractory), which does not "
              "leak into any event's own features or outcome but does mean "
              "the live row set differs near a cut.", ""]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "causality_audit.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
