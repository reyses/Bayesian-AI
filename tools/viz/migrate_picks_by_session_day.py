"""One-time cleanup: re-file human cusp picks by each pick's TRUE session-day.

The old cusp_marker._save keyed all picks to the loaded --date, so picks made after panning into the
next day were stranded in the wrong picks_<day>_multi.json. This re-distributes every pick in the
canonical picks_*_multi.json files into picks_<true-session-day>_multi.json by its own timestamp.

Safe: backs up all canonical files to DATA/cusp_picks/_backup_multi_<ts>/ before touching anything.
Only the *_multi.json canonical files are rewritten; timestamped snapshots and ai_cusp_picks are untouched.
"""
import glob
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pytz

PICKS = "DATA/cusp_picks"
EST = pytz.timezone("US/Eastern")


def session_day(ts):
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone(EST)
    d = dt.date() + (timedelta(days=1) if dt.hour >= 18 else timedelta(0))
    return d.strftime("%Y-%m-%d")


def main():
    files = sorted(glob.glob(os.path.join(PICKS, "picks_*_multi.json")))
    if not files:
        print("no canonical pick files found"); return
    bak = os.path.join(PICKS, f"_backup_multi_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(bak, exist_ok=True)

    pool = {}                          # (ts_int, tf) -> pick  (dedup)
    before = {}
    fwd = cubic = None
    for f in files:
        shutil.copy2(f, os.path.join(bak, os.path.basename(f)))
        d = json.load(open(f))
        before[os.path.basename(f).replace("picks_", "").replace("_multi.json", "")] = len(d.get("picks", []))
        fwd = d.get("fwd_mins", fwd); cubic = d.get("cubic_n", cubic)
        for p in d.get("picks", []):
            pool.setdefault((int(round(p["timestamp"])), p.get("timeframe", "1m")), p)
    print(f"backed up {len(files)} files -> {bak}")
    print(f"pooled {len(pool)} unique picks (from {sum(before.values())} rows; "
          f"{sum(before.values())-len(pool)} dups removed)")

    by_day = defaultdict(list)
    for p in pool.values():
        by_day[session_day(p["timestamp"])].append(p)

    for f in files:                    # remove old canonical files (backed up); rewrite fresh
        os.remove(f)
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    for day, picks in sorted(by_day.items()):
        picks.sort(key=lambda x: x["timestamp"])
        for i, p in enumerate(picks):
            p["pick_id"] = i
        json.dump({"date_range": day, "marked_timeframes": sorted({p.get("timeframe", "1m") for p in picks}),
                   "created": ts_tag, "n_picks": len(picks), "fwd_mins": fwd, "cubic_n": cubic,
                   "picks": picks}, open(os.path.join(PICKS, f"picks_{day}_multi.json"), "w"), indent=2)

    print(f"\n{'day':>22} | before -> after")
    print("-" * 42)
    allkeys = sorted(set(before) | set(by_day))
    for k in allkeys:
        b = before.get(k, 0); a = len(by_day.get(k, []))
        flag = "  <-- moved" if b != a else ""
        print(f"{k:>22} | {b:>5} -> {a:<5}{flag}")
    print(f"\n{len(by_day)} true-session-day files written; originals in {bak}")


if __name__ == "__main__":
    main()
