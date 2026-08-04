"""Sweep the ATLAS corpus and materialise the EVENT LIBRARY.

One parquet per event type in research/event_library/events/. Each row is one
causally-stamped event plus its measured forward outcome.

Day 2024_09_16 (pocket-dojo live sim) is EXCLUDED — see common.EXCLUDED_DAYS.

Run from repo root:
  python research/event_library/builders/build_event_library.py [--workers N]
                                                                [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline.common import EVENTS_DIR, day_list, load_day       # noqa: E402
from pipeline import detectors as det                            # noqa: E402
from pipeline import outcomes as out                             # noqa: E402

TABLES = ("ultra_chop", "chop_control", "leg_descent", "fakeout_poke",
          "stall", "defended_poke_shelf", "flush_v_day")


def one_day(day: str) -> dict[str, list[dict]]:
    res: dict[str, list[dict]] = {k: [] for k in TABLES}

    d1s = load_day(day, "1s")
    if d1s is not None:
        res["ultra_chop"] = out.outcomes_ultra_chop(
            d1s, det.detect_ultra_chop(d1s))
        # seed from the date so the control anchors are reproducible per day
        res["chop_control"] = out.random_controls(
            d1s, seed=int(day.replace("_", "")))

    d5s = load_day(day, "5s")
    if d5s is not None:
        s5 = det.scan_5s(d5s)
        res["leg_descent"] = out.outcomes_leg_descent(d5s, s5["leg_descent"])
        res["fakeout_poke"] = out.outcomes_fakeout_poke(d5s, s5["fakeout_poke"])
        res["stall"] = out.outcomes_stall(d5s, s5["stall"])

    d1m = load_day(day, "1m")
    if d1m is not None:
        s1 = det.scan_1m(d1m)
        res["defended_poke_shelf"] = out.outcomes_shelf(
            d1m, s1["defended_poke_shelf"])
        res["flush_v_day"] = out.outcomes_flush_v(d1m, s1["flush_v_day"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="first N days (debug)")
    args = ap.parse_args()

    days = day_list()
    if args.limit:
        days = days[:args.limit]
    print(f"days: {len(days)} (2024_09_16 excluded) | workers {args.workers}")

    acc: dict[str, list[dict]] = {k: [] for k in TABLES}
    if args.workers <= 1:
        for day in tqdm(days, desc="days"):
            for k, v in one_day(day).items():
                acc[k].extend(v)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(one_day, day): day for day in days}
            for f in tqdm(as_completed(futs), total=len(futs), desc="days"):
                try:
                    for k, v in f.result().items():
                        acc[k].extend(v)
                except Exception as exc:                       # noqa: BLE001
                    print(f"\nFAILED {futs[f]}: {type(exc).__name__}: {exc}")
                    raise

    os.makedirs(EVENTS_DIR, exist_ok=True)
    for k, rows in acc.items():
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values(["day", "ts"]).reset_index(drop=True)
        path = os.path.join(EVENTS_DIR, f"{k}.parquet")
        df.to_parquet(path, index=False)
        print(f"{k:22s} {len(df):8d} rows -> {path}")


if __name__ == "__main__":
    main()
