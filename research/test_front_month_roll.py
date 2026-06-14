"""
Pure-logic guard test for the NT8-calendar-keyed FrontMonthSelector in
DATA/pipeline/databento_to_atlas.py — validates the roll picker WITHOUT touching
real data or running ingestion. Covers:
  (1) the roll-date rule reproduces NT8's published dates (Monday of expiry week),
  (2) calendar spreads (symbols with '-') are excluded,
  (3) the picker holds the front contract until the CALENDAR roll date even when
      the next contract is already more liquid (the key date-keyed behavior),
  (4) post-expiry days are RECOVERED (the old calendar-MONTH-map bug dropped them).

Run: python research/test_front_month_roll.py
"""
import os
import sys
from datetime import datetime, date, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # so databento_to_atlas can import core_v2.sessions
sys.path.insert(0, os.path.join(ROOT, 'DATA', 'pipeline'))
from databento_to_atlas import (  # noqa: E402
    FrontMonthSelector, _roll_date, _front_for_day, _build_roll_calendar)


def _ts(y, m, d, i=0):
    return int(datetime(y, m, d, 12, 0, tzinfo=timezone.utc).timestamp()) + i


def _rows(y, m, d, vols):
    rows, i = [], 0
    for sym, v in vols.items():
        rows.append({'timestamp': _ts(y, m, d, i), 'symbol': sym, 'volume': v,
                     'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0})
        i += 1
    return rows


def build_frame():
    # March 2024: NT8 rolls H4->M4 on Mon 2024-03-11 (expiry Fri 03-15).
    days = [
        # before the roll date: M4 already MORE liquid + a huge spread -> still hold H4
        (2024, 3, 7,  {'MNQH4': 800, 'MNQM4': 900, 'MNQH4-MNQM4': 9999}),
        (2024, 3, 11, {'MNQH4': 600, 'MNQM4': 900}),   # roll date -> M4
        (2024, 3, 18, {'MNQM4': 1000}),                # post-expiry: H4 gone -> RECOVERED
    ]
    rows = []
    for (y, m, d, vols) in days:
        rows.extend(_rows(y, m, d, vols))
    return pd.DataFrame(rows)


def main():
    fails = []

    # (1) roll-date rule == NT8's confirmed published dates
    checks = {
        (2024, 3): date(2024, 3, 11),   # H4->M4 (web-confirmed)
        (2024, 6): date(2024, 6, 17),
        (2024, 9): date(2024, 9, 16),
        (2024, 12): date(2024, 12, 16),
        (2025, 3): date(2025, 3, 17),
        (2025, 6): date(2025, 6, 16),   # M5->U5 (web-confirmed)
        (2025, 9): date(2025, 9, 15),   # U5->Z5 (web-confirmed)
        (2025, 12): date(2025, 12, 15),  # Z5->H6 (web-confirmed)
    }
    for (y, m), expect in checks.items():
        got = _roll_date(y, m)
        if got != expect:
            fails.append(f"_roll_date({y},{m}) = {got} != NT8 {expect}")

    # (2) _front_for_day: holds the contract across its window, switches on the date
    ev = _build_roll_calendar(2023, 2026)
    fday = {
        date(2024, 3, 10): 'MNQH4',   # day before roll -> still H4
        date(2024, 3, 11): 'MNQM4',   # roll date -> M4
        date(2024, 6, 16): 'MNQM4',   # day before next roll -> still M4
        date(2024, 6, 17): 'MNQU4',   # next roll -> U4
        date(2025, 6, 16): 'MNQU5',   # 2025 M5->U5
    }
    for d, expect in fday.items():
        got = _front_for_day(d, ev)
        if got != expect:
            fails.append(f"_front_for_day({d}) = {got} != {expect}")

    # (3)+(4) selector behavior on synthetic data
    sel = FrontMonthSelector()
    out = sel.filter(build_frame())
    man = pd.DataFrame(sel.manifest).sort_values('day').reset_index(drop=True)
    print("Per-day decision manifest:")
    print(man[['day', 'chosen', 'rolled', 'calendar_fallback', 'n_outrights']].to_string(index=False))

    chosen = man['chosen'].tolist()
    if chosen != ['MNQH4', 'MNQM4', 'MNQM4']:
        fails.append(f"chosen {chosen} != ['MNQH4','MNQM4','MNQM4'] "
                     "(date-keyed should hold H4 on 03-07 despite M4 being more liquid)")

    if 'MNQH4-MNQM4' in out['symbol'].unique() or any('-' in c for c in chosen):
        fails.append("spread symbol leaked into selection/output")

    rolled_days = man.loc[man['rolled'], 'day'].tolist()
    if rolled_days != ['2024_03_11']:
        fails.append(f"roll seams {rolled_days} != ['2024_03_11']")

    out_days = out['timestamp'].apply(
        lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y_%m_%d')).unique()
    if '2024_03_18' not in out_days:
        fails.append("post-expiry day 2024_03_18 was DROPPED (bug not fixed)")

    for day, g in out.groupby(out['timestamp'].apply(
            lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y_%m_%d'))):
        if g['symbol'].nunique() != 1:
            fails.append(f"{day}: output has {g['symbol'].nunique()} contracts (expected 1)")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS: NT8-calendar roll picker is correct "
          "(roll dates match NT8, spread-excluded, holds-until-roll-date, post-expiry recovered).")


if __name__ == '__main__':
    main()
