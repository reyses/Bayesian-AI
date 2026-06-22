"""Guard test for core_v2/sessions.py session-day boundary (DST-aware, CME 17:00 CT).
Run: python research/test_session_day.py
"""
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.sessions import session_day, session_day_array  # noqa: E402


def ts(y, m, d, H, M=0):
    return int(datetime(y, m, d, H, M, tzinfo=timezone.utc).timestamp())


CASES = [
    # --- SUMMER (CDT, UTC-5): reopen 17:00 CT = 22:00 UTC ---
    (ts(2024, 6, 16, 22, 0),  '2024_06_17', "Sun 17:00 CDT reopen -> Monday session"),
    (ts(2024, 6, 17, 21, 59), '2024_06_17', "Mon 16:59 CDT (pre-halt) -> Monday close"),
    (ts(2024, 6, 17, 22, 0),  '2024_06_18', "Mon 17:00 CDT reopen -> Tuesday session"),
    (ts(2024, 6, 17, 13, 30), '2024_06_17', "Mon 08:30 CDT (RTH open) -> Monday"),
    # --- WINTER (CST, UTC-6): reopen 17:00 CT = 23:00 UTC (boundary shifts +1h) ---
    (ts(2024, 12, 16, 22, 59), '2024_12_16', "Mon 16:59 CST (pre-halt) -> Monday close"),
    (ts(2024, 12, 16, 23, 0),  '2024_12_17', "Mon 17:00 CST reopen -> Tuesday session"),
    (ts(2024, 12, 15, 23, 0),  '2024_12_16', "Sun 17:00 CST reopen -> Monday session"),
    # --- DST transition days (must not crash; boundary is far from the 02:00 CT flip) ---
    (ts(2024, 11, 4, 6, 0),   '2024_11_04', "Mon 00:00 CST after fall-back -> Monday"),
    (ts(2024, 3, 11, 5, 0),   '2024_03_11', "Mon 00:00 CDT after spring-forward -> Monday"),
]


def main():
    fails = []
    for epoch, expect, desc in CASES:
        got = session_day(epoch)
        if got != expect:
            fails.append(f"session_day({epoch}) = {got} != {expect}  [{desc}]")

    # vectorized must match scalar exactly
    arr = np.array([c[0] for c in CASES], dtype='int64')
    vec = session_day_array(arr)
    for i, (epoch, expect, desc) in enumerate(CASES):
        if vec[i] != expect:
            fails.append(f"session_day_array[{i}] = {vec[i]} != {expect}  [{desc}]")
        if vec[i] != session_day(epoch):
            fails.append(f"vectorized != scalar at {epoch}: {vec[i]} vs {session_day(epoch)}")

    print("Session-day boundary checks:")
    for epoch, expect, desc in CASES:
        utc = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        print(f"  {utc} -> {session_day(epoch)}   ({desc})")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS: DST-aware session-day boundary correct (summer 22:00 UTC, winter 23:00 UTC, "
          "Sunday reopen -> Monday, vectorized == scalar).")


if __name__ == '__main__':
    main()
