"""
nt8_trades_by_hour.py -- Hour-of-day breakdown of NT8 trade-export ledger.

Parses the Period column (e.g. "1/2/2026 5:53 AM") into hour-of-day, then
aggregates Net profit per hour to find time-of-day edges.

Usage:
    python tools/nt8_trades_by_hour.py "examples/trades.csv"
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict


def parse_money(s: str) -> float:
    s = s.replace("$", "").replace(",", "").strip()
    if s == "" or s == "n/a":
        return 0.0
    if s.startswith("(") and s.endswith(")"):
        return -float(s[1:-1])
    return float(s)


def parse_hour(period: str) -> int | None:
    """Parse '1/2/2026 5:53 AM' -> 5 (or 17 for PM-equivalent)."""
    try:
        # Split off date, take time portion
        parts = period.strip().split(" ")
        if len(parts) < 3:
            return None
        time_str = parts[1]   # "5:53"
        ampm     = parts[2]   # "AM" or "PM"
        hr_str   = time_str.split(":")[0]
        hr       = int(hr_str)
        if ampm.upper() == "PM" and hr != 12:
            hr += 12
        if ampm.upper() == "AM" and hr == 12:
            hr = 0
        return hr
    except Exception:
        return None


def parse_dow(period: str) -> int | None:
    """Day of week from Period; 0=Mon..6=Sun"""
    try:
        from datetime import datetime
        # Period: "M/D/YYYY h:MM AM/PM"
        return datetime.strptime(period.strip(), "%m/%d/%Y %I:%M %p").weekday()
    except Exception:
        return None


def main(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            hr = parse_hour(r["Period"])
            if hr is None:
                continue
            pnl = parse_money(r["Net profit"])
            rows.append({"hour": hr, "dow": parse_dow(r["Period"]), "pnl": pnl})

    # ── Hour-of-day aggregation ────────────────────────────────────────────
    by_hour = defaultdict(list)
    for r in rows:
        by_hour[r["hour"]].append(r["pnl"])

    print("=" * 76)
    print(f"HOUR OF DAY breakdown ({len(rows)} trades total)")
    print("=" * 76)
    print(f"{'hour':>5} {'N':>5} {'total_$':>10} {'mean_$':>9} {'median':>9} {'WR':>5} {'best':>9} {'worst':>9}")
    print("-" * 76)

    total_all   = sum(r["pnl"] for r in rows)
    total_chk   = 0.0
    for h in sorted(by_hour):
        ps = by_hour[h]
        total = sum(ps)
        wins  = sum(1 for p in ps if p > 0)
        n     = len(ps)
        mean  = total / n
        med   = sorted(ps)[n // 2]
        total_chk += total
        wr_pct = 100.0 * wins / n if n else 0.0
        print(f"{h:>5} {n:>5} ${total:>+9.0f} ${mean:>+8.2f} ${med:>+8.2f} {wr_pct:>4.0f}% ${max(ps):>+8.0f} ${min(ps):>+8.0f}")

    print("-" * 76)
    print(f"{'TOTAL':>5} {len(rows):>5} ${total_all:>+9.0f}")
    print()

    # Pick top 3 winning hours and top 3 losing hours
    hour_totals = {h: sum(by_hour[h]) for h in by_hour}
    sorted_h = sorted(hour_totals.items(), key=lambda kv: kv[1])
    print("Worst 5 hours (most negative total):")
    for h, t in sorted_h[:5]:
        n = len(by_hour[h])
        print(f"  hr {h:>2}  N={n:>4}  total=${t:>+9.0f}  mean=${t/n:>+7.2f}")
    print("\nBest 5 hours (most positive total):")
    for h, t in sorted_h[-5:][::-1]:
        n = len(by_hour[h])
        print(f"  hr {h:>2}  N={n:>4}  total=${t:>+9.0f}  mean=${t/n:>+7.2f}")
    print()

    # ── Day-of-week aggregation (for completeness) ─────────────────────────
    by_dow = defaultdict(list)
    for r in rows:
        if r["dow"] is not None:
            by_dow[r["dow"]].append(r["pnl"])
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("=" * 76)
    print("DAY OF WEEK breakdown")
    print("=" * 76)
    print(f"{'dow':>5} {'N':>5} {'total_$':>10} {'mean':>9} {'WR':>5}")
    for d in sorted(by_dow):
        ps = by_dow[d]
        total = sum(ps)
        wr = 100.0 * sum(1 for p in ps if p > 0) / len(ps)
        print(f"{dow_names[d]:>5} {len(ps):>5} ${total:>+9.0f} ${total/len(ps):>+8.2f} {wr:>4.0f}%")
    print()

    # ── If we filtered to ONLY trades in the best half of hours ────────────
    # Find the threshold where cumulative hour-totals crosses positive
    sorted_hr_by_total = sorted(by_hour.keys(), key=lambda h: -sum(by_hour[h]))
    cum, kept_total, kept_n = 0.0, 0.0, 0
    keep = set()
    print("=" * 76)
    print("FILTER SCENARIO: keep only PROFITABLE hours, skip the rest")
    print("=" * 76)
    for h in sorted_hr_by_total:
        t = sum(by_hour[h])
        if t > 0:
            keep.add(h)
            kept_total += t
            kept_n     += len(by_hour[h])
    skip_total = total_all - kept_total
    skip_n     = len(rows) - kept_n
    print(f"Hours kept (positive total):  {sorted(keep)}")
    print(f"  N trades: {kept_n}/{len(rows)}  ({100*kept_n/len(rows):.0f}% of trades)")
    print(f"  Total:    ${kept_total:+,.0f}  (vs ${total_all:+,.0f} unfiltered)")
    print(f"  Mean:     ${kept_total/kept_n:+,.2f}/trade")
    print(f"Hours skipped (negative total): {sorted(set(by_hour) - keep)}")
    print(f"  N trades: {skip_n}/{len(rows)}  ({100*skip_n/len(rows):.0f}%)")
    print(f"  Total:    ${skip_total:+,.0f}")
    print(f"  Avoiding these would save ${-skip_total:+,.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/trades.csv")
