"""
v15_hour_filter_walkforward.py — Test if the hour-of-day filter generalizes
walk-forward, or is a pure-IS finding.

Process:
  1. Split 95 days into IS (first half) / OOS (second half).
  2. Compute per-hour total PnL on IS.
  3. Identify "tradable hours" (positive PnL on IS).
  4. Apply that hour mask to OOS — what would total OOS PnL be?
  5. Compare to OOS unfiltered.

Combine with the day-level bleed filter to see if compounding helps.

Usage:
    python tools/v15_hour_filter_walkforward.py
"""
from __future__ import annotations
import csv
from datetime import datetime
from collections import defaultdict
import pandas as pd

TRADES_PATH = "examples/trades.csv"
DAY_LABELS  = "reports/findings/2026-04-27_bleed_harvest_forward/day_labels_with_score.csv"


def parse_money(s):
    s = s.replace("$","").replace(",","").strip()
    if s == "" or s == "n/a": return 0.0
    if s.startswith("(") and s.endswith(")"): return -float(s[1:-1])
    return float(s)


def main():
    # Load trades
    trades = []
    with open(TRADES_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try: ts = datetime.strptime(r["Period"].strip(), "%m/%d/%Y %I:%M %p")
            except: continue
            trades.append({"date": ts.date(), "hour": ts.hour, "dow": ts.weekday(),
                           "pnl": parse_money(r["Net profit"])})

    # Load day labels for IS/OOS split
    day_df = pd.read_csv(DAY_LABELS)
    day_df["date"] = pd.to_datetime(day_df["date"]).dt.date
    is_dates  = set(day_df[day_df["set"] == "IS" ]["date"])
    oos_dates = set(day_df[day_df["set"] == "OOS"]["date"])
    is_pass_dates  = set(day_df[(day_df["set"] == "IS")  & (day_df["bleed_score"] <= -0.34)]["date"])
    oos_pass_dates = set(day_df[(day_df["set"] == "OOS") & (day_df["bleed_score"] <= -0.34)]["date"])

    is_trades  = [t for t in trades if t["date"] in is_dates]
    oos_trades = [t for t in trades if t["date"] in oos_dates]

    # ── Phase A: Hour-of-day filter ALONE (no bleed filter) ───────────────────
    # Compute IS per-hour PnL, identify positive hours
    is_hour = defaultdict(lambda: {"pnl": 0.0, "n": 0})
    for t in is_trades:
        is_hour[t["hour"]]["pnl"] += t["pnl"]
        is_hour[t["hour"]]["n"]   += 1

    # Tradable hours = positive total PnL on IS, with min N
    tradable_hours = sorted([h for h, s in is_hour.items() if s["pnl"] > 0 and s["n"] >= 5])
    print(f"IS-derived tradable hours (positive PnL, N>=5): {tradable_hours}")
    print()

    # Apply to OOS
    is_total       = sum(t["pnl"] for t in is_trades)
    oos_total      = sum(t["pnl"] for t in oos_trades)
    is_kept        = sum(t["pnl"] for t in is_trades  if t["hour"] in tradable_hours)
    oos_kept       = sum(t["pnl"] for t in oos_trades if t["hour"] in tradable_hours)

    print("=" * 80)
    print("HOUR-OF-DAY FILTER ALONE (no bleed-day filter)")
    print("=" * 80)
    print(f"IS total:     ${is_total:+,.2f}  ({len(is_trades)} trades)")
    print(f"IS kept:      ${is_kept:+,.2f}  ({sum(1 for t in is_trades if t['hour'] in tradable_hours)} trades)")
    print(f"OOS total:    ${oos_total:+,.2f}  ({len(oos_trades)} trades)")
    print(f"OOS kept:     ${oos_kept:+,.2f}  ({sum(1 for t in oos_trades if t['hour'] in tradable_hours)} trades)")
    print(f"OOS lift:     ${oos_kept - oos_total:+,.2f}")
    print()

    # ── Phase B: Bleed filter ALONE (no hour filter) ──────────────────────────
    is_bleed_kept = sum(t["pnl"] for t in is_trades  if t["date"] in is_pass_dates)
    oos_bleed_kept= sum(t["pnl"] for t in oos_trades if t["date"] in oos_pass_dates)
    print("=" * 80)
    print("BLEED FILTER ALONE (the MVP rule)")
    print("=" * 80)
    print(f"IS kept:      ${is_bleed_kept:+,.2f}")
    print(f"OOS kept:     ${oos_bleed_kept:+,.2f}")
    print(f"OOS lift:     ${oos_bleed_kept - oos_total:+,.2f}")
    print()

    # ── Phase C: COMBINED — bleed filter AND hour filter ─────────────────────
    is_combined = sum(t["pnl"] for t in is_trades
                      if t["date"] in is_pass_dates and t["hour"] in tradable_hours)
    oos_combined= sum(t["pnl"] for t in oos_trades
                      if t["date"] in oos_pass_dates and t["hour"] in tradable_hours)
    print("=" * 80)
    print("COMBINED FILTER: bleed-day pass AND tradable hour")
    print("=" * 80)
    print(f"IS kept:      ${is_combined:+,.2f}")
    print(f"OOS kept:     ${oos_combined:+,.2f}")
    print(f"OOS lift:     ${oos_combined - oos_total:+,.2f}")
    print()

    # ── Compare ──────────────────────────────────────────────────────────────
    print("=" * 80)
    print("COMPARISON (OOS only — the honest metric)")
    print("=" * 80)
    print(f"{'config':<35} {'kept_pnl':>12} {'lift':>10} {'n_kept':>8}")
    print("-" * 70)
    print(f"{'unfiltered':<35} ${oos_total:>+10.2f} ${0:>+8.0f} {len(oos_trades):>8}")
    print(f"{'hour-only':<35} ${oos_kept:>+10.2f} ${oos_kept-oos_total:>+8.0f} {sum(1 for t in oos_trades if t['hour'] in tradable_hours):>8}")
    print(f"{'bleed-only':<35} ${oos_bleed_kept:>+10.2f} ${oos_bleed_kept-oos_total:>+8.0f} {sum(1 for t in oos_trades if t['date'] in oos_pass_dates):>8}")
    print(f"{'COMBINED (bleed AND hour)':<35} ${oos_combined:>+10.2f} ${oos_combined-oos_total:>+8.0f} {sum(1 for t in oos_trades if t['date'] in oos_pass_dates and t['hour'] in tradable_hours):>8}")


if __name__ == "__main__":
    main()
