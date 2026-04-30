"""
v15_filter_apply.py -- Apply the v1.5-RC bleed-score filter retroactively
to the NT8 trade-export CSV. Reports actual PnL with vs without the filter.

Implements exactly the rule that v1.5-RC will run in NT8:
  bleed_score = z(prior_range) + z(range_compression)
  trade_today = (bleed_score <= BleedThresholdZ)

Z-scores use IS-calibrated constants (computed earlier this session).

Usage:
    python tools/v15_filter_apply.py
    python tools/v15_filter_apply.py --threshold 0.0
"""
from __future__ import annotations
import argparse
import csv
import os
from datetime import datetime
from collections import defaultdict
import pandas as pd

# IS-calibrated constants (from 2026-04-27_bleed_harvest_forward run)
MEAN_PRIOR_RANGE       = 385.32
STD_PRIOR_RANGE        = 219.83
MEAN_RANGE_COMPRESSION = 1.0315
STD_RANGE_COMPRESSION  = 0.5502


def parse_money(s):
    s = s.replace("$","").replace(",","").strip()
    if s == "" or s == "n/a": return 0.0
    if s.startswith("(") and s.endswith(")"): return -float(s[1:-1])
    return float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="examples/trades.csv")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="bleed_score threshold; days with score <= threshold trade. Default 0.0 (~X=40%)")
    ap.add_argument("--day-features", default="reports/findings/2026-04-27_bleed_harvest_forward/day_labels_with_score.csv",
                    help="CSV with bleed_score per day (output of nt8_bleed_harvest_classifier.py)")
    args = ap.parse_args()

    # Load day-level features + bleed score
    day_df = pd.read_csv(args.day_features)
    day_df["date"] = pd.to_datetime(day_df["date"]).dt.date

    # Apply filter
    day_df["filter_pass"] = day_df["bleed_score"] <= args.threshold
    pass_dates = set(day_df[day_df["filter_pass"]]["date"])

    # Load trades
    trades = []
    with open(args.trades, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                ts = datetime.strptime(r["Period"].strip(), "%m/%d/%Y %I:%M %p")
            except Exception:
                continue
            trades.append({
                "date": ts.date(),
                "ts":   ts,
                "pnl":  parse_money(r["Net profit"]),
            })

    n_trades = len(trades)
    total_unfiltered = sum(t["pnl"] for t in trades)

    # Filter
    kept = [t for t in trades if t["date"] in pass_dates]
    skipped = [t for t in trades if t["date"] not in pass_dates]

    n_kept     = len(kept)
    n_skipped  = len(skipped)
    pnl_kept   = sum(t["pnl"] for t in kept)
    pnl_skipped= sum(t["pnl"] for t in skipped)

    days_total = day_df["date"].nunique()
    days_pass  = day_df["filter_pass"].sum()
    days_skip  = days_total - days_pass

    print("=" * 76)
    print(f"v1.5-RC filter application -- threshold = {args.threshold:+.2f}")
    print("=" * 76)
    print(f"Days total:      {days_total}")
    print(f"Days kept:       {days_pass}  ({100*days_pass/days_total:.1f}%)")
    print(f"Days skipped:    {days_skip}  ({100*days_skip/days_total:.1f}%)")
    print()
    print(f"Trades total:    {n_trades}")
    print(f"Trades kept:     {n_kept}  ({100*n_kept/n_trades:.1f}%)")
    print(f"Trades skipped:  {n_skipped}  ({100*n_skipped/n_trades:.1f}%)")
    print()
    print(f"PnL UNFILTERED:  ${total_unfiltered:+,.2f}")
    print(f"PnL kept:        ${pnl_kept:+,.2f}  (the filtered-strategy PnL)")
    print(f"PnL skipped:     ${pnl_skipped:+,.2f}  (= what the filter saved us from)")
    print(f"Net lift:        ${pnl_kept - total_unfiltered:+,.2f}")
    print()
    print(f"Per kept trade:  ${pnl_kept / max(n_kept, 1):+.2f}")
    print(f"Per kept day:    ${pnl_kept / max(days_pass, 1):+.2f}")

    # Per-set breakdown (IS vs OOS)
    is_dates  = set(day_df[day_df["set"] == "IS"]["date"])
    oos_dates = set(day_df[day_df["set"] == "OOS"]["date"])
    is_kept   = [t for t in kept if t["date"] in is_dates]
    oos_kept  = [t for t in kept if t["date"] in oos_dates]
    is_total_pnl  = sum(t["pnl"] for t in trades if t["date"] in is_dates)
    oos_total_pnl = sum(t["pnl"] for t in trades if t["date"] in oos_dates)
    is_kept_pnl   = sum(t["pnl"] for t in is_kept)
    oos_kept_pnl  = sum(t["pnl"] for t in oos_kept)

    print()
    print("--- IS / OOS breakdown ---")
    print(f"{'set':<5} {'unfiltered':>12} {'filtered':>12} {'lift':>10} {'days_kept':>10}")
    print(f"{'IS':<5} ${is_total_pnl:>+10.0f} ${is_kept_pnl:>+10.0f} ${is_kept_pnl-is_total_pnl:>+8.0f} {len(is_kept)} trades / {sum(1 for d in pass_dates if d in is_dates)} days")
    print(f"{'OOS':<5} ${oos_total_pnl:>+10.0f} ${oos_kept_pnl:>+10.0f} ${oos_kept_pnl-oos_total_pnl:>+8.0f} {len(oos_kept)} trades / {sum(1 for d in pass_dates if d in oos_dates)} days")
    print(f"{'ALL':<5} ${total_unfiltered:>+10.0f} ${pnl_kept:>+10.0f} ${pnl_kept-total_unfiltered:>+8.0f}")

    # Trade-day distribution
    print()
    print("--- Day PnL distribution (filter-pass days) ---")
    by_day_kept = defaultdict(lambda: {"pnl": 0.0, "n": 0})
    for t in kept:
        by_day_kept[t["date"]]["pnl"] += t["pnl"]
        by_day_kept[t["date"]]["n"] += 1
    pnls_per_day = sorted([s["pnl"] for s in by_day_kept.values()])
    if pnls_per_day:
        print(f"  N kept days:   {len(pnls_per_day)}")
        print(f"  mean $/day:    ${sum(pnls_per_day)/len(pnls_per_day):+.2f}")
        print(f"  median $/day:  ${pnls_per_day[len(pnls_per_day)//2]:+.2f}")
        print(f"  best day:      ${pnls_per_day[-1]:+.2f}")
        print(f"  worst day:     ${pnls_per_day[0]:+.2f}")
        n_pos = sum(1 for p in pnls_per_day if p > 0)
        print(f"  Day WR:        {100*n_pos/len(pnls_per_day):.1f}%  ({n_pos}/{len(pnls_per_day)})")


if __name__ == "__main__":
    main()
