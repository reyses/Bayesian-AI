"""
v15_cum_pnl_chart.py — Visualize v1.5-RC filter impact on cumulative PnL
across the 95-day window.

Output: reports/findings/2026-04-27_cum_pnl_filtered_vs_unfiltered.png
"""
from __future__ import annotations
import csv
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def parse_money(s):
    s = s.replace("$","").replace(",","").strip()
    if s == "" or s == "n/a": return 0.0
    if s.startswith("(") and s.endswith(")"): return -float(s[1:-1])
    return float(s)


def main():
    # Load trades
    trades = []
    with open("examples/trades.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try: ts = datetime.strptime(r["Period"].strip(), "%m/%d/%Y %I:%M %p")
            except: continue
            trades.append({"ts": ts, "date": ts.date(),
                           "pnl": parse_money(r["Net profit"])})

    # Load filter decisions
    df = pd.read_csv("reports/findings/2026-04-27_bleed_harvest_forward/day_labels_with_score.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    pass_dates = set(df[df["bleed_score"] <= -0.34]["date"])

    # Sort by time and compute cum PnL series
    trades.sort(key=lambda t: t["ts"])

    cum_unfilt = []
    cum_filt   = []
    cum_skip   = []
    times      = []
    u, f, s = 0.0, 0.0, 0.0
    for t in trades:
        u += t["pnl"]
        if t["date"] in pass_dates:
            f += t["pnl"]
        else:
            s += t["pnl"]
        cum_unfilt.append(u)
        cum_filt.append(f)
        cum_skip.append(s)
        times.append(t["ts"])

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(times, cum_unfilt, label=f"Unfiltered (final ${cum_unfilt[-1]:+,.0f})",
             color="#888888", linewidth=1.5)
    ax.plot(times, cum_filt, label=f"v1.5-RC filter ON (final ${cum_filt[-1]:+,.0f})",
             color="#2ca02c", linewidth=2.5)
    ax.plot(times, cum_skip, label=f"Skipped days only (saved ${-cum_skip[-1]:+,.0f})",
             color="#d62728", linewidth=1.5, alpha=0.7, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axvline(datetime(2026, 2, 26), color="purple", linestyle="--", alpha=0.5)
    ax.text(datetime(2026, 2, 26), max(cum_filt) * 0.95, "  Regime change\n  2026-02-26",
            fontsize=9, color="purple")
    ax.axvline(datetime(2026, 3, 1), color="blue", linestyle=":", alpha=0.4)
    ax.text(datetime(2026, 3, 1), min(cum_unfilt) * 0.95, " IS|OOS",
            fontsize=8, color="blue")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("v1.5-RC chop-specialist filter — cumulative PnL\n"
                 "MVP threshold z=-0.34 on the v1.0.x backtest ledger (1,678 trades, 95 days)")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Per-day bars (lower panel)
    ax2 = axes[1]
    by_day = {}
    for t in trades:
        by_day.setdefault(t["date"], 0.0)
        by_day[t["date"]] += t["pnl"]
    days_sorted = sorted(by_day.keys())
    pnls = [by_day[d] for d in days_sorted]
    colors = ["#2ca02c" if d in pass_dates else "#d62728" for d in days_sorted]
    ax2.bar(days_sorted, pnls, color=colors, alpha=0.8, width=1)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Per-day PnL ($)")
    ax2.set_xlabel("Date")
    ax2.set_title("Per-day PnL: green = filter passes (TRADE), red = filter skips")
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "reports/findings/2026-04-27_cum_pnl_filtered_vs_unfiltered.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")

    # Stats summary
    print()
    print(f"Total trades: {len(trades)}")
    print(f"Days total:   {len(days_sorted)}")
    print(f"Days kept:    {sum(1 for d in days_sorted if d in pass_dates)}")
    print(f"Days skipped: {sum(1 for d in days_sorted if d not in pass_dates)}")
    print()
    print(f"Unfiltered final:  ${cum_unfilt[-1]:+,.2f}")
    print(f"Filter ON final:   ${cum_filt[-1]:+,.2f}")
    print(f"Lift:              ${cum_filt[-1] - cum_unfilt[-1]:+,.2f}")
    print(f"Skipped days $$$:  ${cum_skip[-1]:+,.2f} (= what filter saved us from)")


if __name__ == "__main__":
    main()
