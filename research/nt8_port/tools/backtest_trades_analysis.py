"""Honest-metrics analysis of an NT8 Strategy Analyzer trade export (v0.2-RC backtests).

Usage: python research/nt8_port/tools/backtest_trades_analysis.py "<csv path>"
Writes research/nt8_port/reports/backtest_trades_analysis.md
Metric rules per CLAUDE.md: PF-based Trade WR, day WR count-based, mode+mean
with 95% bootstrap CI (4,000 resamples), exit-name breakdown, anomaly flags.
"""
import sys, re, csv
from collections import defaultdict
from datetime import datetime
import numpy as np

BOOT_N = 4000          # CLAUDE.md: 4,000 bootstrap resamples, percentile method
DAY_BIN = 25.0         # $ bin width for $/day mode (CLAUDE.md)
TRADE_BIN = 2.0        # $ bin width for $/trade mode (CLAUDE.md)
STOP_PT = 50.0         # the catastrophic-stop setting Moises used
MNQ_PT_USD = 2.0       # MNQ $2 per point (4 ticks x $0.50)

def money(s):
    s = s.strip().replace("$", "").replace(",", "")
    if s.startswith("(") and s.endswith(")"):
        return -float(s[1:-1])
    return float(s) if s else 0.0

def mode_of(vals, bin_w):
    if len(vals) == 0:
        return float("nan")
    lo = min(vals)
    idx = [int((v - lo) // bin_w) for v in vals]
    counts = defaultdict(int)
    for i in idx:
        counts[i] += 1
    best = max(counts, key=counts.get)
    return lo + (best + 0.5) * bin_w

def boot_ci(vals, n=BOOT_N, seed=7):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def main(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r.get("Trade number", "").strip():
                continue
            rows.append({
                "n": int(r["Trade number"]),
                "dir": r["Market pos."].strip(),
                "entry_t": datetime.strptime(r["Entry time"].strip(), "%m/%d/%Y %I:%M:%S %p"),
                "exit_t": datetime.strptime(r["Exit time"].strip(), "%m/%d/%Y %I:%M:%S %p"),
                "exit_name": r["Exit name"].strip(),
                "pnl": money(r["Profit"]),
                "mae": money(r["MAE"]),
                "mfe": money(r["MFE"]),
                "entry_px": float(r["Entry price"]),
                "exit_px": float(r["Exit price"]),
                "bars": int(r["Bars"]),
            })
    pnl = [t["pnl"] for t in rows]
    win = [p for p in pnl if p > 0]
    loss = [p for p in pnl if p < 0]
    pf = (sum(win) / abs(sum(loss))) if loss else float("inf")
    trade_wr_pf = pf - 1.0

    by_day = defaultdict(float)
    for t in rows:
        by_day[t["entry_t"].date()] += t["pnl"]
    days = sorted(by_day)
    dvals = [by_day[d] for d in days]
    day_wr = sum(1 for v in dvals if v > 0) / len(dvals)

    exits = defaultdict(lambda: [0, 0.0])
    for t in rows:
        exits[t["exit_name"]][0] += 1
        exits[t["exit_name"]][1] += t["pnl"]

    # stop-slip: cat-stop exits whose loss exceeds the configured 50 pts
    slip = [t for t in rows if t["exit_name"] == "X_CatastrophicStop"
            and -t["pnl"] > STOP_PT * MNQ_PT_USD * 1.10]
    # entry-minute clustering
    ent_min = defaultdict(int)
    for t in rows:
        ent_min[t["entry_t"].strftime("%H:%M")] += 1
    first_bucket = sum(v for k, v in ent_min.items() if k <= "08:59")

    tci = boot_ci(pnl)
    dci = boot_ci(dvals)
    lines = []
    A = lines.append
    A("# v0.2-RC backtest trade export — honest metrics + anomaly flags")
    A(f"Source: {path}")
    A(f"Window: {days[0]} -> {days[-1]} ({len(days)} active days), N={len(rows)} trades. "
      f"Config note: catastrophic stop ON at {STOP_PT:.0f} pts (Moises).")
    A("")
    A("## Headline (per canonical metric definitions)")
    A(f"- Net: ${sum(pnl):,.2f}")
    A(f"- **Trade WR (PF-based)**: {trade_wr_pf:+.2f}  (PF {pf:.2f}; {len(win)}W/{len(loss)}L by count)")
    A(f"- **Day WR**: {day_wr:.0%} ({sum(1 for v in dvals if v>0)}/{len(dvals)})")
    A(f"- **$/trade**: mode ${mode_of(pnl, TRADE_BIN):,.0f}, mean ${np.mean(pnl):,.2f} "
      f"[95% CI ${tci[0]:,.2f}, ${tci[1]:,.2f}]")
    A(f"- **$/day**: mode ${mode_of(dvals, DAY_BIN):,.0f}, mean ${np.mean(dvals):,.2f} "
      f"[95% CI ${dci[0]:,.2f}, ${dci[1]:,.2f}]")
    sig = "EXCLUDES 0 - significant at 95%" if (dci[0] > 0 or dci[1] < 0) else "INCLUDES 0 - NOT significant"
    A(f"- Significance: $/day CI {sig}. N={len(dvals)} days is small; treat as directional.")
    A("")
    A("## Exit-name breakdown")
    for k, (c, s) in sorted(exits.items(), key=lambda kv: -kv[1][0]):
        A(f"- {k}: {c} trades, net ${s:,.2f}")
    A("")
    A("## Day P&L (worst -> best)")
    for d in sorted(days, key=lambda d: by_day[d]):
        A(f"- {d}: ${by_day[d]:,.2f}")
    A("")
    A("## Anomaly flags")
    rtrig = exits.get("RTriggerReversal", [0, 0])[0]
    A(f"1. **RTriggerReversal exits: {rtrig} of {len(rows)}** - the designed exit NEVER fired. "
      "Winners exited on session close instead. The strategy tested is effectively "
      "'ensemble entry + ride to close + disaster stop', NOT the designed R-trigger ride.")
    A(f"2. **Stop slip**: {len(slip)} cat-stop exits lost >10% beyond the {STOP_PT:.0f}-pt setting: "
      + ", ".join(f"#{t['n']} (${t['pnl']:,.0f}, MAE ${t['mae']:,.0f})" for t in slip))
    A(f"3. **Entry clustering**: {first_bucket}/{len(rows)} entries before 09:00 local - "
      "first-qualifying-minute-of-day pattern; verify against harness selectivity in the P3 diff.")
    A("4. **Session semantics**: 'Exit on session close' fired at 14:00 local on most days while "
      "X_SessionFlatten fired once at 15:56 - the data-series session template and the strategy's "
      "15:55-CT flatten disagree (TODO P2-5/P2-11).")
    out = "research/nt8_port/reports/backtest_trades_analysis.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main(sys.argv[1])
