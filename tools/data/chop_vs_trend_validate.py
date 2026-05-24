"""
chop_vs_trend_validate.py -- Verify the chop-edge hypothesis.

Loads:
  - Per-day strategy PnL from examples/trades.csv (95 days, 1/2-4/24)
  - Daily OHLC from DATA/ATLAS/1D + DATA/ATLAS_NT8/1s (synthesized)
Computes efficiency ratio per day:
  efficiency = |close - open| / (high - low)
  ~0 = pure chop (price went everywhere, ended where it started)
  ~1 = pure trend (all the move was directional)
Buckets by efficiency, aggregates strategy PnL per bucket.
"""
import csv
import os
import sys
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

PT_VALUE = 2.0  # MNQ $/pt


def parse_money(s):
    s = s.replace("$","").replace(",","").strip()
    if s == "" or s == "n/a": return 0.0
    if s.startswith("(") and s.endswith(")"): return -float(s[1:-1])
    return float(s)


def _load_per_day_strategy_pnl(path: str) -> dict:
    """Aggregate per-trade Net profit by date."""
    by_day = defaultdict(lambda: {"pnl": 0.0, "n": 0, "wins": 0})
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                ts = datetime.strptime(r["Period"].strip(), "%m/%d/%Y %I:%M %p")
            except Exception:
                continue
            d = ts.date()
            pnl = parse_money(r["Net profit"])
            by_day[d]["pnl"]  += pnl
            by_day[d]["n"]    += 1
            if pnl > 0: by_day[d]["wins"] += 1
    return dict(by_day)


def _load_daily_ohlc_atlas(atlas_root: str) -> pd.DataFrame:
    """Load all 1D parquets from ATLAS, return DataFrame of OHLC by date."""
    rows = []
    folder = os.path.join(atlas_root, "1D")
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".parquet"): continue
        try:
            df = pd.read_parquet(os.path.join(folder, f))
        except Exception:
            continue
        if df.empty: continue
        # day_label "2026_01_05"
        date_str = f[:-8].replace("_", "-")  # "2026-01-05"
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        # Single-row daily file
        r = df.iloc[0]
        rows.append({
            "date": d,
            "open":  float(r["open"]),
            "high":  float(r["high"]),
            "low":   float(r["low"]),
            "close": float(r["close"]),
        })
    return pd.DataFrame(rows)


def _load_daily_from_nt8_dump_drift(path: str) -> pd.DataFrame:
    """Load the daily file produced earlier by nt8_dump_parity_and_drift."""
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["day"], format="%Y_%m_%d").dt.date
    return df[["date","open","high","low","close"]].copy()


def main():
    pnl_by_day = _load_per_day_strategy_pnl(r"examples/trades.csv")
    print(f"Strategy PnL days loaded: {len(pnl_by_day)}")

    # Try ATLAS first; fall back to NT8 daily dump file for newer days
    atlas = _load_daily_ohlc_atlas("DATA/ATLAS")
    nt8_daily = _load_daily_from_nt8_dump_drift("reports/findings/nt8_dump_validation/daily_MNQ_06-26.csv")
    print(f"ATLAS daily bars:    {len(atlas)}")
    print(f"NT8 dump daily bars: {len(nt8_daily)}")

    # Combine: prefer NT8 dump for overlap (newer, more authoritative for current contract)
    combined = pd.concat([atlas, nt8_daily], ignore_index=True)
    combined = combined.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)

    # Restrict to dates in strategy PnL
    combined = combined[combined["date"].isin(pnl_by_day.keys())].reset_index(drop=True)
    combined["range"]      = combined["high"] - combined["low"]
    combined["body"]       = (combined["close"] - combined["open"]).abs()
    combined["efficiency"] = combined["body"] / combined["range"].replace(0, np.nan)
    combined["body_signed"] = combined["close"] - combined["open"]

    # Merge with strategy PnL
    combined["strat_pnl"]  = combined["date"].apply(lambda d: pnl_by_day.get(d, {}).get("pnl", 0.0))
    combined["strat_n"]    = combined["date"].apply(lambda d: pnl_by_day.get(d, {}).get("n", 0))
    combined["strat_wins"] = combined["date"].apply(lambda d: pnl_by_day.get(d, {}).get("wins", 0))

    # Drop days with zero trades or no OHLC
    combined = combined[(combined["strat_n"] > 0) & combined["range"].notna() & (combined["range"] > 0)].reset_index(drop=True)

    n = len(combined)
    print(f"Days with both PnL and OHLC: {n}")
    print()

    # ── Efficiency-ratio quartile analysis ─────────────────────────────────
    combined["eff_q"] = pd.qcut(combined["efficiency"], 4, labels=["Q1 chop", "Q2", "Q3", "Q4 trend"])
    print("=" * 80)
    print("STRATEGY PnL by EFFICIENCY RATIO quartile  (low = chop, high = trend)")
    print("=" * 80)
    print(f"{'quartile':<14} {'days':>5} {'eff_range':>16} {'total_$':>10} {'mean':>8} {'wr':>5} {'trades':>7}")
    print("-" * 80)
    for q in ["Q1 chop", "Q2", "Q3", "Q4 trend"]:
        sub = combined[combined["eff_q"] == q]
        if len(sub) == 0: continue
        eff_lo = sub["efficiency"].min()
        eff_hi = sub["efficiency"].max()
        tot = sub["strat_pnl"].sum()
        n_days = len(sub)
        mean = sub["strat_pnl"].mean()
        wr = (sub["strat_pnl"] > 0).mean()
        n_trades = sub["strat_n"].sum()
        print(f"{q:<14} {n_days:>5} {eff_lo:>6.3f}–{eff_hi:>5.3f}   ${tot:>+9.0f} ${mean:>+7.0f} {wr*100:>4.0f}% {n_trades:>7}")
    print()

    # ── Range-expansion analysis (alternative chop indicator) ──────────────
    # Range relative to recent: rolling 20-day mean range
    combined = combined.sort_values("date").reset_index(drop=True)
    combined["range_20d_mean"] = combined["range"].shift(1).rolling(20, min_periods=10).mean()
    combined["range_ratio"]    = combined["range"] / combined["range_20d_mean"]
    sub = combined.dropna(subset=["range_ratio"])
    if len(sub) > 0:
        sub = sub.copy()
        sub["range_q"] = pd.qcut(sub["range_ratio"], 4, labels=["Q1 compressed", "Q2", "Q3", "Q4 expanded"])
        print("=" * 80)
        print("STRATEGY PnL by RANGE-EXPANSION quartile (today's range / 20d mean)")
        print("=" * 80)
        print(f"{'quartile':<16} {'days':>5} {'r_range':>14} {'total_$':>10} {'mean':>8} {'wr':>5} {'trades':>7}")
        print("-" * 80)
        for q in ["Q1 compressed", "Q2", "Q3", "Q4 expanded"]:
            ssub = sub[sub["range_q"] == q]
            if len(ssub) == 0: continue
            lo = ssub["range_ratio"].min()
            hi = ssub["range_ratio"].max()
            tot = ssub["strat_pnl"].sum()
            mean = ssub["strat_pnl"].mean()
            wr = (ssub["strat_pnl"] > 0).mean()
            print(f"{q:<16} {len(ssub):>5} {lo:>5.3f}–{hi:>5.3f}    ${tot:>+9.0f} ${mean:>+7.0f} {wr*100:>4.0f}% {ssub['strat_n'].sum():>7}")
        print()

    # ── Working-vs-bleed window comparison ─────────────────────────────────
    combined["window"] = combined["date"].apply(
        lambda d: "WORKING (1/2-2/26)" if d <= datetime(2026,2,26).date() else "BLEED (2/27-4/24)"
    )
    print("=" * 80)
    print("Mean efficiency / range by window")
    print("=" * 80)
    g = combined.groupby("window").agg(
        n_days=("date", "count"),
        mean_eff=("efficiency", "mean"),
        median_eff=("efficiency", "median"),
        mean_range=("range", "mean"),
        total_pnl=("strat_pnl", "sum"),
        days_pos=("strat_pnl", lambda s: (s > 0).sum()),
    ).round(3)
    print(g.to_string())
    print()

    # ── Save ───────────────────────────────────────────────────────────────
    out = "reports/findings/chop_vs_trend"
    os.makedirs(out, exist_ok=True)
    combined.to_csv(os.path.join(out, "_per_day.csv"), index=False)
    print(f"Saved: {out}/_per_day.csv  ({len(combined)} days)")


if __name__ == "__main__":
    main()
