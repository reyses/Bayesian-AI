"""
tier9_bleed_filter.py -- Apply v1.5-RC bleed-score filter to the 9-tier
iso pipeline trade ledger. Tests the hypothesis from
reports/findings/2026-04-27_9tier_review.md that the same regime filter
that rescues ZigzagRunner also rescues the counter-trend tiers.

For each tier x threshold combination, reports:
  - unfiltered $/day  (baseline)
  - filtered $/day    (tier when only trading low-bleed days)
  - inverse-filtered $/day  (tier when only trading HIGH-bleed days,
                              hypothesis: trend tiers prefer this)
  - lift               (filtered minus unfiltered)

Bleed score formula (same as v1.5-RC):
  prior_range       = yesterday's high - low
  range_compression = prior_range / mean(range over last 20 days)
  bleed_score       = z(prior_range) + z(range_compression)
                    using IS-calibrated constants 385.32 / 219.83 / 1.0315 / 0.5502

Usage:
    python tools/tier9_bleed_filter.py
    python tools/tier9_bleed_filter.py --threshold -0.34
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime, date
from collections import defaultdict
import pandas as pd

# IS-calibrated constants (from v1.5-RC, validated stable across 2025-2026)
MEAN_PRIOR_RANGE       = 385.32
STD_PRIOR_RANGE        = 219.83
MEAN_RANGE_COMPRESSION = 1.0315
STD_RANGE_COMPRESSION  = 0.5502


def load_atlas_daily(folder: str) -> pd.DataFrame:
    """Load DATA/ATLAS/1D/*.parquet into a daily OHLC dataframe."""
    rows = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".parquet"):
            continue
        try:
            df = pd.read_parquet(os.path.join(folder, fname))
        except Exception:
            continue
        if df.empty:
            continue
        date_str = fname[:-8]  # strip .parquet
        try:
            d = datetime.strptime(date_str, "%Y_%m_%d").date()
        except ValueError:
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
        r = df.iloc[0]
        rows.append({
            "date": d,
            "high": float(r["high"]),
            "low":  float(r["low"]),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["range"] = df["high"] - df["low"]
    return df


def compute_bleed_score(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute prior_range, range_compression, bleed_score per session day.
    Both features are forward-available (computed on yesterday's close)."""
    df = daily.copy()
    df["prior_range"] = df["range"].shift(1)
    df["mean_range_20d"] = df["range"].shift(1).rolling(20, min_periods=10).mean()
    df["range_compression"] = df["prior_range"] / df["mean_range_20d"]
    df["z_prior_range"] = (df["prior_range"] - MEAN_PRIOR_RANGE) / STD_PRIOR_RANGE
    df["z_range_compression"] = (df["range_compression"] - MEAN_RANGE_COMPRESSION) / STD_RANGE_COMPRESSION
    df["bleed_score"] = df["z_prior_range"] + df["z_range_compression"]
    return df.dropna(subset=["bleed_score"])


def load_trade_ledger(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["day_dt"] = pd.to_datetime(df["day"], format="%Y_%m_%d", errors="coerce").dt.date
    df = df.dropna(subset=["day_dt"]).reset_index(drop=True)
    return df


def per_tier_filter_table(trades: pd.DataFrame, day_score: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return per-tier lift table at a single threshold."""
    score_map = dict(zip(day_score["date"], day_score["bleed_score"]))

    # Annotate each trade with its day's bleed score (NaN if missing)
    trades = trades.copy()
    trades["bleed_score"] = trades["day_dt"].map(score_map)
    trades = trades.dropna(subset=["bleed_score"])

    rows = []
    tiers = sorted(trades["entry_tier"].unique())
    for tier in tiers:
        sub = trades[trades["entry_tier"] == tier]
        n = len(sub)
        days = sub["day_dt"].nunique()
        unfiltered_pnl = float(sub["pnl"].sum())
        per_day_unf = unfiltered_pnl / max(days, 1)

        # Forward filter: trade on low-bleed days only
        kept = sub[sub["bleed_score"] <= threshold]
        n_kept = len(kept)
        days_kept = kept["day_dt"].nunique()
        kept_pnl = float(kept["pnl"].sum())
        per_day_fwd = kept_pnl / max(days_kept, 1) if days_kept > 0 else 0.0

        # Inverse filter: trade on high-bleed days only
        kept_inv = sub[sub["bleed_score"] > threshold]
        n_inv = len(kept_inv)
        days_inv = kept_inv["day_dt"].nunique()
        inv_pnl = float(kept_inv["pnl"].sum())
        per_day_inv = inv_pnl / max(days_inv, 1) if days_inv > 0 else 0.0

        rows.append({
            "tier": tier,
            "n_trades": n,
            "days": days,
            "unf_pnl": unfiltered_pnl,
            "unf_$/day": per_day_unf,
            "n_kept_fwd": n_kept,
            "days_kept_fwd": days_kept,
            "fwd_pnl": kept_pnl,
            "fwd_$/day": per_day_fwd,
            "fwd_lift": kept_pnl - unfiltered_pnl,
            "n_kept_inv": n_inv,
            "days_kept_inv": days_inv,
            "inv_pnl": inv_pnl,
            "inv_$/day": per_day_inv,
            "inv_lift": inv_pnl - unfiltered_pnl,
        })
    return pd.DataFrame(rows)


def fmt_dollar(v: float) -> str:
    return f"${v:+,.0f}"


def print_tier_table(df: pd.DataFrame, threshold: float, label: str):
    print()
    print("=" * 124)
    print(f"{label}  (threshold z = {threshold:+.2f})")
    print("=" * 124)
    if len(df) == 0:
        print("(no overlap between trade ledger dates and atlas dates -- table empty)")
        return
    print(f"{'tier':<22} {'N':>6} {'days':>5} | {'unf_$':>9} {'unf_$/d':>8} | "
          f"{'fwd_$':>9} {'fwd_$/d':>8} {'fwd_lift':>10} {'days_f':>7} | "
          f"{'inv_$':>9} {'inv_$/d':>8} {'inv_lift':>10} {'days_i':>7}")
    print("-" * 124)
    for _, r in df.iterrows():
        print(f"{r['tier']:<22} {int(r['n_trades']):>6} {int(r['days']):>5} | "
              f"{fmt_dollar(r['unf_pnl']):>9} {fmt_dollar(r['unf_$/day']):>8} | "
              f"{fmt_dollar(r['fwd_pnl']):>9} {fmt_dollar(r['fwd_$/day']):>8} {fmt_dollar(r['fwd_lift']):>10} {int(r['days_kept_fwd']):>7} | "
              f"{fmt_dollar(r['inv_pnl']):>9} {fmt_dollar(r['inv_$/day']):>8} {fmt_dollar(r['inv_lift']):>10} {int(r['days_kept_inv']):>7}")
    # Engine total
    tot_n = int(df["n_trades"].sum())
    tot_days_unf = int(df["days"].max()) if len(df) > 0 else 0  # tiers share the same day-set
    tot_unf = float(df["unf_pnl"].sum())
    tot_fwd = float(df["fwd_pnl"].sum())
    tot_inv = float(df["inv_pnl"].sum())
    print("-" * 124)
    print(f"{'ENGINE TOTAL':<22} {tot_n:>6} {tot_days_unf:>5} | "
          f"{fmt_dollar(tot_unf):>9} {fmt_dollar(tot_unf/max(tot_days_unf,1)):>8} | "
          f"{fmt_dollar(tot_fwd):>9} {'':>8} {fmt_dollar(tot_fwd-tot_unf):>10} {'':>7} | "
          f"{fmt_dollar(tot_inv):>9} {'':>8} {fmt_dollar(tot_inv-tot_unf):>10} {'':>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="DATA/ATLAS/1D")
    ap.add_argument("--is", dest="is_path", default="training_iso/output/trades/iso_is.csv")
    ap.add_argument("--oos", default="training_iso/output/trades/iso_oos.csv")
    ap.add_argument("--threshold", type=float, default=-0.34,
                    help="bleed-score threshold; days with score <= threshold pass forward filter")
    ap.add_argument("--sweep", action="store_true",
                    help="Run threshold sweep across [-1.0, +1.0]")
    ap.add_argument("--out-md", default="reports/findings/2026-04-27_9tier_bleed_filter.md")
    args = ap.parse_args()

    print("=" * 124)
    print("9-TIER BLEED-SCORE FILTER -- testing v1.5-RC hypothesis on the iso pipeline")
    print("=" * 124)

    # Load daily OHLC + bleed score
    daily = load_atlas_daily(args.atlas)
    day_score = compute_bleed_score(daily)
    print(f"Loaded {len(daily)} daily bars. {len(day_score)} days have bleed_score.")
    print(f"  date range: {day_score['date'].min()} -> {day_score['date'].max()}")

    # Bleed-score distribution overview
    print()
    print(f"Bleed score distribution: min {day_score['bleed_score'].min():+.2f}  "
          f"q25 {day_score['bleed_score'].quantile(0.25):+.2f}  "
          f"med {day_score['bleed_score'].median():+.2f}  "
          f"q75 {day_score['bleed_score'].quantile(0.75):+.2f}  "
          f"max {day_score['bleed_score'].max():+.2f}")

    # Load trade ledgers
    is_trades  = load_trade_ledger(args.is_path)
    oos_trades = load_trade_ledger(args.oos)
    print(f"\nLoaded {len(is_trades)} IS trades, {len(oos_trades)} OOS trades.")
    print(f"  IS tiers:  {sorted(is_trades['entry_tier'].unique())}")
    print(f"  OOS tiers: {sorted(oos_trades['entry_tier'].unique())}")

    # Single threshold table (IS + OOS)
    is_tbl  = per_tier_filter_table(is_trades,  day_score, args.threshold)
    oos_tbl = per_tier_filter_table(oos_trades, day_score, args.threshold)
    print_tier_table(is_tbl,  args.threshold, "IS  --  per-tier filter results")
    print_tier_table(oos_tbl, args.threshold, "OOS --  per-tier filter results")

    if args.sweep:
        print()
        print("=" * 124)
        print("THRESHOLD SWEEP (IS engine totals)")
        print("=" * 124)
        print(f"{'threshold':>10} | {'days_fwd':>9} {'unf_$':>10} {'fwd_$':>10} {'fwd_lift':>10} | {'inv_$':>10} {'inv_lift':>10}")
        print("-" * 90)
        for thr in [-1.0, -0.75, -0.5, -0.34, -0.25, 0.0, +0.25, +0.5, +0.75, +1.0]:
            t = per_tier_filter_table(is_trades, day_score, thr)
            unf = float(t["unf_pnl"].sum())
            fwd = float(t["fwd_pnl"].sum())
            inv = float(t["inv_pnl"].sum())
            d_fwd = day_score[day_score["bleed_score"] <= thr]["date"].nunique()
            print(f"{thr:>+9.2f}  | {d_fwd:>9} {fmt_dollar(unf):>10} {fmt_dollar(fwd):>10} {fmt_dollar(fwd-unf):>10} | {fmt_dollar(inv):>10} {fmt_dollar(inv-unf):>10}")

    # Engine-day-level analysis (sum all tiers per day, then apply day filter)
    print()
    print("=" * 124)
    print("ENGINE-DAY LEVEL ANALYSIS  (sum all tiers per day, apply day filter)")
    print("=" * 124)
    score_map = dict(zip(day_score["date"], day_score["bleed_score"]))
    for label, ledger in [("IS", is_trades), ("OOS", oos_trades)]:
        ledger = ledger.copy()
        ledger["bleed_score"] = ledger["day_dt"].map(score_map)
        ledger = ledger.dropna(subset=["bleed_score"])
        by_day = ledger.groupby("day_dt").agg(pnl=("pnl", "sum"), n=("pnl", "count"), bleed=("bleed_score", "first")).reset_index()
        unf_total = float(by_day["pnl"].sum())
        unf_days  = len(by_day)
        unf_winR  = (by_day["pnl"] > 0).mean()
        print()
        print(f"--- {label} engine-day ({unf_days} days, ${unf_total:+,.0f} unfiltered, day WR {unf_winR*100:.1f}%) ---")
        print(f"{'threshold':>10} | {'days_fwd':>9} {'fwd_$':>10} {'fwd_$/d':>8} {'fwd_winR':>9} {'fwd_lift':>10} | "
              f"{'days_inv':>9} {'inv_$':>10} {'inv_$/d':>8} {'inv_winR':>9} {'inv_lift':>10}")
        for thr in [-1.0, -0.75, -0.5, -0.34, -0.25, 0.0, +0.25, +0.5, +0.75, +1.0]:
            fwd = by_day[by_day["bleed"] <= thr]
            inv = by_day[by_day["bleed"] > thr]
            f_pnl = float(fwd["pnl"].sum()); f_n = len(fwd); f_wr = (fwd["pnl"] > 0).mean() if f_n else 0
            i_pnl = float(inv["pnl"].sum()); i_n = len(inv); i_wr = (inv["pnl"] > 0).mean() if i_n else 0
            print(f"{thr:>+9.2f}  | {f_n:>9} {fmt_dollar(f_pnl):>10} {fmt_dollar(f_pnl/max(f_n,1)):>8} {f_wr*100:>8.1f}% {fmt_dollar(f_pnl-unf_total):>10} | "
                  f"{i_n:>9} {fmt_dollar(i_pnl):>10} {fmt_dollar(i_pnl/max(i_n,1)):>8} {i_wr*100:>8.1f}% {fmt_dollar(i_pnl-unf_total):>10}")

    # Write markdown report (manual, no tabulate dependency)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        def df_to_md(df):
            cols = list(df.columns)
            out = "| " + " | ".join(cols) + " |\n"
            out += "|" + "|".join(["---"] * len(cols)) + "|\n"
            for _, r in df.iterrows():
                vals = []
                for c in cols:
                    v = r[c]
                    if isinstance(v, float):
                        vals.append(f"{v:.0f}")
                    else:
                        vals.append(str(v))
                out += "| " + " | ".join(vals) + " |\n"
            return out
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(f"# 9-tier bleed-score filter results\n\n")
            f.write(f"Generated: {datetime.now().isoformat(timespec='minutes')}\n\n")
            f.write(f"Threshold tested: z = {args.threshold:+.2f}\n\n")
            f.write(f"Source: `{args.is_path}` and `{args.oos}` against `{args.atlas}`\n\n")
            f.write(f"## IS results\n\n")
            f.write(df_to_md(is_tbl))
            f.write("\n\n")
            f.write(f"## OOS results\n\n")
            f.write(df_to_md(oos_tbl))
            f.write("\n")
        print(f"\nReport written: {args.out_md}")


if __name__ == "__main__":
    main()
