"""
nt8_bleed_harvest_classifier.py — Day-level BLEED vs HARVEST classifier
adapted from tools/tier_day_classifier.py methodology, but for the NT8
trade-export CSV format (Period, Net profit columns).

Process (mirrors tier_day_classifier):
  1. Load NT8 trade-export CSV (e.g. examples/trades.csv).
  2. Aggregate per-day strategy PnL.
  3. Label days BLEED (PnL <= -threshold) / HARVEST (PnL >= +threshold) /
     NEUTRAL (excluded).
  4. Compute day-level FORWARD-AVAILABLE features per day:
       - prior_day_range, prior_day_drift, prior_day_efficiency
       - mean_range_5d, mean_efficiency_5d
       - range_compression_ratio
       - variance_ratio_5_20
       - days_since_high (breakout indicator)
       - day_of_week
       - prior_5d_cum_drift_sign
  5. Cohen d for each feature: bleed cohort vs harvest cohort.
  6. Walk-forward split (first half = IS, second half = OOS).
     Shortlist features where sign(d_IS) == sign(d_OOS) AND
     min(|d_IS|, |d_OOS|) >= 0.3.
  7. Build combined-z-score rule. Threshold sweep. Measure $ lift.

Usage:
    python tools/nt8_bleed_harvest_classifier.py
    python tools/nt8_bleed_harvest_classifier.py --threshold 50
    python tools/nt8_bleed_harvest_classifier.py --is-end 2026-03-13
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def parse_money(s):
    s = s.replace("$", "").replace(",", "").strip()
    if s == "" or s == "n/a":
        return 0.0
    if s.startswith("(") and s.endswith(")"):
        return -float(s[1:-1])
    return float(s)


def load_trades(path):
    """Load NT8 trade-export CSV → list of (date, hour, dow, pnl) tuples."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                ts = datetime.strptime(r["Period"].strip(), "%m/%d/%Y %I:%M %p")
            except Exception:
                continue
            rows.append({
                "date":      ts.date(),
                "hour":      ts.hour,
                "dow":       ts.weekday(),
                "pnl":       parse_money(r["Net profit"]),
                "mae":       parse_money(r.get("Avg. MAE", "0")),
                "mfe":       parse_money(r.get("Avg. MFE", "0")),
                "etd":       parse_money(r.get("Avg. ETD", "0")),
            })
    return rows


def aggregate_per_day(rows):
    """Group trades by date, return per-day summary dict."""
    by_day = defaultdict(lambda: {"pnl": 0.0, "n": 0, "wins": 0, "first_hour": 24, "hours": []})
    for r in rows:
        d = r["date"]
        by_day[d]["pnl"]   += r["pnl"]
        by_day[d]["n"]     += 1
        by_day[d]["wins"]  += 1 if r["pnl"] > 0 else 0
        by_day[d]["first_hour"] = min(by_day[d]["first_hour"], r["hour"])
        by_day[d]["hours"].append(r["hour"])
    # Add dow + sorted list
    out = []
    for d in sorted(by_day):
        s = by_day[d]
        out.append({
            "date":       d,
            "dow":        d.weekday(),
            "pnl":        s["pnl"],
            "n_trades":   s["n"],
            "wr":         s["wins"] / s["n"] if s["n"] else 0.0,
            "first_hour": s["first_hour"],
            "hour_count_morning": sum(1 for h in s["hours"] if 5 <= h <= 11),
            "hour_count_midday":  sum(1 for h in s["hours"] if 12 <= h <= 13),
        })
    return out


def load_ohlc(atlas_root):
    """Load all daily OHLC bars from ATLAS/1D/, return DataFrame."""
    folder = os.path.join(atlas_root, "1D")
    rows = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".parquet"):
            continue
        try:
            df = pd.read_parquet(os.path.join(folder, f))
        except Exception:
            continue
        if df.empty:
            continue
        date_str = f[:-8].replace("_", "-")
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        r = df.iloc[0]
        rows.append({
            "date":  d,
            "open":  float(r["open"]),
            "high":  float(r["high"]),
            "low":   float(r["low"]),
            "close": float(r["close"]),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def add_nt8_dump_daily(ohlc_df, daily_csv_path):
    """Merge in newer days from NT8-dump-derived daily CSV (overrides for overlap)."""
    if not os.path.exists(daily_csv_path):
        return ohlc_df
    nt8 = pd.read_csv(daily_csv_path)
    nt8["date"] = pd.to_datetime(nt8["day"], format="%Y_%m_%d").dt.date
    nt8 = nt8[["date", "open", "high", "low", "close"]]
    combined = pd.concat([ohlc_df, nt8], ignore_index=True)
    return combined.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)


def compute_intraday_variance_ratio(atlas_root: str, dates: list, n_short: int = 5, n_long: int = 60) -> dict:
    """For each date in `dates`, load 1m bars and compute variance ratio.
    VR(N) = var(N-bar return) / (N * var(1-bar return))
    < 1 = mean-reverting, > 1 = trending.

    Returns dict {date -> {vr_5_60, vr_60_240, prior_intraday_range_zscore}}
    """
    folder = os.path.join(atlas_root, "1m")
    out = {}
    for d in dates:
        fname = d.strftime("%Y_%m_%d") + ".parquet"
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            out[d] = {"vr_5_60": np.nan, "vr_60_240": np.nan,
                      "intraday_range": np.nan, "intraday_efficiency": np.nan,
                      "intraday_atr_5m": np.nan}
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            out[d] = {"vr_5_60": np.nan, "vr_60_240": np.nan,
                      "intraday_range": np.nan, "intraday_efficiency": np.nan,
                      "intraday_atr_5m": np.nan}
            continue
        if df.empty:
            out[d] = {"vr_5_60": np.nan, "vr_60_240": np.nan,
                      "intraday_range": np.nan, "intraday_efficiency": np.nan,
                      "intraday_atr_5m": np.nan}
            continue

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        # 1m return
        r1 = np.diff(close) / close[:-1]
        var1 = np.var(r1)

        def vr(n):
            if len(close) < n + 1 or var1 == 0: return np.nan
            r_n = (close[n:] - close[:-n]) / close[:-n]
            return float(np.var(r_n) / (n * var1))

        # 5m vs 1m: VR(5)
        vr5 = vr(5)
        # 60m vs 1m: VR(60)
        vr60 = vr(60)
        # Intraday total range
        range_pts = float(high.max() - low.min())
        # Open-close drift
        drift = float(close[-1] - close[0]) if len(close) else 0.0
        eff = abs(drift) / range_pts if range_pts > 0 else 0.0
        # 5m ATR-like measure: average 5m bar range
        atr_5m = 0.0
        if len(close) >= 5:
            n5 = len(close) // 5
            atr_5m_list = []
            for i in range(n5):
                seg = slice(i*5, (i+1)*5)
                atr_5m_list.append(float(high[seg].max() - low[seg].min()))
            atr_5m = float(np.mean(atr_5m_list)) if atr_5m_list else 0.0

        out[d] = {
            "vr_5_60":          vr5,
            "vr_60_240":        vr60,
            "intraday_range":   range_pts,
            "intraday_efficiency": eff,
            "intraday_atr_5m":  atr_5m,
        }
    return out


def compute_day_features(ohlc):
    """For each day, compute forward-available features (using prior days only)."""
    df = ohlc.copy()
    df["range"]      = df["high"] - df["low"]
    df["drift"]      = df["close"] - df["open"]
    df["efficiency"] = df["drift"].abs() / df["range"].replace(0, np.nan)

    # All "prior_*" features SHIFT by 1 day (available BEFORE today's open)
    df["prior_range"]      = df["range"].shift(1)
    df["prior_drift"]      = df["drift"].shift(1)
    df["prior_efficiency"] = df["efficiency"].shift(1)

    df["mean_range_5d"]      = df["range"].shift(1).rolling(5,  min_periods=3).mean()
    df["mean_range_20d"]     = df["range"].shift(1).rolling(20, min_periods=10).mean()
    df["mean_efficiency_5d"] = df["efficiency"].shift(1).rolling(5, min_periods=3).mean()

    df["range_compression"] = df["prior_range"] / df["mean_range_20d"]   # < 1 = compressed

    # Variance ratio: 5d / 20d on close pct change
    pct = df["close"].pct_change()
    var5  = pct.shift(1).rolling(5,  min_periods=3).var()
    var20 = pct.shift(1).rolling(20, min_periods=10).var()
    df["variance_ratio_5_20"] = (var5 * 4.0) / var20.replace(0, np.nan)

    # Trend persistence: sign of cumulative drift over last 5 days
    df["cum_drift_5d"] = df["drift"].shift(1).rolling(5, min_periods=3).sum()
    df["cum_drift_5d_sign"] = np.sign(df["cum_drift_5d"])

    # Days since last 20-day high (breakout indicator). 0 = today is at 20d high.
    df["close_vs_20d_max"] = df["close"].shift(1).rolling(20, min_periods=10).max() / df["close"].shift(1)
    df["close_above_20d_max"] = (df["close_vs_20d_max"] < 1.0).astype(int)

    return df


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    if pooled_std == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled_std


FEATURES = [
    "prior_range", "prior_drift", "prior_efficiency",
    "mean_range_5d", "mean_range_20d", "mean_efficiency_5d",
    "range_compression", "variance_ratio_5_20",
    "cum_drift_5d", "cum_drift_5d_sign", "close_above_20d_max",
    # Day-of-week as feature; treat 0/1/.../6 numerically (works for separation rank)
    "dow",
    # Multi-TF features (added 2026-04-27 per user hint about RIDE_AGAINST tier work)
    # All forward-available: computed from PRIOR day's intra-day 1m bars.
    "prior_vr_5_60",          # mean-reversion vs trend at 5m vs 1m
    "prior_vr_60_240",        # mean-reversion vs trend at 1h vs 1m
    "prior_intraday_range",   # yesterday's high-low (similar to prior_range)
    "prior_intraday_efficiency",  # yesterday's close-open / range
    "prior_intraday_atr_5m",  # yesterday's mean 5m bar range
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="examples/trades.csv")
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="$/day threshold defining BLEED (<=-T) and HARVEST (>=+T)")
    ap.add_argument("--atlas",  default="DATA/ATLAS")
    ap.add_argument("--nt8-daily", default="reports/findings/nt8_dump_validation/daily_MNQ_06-26.csv")
    ap.add_argument("--is-end", default=None,
                    help="ISO date YYYY-MM-DD; days <= this are IS, > are OOS. Default = median date.")
    ap.add_argument("--out", default="reports/findings/2026-04-27_bleed_harvest")
    ap.add_argument("--forward-only", action="store_true",
                    help="Use ONLY strict pre-open features (no intra-day hour_count_morning etc.)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load + aggregate trades
    trades = load_trades(args.trades)
    days   = aggregate_per_day(trades)
    print(f"Loaded {len(trades)} trades across {len(days)} days")

    # Load OHLC
    ohlc = load_ohlc(args.atlas)
    ohlc = add_nt8_dump_daily(ohlc, args.nt8_daily)
    print(f"OHLC bars after merge: {len(ohlc)}")

    # Compute features
    feat_df = compute_day_features(ohlc)

    # Multi-TF intraday features computed from ATLAS 1m bars per day
    print("Computing intraday VR features from ATLAS 1m...")
    intraday = compute_intraday_variance_ratio(args.atlas, list(feat_df["date"]))
    intraday_df = pd.DataFrame([
        {"date": d,
         "intraday_vr_5_60":      v["vr_5_60"],
         "intraday_vr_60_240":    v["vr_60_240"],
         "intraday_range_atlas":  v["intraday_range"],
         "intraday_efficiency_atlas": v["intraday_efficiency"],
         "intraday_atr_5m":       v["intraday_atr_5m"]}
        for d, v in intraday.items()
    ])
    feat_df = feat_df.merge(intraday_df, on="date", how="left")
    # Shift by 1 day to make these PRIOR-day features (forward-available at session open)
    feat_df = feat_df.sort_values("date").reset_index(drop=True)
    for col in ["intraday_vr_5_60", "intraday_vr_60_240",
                "intraday_range_atlas", "intraday_efficiency_atlas", "intraday_atr_5m"]:
        feat_df[f"prior_vr_5_60"      if col == "intraday_vr_5_60"      else
                f"prior_vr_60_240"    if col == "intraday_vr_60_240"    else
                f"prior_intraday_range" if col == "intraday_range_atlas" else
                f"prior_intraday_efficiency" if col == "intraday_efficiency_atlas" else
                f"prior_intraday_atr_5m"] = feat_df[col].shift(1)

    # Merge with day PnL
    df = pd.DataFrame(days)
    df["dow"] = df["date"].apply(lambda d: d.weekday())
    df = df.merge(feat_df, on="date", how="left")

    # Drop days lacking 20d window
    df = df.dropna(subset=["mean_range_20d", "variance_ratio_5_20"]).reset_index(drop=True)
    print(f"Days with full 20-day feature window: {len(df)}")

    # Add dow_num if not in features
    df["dow_num"] = df["dow"]

    # Label
    df["label"] = "NEUTRAL"
    df.loc[df["pnl"] <= -args.threshold, "label"] = "BLEED"
    df.loc[df["pnl"] >=  args.threshold, "label"] = "HARVEST"
    n_bleed   = (df["label"] == "BLEED").sum()
    n_harvest = (df["label"] == "HARVEST").sum()
    n_neutral = (df["label"] == "NEUTRAL").sum()
    print(f"\nLabels at threshold ${args.threshold:.0f}: "
          f"BLEED={n_bleed}, HARVEST={n_harvest}, NEUTRAL={n_neutral}")
    print(f"Total $ on bleed days:   ${df.loc[df['label']=='BLEED', 'pnl'].sum():+,.0f}")
    print(f"Total $ on harvest days: ${df.loc[df['label']=='HARVEST', 'pnl'].sum():+,.0f}")
    print(f"Total $ on neutral days: ${df.loc[df['label']=='NEUTRAL', 'pnl'].sum():+,.0f}")

    # IS / OOS split
    if args.is_end:
        is_end_date = datetime.strptime(args.is_end, "%Y-%m-%d").date()
    else:
        sorted_dates = sorted(df["date"])
        is_end_date  = sorted_dates[len(sorted_dates) // 2]
    df["set"] = df["date"].apply(lambda d: "IS" if d <= is_end_date else "OOS")
    print(f"\nIS/OOS split at {is_end_date}: "
          f"IS={(df['set']=='IS').sum()} days, OOS={(df['set']=='OOS').sum()} days")

    # Cohen d per feature, IS and OOS
    # Two pools of features:
    #   STRICT FORWARD-AVAILABLE: only prior-day data (deployable at session OPEN)
    #   INTRA-DAY:                 feature seen after morning (deployable at hr 11)
    forward_only = list(FEATURES)
    intra_day_extra = ["first_hour", "hour_count_morning", "hour_count_midday"]
    if args.forward_only:
        feature_list = forward_only
        print("\n[FORWARD-ONLY MODE] Excluding intra-day features (first_hour, hour_count_*).")
    else:
        feature_list = forward_only + intra_day_extra
    print()
    print("=" * 92)
    print(f"{'feature':<28} {'d_IS':>7} {'d_OOS':>7} {'d_all':>7} {'sign_match':>10} {'min_|d|':>8}")
    print("-" * 92)
    rows = []
    for f in feature_list:
        if f not in df.columns:
            continue
        for setname in ("IS", "OOS", "ALL"):
            sub = df[df["label"] != "NEUTRAL"]
            if setname == "IS":
                sub = sub[sub["set"] == "IS"]
            elif setname == "OOS":
                sub = sub[sub["set"] == "OOS"]
        d_is  = cohen_d(df[(df["label"]=="BLEED") & (df["set"]=="IS")][f].values,
                        df[(df["label"]=="HARVEST") & (df["set"]=="IS")][f].values)
        d_oos = cohen_d(df[(df["label"]=="BLEED") & (df["set"]=="OOS")][f].values,
                        df[(df["label"]=="HARVEST") & (df["set"]=="OOS")][f].values)
        d_all = cohen_d(df[df["label"]=="BLEED"][f].values,
                        df[df["label"]=="HARVEST"][f].values)
        sign_match = (np.sign(d_is) == np.sign(d_oos)) and (np.isfinite(d_is) and np.isfinite(d_oos))
        min_d = min(abs(d_is) if np.isfinite(d_is) else 0,
                    abs(d_oos) if np.isfinite(d_oos) else 0)
        rows.append((f, d_is, d_oos, d_all, sign_match, min_d))
        print(f"{f:<28} {d_is:>+7.3f} {d_oos:>+7.3f} {d_all:>+7.3f} "
              f"{'YES' if sign_match else ' no':>10} {min_d:>+8.3f}")

    rows.sort(key=lambda x: -x[5])
    print()
    shortlist = [r for r in rows if r[4] and r[5] >= 0.30]
    print(f"Walk-forward stable shortlist (sign match + min|d| >= 0.30):")
    for r in shortlist:
        print(f"  {r[0]:<28}  d_IS={r[1]:+.3f}  d_OOS={r[2]:+.3f}  min|d|={r[5]:.3f}")
    print()

    # Save outputs
    df.to_csv(os.path.join(args.out, "day_labels_features.csv"), index=False)
    pd.DataFrame(rows, columns=["feature", "d_IS", "d_OOS", "d_all", "sign_match", "min_abs_d"]
                 ).to_csv(os.path.join(args.out, "cohen_d.csv"), index=False)
    pd.DataFrame(shortlist, columns=["feature", "d_IS", "d_OOS", "d_all", "sign_match", "min_abs_d"]
                 ).to_csv(os.path.join(args.out, "shortlist.csv"), index=False)
    print(f"Saved to: {args.out}/")

    # Quick rule backtest: if any shortlist features, build z-score combined rule
    if shortlist:
        print()
        print("=" * 92)
        print("RULE BACKTEST: skip top-X% bleed-scored days (live-readable rule)")
        print("=" * 92)
        # Build z-score rule
        # Use top-K by min|d| where K is configurable. Default 3 (prior research
        # showed simpler rules walk-forward better; 5 features overfits IS).
        K = max(2, min(3, len(shortlist)))
        rule_features = [r[0] for r in shortlist[:K]]
        rule_signs    = [np.sign(r[1]) for r in shortlist[:K]]   # use IS sign
        is_df  = df[df["set"] == "IS"].copy()
        oos_df = df[df["set"] == "OOS"].copy()
        for f, sign in zip(rule_features, rule_signs):
            mu = is_df[f].mean()
            sd = is_df[f].std() or 1.0
            df[f"_z_{f}"] = (df[f] - mu) / sd
            df[f"_score_{f}"] = sign * df[f"_z_{f}"]
        df["bleed_score"] = sum(df[f"_score_{f}"] for f in rule_features)

        # IS thresholds
        is_df = df[df["set"] == "IS"].copy()
        oos_df = df[df["set"] == "OOS"].copy()
        print(f"\nFeatures used: {rule_features}")
        print(f"Signs:         {[int(s) for s in rule_signs]}")
        print()
        print(f"{'X%':>4} {'thr':>8} {'IS_skipped':>10} {'IS_$_lift':>10} "
              f"{'OOS_skipped':>11} {'OOS_$_lift':>11} {'bleed_caught':>13}")
        print("-" * 92)
        for x_pct in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
            thr = is_df["bleed_score"].quantile(1 - x_pct/100.0)
            is_skipped  = is_df["bleed_score"]  > thr
            oos_skipped = oos_df["bleed_score"] > thr
            is_lift  = -is_df.loc[is_skipped, "pnl"].sum()    # we save the negative
            oos_lift = -oos_df.loc[oos_skipped, "pnl"].sum()
            n_bleed_oos_skip = ((oos_df["label"] == "BLEED") & oos_skipped).sum()
            n_bleed_oos      = (oos_df["label"] == "BLEED").sum()
            caught_pct = 100.0 * n_bleed_oos_skip / n_bleed_oos if n_bleed_oos else 0.0
            print(f"{x_pct:>3}% ${thr:>+7.2f} {is_skipped.sum():>10} "
                  f"${is_lift:>+9.0f} {oos_skipped.sum():>11} ${oos_lift:>+10.0f} "
                  f"{n_bleed_oos_skip}/{n_bleed_oos} ({caught_pct:.0f}%)")
        df.to_csv(os.path.join(args.out, "day_labels_with_score.csv"), index=False)


if __name__ == "__main__":
    main()
