"""
regime_classifier_oracle.py -- Test simple regime classifiers on the 32-day
MNQ_06-26 daily drift data.

Each classifier produces a per-day action (LONG / FLAT / SHORT) using ONLY
data from days strictly before the decision day. PnL is action_sign × drift_D.

Compares to:
  - Always-long benchmark         (the "regime exposure" floor)
  - Perfect oracle                (upper bound)
  - Strategy results from NT8 backtests (counter / with)

Output:
  reports/findings/regime_classifier/_summary.csv
  reports/findings/regime_classifier/_per_day.csv
  Console table with totals + alphas.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATA = r"reports/findings/nt8_dump_validation/daily_MNQ_06-26.csv"
OUT  = r"reports/findings/regime_classifier"


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA).sort_values("day").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["day"], format="%Y_%m_%d")
    drift = df["drift_usd"].values
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values

    n = len(df)

    # ── Classifier rules ────────────────────────────────────────────────────
    # Each returns +1 (LONG), -1 (SHORT), or 0 (FLAT) for day i, using ONLY
    # information available BEFORE the open of day i (= shifted prior-day data).

    def rule_always_long(i):
        return +1

    def rule_always_flat(i):
        return 0

    def rule_oracle(i):
        return int(np.sign(drift[i])) if drift[i] != 0 else 0

    def rule_yesterday_sign(i):
        if i == 0: return +1
        return int(np.sign(drift[i-1])) if drift[i-1] != 0 else 0

    def rule_5d_drift_sign(i):
        # Sum of prior 5 days' drift
        if i < 5: return +1
        s = drift[i-5:i].sum()
        return int(np.sign(s)) if s != 0 else 0

    def rule_5d_drift_long_only(i):
        # Long if 5d drift positive, FLAT otherwise (no shorts)
        if i < 5: return +1
        s = drift[i-5:i].sum()
        return +1 if s > 0 else 0

    def rule_3d_drift_sign(i):
        if i < 3: return +1
        s = drift[i-3:i].sum()
        return int(np.sign(s)) if s != 0 else 0

    def rule_close_above_5d_ma(i):
        # Long if yesterday's close > 5-day MA of close, short otherwise
        if i < 6: return +1
        prior_close = close[i-1]
        ma = close[i-6:i-1].mean()
        return +1 if prior_close > ma else -1

    def rule_close_above_5d_ma_long_only(i):
        if i < 6: return +1
        prior_close = close[i-1]
        ma = close[i-6:i-1].mean()
        return +1 if prior_close > ma else 0

    def rule_3of5_up(i):
        # Long if at least 3 of last 5 days were up
        if i < 5: return +1
        ups = (drift[i-5:i] > 0).sum()
        return +1 if ups >= 3 else 0

    def rule_4of5_up_long_only(i):
        if i < 5: return +1
        ups = (drift[i-5:i] > 0).sum()
        return +1 if ups >= 4 else 0

    def rule_range_compression(i):
        # Long if 5-day mean range is in lower half of last-20-day distribution
        if i < 20: return +1
        ranges = (high[i-20:i] - low[i-20:i])
        recent_5 = ranges[-5:].mean()
        return +1 if recent_5 <= np.median(ranges) else 0

    rules = {
        "Always LONG (= drift)":         rule_always_long,
        "Always FLAT":                   rule_always_flat,
        "Perfect ORACLE":                rule_oracle,
        "Yesterday's sign":              rule_yesterday_sign,
        "Sign of 5d drift sum":          rule_5d_drift_sign,
        "Sign of 5d drift sum, LONG/FLAT": rule_5d_drift_long_only,
        "Sign of 3d drift sum":          rule_3d_drift_sign,
        "Close > 5d MA (long/short)":    rule_close_above_5d_ma,
        "Close > 5d MA, LONG/FLAT":      rule_close_above_5d_ma_long_only,
        "3 of 5 prior days up, LONG/FLAT":rule_3of5_up,
        "4 of 5 prior days up, LONG/FLAT":rule_4of5_up_long_only,
        "Range compression, LONG/FLAT":  rule_range_compression,
    }

    # ── Evaluate ────────────────────────────────────────────────────────────
    per_day = pd.DataFrame({
        "day":   df["day"],
        "drift": drift,
    })
    summary_rows = []
    for name, fn in rules.items():
        actions = np.array([fn(i) for i in range(n)], dtype=int)
        pnl = actions * drift
        per_day[name] = pnl
        n_long  = int((actions == +1).sum())
        n_short = int((actions == -1).sum())
        n_flat  = int((actions ==  0).sum())
        days_pos = int((pnl > 0).sum())
        summary_rows.append({
            "rule":         name,
            "total_pnl":    pnl.sum(),
            "mean_pnl":     pnl.mean(),
            "median_pnl":   np.median(pnl),
            "days_pos":     days_pos,
            "n_long":       n_long,
            "n_short":      n_short,
            "n_flat":       n_flat,
            "n_days":       n,
        })
    summary = pd.DataFrame(summary_rows)
    bench = summary[summary["rule"] == "Always LONG (= drift)"]["total_pnl"].iloc[0]
    summary["alpha_vs_long"] = summary["total_pnl"] - bench
    summary["alpha_per_day"] = summary["alpha_vs_long"] / n

    # ── Print ───────────────────────────────────────────────────────────────
    print(f"32 days. Always-long total = ${bench:+,.0f}, oracle ceiling = "
          f"${summary[summary['rule']=='Perfect ORACLE']['total_pnl'].iloc[0]:+,.0f}.\n")
    print(f"{'rule':<36} {'total':>10} {'$/day':>9} {'days+':>7} "
          f"{'L/S/F':>9} {'alpha':>9} {'a/day':>8}")
    print("-" * 96)
    for _, r in summary.iterrows():
        print(f"{r['rule']:<36} ${r['total_pnl']:>+9,.0f} ${r['mean_pnl']:>+8.0f} "
              f"{r['days_pos']:>3}/{int(n):<3} "
              f"{r['n_long']:>2}/{r['n_short']:>2}/{r['n_flat']:>2} "
              f"${r['alpha_vs_long']:>+8,.0f} ${r['alpha_per_day']:>+7.0f}")

    summary.to_csv(os.path.join(OUT, "_summary.csv"), index=False)
    per_day.to_csv (os.path.join(OUT, "_per_day.csv" ), index=False)
    print(f"\nSaved {OUT}/_summary.csv and _per_day.csv")


if __name__ == "__main__":
    main()
