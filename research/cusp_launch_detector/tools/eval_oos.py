"""Direction 1 (scale-ready): honestly evaluate the cubic+classifier cusp/launch detector.

Day-disjoint is the rule (the day is the unit of independence). This harness is built to stay honest
as the labeler grows the day count:
  - day-group CV (LODO when <=6 days, else k=5 day-folds) — never splits within a day.
  - day-block bootstrap CI on pooled OOS AUC — honest error bars (resamples DAYS, not rows).
  - TWO targets: 'target' (matches a human pick) AND 'paid' (objective forward MFE >= threshold,
    computed from price — independent of any model). The paid target is what we actually care about.
  - the leaky random 80/20 number is kept only as a contrast (shows the same-day leakage).
Run: python research/cusp_launch_detector/tools/eval_oos.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CSV = os.path.join(ROOT, "DATA", "cusp_picks", "features", "candidate_primitives.csv")
ONE_M = os.path.join(ROOT, "DATA", "ATLAS", "1m")
REPORT = os.path.join(ROOT, "research", "cusp_launch_detector", "reports", "eval_oos.md")

FEATURES = ["z_15s", "z_1m", "z_15m", "slope_15s_3m", "slope_15s_10m", "slope_1m_10m",
            "slope_15m_5m", "slope_15m_15m", "slope_15m_decel", "curv_15m",
            "band_width", "band_rank_60", "sigma_15m_rank_60", "fan_width",
            "align_up_count", "align_down_count"]
FWD_MINS = 60        # forward window for MFE (matches the human picks' fwd_mins)
MFE_THRESH_PTS = 5.0  # 'paid' = forward MFE >= this many points in the cusp's direction
RNG = np.random.default_rng(0)


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def add_mfe_target(df):
    """Objective forward MFE per candidate from price -> 'paid' label. No model, no human."""
    mfe = np.full(len(df), np.nan)
    for date, sub in df.groupby("date"):
        p = os.path.join(ONE_M, f"{date.replace('-', '_')}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        ts = d["timestamp"].to_numpy(np.float64)
        hi = d["high"].to_numpy(np.float64); lo = d["low"].to_numpy(np.float64)
        cl = d["close"].to_numpy(np.float64)
        for i, row in sub.iterrows():
            k = int(np.searchsorted(ts, row["timestamp"]))
            if k <= 0 or k >= len(ts) - 1:
                continue
            entry = cl[k]
            w_hi = hi[k + 1:k + 1 + FWD_MINS]; w_lo = lo[k + 1:k + 1 + FWD_MINS]
            if w_hi.size == 0:
                continue
            up = str(row.get("direction", "")).upper() == "LONG"
            mfe[df.index.get_loc(i)] = (w_hi.max() - entry) if up else (entry - w_lo.min())
    df["mfe_pts"] = mfe
    df["paid"] = (df["mfe_pts"] >= MFE_THRESH_PTS).astype(float)
    df.loc[df["mfe_pts"].isna(), "paid"] = np.nan
    return df


def day_cv_oos(X, y, days):
    uniq = sorted(set(days))
    oos = np.full(len(y), np.nan)
    if len(uniq) <= 6:
        folds = [[d] for d in uniq]                       # leave-one-day-out
    else:
        du = RNG.permutation(uniq); folds = [list(du[i::5]) for i in range(5)]
    perfold = []
    for fold in folds:
        te = np.isin(days, fold); tr = ~te
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        oos[te] = fit_predict(X[tr], y[tr], X[te])
        if len(np.unique(y[te])) > 1:
            perfold.append(roc_auc_score(y[te], oos[te]))
    return oos, perfold


def dayblock_ci(y, oos, days, n=4000):
    uniq = np.array(sorted(set(days))); aucs = []
    by = {d: np.where(days == d)[0] for d in uniq}
    for _ in range(n):
        idx = np.concatenate([by[d] for d in RNG.choice(uniq, len(uniq), replace=True)])
        yy, pp = y[idx], oos[idx]
        m = ~np.isnan(pp)
        if len(np.unique(yy[m])) > 1:
            aucs.append(roc_auc_score(yy[m], pp[m]))
    return (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)) if aucs else (np.nan, np.nan)


def evaluate(df, target, w):
    sub = df.dropna(subset=FEATURES + [target]).copy()
    X = sub[FEATURES].values
    y = sub[target].values.astype(int)
    days = sub["date"].values
    if len(np.unique(y)) < 2:
        w(f"### target '{target}': only one class — skip"); return
    oos, perfold = day_cv_oos(X, y, days)
    m = ~np.isnan(oos)
    pooled = roc_auc_score(y[m], oos[m])
    lo, hi = dayblock_ci(y[m], oos[m], days[m])
    gap = pooled - 0.5
    verdict = "REAL" if gap >= 0.10 else "CONDITIONAL" if gap >= 0.05 else "NOISE"
    w(f"### target '{target}'  (n={len(sub)}, pos={int(y.sum())} = {y.mean():.1%}, "
      f"{len(set(days))} days)")
    w(f"- **pooled OOS AUC {pooled:.3f}**  day-block 95% CI [{lo:.3f}, {hi:.3f}]  -> **{verdict}**")
    w(f"- per-day OOS AUC spread: {np.nanmin(perfold):.3f} – {np.nanmax(perfold):.3f} "
      f"(mean {np.nanmean(perfold):.3f}, k={len(perfold)})")
    sig = "CI excludes 0.5 -> signal" if lo > 0.5 else "CI includes 0.5 -> NOT significant"
    w(f"- {sig}\n")


def main():
    df = pd.read_csv(CSV)
    df = add_mfe_target(df)
    days = sorted(df["date"].unique())
    L = []
    def w(s):
        print(s); L.append(s)
    w("# Cusp/launch detector — honest day-disjoint evaluation (scale-ready)")
    w(f"{len(df)} candidate cubic turns, {len(days)} labeled days: {days}\n")

    # leaky contrast
    sub = df.dropna(subset=FEATURES).copy()
    Xa, ya = sub[FEATURES].values, sub["target"].values.astype(int)
    rs = [roc_auc_score(yt, fit_predict(Xtr, ytr, Xt))
          for s in range(20)
          for Xtr, Xt, ytr, yt in [train_test_split(Xa, ya, test_size=.2, random_state=s, stratify=ya)]
          if len(np.unique(yt)) > 1]
    w(f"**Leaky random-split (shipped) AUC {np.mean(rs):.3f}** — same-day leakage, NOT trustworthy.\n")
    w("## Honest day-disjoint")
    evaluate(df, "target", w)     # matches a human pick
    evaluate(df, "paid", w)       # objective forward MFE
    w("## Read")
    w(f"Day-disjoint + day-block CI is the verdict; the random number is leakage. With {len(days)} days")
    w("this is underpowered (wide CI) — the harness is built so that when the labeler grows the day")
    w("count, the SAME run yields a conclusive CI. 'paid' (objective MFE) is the target that matters;")
    w("'target' (human-match) is a proxy. A tight CI above 0.5 on 'paid' across many days = real edge.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
