"""Direction 1: HONESTLY evaluate the cubic+classifier cusp/launch detector.

The shipped train_picks_classifier.py uses a RANDOM 80/20 split. Picks from the same day share a
regime, so random split leaks regime info train->test and inflates AUC. Correct unit of independence
= the DAY. We re-evaluate day-disjoint (leave-one-day-out), report per-day OOS AUC + pooled, vs the
random-split number, held to the signal bar (gap>=0.10 real, 0.05-0.10 conditional, <0.05 noise).

Brutal caveat baked in: candidate_primitives.csv has only 4 labeled days -> day-disjoint power is ~nil.
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
REPORT = os.path.join(ROOT, "research", "cusp_launch_detector", "reports", "eval_oos.md")

FEATURES = ["z_15s", "z_1m", "z_15m", "slope_15s_3m", "slope_15s_10m", "slope_1m_10m",
            "slope_15m_5m", "slope_15m_15m", "slope_15m_decel", "curv_15m",
            "band_width", "band_rank_60", "sigma_15m_rank_60", "fan_width",
            "align_up_count", "align_down_count"]


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def main():
    df = pd.read_csv(CSV).dropna(subset=FEATURES).copy()
    days = sorted(df["date"].unique())
    X = df[FEATURES].values
    y = df["target"].values.astype(int)

    L = []
    def w(s):
        print(s); L.append(s)
    w("# Cusp/launch detector — honest day-disjoint evaluation")
    w(f"{len(df)} candidate cubic turns, {y.sum()} human-accepted ({y.mean():.1%}), "
      f"**{len(days)} labeled days**: {days}\n")

    # 1) the shipped optimistic number: random 80/20 (leaks same-day regime)
    rng_aucs = []
    for s in range(20):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
        if len(np.unique(yte)) > 1:
            rng_aucs.append(roc_auc_score(yte, fit_predict(Xtr, ytr, Xte)))
    w(f"## Random 80/20 split (the shipped method — LEAKY): AUC {np.mean(rng_aucs):.3f} "
      f"(±{np.std(rng_aucs):.3f})  <- optimistic, same-day leakage\n")

    # 2) honest: leave-one-DAY-out
    w("## Leave-one-DAY-out (honest, day-disjoint)")
    w(f"{'held-out day':>14} | n / pos | OOS AUC")
    w("-" * 44)
    oos_prob = np.full(len(df), np.nan)
    perday = []
    for d in days:
        te = df["date"].values == d
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        p = fit_predict(X[tr], y[tr], X[te])
        oos_prob[te] = p
        auc = roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan
        perday.append(auc)
        w(f"{d:>14} | {te.sum():>4}/{int(y[te].sum()):<3} | {auc:.3f}")
    pooled = roc_auc_score(y[~np.isnan(oos_prob)], oos_prob[~np.isnan(oos_prob)])
    w("-" * 44)
    w(f"{'POOLED OOS':>14} |         | **{pooled:.3f}**")
    w(f"{'per-day mean':>14} |         | {np.nanmean(perday):.3f}  (spread {np.nanmin(perday):.3f}-{np.nanmax(perday):.3f})\n")

    gap = pooled - 0.5
    verdict = "REAL" if gap >= 0.10 else "CONDITIONAL" if gap >= 0.05 else "NOISE"
    w(f"## Verdict: pooled OOS gap +{gap:.3f} -> nominal **{verdict}** by the signal bar")
    w("\n## BUT — the honest caveat that dominates everything")
    w(f"- Only **{len(days)} labeled days**. The unit of independence is the day (one regime/day), so")
    w(f"  day-disjoint validation has ~{len(days)} effective data points. The per-day AUC spread")
    w(f"  ({np.nanmin(perday):.2f}-{np.nanmax(perday):.2f}) is the real uncertainty — a pooled point estimate hides it.")
    w("- The random-split number is NOT trustworthy (same-day leakage); the day-disjoint number is")
    w("  honest but UNDERPOWERED. We cannot conclude the detector has durable edge from 4 days.")
    w("- Also: target='matches a human pick', NOT 'the launch paid' (MFE). A profit-targeted eval needs")
    w("  the MFE/MAE labels. ACTION: label more days (>=15-20 disjoint) before trusting any AUC here.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
