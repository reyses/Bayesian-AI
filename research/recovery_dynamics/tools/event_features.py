"""WHAT describes each oscillation event? Causal feature attribution for the DEATH event.

For every wrong-trade event we compute interpretable CAUSAL features at entry (visible state)
and ask: which features / feature GROUPS best separate a quick bounce from a trade that NEVER
comes back (death)? Death is the high-value target — it's the regime where holding ruins you.

Honest validation: train 2024, test 2025 (date-disjoint OOS). Report test AUC + permutation
importance + GROUP drop-importance. Held to the signal bar: AUC-0.5 gap >=0.10 real,
0.05-0.10 conditional, <0.05 noise. (Direction was unpredictable; death has regime structure.)

Features are TRADE-DIRECTION-RELATIVE where it matters: a market trending AGAINST the trade is
what kills it, so we feed 'adverse drift/momentum' (= -dir * move), not raw move.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import opportunity_cost as oc  # noqa: E402

ROOT, ONE_M = oc.ROOT, oc.ONE_M
REPORT = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "event_features.md")

WFEAT = 30               # trailing bars for causal features
H_DEATH = 60             # death = not back to breakeven within this many bars of going underwater
EPS = 1e-9               # (fixed horizon -> every event judged on equal runway, no EOD censoring)

FEATS = ["realized_vol", "efficiency_ratio", "vol_accel", "range_pts",
         "adverse_drift", "adverse_mom", "time_of_day"]
GROUPS = {"VOL": ["realized_vol", "vol_accel", "range_pts"],
          "TREND": ["efficiency_ratio"],
          "ADVERSE": ["adverse_drift", "adverse_mom"],   # market moving against THIS trade
          "TIME": ["time_of_day"]}


def features(close, e, d, n):
    seg = close[e - WFEAT:e + 1]
    dif = np.diff(seg)
    rv = float(np.std(dif))
    er = float(abs(seg[-1] - seg[0]) / (np.abs(dif).sum() + EPS))      # Kaufman efficiency ratio
    half = len(dif) // 2
    vacc = float((np.std(dif[half:]) + EPS) / (np.std(dif[:half]) + EPS))
    rng = float(seg.max() - seg.min())
    drift = (seg[-1] - seg[0]) / WFEAT
    mom = close[e] - close[e - 5]
    return np.array([rv, er, vacc, rng, -d * drift, -d * mom, e / n], dtype=np.float64)


def capture_day(close, rng):
    n = len(close)
    hi = int(n * oc.ENTRY_FRAC)
    rows = []
    if hi <= WFEAT + 5:
        return rows
    for e in rng.integers(WFEAT, hi, size=oc.N_PER_DAY):
        e = int(e)
        d = 1 if rng.random() < 0.5 else -1
        pnl = d * (close[e + 1:] - close[e])
        if pnl.size < 2:
            continue
        adv = np.where(pnl <= -oc.MIN_ADVERSE_PTS)[0]
        pro = np.where(pnl >= oc.PROFIT_FIRST_PTS)[0]
        if len(adv) == 0 or (len(pro) and pro[0] < adv[0]):
            continue                                          # not a wrong trade
        a = adv[0]
        if a + H_DEATH >= pnl.size:
            continue                                          # insufficient runway -> don't judge (no censoring)
        death = 0 if (pnl[a:a + H_DEATH] >= 0).any() else 1   # back to breakeven within H bars?
        rows.append((features(close, e, d, n), death))
    return rows


def gather(year, rng):
    X, y = [], []
    for f in tqdm(sorted(glob.glob(os.path.join(ONE_M, f"{year}_*.parquet"))), desc=year, unit="day"):
        try:
            close = pd.read_parquet(f)["close"].to_numpy(np.float64)
        except Exception:
            continue
        for fv, d in capture_day(close, rng):
            X.append(fv); y.append(d)
    return np.array(X), np.array(y)


def main():
    rng = np.random.default_rng(oc.SEED)
    Xtr, ytr = gather("2024", rng)
    Xte, yte = gather("2025", rng)

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    gap = auc - 0.5
    verdict = ("REAL" if gap >= 0.10 else "CONDITIONAL" if gap >= 0.05 else "NOISE")

    perm = permutation_importance(clf, Xte, yte, scoring="roc_auc", n_repeats=8, random_state=0)
    order = np.argsort(perm.importances_mean)[::-1]

    # group drop-importance: shuffle a whole group, measure AUC drop
    rng2 = np.random.default_rng(1)
    grp = {}
    for g, cols in GROUPS.items():
        Xp = Xte.copy()
        for c in cols:
            j = FEATS.index(c)
            Xp[:, j] = rng2.permutation(Xp[:, j])
        grp[g] = auc - roc_auc_score(yte, clf.predict_proba(Xp)[:, 1])

    # direction of effect: corr(feature, death) on test
    corr = {FEATS[j]: float(np.corrcoef(Xte[:, j], yte)[0, 1]) for j in range(len(FEATS))}

    L = []
    def w(s):
        print(s.encode("ascii", "replace").decode()); L.append(s)
    w("# What features describe a DEATH event (wrong trade that never comes back)")
    w(f"train 2024 (n={len(ytr)}, death {ytr.mean():.0%}) -> test 2025 (n={len(yte)}, death {yte.mean():.0%})\n")
    w(f"## OOS death-prediction AUC = **{auc:.3f}**  (gap +{gap:.3f} -> **{verdict}** by the signal bar)\n")
    w("## Feature importance (permutation, OOS AUC drop) + direction")
    w("```")
    w(f"{'feature':>17} | {'importance':>10} | corr w/ death")
    for j in order:
        f = FEATS[j]
        w(f"{f:>17} | {perm.importances_mean[j]:>10.4f} | {corr[f]:+.3f} "
          f"({'more death' if corr[f]>0 else 'less death'})")
    w("```")
    w("## Feature-GROUP importance (shuffle whole group, OOS AUC drop)")
    w("```")
    for g, v in sorted(grp.items(), key=lambda x: -x[1]):
        w(f"{g:>9} ({'+'.join(GROUPS[g])[:34]:<34}) | AUC drop {v:.4f}")
    w("```")
    w("\n## Read")
    w(f"Death prediction is {verdict}. The top group/feature is what to READ live to decide hold-vs-cut.")
    w("If ADVERSE/TREND dominate -> a trade dies when the market trends against it (the regime read);")
    w("if VOL dominates -> death is about volatility magnitude. Either way this is the causal gauge the")
    w("fixed clock lacked. Next: turn the top features into the live two-gauge meter.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
