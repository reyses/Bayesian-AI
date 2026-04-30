"""
v15_alt_classifiers.py — Alternative classifiers for BLEED vs HARVEST.

Compares:
  1. Linear z-score (= MVP rule)
  2. Decision tree (depth 2, depth 3)
  3. Logistic regression
  4. Simple AND-gate rules (e.g., prior_range > X1 AND range_compression > X2)

All trained on IS, evaluated on OOS using identical IS-trained boundaries.

Usage:
    python tools/v15_alt_classifiers.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = "reports/findings/2026-04-27_bleed_harvest_forward/day_labels_with_score.csv"

FEATURES = [
    "prior_range", "prior_drift", "prior_efficiency",
    "mean_range_5d", "mean_range_20d", "mean_efficiency_5d",
    "range_compression", "variance_ratio_5_20",
    "cum_drift_5d", "dow",
]


def lift(test_df, mask_skip):
    pnl_unfilt = test_df["pnl"].sum()
    pnl_kept   = test_df.loc[~mask_skip, "pnl"].sum()
    return pnl_kept - pnl_unfilt, pnl_kept, mask_skip.sum(), (~mask_skip).sum()


def main():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=FEATURES + ["pnl"]).reset_index(drop=True)
    is_df  = df[df["set"] == "IS"].reset_index(drop=True)
    oos_df = df[df["set"] == "OOS"].reset_index(drop=True)

    # Binary label: BLEED = 1, HARVEST = 0; NEUTRAL excluded for training
    is_train  = is_df[is_df["label"] != "NEUTRAL"].copy()
    is_train["y"] = (is_train["label"] == "BLEED").astype(int)
    oos_eval  = oos_df.copy()  # apply rule to ALL OOS days (not just labeled)

    print(f"IS training: {len(is_train)} labeled days "
          f"(BLEED={is_train['y'].sum()}, HARVEST={(is_train['y']==0).sum()})")
    print(f"OOS test (all):  {len(oos_eval)} days, total PnL = ${oos_eval['pnl'].sum():+.2f}")
    print()

    X_tr = is_train[FEATURES].values
    y_tr = is_train["y"].values
    X_te = oos_eval[FEATURES].values

    # ── 1. Linear z-score rule (the MVP) ────────────────────────────────────
    print("=" * 80)
    print("MVP linear z-score (prior_range + range_compression) at threshold z=-0.34")
    print("=" * 80)
    skip = oos_eval["bleed_score"] > -0.34
    lift_v, kept_pnl, n_skip, n_keep = lift(oos_eval, skip)
    print(f"  N test: {len(oos_eval)}, skipped: {n_skip}, kept: {n_keep}")
    print(f"  OOS lift: ${lift_v:+,.0f}, kept PnL: ${kept_pnl:+,.0f}")
    print()

    # ── 2. Decision Tree (depth 2 / 3) ──────────────────────────────────────
    for depth in [2, 3, 4]:
        print("=" * 80)
        print(f"Decision Tree (depth={depth})")
        print("=" * 80)
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42, min_samples_leaf=4)
        clf.fit(X_tr, y_tr)
        # Predict probability of bleed
        p_bleed = clf.predict_proba(X_te)[:, 1] if clf.n_classes_ > 1 else np.zeros(len(X_te))
        # Try various skip thresholds
        best_lift = -np.inf
        best_thr  = 0.5
        best_meta = None
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            skip = p_bleed >= thr
            lift_v, kept_pnl, n_skip, n_keep = lift(oos_eval, pd.Series(skip, index=oos_eval.index))
            if lift_v > best_lift:
                best_lift = lift_v
                best_thr  = thr
                best_meta = (n_skip, n_keep, kept_pnl)
        n_skip, n_keep, kept_pnl = best_meta
        # IS AUC
        is_auc = roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1]) if len(set(y_tr)) > 1 else 0.5
        print(f"  IS AUC: {is_auc:.3f}  best OOS skip threshold: p>={best_thr}")
        print(f"  N test: {len(oos_eval)}, skipped: {n_skip}, kept: {n_keep}")
        print(f"  OOS lift: ${best_lift:+,.0f}, kept PnL: ${kept_pnl:+,.0f}")
        # Print first 3-4 split rules
        tree_text = export_text(clf, feature_names=FEATURES, max_depth=depth)
        for line in tree_text.split("\n")[:18]:
            print(f"  {line}")
        print()

    # ── 3. Logistic Regression ──────────────────────────────────────────────
    print("=" * 80)
    print("Logistic Regression (all features)")
    print("=" * 80)
    if len(set(y_tr)) > 1:
        # Standardize using IS only
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0) + 1e-9
        X_tr_n = (X_tr - mu) / sd
        X_te_n = (X_te - mu) / sd
        lr = LogisticRegression(max_iter=200, C=1.0)
        lr.fit(X_tr_n, y_tr)
        p_bleed = lr.predict_proba(X_te_n)[:, 1]
        best_lift = -np.inf
        best_thr = 0.5
        best_meta = None
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            skip = p_bleed >= thr
            lift_v, kept_pnl, n_skip, n_keep = lift(oos_eval, pd.Series(skip, index=oos_eval.index))
            if lift_v > best_lift:
                best_lift = lift_v
                best_thr  = thr
                best_meta = (n_skip, n_keep, kept_pnl)
        n_skip, n_keep, kept_pnl = best_meta
        is_auc = roc_auc_score(y_tr, lr.predict_proba(X_tr_n)[:, 1])
        print(f"  IS AUC: {is_auc:.3f}  best OOS skip p>={best_thr}")
        print(f"  N test: {len(oos_eval)}, skipped: {n_skip}, kept: {n_keep}")
        print(f"  OOS lift: ${best_lift:+,.0f}, kept PnL: ${kept_pnl:+,.0f}")
        # Show top features by |coef|
        coefs = sorted(zip(FEATURES, lr.coef_[0]), key=lambda x: -abs(x[1]))
        print(f"  Top coefs (abs): {coefs[:5]}")
    print()

    # ── 4. Simple AND-gate rules ────────────────────────────────────────────
    print("=" * 80)
    print("AND-gate rules (interpretable)")
    print("=" * 80)
    # Try a few hand-chosen thresholds based on Phase 1 finding
    rules = [
        ("prior_range > 500", lambda r: r["prior_range"] > 500),
        ("prior_range > 600", lambda r: r["prior_range"] > 600),
        ("range_compression > 1.2", lambda r: r["range_compression"] > 1.2),
        ("prior_range > 500 AND range_compression > 1.0",
         lambda r: (r["prior_range"] > 500) & (r["range_compression"] > 1.0)),
        ("prior_range > 500 OR range_compression > 1.4",
         lambda r: (r["prior_range"] > 500) | (r["range_compression"] > 1.4)),
        ("prior_range > 450 AND mean_range_5d > 350",
         lambda r: (r["prior_range"] > 450) & (r["mean_range_5d"] > 350)),
    ]
    for name, fn in rules:
        skip = fn(oos_eval).fillna(False)
        lift_v, kept_pnl, n_skip, n_keep = lift(oos_eval, skip)
        print(f"  {name:<55}  skip={n_skip:>2}  lift=${lift_v:+>7.0f}  kept=${kept_pnl:+>7.0f}")


if __name__ == "__main__":
    main()
