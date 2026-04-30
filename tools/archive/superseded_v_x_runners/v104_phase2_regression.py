"""
v104_phase2_regression.py -- Phase 2 EDA: feature-level analysis on v1.0.4
trade ledger using ATLAS_NT8 FEATURES_5s_v2 (185 features per bar).

Goal: predict trade outcome (winner/loser) from features at entry, OR
features evolving over the first N pivot-bars after entry.

Pipeline:
  1. Load NT8 trade CSV (Playback by default — 506 ground-truth trades)
  2. For each trade, find the 5s-anchor bar at-or-after entry timestamp
  3. Extract feature vector at entry (185 features) + optionally at +N min
  4. Per-feature Cohen-d ranking (winners vs losers)
  5. Lasso regression for feature selection (L1)
  6. Logistic regression on top-K features (5-fold CV)
  7. Polynomial expansion (degree 2) on top features, refit
  8. Report top predictors + AUC + held-out PnL impact

Usage:
    python tools/v104_phase2_regression.py
    python tools/v104_phase2_regression.py --trades "examples/trades v1.0.4 playback.csv"
    python tools/v104_phase2_regression.py --window-bars 5  # entry + 5 minutes
"""
from __future__ import annotations
import argparse
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import LogisticRegression, LassoCV, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def parse_money(s: str) -> float:
    s = s.replace("$", "").replace(",", "").strip()
    if not s: return 0.0
    if s.startswith("(") and s.endswith(")"): return -float(s[1:-1])
    return float(s)


def parse_dt(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%m/%d/%Y %I:%M:%S %p")


def load_trades(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                entry_dt = parse_dt(r["Entry time"])
            except (ValueError, KeyError):
                continue
            rows.append({
                "trade_id":   int(r["Trade number"]),
                "side":       r["Market pos."].strip(),
                "entry_dt":   entry_dt,
                "entry_px":   float(r["Entry price"]),
                "pnl":        parse_money(r["Profit"]),
                "mfe":        parse_money(r["MFE"]),
                "mae":        parse_money(r["MAE"]),
                "bars":       int(r["Bars"]),
            })
    df = pd.DataFrame(rows)
    df["dir"] = df["side"].map({"Long": +1, "Short": -1})
    df["is_win"] = df["pnl"] > 0
    return df


def load_features_5s(features_root: str) -> pd.DataFrame:
    """Load ATLAS_NT8 FEATURES_5s_v2 — concatenate all layer families on
    timestamp into a single wide DataFrame."""
    root = Path(features_root)
    family_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"  Loading {len(family_dirs)} layer families from {root}")
    by_family = []
    for fd in family_dirs:
        parts = []
        for f in sorted(fd.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
            if df.empty: continue
            parts.append(df)
        if not parts: continue
        merged = pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
        by_family.append(merged)

    # Outer-join all families on timestamp
    if not by_family: return pd.DataFrame()
    out = by_family[0]
    for df in by_family[1:]:
        out = out.merge(df, on="timestamp", how="outer")
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["dt_utc"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out


def feature_lookup(features: pd.DataFrame, target_dt_utc: pd.Timestamp) -> pd.Series | None:
    """Find the FIRST features row at-or-after target_dt_utc.
    Returns a Series of feature values (excluding timestamp/dt cols)."""
    mask = features["dt_utc"] >= target_dt_utc
    if not mask.any():
        return None
    idx = mask.idxmax()
    row = features.iloc[idx]
    drop_cols = ["timestamp", "dt_utc"]
    return row.drop([c for c in drop_cols if c in row.index])


def build_design_matrix(trades: pd.DataFrame, features: pd.DataFrame, user_tz_offset_h: int = 7,
                        window_bars: int = 0) -> tuple:
    """For each trade, lookup features at entry (and optionally at +N×60s).
    Returns (X DataFrame, y Series, included_indices)."""
    feature_cols = [c for c in features.columns if c not in ("timestamp", "dt_utc")]
    print(f"  Total feature columns: {len(feature_cols)}")

    rows = []
    keep_idx = []
    for i, t in trades.iterrows():
        entry_utc = pd.Timestamp(t["entry_dt"] + timedelta(hours=user_tz_offset_h), tz="UTC")
        feat_entry = feature_lookup(features, entry_utc)
        if feat_entry is None:
            continue
        row = feat_entry.to_dict()

        # Optionally append features at +N × 60s
        if window_bars > 0:
            for k in range(1, window_bars + 1):
                target = entry_utc + pd.Timedelta(seconds=60 * k)
                feat_k = feature_lookup(features, target)
                if feat_k is None:
                    feat_k = feat_entry  # fallback (rare; near data end)
                # Add as delta (change) features — helps regression
                for c in feature_cols:
                    row[f"{c}__d{k}m"] = float(feat_k.get(c, np.nan)) - float(feat_entry.get(c, np.nan))

        rows.append(row)
        keep_idx.append(i)

    X = pd.DataFrame(rows)
    y = trades.loc[keep_idx, "is_win"].astype(int).values
    pnl = trades.loc[keep_idx, "pnl"].values
    print(f"  Joined trades: {len(X)} of {len(trades)}")
    print(f"  Feature columns in X: {X.shape[1]}")
    return X, y, pnl, keep_idx


def cohen_d(x_w: np.ndarray, x_l: np.ndarray) -> float:
    n_w, n_l = len(x_w), len(x_l)
    if n_w < 2 or n_l < 2: return 0.0
    s_w, s_l = np.std(x_w, ddof=1), np.std(x_l, ddof=1)
    pooled = np.sqrt(((n_w-1)*s_w**2 + (n_l-1)*s_l**2) / (n_w + n_l - 2)) if (n_w+n_l) > 2 else 1.0
    if pooled == 0 or np.isnan(pooled): return 0.0
    return (np.mean(x_w) - np.mean(x_l)) / pooled


def cohen_d_ranking(X: pd.DataFrame, y: np.ndarray, top_n: int = 30) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        vals = X[col].dropna()
        valid_mask = X[col].notna().values
        if valid_mask.sum() < 30: continue
        x_w = X.loc[(y==1) & valid_mask, col].values
        x_l = X.loc[(y==0) & valid_mask, col].values
        if len(x_w) < 5 or len(x_l) < 5: continue
        d = cohen_d(x_w, x_l)
        rows.append({"feature": col, "cohen_d": d, "abs_d": abs(d), "n_w": len(x_w), "n_l": len(x_l),
                     "mean_w": float(np.mean(x_w)), "mean_l": float(np.mean(x_l))})
    return pd.DataFrame(rows).sort_values("abs_d", ascending=False).head(top_n).reset_index(drop=True)


def lasso_select(X: pd.DataFrame, pnl: np.ndarray) -> tuple:
    """Lasso regression on PnL with CV for alpha. Return non-zero coefficient features."""
    X_clean = X.fillna(0).values  # naive impute for lasso
    # Standardize features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_clean)
    lasso = LassoCV(cv=5, max_iter=10000, random_state=42, n_alphas=50)
    lasso.fit(Xs, pnl)
    coefs = pd.DataFrame({"feature": X.columns, "coef": lasso.coef_})
    coefs = coefs[coefs["coef"].abs() > 1e-6].sort_values("coef", key=abs, ascending=False).reset_index(drop=True)
    return coefs, lasso.alpha_, scaler


def stepwise_forward_logistic(X: pd.DataFrame, y: np.ndarray, max_features: int = 15,
                                min_improvement: float = 0.003, candidate_pool: list = None) -> dict:
    """Forward stepwise selection: at each round, add the feature that most improves
    5-fold CV AUC. Stop when improvement < min_improvement OR max_features reached.

    Returns dict with:
      - selected: list of features in order added
      - auc_history: list of CV AUCs after each step
      - delta_history: list of improvements per step
    """
    Xfilled = X.fillna(0)
    candidates = list(candidate_pool) if candidate_pool is not None else list(Xfilled.columns)
    selected = []
    auc_history = [0.5]  # baseline AUC with no features
    delta_history = []

    cv = KFold(n_splits=5, shuffle=False)

    def cv_auc(feature_set):
        if len(feature_set) == 0: return 0.5
        Xs = StandardScaler().fit_transform(Xfilled[feature_set].values)
        aucs = []
        for tr, te in cv.split(Xs):
            try:
                lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
                lr.fit(Xs[tr], y[tr])
                if len(np.unique(y[te])) < 2: continue
                prob = lr.predict_proba(Xs[te])[:, 1]
                aucs.append(roc_auc_score(y[te], prob))
            except Exception:
                continue
        return float(np.mean(aucs)) if aucs else 0.5

    for step in range(max_features):
        best_feat = None
        best_auc = auc_history[-1]
        for cand in candidates:
            if cand in selected: continue
            auc = cv_auc(selected + [cand])
            if auc > best_auc:
                best_auc = auc
                best_feat = cand
        if best_feat is None:
            break
        improvement = best_auc - auc_history[-1]
        if improvement < min_improvement:
            break
        selected.append(best_feat)
        auc_history.append(best_auc)
        delta_history.append(improvement)
        print(f"    step {step+1}: +{best_feat} -> AUC {best_auc:.3f} (delta +{improvement:.4f})")

    return {"selected": selected, "auc_history": auc_history, "delta_history": delta_history}


def logistic_top_features(X: pd.DataFrame, y: np.ndarray, top_features: list, with_poly: bool = False) -> dict:
    """Fit logistic regression on top_features. Optionally with polynomial expansion."""
    sub = X[top_features].fillna(0).values
    feature_names = list(top_features)
    if with_poly:
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        sub = poly.fit_transform(sub)
        feature_names = poly.get_feature_names_out(top_features).tolist()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(sub)
    # Cross-validated AUC
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    cv = KFold(n_splits=5, shuffle=False)  # chronological folds via no shuffle
    aucs = []
    for tr, te in cv.split(Xs):
        lr.fit(Xs[tr], y[tr])
        prob = lr.predict_proba(Xs[te])[:, 1]
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], prob))
    auc_mean = np.mean(aucs) if aucs else np.nan
    auc_std  = np.std(aucs)  if aucs else np.nan

    # Refit on full data for coefficient inspection
    lr.fit(Xs, y)
    coefs = pd.DataFrame({"feature": feature_names, "coef": lr.coef_[0]}).sort_values("coef", key=abs, ascending=False)
    return {"auc_mean": auc_mean, "auc_std": auc_std, "coefs": coefs, "n_features": len(feature_names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="examples/trades v1.0.4 playback.csv")
    ap.add_argument("--features", default="DATA/ATLAS_NT8/FEATURES_5s_v2")
    ap.add_argument("--out-md", default="reports/findings/2026-04-27_v104_phase2_regression.md")
    ap.add_argument("--user-tz-offset", type=int, default=7,
                    help="NT8 CSV uses local time. Offset to UTC in hours (PDT=7).")
    ap.add_argument("--window-bars", type=int, default=5,
                    help="0 = entry features only; N>0 = also include feature deltas at +1m..+Nm.")
    ap.add_argument("--top-n", type=int, default=15,
                    help="Number of top features to keep for logistic + polynomial fit.")
    args = ap.parse_args()

    if not HAS_SKLEARN:
        print("FATAL: scikit-learn required (pip install scikit-learn)")
        return

    print(f"Loading trades: {args.trades}")
    trades = load_trades(args.trades)
    print(f"  {len(trades)} trades  (winners: {trades['is_win'].sum()}, losers: {(~trades['is_win']).sum()})")

    print(f"Loading features: {args.features}")
    features = load_features_5s(args.features)
    print(f"  Features rows: {len(features)}, columns: {features.shape[1]}")
    print(f"  Time range: {features['dt_utc'].min()} -- {features['dt_utc'].max()}")

    print(f"\nBuilding design matrix (window_bars={args.window_bars})...")
    X, y, pnl, keep_idx = build_design_matrix(trades, features, args.user_tz_offset, args.window_bars)
    if len(X) < 50:
        print(f"FATAL: too few joined trades ({len(X)}). Check timezone offset and feature coverage.")
        return

    # Drop columns that are all-NaN or have <90% coverage
    coverage = X.notna().mean()
    keep_cols = coverage[coverage >= 0.9].index.tolist()
    X = X[keep_cols]
    print(f"  Features with >=90% coverage: {len(keep_cols)}")

    sections = []
    sections.append(f"# v1.0.4 Phase 2 EDA — feature-level regression\n")
    sections.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n")
    sections.append(f"Trades: `{args.trades}`\n")
    sections.append(f"Features: `{args.features}`\n")
    sections.append(f"window_bars: {args.window_bars}  (0 = entry only; >0 includes feature deltas at +1m..+Nm after entry)\n")
    sections.append(f"Joined trades: {len(X)} (winners: {y.sum()}, losers: {len(y)-y.sum()})\n")
    sections.append(f"Features in design matrix: {X.shape[1]}\n")
    sections.append(f"Base WR: {y.mean()*100:.1f}%\n\n")

    # 1. Cohen-d ranking
    print("\n[1/4] Cohen-d ranking (winners vs losers)...")
    cohen_df = cohen_d_ranking(X, y, top_n=30)
    sections.append("## Top 30 features by |Cohen-d| (winners vs losers at entry)\n")
    sections.append("| Rank | Feature | Cohen-d | mean(W) | mean(L) | n(W) | n(L) |")
    sections.append("|---:|---|---:|---:|---:|---:|---:|")
    for i, r in cohen_df.iterrows():
        sections.append(f"| {i+1} | `{r['feature']}` | {r['cohen_d']:+.3f} | {r['mean_w']:+.4f} | {r['mean_l']:+.4f} | {r['n_w']} | {r['n_l']} |")
    sections.append("")
    print(f"  Top |d|: {cohen_df.iloc[0]['cohen_d']:+.3f} on `{cohen_df.iloc[0]['feature']}`")

    # 2. Lasso regression on PnL (feature selection)
    print("\n[2/4] Lasso regression on PnL (feature selection)...")
    lasso_coefs, alpha, _ = lasso_select(X, pnl)
    sections.append(f"## Lasso regression on PnL — feature selection\n")
    sections.append(f"Best alpha (5-fold CV): {alpha:.4f}\n")
    sections.append(f"Non-zero coefficients: {len(lasso_coefs)}\n")
    if len(lasso_coefs) == 0:
        sections.append("**No features selected.** Lasso shrunk all to zero — no linear PnL predictor found.\n")
    else:
        sections.append("| Rank | Feature | Coefficient |")
        sections.append("|---:|---|---:|")
        for i, r in lasso_coefs.head(20).iterrows():
            sections.append(f"| {i+1} | `{r['feature']}` | {r['coef']:+.4f} |")
    sections.append("")
    print(f"  Lasso selected {len(lasso_coefs)} non-zero coefficients")

    # 3. Logistic regression — top-N features (linear)
    print(f"\n[3/4] Logistic regression — top {args.top_n} features (linear)...")
    top_features = cohen_df["feature"].head(args.top_n).tolist()
    log_lin = logistic_top_features(X, y, top_features, with_poly=False)
    sections.append(f"## Logistic regression — top {args.top_n} features (linear)\n")
    sections.append(f"5-fold CV AUC: **{log_lin['auc_mean']:.3f} ± {log_lin['auc_std']:.3f}**")
    sections.append(f"(0.5 = no signal; >0.6 = weak; >0.7 = decent; >0.8 = strong)\n")
    sections.append("Coefficients (refit on full data, standardized):\n")
    sections.append("| Rank | Feature | Coefficient |")
    sections.append("|---:|---|---:|")
    for i, r in log_lin["coefs"].head(20).iterrows():
        sections.append(f"| {i+1} | `{r['feature']}` | {r['coef']:+.4f} |")
    sections.append("")

    # 4a. Stepwise forward selection (logistic on win/loss)
    print(f"\n[4a] Forward stepwise selection on win/loss (candidate pool = top 50 by Cohen-d)...")
    candidate_pool = cohen_df["feature"].head(50).tolist()
    stepwise_result = stepwise_forward_logistic(X, y, max_features=15, min_improvement=0.003,
                                                  candidate_pool=candidate_pool)
    sections.append(f"## Stepwise forward selection (win/loss prediction)\n")
    sections.append(f"Candidate pool: top 50 features by |Cohen-d|.  Stop criterion: AUC improvement < 0.003.\n")
    if not stepwise_result["selected"]:
        sections.append("**No features selected** — none improved AUC by min threshold.\n")
    else:
        sections.append("| Step | Feature added | CV AUC | Δ AUC |")
        sections.append("|---:|---|---:|---:|")
        for i, (feat, auc, delta) in enumerate(zip(
            stepwise_result["selected"],
            stepwise_result["auc_history"][1:],
            stepwise_result["delta_history"]
        )):
            sections.append(f"| {i+1} | `{feat}` | {auc:.3f} | +{delta:.4f} |")
        sections.append("")
        sections.append(f"**Final selected: {len(stepwise_result['selected'])} features.** "
                        f"Final CV AUC: **{stepwise_result['auc_history'][-1]:.3f}**\n")

    # 4b. Polynomial expansion on top features
    print(f"\n[4b] Logistic regression -- top {args.top_n} features (polynomial degree 2)...")
    log_poly = logistic_top_features(X, y, top_features, with_poly=True)
    sections.append(f"## Logistic regression — top {args.top_n} features × degree-2 polynomial\n")
    sections.append(f"Total polynomial terms: {log_poly['n_features']}")
    sections.append(f"5-fold CV AUC: **{log_poly['auc_mean']:.3f} ± {log_poly['auc_std']:.3f}**\n")
    delta = log_poly["auc_mean"] - log_lin["auc_mean"]
    if abs(delta) < 0.02:
        sections.append(f"Δ vs linear: **{delta:+.3f}** — polynomial does not meaningfully improve.\n")
    elif delta > 0:
        sections.append(f"Δ vs linear: **{delta:+.3f}** — polynomial captures real interactions.\n")
    else:
        sections.append(f"Δ vs linear: **{delta:+.3f}** — polynomial overfits.\n")
    sections.append("Top 20 polynomial terms by |coefficient|:\n")
    sections.append("| Rank | Term | Coefficient |")
    sections.append("|---:|---|---:|")
    for i, r in log_poly["coefs"].head(20).iterrows():
        sections.append(f"| {i+1} | `{r['feature']}` | {r['coef']:+.4f} |")
    sections.append("")

    # Honest summary
    sections.append("## Honest summary\n")
    auc = log_lin["auc_mean"]
    if auc < 0.55:
        verdict = "**NO SIGNIFICANT SIGNAL.** AUC near 0.5 = no better than coin flip. Entry features do not predict outcome at this sample size with these features."
    elif auc < 0.60:
        verdict = "**WEAK SIGNAL.** AUC 0.55-0.60 = marginal. Could be noise or true edge — needs OOS validation."
    elif auc < 0.70:
        verdict = "**MODEST SIGNAL.** AUC 0.60-0.70 = real but weak. Could support a weak filter."
    else:
        verdict = "**STRONG SIGNAL.** AUC > 0.70 = meaningful predictive power. Worth implementing as a strategy filter — BUT validate OOS first to confirm."
    sections.append(verdict)
    sections.append("")
    sections.append(f"Caveats:")
    sections.append(f"- Only 506 trades in this Playback ledger. Underdetermined for {X.shape[1]}-feature regression without regularization.")
    sections.append(f"- Per playbook §9c: 'direction at entry is RANDOM on 91D'. We're predicting OUTCOME (win/loss) not direction — different question — but the small-N risk applies.")
    sections.append(f"- Single 32-day window. Need OOS validation on a different window before trusting any selected feature.")
    sections.append(f"- Polynomial expansion can overfit — compare CV AUC linear vs poly to detect.")

    # Write out
    out = "\n".join(sections)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"\nReport: {args.out_md}")
    print(f"\nKey numbers:")
    print(f"  Top |Cohen-d|:   {cohen_df.iloc[0]['cohen_d']:+.3f} on `{cohen_df.iloc[0]['feature']}`")
    print(f"  Lasso non-zero:  {len(lasso_coefs)}")
    print(f"  AUC (linear):    {log_lin['auc_mean']:.3f}")
    print(f"  AUC (polynomial): {log_poly['auc_mean']:.3f}")


if __name__ == "__main__":
    main()
