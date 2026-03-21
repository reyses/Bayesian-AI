"""
Analysis FF -- Conviction Calibration Audit
Source: RESEARCH_SPEC_V_TO_FF.md

Reads existing oracle trade logs (IS + OOS). No ATLAS data needed.
Outputs: reports/research/ff_conviction_audit/results.txt
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]

IS_LOG  = ROOT / "checkpoints" / "oracle_trade_log_old.csv"
OOS_LOG = ROOT / "checkpoints" / "oos_trade_log.csv"
OUT_DIR = ROOT / "reports" / "research" / "ff_conviction_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Features available in trade log for correlation + recalibration
FEATURES = ["dmi_diff", "entry_depth", "hurst", "velocity", "sigma",
            "band_speed", "F_momentum", "F_reversion", "wave_maturity"]


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["won"] = (df["result"] == "WIN").astype(int)
    # Coerce numeric features
    for feat in FEATURES + ["belief_conviction"]:
        df[feat] = pd.to_numeric(df[feat], errors="coerce")
    return df.dropna(subset=["belief_conviction"])


def analyze(df: pd.DataFrame, label: str, f):
    N = len(df)
    conv = df["belief_conviction"].values
    won = df["won"].values

    f.write(f"\n{'='*70}\n")
    f.write(f"  {label}  --  {N:,} trades, WR={won.mean():.1%}\n")
    f.write(f"{'='*70}\n")

    # -- 1. Conviction distribution -----------------------------------
    f.write(f"\n[1] Conviction Distribution\n")
    f.write(f"    Range:  [{conv.min():.4f}, {conv.max():.4f}]\n")
    f.write(f"    Mean:   {conv.mean():.4f}\n")
    f.write(f"    Std:    {conv.std():.4f}\n")
    for pct in [10, 25, 50, 75, 90]:
        f.write(f"    P{pct:02d}:    {np.percentile(conv, pct):.4f}\n")
    spread = conv.max() - conv.min()
    f.write(f"    Spread: {spread:.4f} >> {'NARROW (< 0.05)' if spread < 0.05 else 'Adequate'}\n")

    # -- 2. Winner vs loser conviction --------------------------------
    f.write(f"\n[2] Winner vs Loser Conviction\n")
    w_conv = conv[won == 1]
    l_conv = conv[won == 0]
    f.write(f"    Winners:  mean={w_conv.mean():.4f}  std={w_conv.std():.4f}  N={len(w_conv):,}\n")
    f.write(f"    Losers:   mean={l_conv.mean():.4f}  std={l_conv.std():.4f}  N={len(l_conv):,}\n")
    if len(w_conv) > 5 and len(l_conv) > 5:
        stat, p = mannwhitneyu(w_conv, l_conv, alternative="two-sided")
        f.write(f"    Mann-Whitney U: stat={stat:.0f}, p={p:.4f}\n")
        f.write(f"    >> {'SIGNIFICANT' if p < 0.05 else 'NOT significant'} difference\n")

    # -- 3. Calibration curve -----------------------------------------
    f.write(f"\n[3] Calibration Curve (conviction bins vs actual WR)\n")
    f.write(f"    {'Bin':>16s}  {'N':>6s}  {'WR':>6s}  {'Expected':>8s}  {'Gap':>6s}\n")
    f.write(f"    {'-'*16}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}\n")

    # Dynamic bins based on actual distribution
    edges = np.percentile(conv, np.arange(0, 101, 10))
    edges = np.unique(np.round(edges, 3))
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conv >= lo) & (conv < hi)
        if mask.sum() < 5:
            continue
        wr = won[mask].mean()
        expected = (lo + hi) / 2
        gap = wr - expected
        f.write(f"    [{lo:.3f}, {hi:.3f})  {mask.sum():6,}  {wr:5.1%}  {expected:7.1%}  {gap:+5.1%}\n")

    # -- 4. Monotonicity check ----------------------------------------
    f.write(f"\n[4] Monotonicity: Does higher conviction >> higher WR?\n")
    quartiles = np.percentile(conv, [25, 50, 75])
    bins = [conv.min() - 0.001] + list(quartiles) + [conv.max() + 0.001]
    prev_wr = 0
    monotonic = True
    for i in range(len(bins)-1):
        mask = (conv >= bins[i]) & (conv < bins[i+1])
        if mask.sum() < 5:
            continue
        wr = won[mask].mean()
        label_q = f"Q{i+1}"
        arrow = "UP" if wr >= prev_wr else "DN"
        f.write(f"    {label_q} [{bins[i]:.3f}, {bins[i+1]:.3f}): WR={wr:.1%}  N={mask.sum():,}  {arrow}\n")
        if wr < prev_wr - 0.01 and i > 0:
            monotonic = False
        prev_wr = wr
    f.write(f"    >> {'MONOTONIC' if monotonic else 'NOT monotonic'}\n")

    # -- 5. Feature correlations with conviction ----------------------
    f.write(f"\n[5] What Does Conviction Correlate With?\n")
    f.write(f"    {'Feature':>15s}  {'Pearson r':>10s}  {'p-value':>8s}  {'Spearman rho':>11s}  {'p-value':>8s}\n")
    f.write(f"    {'-'*15}  {'-'*10}  {'-'*8}  {'-'*11}  {'-'*8}\n")
    for feat in FEATURES:
        vals = df[feat].values
        valid = ~np.isnan(vals) & ~np.isnan(conv)
        if valid.sum() < 10:
            continue
        pr, pp = pearsonr(conv[valid], vals[valid])
        sr, sp = spearmanr(conv[valid], vals[valid])
        sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
        f.write(f"    {feat:>15s}  {pr:+10.3f}  {pp:8.4f}{sig:3s}  {sr:+11.3f}  {sp:8.4f}\n")

    # -- 6. Feature correlations with WINNING -------------------------
    f.write(f"\n[6] What Actually Predicts Winning?\n")
    f.write(f"    {'Feature':>15s}  {'Pearson r':>10s}  {'p-value':>8s}\n")
    f.write(f"    {'-'*15}  {'-'*10}  {'-'*8}\n")
    for feat in FEATURES + ["belief_conviction"]:
        vals = df[feat].values
        valid = ~np.isnan(vals) & ~np.isnan(won.astype(float))
        if valid.sum() < 10:
            continue
        pr, pp = pearsonr(won[valid].astype(float), vals[valid])
        sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
        f.write(f"    {feat:>15s}  {pr:+10.3f}  {pp:8.4f}{sig:3s}\n")

    # -- 7. Recalibration attempt -------------------------------------
    f.write(f"\n[7] Recalibrated Conviction (Logistic Regression)\n")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        feat_cols = [c for c in FEATURES if df[c].notna().sum() > N * 0.8]
        X = df[feat_cols].fillna(0).values
        y = won

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr = LogisticRegression(max_iter=1000, C=1.0)

        # 5-fold CV
        auc_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc")
        acc_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring="accuracy")
        f.write(f"    Features used: {feat_cols}\n")
        f.write(f"    5-fold AUC:    {auc_scores.mean():.3f} +/- {auc_scores.std():.3f}\n")
        f.write(f"    5-fold Acc:    {acc_scores.mean():.3f} +/- {acc_scores.std():.3f}\n")

        # Feature importances
        lr.fit(X_scaled, y)
        f.write(f"\n    Feature Importances (logistic coefs):\n")
        coefs = sorted(zip(feat_cols, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
        for feat, coef in coefs:
            f.write(f"      {feat:>15s}: {coef:+.4f}\n")

        f.write(f"\n    >> Recalibrated AUC {'> 0.62 >> PROMOTE candidate' if auc_scores.mean() >= 0.62 else '< 0.62 >> KILL candidate'}\n")

    except ImportError:
        f.write(f"    sklearn not available -- skipping recalibration\n")

    # -- 8. Bypass rate -----------------------------------------------
    f.write(f"\n[8] Template vs Bypass Breakdown\n")
    bypass = df[df["template_id"] == -1] if -1 in df["template_id"].values else df[df["playbook"].str.contains("BYPASS", case=False, na=False)]
    template = df[~df.index.isin(bypass.index)]
    f.write(f"    Template-matched: {len(template):,} ({len(template)/N:.0%})\n")
    f.write(f"    Worker bypass:    {len(bypass):,} ({len(bypass)/N:.0%})\n")
    if len(bypass) > 5:
        f.write(f"    Bypass WR:    {bypass['won'].mean():.1%}\n")
        f.write(f"    Template WR:  {template['won'].mean():.1%}\n")


def main():
    results_path = OUT_DIR / "results.txt"
    with open(results_path, "w") as f:
        f.write("Analysis FF -- Conviction Calibration Audit\n")
        f.write(f"{'='*70}\n")

        for label_name, path in [("IN-SAMPLE", IS_LOG), ("OUT-OF-SAMPLE", OOS_LOG)]:
            if not path.exists():
                f.write(f"\nWARNING: {label_name} log not found: {path}\n")
                continue
            df = load_trades(path)
            f.write(f"\nLoaded {label_name}: {len(df):,} trades from {path.name}\n")
            analyze(df, label_name, f)

        # -- GATE decision --------------------------------------------
        f.write(f"\n\n{'='*70}\n")
        f.write(f"  GATE DECISION\n")
        f.write(f"{'='*70}\n")
        f.write(f"  PROMOTE if: Recalibrated AUC >= 0.62 AND calibration slope 0.6-1.4\n")
        f.write(f"  KILL if: AUC < 0.55 AND conviction range < 0.05\n")

    print(f"Results written to {results_path}")
    print(f"\nQuick preview:")
    with open(results_path) as f:
        for line in f:
            print(line, end="")


if __name__ == "__main__":
    main()
