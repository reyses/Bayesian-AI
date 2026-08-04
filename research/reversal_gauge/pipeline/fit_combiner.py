"""Fit the reversal-gauge combiner: logistic regression over freeze-event features.

Reads research/reversal_gauge/events.parquet (built by
builders/extract_freeze_events.py), fits a day-grouped cross-validated
logistic model for label_resume on resolved events only, and writes
research/reversal_gauge/coeffs.json (consumed by tools/gauge.py) plus a
full-numbers report at research/reversal_gauge/reports/reversal_gauge_v0.md.

Run from the repo root:
    /home/moi/miniforge3/envs/bayesian/bin/python \
        research/reversal_gauge/pipeline/fit_combiner.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

PROJECT_DIR = Path("research/reversal_gauge")
EVENTS_PATH = PROJECT_DIR / "events.parquet"
COEFFS_PATH = PROJECT_DIR / "coeffs.json"
REPORT_PATH = PROJECT_DIR / "reports" / "reversal_gauge_v0.md"

# Pinned feature order — must match coeffs.json and tools/gauge.py exactly.
FEATURES = [
    "giveback_frac",
    "pace_pts_s",
    "spike_score",
    "repoke",
    "worn_touches",
    "is_flushV",
    "clock_sin",
    "clock_cos",
]

N_FOLDS = 5
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000
N_RELIABILITY_BINS = 10
RTH_START_MINUTE = 9 * 60 + 30  # 09:30 ET session open
RTH_SESSION_MINUTES = 6 * 60    # 09:30-15:30 ET session length
PROGRAM_AUC_CEILING = 0.57      # program-wide oscillator/runaway AUC ceiling


def _clock_angle(bucket: str) -> float:
    """Half-hour bucket 'HH:MM' (ET) -> angle in radians over the RTH session.

    09:30 maps to 0, each half-hour advances 2*pi/12; the session is not
    truly periodic but the full-circle mapping keeps every bucket distinct.
    """
    hh, mm = bucket.split(":")
    minute = int(hh) * 60 + int(mm) - RTH_START_MINUTE
    return 2.0 * math.pi * minute / RTH_SESSION_MINUTES


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["giveback_frac"] = df["giveback_frac"].astype(float)
    feats["pace_pts_s"] = df["pace_pts_s"].astype(float)
    feats["spike_score"] = df["spike_score"].astype(float)
    feats["repoke"] = df["repoke"].astype(float)
    feats["worn_touches"] = df["worn_touches"].astype(float)
    feats["is_flushV"] = (df["day_class"] == "flushV").astype(float)
    angles = df["clock_bucket"].map(_clock_angle).astype(float)
    feats["clock_sin"] = np.sin(angles)
    feats["clock_cos"] = np.cos(angles)
    return feats[FEATURES]


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    # A constant column (e.g. zero flushV days in a fold) would divide by 0;
    # std=1 leaves it centered and lets the coefficient go to ~0 harmlessly.
    stds = np.where(stds == 0.0, 1.0, stds)
    return means, stds


def cv_evaluate(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """GroupKFold CV by day; scaler fit on TRAIN folds only.

    Returns (fold AUCs, out-of-fold predicted probabilities).
    """
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_aucs = []
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in gkf.split(x, y, groups):
        means, stds = fit_standardizer(x[train_idx])
        z_train = (x[train_idx] - means) / stds
        z_test = (x[test_idx] - means) / stds
        model = LogisticRegression(
            C=LOGREG_C, solver="lbfgs", max_iter=LOGREG_MAX_ITER
        )
        model.fit(z_train, y[train_idx])
        p = model.predict_proba(z_test)[:, 1]
        oof[test_idx] = p
        if len(np.unique(y[test_idx])) < 2:
            fold_aucs.append(np.nan)  # degenerate fold; excluded via nanmean
        else:
            fold_aucs.append(roc_auc_score(y[test_idx], p))
    return np.array(fold_aucs, dtype=float), oof


def reliability_table(y: np.ndarray, p: np.ndarray) -> list[dict]:
    edges = np.linspace(0.0, 1.0, N_RELIABILITY_BINS + 1)
    rows = []
    for i in range(N_RELIABILITY_BINS):
        lo, hi = edges[i], edges[i + 1]
        last = i == N_RELIABILITY_BINS - 1
        mask = (p >= lo) & ((p <= hi) if last else (p < hi))
        n_bin = int(mask.sum())
        closer = "]" if last else ")"
        rows.append(
            {
                "bin": f"[{lo:.1f}, {hi:.1f}{closer}",
                "n": n_bin,
                "pred_mean": float(p[mask].mean()) if n_bin else float("nan"),
                "obs_rate": float(y[mask].mean()) if n_bin else float("nan"),
            }
        )
    return rows


def verdict_line(auc_mean: float, auc_std: float) -> str:
    if auc_mean - auc_std > PROGRAM_AUC_CEILING:
        return (
            f"VERDICT: cv AUC {auc_mean:.3f} +- {auc_std:.3f} clears the "
            f"program's {PROGRAM_AUC_CEILING:.2f} ceiling by more than one "
            f"fold-std — a real break, pending replication."
        )
    if auc_mean > PROGRAM_AUC_CEILING:
        return (
            f"VERDICT: cv AUC {auc_mean:.3f} +- {auc_std:.3f} is nominally "
            f"above the {PROGRAM_AUC_CEILING:.2f} ceiling but within one "
            f"fold-std — not a clear break."
        )
    return (
        f"VERDICT: cv AUC {auc_mean:.3f} +- {auc_std:.3f} does not beat the "
        f"program's {PROGRAM_AUC_CEILING:.2f} AUC ceiling."
    )


def write_report(
    n_total: int,
    n_resolved: int,
    n_dropped: int,
    n: int,
    n_days: int,
    base_rate: float,
    fold_aucs: np.ndarray,
    auc_mean: float,
    auc_std: float,
    gb_fold_aucs: np.ndarray,
    gb_mean: float,
    gb_std: float,
    const_auc: float,
    rel_rows: list[dict],
    coef: np.ndarray,
    intercept: float,
) -> None:
    lines = [
        "# Reversal Gauge v0 — combiner fit report",
        "",
        f"- events total: {n_total}",
        f"- resolved (label_resolved==1): {n_resolved}",
        f"- dropped non-finite feature rows: {n_dropped}",
        f"- used: n={n} events across {n_days} days",
        f"- base rate p(resume): {base_rate:.4f}",
        "",
        f"## Cross-validated AUC (GroupKFold {N_FOLDS} by day, label_resume)",
        "",
        "- fold AUCs: " + ", ".join(f"{a:.4f}" for a in fold_aucs),
        f"- mean +- std: {auc_mean:.4f} +- {auc_std:.4f}",
        "",
        "## Baselines",
        "",
        f"- giveback_frac alone (same CV protocol): "
        f"{gb_mean:.4f} +- {gb_std:.4f} "
        f"(folds: {', '.join(f'{a:.4f}' for a in gb_fold_aucs)})",
        f"- constant predictor: {const_auc:.4f} (0.5 by construction)",
        f"- combiner delta over giveback-only: {auc_mean - gb_mean:+.4f}",
        "",
        f"## Reliability ({N_RELIABILITY_BINS}-bin, out-of-fold predictions)",
        "",
        "| bin | n | mean predicted | observed resume rate |",
        "|---|---|---|---|",
    ]
    for row in rel_rows:
        pred = f"{row['pred_mean']:.3f}" if row["n"] else "-"
        obs = f"{row['obs_rate']:.3f}" if row["n"] else "-"
        lines.append(f"| {row['bin']} | {row['n']} | {pred} | {obs} |")
    lines += [
        "",
        "## Coefficients (standardized features, sorted by |coef|)",
        "",
        "| feature | coef |",
        "|---|---|",
    ]
    order = np.argsort(-np.abs(coef))
    for i in order:
        lines.append(f"| {FEATURES[i]} | {coef[i]:+.4f} |")
    lines += [
        f"| (intercept) | {intercept:+.4f} |",
        "",
        verdict_line(auc_mean, auc_std),
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    df = pd.read_parquet(EVENTS_PATH)
    n_total = len(df)
    df = df[df["label_resolved"] == 1].reset_index(drop=True)
    n_resolved = len(df)

    x_df = build_feature_matrix(df)
    finite = np.isfinite(x_df.to_numpy(dtype=float)).all(axis=1)
    n_dropped = int((~finite).sum())
    df = df[finite].reset_index(drop=True)
    x = x_df[finite].to_numpy(dtype=float)
    y = df["label_resume"].to_numpy(dtype=int)
    groups = df["day"].to_numpy()

    n = len(y)
    base_rate = float(y.mean())
    n_days = len(np.unique(groups))
    if n_days < N_FOLDS:
        raise RuntimeError(
            f"only {n_days} unique days; need >= {N_FOLDS} for GroupKFold"
        )

    fold_aucs, oof = cv_evaluate(x, y, groups)
    auc_mean = float(np.nanmean(fold_aucs))
    auc_std = float(np.nanstd(fold_aucs))

    gb_idx = FEATURES.index("giveback_frac")
    gb_fold_aucs, _ = cv_evaluate(x[:, [gb_idx]], y, groups)
    gb_mean = float(np.nanmean(gb_fold_aucs))
    gb_std = float(np.nanstd(gb_fold_aucs))

    const_auc = float(roc_auc_score(y, np.full(n, base_rate)))

    rel_rows = reliability_table(y, oof)

    # Final deployable model: standardized on ALL resolved rows.
    means, stds = fit_standardizer(x)
    z = (x - means) / stds
    final = LogisticRegression(
        C=LOGREG_C, solver="lbfgs", max_iter=LOGREG_MAX_ITER
    )
    final.fit(z, y)
    coef = final.coef_[0]
    intercept = float(final.intercept_[0])

    coeffs = {
        "features": FEATURES,
        "means": [float(v) for v in means],
        "stds": [float(v) for v in stds],
        "coef": [float(v) for v in coef],
        "intercept": intercept,
        "auc_cv_mean": auc_mean,
        "auc_cv_std": auc_std,
        "n": int(n),
        "base_rate": base_rate,
    }
    COEFFS_PATH.write_text(json.dumps(coeffs, indent=2) + "\n")

    write_report(
        n_total=n_total,
        n_resolved=n_resolved,
        n_dropped=n_dropped,
        n=n,
        n_days=n_days,
        base_rate=base_rate,
        fold_aucs=fold_aucs,
        auc_mean=auc_mean,
        auc_std=auc_std,
        gb_fold_aucs=gb_fold_aucs,
        gb_mean=gb_mean,
        gb_std=gb_std,
        const_auc=const_auc,
        rel_rows=rel_rows,
        coef=coef,
        intercept=intercept,
    )

    print(
        f"n={n} (of {n_resolved} resolved / {n_total} total; "
        f"{n_dropped} non-finite dropped), {n_days} days, "
        f"base rate {base_rate:.4f}"
    )
    print(f"cv AUC {auc_mean:.4f} +- {auc_std:.4f} | "
          f"giveback-only {gb_mean:.4f} +- {gb_std:.4f} | "
          f"constant {const_auc:.4f}")
    print(f"wrote {COEFFS_PATH} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
