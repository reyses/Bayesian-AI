"""Parity check: GPU/Torch FISTA solvers vs sklearn (the stage1-vs-stage2 question).

Stage 1 selects polynomial terms with `elasticnet_fista_cv` (Torch FISTA); stage 2
selects them with sklearn `ElasticNetCV`. This script measures how much that choice
actually matters:

  1. Synthetic ElasticNet parity  (active-set Jaccard, FISTA-CV vs sklearn-CV)
  2. Synthetic Group Lasso parity (active-set Jaccard)               <- was missing
  3. REAL segment-data parity     (active-set Jaccard AND, the number that drives
     the unification decision, the rate at which FISTA vs sklearn produce a
     DIFFERENT volatility tier on the same block)

Run:
    python "research/Regression segments/test_fista_parity.py"                 # synthetic only if no data
    python "research/Regression segments/test_fista_parity.py" --day 2025_01_02

Writes a report to reports/findings/<date>_fista_solver_parity.md
"""
import os
import sys
import time
import argparse
import datetime
import numpy as np
import torch
from sklearn.linear_model import ElasticNetCV
from group_lasso import GroupLasso

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.dirname(__file__))  # so we can import the real pipeline module

from core_v2.math.fista_gpu import elasticnet_fista_cv, group_lasso_fista_cv

# Active-set membership threshold (matches evaluate_block in stage1/stage2).
COEF_ACTIVE_THR = 1e-6
# Pass bar: mean active-set Jaccard at/above this is "solver-equivalent for selection".
JACCARD_PASS = 0.90
# Error band used for the real-data tier comparison (fraction of block price range).
ERROR_BAND_FRACTION = 0.10


def jaccard(a, b):
    a, b = set(np.asarray(a).astype(int).tolist()), set(np.asarray(b).astype(int).tolist())
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def device_default():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ───────────────────────── synthetic tests ─────────────────────────
def synthetic_elasticnet_parity(report, n_seeds=5, N=150, p=50, device=None):
    device = device or device_default()
    report.append("\n## 1. Synthetic ElasticNet parity (FISTA-CV vs sklearn-CV)")
    jacc, alpha_rows = [], []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        X = rng.randn(N, p)
        true_w = np.zeros(p)
        true_w[rng.choice(p, 10, replace=False)] = rng.randn(10) * 5
        y = X @ true_w + rng.randn(N) * 0.1

        enet = ElasticNetCV(cv=3, l1_ratio=0.5, fit_intercept=False, max_iter=1000)
        enet.fit(X, y)
        active_cpu = np.where(np.abs(enet.coef_) > COEF_ACTIVE_THR)[0]

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        w_gpu, best_alpha = elasticnet_fista_cv(X_t, y_t, l1_ratio=0.5, cv=3, alphas=50)
        active_gpu = np.where(np.abs(w_gpu.cpu().numpy()) > COEF_ACTIVE_THR)[0]

        j = jaccard(active_cpu, active_gpu)
        jacc.append(j)
        alpha_rows.append((seed, enet.alpha_, best_alpha, len(active_cpu), len(active_gpu), j))

    report.append("| seed | sklearn alpha | FISTA alpha | |sk| | |fista| | Jaccard |")
    report.append("|---|---|---|---|---|---|")
    for s, a_sk, a_f, n_sk, n_f, j in alpha_rows:
        report.append(f"| {s} | {a_sk:.5f} | {a_f:.5f} | {n_sk} | {n_f} | {j:.3f} |")
    _verdict(report, jacc)
    return jacc


def synthetic_group_lasso_parity(report, n_seeds=5, N=150, p=60, device=None):
    device = device or device_default()
    report.append("\n## 2. Synthetic Group Lasso parity (FISTA-CV vs sklearn)")
    jacc, rows = [], []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        X = rng.randn(N, p)
        groups = np.repeat(np.arange(p // 3), 3)[:p]
        true_w = np.zeros(p)
        true_w[:6] = rng.randn(6) * 5
        y = X @ true_w + rng.randn(N) * 0.1

        gl = GroupLasso(groups=groups, group_reg=0.01, l1_reg=0.0, n_iter=100,
                        fit_intercept=False, supress_warning=True)
        gl.fit(X, y)
        active_cpu = np.where(np.abs(gl.coef_.flatten()) > COEF_ACTIVE_THR)[0]

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        w_gpu, _ = group_lasso_fista_cv(X_t, y_t, groups, cv=3)
        active_gpu = np.where(np.abs(w_gpu.cpu().numpy()) > COEF_ACTIVE_THR)[0]

        j = jaccard(active_cpu, active_gpu)
        jacc.append(j)
        rows.append((seed, len(active_cpu), len(active_gpu), j))

    report.append("| seed | |sklearn| | |fista| | Jaccard |")
    report.append("|---|---|---|---|")
    for s, n_sk, n_f, j in rows:
        report.append(f"| {s} | {n_sk} | {n_f} | {j:.3f} |")
    _verdict(report, jacc)
    return jacc


# ───────────────────────── real-data test ─────────────────────────
def _select_terms(w_all, block_len):
    """Replicate evaluate_block's term selection: active set, then top-|w| cap."""
    terms = np.where(np.abs(w_all) > COEF_ACTIVE_THR)[0]
    max_features = min(15, max(1, block_len - 2))
    if len(terms) > max_features:
        # .copy() to avoid a negative-stride view (torch can't index with it) —
        # same guard evaluate_block uses.
        terms = np.argsort(np.abs(w_all))[::-1][:max_features].copy()
    return np.ascontiguousarray(terms)


def _tier_of(s1, X_poly_t, terms, Y_t, Y_cpu, E):
    if len(terms) == 0:
        return None
    Xf = X_poly_t[:, terms]
    beta = s1.ols_fit_pytorch(Xf, Y_t)
    if beta is None:
        return None
    preds = (Xf @ beta).cpu().numpy().flatten()
    return s1.categorize_segment(Y_cpu, preds, E)


def real_data_parity(report, day, atlas_root, n_blocks, block_len, device=None):
    device = device or device_default()
    report.append(f"\n## 3. Real segment-data parity (day {day}, {n_blocks} blocks x {block_len} bars)")
    try:
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        import stage1_speed_pass as s1
    except Exception as e:
        report.append(f"SKIPPED — could not import pipeline: {e}")
        return None

    try:
        features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
        df = s1.load_features([day], root=features_root)
        ohlcv = pd.read_parquet(os.path.join(atlas_root, '5s', f'{day}.parquet'))
    except Exception as e:
        report.append(f"SKIPPED — could not load data for {day}: {e}")
        return None

    min_len = min(len(df), len(ohlcv))
    df = df.iloc[:min_len]
    ohlcv = ohlcv.iloc[:min_len]
    feat_cols = [c for c in df.columns
                 if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]
    X = StandardScaler().fit_transform(df[feat_cols].values)
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    close = ohlcv['close'].values[valid]
    groups = s1.build_groups_from_columns(feat_cols)

    N = len(X)
    if N < block_len + 1:
        report.append(f"SKIPPED — only {N} valid rows, need >{block_len}.")
        return None

    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    cg = torch.tensor(close, dtype=torch.float32, device=device)
    starts = np.unique(np.linspace(0, N - block_len - 1, n_blocks).astype(int))

    jacc, tier_pairs, n_screened, n_tier_disagree = [], [], 0, 0
    for s in starts:
        X_t = Xg[s:s + block_len]
        raw = cg[s:s + block_len]
        Y_t = (raw - raw[0]).unsqueeze(1)
        X_cpu = X_t.cpu().numpy()
        Y_cpu = Y_t.cpu().numpy().flatten()

        active, _, _ = s1.screen_pipeline_cpu(X_cpu, Y_cpu, groups)
        if len(active) == 0:
            continue
        n_screened += 1
        X_poly_t = s1.poly_expand_gpu(X_t[:, active])

        # Stage 1 path: Torch FISTA
        w_f, _ = elasticnet_fista_cv(X_poly_t, Y_t, l1_ratio=0.5, cv=3, alphas=50)
        terms_f = _select_terms(w_f.cpu().numpy(), block_len)

        # Stage 2 path: sklearn
        enet = ElasticNetCV(l1_ratio=0.5, cv=3, n_jobs=1, fit_intercept=False, max_iter=1000)
        enet.fit(X_poly_t.cpu().numpy(), Y_cpu)
        terms_s = _select_terms(enet.coef_, block_len)

        jacc.append(jaccard(terms_f, terms_s))

        E = max(float(raw.max() - raw.min()) * ERROR_BAND_FRACTION, 1e-3)
        t_f = _tier_of(s1, X_poly_t, terms_f, Y_t, Y_cpu, E)
        t_s = _tier_of(s1, X_poly_t, terms_s, Y_t, Y_cpu, E)
        tier_pairs.append((int(s), t_f, t_s))
        if t_f != t_s:
            n_tier_disagree += 1

    if not jacc:
        report.append("SKIPPED — no blocks survived screening.")
        return None

    report.append(f"- Blocks screened with >=1 active cell: {n_screened}/{len(starts)}")
    report.append(f"- Mean term-selection Jaccard (FISTA vs sklearn): **{np.mean(jacc):.3f}** "
                  f"(min {np.min(jacc):.3f})")
    report.append(f"- **Tier disagreement rate: {n_tier_disagree}/{len(tier_pairs)} "
                  f"({100*n_tier_disagree/len(tier_pairs):.1f}%)**  <- the decision number")
    report.append("\n| block start | tier(FISTA) | tier(sklearn) | match |")
    report.append("|---|---|---|---|")
    for s, t_f, t_s in tier_pairs:
        report.append(f"| {s} | {t_f} | {t_s} | {'Y' if t_f == t_s else 'N'} |")

    report.append("\n**Interpretation:** if the tier-disagreement rate is near 0%, the "
                  "stage1/stage2 solver split is cosmetic and unifying stage2 to CPU-FISTA "
                  "is safe & low-value. A high rate means the current cross-stage tiers are "
                  "NOT comparable and unification is worth doing.")
    return jacc


def _verdict(report, jacc):
    if not jacc:
        report.append("_(no results)_")
        return
    m = float(np.mean(jacc))
    verdict = "PASS (solver-equivalent)" if m >= JACCARD_PASS else "DIVERGENT"
    report.append(f"\n-> mean Jaccard {m:.3f}, min {min(jacc):.3f} — **{verdict}** "
                  f"(threshold {JACCARD_PASS})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=str, default='2025_01_02', help="YYYY_MM_DD for real-data test")
    parser.add_argument('--atlas_root', type=str,
                        default="C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS")
    parser.add_argument('--n-blocks', type=int, default=25)
    parser.add_argument('--block-len', type=int, default=120)
    parser.add_argument('--skip-real', action='store_true', help="synthetic tests only")
    args = parser.parse_args()

    device = device_default()
    report = [f"# FISTA vs sklearn solver parity",
              f"_device: {device} | generated {datetime.date.today().isoformat()}_"]

    t0 = time.time()
    synthetic_elasticnet_parity(report, device=device)
    synthetic_group_lasso_parity(report, device=device)
    if not args.skip_real:
        real_data_parity(report, args.day, args.atlas_root, args.n_blocks, args.block_len, device)
    report.append(f"\n_total run time: {time.time()-t0:.1f}s_")

    out = "\n".join(report)
    # Write the report FIRST so a console-encoding hiccup never loses results.
    os.makedirs('reports/findings', exist_ok=True)
    out_path = f"reports/findings/{datetime.date.today().isoformat()}_fista_solver_parity.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out + "\n")
    # Print defensively (Windows cp1252 consoles choke on non-ASCII).
    try:
        print(out)
    except UnicodeEncodeError:
        print(out.encode('ascii', 'replace').decode('ascii'))
    print(f"\n[REPORT] saved to {out_path}")


if __name__ == "__main__":
    main()
