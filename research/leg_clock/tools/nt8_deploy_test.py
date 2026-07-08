"""THE decisive test: does AG's entry signal make money on pristine NT8 tape?

Spec (agreed 2026-07-08):
- Train: logistic on canonical V2 stretch/velocity features (the same families
  AG's top-10 found: z_high/z_low/z_se/velocity/vwap on 1m+5m), 2024 Databento
  label entries vs matched same-day nulls. Features served through the
  ForwardPassSystem (FPS) — the leakproof harness (_last_closed_idx causality,
  handles NT8 end-of-bar timestamps).
- Deploy: iterate FPS over ATLAS_NT8 (138 days, Dec2025-Jun2026, ZERO labeler
  contamination, the tape live trades). Score every 1m close. Enter when the
  score clears a tier threshold (set on train quantiles), direction = FADE the
  stretch (labels' own geometry). Exit = causal 20t trail on 5s closes +
  15:55 CT flatten. Costs 4t/round-trip.
- Judge: $/day with day-block bootstrap 95% CI, PF-based trade WR, trades/day,
  per tier. Null = same machinery with within-day shuffled scores.
Registered prediction (from project history): flat-to-negative.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, _REPO)

from core_v2.FPS.forward_pass_system import ForwardPassSystem  # noqa: E402
from core_v2.features import FEATURE_NAMES  # noqa: E402

TICK = 0.25
TICK_VALUE = 0.50
COST_TICKS = 4.0
FAMS = ['z_high', 'z_low', 'z_se', 'z_close', 'price_velocity', 'price_accel', 'vwap']
COLS = [n for n in FEATURE_NAMES if any(f in n for f in FAMS)
        and ('_1m_' in n or '_5m_' in n)]
IDX = np.array([FEATURE_NAMES.index(c) for c in COLS])
ZH = COLS.index('L3_1m_z_high_30')
ZL = COLS.index('L3_1m_z_low_30')

ATLAS = os.path.join(_REPO, 'DATA', 'ATLAS')
NT8 = os.path.join(_REPO, 'DATA', 'ATLAS_NT8')
REPORT_DIR = os.path.join(_REPO, 'research', 'leg_clock', 'reports')
lines = []


def log(s):
    print(s, flush=True); lines.append(s)


def day_stream(day, root):
    """(ts[], feat[N,F], price[]) at 1m closes + full 5s (ts5, close5) via FPS."""
    fps = ForwardPassSystem(day=day, atlas_root=root,
                            features_root=os.path.join(root, 'FEATURES_5s_v2'),
                            labels_csv=os.path.join(ATLAS, 'regime_labels_2d.csv'),
                            build_v2_dict=False)
    ts_m, feats, px_m, ts5, px5 = [], [], [], [], []
    for bar in fps:
        if bar.v2_vector is None:
            continue
        ts5.append(bar.timestamp)
        px5.append(bar.price)
        if bar.is_1m_close:
            ts_m.append(bar.timestamp)
            feats.append(bar.v2_vector[IDX])
            px_m.append(bar.price)
    return (np.array(ts_m), np.array(feats, dtype=np.float64),
            np.array(px_m), np.array(ts5), np.array(px5))


def fit_logistic(X, y, epochs=400, lr=0.05):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    w = np.zeros(X.shape[1]); b = 0.0
    n1 = y.sum(); n0 = len(y) - n1
    wpos = n0 / max(n1, 1)
    for _ in range(epochs):
        z = Xn @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = (p - y) * np.where(y == 1, wpos, 1.0)
        w -= lr * (Xn.T @ g / len(y) + 1e-4 * w)
        b -= lr * g.mean()
    return w, b, mu, sd


def score(X, model):
    w, b, mu, sd = model
    return ((X - mu) / sd) @ w + b


def train_2024(rng):
    days = sorted(os.path.basename(f).replace('.parquet', '')
                  for f in glob.glob(os.path.join(ATLAS, '1m', '2024_*.parquet')))
    Xe, Xn_, bar_scores_pool = [], [], []
    used = 0
    for day in days:
        pick = os.path.join(_REPO, 'DATA', 'ai_cusp_picks',
                            f"ai_picks_{day.replace('_', '-')}_multi.json")
        if not os.path.exists(pick):
            continue
        try:
            ts_m, F, _, _, _ = day_stream(day, ATLAS)
        except Exception:
            continue
        if len(ts_m) < 100:
            continue
        entries = [t['entry_ts'] for t in json.load(open(pick)).get('trades', [])]
        e_idx = set()
        for ets in entries:
            i = np.searchsorted(ts_m, ets) - 1   # last 1m close strictly before entry
            if i >= 30:
                e_idx.add(int(i))
        if not e_idx:
            continue
        all_i = np.arange(30, len(ts_m))
        far = np.array([i for i in all_i
                        if min(abs(i - j) for j in e_idx) > 5])
        if len(far) < len(e_idx):
            continue
        n_idx = rng.choice(far, size=len(e_idx), replace=False)
        Xe.append(F[sorted(e_idx)])
        Xn_.append(F[n_idx])
        used += 1
    Xe, Xn_ = np.vstack(Xe), np.vstack(Xn_)
    Xe = np.nan_to_num(Xe); Xn_ = np.nan_to_num(Xn_)
    X = np.vstack([Xe, Xn_]); y = np.r_[np.ones(len(Xe)), np.zeros(len(Xn_))]
    model = fit_logistic(X, y)
    s = score(X, model)
    # AUC sanity
    order = s.argsort(); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    log(f"train: {used} days, {n1} entries vs {n0} nulls, train AUC {auc:.3f}")
    # thresholds from the NULL-bar score distribution (ordinary bars)
    thr = {tier: np.quantile(score(Xn_, model), q)
           for tier, q in [('tier_99.5', 0.995), ('tier_98', 0.98)]}
    return model, thr


def deploy_nt8(model, thr, shuffle=False, rng=None):
    days = sorted(os.path.basename(f).replace('.parquet', '')
                  for f in glob.glob(os.path.join(NT8, '1m', '*.parquet')))
    results = {k: [] for k in thr}          # tier -> list of (day, [trade $])
    import pytz, datetime as dtm
    central = pytz.timezone('US/Central')
    for day in days:
        try:
            ts_m, F, px_m, ts5, px5 = day_stream(day, NT8)
        except Exception:
            continue
        if len(ts_m) < 100:
            continue
        s = score(np.nan_to_num(F), model)
        if shuffle:
            s = rng.permutation(s)
        zsum = np.nan_to_num(F[:, ZH] + F[:, ZL])
        # 15:55 CT flatten epoch
        d0 = dtm.datetime.fromtimestamp(ts5[-1], tz=dtm.timezone.utc).astimezone(central)
        cut = central.localize(dtm.datetime(d0.year, d0.month, d0.day, 15, 55)).timestamp()
        for tier, th in thr.items():
            trades = []
            pos, entry, ext = 0, 0.0, 0.0
            k5 = 0
            for i in range(30, len(ts_m)):
                t = ts_m[i]
                # advance 5s pointer & manage open trade up to this minute
                while k5 < len(ts5) and ts5[k5] <= t:
                    p = px5[k5]
                    if pos != 0:
                        ext = max(ext, p) if pos > 0 else min(ext, p)
                        hit = (ext - p >= 20 * TICK) if pos > 0 else (p - ext >= 20 * TICK)
                        if hit or ts5[k5] >= cut:
                            pnl = (p - entry) / TICK * pos - COST_TICKS
                            trades.append(pnl * TICK_VALUE)
                            pos = 0
                    k5 += 1
                if pos == 0 and t < cut and s[i] >= th:
                    pos = -1 if zsum[i] > 0 else 1      # fade the stretch
                    entry = px_m[i]; ext = px_m[i]
            if pos != 0:
                pnl = (px5[-1] - entry) / TICK * pos - COST_TICKS
                trades.append(pnl * TICK_VALUE)
            results[tier].append(trades)
    return results


def report(tag, results):
    for tier, per_day in results.items():
        days_n = len(per_day)
        daily = np.array([sum(t) for t in per_day])
        allt = np.concatenate([np.array(t) for t in per_day if t]) if any(per_day) else np.array([])
        if len(allt) == 0:
            log(f"[{tag}:{tier}] no trades"); continue
        wins = allt[allt > 0].sum(); losses = -allt[allt < 0].sum()
        pf = wins / losses if losses > 0 else float('inf')
        rng = np.random.default_rng(1)
        boots = np.array([daily[rng.integers(0, days_n, days_n)].mean() for _ in range(4000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = 'NOT significant' if lo <= 0 <= hi else 'SIGNIFICANT'
        log(f"[{tag}:{tier}] {len(allt)/days_n:.1f} trades/d | $/trade {allt.mean():+.2f} | "
            f"$/day {daily.mean():+.1f} [CI {lo:+.1f},{hi:+.1f}] {sig} | "
            f"PF {pf:.2f} | tradeWR(PF-1) {pf-1:+.2f} | N={days_n}d/{len(allt)}tr")


def main():
    rng = np.random.default_rng(0)
    log(f"features ({len(COLS)}): {COLS}")
    model, thr = train_2024(rng)
    log(f"thresholds (null-score quantiles): { {k: round(v,3) for k,v in thr.items()} }")
    log("\n== NT8 deploy (pristine, causal exits) ==")
    report('REAL', deploy_nt8(model, thr))
    log("\n== NULL (within-day shuffled scores) ==")
    report('NULL', deploy_nt8(model, thr, shuffle=True, rng=np.random.default_rng(7)))

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, 'nt8_deploy_test.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
