"""Direction predictor — cheap nonlinear PRE-SCREEN (funnel step 1+2), null-anchored, day-disjoint OOS.

QUESTION: do CAUSAL F-space features at a regime's START predict the FORWARD direction (sign of the move),
BEYOND a Fourier null? This is the gate before investing in a Mamba.

- ANCHORS: pristine-regime start indices (raw_start_idx) from the enriched Stage-1 (tiled B2T).
- X: the causal F-space row AT the anchor (trailing-window features; drop timestamp/price_mean/vwap).
- y: sign(close[anchor + h] - close[anchor]) for several forward horizons h (incl. MACRO minutes,
  since the micro 1-60s linear test found nothing).
- MODEL: HistGradientBoostingClassifier (nonlinear, NaN-safe, no extra deps).
- OOS: train on days 1-3, test on disjoint days 4-5. Score REAL vs FOURIER null (same pipeline on the surrogate).
- VERDICT: signal iff REAL test-AUC > 0.5 AND > null test-AUC. If not, do NOT build the Mamba.

Firewall: features at the anchor are causal (trailing). The label is a strictly-forward realized move.
The regime's own (non-causal) duration is NOT used — horizons are fixed.
"""
import os, sys, json
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

COMMON_COLS = None   # feature columns present in ALL day/kind parquets (set in main; aligns ragged schemas)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ART = os.path.join(ROOT, "artifacts")
FEAT_DIR = "FEATURES_RUN_B2T"   # --feat_dir to override (e.g. FEATURES_RUN_B2TF for L4/L5)
WAIT = 0                         # --wait N: anchor at regime start + N seconds (wait-N confirmation)
OHLCV = os.path.join(ROOT, "DATA", "ATLAS", "1s")
DAYS = ['2024_02_20', '2024_02_21', '2024_02_22', '2024_02_23', '2024_02_26']
TRAIN_DAYS, TEST_DAYS = DAYS[:3], DAYS[3:]
HORIZONS = [30, 60, 120, 300]   # forward seconds (micro -> macro/psychohistory)


def seg_path(day, kind):
    if day == '2024_02_20':
        return os.path.join(ART, "stage1_B2Tmap_REAL_segments_2024_02_20.json") if kind == 'REAL' \
            else os.path.join(ART, "stage1_B2Tmap_FOUR_segments_2024_02_20_FOUR.json")
    if kind == 'REAL':
        return os.path.join(ART, f"stage1_WK_{day}_REAL_segments_{day}.json")
    return os.path.join(ART, f"stage1_WK_{day}_FOUR_segments_{day}_FOUR.json")


def build_day(day, kind):
    dk = day if kind == 'REAL' else f"{day}_FOUR"
    feats = pd.read_parquet(os.path.join(ROOT, "DATA", "ATLAS", FEAT_DIR, f"{dk}.parquet"))
    close = pd.read_parquet(os.path.join(OHLCV, f"{dk}.parquet"))['close'].to_numpy(np.float64)
    segs = json.load(open(seg_path(day, kind)))
    anchors = [s['raw_start_idx'] + WAIT for s in segs if s['status'] == 'PRISTINE']
    Xmat = feats[COMMON_COLS].to_numpy(np.float64)
    X, Y = [], {h: [] for h in HORIZONS}
    for a in anchors:
        if a >= len(Xmat) or a >= len(close):
            continue
        X.append(Xmat[a])
        for h in HORIZONS:
            Y[h].append(1 if (a + h < len(close) and close[a + h] - close[a] > 0)
                        else (0 if a + h < len(close) else np.nan))
    return np.array(X), {h: np.array(Y[h]) for h in HORIZONS}


def gather(days, kind):
    Xs, Ys = [], {h: [] for h in HORIZONS}
    for d in days:
        X, Y = build_day(d, kind)
        if len(X) == 0:
            continue
        Xs.append(X)
        for h in HORIZONS:
            Ys[h].append(Y[h])
    return np.vstack(Xs), {h: np.concatenate(Ys[h]) for h in HORIZONS}


def screen(kind, p):
    Xtr, Ytr = gather(TRAIN_DAYS, kind)
    Xte, Yte = gather(TEST_DAYS, kind)
    out = {}
    for h in HORIZONS:
        ytr, yte = Ytr[h], Yte[h]
        mtr = ~np.isnan(ytr); mte = ~np.isnan(yte)
        if mtr.sum() < 200 or mte.sum() < 100 or len(np.unique(ytr[mtr])) < 2:
            out[h] = None; continue
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(Xtr[mtr], ytr[mtr].astype(int))
        pr = clf.predict_proba(Xte[mte])[:, 1]
        auc = roc_auc_score(yte[mte].astype(int), pr)
        acc = accuracy_score(yte[mte].astype(int), (pr > 0.5).astype(int))
        upr = float(np.mean(yte[mte]))   # class balance (P up) on test
        out[h] = (auc, acc, upr, int(mte.sum()))
    return out


def _common_cols():
    cols = None
    for d in DAYS:
        for kind in ('REAL', 'FOUR'):
            dk = d if kind == 'REAL' else f"{d}_FOUR"
            p = os.path.join(ROOT, "DATA", "ATLAS", FEAT_DIR, f"{dk}.parquet")
            if not os.path.exists(p):
                continue
            names = set(x for x in pq.read_schema(p).names
                        if x != 'timestamp' and 'price_mean' not in x and 'vwap' not in x)
            cols = names if cols is None else (cols & names)
    return sorted(cols)


def main():
    global FEAT_DIR, WAIT, COMMON_COLS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat_dir', default=FEAT_DIR)
    ap.add_argument('--wait', type=int, default=WAIT)
    a = ap.parse_args(); FEAT_DIR = a.feat_dir; WAIT = a.wait
    COMMON_COLS = _common_cols()
    lines = []
    def w(s): print(s); lines.append(s)
    w(f"# Direction pre-screen — {FEAT_DIR} | wait={WAIT}s (anchor = regime start + wait) -> forward direction")
    w(f"train days {TRAIN_DAYS} -> test days {TEST_DAYS} (disjoint) | HistGradientBoosting | horizons {HORIZONS}s\n")
    real = screen('REAL', 'r'); null = screen('FOUR', 'n')
    w(f"{'horizon':>8} | {'REAL AUC':>9} {'acc':>6} | {'NULL AUC':>9} | {'AUC gap':>8} | nTest(real)")
    w("-" * 64)
    any_sig = False
    for h in HORIZONS:
        r, n = real.get(h), null.get(h)
        if r is None:
            w(f"{h:>7}s | {'n/a':>9}"); continue
        ns = f"{n[0]:>9.3f}" if n else "   no-null"
        gap = (r[0] - n[0]) if n else float('nan')
        flag = " <-- REAL>0.5 & >null" if (r[0] > 0.52 and n and r[0] > n[0] + 0.02) else ""
        if flag: any_sig = True
        w(f"{h:>7}s | {r[0]:>9.3f} {r[1]:>5.0%} | {ns} | {gap:>+8.3f} | {r[3]} (P_up={r[2]:.0%}){flag}")
    w("")
    w("VERDICT: " + ("REAL beats null+0.5 at >=1 horizon -> SIGNAL; proceed to Mamba." if any_sig
                     else "REAL ~ null ~ 0.5 at all horizons -> NO forecastable direction; do NOT build Mamba."))
    w("(AUC 0.5 = chance. Need REAL AUC > ~0.52 AND clearly above the Fourier null to claim signal. n=5 days, 1 null draw.)")
    out = os.path.join(ROOT, "research", "fspace_cadence", "reports",
                       f"direction_screen_{FEAT_DIR.replace('FEATURES_RUN_','')}_w{WAIT}.md")
    open(out, 'w').write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
