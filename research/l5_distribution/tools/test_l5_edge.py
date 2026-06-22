"""PRELIMINARY L5 edge first-look (2024, self-contained — no NMP trades needed).

Question: at the NMP entry tail (|z|>Z_ENTRY on L3_1m_z_se_15), do any L5_1m intra-bar
distribution features separate the FORWARD snap-back outcome? Wedge methodology:
per-day Spearman(feature, snap_back_return), then day-block bootstrap CI on the mean.
A feature "separates" only if its 95% day-block CI excludes 0.

CAVEATS (read the report): (1) this is forward-RETURN separation, NOT NMP-trade PnL
(L5 is on 2024; NMP trades are 2025 w/ no 1s). (2) 12 features tested -> ~0.6 expected
false-positives at 95%; treat single hits skeptically (multiple comparisons). (3) snap-back
return is a directional proxy, not the actual R-trigger exit. PRELIMINARY — for user review.

Run: python research/test_l5_edge.py
"""
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ATLAS = 'DATA/ATLAS'
FEAT = f'{ATLAS}/FEATURES_5s_v2'
Z_ENTRY = 1.8481           # recalibrated NMP entry threshold (memory)
ZCOL = 'L3_1m_z_se_15'
K = 60                      # forward horizon in 5s bars = 5 minutes
MIN_TAIL = 30              # min tail rows/day to compute a daily Spearman
NB = 4000
RNG = np.random.RandomState(42)
L5_FEATS = [f'L5_1m_ldist_{s}' for s in
            ('n','min','q1','median','q3','max','mean','std','skew','kurtosis','outlier_pct','level')]
OUT = 'reports/findings/L5_edge_2024_preliminary.md'


def dayblock_ci(vals):
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    if len(vals) < 5:
        return np.nan, (np.nan, np.nan)
    boots = [np.mean(vals[RNG.randint(0, len(vals), len(vals))]) for _ in range(NB)]
    return float(vals.mean()), tuple(np.percentile(boots, [2.5, 97.5]))


def main():
    days = sorted(os.path.basename(p)[:-8] for p in glob.glob(f'{FEAT}/L5_1m/2024_*.parquet'))
    rows = []
    for day in days:
        try:
            z = pd.read_parquet(f'{FEAT}/L3_1m/{day}.parquet', columns=['timestamp', ZCOL])
            l5 = pd.read_parquet(f'{FEAT}/L5_1m/{day}.parquet')
            px = pd.read_parquet(f'{ATLAS}/5s/{day}.parquet', columns=['timestamp', 'close'])
        except (FileNotFoundError, KeyError):
            continue
        df = z.merge(l5, on='timestamp').merge(px, on='timestamp').sort_values('timestamp').reset_index(drop=True)
        if len(df) < K + MIN_TAIL:
            continue
        zz = df[ZCOL].to_numpy()
        fwd = df['close'].shift(-K) - df['close']
        snap = -np.sign(zz) * fwd                      # profitable snap-back > 0
        df['_snap'] = snap
        tail = df[(np.abs(zz) > Z_ENTRY) & df['_snap'].notna()]
        if len(tail) < MIN_TAIL:
            continue
        rec = {'day': day, 'n_tail': len(tail), 'base_snap': float(tail['_snap'].mean())}
        for f in L5_FEATS:
            x = tail[f]
            v = x.notna() & tail['_snap'].notna()
            rec[f] = spearmanr(x[v], tail['_snap'][v]).statistic if v.sum() >= MIN_TAIL else np.nan
        rows.append(rec)

    st = pd.DataFrame(rows)
    lines = []
    lines.append(f"# PRELIMINARY L5 edge first-look (2024)\n")
    lines.append(f"- Days with tail data: **{len(st)}**  |  total tail entries: **{int(st['n_tail'].sum()):,}**")
    bm, bci = dayblock_ci(st['base_snap'].tolist())
    sig0 = "" if (bci[0] <= 0 <= bci[1]) else " (CI excludes 0)"
    lines.append(f"- Baseline snap-back return at |z|>{Z_ENTRY} (fwd {K*5//60}min), per-day mean: "
                 f"**{bm:+.3f} pts**  95% day-block CI [{bci[0]:+.3f}, {bci[1]:+.3f}]{sig0}")
    lines.append(f"  (context: is the raw fade entry even +EV on 2024? — NOT the L5 question)\n")
    lines.append("## Per-feature separation: Spearman(L5 feature, snap-back return) at the |z| tail")
    lines.append("| L5_1m feature | mean daily rho | 95% day-block CI | separates? |")
    lines.append("|---|---|---|---|")
    hits = []
    for f in L5_FEATS:
        m, ci = dayblock_ci(st[f].tolist())
        sep = not (ci[0] <= 0 <= ci[1]) if not np.isnan(m) else False
        if sep:
            hits.append(f.replace('L5_1m_ldist_', ''))
        lines.append(f"| {f.replace('L5_1m_ldist_','')} | {m:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                     f"{'**YES**' if sep else 'no'} |")
    lines.append("")
    lines.append(f"## Verdict (PRELIMINARY)")
    if hits:
        lines.append(f"- Features whose 95% day-block CI excludes 0: **{', '.join(hits)}** "
                     f"({len(hits)}/12).")
        lines.append(f"- Multiple-comparisons: ~0.6 false-positives expected at 95% over 12 tests — "
                     f"{'plausibly real if >2 and large' if len(hits) > 2 else 'TREAT A SINGLE HIT AS NOISE until replicated'}.")
    else:
        lines.append("- **No L5_1m feature separates the forward snap-back at the |z| tail** "
                     "(all CIs include 0). On this 2024 forward-return proxy, L5 adds no entry-filter edge.")
    lines.append("\n### Caveats")
    lines.append("- Forward-return proxy, NOT NMP-trade PnL (R-trigger exit). Necessary-not-sufficient.")
    lines.append("- 1m TF only; other TFs untested here. Validate-before-bake: do NOT add to FEATURE_NAMES on this alone.")
    lines.append(f"- Horizon K={K} (={K*5//60}min); Z_ENTRY={Z_ENTRY}; day-block bootstrap {NB} resamples.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(lines))
    print('\n'.join(lines))
    print(f"\n[written to {OUT}]")


if __name__ == '__main__':
    main()
