"""Macro context via R-CURVE slope (trailing linear regression) over 1h & 4h, not just net-move.
Re-tests the macro-alignment finding with a robust regression trend, AND validates across OOS
sub-periods (does the with-macro edge hold in 2025-H1 / 2025-H2 / 2026, not just pooled?).
Trade-level (no day stats). CLEAN trades.

Output: reports/findings/kalman_macro_rcurve.md
"""
import os
import numpy as np
import pandas as pd

TRADES = 'reports/findings/kalman_clean_trades.csv'
ONE_S = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
H1, H4 = 3600, 14400
USD, COST = 2.0, 2.5


def roll_lin_slope(p, W):
    """Trailing linear-regression slope at each bar over window W (O(n), price/bar)."""
    n = len(p); out = np.full(n, np.nan)
    if n <= W:
        return out
    idx = np.arange(n, dtype=np.float64)
    csp = np.concatenate([[0.0], np.cumsum(p)])
    csg = np.concatenate([[0.0], np.cumsum(idx * p)])
    i = np.arange(W - 1, n)
    Sp = csp[i + 1] - csp[i - W + 1]
    Sg = csg[i + 1] - csg[i - W + 1]
    start = (i - W + 1).astype(np.float64)
    Stp = Sg - start * Sp
    St = W * (W - 1) / 2.0; Stt = (W - 1) * W * (2 * W - 1) / 6.0; vart = Stt - St * St / W
    out[i] = (Stp - St * Sp / W) / vart
    return out


def _arch(r, p75):
    g = r['net_usd'] / 2 + 1.25
    kept = g / r['mfe_pts'] if r['mfe_pts'] > 1e-6 else 0
    if r.get('gap_close', 0) == 1 and (r['exit_ts'] - r['entry_ts']) <= 120: return 'GAP_TRUNCATED'
    if r['net_usd'] <= -90: return 'STOPPED'
    if r['mfe_pts'] >= 20 and kept < 0.4: return 'GAVE_BACK'
    if r['mfe_pts'] < 10: return 'CHOP'
    if r['mfe_pts'] >= p75 and r['net_usd'] > 0 and kept >= 0.6: return 'CLEAN_RIDE'
    return 'SMALL_WIN' if r['net_usd'] > 0 else 'SMALL_LOSS'


def subperiod(day):
    if day.startswith('2026'): return '2026'
    if day[:7] in ('2025_01', '2025_02', '2025_03', '2025_04', '2025_05', '2025_06'): return '2025H1'
    if day.startswith('2025'): return '2025H2'
    return 'IS_2024'


def main():
    tr = pd.read_csv(TRADES)
    p75 = tr['mfe_pts'].quantile(0.75)
    tr['arch'] = tr.apply(lambda r: _arch(r, p75), axis=1)
    s1, s4 = [np.nan] * len(tr), [np.nan] * len(tr)
    tr = tr.reset_index(drop=True)
    for day, g in tr.groupby('day'):
        f = f'{ONE_S}/{day}.parquet'
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f, columns=['timestamp', 'close']).sort_values('timestamp')
        ts = d['timestamp'].to_numpy(np.int64); px = d['close'].to_numpy(np.float64)
        sl1 = roll_lin_slope(px, H1); sl4 = roll_lin_slope(px, H4)
        for idx, t in g.iterrows():
            ie = np.searchsorted(ts, int(t['entry_ts']))
            if ie < len(ts):
                s1[idx] = sl1[ie]; s4[idx] = sl4[ie]
    sgn = np.where(tr['dir'] == 'LONG', 1, -1)
    tr['with_1h'] = np.sign(s1) * sgn      # +1 = R-curve(1h) agrees with trade dir
    tr['with_4h'] = np.sign(s4) * sgn
    tr['sp'] = tr['day'].map(subperiod)
    tr['oos'] = tr['sp'] != 'IS_2024'

    def stat(sub):
        if not len(sub): return "0 | - | -"
        net = sub['net_usd'].to_numpy()
        pf = net[net > 0].sum() / abs(net[net < 0].sum()) if (net < 0).any() else np.inf
        return f"{len(sub)} | {net.mean():+.2f} | {pf:.2f}"

    L = ["# Macro context via R-CURVE slope (1h & 4h trailing regression) — CLEAN, trade-level\n",
         "with_macro = R-curve slope sign agrees with trade direction.\n",
         "## OOS — % AGAINST macro by archetype (R-curve)",
         "| archetype | n | %against 1h-R | %against 4h-R |", "|---|---|---|---|"]
    oos = tr[tr['oos']]
    for a in ['STOPPED', 'GAVE_BACK', 'CHOP', 'SMALL_LOSS', 'SMALL_WIN', 'CLEAN_RIDE']:
        s = oos[oos['arch'] == a]
        a1 = (s['with_1h'] < 0).mean() * 100 if s['with_1h'].notna().any() else np.nan
        a4 = (s['with_4h'] < 0).mean() * 100 if s['with_4h'].notna().any() else np.nan
        L.append(f"| {a} | {len(s)} | {a1:.0f}% | {a4:.0f}% |")

    L += ["", "## OOS — expectancy: with vs against R-curve macro (trades | net $/tr | PF)",
          f"- all OOS:        {stat(oos)}",
          f"- 1h-R WITH:      {stat(oos[oos['with_1h']>0])}",
          f"- 1h-R AGAINST:   {stat(oos[oos['with_1h']<0])}",
          f"- 4h-R WITH:      {stat(oos[oos['with_4h']>0])}",
          f"- 4h-R AGAINST:   {stat(oos[oos['with_4h']<0])}",
          f"- 1h&4h-R WITH:   {stat(oos[(oos['with_1h']>0)&(oos['with_4h']>0)])}"]

    L += ["", "## SUB-PERIOD VALIDATION — net $/tr (PF) per period: all vs 1h-R-WITH vs 4h-R-WITH",
          "| period | all | 1h-R WITH | 4h-R WITH | 1h&4h WITH |", "|---|---|---|---|---|"]
    for sp in ['IS_2024', '2025H1', '2025H2', '2026']:
        d = tr[tr['sp'] == sp]
        L.append(f"| {sp} | {stat(d)} | {stat(d[d['with_1h']>0])} | {stat(d[d['with_4h']>0])} | "
                 f"{stat(d[(d['with_1h']>0)&(d['with_4h']>0)])} |")
    L += ["", "Read: the with-macro edge is REAL only if 1h-R-WITH (or 4h) beats 'all' in EVERY OOS "
          "sub-period. If it flips sign across periods, it's period-luck, not an edge."]
    rep = "\n".join(L)
    os.makedirs('reports/findings', exist_ok=True)
    open('reports/findings/kalman_macro_rcurve.md', 'w', encoding='utf-8').write(rep)
    print(rep)


if __name__ == '__main__':
    main()
