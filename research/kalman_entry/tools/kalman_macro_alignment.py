"""Test the hypothesis: STOPPED trades are entered AGAINST the macro trend.
The entry only checks the 24-min blue slope, so a trade can be with-24m but against the
1h/4h macro -> snaps back -> stops. Tag each trade with/against macro (1h & 4h net move
into entry), and check (a) is STOPPED disproportionately against-macro, (b) does a
with-macro entry filter fix it. Trade-level (no day stats). CLEAN trades.

Output: reports/findings/kalman_macro_alignment.md
"""
import os
import numpy as np
import pandas as pd

TRADES = 'reports/findings/kalman_clean_trades.csv'
ONE_S = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
H1, H4 = 3600, 14400          # macro lookbacks (sec) into the entry
USD, COST = 2.0, 2.5


def _arch(r, p75):
    g = r['net_usd'] / 2 + 1.25
    kept = g / r['mfe_pts'] if r['mfe_pts'] > 1e-6 else 0
    if r.get('gap_close', 0) == 1 and (r['exit_ts'] - r['entry_ts']) <= 120: return 'GAP_TRUNCATED'
    if r['net_usd'] <= -90: return 'STOPPED'
    if r['mfe_pts'] >= 20 and kept < 0.4: return 'GAVE_BACK'
    if r['mfe_pts'] < 10: return 'CHOP'
    if r['mfe_pts'] >= p75 and r['net_usd'] > 0 and kept >= 0.6: return 'CLEAN_RIDE'
    return 'SMALL_WIN' if r['net_usd'] > 0 else 'SMALL_LOSS'


def main():
    tr = pd.read_csv(TRADES)
    p75 = tr['mfe_pts'].quantile(0.75)
    tr['arch'] = tr.apply(lambda r: _arch(r, p75), axis=1)
    m1, m4 = [np.nan] * len(tr), [np.nan] * len(tr)
    tr = tr.reset_index(drop=True)
    for day, g in tr.groupby('day'):
        f = f'{ONE_S}/{day}.parquet'
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f, columns=['timestamp', 'close']).sort_values('timestamp')
        ts = d['timestamp'].to_numpy(np.int64); px = d['close'].to_numpy(np.float64)
        for idx, t in g.iterrows():
            ie = np.searchsorted(ts, int(t['entry_ts']))
            if ie >= len(ts):
                continue
            sgn = 1 if t['dir'] == 'LONG' else -1
            i1 = np.searchsorted(ts, int(t['entry_ts']) - H1)
            i4 = np.searchsorted(ts, int(t['entry_ts']) - H4)
            if ts[ie] - ts[i1] >= H1 * 0.7:
                m1[idx] = np.sign(px[ie] - px[i1]) * sgn      # +1 = WITH macro-1h
            if ts[ie] - ts[i4] >= H4 * 0.7:
                m4[idx] = np.sign(px[ie] - px[i4]) * sgn
    tr['with_1h'] = m1; tr['with_4h'] = m4
    tr['oos'] = tr['split'].isin(['OOS_H2_24', 'OOS_25_26'])

    L = ["# STOPPED = against-macro? (with/against 1h & 4h trend at entry) — CLEAN, trade-level\n",
         "with_macro = sign(price move over lookback into entry) matches trade direction.\n"]
    for scope, d in [('OOS', tr[tr['oos']]), ('IS', tr[~tr['oos']])]:
        L.append(f"## {scope}")
        # (a) % against-macro by archetype
        L.append("### (a) % AGAINST-macro by archetype (hypothesis: STOPPED is high)")
        L.append("| archetype | n | % against 1h | % against 4h |")
        L.append("|---|---|---|---|")
        for a in ['STOPPED', 'CHOP', 'GAVE_BACK', 'SMALL_LOSS', 'SMALL_WIN', 'CLEAN_RIDE']:
            s = d[d['arch'] == a]
            a1 = (s['with_1h'] < 0).sum() / s['with_1h'].notna().sum() * 100 if s['with_1h'].notna().sum() else np.nan
            a4 = (s['with_4h'] < 0).sum() / s['with_4h'].notna().sum() * 100 if s['with_4h'].notna().sum() else np.nan
            L.append(f"| {a} | {len(s)} | {a1:.0f}% | {a4:.0f}% |")
        # (b) with vs against macro outcomes + (c) entry-filter
        L.append("\n### (b) outcomes: with vs against macro  |  (c) entry-filter effect")
        L.append("| macro | filter | trades | net $/tr | PF | stop-rate | mfe |")
        L.append("|---|---|---|---|---|---|---|")
        def row(name, sub):
            if not len(sub): return f"| {name} | - | 0 | - | - | - | - |"
            net = sub['net_usd'].to_numpy()
            pf = net[net > 0].sum() / abs(net[net < 0].sum()) if (net < 0).any() else np.inf
            stop = (sub['arch'] == 'STOPPED').mean() * 100
            return f"| {name} | {len(sub)} | {net.mean():+.2f} | {pf:.2f} | {stop:.0f}% | {sub['mfe_pts'].mean():.0f} |"
        L.append(row("(all)", d))
        L.append(row("1h WITH", d[d['with_1h'] > 0]))
        L.append(row("1h AGAINST", d[d['with_1h'] < 0]))
        L.append(row("4h WITH", d[d['with_4h'] > 0]))
        L.append(row("4h AGAINST", d[d['with_4h'] < 0]))
        L.append(row("1h&4h WITH", d[(d['with_1h'] > 0) & (d['with_4h'] > 0)]))
        L.append("")
    rep = "\n".join(L)
    os.makedirs('reports/findings', exist_ok=True)
    open('reports/findings/kalman_macro_alignment.md', 'w', encoding='utf-8').write(rep)
    print(rep)


if __name__ == '__main__':
    main()
