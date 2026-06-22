"""#1: Curate the EDGE-CASE sample from the CLEAN (gap-guarded) trades for 3-way verification
(Gemini/Claude/human) → Gemma few-shot teaching set. Self-extracts each trade's path from 1s
(no dependency on the stale contaminated trade_paths.parquet).

Archetypes: CLEAN_RIDE, GAVE_BACK, CHOP, STOPPED, SMALL_WIN, SMALL_LOSS + extremes.
Output: reports/findings/edge_cases_clean/trade_*.png + edge_case_manifest.csv (+ .md)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRADES = 'reports/findings/kalman_clean_trades.csv'
ONE_S = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
OUT = 'reports/findings/edge_cases_clean'
PER = 6
RNG = np.random.RandomState(42)


def archetype(r):
    # 3-way-verified rules (2026-06-16): GAP_TRUNCATED (Gemini-caught, short gap-closes only);
    # GAVE_BACK no longer gated on p75 (caught misclassified small-losses with real given-back MFE).
    g = r['net_usd'] / 2 + 1.25
    kept = g / r['mfe_pts'] if r['mfe_pts'] > 1e-6 else 0
    if r.get('gap_close', 0) == 1 and r['dur_s'] <= 120: return 'GAP_TRUNCATED'
    if r['net_usd'] <= -90: return 'STOPPED'
    if r['mfe_pts'] >= 20 and kept < 0.4: return 'GAVE_BACK'   # real move given back (any MFE>=20pt)
    if r['mfe_pts'] < 10: return 'CHOP'
    if r['big_mfe'] and r['net_usd'] > 0 and kept >= 0.6: return 'CLEAN_RIDE'
    return 'SMALL_WIN' if r['net_usd'] > 0 else 'SMALL_LOSS'


PROPOSED = {
    'CLEAN_RIDE': ('ok', 'ok', 'caught a real move, kept most of it'),
    'GAVE_BACK': ('ok-but-late', 'too-wide (79pt trail)', 'real move, surrendered most of peak'),
    'CHOP': ('questionable', 'n/a', 'never developed (MFE<10pt) — likely false-start entry'),
    'STOPPED': ('questionable', 'stop (ok)', 'reversed to -50pt stop; check if bad entry'),
    'SMALL_WIN': ('ok?', 'ok?', 'marginal'),
    'SMALL_LOSS': ('ok?', 'ok?', 'marginal'),
}


def main():
    tr = pd.read_csv(TRADES)
    # self-extract each trade's signed path from 1s
    paths = [None] * len(tr)
    tr = tr.reset_index(drop=True)
    for day, g in tr.groupby('day'):
        f = f'{ONE_S}/{day}.parquet'
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f, columns=['timestamp', 'close']).sort_values('timestamp')
        ts = d['timestamp'].to_numpy(np.int64); px = d['close'].to_numpy(np.float64)
        for idx, t in g.iterrows():
            ie = np.searchsorted(ts, int(t['entry_ts'])); ix = np.searchsorted(ts, int(t['exit_ts']))
            if ie < len(ts) and ix >= ie:
                seg = px[ie:ix + 1]
                paths[idx] = (seg - t['entry_price']) if t['dir'] == 'LONG' else (t['entry_price'] - seg)
    tr['path'] = paths
    tr = tr[tr['path'].apply(lambda p: p is not None and len(p) >= 2)].reset_index(drop=True)
    tr['big_mfe'] = tr['mfe_pts'] >= tr['mfe_pts'].quantile(0.75)
    tr['dur_s'] = tr['exit_ts'] - tr['entry_ts']
    tr['arch'] = tr.apply(archetype, axis=1)
    tr['tid'] = tr.index

    pick = []
    for a in ['CLEAN_RIDE', 'GAVE_BACK', 'CHOP', 'STOPPED', 'SMALL_WIN', 'SMALL_LOSS']:
        sub = tr[tr['arch'] == a]
        if len(sub):
            pick += list(sub.sample(min(PER, len(sub)), random_state=42)['tid'])
    for col, asc in [('net_usd', False), ('net_usd', True), ('dur_s', False), ('dur_s', True), ('mfe_pts', False)]:
        pick.append(tr.sort_values(col, ascending=asc)['tid'].iloc[0])
    pick = list(dict.fromkeys(pick))

    os.makedirs(OUT, exist_ok=True)
    man = []
    for tid in pick:
        r = tr.loc[tid]; path = np.asarray(r['path'], float)
        pk = int(np.argmax(path)); tr_idx = int(np.argmin(path)); t = np.arange(len(path)) / 60.0
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(t, path, color='#1565c0', lw=1.2); ax.axhline(0, color='gray', lw=0.6)
        ax.scatter([0], [0], color='green', s=60, zorder=5, label='entry')
        ax.scatter([t[-1]], [path[-1]], color='red', s=60, zorder=5, label='exit')
        ax.scatter([t[pk]], [path[pk]], color='orange', s=50, marker='^', zorder=5, label='MFE')
        ax.scatter([t[tr_idx]], [path[tr_idx]], color='purple', s=40, marker='v', zorder=5, label='MAE')
        el, xl, note = PROPOSED.get(r['arch'], ('?', '?', ''))
        ax.set_title(f"#{tid} {r['arch']} | {r['dir']} {r['split']} | net ${r['net_usd']:.0f} "
                     f"mfe {r['mfe_pts']:.0f}pt | entry:{el} exit:{xl}", fontsize=8)
        ax.set_xlabel('minutes in trade'); ax.set_ylabel('PnL (pts, signed)'); ax.legend(fontsize=6)
        fig.tight_layout(); p = f'{OUT}/trade_{tid:05d}_{r["arch"]}.png'; fig.savefig(p, dpi=90); plt.close(fig)
        man.append(dict(tid=tid, day=r['day'], split=r['split'], dir=r['dir'], arch=r['arch'],
                        net_usd=round(r['net_usd'], 1), mfe_pts=round(r['mfe_pts'], 1),
                        mae_pts=round(r['mae_pts'], 1), dur_min=round(r['dur_s'] / 60, 1),
                        gap_close=int(r.get('gap_close', 0)),
                        proposed_entry=el, proposed_exit=xl, note=note,
                        verify_gemini='', verify_claude='', verify_human='', plot=os.path.basename(p)))
    m = pd.DataFrame(man); m.to_csv(f'{OUT}/edge_case_manifest.csv', index=False)
    L = [f"# Edge-case teaching set (CLEAN trades) — {len(m)} for 3-way verify → Gemma\n",
         f"archetype counts (clean 5k): " + ", ".join(f"{k}={v}" for k, v in tr['arch'].value_counts().items()),
         f"worst loss in clean set: ${tr['net_usd'].min():.0f} (gap artifacts removed)\n",
         "| tid | arch | dir | net$ | mfe | dur(m) | gap | Claude: entry/exit | note |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in m.iterrows():
        L.append(f"| {r['tid']} | {r['arch']} | {r['dir']} | {r['net_usd']} | {r['mfe_pts']} | "
                 f"{r['dur_min']} | {r['gap_close']} | {r['proposed_entry']}/{r['proposed_exit']} | {r['note']} |")
    open(f'{OUT}/edge_case_manifest.md', 'w', encoding='utf-8').write("\n".join(L))
    print("\n".join(L).encode('ascii', 'replace').decode())
    print(f"\n[{len(m)} plots + manifest -> {OUT}/]")


if __name__ == '__main__':
    main()
