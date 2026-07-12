"""
FPS HORIZON EXPLORER — exploration-level response measurement, NO stops/targets.

Moises' ruling (2026-07-12): mechanical stops are TRADE-MANAGEMENT level; the
exploration level measures the raw response — no risk is involved in a backtest,
and barriers censor the distribution (a stop is a clamp on the adverse path).

For every audited event: signed forward drift in the ARTICLE direction from the
trigger bar's 5s close to fixed horizons — no exits, no censoring:
    H = 1m, 5m, 15m, 30m, 1h, EOD
Per dossier x horizon x year: mean drift + day-block CI, median, MODE (2pt bins),
%>0, plus running MFE/MAE to each horizon (stored for the LATER management stage).
Flip direction = exact negation (no barriers -> antisymmetric), so stated numbers
cover both signs.

Bars come from the same canonical FPS stream (use_5s_price mode) — one engine.
Output: reports/AG_cat_00_HORIZON.md + per-event parquet reports/fps_horizons.parquet
"""
import os, sys, glob, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(BASE, '../..'))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from core_v2.FPS.forward_pass_system import ForwardPassSystem  # noqa: E402
import ag_phase5_doe as D                                       # noqa: E402
from fps_catalog_runner import rth_ts, load_triggers, FEATURES, LABELS  # noqa: E402

HOR = {'1m': 12, '5m': 60, '15m': 180, '30m': 360, '1h': 720}   # 5s bars
MODE_BIN = 2.0

def run(dossiers, days_limit=None):
    trig = load_triggers(dossiers)
    days = sorted(trig)
    if days_limit:
        days = days[:days_limit]
    rows = []
    t0, bars = time.time(), 0
    for k, day in enumerate(days):
        ts_map = rth_ts(day)
        if ts_map is None:
            continue
        try:
            fps = ForwardPassSystem(day=day, atlas_root=os.path.join(ROOT, 'DATA', 'ATLAS'),
                                    features_root=FEATURES, labels_csv=LABELS,
                                    tfs=['5s'], layers=['L1'], build_v2_dict=False,
                                    use_5s_price=True)
        except FileNotFoundError:
            continue
        # collect the day's canonical bar stream once
        ts_l, cl_l, hi_l, lo_l = [], [], [], []
        for st in fps:
            ts_l.append(int(st.ohlcv_5s['timestamp']))
            cl_l.append(st.ohlcv_5s['close']); hi_l.append(st.ohlcv_5s['high']); lo_l.append(st.ohlcv_5s['low'])
        bars += len(ts_l)
        ts_a = np.array(ts_l, dtype=np.int64)
        cl = np.array(cl_l); hi = np.array(hi_l); lo = np.array(lo_l)
        n = len(ts_a)
        pos_of_ts = {t: i for i, t in enumerate(ts_a)}
        for doss, idxs, bulls in trig[day]:
            ok = idxs < len(ts_map)
            for ei, bl in zip(idxs[ok], bulls[ok]):
                ets = int(ts_map[ei])
                i0 = pos_of_ts.get(ets)
                if i0 is None or i0 >= n - 13:
                    continue
                dirn = 1.0 if bl else -1.0
                e = cl[i0]
                rec = dict(doss=doss, day=day, year=day[:4], entry_ts=ets, is_long=bool(bl))
                for hname, hb in HOR.items():
                    j = min(i0 + hb, n - 1)
                    rec[f'pnl_{hname}'] = dirn * (cl[j] - e)
                    seg_h = hi[i0 + 1:j + 1]; seg_l = lo[i0 + 1:j + 1]
                    if len(seg_h):
                        rec[f'mfe_{hname}'] = (seg_h.max() - e) if bl else (e - seg_l.min())
                        rec[f'mae_{hname}'] = (e - seg_l.min()) if bl else (seg_h.max() - e)
                rec['pnl_eod'] = dirn * (cl[n - 1] - e)
                rows.append(rec)
        if (k + 1) % 100 == 0:
            print(f'  {k+1}/{len(days)} days, {len(rows)} events, {bars/(time.time()-t0):.0f} bars/s')
    return pd.DataFrame(rows)

def report(df):
    hs = list(HOR) + ['eod']
    lines = ["# Horizon Explorer — raw unstopped drift after each event (exploration level)\n",
             "Signed drift in the ARTICLE direction, no stops/targets (Moises 2026-07-12: "
             "management is a later level; barriers censor the distribution). Flip = exact "
             "negation. Day-block CIs. Mode = 2pt bins.\n"]
    flags = []
    for doss, g in df.groupby('doss'):
        lines.append(f"\n## {doss}")
        years = sorted(g['year'].unique())
        for h in hs:
            col = f'pnl_{h}'
            if col not in g:
                continue
            cells, both = [], True
            for y in years[:2]:
                gy = g[g['year'] == y]
                v = gy[col].dropna().values.astype(float)
                if len(v) < 50:
                    both = False; cells.append(f"{y}: thin"); continue
                ev, lo_, hi_ = D.day_ci(v, gy['day'].values)
                md = float(pd.Series((np.round(v / MODE_BIN) * MODE_BIN)).mode().iloc[0])
                cells.append(f"{y}: {ev:+.2f} [{lo_:+.2f},{hi_:+.2f}] med {np.median(v):+.1f} mode {md:+.0f} %>0 {(v>0).mean():.2f}")
                if not (lo_ > 0 or hi_ < 0):
                    both = False
            mark = "  <== both-year drift" if both else ""
            lines.append(f"- {h:>4}: " + " | ".join(cells) + mark)
            if both:
                flags.append((doss, h))
    lines.append(f"\n## Both-year significant drift (either sign): {len(flags)}")
    for doss, h in flags:
        lines.append(f"- {doss} @ {h}")
    out = os.path.join(BASE, 'reports', 'AG_cat_00_HORIZON.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f'wrote {out}  ({len(flags)} both-year drift cells)')

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    all_doss = sorted(os.path.basename(os.path.dirname(p))
                      for p in glob.glob(os.path.join(BASE, 'tests', '*', 'events.parquet')))
    df = run(args or all_doss, days_limit=3 if '--smoke' in sys.argv else None)
    print(f'{len(df)} events')
    if len(df):
        df.to_parquet(os.path.join(BASE, 'reports', 'fps_horizons.parquet'))
        report(df)
