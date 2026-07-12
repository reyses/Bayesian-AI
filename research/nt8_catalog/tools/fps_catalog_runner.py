"""
FPS CATALOG RUNNER — all catalog strategies through the canonical causal engine.

Design (doc 043):
  - Triggers: the AUDITED events.parquet of each dossier (article-faithful entries).
    Each event is mapped to its entry TIMESTAMP (RTH 5s bar at event_idx) — timestamp
    anchoring, never row-index joins across files (the bug class of docs 036/042).
  - Execution: ForwardPassSystem(use_5s_price=True) streams the day; entries fill at
    the trigger bar's 5s close; exits walk SUBSEQUENT 5s bars: stop-first-in-bar
    (conservative), target, else EOD close. Both directions (stated + flip), grid
    T/S in {(10,10),(10,20),(15,15),(20,20)}.
  - Heavy dossiers subsampled (cap events/day) — pre-registered, random_state=0.
  - Verdict gate (pre-registered, doc 043): TRADABLE-CANDIDATE iff day-block CI_lo
    > 0.5 pt (2 ticks friction) BOTH years at same config, N>=100/yr.

Run:  python tools/fps_catalog_runner.py [DOSSIER ...]   (default: all)
Smoke: python tools/fps_catalog_runner.py --smoke        (3 days, ROUND+PIVOT)
"""
import os, sys, glob, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(BASE, '../..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from core_v2.FPS.forward_pass_system import ForwardPassSystem  # noqa: E402
import ag_phase5_doe as D                                       # noqa: E402  (day_ci)

RAW5S = os.path.join(ROOT, 'DATA', 'ATLAS', '5s')
FEATURES = os.path.join(ROOT, 'DATA', 'ATLAS', 'FEATURES_5s_v2')
LABELS = os.path.join(ROOT, 'DATA', 'ATLAS', 'regime_labels_2d.csv')
GRID = [(10.0, 10.0), (10.0, 20.0), (15.0, 15.0), (20.0, 20.0)]
MAX_EV_PER_DAY = 30          # cap per dossier per day (heavy dossiers)
FRICTION_PTS = 0.5           # 2 MNQ ticks
RTH0, RTH1 = '08:30', '15:15'

def rth_ts(day_fmt):
    """RTH 5s bar timestamps for a day (entry-ts mapping for RTH-relative event_idx)."""
    p = os.path.join(RAW5S, f'{day_fmt}.parquet')
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=['timestamp'])
    dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    m = (dt.dt.time >= pd.Timestamp(RTH0).time()) & (dt.dt.time <= pd.Timestamp(RTH1).time())
    return df['timestamp'].values[m.values].astype(np.int64)

def load_triggers(dossiers):
    """{day_fmt: [(dossier, entry_ts, stated_long), ...]} using timestamp anchoring."""
    per_day = {}
    for doss in dossiers:
        ev = pd.read_parquet(os.path.join(BASE, 'tests', doss, 'events.parquet'))
        if int(ev['event_idx'].max()) >= 5000:   # non-RTH index space (SEASON full-session, RENKO bricks)
            print(f'[skip-trigger-space] {doss}')
            continue
        for day, de in ev.groupby('day'):
            if len(de) > MAX_EV_PER_DAY:
                de = de.sample(MAX_EV_PER_DAY, random_state=0)
            per_day.setdefault(day.replace('-', '_'), []).append(
                (doss, de['event_idx'].astype(int).values,
                 de['mode'].astype(str).str.startswith('bull').values))
    return per_day

def run(dossiers, days_limit=None):
    trig = load_triggers(dossiers)
    days = sorted(trig)
    if days_limit:
        days = days[:days_limit]
    trades = []          # (dossier, dir, T, S, day, pnl)
    t0 = time.time(); bars_done = 0
    for k, day in enumerate(days):
        ts_map = rth_ts(day)
        if ts_map is None:
            continue
        # entry timestamp per event (RTH index -> timestamp)
        sched = {}       # ts -> list of (dossier, stated_long)
        for doss, idxs, bulls in trig[day]:
            ok = idxs < len(ts_map)
            for ei, bl in zip(idxs[ok], bulls[ok]):
                sched.setdefault(int(ts_map[ei]), []).append((doss, bool(bl)))
        if not sched:
            continue
        try:
            fps = ForwardPassSystem(day=day, atlas_root=os.path.join(ROOT, 'DATA', 'ATLAS'),
                                    features_root=FEATURES, labels_csv=LABELS,
                                    tfs=['5s'], layers=['L1'],
                                    build_v2_dict=False, use_5s_price=True)
        except FileNotFoundError:
            continue
        open_pos = []    # dicts: dossier,dirn('stated'/'flip'),long,T,S,entry,day
        last_price = None
        for st in fps:
            bars_done += 1
            bar_ts = int(st.ohlcv_5s['timestamp'])
            hi, lo = st.ohlcv_5s['high'], st.ohlcv_5s['low']
            px = st.price
            last_price = px
            # exits first (positions opened on earlier bars)
            still = []
            for p in open_pos:
                if p['opened_ts'] >= bar_ts:
                    still.append(p); continue
                if p['long']:
                    hit_st, hit_tp = lo <= p['stp'], hi >= p['tpp']
                else:
                    hit_st, hit_tp = hi >= p['stp'], lo <= p['tpp']
                if hit_st:      # stop dominates in-bar (conservative)
                    trades.append((p['doss'], p['dirn'], p['T'], p['S'], p['day'], -p['S']))
                elif hit_tp:
                    trades.append((p['doss'], p['dirn'], p['T'], p['S'], p['day'], p['T']))
                else:
                    still.append(p)
            open_pos = still
            # entries at this bar's close
            for doss, stated_long in sched.get(bar_ts, []):
                for dirn, is_long in [('stated', stated_long), ('flip', not stated_long)]:
                    for (T, S) in GRID:
                        tpp = px + T if is_long else px - T
                        stp = px - S if is_long else px + S
                        open_pos.append(dict(doss=doss, dirn=dirn, long=is_long, T=T, S=S,
                                             entry=px, tpp=tpp, stp=stp, day=day,
                                             opened_ts=bar_ts))
        # EOD close leftovers
        for p in open_pos:
            pnl = (last_price - p['entry']) if p['long'] else (p['entry'] - last_price)
            trades.append((p['doss'], p['dirn'], p['T'], p['S'], p['day'], pnl))
        if (k + 1) % 50 == 0:
            print(f'  {k+1}/{len(days)} days, {len(trades)} trades, {bars_done/(time.time()-t0):.0f} bars/s')
    return pd.DataFrame(trades, columns=['doss', 'dirn', 'T', 'S', 'day', 'pnl'])

def report(tr, out_name='AG_cat_00_FPS_RESULTS.md'):
    lines = ["# FPS Catalog Run — all strategies through the canonical engine\n",
             "Entries = audited events (timestamp-anchored); fills at 5s close via "
             "ForwardPassSystem(use_5s_price=True); stop-first-in-bar; EOD close. "
             f"Gate: day-block CI_lo > {FRICTION_PTS}pt (2-tick friction) BOTH years, N>=100/yr.\n"]
    findings = []
    for (doss, dirn, T, S), g in tr.groupby(['doss', 'dirn', 'T', 'S']):
        years = sorted({d[:4] for d in g['day']})
        if len(years) < 2:
            continue
        cells, ok = [], True
        for y in years[:2]:
            gy = g[g['day'].str.startswith(y)]
            if len(gy) < 100:
                ok = False
            ev, lo, hi = D.day_ci(gy['pnl'].values.astype(float), gy['day'].values)
            cells.append(f"{y}: {ev:+.2f} [{lo:+.2f},{hi:+.2f}] N={len(gy)}")
            if not (lo > FRICTION_PTS):
                ok = False
        line = f"- {doss[:22]:24} {dirn:6} T{T:.0f}/S{S:.0f}: " + " | ".join(cells) + ("  <== CANDIDATE" if ok else "")
        lines.append(line)
        if ok:
            findings.append(line)
    lines.append(f"\n## CANDIDATES past the pre-registered gate: {len(findings)}")
    lines += findings if findings else ["- none. The catalog holds no strategy that clears "
                                        "friction in both years under canonical-engine execution."]
    out = os.path.join(BASE, 'reports', out_name)
    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f'\nwrote {out}  ({len(findings)} candidates)')
    return findings

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    smoke = '--smoke' in sys.argv
    all_doss = sorted(os.path.basename(os.path.dirname(p))
                      for p in glob.glob(os.path.join(BASE, 'tests', '*', 'events.parquet')))
    dossiers = args or (['ROUND-05_Psych_Numbers', 'PIVOT-16_Floor_Levels'] if smoke else all_doss)
    tr = run(dossiers, days_limit=3 if smoke else None)
    print(f'{len(tr)} trades total')
    if len(tr):
        report(tr, 'AG_cat_00_FPS_SMOKE.md' if smoke else 'AG_cat_00_FPS_RESULTS.md')
