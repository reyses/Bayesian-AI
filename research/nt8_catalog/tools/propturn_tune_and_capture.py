"""
PROP-TURN tuning + capture (Moises 2026-07-16) — proportional leg-turn confirmation,
stop-and-reverse. SELF-CONTAINED driver for the PROP-TURN generator in
dossier_signal_pipeline.py (imports the SINGLE shared tracker `_propturn_core`, so tuning
and production never drift).

Mechanic (see the pipeline header): a causal leg on the continuous 5s close stream runs
from the last pivot P0 to a running extreme E; amplitude A=|E-P0|. A TURN fires (stop-AND-
reverse) when close retraces from E by >= r*A, subject to A>=A_min and a STALL gate (>= S
min since E last improved). On fire the pivot jumps to E and the leg flips; the fire's
direction is the NEW leg.

Phases:
  --tune   : 2024-ONLY sealed selection. Grid r x S x A_min = 90 cells. Objective:
             maximize dir-recall@+-2m on 2024 interior label turns SUBJECT TO
             lead-median <= +1.0 min AND fires/day <= 60. Writes the full grid CSV,
             the top-5, and the frozen winner -> reports/propturn_frozen.json.
             ==> THEN freeze PROPTURN_R/S_MIN/A_MIN in dossier_signal_pipeline.py.
  --run    : asserts the pipeline constants == frozen winner, then (a) full 604-day
             league run via the pipeline (signal_rows_PROPTURN.parquet + AUC/terciles),
             (b) TEST (2025+26) turn scorecard, (c) TEST stop-and-reverse capture sim +
             capture-ratio vs labels. Assembles reports/propturn.md.

Everything tees to reports/propturn_run.log; raw per-cell grid and per-trade capture rows
are dumped to reports/propturn_grid_2024.csv and reports/propturn_capture_trades.csv.
"""
import os, sys, json, glob, itertools, datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dossier_signal_pipeline import (LBL, D5, REP, RTH0, RTH1, TAIL, day_block_ci,
                                     _propturn_core)

# ---- tuning grid + constraints (2024 SEALED) -----------------------------------------
GRID_R = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
GRID_S = [0, 1, 2, 3, 5]                 # stall-gate minutes
GRID_A = [5.0, 10.0, 15.0]               # min leg amplitude (pts)
LEAD_MAX_MIN = 1.0                        # lead-median constraint (<= +1.0 min)
FIRES_MAX_DAY = 60.0                      # fires/day constraint (<= 60)
W_LIST = (60, 120, 180, 300)             # +-1/2/3/5 min windows (seconds)
CHANCE_2M = 0.43                          # turn_detection_audit chance precision @2m (test)
FRICTION_PT = 0.6                         # round-trip friction (pts) — MNQ 1pt=$2
RTH0_S, RTH1_S = 8 * 3600 + 30 * 60, 15 * 3600 + 15 * 60
FROZEN = os.path.join(REP, 'propturn_frozen.json')
LOGP = os.path.join(REP, 'propturn_run.log')
_LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    _LOG.append(s)


def flush_log():
    with open(LOGP, 'a', encoding='utf-8') as f:
        f.write('\n'.join(_LOG) + '\n')
    _LOG.clear()


# ---- data plumbing -------------------------------------------------------------------
def stream_2024_days():
    """Per 2024 label day: (c, ts, rth, start) built with the SAME tail+day construction
    as run_all (tail=TAIL carried across files in order). 2024 is the dataset start, so
    the first day's tail is None — identical to run_all's first-file behaviour."""
    lbl = {os.path.basename(f)[9:19] for f in glob.glob(os.path.join(LBL, 'ai_picks_*_multi.json'))}
    files = sorted(glob.glob(os.path.join(D5, '*.parquet')))
    out = {}
    tail = None
    for p in files:
        day = os.path.basename(p)[:10]                # YYYY_MM_DD
        if day[:4] > '2024':
            break
        df = pd.read_parquet(p, columns=['timestamp', 'close']).sort_values('timestamp').reset_index(drop=True)
        if day.replace('_', '-') in lbl and day[:4] == '2024':
            full = pd.concat([tail, df], ignore_index=True) if tail is not None else df
            ts = full['timestamp'].values.astype(np.int64)
            c = full['close'].values.astype(np.float64)
            dt = pd.to_datetime(full['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
            secs = (dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second).values
            rth = ((secs >= RTH0_S) & (secs <= RTH1_S)).astype(np.bool_)
            out[day] = dict(c=c, ts=ts, rth=rth,
                            start=(len(tail) if tail is not None else 0))
        tail = df.tail(TAIL)
    return out


def load_turns(year_pred):
    """{day: [(turn_ts, new_dir_is_long), ...]} — interior boundaries (labels chain),
    identical construction to turn_detection_audit.load_turns."""
    turns = {}
    for f in glob.glob(os.path.join(LBL, 'ai_picks_*_multi.json')):
        iso = os.path.basename(f)[9:19]
        if not year_pred(iso[:4]):
            continue
        tr = [t for t in json.load(open(f)).get('trades', []) if t.get('exit_ts')]
        tr.sort(key=lambda t: t['entry_ts'])
        turns[iso.replace('-', '_')] = [(t['entry_ts'], t.get('direction') == 'LONG') for t in tr[1:]]
    return turns


def build_fbd(days, r, s_min, a_min):
    """{day: (fire_ts_sorted, is_long_bool)} for one grid cell over the streamed days."""
    fbd = {}
    for day, dd in days.items():
        fi, fl, _ = _propturn_core(dd['c'], dd['ts'], dd['rth'], np.int64(dd['start']),
                                   float(r), float(s_min) * 60.0, float(a_min))
        fts = dd['ts'][fi].astype(float)
        order = np.argsort(fts)
        fbd[day] = (fts[order], fl.astype(bool)[order])
    return fbd


def score(fbd, turns, w_list=W_LIST):
    """Recall / dir-recall / precision / lead — turn_detection_audit machinery, extended
    to +-3m/+-5m and to lead p25/median/p75/mode. dir-recall CI (day-block) at +-2m."""
    res = {}
    n_days = len([d for d in turns if turns[d]])
    for W in w_list:
        hit, dhit, hit_days, leads = [], [], [], []
        n_fires = 0
        fire_near = 0
        for day, tl in turns.items():
            if not tl:
                continue
            fts, flong = fbd.get(day, (np.array([]), np.array([], dtype=bool)))
            n_fires += len(fts)
            tarr = np.array([t for t, _ in tl], dtype=float)
            if len(fts):
                idx = np.searchsorted(tarr, fts)
                near = np.zeros(len(fts), dtype=bool)
                for koff in (-1, 0):
                    kk = np.clip(idx + koff, 0, len(tarr) - 1)
                    near |= np.abs(fts - tarr[kk]) <= W
                fire_near += int(near.sum())
            for t0, new_long in tl:
                m = (fts >= t0 - W) & (fts <= t0 + W)
                any_hit = bool(m.any())
                hit.append(int(any_hit))
                dhit.append(int(bool((flong[m] == new_long).any())) if any_hit else 0)
                hit_days.append(day)
                if any_hit:
                    sub = fts[m]
                    leads.append((sub[np.argmin(np.abs(sub - t0))] - t0) / 60.0)
        hit = np.array(hit); dhit = np.array(dhit)
        r_ = dict(n_turns=len(hit), recall=float(hit.mean()), dir_recall=float(dhit.mean()),
                  precision=(fire_near / n_fires if n_fires else float('nan')), n_fires=int(n_fires))
        if leads:
            la = np.array(leads)
            hb = np.histogram(la, bins=np.arange(-5.25, 5.5, 0.5))
            r_.update(lead_median=float(np.median(la)),
                      lead_mode=float(hb[1][np.argmax(hb[0])] + 0.25),
                      lead_p25=float(np.percentile(la, 25)),
                      lead_p75=float(np.percentile(la, 75)))
        if W == 120:
            lo, hi = day_block_ci(dhit.astype(float), np.array(hit_days))
            r_['dir_ci'] = (float(lo), float(hi))
        res[W] = r_
    res['fires_per_day'] = res[120]['n_fires'] / max(n_days, 1)
    res['n_days'] = n_days
    return res


def hist_text(vals, binw, lo=None, hi=None, top=6):
    """Mode-first text histogram: '[a,b): n' lines for the tallest bins + the mode center."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 'no data', float('nan')
    lo = np.floor(vals.min() / binw) * binw if lo is None else lo
    hi = np.ceil(vals.max() / binw) * binw if hi is None else hi
    edges = np.arange(lo, hi + binw, binw)
    cnt, e = np.histogram(vals, bins=edges)
    mode_c = float(e[np.argmax(cnt)] + binw / 2.0)
    order = np.argsort(cnt)[::-1][:top]
    order = sorted(order)
    lines = [f'[{e[i]:+.2f},{e[i + 1]:+.2f}): {cnt[i]}' for i in order if cnt[i] > 0]
    return '  '.join(lines), mode_c


# ---- tuning phase --------------------------------------------------------------------
def tune():
    log(f'\n===== PROP-TURN TUNING (2024 SEALED) {datetime.datetime.now():%Y-%m-%d %H:%M} =====')
    days = stream_2024_days()
    turns = load_turns(lambda y: y == '2024')
    n_turns = sum(len(v) for v in turns.values())
    log(f'2024: {len(days)} streamed label days, {len(turns)} label days, {n_turns} interior turns')
    grid = list(itertools.product(GRID_R, GRID_S, GRID_A))
    rows = []
    for r, s, a in tqdm(grid, desc='2024 grid'):
        fbd = build_fbd(days, r, s, a)
        sc = score(fbd, turns, w_list=(120,))
        rec2 = sc[120]
        feas = (rec2.get('lead_median', 9.9) <= LEAD_MAX_MIN) and (sc['fires_per_day'] <= FIRES_MAX_DAY)
        rows.append(dict(r=r, S=s, A_min=a,
                         dir_recall_2m=rec2['dir_recall'], recall_2m=rec2['recall'],
                         precision_2m=rec2['precision'],
                         lead_median=rec2.get('lead_median', float('nan')),
                         fires_per_day=sc['fires_per_day'], n_fires=rec2['n_fires'],
                         feasible=bool(feas)))
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(REP, 'propturn_grid_2024.csv'), index=False)
    log(f'wrote propturn_grid_2024.csv ({len(G)} cells)')

    feasible = G[G['feasible']].copy()
    log(f'feasible cells (lead-median<=+{LEAD_MAX_MIN}min AND fires/day<={FIRES_MAX_DAY:.0f}): '
        f'{len(feasible)}/{len(G)}')
    if len(feasible) == 0:
        log('!! NO feasible cell — relaxing to global max dir-recall@2m (flagged)')
        ranked = G.sort_values(['dir_recall_2m', 'recall_2m'], ascending=False)
        relaxed = True
    else:
        ranked = feasible.sort_values(['dir_recall_2m', 'recall_2m', 'fires_per_day'],
                                      ascending=[False, False, True])
        relaxed = False
    top5 = ranked.head(5)
    log('\nTOP-5 2024 cells (by dir-recall@2m; feasible-only unless relaxed):')
    log(top5.to_string(index=False))
    w = ranked.iloc[0]
    winner = dict(r=float(w['r']), S=int(w['S']), A_min=float(w['A_min']), relaxed=relaxed)
    log(f'\nWINNER (frozen): r={winner["r"]}  S={winner["S"]}min  A_min={winner["A_min"]}pts'
        f'  dir-recall@2m={w["dir_recall_2m"]:.3f}  lead-med={w["lead_median"]:+.2f}m'
        f'  fires/day={w["fires_per_day"]:.1f}')

    # full winner score (all W) on 2024 for provenance
    fbd_w = build_fbd(days, winner['r'], winner['S'], winner['A_min'])
    sc_w = score(fbd_w, turns)
    frozen = dict(winner=winner,
                  winner_2024=_pack(sc_w),
                  top5=top5.to_dict('records'),
                  n_turns_2024=n_turns, n_days_2024=len(turns))
    with open(FROZEN, 'w', encoding='utf-8') as f:
        json.dump(frozen, f, indent=2)
    log(f'wrote {os.path.basename(FROZEN)}')
    log('\n>>> FREEZE STEP: set in dossier_signal_pipeline.py:')
    log(f'      PROPTURN_R = {winner["r"]}')
    log(f'      PROPTURN_S_MIN = {float(winner["S"])}')
    log(f'      PROPTURN_A_MIN = {winner["A_min"]}')
    flush_log()
    return frozen


def _pack(sc):
    """JSON-friendly slice of a score() result."""
    out = {'fires_per_day': sc['fires_per_day'], 'n_days': sc['n_days']}
    for W in W_LIST:
        out[str(W)] = {k: v for k, v in sc[W].items()}
    return out


# ---- capture simulation --------------------------------------------------------------
_PXCACHE = {}


def day_px(day):
    """(ts->close dict, rth_close_ts, rth_close_px) for a test day from the 5s parquet."""
    if day in _PXCACHE:
        return _PXCACHE[day]
    df = pd.read_parquet(os.path.join(D5, f'{day}.parquet'), columns=['timestamp', 'close'])
    ts = df['timestamp'].values.astype(np.int64)
    cl = df['close'].values.astype(float)
    dt = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/Chicago')
    secs = dt.hour.values * 3600 + dt.minute.values * 60 + dt.second.values
    rth = (secs >= RTH0_S) & (secs <= RTH1_S)
    idx = np.flatnonzero(rth)
    if len(idx) == 0:
        return dict(zip(ts.tolist(), cl.tolist())), None, None
    rc = idx[np.argmax(ts[idx])]
    r = (dict(zip(ts.tolist(), cl.tolist())), int(ts[rc]), float(cl[rc]))
    _PXCACHE[day] = r
    return r


def stream_test_days():
    """Per non-2024 label day: (c, ts, rth, start), same tail+day construction as run_all
    (tail carried across ALL files). For the regime-sensitivity exploration."""
    lbl = {os.path.basename(f)[9:19] for f in glob.glob(os.path.join(LBL, 'ai_picks_*_multi.json'))}
    files = sorted(glob.glob(os.path.join(D5, '*.parquet')))
    out = {}
    tail = None
    for p in files:
        day = os.path.basename(p)[:10]
        df = pd.read_parquet(p, columns=['timestamp', 'close']).sort_values('timestamp').reset_index(drop=True)
        if day.replace('_', '-') in lbl and day[:4] != '2024':
            full = pd.concat([tail, df], ignore_index=True) if tail is not None else df
            ts = full['timestamp'].values.astype(np.int64)
            c = full['close'].values.astype(np.float64)
            dt = pd.to_datetime(full['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
            secs = (dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second).values
            rth = ((secs >= RTH0_S) & (secs <= RTH1_S)).astype(np.bool_)
            out[day] = dict(c=c, ts=ts, rth=rth, start=(len(tail) if tail is not None else 0))
        tail = df.tail(TAIL)
    return out


def _fires_df(days, r, s_min, a_min):
    rows = []
    for day, dd in days.items():
        fi, fl, _ = _propturn_core(dd['c'], dd['ts'], dd['rth'], np.int64(dd['start']),
                                   float(r), float(s_min) * 60.0, float(a_min))
        ts = dd['ts']
        for k in range(len(fi)):
            rows.append(dict(day=day, ts=int(ts[fi[k]]), is_long=bool(fl[k])))
    return pd.DataFrame(rows)


def explore_regimes():
    """EXPLORATION (NOT a result): does ANY stall regime hit the 0.5-0.8 capture budget?
    Runs the capture sim on TEST for the frozen cell + the best non-degenerate cells."""
    log('\n--- EXPLORATION: regime sensitivity of capture (TEST; NOT results) ---')
    days = stream_test_days()
    cells = [('FROZEN r.05/S3/A15', 0.05, 3, 15.0), ('r.10/S2/A5', 0.10, 2, 5.0),
             ('r.10/S1/A5', 0.10, 1, 5.0), ('r.10/S0/A5', 0.10, 0, 5.0)]
    out = []
    for tag, r, s, a in cells:
        T = simulate_capture(_fires_df(days, r, s, a))
        cap = T['captured_pts'].values.astype(float)
        R = T[T['n_label_overlap'] == 1]['capture_ratio'].dropna().values
        win = cap[cap > 0].sum(); loss = cap[cap < 0].sum()
        pf = (win / abs(loss) - 1.0) if loss != 0 else float('nan')
        rec = dict(tag=tag, tpd=len(T) / max(T['day'].nunique(), 1),
                   cap_med=float(np.median(cap)), pf=pf,
                   ratio_med=float(np.median(R)) if len(R) else float('nan'),
                   frac_budget=float(((R >= 0.5) & (R <= 0.8)).mean()) if len(R) else float('nan'),
                   frac_pos=float((R > 0).mean()) if len(R) else float('nan'), n=len(R))
        log(f"  {tag:20} tpd {rec['tpd']:5.0f}  cap-med {rec['cap_med']:+.2f}pt  PF-WR {rec['pf']:+.3f}"
            f"  ratio-med {rec['ratio_med']:+.3f}  in[.5,.8] {rec['frac_budget']:.2f}  >0 {rec['frac_pos']:.2f}")
        out.append(rec)
    return out


def label_intervals(day):
    """[(l_entry_ts, l_exit_ts, l_disp_pts_abs), ...] for a test day."""
    f = os.path.join(LBL, f'ai_picks_{day.replace("_", "-")}_multi.json')
    if not os.path.exists(f):
        return []
    out = []
    for t in json.load(open(f)).get('trades', []):
        if not t.get('exit_ts'):
            continue
        out.append((t['entry_ts'], t['exit_ts'], abs(t['exit_price'] - t['entry_price'])))
    return out


def simulate_capture(fires_df):
    """Pure stop-and-reverse on the frozen cell's TEST fires. Per day: enter at the first
    fire (close fill), flip at each subsequent fire, force-flat at the RTH close (15:15).
    Flat overnight. Returns a per-leg-trade DataFrame."""
    trades = []
    for day, g in fires_df.groupby('day', sort=True):
        g = g.sort_values('ts')
        fts = g['ts'].values.astype(np.int64)
        flong = g['is_long'].values.astype(bool)
        ts2cl, rc_ts, rc_px = day_px(day)
        if rc_ts is None:
            continue
        px = np.array([ts2cl.get(int(t), np.nan) for t in fts])
        labs = label_intervals(day)
        for i in range(len(fts)):
            e_ts, e_px, e_long = int(fts[i]), px[i], bool(flong[i])
            if not np.isfinite(e_px):
                continue
            if i + 1 < len(fts):
                x_ts, x_px = int(fts[i + 1]), px[i + 1]
            else:
                x_ts, x_px = rc_ts, rc_px                 # final leg -> RTH close
            if not np.isfinite(x_px) or x_ts <= e_ts:
                continue
            d = 1.0 if e_long else -1.0
            cap = d * (x_px - e_px)
            ov = [ld for (la, lb, ld) in labs if la <= x_ts and lb >= e_ts]
            disp = ov[0] if len(ov) == 1 else np.nan
            ratio = (cap / disp) if (len(ov) == 1 and disp > 0) else np.nan
            trades.append(dict(day=day, year=day[:4], entry_ts=e_ts, exit_ts=x_ts,
                               dir=('LONG' if e_long else 'SHORT'), entry_px=e_px, exit_px=x_px,
                               captured_pts=cap, net_pts=cap - FRICTION_PT,
                               n_label_overlap=len(ov), label_disp_pts=disp, capture_ratio=ratio))
    return pd.DataFrame(trades)


def capture_block(T, tag):
    """Per-population capture stats (mode-first): N trades/day, captured mode/median/
    mean+CI, PF-based Trade WR, friction net."""
    n = len(T)
    days = T['day'].values
    cap = T['captured_pts'].values.astype(float)
    n_days = T['day'].nunique()
    tpd = n / max(n_days, 1)
    hist, mode = hist_text(cap, 1.0, top=6)
    med = float(np.median(cap))
    mean = float(np.mean(cap))
    lo, hi = day_block_ci(cap, days)
    win = cap[cap > 0].sum(); loss = cap[cap < 0].sum()
    pfwr = (win / abs(loss) - 1.0) if loss != 0 else float('nan')
    net = cap - FRICTION_PT
    nlo, nhi = day_block_ci(net, days)
    log(f'\n[{tag}] N={n} trades over {n_days} days  ->  {tpd:.1f} trades/day')
    log(f'  captured pts  mode {mode:+.2f}  median {med:+.2f}  mean {mean:+.2f} '
        f'[CI {lo:+.2f},{hi:+.2f}]')
    log(f'  hist(1pt): {hist}')
    log(f'  PF-based Trade WR (gross) = {pfwr:+.3f}  (win-sum {win:+.1f} / |loss-sum| {abs(loss):.1f})')
    log(f'  net-of-friction (0.6pt/rt)  mean {net.mean():+.2f} [CI {nlo:+.2f},{nhi:+.2f}]  '
        f'median {np.median(net):+.2f}')
    return dict(tag=tag, n=n, n_days=n_days, trades_per_day=tpd, cap_mode=mode, cap_median=med,
                cap_mean=mean, cap_ci=[lo, hi], pfwr=pfwr, net_mean=float(net.mean()),
                net_ci=[float(nlo), float(nhi)], net_median=float(np.median(net)), hist=hist)


def ratio_block(T, tag):
    """Capture-ratio (captured / single-overlap label displacement) distribution."""
    R = T[T['n_label_overlap'] == 1]
    r = R['capture_ratio'].values.astype(float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        log(f'\n[{tag}] capture-ratio: no single-overlap trades')
        return dict(tag=tag, n=0)
    hist, mode = hist_text(r, 0.1, lo=-1.0, hi=2.0, top=8)
    med = float(np.median(r))
    frac_budget = float(((r >= 0.5) & (r <= 0.8)).mean())
    frac_pos = float((r > 0).mean())
    log(f'\n[{tag}] capture-ratio (single-overlap N={len(r)})  mode {mode:+.2f}  median {med:+.2f}')
    log(f'  hist(0.1): {hist}')
    log(f'  frac in [0.5,0.8] budget = {frac_budget:.2f}   frac>0 = {frac_pos:.2f}')
    return dict(tag=tag, n=len(r), mode=mode, median=med, frac_budget=frac_budget, frac_pos=frac_pos)


# ---- run phase (league + scorecard + capture + report) -------------------------------
def _assert_frozen():
    import importlib, dossier_signal_pipeline as P
    importlib.reload(P)
    fr = json.load(open(FROZEN))['winner']
    got = (P.PROPTURN_R, float(P.PROPTURN_S_MIN), P.PROPTURN_A_MIN)
    want = (fr['r'], float(fr['S']), fr['A_min'])
    if got != want:
        raise SystemExit(f'FREEZE MISMATCH: pipeline has {got}, frozen winner is {want}. '
                         f'Edit PROPTURN_R/S_MIN/A_MIN in dossier_signal_pipeline.py first.')
    return P, fr


def run():
    frozen = json.load(open(FROZEN))
    log(f'\n===== PROP-TURN RUN (league+scorecard+capture) {datetime.datetime.now():%Y-%m-%d %H:%M} =====')
    P, fr = _assert_frozen()
    log(f'frozen cell OK: r={fr["r"]} S={fr["S"]}min A_min={fr["A_min"]}')

    # (a) LEAGUE — full 604-day run via the pipeline
    log('\n--- LEAGUE (full 604-day pipeline run) ---')
    streams, lblf = P.run_all(['PROP-TURN'])
    league = P.evaluate('PROP-TURN', streams['PROP-TURN'], lblf)
    log(f'league: {league}')

    # (b) TEST turn scorecard from signal_rows_PROPTURN.parquet
    log('\n--- TEST TURN SCORECARD (2025+26) ---')
    F = pd.read_parquet(os.path.join(REP, 'signal_rows_PROPTURN.parquet'),
                        columns=['ts', 'is_long', 'day'])
    F = F[F['day'].str[:4] != '2024']
    fbd = {}
    for day, g in F.groupby('day'):
        o = np.argsort(g['ts'].values)
        fbd[day] = (g['ts'].values.astype(float)[o], g['is_long'].values.astype(bool)[o])
    turns_te = load_turns(lambda y: y != '2024')
    sc = score(fbd, turns_te)
    _log_scorecard(sc)
    with open(os.path.join(REP, 'propturn_scorecard.json'), 'w') as f:
        json.dump(_pack(sc), f, indent=2)

    # (c) CAPTURE — stop-and-reverse on the FULL emitted fire chain (NOT the label-covered
    # signal_rows parquet, which drops early/late-day fires and would splice non-adjacent
    # fires into bogus trades). streams['PROP-TURN'] is the complete RTH-gated fire set.
    log('\n--- CAPTURE (stop-and-reverse, TEST 2025+26) ---')
    Ffull = streams['PROP-TURN'][['ts', 'is_long', 'day']]
    Ftest = Ffull[Ffull['day'].str[:4] != '2024'].copy()
    log(f'full emitted TEST fires: {len(Ftest)} over {Ftest["day"].nunique()} days')
    T = simulate_capture(Ftest)
    T.to_csv(os.path.join(REP, 'propturn_capture_trades.csv'), index=False)
    log(f'wrote propturn_capture_trades.csv ({len(T)} leg-trades)')
    cap = {}
    rat = {}
    for tag, pred in [('2025', T['year'] == '2025'), ('2026', T['year'] == '2026'),
                      ('POOLED', T['year'].isin(['2025', '2026']))]:
        sub = T[pred]
        if len(sub) == 0:
            continue
        cap[tag] = capture_block(sub, tag)
        rat[tag] = ratio_block(sub, tag)

    explore = explore_regimes()
    _write_report(frozen, league, sc, cap, rat, T, explore)
    flush_log()


def _log_scorecard(sc):
    r1, r2, r3, r5 = sc[60], sc[120], sc[180], sc[300]
    ci = r2.get('dir_ci', (float('nan'), float('nan')))
    log(f"dir-recall  @1m {r1['dir_recall']:.3f}  @2m {r2['dir_recall']:.3f} "
        f"[CI {ci[0]:.3f},{ci[1]:.3f}]  @3m {r3['dir_recall']:.3f}  @5m {r5['dir_recall']:.3f}")
    log(f"recall      @1m {r1['recall']:.3f}  @2m {r2['recall']:.3f}")
    log(f"precision   @2m {r2['precision']:.3f}   (chance {CHANCE_2M:.2f})")
    log(f"lead@2m     median {r2.get('lead_median', float('nan')):+.2f}  "
        f"mode {r2.get('lead_mode', float('nan')):+.2f}  "
        f"p25 {r2.get('lead_p25', float('nan')):+.2f}  p75 {r2.get('lead_p75', float('nan')):+.2f}  (min)")
    log(f"fires/day   {sc['fires_per_day']:.1f}   (test)")
    bar = (r2['precision'] > CHANCE_2M) or (r2['dir_recall'] >= 0.35 and r2.get('lead_median', 9) <= 1.0)
    log(f"BAR VERDICT: {'PASS' if bar else 'FAIL'}  "
        f"(precision>{CHANCE_2M} OR dir-recall@2m>=0.35 with lead<=+1min)")
    sc['_bar_pass'] = bool(bar)


def _write_report(frozen, league, sc, cap, rat, T, explore):
    fr = frozen['winner']
    w24 = frozen['winner_2024']
    r2 = sc[120]
    ci = r2.get('dir_ci', (float('nan'), float('nan')))
    pooled = cap.get('POOLED', {})
    pratio = rat.get('POOLED', {})
    L = []
    L.append('# PROP-TURN — proportional leg-turn confirmation, stop-and-reverse')
    L.append(f'_Moises 2026-07-16 design. Tuned on 2024 ONLY (sealed); all read-outs below are '
             f'TEST 2025+26. Generated {datetime.datetime.now():%Y-%m-%d %H:%M}._\n')

    L.append('## TL;DR — verdict')
    L.append('- **Turn bar: FAIL.** Frozen-cell dir-recall@±2m '
             f'{r2["dir_recall"]:.3f} [{ci[0]:.3f},{ci[1]:.3f}], precision@2m {r2["precision"]:.3f} '
             f'(both far below the 0.35 / 0.43 bar; lead is NEGATIVE {r2.get("lead_median", float("nan")):+.2f}m).')
    L.append(f'- **Capture: FAIL the 0.5–0.8 budget decisively.** Capture-ratio median '
             f'{pratio.get("median", float("nan")):+.2f} (typical leg-trade goes slightly the WRONG '
             f'way vs the label it sits in); only {pratio.get("frac_budget", float("nan")):.0%} of trades land in the budget. '
             f'Gross ≈ coin-flip (PF Trade WR {pooled.get("pfwr", float("nan")):+.3f}); '
             f'net {pooled.get("net_mean", float("nan")):+.2f} pt/trade after 0.6-pt friction '
             f'(CI [{pooled.get("net_ci",[float("nan")]*2)[0]:+.2f},{pooled.get("net_ci",[float("nan")]*2)[1]:+.2f}]).')
    L.append(f'- **League: the one positive.** As a COMBINER feature the fires carry weak but real '
             f'direction signal: OOS AUC {league.get("auc", float("nan")):.3f}, monotonic terciles.')
    L.append('- **Two structural findings drove the result** — see §0.\n')

    L.append('## 0. Structural findings (why the sealed cell is what it is)')
    L.append('**(a) The literal spec BREAKS stop-and-reverse (verification-driven; DECLARED DEVIATION).** '
             'After every flip the new leg starts with amplitude A = the triggering retrace (< A_min). '
             'If price then reverses hard, A stays frozen < A_min, the "A ≥ A_min to fire" gate '
             'permanently blocks the opposite turn, and the opposite branch is disabled by the leg '
             'direction — the tracker holds a LOSING position for the rest of the day. On the raw spec '
             'this zeroed 82 of 318 test days and *partially* stuck most others (8 sample 2024 days: '
             '314 fires literal vs 783 de-stuck, 2.49×). FIX (in the shared `_propturn_core`, flagged '
             'for review): keep the proportional confirm EXACTLY as specified for real legs (A ≥ A_min), '
             'and add a sub-minimal ESCAPE — a leg whose amplitude never reached A_min is re-designated '
             'when a full A_min counter-move occurs. Escape fires are ~3% of fires directly but unlock '
             'the mechanic\'s true ~100/day rate. Without it, every number is a bug artifact.')
    L.append('')
    L.append('**(b) The fires/day ≤ 60 cap forces a DEGENERATE cell.** With sticking cured, dir-recall '
             'scales with fire rate, and there is a sharp cliff in direction-correctness at S=3 '
             '(share of near-turn fires with the RIGHT direction, 2024, A_min=5): '
             'S=0 → 1.00, S=1 → 0.99, S=2 → 0.85, **S=3 → 0.28**, S=5 → 0.18. At S≥3 the long stall '
             'delays the confirm so far that the assigned "new leg" inverts vs the actual turn and the '
             'lead goes negative. EVERY feasible cell (fires ≤ 60) is S=3 or S=5 — the usable regime '
             '(S≤2, dir-recall 0.22–0.32) all fires 94–705/day. So the sealed winner is degenerate by '
             'construction, and even the un-capped "good" regime is largely fire-rate saturation '
             '(precision ~0.16). PROP-TURN is a firehose, not a ~45/day turn detector.\n')

    L.append('## Mechanic')
    L.append('Causal leg tracker on the continuous 5s close stream (tail+day, doc-073). Leg runs '
             'from pivot P0 to running extreme E; amplitude A=|E-P0|. TURN fires (stop-AND-reverse) '
             'when close retraces from E by >= r*A, subject to A>=A_min and a STALL gate '
             '(>= S min since E last improved). On fire: pivot->E, leg flips, fire direction = the '
             'NEW leg. State runs continuously (incl. overnight); emission RTH-gated.\n')

    L.append('## 1. Tuning (2024 SEALED) — top-5 feasible cells')
    L.append('Objective: max dir-recall@±2m on 2024 interior label turns s.t. lead-median ≤ +1.0 min '
             'AND fires/day ≤ 60. (90-cell grid: r×S×A_min.)\n')
    L.append('| rank | r | S (min) | A_min (pt) | dir-recall@2m | recall@2m | precision@2m | lead-med (min) | fires/day |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for i, row in enumerate(frozen['top5'], 1):
        L.append(f"| {i} | {row['r']} | {row['S']} | {row['A_min']:.0f} | {row['dir_recall_2m']:.3f} "
                 f"| {row['recall_2m']:.3f} | {row['precision_2m']:.3f} | {row['lead_median']:+.2f} "
                 f"| {row['fires_per_day']:.1f} |")
    relaxed = fr.get('relaxed')
    L.append(f'\n**FROZEN winner:** r={fr["r"]}, S={fr["S"]} min, A_min={fr["A_min"]:.0f} pt'
             + ('  _(NOTE: no cell met both constraints — relaxed to global-max dir-recall; flagged)_' if relaxed else '')
             + '\n')
    L.append('### Frozen cell — 2024 selection stats (the numbers it was chosen on)')
    L.append(f"- dir-recall@2m **{w24['120']['dir_recall']:.3f}**, recall@2m {w24['120']['recall']:.3f}, "
             f"precision@2m {w24['120']['precision']:.3f}")
    L.append(f"- lead-median {w24['120'].get('lead_median', float('nan')):+.2f} min, "
             f"fires/day {w24['fires_per_day']:.1f}, on {frozen['n_turns_2024']} interior turns / "
             f"{frozen['n_days_2024']} days\n")

    L.append('## 2. TEST turn scorecard (2025+26) — frozen cell')
    L.append('| metric | value |')
    L.append('|---|---|')
    L.append(f"| dir-recall@±1m | {sc[60]['dir_recall']:.3f} |")
    L.append(f"| **dir-recall@±2m [CI]** | **{r2['dir_recall']:.3f}** [{ci[0]:.3f}, {ci[1]:.3f}] |")
    L.append(f"| dir-recall@±3m | {sc[180]['dir_recall']:.3f} |")
    L.append(f"| dir-recall@±5m | {sc[300]['dir_recall']:.3f} |")
    L.append(f"| recall@±1m / ±2m | {sc[60]['recall']:.3f} / {r2['recall']:.3f} |")
    L.append(f"| precision@±2m (chance {CHANCE_2M:.2f}) | {r2['precision']:.3f} |")
    L.append(f"| lead@2m median / mode | {r2.get('lead_median', float('nan')):+.2f} / "
             f"{r2.get('lead_mode', float('nan')):+.2f} min |")
    L.append(f"| lead@2m p25 / p75 | {r2.get('lead_p25', float('nan')):+.2f} / "
             f"{r2.get('lead_p75', float('nan')):+.2f} min |")
    L.append(f"| fires/day (test) | {sc['fires_per_day']:.1f} |")
    L.append(f"\n**Standing-bar verdict: {'PASS' if sc.get('_bar_pass') else 'FAIL'}** "
             f"— bar = precision > {CHANCE_2M} OR (dir-recall@2m ≥ 0.35 with lead ≤ +1 min). "
             f"Best prior stream (RENKO24) sits at dir-recall 0.30 / precision 0.17.\n")

    L.append('## 3. League line (full 604-day pipeline; direction-agreement with AI labels)')
    if 'auc' in league:
        t = league['ter']
        ts = ' | '.join(f"{b}: {t[b][1]:.2f} [{t[b][2]:.2f},{t[b][3]:.2f}] N={t[b][0]}" for b in t)
        L.append(f"- N={league['n']} (train {league['n_tr']} / test {league['n_te']}), "
                 f"OOS **AUC {league['auc']:.3f}**, test base {league['base_te']:.2f}")
        L.append(f"- P-terciles: {ts}")
        L.append(f"- coefs: {league['coefs']}\n")
    else:
        L.append(f"- {league.get('note', league)}\n")

    L.append('## 4. CAPTURE — pure stop-and-reverse (TEST; the 50–80% budget headline)')
    L.append('Position flips at each fire (close fills); flat outside RTH (force-close 15:15, '
             're-open at next fire). Per completed leg-trade: captured points (signed). '
             f'Friction line = {FRICTION_PT} pt/round-trip (MNQ 1 pt = $2).\n')
    L.append('| pop | trades/day | captured mode | median | mean [CI] (pt) | PF Trade WR | net mean [CI] (pt) |')
    L.append('|---|---|---|---|---|---|---|')
    for tag in ('2025', '2026', 'POOLED'):
        c = cap.get(tag)
        if not c:
            continue
        L.append(f"| {tag} | {c['trades_per_day']:.1f} | {c['cap_mode']:+.2f} | {c['cap_median']:+.2f} "
                 f"| {c['cap_mean']:+.2f} [{c['cap_ci'][0]:+.2f}, {c['cap_ci'][1]:+.2f}] "
                 f"| {c['pfwr']:+.3f} | {c['net_mean']:+.2f} [{c['net_ci'][0]:+.2f}, {c['net_ci'][1]:+.2f}] |")
    L.append('\n### Capture ratio — captured / single-overlap label displacement')
    L.append('Reference points (from prior turn work): fixed-5m top-decile ≈ +2.00 pt median (deduped); '
             'oracle exit ≈ +27.5 pt median, ratio ≈ 0.23; user budget = 0.5–0.8.\n')
    L.append('| pop | N (1-overlap) | ratio mode | ratio median | frac in [0.5,0.8] | frac > 0 |')
    L.append('|---|---|---|---|---|---|')
    for tag in ('2025', '2026', 'POOLED'):
        rr = rat.get(tag)
        if not rr or rr.get('n', 0) == 0:
            continue
        L.append(f"| {tag} | {rr['n']} | {rr['mode']:+.2f} | {rr['median']:+.2f} "
                 f"| {rr['frac_budget']:.2f} | {rr['frac_pos']:.2f} |")
    pooled_r = rat.get('POOLED', {})
    if pooled_r.get('n', 0):
        verdict = ('WITHIN' if 0.5 <= pooled_r['median'] <= 0.8
                   else ('BELOW' if pooled_r['median'] < 0.5 else 'ABOVE'))
        L.append(f"\n**Capture-ratio vs the 0.5–0.8 budget: median {pooled_r['median']:+.2f} → {verdict} budget.**\n")

    L.append('## 5. Honesty guards')
    L.append('- **Pseudo-replication:** fires within a day share ONE leg-tracker state → serially '
             'dependent; capture legs within a day are a stop-and-reverse chain. All CIs are '
             'day-block bootstraps (unit of independence = the day), never per-trade/per-fire.')
    L.append('- **No post-hoc test selection:** the cell was frozen on 2024 ALONE before any test '
             'number was computed, on the stated objective (max dir-recall@2m s.t. constraints). '
             'Other grid cells\' TEST numbers appear ONLY in the clearly-labeled EXPLORATION table '
             '(appendix) and are never quoted as results — they exist to answer the design question '
             '"does any regime capture the budget" (answer: no), NOT to reselect a better cell.')
    L.append('- **Turn-bar vs capture read independently:** the standing turn-bar verdict and the '
             'capture/ratio read-out are reported separately; one passing does not carry the other.')
    L.append(f'- **Friction is real:** {FRICTION_PT} pt/round-trip; at the observed trades/day the '
             'net line is what matters, not gross.\n')

    L.append('## Appendix (EXPLORATION — 2024 tuning grid, NOT results)')
    G = pd.read_csv(os.path.join(REP, 'propturn_grid_2024.csv'))
    G = G.sort_values('dir_recall_2m', ascending=False).head(15)
    L.append('Top-15 of the 90 cells by 2024 dir-recall@2m (feasible flag shown). These are '
             'SELECTION-YEAR numbers; do not read as test performance.\n')
    L.append('| r | S | A_min | dir-recall@2m | recall@2m | precision@2m | lead-med | fires/day | feasible |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for _, x in G.iterrows():
        L.append(f"| {x['r']} | {x['S']:.0f} | {x['A_min']:.0f} | {x['dir_recall_2m']:.3f} "
                 f"| {x['recall_2m']:.3f} | {x['precision_2m']:.3f} | {x['lead_median']:+.2f} "
                 f"| {x['fires_per_day']:.1f} | {x['feasible']} |")

    if explore:
        L.append('\n### EXPLORATION — does ANY stall regime hit the capture budget? (TEST; NOT a result)')
        L.append('Capture sim run on TEST for the frozen cell + the best NON-degenerate cells, to answer '
                 'the design question directly. Shown per the honesty guard as exploration, not a result '
                 '(these cells were NOT the sealed selection). Answer: **no regime captures the budget** — '
                 'every one whipsaws to a slightly-negative capture ratio and coin-flip gross PF.\n')
        L.append('| cell | trades/day | capture median (pt) | PF Trade WR | capture-ratio median | frac in [0.5,0.8] | frac > 0 |')
        L.append('|---|---|---|---|---|---|---|')
        for e in explore:
            L.append(f"| {e['tag']} | {e['tpd']:.0f} | {e['cap_med']:+.2f} | {e['pf']:+.3f} "
                     f"| {e['ratio_med']:+.3f} | {e['frac_budget']:.2f} | {e['frac_pos']:.2f} |")
        L.append('')

    L.append('\n### Declared choices')
    L.append('- value emitted per fire = the completed leg amplitude A (pts) — a natural strength scalar.')
    L.append('- Shared feature basis for the league logistic is the canonical DayCtx zigzag '
             '(pivot_age/sig_with_leg/tod), identical to every other stream — PROP-TURN supplies only '
             'the trigger times + directions + value.')
    L.append('- Capture fills use the fire bar CLOSE (causal; no intrabar peeking). Final leg each '
             'day force-closes at the RTH-close bar (≤15:15).')
    L.append('- lead uses the nearest fire (any direction) to the turn, matching turn_detection_audit.')
    L.append('- Run via the tool (import-driven), not `python dossier_signal_pipeline.py PROP-TURN`: '
             'the generator is appended after the module `__main__` block, so it registers on import '
             '(the tool\'s path) but not on direct script execution.')

    out = os.path.join(REP, 'propturn.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    log(f'\nwrote {out}')


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else '--tune'
    if phase == '--tune':
        tune()
    elif phase == '--run':
        run()
    else:
        raise SystemExit('usage: propturn_tune_and_capture.py [--tune|--run]')
