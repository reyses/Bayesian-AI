"""
GOLDEN VECTOR GENERATOR  --  P0 of the NT8 native port parity harness (doc 129).

Runs the PYTHON REFERENCE decider over 20 reference days and emits per-1m-bar
"golden records" that every later C# component must reproduce bar-by-bar. The
reference decider = the pooled dossier combiner (entry P + top-decile threshold)
+ the live R-trigger zigzag (pivot state).

WHAT IS FROZEN AND WHY
----------------------
1. COMBINER FIT: reproduced EXACTLY from combiner_preview.py conventions
   (load_pool -> BASE + consensus + per-stream one-hots, standardized, 2024-train
   LogisticRegression(max_iter=2000)).  We extract cols / mu / sd / coef / intercept
   and the 2024-train P distribution.  The FROZEN top-decile threshold = the 90th
   percentile of the 2024-train pooled P (computed once, frozen).

2. TOP-K STREAMS: rank the per-stream one-hot |coef|; K = smallest set whose
   cumulative |coef| >= 80% of the STREAM coefficient mass (sum of |coef| over the
   is_<det> one-hots only).  The grand-total interpretation (BASE+consensus in the
   denominator) is DEGENERATE: sum(|stream coef|) < 0.80 * sum(|all coef|), so no
   stream subset can ever reach it -> the stream-mass denominator is the only
   well-defined reading.  Documented in the report.

3. R-TRIGGER PIVOTS: verbatim port of training/strategies/zigzag.py::ZigzagStrategy
   (the LIVE path: extreme +- R flip, min_bars_5s=36) over the continuous 5s close
   stream.  Per-day R (min_reversal_ticks) = round(ATR(14 1m)x4 / TICK), ATR taken
   CAUSALLY at the first RTH 5s bar (reuses DayCtx.zz_thr = ATR(14)x4 in points).
   This is the causal choice; the archived offline builder used whole-day median-TR
   (build_zigzag_pivot_dataset.compute_atr) -- flagged for P2 to reconcile.

REPRODUCIBILITY OF STREAM FIRES ON NEW DAYS
-------------------------------------------
Fires are regenerated on the reference days with the CANONICAL CAUSAL generators
(dossier_signal_pipeline.GENS, ~53 streams) + TMPL0 (template_stream_builder, frozen
2024 codebook, reuses DayCtx).  Two combiner streams are NOT primitive causal
generators and are EXCLUDED from the reproduced pool (both rank far below top-K, so
top-K is unaffected):
  * ADX08         (coef ~+0.010) -- separate ADX tool, not in the causal pipeline.
  * FOOTPRINT-IMB (not in the 55-snapshot) -- a META stream over econ_drift_rows
    (second-order on the combiner's own P); circular, not a port primitive.
Their combiner one-hots simply stay 0 on the reproduced fires (exactly as the
combiner sees a non-firing stream), so the frozen model applies unchanged.

CONSENSUS CAVEAT (documented deviation)
---------------------------------------
The combiner FIT computes `consensus` over the y-filtered signal_rows pool (fires
inside a labeled trade window only -- combiner_preview.load_pool).  At GENERATION we
have no labels, so consensus is computed over ALL reproduced fires of the day
(the live-valid definition).  The frozen mu/sd for consensus are applied unchanged.
Standardization is monotone so within-day P ordering is preserved; absolute P (hence
the top-decile crossing) may shift slightly.  P1's compact re-fit should define
consensus over its own K streams consistently (fit==deploy).  The pooled P here is
the FULL-combiner reference, not the P1 compact model.

DETERMINISM
-----------
No randomness anywhere: lbfgs logistic is deterministic given the (fixed) signal_rows
pool; consensus / features / generators / TMPL0 nearest-centroid assign are all
deterministic.  Re-running yields byte-identical golden parquets (verified with
--verify-determinism, which regenerates one day twice and compares sha256).

Usage:
  python3.11 research/nt8_port/tools/golden_vector_gen.py            # full 20-day run
  python3.11 research/nt8_port/tools/golden_vector_gen.py --verify-determinism
Outputs:
  research/nt8_port/golden/<day>.parquet          per-1m-bar golden records
  research/nt8_port/reports/golden_manifest.json  fit + top-K + threshold + day list
  research/nt8_port/reports/golden_sanity.md       per-day fire/entry/pivot counts
"""
import os
import sys
import json
import glob
import hashlib
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, '..'))          # research/nt8_port
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))    # repo root
CAT_TOOLS = os.path.join(ROOT, 'research', 'nt8_catalog', 'tools')
sys.path.insert(0, CAT_TOOLS)

import dossier_signal_pipeline as dsp                      # DayCtx, GENS, paths, TAIL
import combiner_preview as cp                              # load_pool, BASE, CONSENSUS_S
import template_stream_builder as tmpl                     # TMPL0 causal stream

GOLD = os.path.join(PROJ, 'golden')
REP = os.path.join(PROJ, 'reports')
os.makedirs(GOLD, exist_ok=True)
os.makedirs(REP, exist_ok=True)

TICK = 0.25
ZZ_MIN_BARS_5S = 36              # ZigzagStrategy.MIN_BARS_5S_DEFAULT (canonical live R-trigger)
TOPK_MASS = 0.80                 # cumulative |coef| target on the STREAM coefficient mass
TOPDECILE_PCTL = 90.0            # frozen entry threshold = 90th pct of 2024-train pooled P
N_2024 = 10                      # reference days from 2024
N_2526 = 10                      # reference days from 2025+2026
TMPL_JSON = os.path.join(dsp.REP, 'tmpl0_templates_2024.json')

# streams the combiner pools but which are NOT reproduced here (see module header)
EXTERNAL_STREAMS = {'ADX08', 'FOOTPRINTIMB'}


# --------------------------------------------------------------------------------------
# 1.  FROZEN COMBINER FIT  (combiner_preview conventions, verbatim)
# --------------------------------------------------------------------------------------
def fit_combiner():
    """Reproduce the combiner_preview fit and return the frozen model + top-K."""
    P = cp.load_pool()
    P = P.dropna(subset=['y']).copy()
    P['year'] = P['day'].str[:4]
    dets = sorted(P['det'].unique())
    for d in dets:
        P[f'is_{d}'] = (P['det'] == d).astype(int)
    cols = cp.BASE + ['consensus'] + [f'is_{d}' for d in dets]
    trm = P['year'] == '2024'
    Xtr = P.loc[trm, cols].values.astype(float)
    ytr = P.loc[trm, 'y'].astype(int).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, ytr)
    coef = clf.coef_[0]
    b0 = float(clf.intercept_[0])
    ptr = clf.predict_proba((Xtr - mu) / sd)[:, 1]
    thr = float(np.percentile(ptr, TOPDECILE_PCTL))

    # ---- top-K by stream coefficient mass ----
    stream_idx = [i for i, c in enumerate(cols) if c.startswith('is_')]
    stream_names = [cols[i][3:] for i in stream_idx]          # dashless det names
    stream_abs = np.array([abs(coef[i]) for i in stream_idx])
    order = np.argsort(-stream_abs)                            # descending |coef|
    total_stream_mass = float(stream_abs.sum())
    total_all_mass = float(np.abs(coef).sum())
    cum = 0.0
    topk = []
    for j in order:
        topk.append(stream_names[j])
        cum += stream_abs[j]
        if cum >= TOPK_MASS * total_stream_mass:
            break
    K = len(topk)
    ranking = [(stream_names[j], float(stream_abs[j])) for j in order]

    model = dict(cols=cols, dets=dets, mu=mu, sd=sd, coef=coef, b0=b0, thr=thr,
                 topk=topk, K=K, ranking=ranking,
                 total_stream_mass=total_stream_mass, total_all_mass=total_all_mass,
                 n_fires=int(len(P)), n_train=int(trm.sum()))
    return model


# --------------------------------------------------------------------------------------
# 2.  TMPL0 causal stream on a ref-day ctx (frozen 2024 codebook)
# --------------------------------------------------------------------------------------
_TMPL_CB = None


def _tmpl_codebook():
    global _TMPL_CB
    if _TMPL_CB is None:
        with open(TMPL_JSON, encoding='utf-8') as f:
            j = json.load(f)
        mean = np.array(j['scaler_mean'], float)
        scale = np.array(j['scaler_scale'], float)
        C_raw = np.array([t['centroid'] for t in j['templates']], float)
        Cs = (C_raw - mean) / scale                    # standardize centroids
        lf = np.array([np.nan if t['long_frac'] is None else t['long_frac']
                       for t in j['templates']], float)
        mc = np.array([t['member_count'] for t in j['templates']], float)
        _TMPL_CB = dict(mean=mean, scale=scale, Cs=Cs, lf=lf, mc=mc)
    return _TMPL_CB


def gen_tmpl0(ctx):
    """TMPL0 fires on one ctx: day_events -> standardize -> nearest frozen centroid ->
    template long_frac filter (verbatim template_stream_builder STREAM logic)."""
    cb = _tmpl_codebook()
    evs = tmpl.day_events(ctx)
    if not evs:
        return []
    E = pd.DataFrame(evs)
    X = E[['f0', 'f1', 'f2', 'f3', 'f4', 'f5']].values.astype(float)
    Xs = (X - cb['mean']) / cb['scale']
    tid, _, _ = tmpl.assign(Xs, cb['Cs'])
    lf = cb['lf'][tid]
    mc = cb['mc'][tid]
    conviction = np.abs(lf - 0.5)
    keep = (mc >= tmpl.MIN_MEMBERS_STREAM) & np.isfinite(lf) & (conviction >= tmpl.MIN_CONVICTION)
    is_long = lf > 0.5
    leg = E['leg'].values
    swl = np.where(leg != 0, ((leg > 0) == is_long).astype(int), 0)
    out = []
    for k in np.flatnonzero(keep):
        out.append(dict(ts=int(E['ts'].values[k]), is_long=bool(is_long[k]),
                        value=float(conviction[k]),
                        pivot_age_min=float(E['pivot_age_min'].values[k]),
                        sig_with_leg=int(swl[k]), tod=float(E['tod'].values[k]),
                        day=ctx.day))
    return out


# --------------------------------------------------------------------------------------
# 3.  R-trigger zigzag  (verbatim ZigzagStrategy port; live path, min_bars_5s=36)
# --------------------------------------------------------------------------------------
def zigzag_rtrigger(ctx):
    """Stream the ZigzagStrategy state machine over the full 5s close stream (ticks).
    Returns per-row (direction[-1/0/1], flip[-1/0/1], last_pivot_bar, last_pivot_price).
    R = round(zz_thr[first_rth]/TICK), min(4) floor; zz_thr = ATR(14 1m)x4 (causal)."""
    n = len(ctx.c)
    price_t = ctx.c / TICK                                    # closes in ticks
    rth_day = ctx.rth & (np.arange(n) >= ctx.start)
    rr = np.flatnonzero(rth_day)
    first_rth = int(rr[0]) if len(rr) else ctx.start
    thr_pts = ctx.zz_thr[first_rth]
    if not np.isfinite(thr_pts):
        # fall back to first finite zz_thr at/after first_rth
        fin = np.flatnonzero(np.isfinite(ctx.zz_thr[first_rth:]))
        thr_pts = ctx.zz_thr[first_rth + fin[0]] if len(fin) else 0.0
    min_rev = max(4, int(round(thr_pts / TICK)))              # ticks

    direction = np.zeros(n, dtype=np.int8)
    flip = np.zeros(n, dtype=np.int8)
    piv_bar = np.zeros(n, dtype=np.int64)
    piv_px = np.zeros(n, dtype=np.float64)

    d = 0
    ext = price_t[0]                                          # extreme in ticks
    ext_bar = 0
    first_close = price_t[0]
    last_piv_bar = 0
    last_piv_px = price_t[0]
    for i in range(1, n):
        p = price_t[i]
        f = 0
        if d == 0:
            if p > ext:
                ext, ext_bar = p, i
            if p < first_close and (first_close - p) >= min_rev:
                d, ext, ext_bar, f = -1, p, i, -1
            elif p > first_close and (p - first_close) >= min_rev:
                d, ext, ext_bar, f = 1, p, i, 1
            if f != 0:
                last_piv_bar, last_piv_px = i, first_close     # seed pivot = first close
        elif d == 1:
            if p >= ext:
                ext, ext_bar = p, i
            elif (ext - p) >= min_rev and (i - ext_bar) >= ZZ_MIN_BARS_5S:
                last_piv_bar, last_piv_px = ext_bar, ext       # confirmed swing HIGH
                d, ext, ext_bar, f = -1, p, i, -1
        else:  # d == -1
            if p <= ext:
                ext, ext_bar = p, i
            elif (p - ext) >= min_rev and (i - ext_bar) >= ZZ_MIN_BARS_5S:
                last_piv_bar, last_piv_px = ext_bar, ext       # confirmed swing LOW
                d, ext, ext_bar, f = 1, p, i, 1
        direction[i] = d
        flip[i] = f
        piv_bar[i] = last_piv_bar
        piv_px[i] = last_piv_px * TICK                         # back to points
    return dict(direction=direction, flip=flip, piv_bar=piv_bar, piv_px=piv_px,
                min_rev_ticks=min_rev)


# --------------------------------------------------------------------------------------
# 4.  consensus (combiner_preview convention, per-day pool)
# --------------------------------------------------------------------------------------
def day_consensus(P):
    """Same-direction OTHER-stream fires within +-CONSENSUS_S. Vectorized, identical to
    combiner_preview.load_pool (days are >>window apart, so per-day == pooled)."""
    P = P.sort_values('ts').reset_index(drop=True)
    ts = P['ts'].values.astype(np.int64)
    lng = P['is_long'].values.astype(bool)
    lo = np.searchsorted(ts, ts - cp.CONSENSUS_S, 'left')
    hi = np.searchsorted(ts, ts + cp.CONSENSUS_S, 'right')

    def wcount(flags):
        c = np.concatenate([[0], np.cumsum(flags)])
        return c[hi] - c[lo]

    same_dir = np.where(lng, wcount(lng), wcount(~lng))
    own = np.zeros(len(P), dtype=np.int64)
    for (d, is_l), g in P.groupby(['det', 'is_long'], sort=False):
        flags = np.zeros(len(P), dtype=np.int64)
        flags[g.index.values] = 1
        own[g.index.values] = wcount(flags)[g.index.values]
    P['consensus'] = (same_dir - own).astype(np.int64)
    return P


# --------------------------------------------------------------------------------------
# 5.  per-day driver
# --------------------------------------------------------------------------------------
def build_ctx(files, j):
    """Warm context (20 prior files -> prior_daily + tail), then build the ref-day ctx.
    Mirrors dossier_signal_pipeline.run_all's per-day context exactly."""
    prior_daily = []
    tail = None
    for pf in files[max(0, j - 20):j]:
        df = pd.read_parquet(pf, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        m = ((dt.dt.time >= dsp.RTH0) & (dt.dt.time <= dsp.RTH1)).values
        if m.any():
            entry = dict(high=float(df['high'].values[m].max()),
                         low=float(df['low'].values[m].min()),
                         close=float(df['close'].values[m][-1]))
            entry.update(dsp._day_profile(df['close'].values[m], df['volume'].values[m]))
            prior_daily.append(entry)
            prior_daily = prior_daily[-20:]
        tail = df.tail(dsp.TAIL)

    p = files[j]
    day = os.path.basename(p)[:10]
    df = pd.read_parquet(p, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    full = pd.concat([tail, df], ignore_index=True) if tail is not None else df
    start = len(tail) if tail is not None else 0
    ctx = dsp.DayCtx(full, start, day, prior_daily)
    # z_se for NMP / NMP9 head streams (run_all convention)
    zp = os.path.join(ROOT, 'DATA', 'ATLAS', 'FEATURES_1s_v2', 'L3_1m', f'{day}.parquet')
    ctx.zse = None
    if os.path.exists(zp):
        zf = pd.read_parquet(zp, columns=['timestamp', 'L3_1m_z_se_15'])
        ctx.zse = pd.Series(full['timestamp']).map(
            dict(zip(zf['timestamp'].values, zf['L3_1m_z_se_15'].values))).values
    return ctx, day


def all_fires(ctx):
    """Run every reproduced stream; return one pooled DataFrame of fires with `det`."""
    frames = []
    for key, gen in dsp.GENS.items():
        det = key.replace('-', '')
        if det in EXTERNAL_STREAMS:
            continue
        fires = gen(ctx)
        if fires:
            g = pd.DataFrame(fires)
            g['det'] = det
            frames.append(g)
    t0 = gen_tmpl0(ctx)
    if t0:
        g = pd.DataFrame(t0)
        g['det'] = 'TMPL0'
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=['ts', 'is_long', 'value', 'pivot_age_min',
                                     'sig_with_leg', 'tod', 'day', 'det'])
    return pd.concat(frames, ignore_index=True)


def score_fires(F, model):
    """Attach consensus + pooled combiner P + entry flag to each fire row."""
    if len(F) == 0:
        F = F.copy()
        F['consensus'] = np.array([], dtype=np.int64)
        F['P'] = np.array([], dtype=float)
        return F
    F = day_consensus(F)
    F['inter'] = F['sig_with_leg'].values * F['pivot_age_min'].values
    cols = model['cols']
    n = len(F)
    X = np.zeros((n, len(cols)), dtype=float)
    base_map = {'pivot_age_min': F['pivot_age_min'].values,
                'sig_with_leg': F['sig_with_leg'].values,
                'tod': F['tod'].values, 'inter': F['inter'].values,
                'consensus': F['consensus'].values}
    det = F['det'].values
    for ci, c in enumerate(cols):
        if c in base_map:
            X[:, ci] = base_map[c]
        elif c.startswith('is_'):
            X[:, ci] = (det == c[3:]).astype(float)
    Z = (X - model['mu']) / model['sd']
    logit = Z @ model['coef'] + model['b0']
    F['P'] = 1.0 / (1.0 + np.exp(-logit))
    return F


def aggregate_day(ctx, day, F, zz, model):
    """Collapse fires + zigzag state to per-1m-bar golden records over RTH."""
    topk = model['topk']
    thr = model['thr']
    n = len(ctx.c)
    ar = np.arange(n)
    rth_day = ctx.rth & (ar >= ctx.start)
    # RTH 1m bars of the day (minute-open epochs), ordered
    minute = (ctx.ts // 60) * 60
    day_min = minute[rth_day]
    bars = np.unique(day_min)
    # map fires to their minute
    if len(F):
        f_min = (F['ts'].values // 60) * 60
        f_dir = np.where(F['is_long'].values, 1, -1)
        f_det = F['det'].values
        f_P = F['P'].values
        f_topk = np.isin(f_det, topk)
    else:
        f_min = np.array([], dtype=np.int64)

    # last 5s row index within each minute (for zigzag state @ bar close)
    # rows in the day, grouped by minute -> take max index per minute
    day_rows = ar[rth_day]
    row_min = minute[rth_day]
    last_row = pd.Series(day_rows, index=row_min).groupby(level=0).max()

    recs = []
    for T in bars:
        rec = {'bar_ts': int(T), 'date': day}
        # fire states
        states = {d: 0 for d in topk}
        n_topk = 0
        gov_stream, gov_dir, P_topk, P_any = '', 0, np.nan, np.nan
        if len(f_min):
            inb = f_min == T
            if inb.any():
                dets_b = f_det[inb]
                dir_b = f_dir[inb]
                P_b = f_P[inb]
                topk_b = f_topk[inb]
                P_any = float(np.nanmax(P_b))
                for dd, di in zip(dets_b, dir_b):
                    if dd in states:
                        states[dd] = int(di)          # last fire's direction wins ties
                if topk_b.any():
                    n_topk = int(topk_b.sum())
                    P_tk = P_b[topk_b]
                    kk = int(np.nanargmax(P_tk))
                    P_topk = float(P_tk[kk])
                    gov_stream = str(dets_b[topk_b][kk])
                    gov_dir = int(dir_b[topk_b][kk])
        for d in topk:
            rec[f'f_{d}'] = states[d]
        rec['n_fires_topk'] = n_topk
        rec['gov_stream'] = gov_stream
        rec['gov_dir'] = gov_dir
        rec['P_topk'] = P_topk
        rec['P_any'] = P_any
        entry = int(np.isfinite(P_topk) and P_topk >= thr)
        rec['entry'] = entry
        rec['entry_dir'] = gov_dir if entry else 0
        # zigzag R-trigger state @ bar close (last 5s row of the minute)
        r = int(last_row.loc[T])
        rec['zz_leg'] = int(zz['direction'][r])
        # flip within this minute?
        rows_in = day_rows[row_min == T]
        fl = zz['flip'][rows_in]
        nz = fl[fl != 0]
        rec['zz_confirm'] = int(nz[-1]) if len(nz) else 0
        pb = int(zz['piv_bar'][r])
        rec['zz_pivot_age_min'] = float((r - pb) * 5 / 60.0)
        rec['zz_pivot_price'] = float(zz['piv_px'][r])
        recs.append(rec)
    cols_order = (['bar_ts', 'date'] + [f'f_{d}' for d in topk] +
                  ['n_fires_topk', 'gov_stream', 'gov_dir', 'P_topk', 'P_any',
                   'entry', 'entry_dir', 'zz_leg', 'zz_confirm',
                   'zz_pivot_age_min', 'zz_pivot_price'])
    return pd.DataFrame(recs, columns=cols_order)


def select_days(files):
    """10 evenly-spaced 2024 + 10 evenly-spaced 2025/26 real trading days.
    Candidates are restricted to LABEL days (ai_cusp_picks) so weekends / holidays /
    partial end-of-dump files (no RTH session) are excluded -- selection is purely a
    'real RTH session' filter; labels are NOT used to build the golden vectors."""
    lblf = {os.path.basename(f)[9:19]
            for f in glob.glob(os.path.join(dsp.LBL, 'ai_picks_*_multi.json'))}
    names = [os.path.basename(f)[:10] for f in files]

    def is_label(i):
        return names[i].replace('_', '-') in lblf

    idx24 = [i for i, nm in enumerate(names)
             if nm[:4] == '2024' and i >= 20 and is_label(i)]
    idx2526 = [i for i, nm in enumerate(names)
               if nm[:4] in ('2025', '2026') and is_label(i)]

    def has_rth(i):
        """Cheap check: does this 5s file contain a real RTH session? (drops truncated
        end-of-dump files that carry a label but no 08:30-15:15 CT rows)."""
        ts = pd.read_parquet(files[i], columns=['timestamp'])['timestamp'].values
        dt = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/Chicago')
        tt = dt.time
        return bool(((tt >= dsp.RTH0) & (tt <= dsp.RTH1)).any())

    def even(pool, k):
        if len(pool) <= k:
            picks = list(pool)
        else:
            pos = np.linspace(0, len(pool) - 1, k).round().astype(int)
            picks = [pool[p] for p in sorted(set(pos.tolist()))]
        # step each pick back to the nearest earlier real-RTH file not already taken
        chosen = []
        for i in picks:
            while i in chosen or not has_rth(i):
                i -= 1
                if i < pool[0]:
                    break
            if i >= pool[0] and i not in chosen:
                chosen.append(i)
        return chosen

    return even(idx24, N_2024) + even(idx2526, N_2526)


# --------------------------------------------------------------------------------------
# 6.  main
# --------------------------------------------------------------------------------------
def process_day(files, j, model):
    ctx, day = build_ctx(files, j)
    F = all_fires(ctx)
    F = score_fires(F, model)
    zz = zigzag_rtrigger(ctx)
    G = aggregate_day(ctx, day, F, zz, model)
    return day, F, zz, G


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-determinism', action='store_true',
                    help='regenerate one day twice and compare sha256')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(dsp.D5, '*.parquet')))
    print(f'{len(files)} 5s day files found')
    model = fit_combiner()
    print(f"combiner: {model['n_fires']} fires, {len(model['dets'])} streams; "
          f"K={model['K']} top-K reaches "
          f"{TOPK_MASS*100:.0f}% of stream mass ({model['total_stream_mass']:.3f})")
    print(f"  top-K: {', '.join(model['topk'])}")
    print(f"  frozen top-decile threshold P>= {model['thr']:.4f}")

    day_idx = select_days(files)
    days = [os.path.basename(files[j])[:10] for j in day_idx]
    print(f'reference days ({len(days)}): {days}')

    if args.verify_determinism:
        j = day_idx[0]
        _, _, _, G1 = process_day(files, j, model)
        _, _, _, G2 = process_day(files, j, model)
        p1 = os.path.join(REP, '_det_a.parquet')
        p2 = os.path.join(REP, '_det_b.parquet')
        G1.to_parquet(p1)
        G2.to_parquet(p2)
        h1, h2 = _sha256(p1), _sha256(p2)
        os.remove(p1)
        os.remove(p2)
        print(f'DETERMINISM {days[0]}: sha256 A={h1[:16]} B={h2[:16]} '
              f'-> {"IDENTICAL" if h1 == h2 else "DIFFER"}')
        return

    sanity = []
    for j in day_idx:
        day, F, zz, G = process_day(files, j, model)
        outp = os.path.join(GOLD, f'{day}.parquet')
        G.to_parquet(outp)
        n_fires = int(len(F))
        n_topk_fires = int(F['det'].isin(model['topk']).sum()) if len(F) else 0
        n_entries = int(G['entry'].sum())
        n_pivots = int((G['zz_confirm'] != 0).sum())
        n_bars = int(len(G))
        sanity.append(dict(day=day, bars=n_bars, fires=n_fires,
                           topk_fires=n_topk_fires, entries=n_entries,
                           pivots=n_pivots, min_rev_ticks=int(zz['min_rev_ticks']),
                           sha=_sha256(outp)[:16]))
        flag = '  <== ZERO ENTRIES' if n_entries == 0 else ''
        print(f'{day}: bars={n_bars} fires={n_fires} topk={n_topk_fires} '
              f'entries={n_entries} pivots={n_pivots} R={zz["min_rev_ticks"]}t{flag}')

    # manifest
    manifest = dict(
        generated_by='research/nt8_port/tools/golden_vector_gen.py',
        combiner_n_fires=model['n_fires'], combiner_n_train=model['n_train'],
        n_streams=len(model['dets']), K=model['K'], topk=model['topk'],
        topk_mass_target=TOPK_MASS,
        total_stream_mass=model['total_stream_mass'],
        total_all_mass=model['total_all_mass'],
        note_grand_total=('grand-total interpretation is degenerate: '
                          f"stream_mass {model['total_stream_mass']:.3f} < "
                          f"0.80*all_mass {0.8*model['total_all_mass']:.3f}"),
        stream_ranking=[[nm, round(m, 4)] for nm, m in model['ranking']],
        excluded_external=sorted(EXTERNAL_STREAMS),
        frozen_top_decile_threshold=model['thr'],
        top_decile_pctl=TOPDECILE_PCTL,
        zz_min_bars_5s=ZZ_MIN_BARS_5S,
        reference_days=days, day_sanity=sanity)
    with open(os.path.join(REP, 'golden_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1)

    # sanity md
    L = ['# Golden vectors -- sanity counts', '',
         f'- {len(days)} reference days, K={model["K"]} top-K streams, '
         f'frozen entry threshold P>= {model["thr"]:.4f}', '',
         '| day | bars | fires | top-K fires | entries | pivots | R(ticks) | sha256 |',
         '|---|---|---|---|---|---|---|---|']
    for s in sanity:
        L.append(f"| {s['day']} | {s['bars']} | {s['fires']} | {s['topk_fires']} | "
                 f"{s['entries']} | {s['pivots']} | {s['min_rev_ticks']} | {s['sha']} |")
    zero = [s['day'] for s in sanity if s['entries'] == 0]
    L.append('')
    L.append(f'- zero-entry days: {zero if zero else "none"}')
    with open(os.path.join(REP, 'golden_sanity.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'\nwrote {len(days)} golden parquets + manifest + sanity to {REP}')


if __name__ == '__main__':
    main()
