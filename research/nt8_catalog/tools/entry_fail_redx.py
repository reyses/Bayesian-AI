"""
TASK 122 -- the entry-fail RED X: what separates terminal-good from terminal-bad AT ENTRY?
(research/nt8_catalog/tools/entry_fail_redx.py -- Opus drone, sealed Shainin contrast)

Moises' thesis: "the best way not to bite our nails on ride-or-eject is not getting into
fails." ~half of top-decile entries end <= -4pts because entry P was trained on
DIRECTION-AGREEMENT, not terminal economics. This asks: do ENTRY-TIME features (leg
geometry, lambda-hat, NMP9 tier, tod, trail-vol, det) separate terminal-good from
terminal-bad ABOVE the P-only baseline? P-only is the pre-registered bar -- beating base
alone just re-discovers P.

SEALED PROTOCOL (nothing tuned on test):
  TRAIN = 2024 engagements (split=='train'); TEST = 2025-26 (split=='test').
  Engagement population = select_wrongdir.engagements() machinery: P >= p90(train) frozen,
  60s/day/dir de-dup, MIN_WINDOW>=15m. Terminal drift via swl.scan (eb.signed_drift_path).
  Terminal labels at BAND=4: GOOD terminal>=+4, BAD terminal<=-4, DEAD |terminal|<4
  (DEAD excluded from FIT; INCLUDED in volume accounting).
  Logistic fit on 2024 good-vs-bad; continuous standardized on train; coefs + the three
  operating-point thresholds (retain 70/50/30% of 2024 volume) FROZEN; single shot on test.
  Day-block bootstrap CIs (4000 resamples over distinct days).

ENTRY-TIME FEATURES (all causal at fire ts; joins documented in the report):
  1. entry P                         -- econ_drift_rows (also THE P-only baseline).
  2. det one-hot                     -- which stream fired (econ det).
  3. pivot_age_min, sig_with_leg     -- the fire's own signal_rows_<det> row (join ts).
  4. lambda_hat at ts                -- dsp._nmp_lambda (z_se store, NMP_K=21, NMP_EPS=0.1).
  5. NMP9 tier at ts                 -- dsp._nmp9_events waterfall, as-of last tier ('none').
  6. tod (session-time)              -- signal_rows tod.
  7. trail_vol (ticks)               -- std of last 60 5s closes / TICK (pipeline tvol def).

Reuses swl.engagements/scan, dsp._nmp_lambda/_nmp9_events/_tf_state, eb helpers by import;
edits none of them. New files ONLY: this script + reports/entry_fail_redx.md. Commit NOTHING.

Run: python3.11 research/nt8_catalog/tools/entry_fail_redx.py
"""
import os
import sys
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
EXIT_TOOLS = os.path.join(ROOT, 'research', 'exit_dojo', 'tools')
EXIT_BUILDERS = os.path.join(ROOT, 'research', 'exit_dojo', 'builders')
sys.path.insert(0, HERE)
sys.path.insert(0, EXIT_TOOLS)
sys.path.insert(0, EXIT_BUILDERS)

import select_wrongdir as swl            # engagements() + scan() population machinery
import episode_builder as eb             # signed_drift_path, load_day_data, asof_idx
import telescope_packet_builder as tb    # P_PCTL / DEDUP_S / MIN_WINDOW_MIN / WINDOW_CAP
import dossier_signal_pipeline as dsp     # _nmp_lambda / _nmp9_events / _tf_state / RTH / TICK

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---- constants (house rule: no bare magic numbers) -----------------------------------
BAND = 4.0                    # terminal-good >= +BAND, terminal-bad <= -BAND (pts)
DIP_PTS = 4.0                 # a GOOD "dipped" if min drift <= -DIP before winning
NMP_K = dsp.NMP_K             # 21 -- lambda-hat OLS window (verified derivation)
NMP_EPS = dsp.NMP_EPS         # 0.1 -- log floor
TICK = dsp.TICK               # 0.25 -- MNQ tick (pts->ticks conversion)
TVOL_WIN = 60                 # 60 x 5s = 5-min trailing vol (pipeline tvol, dsp line 1538)
TAIL = dsp.TAIL               # 2500 -- prior-day tail prepend (ctx continuity)
BOOTS = 4000                  # day-block bootstrap resamples
SEED = 20260718
RETAIN_LEVELS = [0.70, 0.50, 0.30]     # PRE-REGISTERED operating points (train volume kept)

NT8_REPORTS = os.path.join(ROOT, 'research', 'nt8_catalog', 'reports')
D5 = dsp.D5
ZDIR = os.path.join(ROOT, 'DATA', 'ATLAS', 'FEATURES_1s_v2', 'L3_1m')
OUT_MD = os.path.join(NT8_REPORTS, 'entry_fail_redx.md')

CONT_FEATURES = ['P', 'pivot_age_min', 'tod', 'trail_vol_ticks', 'lambda_hat']
BIN_FEATURES = ['sig_with_leg']
NMP9_LEVELS = ['none'] + list(dsp.NMP9_TIERS)


# ================= population ==========================================================
def build_engagements(econ, thr, split, years):
    """swl.engagements() logic, parameterized for split/years with the frozen thr."""
    sub = econ[(econ.split == split) & (econ.P >= thr) &
               (econ.day.str[:4].isin(years))].copy()
    sub = sub.sort_values(['day', 'is_long', 'ts', 'det']).reset_index(drop=True)
    last, keep = {}, []
    for r in sub.itertuples():
        k = (r.day, bool(r.is_long))
        if k in last and r.ts - last[k] <= tb.DEDUP_S:
            continue
        last[k] = r.ts
        keep.append(r.Index)
    return sub.loc[keep].reset_index(drop=True)


def scan_to_rows(eng):
    """swl.scan -> flat per-engagement rows with terminal/mindrift/window + keys."""
    day_engs, _ = swl.scan(eng)
    rows = []
    for day, engs in day_engs.items():
        for e in engs:
            rows.append(dict(day=day, ts=int(e['ts']), is_long=bool(e['is_long']),
                             P=float(e['P']), det=e['det'],
                             terminal=float(e['terminal']), mindrift=float(e['mindrift']),
                             window=int(e['window_minutes'])))
    return pd.DataFrame(rows)


# ================= leg-geometry join (signal_rows_<det>, ts) ==========================
def join_leg_geometry(df):
    """Join pivot_age_min / sig_with_leg / tod from each fire's own signal_rows_<det>.
    det name maps to filename verbatim (already dashless). Reports coverage."""
    df = df.copy()
    for c in ['pivot_age_min', 'sig_with_leg', 'tod']:
        df[c] = np.nan
    for det, idx in df.groupby('det').groups.items():
        path = os.path.join(NT8_REPORTS, f'signal_rows_{det.replace("-", "")}.parquet')
        if not os.path.exists(path):
            continue
        sr = pd.read_parquet(path, columns=['ts', 'pivot_age_min', 'sig_with_leg', 'tod'])
        sr = sr.drop_duplicates(subset=['ts'], keep='first').set_index('ts')
        sub = df.loc[idx]
        for c in ['pivot_age_min', 'sig_with_leg', 'tod']:
            df.loc[idx, c] = sub['ts'].map(sr[c])
    return df


# ================= ctx-derived features (lambda_hat, nmp9 tier, trail_vol) =============
class LiteCtx:
    """Minimal DayCtx duck-type carrying only what _nmp_lambda / _nmp9_events / _tf_state
    read (ts,c,o,h,l,v,rth,start,zse). Skips DayCtx's streaming-zigzag loop (unused here)."""
    def __init__(self, full, start, day, zse):
        self.start, self.day, self.zse = start, day, zse
        self.ts = full['timestamp'].values.astype(np.int64)
        self.c = full['close'].values
        self.h = full['high'].values
        self.l = full['low'].values
        self.o = full['open'].values
        self.v = full['volume'].values.astype(float)
        dt = pd.to_datetime(full['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        tt = dt.dt.time
        self.rth = ((tt >= dsp.RTH0) & (tt <= dsp.RTH1)).values


def ctx_features(need_days, fires_by_day):
    """Stream ALL 5s days in order (maintain prior-day tail for ctx continuity like
    run_all); for each engagement-day compute lambda_hat (per-5s ffilled), NMP9 tier
    (as-of last waterfall emission), trail_vol (ticks). Returns {(day,ts): dict}."""
    files = sorted(glob.glob(os.path.join(D5, '*.parquet')))
    out = {}
    cov = dict(lambda_ok=0, lambda_nan=0, zse_missing_days=0, days_done=0)
    tail = None
    for p in tqdm(files, desc='ctx-days'):
        day = os.path.basename(p)[:10]
        df = pd.read_parquet(p, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        if day in need_days:
            full = pd.concat([tail, df], ignore_index=True) if tail is not None else df
            start = len(tail) if tail is not None else 0
            zp = os.path.join(ZDIR, f'{day}.parquet')
            zse = None
            if os.path.exists(zp):
                zf = pd.read_parquet(zp, columns=['timestamp', 'L3_1m_z_se_15'])
                zse = pd.Series(full['timestamp']).map(
                    dict(zip(zf['timestamp'].values, zf['L3_1m_z_se_15'].values))).values
            else:
                cov['zse_missing_days'] += 1
            ctx = LiteCtx(full, start, day, zse)
            lam = dsp._nmp_lambda(ctx) if zse is not None else np.full(len(ctx.c), np.nan)
            tvol = (pd.Series(ctx.c).rolling(TVOL_WIN, min_periods=TVOL_WIN).std(ddof=1)
                    / TICK).values
            ev = dsp._nmp9_events(ctx) if zse is not None else []
            ev_ts = np.array([int(ctx.ts[i]) for i, _, _, _ in ev], dtype=np.int64)
            ev_tier = [t for _, _, t, _ in ev]
            for ts in fires_by_day[day]:
                j = int(np.searchsorted(ctx.ts, ts, side='right') - 1)
                j = max(0, min(j, len(ctx.ts) - 1))
                lv = float(lam[j])
                if np.isfinite(lv):
                    cov['lambda_ok'] += 1
                else:
                    cov['lambda_nan'] += 1
                if len(ev_ts):
                    k = int(np.searchsorted(ev_ts, ts, side='right') - 1)
                    tier = ev_tier[k] if k >= 0 else 'none'
                else:
                    tier = 'none'
                out[(day, ts)] = dict(lambda_hat=lv, trail_vol_ticks=float(tvol[j]),
                                      nmp9_tier=tier)
            cov['days_done'] += 1
        tail = df.tail(TAIL)
    return out, cov


# ================= feature-matrix assembly ============================================
def assemble(df, det_cols, fill):
    """Build the model design matrix. Continuous standardized by `fill['mu']/fill['sd']`
    (train stats); NaN continuous filled with train median + a missing indicator for
    lambda_hat; det + nmp9_tier one-hot aligned to train columns."""
    n = len(df)
    parts, names = [], []
    # continuous (standardized on train stats; median-filled)
    for c in CONT_FEATURES:
        x = df[c].values.astype(float)
        miss = ~np.isfinite(x)
        x = np.where(miss, fill['median'][c], x)
        z = (x - fill['mu'][c]) / (fill['sd'][c] if fill['sd'][c] > 0 else 1.0)
        parts.append(z.reshape(-1, 1)); names.append(c)
        if c == 'lambda_hat':
            parts.append(miss.astype(float).reshape(-1, 1)); names.append('lambda_hat_missing')
    # binary
    for c in BIN_FEATURES:
        x = df[c].values.astype(float)
        x = np.where(np.isfinite(x), x, 0.0)
        parts.append(x.reshape(-1, 1)); names.append(c)
    # nmp9 tier one-hot (drop 'none' as baseline)
    tier = df['nmp9_tier'].fillna('none').values
    for lv in NMP9_LEVELS[1:]:
        parts.append((tier == lv).astype(float).reshape(-1, 1)); names.append(f'nmp9_{lv}')
    # det one-hot (aligned to train det_cols; drop first as baseline)
    det = df['det'].values
    for d in det_cols[1:]:
        parts.append((det == d).astype(float).reshape(-1, 1)); names.append(f'det_{d}')
    return np.hstack(parts), names


def train_fill(df):
    mu, sd, med = {}, {}, {}
    for c in CONT_FEATURES:
        x = df[c].values.astype(float)
        x = x[np.isfinite(x)]
        med[c] = float(np.median(x)) if len(x) else 0.0
    # standardize using median-filled full column (so test uses identical transform)
    for c in CONT_FEATURES:
        x = df[c].values.astype(float)
        x = np.where(np.isfinite(x), x, med[c])
        mu[c] = float(x.mean()); sd[c] = float(x.std())
    return dict(mu=mu, sd=sd, median=med)


# ================= day-block bootstrap ================================================
def _perday(days, arrs):
    """Aggregate per-day sums for each array in arrs (list of 1d). Returns (uq, mats)
    where mats[k] has per-day sums."""
    uq, inv = np.unique(days, return_inverse=True)
    mats = []
    for a in arrs:
        s = np.zeros(len(uq)); np.add.at(s, inv, a); mats.append(s)
    return uq, inv, mats


def rate_delta_ci(days, good, keep_a, keep_b, boots=BOOTS, seed=SEED):
    """delta = rate_a - rate_b where rate = sum(good&keep)/sum(keep). Day-block bootstrap.
    Returns (point_delta, lo, hi, rate_a, rate_b)."""
    na = good * keep_a; da = keep_a.astype(float)
    nb = good * keep_b; db = keep_b.astype(float)
    uq, inv, (Sna, Sda, Snb, Sdb) = _perday(days, [na, da, nb, db])
    ra = Sna.sum() / max(Sda.sum(), 1e-9)
    rb = Snb.sum() / max(Sdb.sum(), 1e-9)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uq), size=(boots, len(uq)))
    da_s = Sda[idx].sum(1); db_s = Sdb[idx].sum(1)
    va = Sna[idx].sum(1) / np.maximum(da_s, 1e-9)
    vb = Snb[idx].sum(1) / np.maximum(db_s, 1e-9)
    d = va - vb
    return float(ra - rb), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), \
        float(ra), float(rb)


def mean_diff_ci(days, feat, is_good, is_bad, boots=BOOTS, seed=SEED):
    """mean(feat|good) - mean(feat|bad), day-block bootstrap over the good+bad rows."""
    m = np.isfinite(feat) & (is_good | is_bad)
    days, feat, g, b = days[m], feat[m], is_good[m].astype(float), is_bad[m].astype(float)
    fg = feat * g; fb = feat * b
    uq, inv, (Sfg, Sg, Sfb, Sb) = _perday(days, [fg, g, fb, b])
    pt = Sfg.sum() / max(Sg.sum(), 1e-9) - Sfb.sum() / max(Sb.sum(), 1e-9)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uq), size=(boots, len(uq)))
    mg = Sfg[idx].sum(1) / np.maximum(Sg[idx].sum(1), 1e-9)
    mb = Sfb[idx].sum(1) / np.maximum(Sb[idx].sum(1), 1e-9)
    d = mg - mb
    return float(pt), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def hist_mode(x, bw):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float('nan')
    lo, hi = np.floor(x.min() / bw) * bw, np.ceil(x.max() / bw) * bw + bw
    edges = np.arange(lo, hi + bw, bw)
    h, e = np.histogram(x, bins=edges)
    k = int(np.argmax(h))
    return float((e[k] + e[k + 1]) / 2)


def fmt_ci(m, lo, hi):
    sig = '' if (lo <= 0 <= hi) else ' *'
    return f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}'


# ================= main ===============================================================
def main():
    print('[load] econ_drift_rows ...')
    econ = pd.read_parquet(os.path.join(NT8_REPORTS, 'econ_drift_rows.parquet'),
                           columns=['ts', 'day', 'det', 'is_long', 'P', 'split'])
    thr = float(np.percentile(econ.loc[econ.split == 'train', 'P'].values, tb.P_PCTL))
    print(f'[cfg] p{tb.P_PCTL}(train) thr={thr:.5f}; BAND={BAND:.0f}; NMP_K={NMP_K}')

    eng_tr = build_engagements(econ, thr, 'train', ['2024'])
    eng_te = build_engagements(econ, thr, 'test', ['2025', '2026'])
    print(f'[pop] pre-scan  train={len(eng_tr)} ({eng_tr.day.nunique()}d)  '
          f'test={len(eng_te)} ({eng_te.day.nunique()}d)')

    print('[scan] train drift/terminal ...')
    tr = scan_to_rows(eng_tr)
    print('[scan] test drift/terminal ...')
    te = scan_to_rows(eng_te)
    print(f'[pop] post-scan train={len(tr)} test={len(te)}')

    # terminal labels
    for d in (tr, te):
        d['is_good'] = (d['terminal'] >= BAND).values
        d['is_bad'] = (d['terminal'] <= -BAND).values
        d['is_dead'] = (~d['is_good'] & ~d['is_bad']).values
        d['good_kind'] = np.where(d['is_good'] & (d['mindrift'] <= -DIP_PTS), 'dipped',
                                  np.where(d['is_good'], 'clean', '-'))

    # leg geometry join
    tr = join_leg_geometry(tr)
    te = join_leg_geometry(te)
    leg_cov = {k: float(tr[k].notna().mean()) for k in ['pivot_age_min', 'sig_with_leg', 'tod']}
    leg_cov_te = {k: float(te[k].notna().mean()) for k in ['pivot_age_min', 'sig_with_leg', 'tod']}

    # ctx features (lambda_hat, nmp9 tier, trail_vol)
    need_days = set(tr['day']) | set(te['day'])
    fires_by_day = defaultdict(list)
    for d in (tr, te):
        for day, ts in zip(d['day'], d['ts']):
            fires_by_day[day].append(int(ts))
    print(f'[ctx] streaming 5s for {len(need_days)} engagement-days ...')
    ctxf, cov = ctx_features(need_days, fires_by_day)
    print(f"[ctx] lambda coverage: {cov['lambda_ok']}/{cov['lambda_ok']+cov['lambda_nan']} "
          f"defined; zse-missing days={cov['zse_missing_days']}")
    for d in (tr, te):
        d['lambda_hat'] = [ctxf.get((day, int(ts)), {}).get('lambda_hat', np.nan)
                           for day, ts in zip(d['day'], d['ts'])]
        d['trail_vol_ticks'] = [ctxf.get((day, int(ts)), {}).get('trail_vol_ticks', np.nan)
                                for day, ts in zip(d['day'], d['ts'])]
        d['nmp9_tier'] = [ctxf.get((day, int(ts)), {}).get('nmp9_tier', 'none')
                          for day, ts in zip(d['day'], d['ts'])]

    lam_cov_tr = float(np.isfinite(pd.to_numeric(tr['lambda_hat'])).mean())
    lam_cov_te = float(np.isfinite(pd.to_numeric(te['lambda_hat'])).mean())
    tier_share_tr = tr['nmp9_tier'].value_counts(normalize=True).to_dict()

    # ---- FIT SET = train good vs bad (dead excluded) ----
    fit = tr[tr['is_good'] | tr['is_bad']].copy()
    det_cols = sorted(tr['det'].unique())            # train det universe (freeze)
    fill = train_fill(tr)                             # standardize on FULL train (incl dead)
    y_fit = fit['is_good'].values.astype(int)

    # ---- Shainin contrast (train good vs bad) ----
    print('[shainin] contrast + univariate AUC ...')
    days_fit = fit['day'].values
    good_m = fit['is_good'].values; bad_m = fit['is_bad'].values
    contrast = []
    bw_map = dict(P=0.02, pivot_age_min=1.0, tod=0.05, trail_vol_ticks=1.0,
                  lambda_hat=0.05, sig_with_leg=1.0)
    for c in CONT_FEATURES + BIN_FEATURES:
        x = pd.to_numeric(fit[c]).values.astype(float)
        pt, lo, hi = mean_diff_ci(days_fit, x, good_m, bad_m)
        ok = np.isfinite(x)
        try:
            auc = roc_auc_score(y_fit[ok], x[ok])
        except Exception:
            auc = 0.5
        gmode = hist_mode(x[good_m & ok], bw_map[c]); bmode = hist_mode(x[bad_m & ok], bw_map[c])
        gmed = float(np.median(x[good_m & ok])); bmed = float(np.median(x[bad_m & ok]))
        contrast.append(dict(feat=c, good_mode=gmode, bad_mode=bmode, good_med=gmed,
                             bad_med=bmed, diff=pt, lo=lo, hi=hi, auc=auc,
                             absauc=abs(auc - 0.5), kind='cont'))
    # categorical: univariate one-hot logistic train AUC
    for cat, levels in [('nmp9_tier', NMP9_LEVELS), ('det', det_cols)]:
        vals = fit[cat].astype(str).values
        oh = np.column_stack([(vals == lv).astype(float) for lv in levels[1:]])
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(oh, y_fit)
        auc = roc_auc_score(y_fit, lr.predict_proba(oh)[:, 1])
        # most good-skewed / bad-skewed level by good-rate
        gr = {}
        for lv in levels:
            sel = vals == lv
            if sel.sum() >= 20:
                gr[lv] = good_m[sel].mean()
        best = max(gr, key=gr.get) if gr else '-'
        worst = min(gr, key=gr.get) if gr else '-'
        contrast.append(dict(feat=cat, good_mode=f'best={best}', bad_mode=f'worst={worst}',
                             good_med=gr.get(best, float('nan')), bad_med=gr.get(worst, float('nan')),
                             diff=float('nan'), lo=float('nan'), hi=float('nan'),
                             auc=auc, absauc=abs(auc - 0.5), kind='cat'))
    contrast.sort(key=lambda r: -r['absauc'])

    # ---- logistic full model + P-only (same protocol) ----
    print('[fit] full logistic + P-only ...')
    X_fit, names = assemble(fit, det_cols, fill)
    full_lr = LogisticRegression(max_iter=5000, C=1.0)
    full_lr.fit(X_fit, y_fit)
    # P-only: single standardized P column
    Pcol_fit = ((pd.to_numeric(fit['P']).values - fill['mu']['P']) / fill['sd']['P']).reshape(-1, 1)
    p_lr = LogisticRegression(max_iter=5000, C=1.0)
    p_lr.fit(Pcol_fit, y_fit)

    # ---- single-shot test scores (whole test incl dead-band) ----
    X_te, _ = assemble(te, det_cols, fill)
    s_full_te = full_lr.predict_proba(X_te)[:, 1]
    Pcol_te = ((pd.to_numeric(te['P']).values - fill['mu']['P']) / fill['sd']['P']).reshape(-1, 1)
    s_p_te = p_lr.predict_proba(Pcol_te)[:, 1]
    te = te.reset_index(drop=True)
    te['s_full'] = s_full_te
    te['s_p'] = s_p_te

    # AUC on test good-vs-bad (dead excluded from AUC)
    gb = (te['is_good'] | te['is_bad']).values
    y_te_gb = te['is_good'].values[gb].astype(int)
    auc_full = roc_auc_score(y_te_gb, s_full_te[gb])
    auc_p = roc_auc_score(y_te_gb, s_p_te[gb])

    # train scores for frozen thresholds (whole train incl dead)
    X_tr_all, _ = assemble(tr, det_cols, fill)
    s_full_tr = full_lr.predict_proba(X_tr_all)[:, 1]
    Pcol_tr = ((pd.to_numeric(tr['P']).values - fill['mu']['P']) / fill['sd']['P']).reshape(-1, 1)
    s_p_tr = p_lr.predict_proba(Pcol_tr)[:, 1]

    # ---- FULL frontier (description only) ----
    base_te = float(te['is_good'].mean())
    days_te = te['day'].values
    good_te = te['is_good'].values.astype(float)
    frontier = []
    for v in np.linspace(0.95, 0.05, 19):
        tau = np.quantile(s_full_tr, 1 - v)
        keep = (s_full_te >= tau)
        rv = float(keep.mean())
        gr = float(good_te[keep].mean()) if keep.sum() else float('nan')
        frontier.append((v, tau, rv, gr, int(keep.sum())))

    # ---- PRE-REGISTERED operating points ----
    print('[frontier] pre-registered operating points ...')
    ops = []
    for v in RETAIN_LEVELS:
        tau = float(np.quantile(s_full_tr, 1 - v))          # frozen on train
        tau_p = float(np.quantile(s_p_tr, 1 - v))           # P-only frozen on train (equal vol)
        keep_m = (s_full_te >= tau)
        keep_p = (s_p_te >= tau_p)
        keep_all = np.ones(len(te), bool)
        # good-rate delta vs base (dead in denom) and vs P-only at equal vol
        d_base, lo_b, hi_b, ra, rbase = rate_delta_ci(days_te, good_te, keep_m, keep_all)
        d_p, lo_p, hi_p, ra2, rp = rate_delta_ci(days_te, good_te, keep_m, keep_p)
        # decomposition
        n_keep = int(keep_m.sum())
        goods_keep = int((te['is_good'].values & keep_m).sum())
        bads_keep = int((te['is_bad'].values & keep_m).sum())
        dead_keep = int((te['is_dead'].values & keep_m).sum())
        goods_all = int(te['is_good'].sum())
        goods_lost = goods_all - goods_keep
        dipped_lost = int((te['is_good'].values & (te['good_kind'].values == 'dipped') & ~keep_m).sum())
        clean_lost = int((te['is_good'].values & (te['good_kind'].values == 'clean') & ~keep_m).sum())
        bads_all = int(te['is_bad'].sum())
        fails_avoided = bads_all - bads_keep
        ops.append(dict(v=v, tau=tau, tau_p=tau_p, retain_vol=float(keep_m.mean()),
                        retain_vol_p=float(keep_p.mean()), good_rate=ra, base=rbase,
                        good_rate_p=rp, d_base=d_base, lo_b=lo_b, hi_b=hi_b,
                        d_p=d_p, lo_p=lo_p, hi_p=hi_p, n_keep=n_keep, goods_keep=goods_keep,
                        bads_keep=bads_keep, dead_keep=dead_keep, goods_lost=goods_lost,
                        dipped_lost=dipped_lost, clean_lost=clean_lost,
                        fails_avoided=fails_avoided, dead_share=dead_keep / max(n_keep, 1)))

    # ---- PASS/FAIL ----
    verdict = []
    for o in ops:
        beats_base = o['good_rate'] > o['base']
        beats_p = (o['d_p'] > 0) and (o['lo_p'] > 0)      # CI-vs-Ponly excludes 0
        vol_ok = o['retain_vol'] >= 0.30
        verdict.append(vol_ok and beats_base and beats_p)
    passed = any(verdict)

    # ================= write report =================
    print('[write] report ...')
    L = []
    A = L.append
    A('# TASK 122 -- the entry-fail RED X: what separates terminal-good from terminal-bad AT ENTRY?')
    A('')
    A('SEALED Shainin contrast. Logistic P(terminal-good | entry-time features) fit on 2024 '
      'good-vs-bad, coefs + the three retained-volume thresholds FROZEN, single shot on the '
      '2025-26 test tape. P-only is the pre-registered bar -- beating base alone re-discovers P.')
    A('')
    A('## Population + labels')
    A(f'- Engagement machinery: select_wrongdir.engagements() -- P>=p{tb.P_PCTL}(train)'
      f'={thr:.5f} FROZEN, {tb.DEDUP_S}s/day/dir de-dup, MIN_WINDOW>={tb.MIN_WINDOW_MIN}m; '
      f'terminal drift via swl.scan (eb.signed_drift_path).')
    A(f'- **TRAIN (2024, split=train): {len(tr)} engagements** over {tr.day.nunique()} days.')
    A(f'- **TEST (2025-26, split=test): {len(te)} engagements** over {te.day.nunique()} days.')
    A(f'- Terminal labels at BAND={BAND:.0f}: GOOD terminal>=+{BAND:.0f}, BAD terminal<=-{BAND:.0f}, '
      f'DEAD |terminal|<{BAND:.0f}. DEAD excluded from FIT, included in volume accounting.')
    tr_g, tr_b, tr_d = int(tr.is_good.sum()), int(tr.is_bad.sum()), int(tr.is_dead.sum())
    te_g, te_b, te_d = int(te.is_good.sum()), int(te.is_bad.sum()), int(te.is_dead.sum())
    A(f'- TRAIN mix: good={tr_g} ({tr_g/len(tr):.1%}), bad={tr_b} ({tr_b/len(tr):.1%}), '
      f'dead={tr_d} ({tr_d/len(tr):.1%}).')
    A(f'- TEST mix: good={te_g} ({te_g/len(te):.1%}), bad={te_b} ({te_b/len(te):.1%}), '
      f'dead={te_d} ({te_d/len(te):.1%}). **Unconditional terminal-good rate (base) = {base_te:.3f}.**')
    A(f'- Moises\' fail fact reproduced: bad(<=-{BAND:.0f}) share of test = {te_b/len(te):.1%}; '
      f'good share = {te_g/len(te):.1%}.')
    A('')
    A('## Feature joins (all causal at fire ts) + coverage')
    A(f'- entry P, det: econ_drift_rows (native).')
    A(f'- pivot_age_min / sig_with_leg / tod: join fire ts into signal_rows_<det> (det verbatim). '
      f'Coverage train {leg_cov}; test {leg_cov_te}.')
    A(f'- lambda_hat: dsp._nmp_lambda (z_se store L3_1m_z_se_15, NMP_K={NMP_K}, NMP_EPS={NMP_EPS}), '
      f'per-5s ffilled, as-of fire ts. Coverage (defined) train {lam_cov_tr:.3f}, test {lam_cov_te:.3f} '
      f'(undefined -> median-fill + missing indicator).')
    A(f'- NMP9 tier: dsp._nmp9_events waterfall (verbatim constants), as-of last emission at/before '
      f'ts (\'none\' if no tier armed). trail_vol: std of last {TVOL_WIN} 5s closes / TICK (ticks).')
    A('')

    A('## 1. THE SHAININ CONTRAST (train good-vs-bad; ranked by univariate |AUC-0.5|)')
    A('Univariate train AUC = how well each feature ALONE ranks good above bad. diff = '
      'mean(good)-mean(bad) with day-block 95% CI (4000; * = excludes 0). Categoricals show '
      'best/worst good-rate level + one-hot AUC.')
    A('')
    A('| rank | feature | uni AUC | good mode | bad mode | good med | bad med | diff (good-bad) [CI] |')
    A('|---|---|---|---|---|---|---|---|')
    for i, r in enumerate(contrast, 1):
        if r['kind'] == 'cont':
            diff = fmt_ci(r['diff'], r['lo'], r['hi'])
            A(f"| {i} | {r['feat']} | {r['auc']:.3f} | {r['good_mode']:+.3f} | {r['bad_mode']:+.3f} | "
              f"{r['good_med']:+.3f} | {r['bad_med']:+.3f} | {diff} |")
        else:
            A(f"| {i} | {r['feat']} | {r['auc']:.3f} | {r['good_mode']} | {r['bad_mode']} | "
              f"{r['good_med']:.3f} | {r['bad_med']:.3f} | (categorical) |")
    A('')
    top5 = contrast[:5]
    A('**Top-5 dominators:** ' + ', '.join(
        f"{r['feat']} (AUC {r['auc']:.3f})" for r in top5) + '.')
    A('')

    A('## 2. Full model vs P-only -- the increment (single-shot test AUC, good-vs-bad)')
    A(f'- **P-only test AUC = {auc_p:.4f}**')
    A(f'- **Full-model test AUC = {auc_full:.4f}**')
    A(f'- **Incremental AUC (full - P-only) = {auc_full - auc_p:+.4f}**')
    A(f'- Reference signal-magnitude bar (MEMORY §2): AUC gap >=0.10 real / 0.05-0.10 conditional '
      f'/ <0.05 noise. This increment is '
      f'{"REAL" if abs(auc_full-auc_p)>=0.10 else "CONDITIONAL" if abs(auc_full-auc_p)>=0.05 else "NOISE-LEVEL"}.')
    A('')

    A('## 3. Pre-registered operating points (thresholds frozen on 2024 volume; single-shot test)')
    A('good-rate = P(terminal-good) among retained (DEAD in denominator -- deployment reality). '
      'delta-vs-base and delta-vs-P-only(equal-vol) with day-block 95% CI (* = excludes 0).')
    A('')
    A('| target vol (2024) | test retain vol | good-rate | base | vs base [CI] | P-only good-rate | vs P-only [CI] |')
    A('|---|---|---|---|---|---|---|')
    for o in ops:
        A(f"| {o['v']:.0%} | {o['retain_vol']:.1%} | {o['good_rate']:.3f} | {o['base']:.3f} | "
          f"{fmt_ci(o['d_base'], o['lo_b'], o['hi_b'])} | {o['good_rate_p']:.3f} | "
          f"{fmt_ci(o['d_p'], o['lo_p'], o['hi_p'])} |")
    A('')

    A('## 4. Decomposition at each operating point (what gets sacrificed)')
    A('| target vol | N kept | goods kept | bads kept | dead kept | dead share | goods lost (dip/clean) | fails avoided |')
    A('|---|---|---|---|---|---|---|---|')
    for o in ops:
        A(f"| {o['v']:.0%} | {o['n_keep']} | {o['goods_keep']} | {o['bads_keep']} | {o['dead_keep']} | "
          f"{o['dead_share']:.1%} | {o['goods_lost']} ({o['dipped_lost']}/{o['clean_lost']}) | "
          f"{o['fails_avoided']} |")
    A('')

    A('## Full frontier (description only -- NOT the verdict)')
    A('| retain vol target | test retain vol | good-rate | N kept |')
    A('|---|---|---|---|')
    for v, tau, rv, gr, nk in frontier:
        A(f"| {v:.0%} | {rv:.1%} | {gr:.3f} | {nk} |")
    A('')

    A('## PRE-REGISTERED BAR + VERDICT')
    A('Bar: at >=1 operating point with test retain vol >=30%, filtered good-rate beats BOTH '
      '(a) unconditional base AND (b) P-only at equal volume, with delta-vs-P-only CI excluding 0.')
    for o, ok in zip(ops, verdict):
        A(f"- vol {o['v']:.0%} (test {o['retain_vol']:.0%}): beats base={o['good_rate']>o['base']} "
          f"(+{o['d_base']:+.3f}), beats P-only={o['d_p']>0 and o['lo_p']>0} "
          f"(vs-P-only {fmt_ci(o['d_p'], o['lo_p'], o['hi_p'])}) -> {'PASS' if ok else 'fail'}")
    A('')
    A(f'## **VERDICT: {"PASS" if passed else "FAIL"}**')
    if not passed:
        A('No pre-registered operating point beats P-only at equal volume with the delta CI '
          'excluding 0. Entry-time leg/lambda/tier/vol features do NOT add terminal-good '
          'separation over entry P alone -- the fail problem is not solvable at entry with '
          'these features; it lives in the path (turn detector), consistent with the '
          'turns-live-in-paths finding (MEMORY §5).')
    else:
        A('At least one pre-registered point beats both base and P-only (CI-clean). Entry-time '
          'features add real terminal-good separation over P.')
    A('')
    A('_Descriptive path/label study on the sealed test tape. Rates only (no trading sim, '
      'friction irrelevant). A retained rule still graduates through the sealed harness._')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'\nwrote {OUT_MD}')

    # console digest
    print('\n===== SHAININ TOP-5 =====')
    for r in top5:
        print(f"  {r['feat']:16s} uniAUC {r['auc']:.3f}")
    print(f'\nAUC  P-only {auc_p:.4f}  full {auc_full:.4f}  increment {auc_full-auc_p:+.4f}')
    print('\n===== OPERATING POINTS =====')
    for o in ops:
        print(f"  vol{o['v']:.0%} test{o['retain_vol']:.0%}  good-rate {o['good_rate']:.3f} "
              f"base {o['base']:.3f}  vsBase {o['d_base']:+.3f}[{o['lo_b']:+.3f},{o['hi_b']:+.3f}]  "
              f"vsP {o['d_p']:+.3f}[{o['lo_p']:+.3f},{o['hi_p']:+.3f}]")
    print(f'\nVERDICT: {"PASS" if passed else "FAIL"}')


if __name__ == '__main__':
    main()
