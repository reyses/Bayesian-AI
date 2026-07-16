"""
TMPL0 — resurrect the 2024-frozen K-means pattern-template stream for the
label-alignment league (task 2026-07-16, executor build).

WHAT THIS IS
------------
The legacy "template engine" (commit 09cd30d8, core/fractal_clustering.py) clustered
pattern events into ~100s of "templates" via recursive K-means, then learned a
per-template direction bias. This resurrects that as a CAUSAL league stream:

  FIT (2024 only): detect candlestick + geometric pattern events on clock-aligned
    1m/5m/15m buckets of the 5s stream; build a per-event feature vector from the
    subset of the legacy 16-D vector that is CAUSALLY computable from raw bars;
    StandardScaler + recursive K-means (legacy splitting rules / constants) → FROZEN
    codebook of centroids; per template, long_frac = fraction of member 2024 events
    whose ACTIVE AI label was LONG (label-side, 2024-only → allowed for FIT).

  STREAM (2024 fit-period + 2025 + 2026): assign each event to the nearest FROZEN
    2024 centroid (features standardized with the frozen 2024 mean/std); fire
    direction = LONG if that template's long_frac > 0.5 else SHORT; value =
    |long_frac-0.5|. SKIP events whose template has <20 2024 members or
    |long_frac-0.5| < 0.05 (no conviction).

The stream is scored by the shared league harness (dossier_signal_pipeline.evaluate):
direction-agreement with the golden AI labels, train-2024 / test-2025+26, OOS AUC.

SOURCE RECOVERY (read-only, git show 09cd30d8 → research/nt8_catalog/templates_v0/):
  fractal_clustering.py, pattern_utils.py, cuda_pattern_detector.py, three_body_state.py

16-D VECTOR SURVIVAL (legacy fractal_clustering.extract_features order in brackets):
  KEPT (6, causally computable from raw bars):
    [0]  |z|            — 21-bar OLS endpoint z, residual std ddof=2 (dsp._z21, V1 formula)
    [1]  log1p(|vel|)   — velocity = 1-bar close delta in TICKS (0.25), log1p-compressed
    [4]  log2(tf_secs)  — timeframe scale (60/300/900 → structural TF separator)
    [7]  adx/100        — Wilder-14 ADX (see DEVIATION A)
    [8]  hurst          — 30-bar R/S single-window estimator (see DEVIATION B)
    [9]  dmi_diff/100   — Wilder-14 (DI+ − DI−)/100 (matches dsp._tf_state DMI basis)
  DROPPED (10, not causally reconstructable from raw bars with a recovered formula):
    [2]  |momentum|     — legacy PatternEvent.momentum source not in the recovered 4
                          files (set by the field engine); no V1 raw-bar formula → DROP
    [3]  coherence      — entropy_normalized: quantum-wavefunction Shannon entropy from
                          StatisticalFieldEngine; no raw-bar formula → DROP
    [5]  depth          — fractal-tree depth (parent_chain); requires the discovery-agent
                          cascade tree (not in recovered files; parent linkage not
                          causal/portable) → DROP
    [6]  parent_is_roche, [10] parent_z, [11] parent_dmi_diff, [12] root_is_roche,
    [13] tf_alignment   — all ANCESTRY / tree-position dims (parent_chain) → DROP (same
                          reason as depth: the tree is not in the recovered code)
    [14] self_pid       — term_pid (nightmare-field PID control term) → DROP
    [15] self_osc_coh   — oscillation_entropy_normalized → DROP
  Split criterion feature (dim 0 = |z|) SURVIVES, so the recursive-split z-variance
  gate is preserved exactly.

DEVIATIONS FROM LEGACY (declared):
  A. ADX formula is NOT present in the recovered 4 files (three_body_state only STORES
     adx_strength; it is computed in the field engine). Used the canonical Wilder-14 ADX
     on the SAME DMI basis as dmi_diff. /100 to match legacy self_adx scaling.
  B. Hurst formula is NOT in the recovered files. Implemented the standard single-window
     R/S estimator H = log(R/S)/log(N), N=30 ("30-bar R/S per legacy" per task spec),
     clipped to [0,1] to match the legacy hurst_exponent 0-1 range.
  C. Geometric priority follows pattern_utils.detect_geometric_pattern (the single-bar,
     returns-FIRST function the task names): COMPRESSION > WEDGE > BREAKDOWN. (The
     vectorized twin detect_geometric_patterns_vectorized overwrites in the opposite
     order — not used.)
  D. Template membership + long_frac are computed by NEAREST-frozen-centroid assignment
     of the 2024 events (a flat codebook), NOT by the recursive K-means leaf membership.
     Rationale: the STREAM assigns by nearest centroid, so the frozen per-template stats
     must reflect the SAME assignment rule (else fit-time stats and stream-time routing
     disagree at cluster boundaries).
  E. pivot_age_min / sig_with_leg / tod come verbatim from the causal streaming zigzag in
     dsp.DayCtx (reused, not reimplemented) → league-consistent shared features.
  F. Candlestick DIRECTION for the emitted fire is NOT the pattern's textbook direction —
     per task spec, fire direction = the template's frozen 2024 long_frac. is_long_raw
     (textbook: engulf-bull long / engulf-bear short / hammer long / doji+geometric =
     sign(close−open)) is recorded only for the transfer diagnostic.

HARD CONSTRAINTS honored: FIT on 2024 ONLY; no hyperparameter selection on 2025-26; all
features strictly trailing (no centered filters / no future context); legacy constants
(n_clusters=1000, max_variance=0.5, min-members=20, recursive k=min(3,max(2,len//20)),
depth<=5). Commits NOTHING.

Usage:  python3.11 research/nt8_catalog/tools/template_stream_builder.py
Outputs (research/nt8_catalog/reports/):
  signal_rows_TMPL0.parquet     — ALL fires (2024+2025+2026) for the league harness
  tmpl0_templates_2024.json     — per-template {id, member_count, long_frac, centroid,...}
  tmpl0_findings.md             — event counts, template counts, margin dist (eval pasted
                                  in by the executor after running dsp.evaluate)
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dossier_signal_pipeline as dsp   # DayCtx, _z21, RTH gate, paths, TAIL, evaluate

D5, LBL, REP, TAIL = dsp.D5, dsp.LBL, dsp.REP, dsp.TAIL
RTH0, RTH1 = dsp.RTH0, dsp.RTH1

# ---- legacy constants (fractal_clustering.py defaults, verbatim) --------------------
N_CLUSTERS   = 1000     # FractalClusteringEngine(n_clusters=1000)
MAX_VARIANCE = 0.5      # FractalClusteringEngine(max_variance=0.5)
MIN_MEMBERS  = 20       # _recursive_split: len(patterns) <= 20 stops splitting
MAX_DEPTH    = 5        # _recursive_split: depth > 5 stops
KM_SEED, KM_NINIT, KM_MAXITER = 42, 3, 300   # _get_kmeans_model CPU branch
TICK = 0.25
HURST_N = 30            # 30-bar R/S window (task spec)

# ---- stream filters (task spec §4) --------------------------------------------------
MIN_MEMBERS_STREAM = 20      # skip templates with <20 2024 members
MIN_CONVICTION     = 0.05    # skip templates with |long_frac-0.5| < 0.05

PERIODS = {'1m': 60, '5m': 300, '15m': 900}


# ===================== per-TF causal feature helpers ================================
def _wilder_dmi_adx(h, l, c):
    """Wilder-14 DI+/DI-/ADX on a bucketed OHLC series (pandas, ewm alpha=1/14).
    Matches dsp._tf_state's DMI basis; ADX = ewm(alpha=1/14) of DX (DEVIATION A)."""
    h, l, c = pd.Series(h), pd.Series(l), pd.Series(c)
    up, dn = h.diff(), -l.diff()
    dmp = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    dmm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    trs = tr.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    dip = 100 * dmp.ewm(alpha=1/14, adjust=False).mean() / trs
    dim = 100 * dmm.ewm(alpha=1/14, adjust=False).mean() / trs
    dmi = (dip - dim)
    dx = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    return dmi.values, adx.values


def _rs_hurst(c, N=HURST_N):
    """Standard single-window rescaled-range Hurst H=log(R/S)/log(N) on the trailing
    N-bar close series, clipped to [0,1] (DEVIATION B). Strictly trailing."""
    c = np.asarray(c, float)
    n = len(c)
    H = np.full(n, np.nan)
    if n < N:
        return H
    sw = sliding_window_view(c, N)                 # (n-N+1, N), window ends at bar k
    yc = sw - sw.mean(1, keepdims=True)
    zc = np.cumsum(yc, axis=1)
    R = zc.max(1) - zc.min(1)
    S = sw.std(1)                                   # population std (ddof=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(S > 0, R / S, np.nan)
        h = np.where(rs > 0, np.log(rs) / np.log(N), np.nan)
    H[N-1:] = np.clip(h, 0.0, 1.0)
    return H


def _candlestick_flags(o, h, l, c):
    """Legacy cascade cuda_pattern_detector @108-126 (DOJI > HAMMER > ENGULF), verbatim,
    vectorized over bucketed bars. Returns int codes: 0 NONE,1 DOJI,2 HAMMER,
    3 ENGULF_BULL,4 ENGULF_BEAR (each bucket at most one)."""
    o, h, l, c = map(lambda a: np.asarray(a, float), (o, h, l, c))
    body = np.abs(c - o)
    rng = np.where(h - l == 0, 1e-10, h - l)
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    po = pd.Series(o).shift(1).values
    pc = pd.Series(c).shift(1).values
    code = np.zeros(len(c), dtype=np.int8)
    doji = (body / rng) < 0.1                                   # DOJI_BODY_RATIO
    hammer = (~doji) & (lower > 2.0 * body) & (upper < 0.1 * rng) & (body < 0.3 * rng)
    ebull = (~doji) & (~hammer) & (pc < po) & (c > o) & (o <= pc) & (c >= po)
    ebear = (~doji) & (~hammer) & (pc > po) & (c < o) & (o >= pc) & (c <= po)
    code[doji] = 1; code[hammer] = 2; code[ebull] = 3; code[ebear] = 4
    return code


def _geometric_flags(h, l):
    """pattern_utils.detect_geometric_pattern priority (COMPRESSION>WEDGE>BREAKDOWN,
    first-match; DEVIATION C), vectorized. Returns 0 NONE,1 COMPRESSION,2 WEDGE,
    3 BREAKDOWN. First 9 buckets forced NONE (needs 10 bars)."""
    hs, ls = pd.Series(h, dtype=float), pd.Series(l, dtype=float)
    rec_range = hs.rolling(5).max() - ls.rolling(5).min()
    prev_range = rec_range.shift(5)
    comp = ((prev_range > 0) & (rec_range < prev_range * 0.7)).fillna(False).values
    wedge = ((ls > ls.shift(4)) & (hs < hs.shift(4))).fillna(False).values
    prev4min = ls.shift(1).rolling(4).min()
    brk = (ls < prev4min).fillna(False).values
    # first-match priority COMPRESSION > WEDGE > BREAKDOWN (detect_geometric_pattern order)
    code = np.zeros(len(hs), dtype=np.int8)
    code[comp] = 1
    code[wedge & (code == 0)] = 2
    code[brk & (code == 0)] = 3
    code[:9] = 0
    return code


# direction (is_long_raw): textbook reading; diagnostic only (fire uses long_frac)
_CDL_LONG = {3: True, 4: False, 2: True}   # ENGULF_BULL long, ENGULF_BEAR short, HAMMER long


def day_events(ctx):
    """Detect all candlestick + geometric events on the day's 1m/5m/15m buckets.
    Returns list of dicts with causal features + emit-context (leg/pivot_age/tod/ts).
    A bar's event is KNOWN at the first row of the NEXT bucket (open-stamped)."""
    n5 = len(ctx.ts)
    rows = []
    for tf, period in PERIODS.items():
        b = ctx.ts // period
        g = pd.DataFrame({'b': b, 'o': ctx.o, 'h': ctx.h, 'l': ctx.l,
                          'c': ctx.c, 'v': ctx.v}).groupby('b')
        o = g['o'].first().values; h = g['h'].max().values
        l = g['l'].min().values;  c = g['c'].last().values
        ids = g['c'].last().index.values
        if len(c) < HURST_N + 2:
            continue
        # --- 6-D causal feature vector (legacy dims [0,1,4,7,8,9]) ---
        z_abs = np.abs(dsp._z21(c))
        vel_ticks = pd.Series(c).diff().values / TICK
        vel_feat = np.log1p(np.abs(vel_ticks))
        tf_feat = np.full(len(c), np.log2(max(1, period)))
        dmi, adx = _wilder_dmi_adx(h, l, c)
        adx_feat = adx / 100.0
        dmi_feat = dmi / 100.0
        hurst_feat = _rs_hurst(c, HURST_N)
        feats = np.column_stack([z_abs, vel_feat, tf_feat, adx_feat, hurst_feat, dmi_feat])
        # --- pattern flags ---
        cdl = _candlestick_flags(o, h, l, c)
        geo = _geometric_flags(h, l)
        sign_long = (c - o) >= 0                   # doji + geometric direction
        # --- bucket → closing 5s row (first row of NEXT bucket) ---
        first_row = pd.Series(np.arange(n5), index=b).groupby(level=0).first()
        close_row = pd.Series(first_row.values, index=first_row.index - 1)  # bucket->close row
        for k in range(len(ids)):
            if cdl[k] == 0 and geo[k] == 0:
                continue
            fv = feats[k]
            if not np.all(np.isfinite(fv)):
                continue
            cr = close_row.get(ids[k])
            if cr is None or not np.isfinite(cr):
                continue
            r = int(cr)
            if r < ctx.start or not ctx.rth[r]:
                continue
            base = dict(ts=int(ctx.ts[r]),
                        pivot_age_min=(r - ctx.piv_i[r]) * 5 / 60.0,
                        tod=float(ctx.tod[r]), leg=int(ctx.leg[r]),
                        f0=fv[0], f1=fv[1], f2=fv[2], f3=fv[3], f4=fv[4], f5=fv[5])
            if cdl[k] != 0:
                ptype = {1: 'DOJI', 2: 'HAMMER', 3: 'ENGULF_BULL', 4: 'ENGULF_BEAR'}[cdl[k]]
                il_raw = _CDL_LONG.get(int(cdl[k]), bool(sign_long[k]))  # doji→sign
                rows.append({**base, 'tf': tf, 'ptype': ptype, 'is_long_raw': bool(il_raw)})
            if geo[k] != 0:
                ptype = {1: 'COMPRESSION', 2: 'WEDGE', 3: 'BREAKDOWN'}[geo[k]]
                rows.append({**base, 'tf': tf, 'ptype': ptype,
                             'is_long_raw': bool(sign_long[k])})
    return rows


# ===================== recursive K-means (fractal_clustering port) ===================
def _split(Xs, depth):
    """fractal_clustering._recursive_split, returns leaf centroids in STANDARDIZED space."""
    z_var = float(np.std(Xs[:, 0]))
    if z_var <= MAX_VARIANCE or len(Xs) <= MIN_MEMBERS or depth > MAX_DEPTH:
        return [Xs.mean(0)]
    k = min(3, max(2, len(Xs) // MIN_MEMBERS))
    km = KMeans(n_clusters=k, random_state=KM_SEED, n_init=KM_NINIT, max_iter=KM_MAXITER)
    labels = km.fit_predict(Xs)
    out = []
    for lbl in range(k):
        m = labels == lbl
        if m.sum() == 0:
            continue
        out += _split(Xs[m], depth + 1)
    return out


def build_codebook(Xs):
    """fractal_clustering.create_templates: coarse KMeans → per-cluster recursive
    refinement. Returns codebook of STANDARDIZED centroids (N_templates x 6)."""
    target_k = min(N_CLUSTERS, len(Xs) // MIN_MEMBERS)
    target_k = max(target_k, 1)
    print(f'  coarse KMeans: {len(Xs)} events -> {target_k} clusters ...', flush=True)
    km = KMeans(n_clusters=target_k, random_state=KM_SEED, n_init=KM_NINIT,
                max_iter=KM_MAXITER)
    labels = km.fit_predict(Xs)
    centroids = []
    splits = 0
    for lbl in tqdm(range(target_k), desc='  refine'):
        m = labels == lbl
        if m.sum() == 0:
            continue
        sub = Xs[m]
        z_var = float(np.std(sub[:, 0]))
        if z_var > MAX_VARIANCE and len(sub) > MIN_MEMBERS:
            splits += 1
            centroids += _split(sub, 0)
        else:
            centroids.append(sub.mean(0))
    C = np.array(centroids)
    print(f'  refinement: {target_k} coarse -> {len(C)} tight templates ({splits} split)')
    return C


def assign(Xs, C):
    """Nearest-centroid (Euclidean, standardized space). Returns (tid, margin, d1)
    where margin = d2 - d1 (2nd-nearest minus nearest)."""
    # (n x k) squared distances via broadcasting in chunks to bound memory
    n, k = len(Xs), len(C)
    tid = np.empty(n, dtype=np.int64)
    d1 = np.empty(n); margin = np.empty(n)
    Cn = (C * C).sum(1)
    CHUNK = 5000
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        Xb = Xs[s:e]
        d2m = (Xb * Xb).sum(1)[:, None] + Cn[None, :] - 2 * Xb @ C.T
        np.maximum(d2m, 0, out=d2m)
        d = np.sqrt(d2m)
        rows = np.arange(e - s)
        nn = np.argmin(d, axis=1)
        tid[s:e] = nn
        dnn = d[rows, nn]
        d1[s:e] = dnn
        if k > 1:
            d[rows, nn] = np.inf                    # mask nearest, take 2nd nearest
            margin[s:e] = d.min(1) - dnn
        else:
            margin[s:e] = 0.0
    return tid, margin, d1


# ===================== main build ===================================================
def run():
    lblf = {os.path.basename(f)[9:19]: f
            for f in glob.glob(os.path.join(LBL, 'ai_picks_*_multi.json'))}
    files = sorted(glob.glob(os.path.join(D5, '*.parquet')))
    n_lab = sum(1 for f in files if os.path.basename(f)[:10].replace('_', '-') in lblf)
    print(f'{len(files)} 5s days ({n_lab} with labels)')

    # ---- single causal streaming pass: collect all events on label days ----
    all_rows = []
    tail = None
    for p in tqdm(files, desc='stream'):
        day = os.path.basename(p)[:10]
        df = pd.read_parquet(p, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        key = day.replace('_', '-')
        if key in lblf:
            full = pd.concat([tail, df], ignore_index=True) if tail is not None else df
            start = len(tail) if tail is not None else 0
            ctx = dsp.DayCtx(full, start, day, [])
            evs = day_events(ctx)
            if evs:
                year = day[:4]
                # active 2024 label direction per event (FIT target; label-side, 2024-only)
                labs = None
                if year == '2024':
                    labs = [(t['entry_ts'], t['exit_ts'], t.get('direction') == 'LONG')
                            for t in json.load(open(lblf[key])).get('trades', [])
                            if t.get('exit_ts')]
                for ev in evs:
                    ev['day'] = day
                    ev['year'] = year
                    if labs is not None:
                        hit = [lg for a, b, lg in labs if a <= ev['ts'] <= b]
                        ev['active_long'] = bool(hit[0]) if hit else None
                    else:
                        ev['active_long'] = None
                all_rows.extend(evs)
        tail = df.tail(TAIL)

    E = pd.DataFrame(all_rows)
    print(f'\ntotal events: {len(E)}')
    fcols = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']

    # ---- FIT: 2024 only ----
    tr = E[E['year'] == '2024'].copy()
    Xtr = tr[fcols].values.astype(float)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    print(f'FIT on {len(tr)} 2024 events; fitting codebook ...')
    C = build_codebook(Xtr_s)

    # ---- per-template stats via nearest-centroid assignment of 2024 events (DEVIATION D) ----
    tid_tr, _, _ = assign(Xtr_s, C)
    tr['tid'] = tid_tr
    ntpl = len(C)
    member_count = np.zeros(ntpl, dtype=int)
    long_frac = np.full(ntpl, np.nan)
    labeled_count = np.zeros(ntpl, dtype=int)
    textbook_agree = np.full(ntpl, np.nan)
    for t, g in tr.groupby('tid'):
        member_count[t] = len(g)
        lab = g.dropna(subset=['active_long'])
        labeled_count[t] = len(lab)
        if len(lab) > 0:
            long_frac[t] = float(lab['active_long'].astype(bool).mean())
            textbook_agree[t] = float((lab['is_long_raw'].astype(bool)
                                       == lab['active_long'].astype(bool)).mean())

    # frozen raw centroids (legacy stored inverse_transform for readability)
    C_raw = scaler.inverse_transform(C)

    # ---- STREAM: assign ALL events, emit fires that pass filters ----
    Xall_s = scaler.transform(E[fcols].values.astype(float))
    tid_all, margin_all, d1_all = assign(Xall_s, C)
    E['tid'] = tid_all
    E['margin'] = margin_all

    lf = long_frac[tid_all]
    mc = member_count[tid_all]
    conviction = np.abs(lf - 0.5)
    keep = (mc >= MIN_MEMBERS_STREAM) & np.isfinite(lf) & (conviction >= MIN_CONVICTION)
    is_long = lf > 0.5
    leg = E['leg'].values
    sig_with_leg = np.where(leg != 0, ((leg > 0) == is_long).astype(int), 0)

    out = pd.DataFrame({
        'ts': E['ts'].values.astype(np.int64),
        'is_long': is_long.astype(bool),
        'value': conviction.astype(float),
        'pivot_age_min': E['pivot_age_min'].values.astype(float),
        'sig_with_leg': sig_with_leg.astype(int),
        'tod': E['tod'].values.astype(float),
        'day': E['day'].values,
    })[keep].reset_index(drop=True)

    os.makedirs(REP, exist_ok=True)
    pq = os.path.join(REP, 'signal_rows_TMPL0.parquet')
    out.to_parquet(pq)
    print(f'wrote {pq}: {len(out)} fires '
          f'(2024 {(out["day"].str[:4]=="2024").sum()}, '
          f'2025 {(out["day"].str[:4]=="2025").sum()}, '
          f'2026 {(out["day"].str[:4]=="2026").sum()})')

    # ---- intermediate artifacts ----
    templates = []
    for t in range(ntpl):
        templates.append(dict(
            id=int(t), member_count=int(member_count[t]),
            labeled_count=int(labeled_count[t]),
            long_frac=(None if not np.isfinite(long_frac[t]) else round(float(long_frac[t]), 4)),
            textbook_agree=(None if not np.isfinite(textbook_agree[t]) else round(float(textbook_agree[t]), 4)),
            centroid=[round(float(x), 5) for x in C_raw[t]]))
    with open(os.path.join(REP, 'tmpl0_templates_2024.json'), 'w') as f:
        json.dump(dict(
            n_templates=ntpl, feature_dims=['abs_z', 'log1p_vel_ticks', 'log2_tf_secs',
                                            'adx_over_100', 'rs_hurst_30', 'dmi_diff_over_100'],
            scaler_mean=[round(float(x), 6) for x in scaler.mean_],
            scaler_scale=[round(float(x), 6) for x in scaler.scale_],
            templates=templates), f, indent=1)

    # ---- findings ----
    _write_findings(E, out, C, member_count, long_frac, labeled_count,
                    textbook_agree, margin_all, tid_all)
    print('wrote tmpl0_templates_2024.json + tmpl0_findings.md')
    return out, lblf


def _pct(a, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return {p: round(float(np.percentile(a, p)), 4) for p in ps} if len(a) else {}


def _write_findings(E, out, C, member_count, long_frac, labeled_count,
                    textbook_agree, margin_all, tid_all):
    L = []
    L.append('# TMPL0 findings — 2024-frozen K-means pattern-template stream\n')
    L.append('Built by research/nt8_catalog/tools/template_stream_builder.py. '
             'FIT 2024-only; STREAM 2024+2025+2026; features strictly trailing.\n')

    L.append('## Event counts (detected, causal, RTH-gated)')
    piv = E.pivot_table(index='ptype', columns='year', values='ts', aggfunc='count',
                        fill_value=0)
    L.append('```')
    L.append(piv.to_string())
    L.append(f'\nTOTAL events: {len(E)}   (1m {int((E.tf=="1m").sum())}, '
             f'5m {int((E.tf=="5m").sum())}, 15m {int((E.tf=="15m").sum())})')
    L.append('```\n')

    L.append('## Templates')
    kept20 = int((member_count >= MIN_MEMBERS_STREAM).sum())
    conv = np.abs(long_frac - 0.5)
    kept_final = int(((member_count >= MIN_MEMBERS_STREAM) & np.isfinite(long_frac)
                      & (conv >= MIN_CONVICTION)).sum())
    L.append(f'- total templates in codebook: **{len(C)}**')
    L.append(f'- with >=20 2024 members: **{kept20}**')
    L.append(f'- also |long_frac-0.5|>=0.05 (FIRING templates): **{kept_final}**')
    L.append(f'- member_count: {_pct(member_count)}')
    L.append(f'- long_frac (templates with labeled members): {_pct(long_frac)}')
    L.append(f'- labeled_count per template: {_pct(labeled_count)}\n')

    L.append('## KILL-POINT 1 — assignment-margin stability (test 2025+26 events)')
    te = E['year'] != '2024'
    L.append(f'- margin = d(2nd-nearest) - d(nearest), standardized L2')
    L.append(f'- ALL events margin pct: {_pct(margin_all)}')
    L.append(f'- TEST events margin pct: {_pct(margin_all[te.values])}')
    zero = float((margin_all[te.values] < 1e-6).mean()) if te.any() else float("nan")
    L.append(f'- fraction of TEST margins < 1e-6 (tie/unstable): {zero:.4f}\n')

    L.append('## KILL-POINT 2 — do 2024 biases transfer? (textbook-agree per template, 2024 in-sample)')
    L.append(f'- textbook is_long_raw vs active 2024 label, per-template mean: '
             f'{_pct(textbook_agree)}')
    L.append('  (this is IN-SAMPLE 2024; the real transfer test is the OOS-AUC + test '
             'terciles from dsp.evaluate below — a flat 0.50 across terciles = no transfer)\n')

    L.append('## KILL-POINT 3 — beat the no-clustering baseline')
    L.append('- bar: PTRN-ENGULF OOS-AUC 0.616 / PTRN-HAMMER 0.615 (same harness). '
             'TMPL0 OOS-AUC below ~0.616 => clustering adds nothing over raw pattern events.\n')

    L.append('## dsp.evaluate() output (pasted verbatim by executor)')
    L.append('```')
    L.append('<<<EVAL_OUTPUT>>>')
    L.append('```\n')

    with open(os.path.join(REP, 'tmpl0_findings.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    run()
