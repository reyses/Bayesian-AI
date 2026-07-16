"""
GENERALIZED dossier signal pipeline (autonomous night, doc 080).

Funnel (Moises 2026-07-15): extract as much CAUSAL signal as possible aligned with the
AI labels -> mix all signals (pooled combiner) -> hand the completed signal to the
Mamba RL engine for trade management. This tool is stage 1+2.

Per stream: causal triggers on the continuous 5s stream (tail-carry, no cold start) ->
shared features (zigzag pivot_age, sig_with_leg, value, tod, interaction) ->
target = direction agreement with the ACTIVE AI label -> logistic train-2024 /
test-2025+26 -> OOS AUC + tercile calibration. League table across all streams.

FAITHFULNESS: every trigger CONDITION is verbatim from the verified detector or the
legacy deep-dive (cited per generator). Documented deviations only:
  - one-shot-per-day latches on VWAP-03 / ROUND-05 are REMOVED (all fires emitted):
    the latch is a frequency knob, not the signal definition — same call as the ADX
    standard (docs 074/077). ORB-02 / CROSS-11 / VWMA-10 keep first-only because the
    scan-break IS the rule there (doc 070) or the event is structurally once-per-day.
  - windows run CONTINUOUSLY across days (doc 073 no-cold-start ruling).
Skip-rather-than-fabricate: MACD-07, RSI-06, FIB-17, SQZ-04, SAR-23, HNS-22, VP-01,
VA-13, ZONE-21, SCALP-18, ORDERFLOW-14, RENKO-24 (see doc 080 §skip).

Usage: python dossier_signal_pipeline.py [DET ...]   (default: all 12)
Output: reports/dossier_signal_league.md + reports/signal_rows_<det>.parquet
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))
D5 = os.path.join(ROOT, 'DATA', 'ATLAS', '5s')
LBL = os.path.join(ROOT, 'DATA', 'ai_cusp_picks')
REP = os.path.abspath(os.path.join(HERE, '..', 'reports'))
RTH0, RTH1 = pd.Timestamp('08:30').time(), pd.Timestamp('15:15').time()
ATR_N, ATR_MULT, BAR_1M = 14, 4.0, 12   # canonical zigzag: 1m ATR(14) x 4
TAIL = 2500                             # > CROSS-11's 2400-bar SMA across day boundary
RTH_OPEN_S, RTH_LEN_S = 8 * 3600 + 30 * 60, 6.75 * 3600
GAP_MIN = 5.0                           # SEASON-12 legacy |gap| gate (batch_a :64)
PDC_GAP = 2.5                           # OHLC-01 setup-3 gate (batch_a :230)
ROUND_GRID, ROUND_PRIME = 50.0, 5.0     # ROUND-05 legacy grid / prime distance
COOLDOWN = 60                           # DOW-19 / TUNNEL-20 legacy 60-bar cooldown


class DayCtx:
    """One day's continuous-stream context (prior-day tail prepended)."""
    def __init__(self, full, start, day, prior_daily):
        self.start, self.day, self.prior_daily = start, day, prior_daily
        self.ts = full['timestamp'].values.astype(np.int64)
        self.c = full['close'].values
        self.h = full['high'].values
        self.l = full['low'].values
        self.v = full['volume'].values.astype(float) if 'volume' in full else np.zeros(len(full))
        dt = pd.to_datetime(full['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        tt = dt.dt.time
        self.rth = ((tt >= RTH0) & (tt <= RTH1)).values
        secs = (dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second).values
        self.tod = np.clip((secs - RTH_OPEN_S) / RTH_LEN_S, 0, 1)
        self.before9 = (secs < 9 * 3600)
        # causal streaming zigzag (adx_prob_logistic.stream_day pattern)
        c1 = pd.Series(self.c).groupby(np.arange(len(self.c)) // BAR_1M).last()
        h1 = pd.Series(self.h).groupby(np.arange(len(self.h)) // BAR_1M).max()
        l1 = pd.Series(self.l).groupby(np.arange(len(self.l)) // BAR_1M).min()
        pc1 = c1.shift(1)
        tr1 = pd.concat([h1 - l1, (h1 - pc1).abs(), (l1 - pc1).abs()], axis=1).max(axis=1)
        atr1 = tr1.rolling(ATR_N, min_periods=ATR_N).mean()
        self.zz_thr = atr1.reindex(np.arange(len(self.c)) // BAR_1M).values * ATR_MULT
        n = len(self.c)
        self.piv_i = np.zeros(n, dtype=np.int64)
        self.leg = np.zeros(n, dtype=np.int8)
        self.piv_confirm = np.zeros(n, dtype=np.int8)   # +1/-1 at the confirmation bar
        hi_i = lo_i = 0; hi_v = lo_v = self.c[0]; d = 0; last = 0
        for i in range(1, n):
            x = self.c[i]; t = self.zz_thr[i] if np.isfinite(self.zz_thr[i]) else np.inf
            if x > hi_v: hi_v, hi_i = x, i
            if x < lo_v: lo_v, lo_i = x, i
            if d >= 0 and hi_v - x >= t:
                last, d = hi_i, -1; lo_v, lo_i = x, i; self.piv_confirm[i] = -1
            elif d <= 0 and x - lo_v >= t:
                last, d = lo_i, 1; hi_v, hi_i = x, i; self.piv_confirm[i] = +1
            self.piv_i[i], self.leg[i] = last, d

    def emit(self, i, is_long, value):
        return dict(ts=int(self.ts[i]), is_long=bool(is_long), value=float(value),
                    pivot_age_min=(i - self.piv_i[i]) * 5 / 60.0,
                    sig_with_leg=int((self.leg[i] > 0) == bool(is_long)) if self.leg[i] != 0 else 0,
                    tod=float(self.tod[i]), day=self.day)

    def rth_idx(self):
        return np.flatnonzero(self.rth & (np.arange(len(self.c)) >= self.start))


# ---- generators (trigger conditions cited to verified detectors / legacy) ------------
def gen_zigzag(ctx):
    """ZIGZAG pivot confirmation (canonical ATR(14)x4 causal spec). Direction = new leg."""
    return [ctx.emit(i, ctx.piv_confirm[i] > 0,
                     ctx.zz_thr[i] if np.isfinite(ctx.zz_thr[i]) else 0.0)
            for i in ctx.rth_idx() if ctx.piv_confirm[i] != 0]

def gen_orb02(ctx):
    """ORB-02 (batch_a_detectors.ORB02Detector: close-based OR 08:30-<09:00, first
    breach >=09:00, ONE-SHOT either direction)."""
    idx = ctx.rth_idx()
    orc = [ctx.c[i] for i in idx if ctx.before9[i]]
    if not orc: return []
    orh, orl = max(orc), min(orc)
    for i in idx:
        if ctx.before9[i]: continue
        if ctx.c[i] > orh: return [ctx.emit(i, True, ctx.c[i] - orh)]
        if ctx.c[i] < orl: return [ctx.emit(i, False, orl - ctx.c[i])]
    return []

def gen_season12(ctx):
    """SEASON-12 (batch_a SEASON12Detector: first RTH bar, gap vs prior RTH close,
    |gap|>=5). Direction = toward fill (the article's registered response)."""
    if not ctx.prior_daily: return []
    pdc = ctx.prior_daily[-1]['close']
    idx = ctx.rth_idx()
    if len(idx) == 0: return []
    gap = ctx.c[idx[0]] - pdc
    return [ctx.emit(idx[0], gap < 0, abs(gap))] if abs(gap) >= GAP_MIN else []

def gen_vwap03(ctx):
    """VWAP-03 z-turn (batch_a VWAP03Detector verbatim: session VWAP over RTH bars,
    rolling-20 close std ddof=1 floor 0.25, prime |z|>2 then turn toward mean).
    Deviation: one-shot latch removed; to avoid fire-on-every-downtick, priming is on
    the CROSSING of |z|=2 (one fire per excursion), the minimal multi-fire
    generalization of the legacy one-shot."""
    out = []; cum_pv = cum_vol = 0.0; buf = []; pb = pbear = False; zprev = 0.0
    for i in range(len(ctx.c)):
        if not ctx.rth[i]:
            cum_pv = cum_vol = 0.0; buf = []; pb = pbear = False; zprev = 0.0
            continue
        cum_pv += ctx.c[i] * ctx.v[i]; cum_vol += ctx.v[i]
        vwap = ctx.c[i] if cum_vol == 0 else cum_pv / cum_vol
        buf.append(ctx.c[i]); buf = buf[-20:]
        if len(buf) < 20: continue
        z = (ctx.c[i] - vwap) / max(0.25, float(np.std(buf, ddof=1)))
        fire_bear = pbear and z < zprev and z > 0
        fire_bull = pb and z > zprev and z < 0
        if z > 2.0 and zprev <= 2.0: pbear = True
        elif fire_bear or z <= 0: pbear = False
        if z < -2.0 and zprev >= -2.0: pb = True
        elif fire_bull or z >= 0: pb = False
        if i >= ctx.start:
            if fire_bear: out.append(ctx.emit(i, False, z))
            if fire_bull: out.append(ctx.emit(i, True, -z))
        zprev = z
    return out

def gen_ohlc01(ctx):
    """OHLC-01 (batch_a OHLC01Detector: PDH touch=short, PDL touch=long, PDC gap-fill
    |open-pdc|>2.5 crossing toward pdc keeps the crossing direction; one-shot each)."""
    if not ctx.prior_daily: return []
    d = ctx.prior_daily[-1]
    idx = ctx.rth_idx()
    if len(idx) == 0: return []
    o = ctx.c[idx[0]]; out = []; s1 = s2 = s3 = False
    for i in idx:
        p = ctx.c[i]
        if not s1 and o < d['high'] and p >= d['high']: out.append(ctx.emit(i, False, p - o)); s1 = True
        if not s2 and o > d['low'] and p <= d['low']: out.append(ctx.emit(i, True, o - p)); s2 = True
        if not s3 and abs(o - d['close']) > PDC_GAP:
            if o < d['close'] and p >= d['close']: out.append(ctx.emit(i, True, abs(o - d['close']))); s3 = True
            elif o > d['close'] and p <= d['close']: out.append(ctx.emit(i, False, abs(o - d['close']))); s3 = True
    return out

def gen_pivot16(ctx):
    """PIVOT-16 (batch_a PIVOT16Detector: PP=(H+L+C)/3; long at S1 touch, short at R1)."""
    if not ctx.prior_daily: return []
    d = ctx.prior_daily[-1]
    pp = (d['high'] + d['low'] + d['close']) / 3.0
    s1v, r1v = 2 * pp - d['high'], 2 * pp - d['low']
    idx = ctx.rth_idx()
    if len(idx) == 0: return []
    o = ctx.c[idx[0]]; out = []; g1 = g2 = False
    for i in idx:
        p = ctx.c[i]
        if not g1 and o > s1v and p <= s1v: out.append(ctx.emit(i, True, o - p)); g1 = True
        if not g2 and o < r1v and p >= r1v: out.append(ctx.emit(i, False, p - o)); g2 = True
    return out

def gen_round05(ctx):
    """ROUND-05 (batch_a ROUND05Detector: 50-pt grid, prime 5 beyond, continuation
    through the level). Deviation: one-shot latch removed."""
    out = []; prim_b = {}; prim_s = {}
    for i in range(len(ctx.c)):
        p = ctx.c[i]
        base = int(p / ROUND_GRID) * ROUND_GRID
        for L in (base - ROUND_GRID, base, base + ROUND_GRID):
            if p >= L and prim_b.get(L):
                prim_b[L] = False
                if ctx.rth[i] and i >= ctx.start: out.append(ctx.emit(i, True, ROUND_PRIME))
            if p <= L and prim_s.get(L):
                prim_s[L] = False
                if ctx.rth[i] and i >= ctx.start: out.append(ctx.emit(i, False, ROUND_PRIME))
            if p < L - ROUND_PRIME: prim_b[L] = True
            elif p >= L: prim_b[L] = False
            if p > L + ROUND_PRIME: prim_s[L] = True
            elif p <= L: prim_s[L] = False
    return out

def gen_cross11(ctx):
    """CROSS-11 (batch_b CROSS11Detector: 600/2400-bar SMAs continuous, FIRST cross of
    the session only — the legacy break IS the rule, doc 070)."""
    c = pd.Series(ctx.c)
    s50 = c.rolling(600, min_periods=600).mean().values
    s200 = c.rolling(2400, min_periods=2400).mean().values
    for i in ctx.rth_idx():
        if i < 2400 or not np.isfinite(s200[i - 1]): continue
        if s50[i - 1] <= s200[i - 1] and s50[i] > s200[i]: return [ctx.emit(i, True, s50[i] - s200[i])]
        if s50[i - 1] >= s200[i - 1] and s50[i] < s200[i]: return [ctx.emit(i, False, s200[i] - s50[i])]
    return []

def gen_vwma10(ctx):
    """VWMA-10 (ag_deepdive_10_vwma.py:36-77: 240-bar VWMA vs 240-bar SMA, FIRST cross
    of the day only; VWMA above = bullish)."""
    c = pd.Series(ctx.c); v = pd.Series(ctx.v)
    vw = ((c * v).rolling(240, min_periods=240).sum() / v.rolling(240, min_periods=240).sum()).values
    sm = c.rolling(240, min_periods=240).mean().values
    for i in ctx.rth_idx():
        if i < 241 or not (np.isfinite(vw[i - 1]) and np.isfinite(sm[i - 1])): continue
        if vw[i - 1] <= sm[i - 1] and vw[i] > sm[i]: return [ctx.emit(i, True, vw[i] - sm[i])]
        if vw[i - 1] >= sm[i - 1] and vw[i] < sm[i]: return [ctx.emit(i, False, sm[i] - vw[i])]
    return []

def gen_dow19(ctx):
    """DOW-19 trap (batch_b DOW19Detector: buffers on ALL bars, prev-10-close extremes
    = close.shift(1).rolling(10), vol<vol_sma20, 60-RTH-bar cooldown; break up on low
    vol = SHORT trap)."""
    c = pd.Series(ctx.c); v = pd.Series(ctx.v)
    vs = v.rolling(20, min_periods=20).mean().values
    hi10 = c.shift(1).rolling(10, min_periods=10).max().values
    lo10 = c.shift(1).rolling(10, min_periods=10).min().values
    out = []; cool = 0
    for i in ctx.rth_idx():
        if i < 21 or not np.isfinite(vs[i]): continue
        if cool > 0: cool -= 1; continue
        if ctx.v[i] < vs[i]:
            if ctx.c[i] > hi10[i]: out.append(ctx.emit(i, False, ctx.c[i] - hi10[i])); cool = COOLDOWN
            elif ctx.c[i] < lo10[i]: out.append(ctx.emit(i, True, lo10[i] - ctx.c[i])); cool = COOLDOWN
    return out

def gen_tunnel20(ctx):
    """TUNNEL-20 (ag_deepdive_20_tunnel.py:39-78: EMA34 of high/low tunnel, close
    crossing above tunnel-high = bullish impulse / below tunnel-low = bearish,
    60-bar cooldown, multiple per day)."""
    eh = pd.Series(ctx.h).ewm(span=34, adjust=False).mean().values
    el = pd.Series(ctx.l).ewm(span=34, adjust=False).mean().values
    out = []; cool = 0
    for i in ctx.rth_idx():
        if i < 1: continue
        if cool > 0: cool -= 1; continue
        if ctx.c[i - 1] <= eh[i - 1] and ctx.c[i] > eh[i]:
            out.append(ctx.emit(i, True, ctx.c[i] - eh[i])); cool = COOLDOWN
        elif ctx.c[i - 1] >= el[i - 1] and ctx.c[i] < el[i]:
            out.append(ctx.emit(i, False, el[i] - ctx.c[i])); cool = COOLDOWN
    return out

def gen_atr09(ctx):
    """ATR-09 fade (batch_b ATR09Detector: TRUE 14-day ATR from prior H/L/C; at the
    FIRST crossing of each range threshold x in {0.5,0.75,1.0}, fade if price within
    0.25 of the running extreme). Running extremes = TODAY's RTH closes only."""
    pdays = ctx.prior_daily
    if len(pdays) < 15: return []
    atr = float(np.mean([max(pdays[j]['high'] - pdays[j]['low'],
                             abs(pdays[j]['high'] - pdays[j - 1]['close']),
                             abs(pdays[j]['low'] - pdays[j - 1]['close']))
                         for j in range(-14, 0)]))
    out = []; rh = -np.inf; rl = np.inf
    trig = {0.5: False, 0.75: False, 1.0: False}
    for i in ctx.rth_idx():
        p = ctx.c[i]
        rh = max(rh, p); rl = min(rl, p)
        for x in trig:
            if not trig[x] and (rh - rl) >= x * atr:
                trig[x] = True
                if p >= rh - 0.25: out.append(ctx.emit(i, False, x))
                elif p <= rl + 0.25: out.append(ctx.emit(i, True, x))
    return out


GENS = {'ZIGZAG': gen_zigzag, 'ORB-02': gen_orb02, 'SEASON-12': gen_season12,
        'VWAP-03': gen_vwap03, 'OHLC-01': gen_ohlc01, 'PIVOT-16': gen_pivot16,
        'ROUND-05': gen_round05, 'CROSS-11': gen_cross11, 'VWMA-10': gen_vwma10,
        'DOW-19': gen_dow19, 'TUNNEL-20': gen_tunnel20, 'ATR-09': gen_atr09}


def run_all(dets):
    """Stream ALL 5s days for continuity (tail + prior-day context); emit signals only
    on label days."""
    lblf = {os.path.basename(f)[9:19]: f for f in glob.glob(os.path.join(LBL, 'ai_picks_*_multi.json'))}
    files = sorted(glob.glob(os.path.join(D5, '*.parquet')))
    n_lab = sum(1 for f in files if os.path.basename(f)[:10].replace('_', '-') in lblf)
    print(f'{len(files)} 5s days ({n_lab} with labels); streams: {dets}')
    rows = {d: [] for d in dets}
    tail = None; prior_daily = []
    for p in tqdm(files, desc='days'):
        day = os.path.basename(p)[:10]
        df = pd.read_parquet(p, columns=['timestamp', 'high', 'low', 'close', 'volume'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        if day.replace('_', '-') in lblf:
            full = pd.concat([tail, df], ignore_index=True) if tail is not None else df
            ctx = DayCtx(full, len(tail) if tail is not None else 0, day, prior_daily)
            for d in dets:
                rows[d] += GENS[d](ctx)
        # today's TRUE RTH H/L/C for tomorrow's prior-day context (audit-fixed: not close-as-high)
        dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        m = ((dt.dt.time >= RTH0) & (dt.dt.time <= RTH1)).values
        if m.any():
            prior_daily.append(dict(high=float(df['high'].values[m].max()),
                                    low=float(df['low'].values[m].min()),
                                    close=float(df['close'].values[m][-1])))
            prior_daily = prior_daily[-20:]
        tail = df.tail(TAIL)
    return {d: pd.DataFrame(r) for d, r in rows.items()}, lblf


COLS = ['pivot_age_min', 'sig_with_leg', 'value', 'tod', 'inter']

def evaluate(det, F, lblf):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if len(F) < 200: return dict(det=det, n=len(F), note='too few signals')
    tgt = []
    for day, g in F.groupby('day', sort=False):
        labs = [(t['entry_ts'], t['exit_ts'], t.get('direction') == 'LONG')
                for t in json.load(open(lblf[day.replace('_', '-')])).get('trades', [])
                if t.get('exit_ts')]
        for ridx, r in g.iterrows():
            hit = [lg for a, b, lg in labs if a <= r['ts'] <= b]
            tgt.append((ridx, int(hit[0] == r['is_long']) if hit else np.nan))
    F = F.copy()
    F['y'] = pd.Series(dict(tgt))
    F = F.dropna(subset=['y'])
    F['year'] = F['day'].str[:4]
    F['inter'] = F['sig_with_leg'] * F['pivot_age_min']
    trm, tem = F['year'] == '2024', F['year'] != '2024'
    if trm.sum() < 100 or tem.sum() < 100: return dict(det=det, n=len(F), note='thin split')
    Xtr, ytr = F.loc[trm, COLS].values, F.loc[trm, 'y'].astype(int).values
    Xte, yte = F.loc[tem, COLS].values, F.loc[tem, 'y'].astype(int).values
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return dict(det=det, n=len(F), note='one-class')
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=1000).fit((Xtr - mu) / sd, ytr)
    pte = clf.predict_proba((Xte - mu) / sd)[:, 1]
    auc = roc_auc_score(yte, pte)
    try:
        q = pd.qcut(pte, 3, labels=['low', 'mid', 'high'])
    except ValueError:
        q = pd.Series(['mid'] * len(pte))
    days_te = F.loc[tem, 'day'].values
    ter = {}
    for b in ['low', 'mid', 'high']:
        m = np.asarray(q == b)
        if m.sum() < 10: continue
        uq = np.unique(days_te[m]); boots = []
        for _ in range(1000):
            s = np.random.choice(uq, len(uq), True)
            vv = np.concatenate([yte[m][days_te[m] == d2] for d2 in s])
            if len(vv): boots.append(vv.mean())
        ter[b] = (int(m.sum()), float(yte[m].mean()),
                  float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    F.to_parquet(os.path.join(REP, f'signal_rows_{det.replace("-", "")}.parquet'))
    return dict(det=det, n=len(F), n_tr=int(trm.sum()), n_te=int(tem.sum()),
                base_te=float(yte.mean()), auc=float(auc), ter=ter,
                coefs=dict(zip(COLS, np.round(clf.coef_[0], 3))))


if __name__ == '__main__':
    dets = sys.argv[1:] or list(GENS)
    streams, lblf = run_all(dets)
    lines = ['# Dossier signal league — direction agreement with AI labels',
             '(train 2024, test 2025+26, day-block bootstrap CIs; baseline 0.50)\n']
    for det in dets:
        r = evaluate(det, streams[det], lblf)
        if 'note' in r:
            print(f'{det:10} N={r["n"]}  {r["note"]}')
            lines.append(f'- **{det}**: N={r["n"]} — {r["note"]}')
            continue
        t = r['ter']
        ts = ' | '.join(f'{b}: {t[b][1]:.2f} [{t[b][2]:.2f},{t[b][3]:.2f}] N={t[b][0]}' for b in t)
        print(f'{det:10} N={r["n"]:6} OOS-AUC {r["auc"]:.3f} base {r["base_te"]:.2f} || {ts}')
        lines.append(f'## {det}\n- N={r["n"]} (train {r["n_tr"]} / test {r["n_te"]}), '
                     f'OOS AUC **{r["auc"]:.3f}**, test base {r["base_te"]:.2f}\n'
                     f'- P-terciles: {ts}\n- coefs: {r["coefs"]}')
    with open(os.path.join(REP, 'dossier_signal_league.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\nwrote reports/dossier_signal_league.md')
