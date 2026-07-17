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
        self.o = full['open'].values if 'open' in full else full['close'].values
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


# ---- batch 2 (2026-07-16, easiest-first port of the doc-080 skip list) ---------------
def gen_sar23(ctx):
    """SAR-23 (ag_deepdive_23_sar.py:10-67,116-135: PSAR af 0.02/0.2 on high/low,
    continuous; flip to bullish = LONG / bearish = SHORT; 60-bar cooldown)."""
    h, l = ctx.h, ctx.l
    n = len(h); bull = np.ones(n, dtype=bool); psar = np.zeros(n)
    psar[0] = l[0]; ep = h[0]; af = 0.02
    for i in range(1, n):
        if bull[i-1]:
            cur = psar[i-1] + af * (ep - psar[i-1])
            cur = min(cur, l[i-1], l[i-2]) if i >= 2 else min(cur, l[i-1])
            if l[i] < cur:
                bull[i] = False; psar[i] = ep; ep = l[i]; af = 0.02
            else:
                bull[i] = True; psar[i] = cur
                if h[i] > ep: ep = h[i]; af = min(af + 0.02, 0.2)
        else:
            cur = psar[i-1] - af * (psar[i-1] - ep)
            cur = max(cur, h[i-1], h[i-2]) if i >= 2 else max(cur, h[i-1])
            if h[i] > cur:
                bull[i] = True; psar[i] = ep; ep = h[i]; af = 0.02
            else:
                bull[i] = False; psar[i] = cur
                if l[i] < ep: ep = l[i]; af = min(af + 0.02, 0.2)
    out = []; cool = 0
    for i in ctx.rth_idx():
        if i < 1: continue
        if cool > 0: cool -= 1; continue
        if bull[i] != bull[i-1]:
            out.append(ctx.emit(i, bool(bull[i]), abs(ctx.c[i] - psar[i]))); cool = COOLDOWN
    return out

def gen_sqz04(ctx):
    """SQZ-04 (ag_deepdive_04_squeeze.py:42-88: BB(20,2.0 ddof=0) vs Keltner(20,
    1.5x SMA20|dClose|); squeeze on = BB inside KC; fire on close crossing outside
    the BB while squeeze on now-or-prev. Deviation: one-shot latch removed,
    60-bar cooldown (frequency knob; condition verbatim)."""
    c = pd.Series(ctx.c)
    sma = c.rolling(20, min_periods=20).mean()
    std = c.rolling(20, min_periods=20).std(ddof=0)
    ubb, lbb = (sma + 2.0 * std).values, (sma - 2.0 * std).values
    patr = c.diff().abs().fillna(0).rolling(20, min_periods=20).mean()
    on = ((sma + 2.0 * std < sma + 1.5 * patr) & (sma - 2.0 * std > sma - 1.5 * patr)).values
    out = []; cool = 0
    for i in ctx.rth_idx():
        if i < 21 or not np.isfinite(ubb[i - 1]): continue
        if cool > 0: cool -= 1; continue
        sq = on[i] or on[i - 1]
        if sq and ctx.c[i-1] <= ubb[i-1] and ctx.c[i] > ubb[i]:
            out.append(ctx.emit(i, True, ctx.c[i] - ubb[i])); cool = COOLDOWN
        elif sq and ctx.c[i-1] >= lbb[i-1] and ctx.c[i] < lbb[i]:
            out.append(ctx.emit(i, False, lbb[i] - ctx.c[i])); cool = COOLDOWN
    return out

def gen_rsi06(ctx):
    """RSI-06 divergence (ag_deepdive_06_rsi.py:24-97: RSI ewm com=167 (14x12 bars),
    360-bar rolling extremes; price AT 30m max with RSI below its 30m max = SHORT,
    mirror = LONG. Deviation: latch removed, 60-bar cooldown."""
    delta = pd.Series(ctx.c).diff()
    ag = delta.clip(lower=0).ewm(com=167, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(com=167, adjust=False).mean()
    rsi = (100 - 100 / (1 + ag / al)).values
    c = pd.Series(ctx.c); r = pd.Series(rsi)
    pmax = c.rolling(360, min_periods=360).max().values
    pmin = c.rolling(360, min_periods=360).min().values
    rmax = r.rolling(360, min_periods=360).max().values
    rmin = r.rolling(360, min_periods=360).min().values
    out = []; cool = 0
    for i in ctx.rth_idx():
        if not np.isfinite(pmax[i]) or not np.isfinite(rmax[i]): continue
        if cool > 0: cool -= 1; continue
        if ctx.c[i] == pmax[i] and rsi[i] < rmax[i]:
            out.append(ctx.emit(i, False, rmax[i] - rsi[i])); cool = COOLDOWN
        elif ctx.c[i] == pmin[i] and rsi[i] > rmin[i]:
            out.append(ctx.emit(i, True, rsi[i] - rmin[i])); cool = COOLDOWN
    return out

def gen_macd07(ctx):
    """MACD-07 divergence (ag_deepdive_07_macd.py:42-80: MACD = EMA144-EMA312,
    360-bar extremes; price >= 30m high with MACD < its 30m high = SHORT, mirror
    = LONG. Deviation: latch removed, 60-bar cooldown."""
    c = pd.Series(ctx.c)
    macd = (c.ewm(span=144, adjust=False).mean() - c.ewm(span=312, adjust=False).mean())
    ph = c.rolling(360, min_periods=360).max().values
    pl = c.rolling(360, min_periods=360).min().values
    mh = macd.rolling(360, min_periods=360).max().values
    ml = macd.rolling(360, min_periods=360).min().values
    m = macd.values
    out = []; cool = 0
    for i in ctx.rth_idx():
        if not np.isfinite(ph[i]) or not np.isfinite(mh[i]): continue
        if cool > 0: cool -= 1; continue
        if ctx.c[i] >= ph[i] and m[i] < mh[i]:
            out.append(ctx.emit(i, False, mh[i] - m[i])); cool = COOLDOWN
        elif ctx.c[i] <= pl[i] and m[i] > ml[i]:
            out.append(ctx.emit(i, True, m[i] - ml[i])); cool = COOLDOWN
    return out

def gen_scalp18(ctx):
    """SCALP-18 (ag_deepdive_18_scalp.py:58-87: session VWAP + EMA240 + RSI(alpha=
    1/168) computed on RTH bars only, first 240 RTH bars skipped; LONG = above VWAP,
    pulled back to EMA, RSI<40; SHORT mirror). Session-scoped by design (VWAP is
    session-anchored). Deviation: latch removed, 60-bar cooldown."""
    idx = ctx.rth_idx()
    if len(idx) < 500: return []
    p = ctx.c[idx]; v = ctx.v[idx]
    cum_v = np.maximum(np.cumsum(v), 1)
    vwap = np.cumsum(p * v) / cum_v
    ema = pd.Series(p).ewm(span=240, adjust=False).mean().values
    d = np.diff(p, prepend=p[0])
    up = pd.Series(np.where(d > 0, d, 0)).ewm(alpha=1/168, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0)).ewm(alpha=1/168, adjust=False).mean().values
    rsi = 100 - 100 / (1 + up / (dn + 1e-10))
    out = []; cool = 0
    for k in range(240, len(idx)):
        if cool > 0: cool -= 1; continue
        if p[k] > vwap[k] and p[k] <= ema[k] and rsi[k] < 40:
            out.append(ctx.emit(idx[k], True, vwap[k] - p[k])); cool = COOLDOWN
        elif p[k] < vwap[k] and p[k] >= ema[k] and rsi[k] > 60:
            out.append(ctx.emit(idx[k], False, p[k] - vwap[k])); cool = COOLDOWN
    return out

def gen_renko24(ctx):
    """RENKO-24 (batch_a_detectors.RENKO24Detector verbatim: 2.0-pt bricks, 2-brick
    reversal; trigger = 2nd consecutive brick right after a direction flip). Brick
    chain seeded at first RTH bar each day, per legacy."""
    BRICK = 2.0
    out = []; prev_close = None; cur_d = 0; prev_d = 0; chain = 0
    for i in ctx.rth_idx():
        p = ctx.c[i]
        if prev_close is None:
            prev_close = np.floor(p / BRICK) * BRICK
            continue
        while True:
            if cur_d == 0:
                if p >= prev_close + BRICK: cur_d, prev_d, prev_close, chain = 1, 0, prev_close + BRICK, 1
                elif p <= prev_close - BRICK: cur_d, prev_d, prev_close, chain = -1, 0, prev_close - BRICK, 1
                else: break
            elif cur_d == 1:
                if p >= prev_close + BRICK:
                    prev_close += BRICK; chain += 1
                    if chain == 2 and prev_d == -1: out.append(ctx.emit(i, True, BRICK))
                elif p <= prev_close - 2 * BRICK:
                    prev_d, cur_d, prev_close, chain = 1, -1, prev_close - 2 * BRICK, 1
                else: break
            else:
                if p <= prev_close - BRICK:
                    prev_close -= BRICK; chain += 1
                    if chain == 2 and prev_d == 1: out.append(ctx.emit(i, False, BRICK))
                elif p >= prev_close + 2 * BRICK:
                    prev_d, cur_d, prev_close, chain = -1, 1, prev_close + 2 * BRICK, 1
                else: break
    return out


def gen_vp01(ctx):
    """VP-01 (ag_deepdive_01_vol_profile.py:126-154 vs YESTERDAY's profile:
    open in [VAH, prior-high] -> first touch of POC = LONG bounce; open in
    [prior-low, VAL] -> first touch of POC = SHORT; open beyond prior extremes =
    runner at the open in the breakout direction; one-shot per day).
    ph/pl = prior TRUE H/L (doc-070 close-as-high defect ruling); POC/VA =
    close-binned volume profile (the profile's own definition)."""
    if not ctx.prior_daily or 'poc' not in ctx.prior_daily[-1]: return []
    d = ctx.prior_daily[-1]
    idx = ctx.rth_idx()
    if len(idx) == 0: return []
    o = ctx.c[idx[0]]
    if d['vah'] < o < d['high']:
        for i in idx:
            if ctx.c[i] <= d['poc']: return [ctx.emit(i, True, d['vah'] - ctx.c[i])]
    elif d['low'] < o < d['val']:
        for i in idx:
            if ctx.c[i] >= d['poc']: return [ctx.emit(i, False, ctx.c[i] - d['val'])]
    elif o > d['high']:
        return [ctx.emit(idx[0], True, o - d['high'])]
    elif o < d['low']:
        return [ctx.emit(idx[0], False, d['low'] - o)]
    return []

def gen_va13(ctx):
    """VA-13 rotation (ag_deepdive_13_va.py:126-154: open INSIDE yesterday's value
    area; first touch of VAH then close back below it = SHORT rotation toward POC;
    mirror at VAL = LONG; one-shot per day)."""
    if not ctx.prior_daily or 'poc' not in ctx.prior_daily[-1]: return []
    d = ctx.prior_daily[-1]
    idx = ctx.rth_idx()
    if len(idx) == 0: return []
    o = ctx.c[idx[0]]
    if not (d['val'] < o < d['vah']): return []
    t_vah = t_val = False
    for i in idx:
        p = ctx.c[i]
        if not t_vah and not t_val:
            if p >= d['vah']: t_vah = True
            elif p <= d['val']: t_val = True
        elif t_vah:
            if p < d['vah']: return [ctx.emit(i, False, abs(p - d['poc']))]
        elif t_val:
            if p > d['val']: return [ctx.emit(i, True, abs(p - d['poc']))]
    return []

def gen_hns22(ctx):
    """HNS-22 (ag_deepdive_22_hns.py:47-110,209-210: 21-bar CENTERED peak/trough
    flags registered 10 bars late (causal); TOP pattern p3<t2<p2<t1<p1 with head
    highest, shoulders within 0.5x(head-RS), volume divergence LS>Head>RS
    (+-2-bar means); trigger = close crossing below the t2->t1 neckline; SHORT
    only (legacy has no inverse); 60-bar cooldown, structure reset on fire)."""
    idx = ctx.rth_idx()
    n = len(idx)
    if n < 120: return []
    hi, lo, cl, vo = ctx.h[idx], ctx.l[idx], ctx.c[idx], ctx.v[idx]
    hs, ls = pd.Series(hi), pd.Series(lo)
    is_pk = ((hs == hs.rolling(21, center=True).max()) & (hs > hs.shift(1))).values
    is_tr = ((ls == ls.rolling(21, center=True).min()) & (ls < ls.shift(1))).values
    out = []; peaks = []; troughs = []; cool = 0
    for i in range(10, n):
        if cool > 0: cool -= 1
        ci = i - 10
        if is_pk[ci]: peaks.append(ci)
        if is_tr[ci]: troughs.append(ci)
        if cool > 0: continue
        if len(peaks) >= 3 and len(troughs) >= 2:
            p3, p2, p1 = peaks[-3], peaks[-2], peaks[-1]
            t2, t1 = troughs[-2], troughs[-1]
            if p3 < t2 < p2 < t1 < p1 and hi[p2] > hi[p3] and hi[p2] > hi[p1] \
               and abs(hi[p3] - hi[p1]) < (hi[p2] - hi[p1]) * 0.5:
                v_ls = vo[max(0, p3-2):p3+3].mean()
                v_h = vo[max(0, p2-2):p2+3].mean()
                v_rs = vo[max(0, p1-2):p1+3].mean()
                if v_ls > v_h > v_rs:
                    slope = (lo[t1] - lo[t2]) / (t1 - t2) if t1 > t2 else 0.0
                    neck = lo[t1] + slope * (i - t1)
                    if cl[i-1] >= neck and cl[i] < neck:
                        out.append(ctx.emit(idx[i], False, hi[p2] - neck))
                        cool = COOLDOWN; peaks.clear()
    return out

def gen_fib17(ctx):
    """FIB-17 (batch_b FIB17Detector + ag_deepdive_17_fib.py:256-292 daily wiring:
    prior-14-day ADX(n=7) SMA-approx > 25 gate; swing = prior-10-day H/L extremes;
    trend = prior close vs 10-day close SMA; UP: retrace into [61.8%,50%] of
    low->high = LONG; DOWN mirror = SHORT; one-shot per day)."""
    pdays = ctx.prior_daily
    if len(pdays) < 15: return []
    d14 = pd.DataFrame(pdays[-14:])
    up, dn = d14['high'].diff(), -d14['low'].diff()
    dmp = np.where((up > dn) & (up > 0), up, 0.0)
    dmm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = d14['close'].shift(1)
    tr = pd.concat([d14['high'] - d14['low'], (d14['high'] - pc).abs(),
                    (d14['low'] - pc).abs()], axis=1).max(axis=1)
    trs = tr.rolling(7).mean()
    dip = 100 * pd.Series(dmp).rolling(7).mean() / trs
    dim = 100 * pd.Series(dmm).rolling(7).mean() / trs
    dx = 100 * (dip - dim).abs() / (dip + dim)
    adx = dx.rolling(7).mean().iloc[-1]
    if not np.isfinite(adx) or adx <= 25.0: return []
    d10 = pdays[-10:]
    sw_h = max(x['high'] for x in d10); sw_l = min(x['low'] for x in d10)
    trend_up = pdays[-1]['close'] > np.mean([x['close'] for x in d10])
    rng = sw_h - sw_l
    if trend_up: f50, f618 = sw_h - 0.5 * rng, sw_h - 0.618 * rng
    else: f50, f618 = sw_l + 0.5 * rng, sw_l + 0.618 * rng
    lo_b, hi_b = min(f50, f618), max(f50, f618)
    for i in ctx.rth_idx():
        p = ctx.c[i]
        if trend_up and lo_b <= p <= hi_b: return [ctx.emit(i, True, float(adx))]
        if not trend_up and lo_b <= p <= hi_b: return [ctx.emit(i, False, float(adx))]
    return []

def gen_zone21(ctx):
    """ZONE-21 virgin supply/demand (ag_deepdive_21_zone.py:38-107: 3 consecutive
    tight bars (range < 0.8xATR14 of h-l) + explosion bar (|close-open| > 1.5xATR
    AND vol > 1.5x vol_sma20) forms a zone from the tight bars' extremes; first
    touch = bounce in the explosion's direction; zone consumed (virgin-only);
    zones per-day as legacy)."""
    atr = pd.Series(ctx.h - ctx.l).rolling(14, min_periods=14).mean().values
    vs = pd.Series(ctx.v).rolling(20, min_periods=20).mean().values
    zones = []; out = []
    for i in ctx.rth_idx():
        if i < 21 or not np.isfinite(atr[i]): continue
        t1 = (ctx.h[i-3] - ctx.l[i-3]) < atr[i-3] * 0.8
        t2 = (ctx.h[i-2] - ctx.l[i-2]) < atr[i-2] * 0.8
        t3 = (ctx.h[i-1] - ctx.l[i-1]) < atr[i-1] * 0.8
        if t1 and t2 and t3:
            if abs(ctx.c[i] - ctx.o[i]) > atr[i] * 1.5 and ctx.v[i] > vs[i] * 1.5:
                zh = max(ctx.h[i-3], ctx.h[i-2], ctx.h[i-1])
                zl = min(ctx.l[i-3], ctx.l[i-2], ctx.l[i-1])
                zones.append((ctx.c[i] > ctx.o[i], zh, zl))
        for k, (is_dem, zh, zl) in enumerate(zones):
            if is_dem and ctx.l[i] <= zh:
                zones.pop(k); out.append(ctx.emit(i, True, zh - zl)); break
            if not is_dem and ctx.h[i] >= zl:
                zones.pop(k); out.append(ctx.emit(i, False, zh - zl)); break
    return out


def gen_curve(ctx):
    """CURVE (declared adaptation, Moises doc-075 roster: 'the Curve regression that
    we have'). CAUSAL endpoint variant of the AI labeler's cubic: the labeler fits a
    CENTERED cubic N=20 on 1m closes (ai_labeler_v2.py:4, cubic_utils.py) — hindsight
    by construction. Here: TRAILING 20-bar cubic OLS on 1m closes, slope/curvature
    evaluated at the RIGHT EDGE; fire on edge-slope sign flip (flip to + with curv>0
    = bottom turn = LONG; flip to - with curv<0 = top = SHORT). value = |curvature|."""
    N = 20
    c1 = pd.Series(ctx.c).groupby(np.arange(len(ctx.c)) // BAR_1M).last().values
    if len(c1) < N + 2: return []
    x = np.arange(N, dtype=float) - (N - 1)          # right edge at x=0
    X = np.vstack([x**3, x**2, x, np.ones(N)]).T
    P = np.linalg.pinv(X)
    w_slope, w_curv = P[2], 2 * P[1]
    sw = np.lib.stride_tricks.sliding_window_view(c1, N)
    slope = np.full(len(c1), np.nan); curv = np.full(len(c1), np.nan)
    slope[N-1:] = sw @ w_slope; curv[N-1:] = sw @ w_curv
    out = []
    n5 = len(ctx.c)
    for k in range(N, len(c1)):
        s, sp = slope[k], slope[k-1]
        if not (np.isfinite(s) and np.isfinite(sp)): continue
        if np.sign(s) == np.sign(sp) or s == 0: continue
        i = min((k + 1) * BAR_1M - 1, n5 - 1)        # 5s bar where 1m bar k completes
        if i < ctx.start or not ctx.rth[i]: continue
        if s > 0 and curv[k] > 0: out.append(ctx.emit(i, True, abs(curv[k])))
        elif s < 0 and curv[k] < 0: out.append(ctx.emit(i, False, abs(curv[k])))
    return out


# ---- NMP master equation (Moises 2026-07-16: add NMP + extended NMP) -----------------
# Canonical column L3_1m_z_se_15 (FEATURES_1s_v2 store — the ONLY store carrying the
# window the verified thresholds live on; 5s store is a _30 build, thresholds don't
# transfer across window drift). Thresholds: recalibration verified 2026-06-11.
Z_ENTRY, Z_EXIT = 1.8481, 0.4752   # quantile-matched to V1 |z_21|>2.0 / <0.5
NMP_EPS = 0.1                      # log-floor (research/nmp_state/derive.py:11-14)
NMP_K = 21                         # lambda_hat OLS window, mid of verified K_SWEEP
                                   # (12,21,30); matches the V1 z_21 window heritage

def _nmp_lambda(ctx):
    """lambda_hat per 5s row: trailing OLS slope (k=NMP_K) of log(|z_se|+EPS) over the
    CLOSED-1m sequence (one sample per minute at ts%60==0 — the anchored value there
    reflects the just-closed 1m bar), forward-filled to the 5s grid. Verified math:
    research/nmp_state/derive.py:120-157 (vectorized, identical estimator)."""
    z = ctx.zse
    m_rows = np.flatnonzero(ctx.ts % 60 == 0)
    z1m = z[m_rows]
    lam1m = np.full(len(m_rows), np.nan)
    ok = np.isfinite(z1m)
    logz = np.where(ok, np.log(np.abs(z1m) + NMP_EPS), np.nan)
    if len(logz) >= NMP_K:
        x = np.arange(NMP_K, dtype=float)
        w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()   # OLS slope weights
        sw = np.lib.stride_tricks.sliding_window_view(logz, NMP_K)
        lam1m[NMP_K - 1:] = sw @ w                          # NaN windows -> NaN
    lam = np.full(len(ctx.c), np.nan)
    lam[m_rows] = lam1m
    return pd.Series(lam).ffill().values

def _nmp_fires(ctx):
    """V1 episode semantics: armed while |z|<Z_EXIT has occurred since last fire;
    fire at first RTH bar with |z|>Z_ENTRY while armed. Yields (i, z_i)."""
    z = ctx.zse
    armed = True
    for i in range(len(z)):
        zi = z[i]
        if not np.isfinite(zi): continue
        if abs(zi) < Z_EXIT: armed = True
        elif armed and abs(zi) > Z_ENTRY:
            armed = False
            if ctx.rth[i] and i >= ctx.start:
                yield i, zi

def _vr_1m(ctx):
    """V1 variance_ratio on 1m closes: rolling std(10)/std(60), ddof=1 — the exact
    formula (research/nmp_state/derive.py:67-72; NMP_V2_FEATURE_MAP §1 row 6).
    Computed on clock-aligned 1m buckets, mapped to each 5s row's LAST CLOSED bar."""
    b = ctx.ts // 60
    c1 = pd.Series(ctx.c).groupby(b).last()
    s10 = c1.rolling(10, min_periods=10).std(ddof=1)
    s60 = c1.rolling(60, min_periods=60).std(ddof=1).replace(0, np.nan)
    vr_by_bucket = (s10 / s60)
    closed = pd.Series(b - 1)                     # last CLOSED 1m bucket per row
    return closed.map(vr_by_bucket).values

def gen_nmp(ctx):
    """NMP (V1 trigger CORRECTED per doc 085): |z_se|>Z_ENTRY AND vr_1m<1.0 -> FADE
    (-sign z); re-arm at |z|<Z_EXIT. The vr<1 gate was V1's de-facto stability term
    (NMP_V2_FEATURE_MAP trap #6: without vr you are not running the NMP trigger).
    value=|z|."""
    if getattr(ctx, 'zse', None) is None: return []
    vr = _vr_1m(ctx)
    return [ctx.emit(i, zi < 0, abs(zi)) for i, zi in _nmp_fires(ctx)
            if np.isfinite(vr[i]) and vr[i] < 1.0]

def gen_nmp_lambda(ctx):
    """NMP-LAMBDA (the lambda-complete trigger, NMP_V2_FEATURE_MAP §3 — the
    never-built branch; formerly mislabeled NMP-EXT): |z|>Z_ENTRY AND lambda_hat<0
    -> FADE; lambda_hat>=0 -> RIDE. Skip if lambda_hat undefined. value=|lambda_hat|."""
    if getattr(ctx, 'zse', None) is None: return []
    lam = _nmp_lambda(ctx)
    out = []
    for i, zi in _nmp_fires(ctx):
        if not np.isfinite(lam[i]): continue
        fade = lam[i] < 0
        out.append(ctx.emit(i, (zi < 0) if fade else (zi > 0), abs(lam[i])))
    return out


# ---- NMP TIER LADDER = "extended NMP" (Moises 2026-07-16: "a bunch of augmented
# NMP") — verbatim port of blended_engine_2026_04_18._classify_full_tier (:663-770)
# with every V1 quantity recomputed EXACTLY from raw bars at the decision layer
# (map §4 recipe A — original thresholds therefore transfer; no window drift).
# V1 units: velocity/bar_range in TICKS (0.25). Evaluated at 1m boundaries,
# edge-triggered on (tier, direction) change. REGIME_FLIP excluded (only reachable
# via manual injection in legacy). PEAK disabled in legacy.
TICK = 0.25
_TIER_C = dict(ROCHE=2.0, VR_ENTRY=1.0, VELOCITY_THRESHOLD=50.0,
               FREIGHT_TRAIN_THRESHOLD=100.0, FREIGHT_TRAIN_VR_MAX=0.85,
               WICK_5M_MIN=0.83, WICK_15M_MIN=0.77, H1_Z_MIN=1.0,
               H1_AGAINST_Z_MIN=1.5, MTF_5M_VEL_MIN=30.0, MTF_1M_VEL_ALIVE=10.0,
               MTF_Z_MIN=1.4, MTF_VR_MIN=0.58, MTF_VOL_MIN=2.0)

def _z21(closes):
    """V1 21-bar OLS endpoint z (derive.py:74-95 exact estimator, ddof=2), vectorized."""
    n, w = len(closes), 21
    z = np.full(n, np.nan)
    if n < w: return z
    x = np.arange(w, dtype=float); xm = x.mean(); xv = ((x - xm) ** 2).sum()
    sw = np.lib.stride_tricks.sliding_window_view(closes, w)
    ym = sw.mean(1)
    slope = ((sw - ym[:, None]) * (x - xm)).sum(1) / xv
    inter = ym - slope * xm
    fit = slope[:, None] * x + inter[:, None]
    resid = sw - fit
    var = (resid ** 2).sum(1) / (w - 2)
    sd = np.sqrt(np.maximum(var, 0))
    z[w - 1:] = np.where(sd > 0, (sw[:, -1] - fit[:, -1]) / np.where(sd > 0, sd, 1), np.nan)
    return z

def _tf_state(ctx, period):
    """Clock-aligned TF buckets from the 5s stream. Returns dict of per-bucket
    arrays (V1 formulas) + per-row index of the LAST CLOSED bucket."""
    b = ctx.ts // period
    g = pd.DataFrame({'b': b, 'o': ctx.o, 'h': ctx.h, 'l': ctx.l,
                      'c': ctx.c, 'v': ctx.v}).groupby('b')
    o = g['o'].first(); h = g['h'].max(); l = g['l'].min()
    c = g['c'].last(); v = g['v'].sum()
    ids = c.index.values
    cl = c.values
    vel = np.diff(cl, prepend=np.nan) / TICK                     # V1 velocity (ticks)
    acc = np.diff(vel, prepend=np.nan)
    rng = np.maximum((h - l).values, 1e-9)
    wick = 1.0 - np.abs(c.values - o.values) / rng               # V1 wick_ratio
    s10 = c.rolling(10, min_periods=10).std(ddof=1)
    s60 = c.rolling(60, min_periods=60).std(ddof=1).replace(0, np.nan)
    vr = (s10 / s60).values
    volr = (v / v.rolling(30, min_periods=30).mean()).values     # V1 vol_rel
    # Wilder-14 dmi_diff = DI+ - DI-
    up, dn = h.diff(), -l.diff()
    dmp = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    dmm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    trs = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    dip = 100 * dmp.ewm(alpha=1 / 14, adjust=False).mean() / trs
    dim = 100 * dmm.ewm(alpha=1 / 14, adjust=False).mean() / trs
    dmi = (dip - dim).values
    pos = pd.Series(np.arange(len(ids)), index=ids)              # bucket id -> position
    row_closed = pd.Series(ctx.ts // period - 1).map(pos).values # NaN if not present
    return dict(z=_z21(cl), vel=vel, acc=acc, wick=wick, vr=vr, volr=volr,
                dmi=dmi, row_closed=row_closed)

def _nmp_tier_events(ctx):
    """Classify every RTH 1m boundary with the verbatim V1 ladder; emit on
    (tier, direction) EDGE (documented adaptation: legacy frequency arose from
    position occupancy, a trade-management artifact, not signal definition)."""
    if getattr(ctx, '_nmpt', None) is not None: return ctx._nmpt
    C = _TIER_C
    m1 = _tf_state(ctx, 60); m5 = _tf_state(ctx, 300)
    m15 = _tf_state(ctx, 900); h1 = _tf_state(ctx, 3600)
    events = []
    prev = None
    rows = np.flatnonzero((ctx.ts % 60 == 0) & ctx.rth &
                          (np.arange(len(ctx.c)) >= ctx.start))
    def at(st, i):
        k = st['row_closed'][i]
        return None if not np.isfinite(k) else int(k)
    for i in rows:
        k1, k5, k15, kh = at(m1, i), at(m5, i), at(m15, i), at(h1, i)
        if None in (k1, k5, k15, kh): continue
        z = m1['z'][k1]
        if not np.isfinite(z): continue
        direction = 'short' if z > 0 else 'long'
        wick_5m, wick_15m = m5['wick'][k5], m15['wick'][k15]
        h1_z, h1_vel = h1['z'][kh], h1['vel'][kh]
        velocity, acceleration = m1['vel'][k1], m1['acc'][k1]
        abs_vel = abs(velocity)
        vr, v5_vel, v5_accel = m1['vr'][k1], m5['vel'][k5], m5['acc'][k5]
        dmi, vol_rel = m1['dmi'][k1], m1['volr'][k1]
        z_5m, z_15m = abs(m5['z'][k5]), abs(m15['z'][k15])
        has_wick = wick_5m > C['WICK_5M_MIN'] and wick_15m > C['WICK_15M_MIN']
        h1_against_fade = ((direction == 'long' and h1_z > C['H1_AGAINST_Z_MIN']) or
                           (direction == 'short' and h1_z < -C['H1_AGAINST_Z_MIN']))
        h1_aligned = ((direction == 'long' and h1_z < -C['H1_Z_MIN']) or
                      (direction == 'short' and h1_z > C['H1_Z_MIN']))
        res = None
        if (np.isfinite(vr) and abs_vel >= C['FREIGHT_TRAIN_THRESHOLD'] and
                velocity * acceleration > 0 and vr < C['FREIGHT_TRAIN_VR_MAX']):
            res = ('long' if velocity > 0 else 'short', 'FREIGHT', abs_vel)
        elif has_wick and not h1_aligned:
            res = (direction, 'KILLSHOT', wick_5m)
        elif has_wick and h1_aligned:
            res = (direction, 'CASCADE', wick_5m)
        elif (((direction == 'long' and h1_vel < -3.0) or
               (direction == 'short' and h1_vel > 3.0)) and not h1_against_fade):
            res = ('long' if h1_vel > 0 else 'short', 'RIDEAGN', abs(h1_vel))
        elif h1_against_fade and abs(v5_vel) < 10.0:
            res = (direction, 'FADEAGN', abs(h1_z))
        elif (np.isfinite(vr) and np.isfinite(vol_rel) and v5_accel < 0 and
              abs(v5_vel) > C['MTF_5M_VEL_MIN'] and abs_vel > C['MTF_1M_VEL_ALIVE'] and
              abs(z) > C['MTF_Z_MIN'] and vr > C['MTF_VR_MIN'] and
              vol_rel > C['MTF_VOL_MIN']):
            res = ('long' if v5_vel > 0 else 'short', 'MTFEXH', abs(v5_vel))
        elif z_5m > 1.3 and z_15m > 1.3:
            bdir = 'long' if z > 0 else 'short'
            if (bdir == 'long' and dmi > -5) or (bdir == 'short' and dmi < 5):
                res = (bdir, 'MTFBRK', min(z_5m, z_15m))
        else:
            hi_opp = ((direction == 'long' and v5_vel < -3 and h1_vel < -3) or
                      (direction == 'short' and v5_vel > 3 and h1_vel > 3))
            if not hi_opp:
                res = (direction, 'FADECALM', abs(z))
        key = (res[0], res[1]) if res else None
        if res and key != prev:
            events.append((i, res[0] == 'long', res[1], res[2]))
        prev = key
    ctx._nmpt = events
    return events

def _cdl_flags(ctx):
    """Legacy candlestick cascade on 1m buckets (core/cuda_pattern_detector.py
    @09cd30d8:108-126 verbatim, incl. priority: DOJI first, then HAMMER, then
    ENGULFING). Returns per-bucket flags + the 5s row where each bucket's bar
    CLOSES (first row of the next bucket)."""
    b = ctx.ts // 60
    g = pd.DataFrame({'b': b, 'o': ctx.o, 'h': ctx.h, 'l': ctx.l, 'c': ctx.c}).groupby('b')
    o = g['o'].first().values; h = g['h'].max().values
    l = g['l'].min().values; c = g['c'].last().values
    ids = g['c'].last().index.values
    body = np.abs(c - o)
    rng = np.where(h - l == 0, 1e-10, h - l)
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    po = np.roll(o, 1); pc = np.roll(c, 1); po[0] = np.nan; pc[0] = np.nan
    doji = body / rng < 0.1
    hammer = ~doji & (lower > 2.0 * body) & (upper < 0.1 * rng) & (body < 0.3 * rng)
    ebull = ~doji & ~hammer & (pc < po) & (c > o) & (o <= pc) & (c >= po)
    ebear = ~doji & ~hammer & (pc > po) & (c < o) & (o >= pc) & (c <= po)
    # bucket k's bar closes at the first row of bucket k+1
    first_row = pd.Series(np.arange(len(ctx.ts)), index=b).groupby(level=0).first()
    close_row = pd.Series(first_row.values, index=first_row.index - 1)  # bucket -> closing row
    pos = pd.Series(np.arange(len(ids)), index=ids)
    return dict(hammer=hammer, ebull=ebull, ebear=ebear, body=body, lower=lower,
                ids=ids, close_row=close_row, pos=pos)

def gen_ptrn_engulf(ctx):
    """PTRN-ENGULF (template-era event layer, legacy formula verbatim): 1m engulfing
    bull = LONG / bear = SHORT — direction is IN the formula. value = body (pts)."""
    F = _cdl_flags(ctx)
    out = []
    for k in range(1, len(F['ids'])):
        if not (F['ebull'][k] or F['ebear'][k]): continue
        r = F['close_row'].get(F['ids'][k])
        if r is None or not np.isfinite(r): continue
        r = int(r)
        if r < ctx.start or not ctx.rth[r]: continue
        out.append(ctx.emit(r, bool(F['ebull'][k]), float(F['body'][k])))
    return out

def gen_ptrn_hammer(ctx):
    """PTRN-HAMMER (template-era event layer, legacy formula verbatim; direction =
    classic bullish-reversal reading — DECLARED adaptation: legacy used patterns as
    state flags, direction was learned by the Bayesian brain). value = lower shadow."""
    F = _cdl_flags(ctx)
    out = []
    for k in range(1, len(F['ids'])):
        if not F['hammer'][k]: continue
        r = F['close_row'].get(F['ids'][k])
        if r is None or not np.isfinite(r): continue
        r = int(r)
        if r < ctx.start or not ctx.rth[r]: continue
        out.append(ctx.emit(r, True, float(F['lower'][k])))
    return out

def _make_tier_gen(tier):
    def gen(ctx):
        return [ctx.emit(i, is_long, float(val))
                for i, is_long, t, val in _nmp_tier_events(ctx) if t == tier]
    gen.__doc__ = (f"NMPT-{tier}: V1 tier ladder port (blended_engine_2026_04_18"
                   f"._classify_full_tier verbatim), edge-triggered at 1m boundaries.")
    return gen


GENS = {'ZIGZAG': gen_zigzag, 'ORB-02': gen_orb02, 'SEASON-12': gen_season12,
        'VWAP-03': gen_vwap03, 'OHLC-01': gen_ohlc01, 'PIVOT-16': gen_pivot16,
        'ROUND-05': gen_round05, 'CROSS-11': gen_cross11, 'VWMA-10': gen_vwma10,
        'DOW-19': gen_dow19, 'TUNNEL-20': gen_tunnel20, 'ATR-09': gen_atr09,
        'SAR-23': gen_sar23, 'SQZ-04': gen_sqz04, 'RSI-06': gen_rsi06,
        'MACD-07': gen_macd07, 'SCALP-18': gen_scalp18, 'RENKO-24': gen_renko24,
        'FIB-17': gen_fib17, 'ZONE-21': gen_zone21, 'VP-01': gen_vp01,
        'VA-13': gen_va13, 'HNS-22': gen_hns22, 'CURVE': gen_curve,
        'NMP': gen_nmp, 'NMP-LAMBDA': gen_nmp_lambda,
        'PTRN-ENGULF': gen_ptrn_engulf, 'PTRN-HAMMER': gen_ptrn_hammer}
for _t in ('FREIGHT', 'KILLSHOT', 'CASCADE', 'RIDEAGN', 'FADEAGN',
           'MTFEXH', 'MTFBRK', 'FADECALM'):
    GENS[f'NMPT-{_t}'] = _make_tier_gen(_t)
NMP_STREAMS = {'NMP', 'NMP-LAMBDA'}   # streams needing the canonical z_se load


# ---- batch 3 (2026-07-16, doc-082: turn/exit-timing concepts as causal 1m streams) ---
# Six concepts ported from TURN_CATALOG_DRAFT (TURN-10/07/06) + EXIT_CATALOG_DRAFT
# (EXIT-05/06/04). ALL operate on CLOCK-ALIGNED 1m buckets (ts//60) over the CONTINUOUS
# tail+day stream (no cold start, doc 073); emission = the 5s row where the bucket
# CLOSES (first row of the NEXT bucket, the identical causal knowledge point used by
# _cdl_flags), gated RTH & >=start. Every DECLARED parameter (the catalogs mark these
# [UNSPECIFIED]) is a named module constant with a comment on its origin.
CLIMAX_K = 2.0        # TURN-CLIMAX volume-spike multiple vs rolling median (declared)
CLIMAX_N = 30         # TURN-CLIMAX / EXIT-TIMESTOP fresh-extreme + vol-norm window (declared)
SWEEP_WICK = 0.50     # TURN-SWEEP rejection-wick fraction of bucket range (declared)
ER_N = 10             # CTX-ER efficiency-ratio window (Kaufman KAMA canonical)
ER_CHOP = 0.30        # CTX-ER chop-onset cross level (declared)
KMDR_L_LO, KMDR_L_HI = 28, 22     # EXIT-KMDR EMA lengths (WPI floors/ceilings verbatim)
KMDR_MOM_LO, KMDR_MOM_HI = 6, 11  # EXIT-KMDR momentum lengths (WPI floors/ceilings verbatim)
KMDR_ATR_N, KMDR_ATRS = 14, 1.5   # EXIT-KMDR ATR (Wilder, doc-082) x band multiple (WPI)
TIMESTOP_STALE_S = 20 * 60        # EXIT-TIMESTOP stale window: catalog illustrative 20-min peak


def _min_bkt(ctx):
    """Clock-aligned 1m OHLCV buckets over the full stream + the 5s row where each
    bucket CLOSES (first row of the next bucket = the _cdl_flags causal point).
    Cached on ctx so the six batch-3 generators share one bucketing per day."""
    if getattr(ctx, '_mb', None) is not None:
        return ctx._mb
    b = ctx.ts // 60
    g = pd.DataFrame({'b': b, 'o': ctx.o, 'h': ctx.h, 'l': ctx.l,
                      'c': ctx.c, 'v': ctx.v}).groupby('b')
    ids = g['c'].last().index.values
    first_row = pd.Series(np.arange(len(ctx.ts)), index=b).groupby(level=0).first()
    close_row = pd.Series(first_row.values, index=first_row.index - 1)  # bucket -> close row
    mb = dict(o=g['o'].first().values, h=g['h'].max().values, l=g['l'].min().values,
              c=g['c'].last().values, v=g['v'].sum().values, ids=ids, close_row=close_row)
    ctx._mb = mb
    return mb

def _bkt_row(ctx, mb, k):
    """Causal emission row for bucket position k (RTH & >=start), else None."""
    r = mb['close_row'].get(mb['ids'][k])
    if r is None or not np.isfinite(r):
        return None
    r = int(r)
    if r < ctx.start or not ctx.rth[r]:
        return None
    return r


def gen_turn_ha(ctx):
    """TURN-HA (TURN_CATALOG_DRAFT TURN-10 HeikinAshi_Color_Flip; four HA equations
    verbatim: HA-close=(o+h+l+c)/4; HA-open=avg(prev HA-open,prev HA-close); HA-high/
    -low unused since value=|body| and no wick gate). HA on clock-aligned 1m buckets,
    continuous across the stream, HA-open seeded (o+c)/2 at the first bucket. FIRE at
    the bucket-close row where HA color flips; direction=NEW color (green=LONG/red=
    SHORT); value=|HA body|=|HA-close-HA-open|.
    DECLARED (doc-082): fire on EVERY color flip — the article's small-body + two-sided-
    wick refinement is a precision knob and is NOT applied as a gate; value stays scalar
    (=|HA body|). Latch-removal philosophy of the harness (all fires emitted)."""
    mb = _min_bkt(ctx)
    o, h, l, c = mb['o'], mb['h'], mb['l'], mb['c']
    n = len(c)
    if n < 2:
        return []
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty(n)
    ha_o[0] = (o[0] + c[0]) / 2.0                       # declared seed (doc-082)
    for k in range(1, n):
        ha_o[k] = (ha_o[k - 1] + ha_c[k - 1]) / 2.0     # HA-open recurrence (verbatim)
    green = ha_c > ha_o
    out = []
    for k in range(1, n):
        if green[k] == green[k - 1]:
            continue
        r = _bkt_row(ctx, mb, k)
        if r is None:
            continue
        out.append(ctx.emit(r, bool(green[k]), abs(ha_c[k] - ha_o[k])))
    return out

def gen_turn_sweep(ctx):
    """TURN-SWEEP (TURN_CATALOG_DRAFT TURN-07 Sweep_And_Reclaim / liquidity-trap).
    Levels: prior-day RTH high/low (prior_daily true H/L) + overnight high/low (this
    day file's pre-RTH rows, ~18:00->08:30 CT, fixed at the RTH open). A 1m bucket that
    trades BEYOND a level then CLOSES back inside WITH a rejection wick. FIRE at bucket
    close; direction=AWAY from the swept level (high swept->SHORT, low swept->LONG);
    value=penetration depth (pts).
    DECLARED (catalog marks the wick threshold [UNSPECIFIED]): rejection wick on the
    swept side >= SWEEP_WICK(0.50) x bucket range (standard rejection-candle criterion).
    Highs swept from below (high>L & close<L); lows from above (low<L & close>L). One
    fire per bucket (largest-penetration qualifying level). Order-flow confirmation
    (delta divergence / footprint absorption) is DATA-BLOCKED for the train year, OMITTED."""
    mb = _min_bkt(ctx)
    o, h, l, c = mb['o'], mb['h'], mb['l'], mb['c']
    ar = np.arange(len(ctx.c))
    day = ar >= ctx.start
    rth_idx = np.flatnonzero(ctx.rth & day)
    if len(rth_idx) == 0:
        return []
    first_rth = rth_idx[0]
    on_mask = day & (~ctx.rth) & (ar < first_rth)
    highs, lows = [], []
    if on_mask.any():
        highs.append(float(ctx.h[on_mask].max()))       # overnight high
        lows.append(float(ctx.l[on_mask].min()))        # overnight low
    if ctx.prior_daily:
        d = ctx.prior_daily[-1]
        highs.append(float(d['high']))                  # prior-day RTH high
        lows.append(float(d['low']))                    # prior-day RTH low
    if not highs and not lows:
        return []
    out = []
    for k in range(len(c)):
        rng = h[k] - l[k]
        if rng <= 0:
            continue
        best = None                                      # (is_long, penetration)
        if (h[k] - max(o[k], c[k])) >= SWEEP_WICK * rng:            # upper rejection wick
            pen = max((h[k] - L for L in highs if h[k] > L > c[k]), default=None)
            if pen is not None:
                best = (False, pen)                      # high swept -> SHORT
        if (min(o[k], c[k]) - l[k]) >= SWEEP_WICK * rng:           # lower rejection wick
            pen = max((L - l[k] for L in lows if l[k] < L < c[k]), default=None)
            if pen is not None and (best is None or pen > best[1]):
                best = (True, pen)                       # low swept -> LONG
        if best is None:
            continue
        r = _bkt_row(ctx, mb, k)
        if r is None:
            continue
        out.append(ctx.emit(r, best[0], best[1]))
    return out

def gen_turn_climax(ctx):
    """TURN-CLIMAX (TURN_CATALOG_DRAFT TURN-06 Climax_Volume_Exhaustion_Bar; OHLCV
    shadow of the DATA-BLOCKED footprint/delta exhaustion). 1m bucket with a volume
    spike AT a fresh extreme that CLOSES in the far third AWAY from the extreme. FIRE
    at bucket close; direction=away from the extreme (high->SHORT, low->LONG);
    value=volume ratio.
    DECLARED (catalog marks m/window/close-frac [UNSPECIFIED]): spike = vol >
    CLIMAX_K(2.0) x trailing rolling-CLIMAX_N(30)-bucket MEDIAN volume (session-relative:
    during RTH the trailing window is entirely same-day, overnight-into-RTH); fresh
    extreme = new CLIMAX_N-bucket high/low vs the PRIOR 30 buckets (excl. current);
    rejection = close in the far third of the bucket range (close<=low+range/3 at a
    high; close>=high-range/3 at a low)."""
    mb = _min_bkt(ctx)
    o, h, l, c, v = mb['o'], mb['h'], mb['l'], mb['c'], mb['v']
    hs, ls, vsr = pd.Series(h), pd.Series(l), pd.Series(v)
    prev_max = hs.rolling(CLIMAX_N, min_periods=CLIMAX_N).max().shift(1).values
    prev_min = ls.rolling(CLIMAX_N, min_periods=CLIMAX_N).min().shift(1).values
    vmed = vsr.rolling(CLIMAX_N, min_periods=CLIMAX_N).median().values
    out = []
    for k in range(len(c)):
        rng = h[k] - l[k]
        if rng <= 0 or not np.isfinite(vmed[k]) or vmed[k] <= 0:
            continue
        vr = v[k] / vmed[k]
        if vr <= CLIMAX_K:
            continue
        is_long = None
        if np.isfinite(prev_max[k]) and h[k] > prev_max[k] and c[k] <= l[k] + rng / 3.0:
            is_long = False                              # fresh high, close low third -> SHORT
        elif np.isfinite(prev_min[k]) and l[k] < prev_min[k] and c[k] >= h[k] - rng / 3.0:
            is_long = True                               # fresh low, close high third -> LONG
        if is_long is None:
            continue
        r = _bkt_row(ctx, mb, k)
        if r is None:
            continue
        out.append(ctx.emit(r, is_long, vr))
    return out

def gen_exit_kmdr(ctx):
    """EXIT-KMDR (EXIT_CATALOG_DRAFT EXIT-05 Keltner_Momentum_Decel_Reversal; WPI
    Appendix D params verbatim). Keltner = EMA(price,L) +- 1.5*ATR(Wilder 14) on 1m
    buckets. At the LOWER band with Mom<0 AND Accel<0 -> LONG (reversal up); MIRROR at
    the UPPER band with Mom>0 AND Accel>0 -> SHORT. Asymmetric source lengths:
    LONG/floor EMA L=28, Mom Lmom=6; SHORT/ceiling EMA L=22, Mom Lmom=11.
    Mom=Momentum(price,Lmom)=c[k]-c[k-Lmom]; Accel=Momentum(Mom,1)=Mom[k]-Mom[k-1].
    value=|Mom|. Edge-triggered: fire only when the full band+decel condition goes
    false->true (marks the exhaustion onset once per event).
    DECLARED (doc-082): centerline = EMA(span,adjust=False) and ATR = Wilder-14 (both
    per doc-082, overriding the source's AverageFC-SMA / ATR(L)); WPI params were fit on
    EURUSD 60m and are used here as STRUCTURE, not re-tuned constants."""
    mb = _min_bkt(ctx)
    c = pd.Series(mb['c']); h = pd.Series(mb['h']); l = pd.Series(mb['l'])
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / KMDR_ATR_N, adjust=False).mean().values      # Wilder-14
    cv = c.values
    ema_lo = c.ewm(span=KMDR_L_LO, adjust=False).mean().values
    ema_hi = c.ewm(span=KMDR_L_HI, adjust=False).mean().values
    mom_lo = cv - np.concatenate([np.full(KMDR_MOM_LO, np.nan), cv[:-KMDR_MOM_LO]])
    mom_hi = cv - np.concatenate([np.full(KMDR_MOM_HI, np.nan), cv[:-KMDR_MOM_HI]])
    acc_lo = np.concatenate([[np.nan], np.diff(mom_lo)])
    acc_hi = np.concatenate([[np.nan], np.diff(mom_hi)])
    lower = cv <= ema_lo - KMDR_ATRS * atr
    upper = cv >= ema_hi + KMDR_ATRS * atr
    cond_long = lower & (mom_lo < 0) & (acc_lo < 0)
    cond_short = upper & (mom_hi > 0) & (acc_hi > 0)
    out = []
    for k in range(1, len(cv)):
        fl = bool(cond_long[k]) and not bool(cond_long[k - 1])
        fs = bool(cond_short[k]) and not bool(cond_short[k - 1])
        if not (fl or fs):
            continue
        r = _bkt_row(ctx, mb, k)
        if r is None:
            continue
        if fl:
            out.append(ctx.emit(r, True, abs(mom_lo[k])))
        if fs:
            out.append(ctx.emit(r, False, abs(mom_hi[k])))
    return out

def gen_ctx_er(ctx):
    """CTX-ER (EXIT_CATALOG_DRAFT EXIT-06 Efficiency_Ratio_Chop_Filter, ported as a
    TIMING MARK). ER = |c[k]-c[k-N]| / sum_i|c[i]-c[i-1]| over the last N 1m buckets
    (bounded 0..1). FIRE when ER crosses DOWN through ER_CHOP(0.30) = chop onset.
    *** DECLARED ADAPTATION (PROMINENT) ***: ER is direction-NEUTRAL; the turn
    scorecard needs a direction per fire, so direction = OPPOSITE the sign of the
    last-N-bucket net move (a dying trend marks a potential turn AGAINST it). This is a
    REGIME/CONTEXT stream FORCED directional purely for scoring — read the verdict with
    that caveat. N=ER_N(10) (Kaufman KAMA canonical); threshold ER_CHOP(0.30) declared;
    value = ER at the cross."""
    mb = _min_bkt(ctx)
    c = mb['c']; n = len(c)
    if n < ER_N + 2:
        return []
    dc = np.abs(np.diff(c, prepend=c[0]))
    denom = pd.Series(dc).rolling(ER_N, min_periods=ER_N).sum().values
    net = c - np.concatenate([np.full(ER_N, np.nan), c[:-ER_N]])
    er = np.abs(net) / np.where(denom > 0, denom, np.nan)
    out = []
    for k in range(ER_N + 1, n):
        if not (np.isfinite(er[k]) and np.isfinite(er[k - 1])):
            continue
        if not (er[k] < ER_CHOP and er[k - 1] >= ER_CHOP):     # cross DOWN through 0.30
            continue
        if not np.isfinite(net[k]) or net[k] == 0:
            continue
        r = _bkt_row(ctx, mb, k)
        if r is None:
            continue
        out.append(ctx.emit(r, bool(net[k] < 0), float(er[k])))  # OPPOSITE the dying move
    return out

def gen_exit_timestop(ctx):
    """EXIT-TIMESTOP (EXIT_CATALOG_DRAFT EXIT-04 MFE-duration/time-stop as a turn mark —
    the WEAKEST, MOST-DECLARED port). Track the day's most-recent fresh CLIMAX_N(30)-
    bucket extreme; FIRE when >= TIMESTOP_STALE_S(20 min) elapse since it with no newer
    extreme (the move has gone stale). direction = OPPOSITE the stale move (last extreme
    a high -> SHORT; a low -> LONG). value = minutes since the extreme.
    DECLARED (doc-082): fresh extreme = new 30-bucket high/low vs the prior 30 (excl.
    current); stale window = 20 min (the catalog's ILLUSTRATIVE 20-min peak, NOT measured
    on our own MFE-duration distribution — flagged as unmeasured); edge-triggered at the
    20-min crossing, re-armed by the next fresh extreme; value = staleness minutes.
    NOTE: direction-neutral clock context forced directional for scoring — weakest port."""
    mb = _min_bkt(ctx)
    h, l, ids = mb['h'], mb['l'], mb['ids']
    n = len(h)
    prev_max = pd.Series(h).rolling(CLIMAX_N, min_periods=CLIMAX_N).max().shift(1).values
    prev_min = pd.Series(l).rolling(CLIMAX_N, min_periods=CLIMAX_N).min().shift(1).values
    out = []
    last_ts = None; last_long = None; fired = False
    for k in range(n):
        is_hi = np.isfinite(prev_max[k]) and h[k] > prev_max[k]
        is_lo = np.isfinite(prev_min[k]) and l[k] < prev_min[k]
        if is_hi or is_lo:
            last_ts = int(ids[k]) * 60                  # bucket-start clock
            last_long = bool(is_lo and not is_hi)       # low extreme -> LONG, high -> SHORT
            fired = False
            continue
        if last_ts is None or fired:
            continue
        r = _bkt_row(ctx, mb, k)
        if r is None:
            continue
        elapsed = int(ctx.ts[r]) - last_ts
        if elapsed >= TIMESTOP_STALE_S:
            out.append(ctx.emit(r, last_long, elapsed / 60.0))
            fired = True
    return out


GENS.update({'TURN-HA': gen_turn_ha, 'TURN-SWEEP': gen_turn_sweep,
             'TURN-CLIMAX': gen_turn_climax, 'EXIT-KMDR': gen_exit_kmdr,
             'CTX-ER': gen_ctx_er, 'EXIT-TIMESTOP': gen_exit_timestop})


def _day_profile(prices, volumes):
    """RTH volume profile (ag_deepdive_01_vol_profile.compute_daily_profile verbatim:
    0.25-tick close-binned volume; POC = argmax; VA = 70% expansion around POC)."""
    hi, lo, tv = float(prices.max()), float(prices.min()), float(volumes.sum())
    if tv == 0: return {}
    bins = np.arange(lo, hi + 0.25, 0.25)
    if len(bins) < 2: return dict(poc=lo, vah=hi, val=lo)
    dig = np.clip(np.digitize(prices, bins) - 1, 0, len(bins) - 1)
    vb = np.zeros(len(bins))
    np.add.at(vb, dig, volumes)
    pi = int(np.argmax(vb))
    target = 0.7 * tv; va = vb[pi]; up, dn = pi + 1, pi - 1
    while va < target:
        vu = vb[up] if up < len(bins) else -1
        vd = vb[dn] if dn >= 0 else -1
        if vu == -1 and vd == -1: break
        if vu > vd: va += vu; up += 1
        else: va += vd; dn -= 1
    return dict(poc=float(bins[pi]), vah=float(bins[min(up, len(bins) - 1)]),
                val=float(bins[max(dn, 0)]))


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
        df = pd.read_parquet(p, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        if day.replace('_', '-') in lblf:
            full = pd.concat([tail, df], ignore_index=True) if tail is not None else df
            ctx = DayCtx(full, len(tail) if tail is not None else 0, day, prior_daily)
            if NMP_STREAMS & set(dets):
                zp = os.path.join(ROOT, 'DATA', 'ATLAS', 'FEATURES_1s_v2', 'L3_1m', f'{day}.parquet')
                ctx.zse = None
                if os.path.exists(zp):
                    zf = pd.read_parquet(zp, columns=['timestamp', 'L3_1m_z_se_15'])
                    ctx.zse = pd.Series(full['timestamp']).map(
                        dict(zip(zf['timestamp'].values, zf['L3_1m_z_se_15'].values))).values
            for d in dets:
                rows[d] += GENS[d](ctx)
        # today's TRUE RTH H/L/C for tomorrow's prior-day context (audit-fixed: not close-as-high)
        dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
        m = ((dt.dt.time >= RTH0) & (dt.dt.time <= RTH1)).values
        if m.any():
            entry = dict(high=float(df['high'].values[m].max()),
                         low=float(df['low'].values[m].min()),
                         close=float(df['close'].values[m][-1]))
            entry.update(_day_profile(df['close'].values[m], df['volume'].values[m]))
            prior_daily.append(entry)
            prior_daily = prior_daily[-20:]
        tail = df.tail(TAIL)
    return {d: pd.DataFrame(r) for d, r in rows.items()}, lblf


COLS = ['pivot_age_min', 'sig_with_leg', 'value', 'tod', 'inter']

def day_block_ci(y, days, boots=1000, seed=0):
    """Day-block bootstrap 95% CI on mean(y), vectorized: per-day sums precomputed,
    each resample = gather + divide (identical statistic to concatenating sampled
    days; ~1000x the naive per-day rescan)."""
    uq, inv = np.unique(days, return_inverse=True)
    s = np.zeros(len(uq)); n = np.zeros(len(uq))
    np.add.at(s, inv, y); np.add.at(n, inv, 1)
    idx = np.random.default_rng(seed).integers(0, len(uq), size=(boots, len(uq)))
    means = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def evaluate(det, F, lblf):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if len(F) == 0: return dict(det=det, n=0, note='no signals')
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
    F.to_parquet(os.path.join(REP, f'signal_rows_{det.replace("-", "")}.parquet'))
    if len(F) < 200:
        agr = float(F['y'].mean()) if len(F) else float('nan')
        return dict(det=det, n=len(F), note=f'too few signals (raw agree {agr:.2f})')
    trm, tem = F['year'] == '2024', F['year'] != '2024'
    if trm.sum() < 100 or tem.sum() < 100:
        return dict(det=det, n=len(F), note=f'thin split (raw agree {F["y"].mean():.2f})')
    Xtr, ytr = F.loc[trm, COLS].values, F.loc[trm, 'y'].astype(int).values
    Xte, yte = F.loc[tem, COLS].values, F.loc[tem, 'y'].astype(int).values
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return dict(det=det, n=len(F), note=f'one-class (raw agree {F["y"].mean():.2f})')
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
        lo, hi = day_block_ci(yte[m], days_te[m])
        ter[b] = (int(m.sum()), float(yte[m].mean()), lo, hi)
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
