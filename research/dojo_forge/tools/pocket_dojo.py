"""POCKET DOJO — fogged causal replay over Telegram (owner 2026-07-28:
"an AI companion so I can chat, propose, see what happens, narrate, improve").

The Telegram bot provides the images; the companion conversation happens in the
session thread. This tool is the deterministic engine: state on disk, one
command per invocation, renders the causal chart PNG and (optionally) sends it
via the bot. EVERYTHING (calls, narration, fills, P&L) logs to JSONL — the
owner-narration + chart-state corpus is the distillation deliverable.

Commands (run via: python research/dojo_forge/tools/pocket_dojo.py <cmd> ...):
  new [--day YYYY_MM_DD] [--send]     start a session (random fogged day if no --day)
  step N [--send]                     advance N 1m bars; fills/target/EOD applied
  chart [--send]                      re-render current frame
  sigma W                             set σ sample size (telescope the bands)
  call {long|short} [--target P] [--stop P]   enter at NEXT bar open (causal)
  exit                                flat at next bar open
  note <text...>                      log free-text narration verbatim
  score                               session P&L summary
State: research/dojo_forge/gate_state/pocket_dojo_state.json
Log  : research/dojo_forge/reports/human_dojo/pocket_<day>.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cubic_regression as _cub                      # noqa: E402
from level_coordinate_system import telescope         # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', '..', 'cubic_wick_sensitivity', 'tools'))
try:                                                   # optional -- render
    from wick_series import wick_bias as _wick_bias    # noqa: E402
except Exception:
    _wick_bias = None

REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
# owner 2026-07-29: dojo days come from DATA/ATLAS (not ATLAS_NT8) — this is
# also the atlas_backtest/F-space source range, so future dojo days can carry
# combiner F-space once atlas_backtest is backfilled to match.
DATA = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
STATE = os.path.join(REPO, 'research', 'dojo_forge', 'gate_state', 'pocket_dojo_state.json')
LOGDIR = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'human_dojo')
PNG = os.path.join(LOGDIR, 'pocket_current.png')

VIEW = 45            # bars shown
WARMUP = 90          # starting cursor (frame + cubic + theme warm)
CUBIC_5S_WINDOW = 90  # 90 x 5s = 7.5min -- matches docs/nt8/2-CubicRegressionEndpoint_v1.0-RC.cs
SIGMA_W0 = 20        # default σ sample
FRICTION_PT = 0.89   # round-trip, points
PT_USD = 2.0
WAKEUP_PT_DEFAULT = 0.0    # DISABLED by default (owner 2026-07-30, same day
                           # this was built: "the only way it works is that
                           # it is a set pattern... valid if we look for a
                           # specific setup"). A raw realized-range threshold
                           # is an OUTCOME-conditioned trigger -- it fires
                           # because a move turned out big, which is a
                           # retroactive selection bias (over-samples
                           # decision points right after large moves) AND a
                           # salience cue in its own right (the tool
                           # choosing to interrupt there implicitly flags
                           # "this moment matters" -- close to the same
                           # category of problem the session's no-hints
                           # protocol exists to prevent). The correct kind of
                           # trigger is STRUCTURE-conditioned: a pre-defined,
                           # validated setup/pattern, independent of how big
                           # the aftermath turns out to be -- same
                           # pre-registration discipline as the project's
                           # edge gate. No such validated pattern exists yet
                           # (wick-bias divergence is the nearest candidate
                           # and was itself flagged post-hoc/N=2, not trusted
                           # as a trigger). Mechanism kept, wired to 0/off,
                           # so a real pattern trigger can reuse this exact
                           # early-stop plumbing once one is validated.
RNG_DAY_SEED = None  # wall-clock seed for day pick (variety is the point)


# ---------- state ----------
def _load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return None


def _save(s):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(s, open(STATE, 'w'), indent=1)


SLICE_F = os.path.join(REPO, 'research', 'dojo_forge', 'gate_state',
                       'pocket_slice_counter.txt')
FSPACE_DIR = os.path.join(REPO, 'research', 'nt8_port', 'atlas_backtest')
_fspace_cache = {}


def _fspace_snapshot(day, ts):
    """Causal F-space snapshot at ts (owner 2026-07-29: 'my brain says the
    answer is in F-space, I just don't know how to access it yet'). Attaching
    the combiner state to every slice lets us later correlate the owner's
    tacit calls with the features his pattern engine is actually reading —
    triangulated access instead of conscious access. Last CLOSED row only
    (bar_ts <= ts, within 120s); None outside the RTH window / missing days."""
    if _fspace_cache.get('day') != day:
        p = os.path.join(FSPACE_DIR, f'{day}.parquet')
        _fspace_cache['day'] = day
        _fspace_cache['df'] = pd.read_parquet(p) if os.path.exists(p) else None
    fdf = _fspace_cache['df']
    if fdf is None or len(fdf) == 0:
        return None
    rows = fdf[fdf['bar_ts'] <= ts]
    if len(rows) == 0 or ts - int(rows['bar_ts'].iloc[-1]) > 120:
        return None
    r = rows.iloc[-1]
    fcols = [c for c in fdf.columns if c.startswith('f_')]
    top = sorted(((c, float(r[c])) for c in fcols), key=lambda kv: -abs(kv[1]))[:3]
    return dict(P_topk=round(float(r['P_topk']), 3), P_any=round(float(r['P_any']), 3),
                gov_dir=int(r['gov_dir']), gov_stream=str(r['gov_stream']),
                n_fires=int(r['n_fires_topk']), zz_leg=float(r['zz_leg']),
                zz_confirm=int(r['zz_confirm']), zz_age_min=float(r['zz_pivot_age_min']),
                top_streams={k: round(v, 3) for k, v in top})


def _next_slice():
    """Global monotonic decision-point counter (owner: 'number the slices so we
    can backtrack'). Increments on `new` and each `step`; every event logged in
    between inherits the current slice, so 'S38' resolves to an exact
    day+bar+context in the corpus."""
    try:
        n = int(open(SLICE_F).read().strip())
    except Exception:
        n = 0
    n += 1
    open(SLICE_F, 'w').write(str(n))
    return n


V2F_DIR = os.path.join(REPO, 'DATA', 'ATLAS', 'FEATURES_5s_v2')
# owner-language F-space picks (NOT the combiner — that was "gibberish" to the
# owner; the ACTUAL F-space is the V2 statistical layer families)
V2F_LAYERS = {
    'L4_1m': ['L4_1m_z_21', 'L4_1m_vr_exact', 'L4_1m_lambda_t_21'],
    'L5_1m': ['L5_1m_ldist_skew', 'L5_1m_ldist_kurtosis', 'L5_1m_ldist_outlier_pct'],
    'L5_5m': ['L5_5m_ldist_skew', 'L5_5m_ldist_kurtosis'],
    'L2_1m': ['L2_1m_price_velocity_30', 'L2_1m_price_accel_30', 'L2_1m_price_sigma_30'],
}
_v2f_cache = {}


def _v2_fspace(day, ts):
    """Causal V2 F-space snapshot at ts — the layer-family features in the
    owner's own vocabulary (z, variance-ratio, lambda pull, bar-internals
    skew/kurtosis, velocity/accel/sigma). None if the day isn't built."""
    if _v2f_cache.get('day') != day:
        _v2f_cache.clear(); _v2f_cache['day'] = day
        merged = None
        for layer, cols in V2F_LAYERS.items():
            p = os.path.join(V2F_DIR, layer, f'{day}.parquet')
            if not os.path.exists(p):
                merged = None; break
            d = pd.read_parquet(p, columns=['timestamp'] + cols)
            merged = d if merged is None else merged.merge(d, on='timestamp', how='inner')
        _v2f_cache['df'] = merged.sort_values('timestamp').reset_index(drop=True) if merged is not None else None
    fdf = _v2f_cache['df']
    if fdf is None or len(fdf) == 0:
        return None
    rows = fdf[fdf['timestamp'] <= ts]
    if len(rows) == 0:
        return None
    r = rows.iloc[-1]
    return {c: (None if pd.isna(r[c]) else round(float(r[c]), 3))
            for cols in V2F_LAYERS.values() for c in cols}


def _fs_line(s, df):
    """Compact ACTUAL-F-space caption line for phone frames ('peppering' —
    ambient exposure for the owner's background pattern engine)."""
    v = _v2_fspace(s['day'], int(df['timestamp'].iloc[s['cur']]))
    if v is None:
        return None
    def g(k, f='{:+.2f}'):
        x = v.get(k)
        return f.format(x) if x is not None else '·'
    return (f"F: z{g('L4_1m_z_21')} vr{g('L4_1m_vr_exact','{:.2f}')} "
            f"λt{g('L4_1m_lambda_t_21')} | 1m sk{g('L5_1m_ldist_skew')} "
            f"ku{g('L5_1m_ldist_kurtosis')} | 5m sk{g('L5_5m_ldist_skew')} "
            f"ku{g('L5_5m_ldist_kurtosis')} | v{g('L2_1m_price_velocity_30')} "
            f"a{g('L2_1m_price_accel_30')} σ{g('L2_1m_price_sigma_30','{:.1f}')}")


# ---- live oscillation watcher (owner 2026-08-01: "let's try it out, as a
# live watcher ... we will advance in 1s increments, we will first observe the
# first 2 oslialation"). Mirrors oscillation_harvest_test.py EXACTLY so what
# he watches on the chart is the same object the 55k-trade study measured --
# same band, same lookback, same edge-triggered traverse rule. Any drift
# between the two would make the live read untestable.
OSC_BAND = 1.5          # sigma band defining an extreme  (== test's BAND)
OSC_LOOKBACK_S = 1800   # 30min traverse-count window     (== test's LOOKBACK_S)
# BASIS (owner 2026-08-01: "we need to switch to 1s watcher ... this is to
# develop the precursor so step at 1s slices"). 1s x450 = 7.5min is the DEPLOYED
# NT8 spec (docs/nt8/2-CubicRegressionEndpoint_v1.0-RC.cs); the previous 5s x90
# matched it by time span but not by basis, so the watcher was deciding on an
# instrument that never ships. 1s bars also remove the halt-straddle lookahead
# entirely -- a 1s bar cannot span the cutoff.
OSC_BASIS = '1s'
CUBIC_1S_WINDOW = 450   # 450 x 1s = 7.5min, deployed spec
OSC_HIST_S = 2400       # trailing seconds loaded: >= cubic(450) + sigma(1200)


# ---- reference REGIONS (owner 2026-08-01: "reference are regions, how large
# the region is should be based on probability of being near"). A level drawn
# as a hairline implies a precision the tape does not have. The honest width is
# where price has actually SPENT TIME around it: build a density of 1s closes
# within +-REGION_SEARCH_PT of the level over a trailing window and take the
# central REGION_MASS of that density. Wide region = price loiters there and
# the level is soft; narrow = it is a genuine edge.
REGION_SEARCH_PT = 15.0     # only observations this close count toward a level
REGION_MASS = 0.68          # central mass of the local density to enclose
REGION_LOOKBACK_S = 3600    # trailing seconds of 1s tape used to build it


def _level_region(s, level, cut, lookback=REGION_LOOKBACK_S):
    """(lo, hi, n) for a reference level, or None when too few observations.

    Purely descriptive and strictly causal -- it reads only tape at or before
    `cut`. It does NOT predict; it reports how tightly price has been packed
    around that level so far."""
    d1 = _bars_tele(s['day'], '1s')
    if d1 is None:
        return None
    w = d1[(d1['timestamp'] <= cut) & (d1['timestamp'] > cut - lookback)]
    if not len(w):
        return None
    c = w['close'].to_numpy()
    near = c[np.abs(c - level) <= REGION_SEARCH_PT]
    if len(near) < 60:
        return None
    # CENTRED on the level. Taking quantiles of the raw prices instead produced
    # regions that did not even CONTAIN their own level (19689.50 -> 19676.50-
    # 19685.75), because the nearby density sat mostly below it. The width that
    # answers "how likely is price to be near this level" is the quantile of
    # the ABSOLUTE DISTANCE from it.
    hw = float(np.quantile(np.abs(near - level), REGION_MASS))
    skew = float(np.mean(near) - level)     # +ve: density sits ABOVE the level
    return level - hw, level + hw, len(near), hw, skew


def _osc_state(s, df):
    """(z_now, band_pt, K, traverses, last_side) at the causal cutoff, or None.

    Runs on the DEPLOYED basis: cubic endpoint over CUBIC_1S_WINDOW 1s bars
    (7.5min), residual sigma over SIGMA_MIN minutes. K counts COMPLETED
    traverses in the trailing 30min. Strictly causal -- and at 1s there is no
    straddle bar, so `<= cut` is exact rather than leaking up to 4s of future
    the way the 5s series did."""
    d1 = _bars_tele(s['day'], OSC_BASIS)
    if d1 is None:
        return None
    cur = s['cur']
    cut = int(s.get('halt_ts5')
              or (int(df['timestamp'].iloc[cur]) + 55 + s.get('peek_offset', 0)))
    sub = d1[(d1['timestamp'] <= cut) & (d1['timestamp'] > cut - OSC_HIST_S)]
    if len(sub) < CUBIC_1S_WINDOW + 300:
        return None
    c = sub['close'].to_numpy()
    cub, _, _ = _cub.rolling(c, CUBIC_1S_WINDOW, 1)
    res = c - cub
    sig = pd.Series(res).rolling(s.get('sigma_w', SIGMA_W0) * 60,
                                 min_periods=5 * 60).std().to_numpy()
    z = np.where(sig > 0, res / sig, np.nan)
    ts = sub['timestamp'].to_numpy()
    side = np.where(z >= OSC_BAND, 1, np.where(z <= -OSC_BAND, -1, 0))
    ff = pd.Series(np.where(side == 0, np.nan, side)).ffill().to_numpy()
    flip = np.flatnonzero((~np.isnan(ff[1:])) & (~np.isnan(ff[:-1]))
                          & (ff[1:] != ff[:-1])) + 1
    K = int((ts[flip] >= cut - OSC_LOOKBACK_S).sum()) if len(flip) else 0
    last = None if not len(flip) else (int(ts[flip[-1]]), int(ff[flip[-1]]))
    return (float(z[-1]),
            float(OSC_BAND * sig[-1]) if np.isfinite(sig[-1]) else float('nan'),
            K, [int(t) for t in ts[flip]], last)


def _osc_line(s, df):
    """One-line live watcher readout, appended to every step caption."""
    st = _osc_state(s, df)
    if st is None:
        return None
    z, band_pt, K, flips, last = st
    # study P(complete) by K — so the live number carries its own base rate
    tbl = {0: 63.6, 1: 66.3, 2: 70.9, 3: 75.0, 4: 78.6}
    p = tbl.get(K, 78.5 if K >= 5 else 63.6)
    ago = ''
    if last:
        cut = int(s.get('halt_ts5')
                  or (int(df['timestamp'].iloc[s['cur']]) + 55
                      + s.get('peek_offset', 0)))
        ago = (f" | last traverse {(cut - last[0]) / 60:.1f}m ago -> "
               f"{'HIGH' if last[1] > 0 else 'LOW'} band")
    zone = ('AT HIGH BAND' if z >= OSC_BAND else
            'AT LOW BAND' if z <= -OSC_BAND else 'mid')
    return (f"OSC[{OSC_BASIS}x{CUBIC_1S_WINDOW}]: z{z:+.2f} [{zone}] "
            f"| band ±{OSC_BAND:g}σ = ±{band_pt:.1f}pt "
            f"| K={K} in 30m (study P(next completes)={p:.1f}%){ago}")


def _peek_1s(s, df):
    """Advance the frame exactly ONE second, rolling over into a real 1m
    _advance (correct stop/target fills) whenever the offset completes a
    minute. Same contract as `peek`, factored out for the live watcher.

    MUST honour an active halt truncation the same way `peek` does. Without
    this the frame cutoff stays PINNED at halt_ts5 while peek_offset runs
    ahead: `watch 300` silently committed five real minutes of bars while the
    display froze on one instant (owner 2026-08-01, watching for an exit --
    he would have been blind through the whole window)."""
    halt = s.get('halt_ts5')
    if halt:
        bar_end = int(df['timestamp'].iloc[s['cur']]) + 55
        if int(halt) + 1 < bar_end:
            s['halt_ts5'] = int(halt) + 1
            return []
        s.pop('halt_ts5', None)          # bar fully revealed; resume normal peek
        # engine halts stamp :56-:59 (past the +55 display max) — carry the
        # consumed edge into peek_offset so the reveal never rewinds onto
        # engine-consumed seconds (audit 3c)
        if int(halt) > bar_end:
            s['peek_offset'] = max(s.get('peek_offset', 0),
                                   int(halt) - bar_end)
    total = s.get('peek_offset', 0) + 1
    whole, s['peek_offset'] = divmod(total, 60)
    ev = []
    if whole:
        s['slice'] = _next_slice()
        ev = _advance(s, df, whole)
    ts_c = int(df['timestamp'].iloc[s['cur']]) + 55
    sub = _check_5s_fill(s, ts_c, ts_c + s['peek_offset'])
    if sub:
        ev.append(sub)
    return ev


def _bar_line(s, df):
    """Last CLOSED 1m bar + the partial one, as rendered (owner 2026-08-01:
    "Correction the current bar is red with a long down wick (1m bar)").

    I had been quoting a ROLLING 60s window ending at the frame cutoff, which
    straddles the minute boundary and can print GREEN while the rendered 1m
    candle is RED — it scored his shape calls against an object he was not
    looking at. The bar he sees is the only one that counts."""
    d5 = _bars_tele(s['day'], '5s')
    if d5 is None:
        return None
    cur = s['cur']
    cut = int(s.get('halt_ts5')
              or (int(df['timestamp'].iloc[cur]) + 55 + s.get('peek_offset', 0)))
    out = []
    for b, tag in ((cur, 'bar'), (cur + 1, 'partial')):
        if b >= len(df):
            continue
        t0 = int(df['timestamp'].iloc[b])
        if t0 > cut:
            continue
        w = d5[(d5['timestamp'] >= t0) & (d5['timestamp'] <= min(t0 + 59, cut))]
        if not len(w):
            continue
        o = float(w['open'].iloc[0]); h = float(w['high'].max())
        l = float(w['low'].min()); c = float(w['close'].iloc[-1])
        uw, lw = h - max(o, c), min(o, c) - l
        et = pd.to_datetime(t0, unit='s', utc=True).tz_convert('America/New_York')
        out.append(f"{tag} {et:%H:%M} O{o:.2f} H{h:.2f} L{l:.2f} C{c:.2f} "
                   f"{'GRN' if c > o else 'RED'} rng{h - l:.1f} "
                   f"uw{uw:.1f}/lw{lw:.1f}->{'LOW' if lw > uw else 'UPP'}")
    return ' | '.join(out) if out else None


def _log(s, event, **kw):
    os.makedirs(LOGDIR, exist_ok=True)
    rec = dict(wall=time.strftime('%Y-%m-%dT%H:%M:%S'), day=s['day'],
               bar=s['cur'], event=event, who=s.get('who', 'owner'),
               slice=s.get('slice'))
    rec.update(kw)          # kw may override (e.g. per-event who= for a claude
                            # THESIS logged mid-owner-session)
    with open(os.path.join(LOGDIR, f"pocket_{s['day']}.jsonl"), 'a') as f:
        f.write(json.dumps(rec) + '\n')
    try:                          # SQL write-through (JSONL stays source of truth)
        import pocket_dojo_db
        pocket_dojo_db.write_event(rec)
    except Exception:
        pass                      # never let corpus indexing break the dojo


def _bars(day):
    df = pd.read_parquet(os.path.join(DATA, f'{day}.parquet'))[
        ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df.reset_index(drop=True)


LIQ_FLOOR = 400      # 1m-volume floor for session START bars (median across the
                     # book; owner trades "whenever free" so sampling is any-hour,
                     # but dead overnight tape (03-07h UTC ~150/bar) makes even
                     # the candles unreliable — 2025_12_19 session had 1-tick
                     # bars. Rolling-30 median volume must clear this.)
LIQ_WINDOW = 30      # trailing bars for the liquidity check
MIN_PLAY_BARS = 180  # a session start leaves at least this many bars of day


def _pick_start(df, rng):
    """Random start bar wherever the tape is ALIVE (any hour, liquidity-gated)."""
    v = pd.Series(df['volume'].to_numpy(float))
    liq = v.rolling(LIQ_WINDOW, min_periods=LIQ_WINDOW).median().to_numpy()
    hi = len(df) - MIN_PLAY_BARS
    elig = [i for i in range(WARMUP, max(WARMUP + 1, hi))
            if np.isfinite(liq[i]) and liq[i] >= LIQ_FLOOR]
    if not elig:                       # fully dead day: start at its liveliest spot
        i = int(np.nanargmax(liq[WARMUP:hi])) + WARMUP if hi > WARMUP else WARMUP
        return i
    return int(rng.choice(elig))


LEVEL_TRAIL_DAYS = 5   # prior sessions to load for the telescope's day/4h scales
                       # (single-day loading was a bug: 'day' scale saw only
                       # today's bars, so a real swing level like yesterday's
                       # high never appeared in the close-up frame)


_sref_cache = {}


def _session_refs(day, df, cur):
    """Causal look-left session references (owner 2026-07-31): prior-day
    high/low/close, overnight (pre-09:30 ET) high/low, today's open, and the
    DEVELOPING day high/low up to cur. Always exist, even in fresh territory
    where the density telescope has no pivots to draw."""
    refs = []
    if day not in _sref_cache:
        static = []
        days = sorted(f[:-8] for f in os.listdir(DATA) if f.endswith('.parquet'))
        if day in days and days.index(day) > 0:
            pd_df = _bars(days[days.index(day) - 1])
            static += [(float(pd_df['high'].max()), 'pdH'),
                       (float(pd_df['low'].min()), 'pdL'),
                       (float(pd_df['close'].iloc[-1]), 'pdC')]
        et = (pd.to_datetime(df['timestamp'], unit='s', utc=True)
              .dt.tz_convert('America/New_York'))
        mins = (et.dt.hour * 60 + et.dt.minute).to_numpy()
        # NOT monotonic: a UTC day file wraps ET 20:00(prev)->19:59, so find
        # the first bar inside [09:30, 20:00) ET rather than searchsorted.
        rth = (mins >= 570) & (mins < 1200)
        o930 = int(np.argmax(rth)) if rth.any() else len(mins)
        _sref_cache[day] = (static, o930)
    static, o930 = _sref_cache[day]
    refs += static
    if 0 < o930 <= cur:
        refs += [(float(df['high'].iloc[:o930].max()), 'onH'),
                 (float(df['low'].iloc[:o930].min()), 'onL')]
    if o930 <= cur:
        refs.append((float(df['open'].iloc[o930]), 'open'))
    refs += [(float(df['high'].iloc[:cur + 1].max()), 'dH'),
             (float(df['low'].iloc[:cur + 1].min()), 'dL')]
    # dedupe near-identical prices (keep first label), 2pt tolerance
    out = []
    for p_, lab in refs:
        if not any(abs(p_ - q) < 2.0 for q, _ in out):
            out.append((p_, lab))
    return out


def _bars_with_history(day):
    """(extended_df, offset) — offset trailing days + today, concatenated, so
    the level telescope's day/4h scales have real multi-day context. `offset`
    is how many rows precede today's bar0 in the extended frame; add it to a
    single-day cursor to index into the extended frame."""
    days = sorted(f[:-8] for f in os.listdir(DATA) if f.endswith('.parquet'))
    i = days.index(day)
    trail = days[max(0, i - LEVEL_TRAIL_DAYS):i]
    frames = [_bars(d) for d in trail] + [_bars(day)]
    offset = sum(len(f) for f in frames[:-1])
    ext = pd.concat(frames, ignore_index=True)
    return ext, offset


# ---------- fills ----------
def _close_and_maybe_reverse(s, p, px, why, d, log_suffix=''):
    """Close the position at px, with two owner-armed cascading behaviors
    (2026-07-30):
    - STOP + stop_reverse: flip to the OPPOSITE direction at the trigger price
      (adverse move -> catch the reversal as a new trade instead of just a
      loss).
    - TARGET + bank_reenter: bank the profit, then immediately RE-ENTER the
      SAME direction at the trigger price with a clean slate (favorable move
      keeps running -> lock the gain, keep riding, fresh stop/target owner
      sets next). Never crossed (a STOP never reenters-same, a TARGET never
      reverses) — each flag only fires on its own trigger type.
    Returns the event string."""
    pts = (px - p['entry']) * d - FRICTION_PT
    s['pnl_pts'] = s.get('pnl_pts', 0.0) + pts
    s['trades'] = s.get('trades', 0) + 1
    _log(s, 'close', reason=why.lower() + log_suffix, price=px, pts=round(pts, 2))
    ev = f"{why} {p['dir']} @ {px:.2f} -> {pts:+.2f}pt (${pts*PT_USD:+.2f})"
    if why == 'STOP' and p.get('stop_reverse'):
        new_dir = 'short' if p['dir'] == 'long' else 'long'
        s['pos'] = dict(dir=new_dir, pending=False, entry=px, entry_bar=s['cur'],
                        target=None, stop=None)
        _log(s, 'fill', dir=new_dir, price=px, reason='stop_reverse')
        ev += f" -> REVERSED to {new_dir} @ {px:.2f}"
    elif why == 'TARGET' and p.get('bank_reenter'):
        s['pos'] = dict(dir=p['dir'], pending=False, entry=px, entry_bar=s['cur'],
                        target=None, stop=None)
        _log(s, 'fill', dir=p['dir'], price=px, reason='bank_reenter')
        ev += f" -> BANKED, REENTERED {p['dir']} @ {px:.2f}"
    else:
        s['pos'] = None
    return ev


def _check_5s_fill(s, ts_from, ts_to):
    """Check the OPEN position against real 5s bars in (ts_from, ts_to] --
    catches stop/target hits inside a PEEKED-but-not-yet-committed partial 1m
    bar (owner 2026-07-30: caught a stop touch at 14:40:20 that a whole-minute
    -only check would have missed until the full bar rolled over). Closes at
    the actual 5s bar's trigger price if hit. Returns an event string or None."""
    p = s.get('pos')
    if not p or p.get('pending') or (p.get('target') is None and p.get('stop') is None):
        return None
    d5 = _bars_tele(s['day'], '5s')
    if d5 is None:
        return None
    win = d5[(d5['timestamp'] > ts_from) & (d5['timestamp'] <= ts_to)]
    if win.empty:
        return None
    d = 1 if p['dir'] == 'long' else -1
    tgt, stp = p.get('target'), p.get('stop')
    for _, r in win.iterrows():          # causal order — first trigger wins
        hit_t = tgt is not None and ((d > 0 and r.high >= tgt) or (d < 0 and r.low <= tgt))
        hit_s = stp is not None and ((d > 0 and r.low <= stp) or (d < 0 and r.high >= stp))
        if hit_t or hit_s:
            px = float(tgt if hit_t else stp)
            why = 'TARGET' if hit_t else 'STOP'
            ev = _close_and_maybe_reverse(s, p, px, why, d, log_suffix='_5s')
            return ev + f" [5s @ {pd.to_datetime(int(r.timestamp), unit='s'):%H:%M:%S}]"
    return None


def _advance(s, df, n, stop_on_fill=False):
    """Step n bars applying open-position logic causally. Returns event strings.

    stop_on_fill (owner 2026-07-30, after a real overshoot): stop the advance
    the moment a target/stop fills, leaving `cur` ON the fill bar instead of
    running out the remaining n. Without it, "advance to the next trigger"
    silently blew 9 bars PAST the trigger with a freshly-reversed naked
    position riding the whole way -- the fill is the decision point, so
    stopping there is the whole point of the instruction."""
    ev = []
    o = df['open'].to_numpy(); h = df['high'].to_numpy(); l = df['low'].to_numpy()
    end = min(s['cur'] + n, len(df) - 1)
    i = s['cur']
    start_i = i
    wk = s.get('wakeup_pt', WAKEUP_PT_DEFAULT)
    run_hi = run_lo = None        # seeded on the first NEWLY-advanced bar
                                   # only (NOT the starting bar -- that one
                                   # was already shown before this step was
                                   # even issued, catching its own range
                                   # would just flag something the owner
                                   # already saw, not a move during the step)
    stopped_early = False        # set by EITHER stop_on_fill or the wakeup;
                                  # both mean "cur is already where we want it,
                                  # don't fast-forward to `end` below"
    while i < end:
        i += 1
        s['cur'] = i                 # keep cur live INSIDE the loop -- a stop
                                      # reversal needs the correct entry_bar,
                                      # not the stale pre-loop cur (real bug
                                      # caught while wiring stop-and-reverse)
        p = s.get('pos')
        if p:                        # entries/exits fill immediately (call/exit
                                      # commands) — only target/stop hits happen
                                      # while stepping through open bars
            d = 1 if p['dir'] == 'long' else -1
            tgt, stp = p.get('target'), p.get('stop')
            hit_t = tgt is not None and ((d > 0 and h[i] >= tgt) or (d < 0 and l[i] <= tgt))
            hit_s = stp is not None and ((d > 0 and l[i] <= stp) or (d < 0 and h[i] >= stp))
            if hit_t or hit_s:
                px = float(tgt if hit_t else stp)
                why = 'TARGET' if hit_t else 'STOP'
                ev.append(_close_and_maybe_reverse(s, p, px, why, d))
                if stop_on_fill:
                    stopped_early = True
                    ev.append(f"(stopped AT the fill -- {i - start_i}/{n} bars advanced)")
                    break
        # WAKEUP (owner 2026-07-30: "like a wakeup timer?" -- a big multi-bar
        # step can silently blow through a real move with only start/end
        # ever shown; see WAKEUP_PT_DEFAULT). Track the running high/low
        # SEEN DURING THIS CALL; if the range crosses the threshold, stop
        # the advance early instead of completing all n bars -- a genuine
        # interrupt, not just a note added after the fact.
        run_hi = h[i] if run_hi is None else max(run_hi, h[i])
        run_lo = l[i] if run_lo is None else min(run_lo, l[i])
        if wk and (run_hi - run_lo) >= wk and i < end:
            stopped_early = True
            ev.append(f"WAKEUP: {run_hi - run_lo:.1f}pt range by bar {i} "
                      f"(stopped early -- {i - start_i}/{n} bars advanced)")
            break
    if not stopped_early:
        s['cur'] = end
    # NOTE: checked against s['cur'] (the REAL stopping point), not the
    # pre-computed `end` -- if a wakeup fired early, `end` can still equal
    # len(df)-1 (that was the original request) even though we stopped
    # well short of it, which would have wrongly forced an EOD close.
    if s['cur'] >= len(df) - 1 and s.get('pos') and not s['pos'].get('pending'):
        p = s['pos']; d = 1 if p['dir'] == 'long' else -1
        px = float(df['close'].iloc[-1])
        pts = (px - p['entry']) * d - FRICTION_PT
        s['pnl_pts'] = s.get('pnl_pts', 0.0) + pts; s['trades'] = s.get('trades', 0) + 1
        ev.append(f"EOD flat @ {px:.2f} -> {pts:+.2f}pt")
        _log(s, 'close', reason='eod', price=px, pts=round(pts, 2))
        s['pos'] = None
    return ev


# ---------- render ----------
_hist_cache = {}
_tele_cache = {}
TELE_BARS = 72            # telescopic sub-panel: bar count shown (span scales with res)
TELE_RES_DEFAULT = '5s'   # owner 2026-07-30: "start looking at 30s windows, even 15s,
                          # with the 1m in view" -- switchable telescope resolution


def _bars_tele(day, res):
    """Load the telescope sub-panel's OHLC at the given resolution, stitching
    up to LEVEL_TRAIL_DAYS trailing days + today (owner 2026-07-30: "go to 1h
    TF and let's look back the past days" -- a single day's 1h file is only
    ~23 bars, nowhere near enough. Mirrors _bars_with_history's trailing-day
    stitch, generalized by resolution. No behavior change for 5s/15s/30s:
    those already have far more same-day bars than TELE_BARS ever requests,
    so .tail(n_bars) downstream still only ever pulls from today for them."""
    key = (day, res)
    if _tele_cache.get('key') != key:
        d = os.path.join(REPO, 'DATA', 'ATLAS', res)
        _tele_cache['key'] = key
        if os.path.isdir(d):
            days = sorted(f[:-8] for f in os.listdir(d) if f.endswith('.parquet'))
            trail = days[max(0, days.index(day) - LEVEL_TRAIL_DAYS):days.index(day)] if day in days else []
            frames = []
            for dd in trail + ([day] if day in days else []):
                p = os.path.join(d, f'{dd}.parquet')
                frames.append(pd.read_parquet(p)[['timestamp', 'open', 'high', 'low', 'close']])
            _tele_cache['df'] = (pd.concat(frames, ignore_index=True)
                                 .sort_values('timestamp').reset_index(drop=True)
                                 if frames else None)
        else:
            _tele_cache['df'] = None
    return _tele_cache['df']


def _draw_tele_panel(ax, s, res, n_bars, ts_cutoff, tele_lines, p, title_prefix='',
                     grid_s=60, t_window=None):
    """Draw ONE telescope sub-panel (candles + level overlay + entry/target/
    stop + minute gridlines) at the given resolution/bar-count onto `ax`.
    Factored out so the switchable panel and the fixed always-5s/3min panel
    (owner 2026-07-30) share identical drawing logic."""
    dtele = _bars_tele(s['day'], res)
    sub = dtele[dtele['timestamp'] <= ts_cutoff].tail(n_bars) if dtele is not None else None
    if sub is None or len(sub) < 4:
        ax.text(0.5, 0.5, f'{res} telescope: no intrabar data here',
               transform=ax.transAxes, ha='center', fontsize=8, color='gray')
        ax.set_xticks([]); ax.set_yticks([])
        return
    x5 = np.arange(len(sub))
    o5 = sub['open'].to_numpy(); h5 = sub['high'].to_numpy()
    l5 = sub['low'].to_numpy(); c5 = sub['close'].to_numpy()
    y5lo, y5hi = l5.min(), h5.max(); pad5 = max(1.0, (y5hi - y5lo) * 0.1)
    # PRICE LABELS on every reference line (owner 2026-08-01: "on the 5s and 1s
    # telescope on the ref lines can you add the price level"). The lines were
    # drawn unlabelled, so a level could be SEEN but not READ -- you had to
    # cross-reference the main panel to find out what you were looking at.
    def _tag(v, colr, weight='normal'):
        ax.text(len(sub) - 0.5, v, f' {v:.2f}', color=colr, fontsize=6.5,
                va='center', ha='right', zorder=9, fontweight=weight,
                clip_on=True,
                bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none',
                          alpha=0.65))
    for lp, colr, lw in tele_lines:
        if y5lo - pad5 <= lp <= y5hi + pad5:
            ax.axhline(lp, color=colr, lw=lw, alpha=0.55, zorder=1)
            _tag(lp, colr)
    for lp in s.get('owner_lines', []):
        if y5lo - pad5 <= lp <= y5hi + pad5:
            reg = _level_region(s, lp, ts_cutoff)
            if reg:
                lo_, hi_ = reg[0], reg[1]
                ax.axhspan(lo_, hi_, color='#6A1B9A', alpha=0.12, zorder=0.8)
                ax.axhline(lo_, color='#6A1B9A', lw=0.6, alpha=0.5, zorder=1.2)
                ax.axhline(hi_, color='#6A1B9A', lw=0.6, alpha=0.5, zorder=1.2)
            ax.axhline(lp, color='#6A1B9A', lw=1.4, alpha=0.75, zorder=1.5)
            _tag(lp, '#6A1B9A', 'bold')
    if p and not p.get('pending'):
        for lvl, colr, ls_ in ((p['entry'], '#000', '--'),
                               (p.get('target'), '#1B5E20', '--'),
                               (p.get('stop'), '#B71C1C', '--')):
            if lvl is not None and y5lo - pad5 <= lvl <= y5hi + pad5:
                ax.axhline(lvl, color=colr, lw=1.2, ls=ls_, alpha=0.85, zorder=5)
                _tag(lvl, colr, 'bold')
    up5 = c5 >= o5
    ax.vlines(x5, l5, h5, color=np.where(up5, '#2E7D32', '#C62828'), lw=0.8, zorder=3)
    ax.bar(x5, np.abs(c5 - o5), bottom=np.minimum(o5, c5), width=0.75,
          color=np.where(up5, '#2E7D32', '#C62828'), edgecolor='none', zorder=3.5)
    ts5 = sub['timestamp'].to_numpy()
    # gridline interval configurable (owner 2026-07-30: "divide the 5s in 30s
    # instead of 1m so I can easily correlate bottom-up" -- so a sub-minute
    # panel's gridlines can line up exactly with the coarser panel's bars)
    grid_starts = np.where(ts5 % grid_s == 0)[0]
    # THIN when the grid interval is finer than the panel can show (owner
    # 2026-08-01, 4d/1h view): a 1h panel has ts%3600==0 on EVERY bar, so all
    # 96 got a gridline AND a rotated '%m/%d %H:%M' label -- the axis rendered
    # as unreadable mush. Cap both at ~14 evenly-spaced marks; the underlying
    # bars are unchanged, only the annotation density.
    MAX_TICKS = 14
    if len(grid_starts) > MAX_TICKS:
        grid_starts = grid_starts[::int(np.ceil(len(grid_starts) / MAX_TICKS))]
    for mi in grid_starts:
        ax.axvline(mi - 0.5, color='#9E9E9E', lw=0.6, ls=':', alpha=0.6, zorder=0.5)
    span_min = (sub['timestamp'].iloc[-1] - sub['timestamp'].iloc[0]) / 60
    ax.set_title(f'{title_prefix}TELESCOPE · {res} · last {span_min:.0f} min ({len(sub)} bars) '
                f'· grid={grid_s}s', fontsize=8, loc='left')
    ax.margins(y=0.15)
    ax.set_xticks(grid_starts - 0.5)
    # multi-day span (owner 2026-07-30, 1h "look back the past days" view):
    # HH:MM alone is ambiguous once bars cross a calendar-day boundary --
    # e.g. two different days both print "14:00". Detect the span and
    # prefix the date whenever it's not all one day.
    # ET everywhere (owner 2026-08-01, asking for 5min markers on the main
    # panel "so I know which hour it is"). The telescopes had been labelling in
    # UTC while the main panel now reads ET -- two clocks on one figure is
    # exactly the kind of thing that gets a level misread. One clock: ET.
    _et = lambda t: (pd.to_datetime(int(t), unit='s', utc=True)
                     .tz_convert('America/New_York'))
    multi_day = _et(ts5[0]).date() != _et(ts5[-1]).date()
    lbl_fmt = '%m/%d %H:%M' if multi_day else ('%H:%M' if grid_s >= 60 else '%H:%M:%S')
    ax.set_xticklabels([_et(t).strftime(lbl_fmt) for t in ts5[grid_starts]],
                       fontsize=6, rotation=(45 if (multi_day or grid_s < 60) else 0))
    # COMMON TIME WINDOW (owner 2026-08-01: "align the 10:05 for both 5s and 1s
    # so the last sections are along vertically"). Each panel plots against BAR
    # INDEX, so two panels covering the same seconds still drifted apart -- the
    # 1s series ends at the cutoff itself while the 5s series ends at its last
    # 5s boundary, up to 5s earlier. Mapping an explicit (T0,T1) onto the x-axis
    # pins identical timestamps to identical x positions across panels.
    if t_window:
        step = max(1, int(round((ts5[-1] - ts5[0]) / max(1, len(ts5) - 1))))
        to_x = lambda t: (t - ts5[0]) / step
        # the pad MUST be in seconds, not bars: half a bar is 0.5s on the 1s
        # panel and 2.5s on the 5s panel, which left the shared gridline ~8px
        # apart. With a constant time pad the axis fraction of any timestamp
        # reduces to (t-T0+pad)/(T1-T0+2*pad) -- independent of `step`, so the
        # two panels map identical times to identical x by construction.
        TELE_PAD_S = 3          # half a 5s candle body; keeps wicks off the edge
        ax.set_xlim(to_x(t_window[0] - TELE_PAD_S), to_x(t_window[1] + TELE_PAD_S))
    _price_grid(ax)


def _price_grid(ax):
    """Dashed price gridlines at the majors + minor ticks (owner 2026-08-03:
    "we are missing levels moving forward — add dashed lines for the ticks
    and add minor ticks"). Step auto-picked from the visible span so the
    grid stays readable at any zoom (~<=12 major lines); minor = major/5."""
    from matplotlib.ticker import MultipleLocator
    y_lo, y_hi = ax.get_ylim()
    span = max(1e-9, y_hi - y_lo)
    stp = next(v for v in (1, 2.5, 5, 10, 25, 50, 100, 250)
               if span / v <= 12)
    ax.yaxis.set_major_locator(MultipleLocator(stp))
    ax.yaxis.set_minor_locator(MultipleLocator(stp / 5))
    ax.grid(axis='y', which='major', ls='--', lw=0.6, color='#78909C',
            alpha=0.45, zorder=0.4)
    ax.grid(axis='y', which='minor', ls=':', lw=0.4, color='#90A4AE',
            alpha=0.28, zorder=0.3)
    ax.tick_params(axis='y', which='minor', length=2)
    ax.tick_params(axis='y', which='major', labelsize=7)


_ROLL_MANIFEST = os.path.join(REPO, 'DATA', 'ATLAS', 'roll_manifest.csv')


def _contracts(day_list):
    """{day: contract} from ATLAS's roll manifest. Multi-session views stitch
    ATLAS day files, which are per-day CHOSEN outrights -- so any span crossing
    a roll silently splices two different instruments and every cross-day level
    on it is meaningless (owner 2026-08-01 caught exactly this on 2024_09_16,
    MNQU4 -> MNQZ4). Returns {} if the manifest is missing rather than
    pretending the span is clean."""
    try:
        m = pd.read_csv(_ROLL_MANIFEST)
    except Exception:
        return {}
    m = m[m['day'].isin(day_list)]
    return dict(zip(m['day'], m['chosen']))


PNG_PREV = os.path.join(LOGDIR, 'pocket_prevday.png')


def _render_prevday(s, back=4, res='1m', label=None):
    """The last `back` COMPLETED sessions plus today up to the current bar, as
    one continuous 1m chart (owner 2026-08-01: "when I use the prevday [N]
    command I want to see the 4 prev days with the current day up to the
    current bar"). Distinct from `mainview 4d`, which is the same span at 1h
    with no level annotations -- this is the full-resolution levels map.

    Strictly causal: today is sliced at `cur`, and if an alarm halt truncated
    the frame the final bar is rebuilt from its own 5s bars, exactly as the
    main panel does. Per-day highs/lows are drawn as DAY-WIDTH segments (5
    full-width lines would be unreadable); only the most recent completed
    session's extremes and today's price get full-width lines, since those are
    the levels actually in play. Returns (path, caption)."""
    rdir = os.path.join(REPO, 'DATA', 'ATLAS', res)
    days = sorted(f[:-8] for f in os.listdir(rdir) if f.endswith('.parquet'))
    if s['day'] not in days:
        return None, f'no {res} data for {s["day"]}'
    i = days.index(s['day'])
    prior = days[max(0, i - back):i]
    if not prior:
        return None, f'no sessions before {s["day"]}'

    # CAUSAL CUT. One cutoff timestamp governs every resolution: the halt point
    # if an alarm truncated the frame, else the current 1m bar's own close.
    # Everything at or before it is real, everything after is unseen — so a
    # coarser view can never smuggle in future bars, and the trailing PARTIAL
    # bar is rebuilt from 5s so it shows only what had printed by the cutoff.
    cur = s['cur']
    d1 = _bars(s['day'])
    cutoff = int(s.get('halt_ts5') or (int(d1['timestamp'].iloc[cur]) + 59))
    d5 = _bars_tele(s['day'], '5s')

    frames, bounds, names = [], [], []
    for d in prior + [s['day']]:
        f = pd.read_parquet(os.path.join(rdir, f'{d}.parquet'))[
            ['timestamp', 'open', 'high', 'low', 'close']].reset_index(drop=True)
        if d == s['day']:
            f = f[f['timestamp'] <= cutoff].copy()
            if not len(f):
                continue
            if d5 is not None:                        # rebuild the partial bar
                t0 = int(f['timestamp'].iloc[-1])
                w = d5[(d5['timestamp'] >= t0) & (d5['timestamp'] <= cutoff)]
                if len(w):
                    f.iloc[-1, f.columns.get_loc('open')] = float(w['open'].iloc[0])
                    f.iloc[-1, f.columns.get_loc('high')] = float(w['high'].max())
                    f.iloc[-1, f.columns.get_loc('low')] = float(w['low'].min())
                    f.iloc[-1, f.columns.get_loc('close')] = float(w['close'].iloc[-1])
        bounds.append((sum(len(x) for x in frames), len(f)))
        frames.append(f); names.append(d)
    ext = pd.concat(frames, ignore_index=True)
    o = ext['open'].to_numpy(); h = ext['high'].to_numpy()
    l = ext['low'].to_numpy(); c = ext['close'].to_numpy()
    et = pd.to_datetime(ext['timestamp'], unit='s', utc=True).dt.tz_convert('America/New_York')
    mins = (et.dt.hour * 60 + et.dt.minute).to_numpy()
    x = np.arange(len(ext))

    fig, ax = plt.subplots(figsize=(15, 6.6), dpi=110)
    up = c >= o
    ax.vlines(x, l, h, color=np.where(up, '#2E7D32', '#C62828'), lw=0.45, zorder=3)

    y_lo, y_hi = float(l.min()), float(h.max())
    pad = (y_hi - y_lo) * 0.05
    ax.set_ylim(y_lo - pad, y_hi + pad)

    # per-day RTH shading, boundary rules, and day-width extreme segments
    for (b0, n), d in zip(bounds, names):
        seg = slice(b0, b0 + n)
        r = (mins[seg] >= 570) & (mins[seg] < 960)
        if r.any():
            ax.axvspan(b0 + int(np.argmax(r)), b0 + int(n - 1 - np.argmax(r[::-1])),
                       color='#5C6BC0', alpha=0.06, zorder=0)
        if b0:
            ax.axvline(b0 - 0.5, color='#455A64', lw=0.9, ls='--', alpha=0.55, zorder=2)
        dh, dl = float(h[seg].max()), float(l[seg].min())
        ax.hlines([dh], b0, b0 + n, color='#C62828', lw=0.9, alpha=0.55, zorder=2)
        ax.hlines([dl], b0, b0 + n, color='#2E7D32', lw=0.9, alpha=0.55, zorder=2)
        if len(names) <= 12 or names.index(d) % 2 == 0:   # avoid label pile-up
            ax.text(b0 + n * 0.5, y_hi + pad * 0.45, d[5:].replace('_', '/'),
                    fontsize=7.5, color='#455A64', ha='center', va='top', zorder=8)

    def _lvl(v, lab, col, ls='-'):
        ax.axhline(v, color=col, lw=1.1, ls=ls, alpha=0.85, zorder=2.5)
        ax.text(len(ext) * 0.998, v, f' {lab} {v:.2f}', color=col, fontsize=8,
                va='bottom', ha='right', zorder=9, clip_on=True)

    # CONTRACT ROLL GUARD — mark it loudly; a spliced span invalidates every
    # cross-day level drawn on it.
    con = _contracts(names)
    rolls = [(bounds[j][0], names[j - 1], names[j])
             for j in range(1, len(names))
             if con.get(names[j]) and con.get(names[j]) != con.get(names[j - 1])]
    for b0, dprev, dnext in rolls:
        ax.axvline(b0 - 0.5, color='#6A1B9A', lw=2.2, alpha=0.9, zorder=6)
        right = b0 > len(ext) * 0.65        # roll near the right edge -> flip
        ax.text(b0, y_lo - pad * 0.2,
                f'CONTRACT ROLL {con[dprev]}->{con[dnext]} — levels do NOT '
                f'cross this line  ', color='#6A1B9A', fontsize=8.5,
                fontweight='bold', ha=('right' if right else 'left'),
                va='bottom', zorder=9, clip_on=True)

    pb0, pn = bounds[-2]                      # last COMPLETED session
    pseg = slice(pb0, pb0 + pn)
    pdh, pdl = float(h[pseg].max()), float(l[pseg].min())
    _lvl(pdh, 'PD HIGH', '#C62828')
    _lvl(pdl, 'PD LOW', '#2E7D32')
    _lvl(float(c[pb0 + pn - 1]), 'PD SETTLE', '#455A64', ':')
    now = float(c[-1])
    _lvl(now, 'NOW', '#E8833A')

    tick = np.arange(0, len(ext), max(1, len(ext) // 14))
    ax.set_xticks(tick)
    ax.set_xticklabels([et.iloc[int(t)].strftime('%m/%d %H:%M') for t in tick],
                       fontsize=7, rotation=45, ha='right')
    ax.set_xlim(-2, len(ext) + 1)
    ax.grid(alpha=0.13, lw=0.5)
    ax.set_title(f'{label or f"{back} PRIOR SESSIONS"} + TODAY to bar {cur} · '
                 f'{names[0][5:]}–{names[-1][5:]} · {len(ext)} {res} bars · '
                 f'span {y_hi - y_lo:.0f}pt · ET (RTH shaded)', fontsize=10)
    fig.tight_layout(); fig.savefig(PNG_PREV); plt.close(fig)

    prng = pdh - pdl
    pos = (now - pdl) / prng * 100 if prng else float('nan')
    warn = ''
    if rolls:
        warn = ('*** CONTRACT ROLL INSIDE THIS SPAN: '
                + '; '.join(f'{con[a]}->{con[b]} at {b[5:]}' for _, a, b in rolls)
                + ' — cross-day levels spanning it are INVALID ***\n')
    elif con.get(s['day']):
        warn = f'contract {con[s["day"]]} (clean span)\n'
    cap = (warn
           + f'{back} prior sessions {names[0][5:]}–{names[-2][5:]} + today to '
           f'bar {cur}\n'
           f'{back}-session span {y_lo:.2f}–{y_hi:.2f} ({y_hi - y_lo:.0f}pt)\n'
           f'prev day {names[-2][5:]}: H {pdh:.2f} L {pdl:.2f} '
           f'settle {c[pb0 + pn - 1]:.2f}\n'
           f'now {now:.2f} = {pos:.0f}% of prev-day range '
           f'({"ABOVE" if now > pdh else "BELOW" if now < pdl else "inside"})')
    return PNG_PREV, cap



def _render(s, df):
    cur = s['cur']; sw = s.get('sigma_w', SIGMA_W0)
    o = df['open'].to_numpy(); h = df['high'].to_numpy()
    l = df['low'].to_numpy(); c = df['close'].to_numpy()
    # PRICE CUBIC ON 5s, matching the ACTUAL deployed NT8 spec (owner
    # 2026-07-30, second time raised: "we should be using the 5s or even 1s
    # for the TF" -- docs/nt8/2-CubicRegressionEndpoint_v1.0-RC.cs specifies
    # 7.5min/450 1s-bars; chart_replay_recorder.py's original, correct
    # implementation used 5s x90 = 7.5min exactly, matching by TIME SPAN not
    # by TF. pocket_dojo had drifted to 1m x8 (=8min, wrong basis, and
    # discards ALL sub-minute shape) -- fixed to match the reference.
    ts_cut_now = int(df['timestamp'].iloc[cur]) + 55 + s.get('peek_offset', 0)
    d5_price = _bars_tele(s['day'], '5s')
    # HALT TRUNCATION (2026-07-31). An alarm/warn-stop halts at 5s precision,
    # but _advance commits the WHOLE 1m bar first -- so without this the frame
    # showed the rest of the minute (the run past the level AND the fade) while
    # asking the owner to decide AT the level. That is lookahead in the
    # decision frame; it would poison every decision logged off an alarm halt.
    # Cutting ts_cut_now truncates the cubic, wick strip and telescope panels
    # automatically (they all key off it); only the main-panel 1m OHLC is read
    # straight from df, so rebuild the halt bar from its own 5s bars.
    halt5 = s.get('halt_ts5')
    if halt5 and d5_price is not None:
        # 5s bars are labelled by their START, so `<= halt` would include the
        # bar SPANNING the halt and leak up to 4s of future (owner's live short,
        # 2026-08-01: the frame printed 19681.00 when the honest price at the
        # halt instant was 19685.75). Exclude it from the 5s series and rebuild
        # the visible tail from 1s bars, which are exact at 1s granularity.
        ts_cut_now = int(halt5) - 1
        t0 = int(df['timestamp'].iloc[cur])
        d1h = _bars_tele(s['day'], '1s')
        src = d1h if d1h is not None else d5_price
        wpart = src[(src['timestamp'] >= t0) & (src['timestamp'] <= int(halt5))]
        if len(wpart):
            o = o.copy(); h = h.copy(); l = l.copy(); c = c.copy()
            o[cur] = float(wpart['open'].iloc[0])
            h[cur] = float(wpart['high'].max())
            l[cur] = float(wpart['low'].min())
            c[cur] = float(wpart['close'].iloc[-1])
    if d5_price is not None:
        sub5 = d5_price[d5_price['timestamp'] <= ts_cut_now]
        c5 = sub5['close'].to_numpy(); ep5 = sub5['timestamp'].to_numpy()
        cub5, slp5, _ = _cub.rolling(c5, CUBIC_5S_WINDOW, 5)   # slp5 units: pts/min
        res5 = c5 - cub5
        sig5 = pd.Series(res5).rolling(sw * 12, min_periods=5 * 12).std().to_numpy()
        # step-fill the 5s cubic/sigma onto the 1m view's bar closes (causal:
        # last 5s value AT OR BEFORE each 1m bar's own close+55s)
        ep1_cut = df['timestamp'].to_numpy() + 55
        k = np.searchsorted(ep5, ep1_cut[:cur + 1], side='right') - 1
        cub = np.where(k >= 0, cub5[np.clip(k, 0, len(cub5) - 1)], np.nan)
        sig = np.where(k >= 0, sig5[np.clip(k, 0, len(sig5) - 1)], np.nan)
        c0_5s, s0_5s = cub5[-1], sig5[-1]        # latest, at the peek cutoff
        slp0_5s = slp5[-1]                        # pts/min, raw endpoint (noisy)
        # damped slope for projection use: the raw cubic endpoint derivative
        # is a genuinely unstable estimator -- observed swinging -44 -> +42
        # pts/min over 75s (15 5s-bars) during a sharp move (2026-07-30 debug).
        # A short rolling mean over the last 4 5s-bars (20s) damps single-
        # point noise without adding lookahead (still strictly n-1).
        slp5_d = float(np.nanmean(slp5[-4:])) if len(slp5) >= 4 else slp0_5s
    else:
        cub = np.full(cur + 1, np.nan); sig = np.full(cur + 1, np.nan)
        c0_5s = s0_5s = slp0_5s = slp5_d = np.nan
    v0 = max(0, cur - int(s.get('view_bars', VIEW)))
    x = np.arange(v0, cur + 1)

    fig, (ax, axw, ax5a, ax5b) = plt.subplots(4, 1, figsize=(10, 11.8), dpi=110,
                                              gridspec_kw={'height_ratios': [3, 0.7, 1.15, 1.15]})
    # telescope frame (causal, MULTI-DAY context — a single day's bars miss
    # real swing levels from prior sessions, e.g. yesterday's high). Built
    # unconditionally -- also feeds the ax5a/ax5b panels below regardless
    # of which main-panel mode is active.
    global _hist_cache
    if _hist_cache.get('day') != s['day']:
        _hist_cache = {'day': s['day'], **dict(zip(('ext', 'off'), _bars_with_history(s['day'])))}
    ext, off = _hist_cache['ext'], _hist_cache['off']
    p = s.get('pos')
    main_view = s.get('main_view', '1m')
    tele_lines = []                      # (price, color, lw) — reused on the 5s/1h panels below
    try:
        for sc in telescope(ext.iloc[:off + cur + 1]):
            for L in sc['lines']:
                tele_lines.append((L['price'], sc['color'], sc['lw']))
                if main_view == '1m' and l[v0:cur + 1].min() - 15 <= L['price'] <= h[v0:cur + 1].max() + 15:
                    ax.axhline(L['price'], color=sc['color'], lw=sc['lw'], alpha=0.55, zorder=2)
                    ax.text(v0, L['price'], f" {sc['name']} {L['price']:.0f}({L['touches']}t)",
                            fontsize=7, color=sc['color'], va='center', fontweight='bold', clip_on=True)
    except Exception:
        pass
    # SESSION REFERENCE LINES (owner 2026-07-31: "I feel we are missing
    # reference lines" -- on fresh-territory days the density telescope has
    # nothing to draw where price has never pivoted, the documented trend-day
    # gap. These classic look-left references always exist causally: prior-day
    # H/L/C, overnight range, today's open, developing day H/L.)
    for rp, rlab in _session_refs(s['day'], df, cur):
        tele_lines.append((rp, '#607D8B', 1.0))
        if main_view == '1m' and l[v0:cur + 1].min() - 15 <= rp <= h[v0:cur + 1].max() + 15:
            ax.axhline(rp, color='#607D8B', lw=1.0, ls='-.', alpha=0.6, zorder=2.2)
            ax.text(cur + 0.5, rp, f' {rlab} {rp:.0f}', fontsize=6.5,
                    color='#607D8B', va='center', clip_on=True)
    ts = pd.to_datetime(df['timestamp'].iloc[cur], unit='s')
    pos_txt = (f"{p['dir']} @ {p['entry']:.1f}" if p and not p.get('pending')
               else ('pending ' + p['dir'] if p else 'flat'))
    ts = ts.tz_localize('UTC').tz_convert('America/New_York') if ts.tzinfo is None else ts.tz_convert('America/New_York')
    hdr = (f"POCKET DOJO · S{s.get('slice', '?')} · bar {cur}/{len(df)-1} · {ts:%H:%M} ET · "
           f"{pos_txt} · day P&L {s.get('pnl_pts', 0.0)*PT_USD:+.0f}$")

    if main_view == '4d':
        # MACRO main panel (owner 2026-07-30: "I want to see in the main
        # panel the last 4 days" -- reuses the exact 1h/96-bar/6h-grid
        # sizing already proven to fully cover a look-back day (Monday) on
        # the second panel minutes earlier. Cubic/sigma-bands/fog projection
        # are all short-timeframe concepts (7.5min cubic, 6min fog) that
        # would be unreadable noise zoomed to 4 days -- dropped here on
        # purpose, not lost: `mainview 1m` switches back to full detail.
        _draw_tele_panel(ax, s, '1h', 96, ts_cut_now, tele_lines, p, title_prefix='MAIN · ')
        # _draw_tele_panel sets its title with loc='left', a SEPARATE text
        # object from the default loc='center' -- setting another title
        # below without clearing this one first stacks both, garbled
        # (2026-07-30 bug, caught before sending).
        ax.set_title('', loc='left')
        ax.set_title(f'{hdr} · [4d/1h view]', fontsize=10)
    else:
        up = c[v0:cur + 1] >= o[v0:cur + 1]
        ax.vlines(x, l[v0:cur + 1], h[v0:cur + 1],
                  color=np.where(up, '#2E7D32', '#C62828'), lw=1.0, zorder=3)
        ax.bar(x, np.abs(c[v0:cur + 1] - o[v0:cur + 1]),
               bottom=np.minimum(o[v0:cur + 1], c[v0:cur + 1]), width=0.7,
               color=np.where(up, '#2E7D32', '#C62828'), edgecolor='none', zorder=3.5)
        cb = cub[v0:cur + 1]; sg = sig[v0:cur + 1]
        if np.isfinite(cb).any():
            ax.plot(x, cb, color='#E8833A', lw=1.8, zorder=4)
            ax.fill_between(x, cb - 2 * sg, cb + 2 * sg, color='#5C6BC0', alpha=0.10, zorder=1)
            ax.fill_between(x, cb - sg, cb + sg, color='#5C6BC0', alpha=0.12, zorder=1.1)
        # current-bar horizontal σ levels
        c0, s0 = c0_5s, s0_5s
        if np.isfinite(c0) and np.isfinite(s0):
            for m, lab in [(2, '+2σ'), (1, '+1σ'), (0, 'µ'), (-1, '−1σ'), (-2, '−2σ')]:
                lv = c0 + m * s0
                ax.axhline(lv, color='#E8833A' if m == 0 else '#3949AB',
                           ls=':' if m == 0 else '-', lw=0.9,
                           alpha=0.7 if abs(m) == 2 else 0.45, zorder=2)
                ax.text(cur + 0.5, lv, f'{lab} {lv:.1f}', fontsize=7, color='#3949AB', va='center', clip_on=True)
        # OWNER lines (hand-called levels — the selection-rule corpus, drawn distinct)
        for lp in s.get('owner_lines', []):
            ax.axhline(lp, color='#6A1B9A', lw=1.6, alpha=0.8, zorder=4.5)
            ax.text(cur + 0.5, lp, f' YOU {lp:.1f}', fontsize=8, color='#6A1B9A',
                    fontweight='bold', va='center', clip_on=True)
        # position marker
        if p and not p.get('pending'):
            ax.axhline(p['entry'], color='#000', lw=1.0, ls='--', zorder=5)
            ax.text(cur + 0.5, p['entry'], f" {p['dir']} {p['entry']:.1f}", fontsize=8, fontweight='bold', clip_on=True)
            if p.get('target'):
                ax.axhline(p['target'], color='#1B5E20', lw=1.0, ls='--', alpha=0.8, zorder=5)
            if p.get('stop'):
                ax.axhline(p['stop'], color='#B71C1C', lw=1.0, ls='--', alpha=0.8, zorder=5)
        # exit markers (owner 2026-08-04: "place a marker of exit") — every
        # booking drops an x at its bar/price; persists in s['exit_marks']
        for mts, mpx in s.get('exit_marks', [])[-10:]:
            _mb = df[df['timestamp'] <= mts]
            if len(_mb):
                mb = int(_mb.index[-1])
                if v0 <= mb <= cur + 1:
                    ax.plot(mb, mpx, marker='x', ms=9, mew=2.2,
                            color='#000', zorder=7)
                    ax.text(mb, mpx, f'  exit {mpx:.2f}', fontsize=7,
                            fontweight='bold', va='bottom', clip_on=True)
        # Y-LIMITS computed BEFORE the fog projection so the projection can be
        # clamped into them (owner bug report precedent: an off-view line like
        # 23400 was force-expanding autoscale and squishing the candles -- a
        # far value must NOT distort the view). Explicit from price data only.
        yv_lo = min(l[v0:cur + 1].min(), np.nanmin(cb - 2 * sg) if np.isfinite(cb).any() else np.inf)
        yv_hi = max(h[v0:cur + 1].max(), np.nanmax(cb + 2 * sg) if np.isfinite(cb).any() else -np.inf)
        ypad = max(1.0, (yv_hi - yv_lo) * 0.10)
        y_lo, y_hi = yv_lo - ypad, yv_hi + ypad
        ax.set_ylim(y_lo, y_hi)
        # fog edge
        ax.axvspan(cur + 0.5, cur + 6, color='#B0BEC5', alpha=0.35, zorder=6)
        # MECHANICAL PROJECTION into the fog (owner 2026-07-30: "make sure its
        # data is based on n-1" -- explicitly NOT a judgment call. A dotted linear
        # extension of the cubic's OWN (damped, see slp5_d above) slope at the
        # current bar, computed purely from data up to n-1 (no peeking).
        # Clamped to the panel's own y-range: text is unclipped by default in
        # matplotlib, so an extreme slope was rendering the label off-canvas in
        # blank space, disconnected from the chart entirely (2026-07-30 bug).
        # Clamping means a projection that would blow through the visible range
        # now visibly runs into the chart edge instead -- which is itself an
        # honest signal (the projection is extreme / the slope is unstable),
        # not hidden information.)
        if np.isfinite(c0_5s) and np.isfinite(slp5_d):
            t_fog = np.arange(0, 6.5)
            proj = np.clip(c0_5s + slp5_d * t_fog, y_lo, y_hi)
            ax.plot(cur + t_fog, proj, color='#E8833A', lw=1.3, ls=':', alpha=0.6,
                    zorder=6.5, clip_on=True)
            ax.text(cur + 6, proj[-1], ' n-1 slope\nprojection', fontsize=6.5,
                   color='#E8833A', va='center', style='italic', clip_on=True)
        # PARTIAL BAR (owner 2026-08-01: "Where is 10:05 in the 1m?"). The main
        # panel plotted COMMITTED 1m bars only, so while the telescopes showed
        # 30s of the forming minute, that minute simply did not exist here --
        # the two halves of the figure covered different time extents and there
        # was no way to see it. Draw the forming bar from its own 5s bars, in
        # outline, so it reads as unfinished rather than closed.
        if d5_price is not None and cur + 1 < len(df):
            t0p = int(df['timestamp'].iloc[cur + 1])
            wp = d5_price[(d5_price['timestamp'] >= t0p)
                          & (d5_price['timestamp'] <= ts_cut_now)]
            if len(wp):
                po = float(wp['open'].iloc[0]); ph = float(wp['high'].max())
                pl = float(wp['low'].min()); pc = float(wp['close'].iloc[-1])
                col = '#2E7D32' if pc >= po else '#C62828'
                ax.vlines(cur + 1, pl, ph, color=col, lw=1.0, alpha=0.55, zorder=3)
                ax.bar(cur + 1, abs(pc - po), bottom=min(po, pc), width=0.7,
                       facecolor='none', edgecolor=col, lw=1.1, alpha=0.9,
                       zorder=3.5, hatch='///')
                # label BELOW the bar: the top-right corner already holds the
                # sigma ladder and the fog projection, and 'forming' collided
                # with the +1sigma tag.
                ax.text(cur + 1, pl, 'forming', fontsize=6, color=col,
                        ha='center', va='top', zorder=8, clip_on=True)
        ax.set_title(f'{hdr} · σW={sw}', fontsize=10)
        ax.set_xlim(v0 - 0.5, cur + 6)
        _price_grid(ax)
        # 5-MINUTE ET CLOCK on the main panel (owner 2026-08-01: "add a marker
        # of every 5 minutes? So I know which hour it is similar to the
        # telescope"). The main x-axis was raw BAR INDEX, which carries no
        # time at all -- you had to read the telescope to know the hour.
        _hi = min(cur + 2, len(df))       # include the forming bar's label
        etm = (pd.to_datetime(df['timestamp'].to_numpy()[v0:_hi], unit='s',
                              utc=True).tz_convert('America/New_York'))
        mm = etm.minute.to_numpy()
        five = np.flatnonzero(mm % 5 == 0)
        if len(five):
            for k in five:
                ax.axvline(v0 + k - 0.5, color='#9E9E9E', lw=0.6, ls=':',
                           alpha=0.55, zorder=0.5)
            # thin labels if a 5min tick every bar would crowd the axis
            step = max(1, int(np.ceil(len(five) / 12)))
            keep = five[::step]
            ax.set_xticks([v0 + k for k in keep])
            ax.set_xticklabels([etm[k].strftime('%H:%M') for k in keep],
                               fontsize=7)
            ax.set_xlabel('ET', fontsize=7, labelpad=1)

    # ---- WICK-BIAS strip (owner 2026-07-30, cubic-wick-sensitivity research
    # agent recommendation: additive companion series, NOT part of the price
    # cubic -- render-only, per-slice logged so the divergence threshold gets
    # validated on real dojo episodes before earning any highlight rule.
    # NOTE basis mismatch, flagged not hidden: this is on 1m/8-bar (the basis
    # the agent actually tested and found the bar107 divergence on); the price
    # cubic above just moved to 5s/90 -- re-validating wick-bias at 5s is a
    # separate follow-up, not assumed equivalent.) ----
    # 5s-DERIVED INTRABAR RATIO, displayed at 1m cadence (owner 2026-07-31:
    # "aggregated to 1m but using the 5s -- the thesis is this will tell us
    # the intrabar ratio"). Per minute: (sum lower wicks - sum upper wicks) /
    # sum range over that minute's 5s bars -- 12 samples of who keeps losing
    # the micro-fights, range-weighted so 1-tick bars don't dominate. Bars =
    # raw per-minute value; thin line = 8-min mean (the old smoothed series'
    # grammar). Falls back to the 1m-candle basis if 5s data is absent.
    # CAVEAT flagged: the +-0.12 noise band was calibrated on the 1m-candle
    # series; not re-validated for this basis yet.
    wb_min = None
    if d5_price is not None:
        s5 = d5_price[d5_price['timestamp'] <= ts_cut_now]
        o5 = s5['open'].to_numpy(); h5 = s5['high'].to_numpy()
        l5 = s5['low'].to_numpy(); c5v = s5['close'].to_numpy()
        agg5 = pd.DataFrame({'k': (s5['timestamp'] // 60).to_numpy(),
                             'lw': np.minimum(o5, c5v) - l5,
                             'uw': h5 - np.maximum(o5, c5v),
                             'rg': h5 - l5}).groupby('k').sum()
        ratio = (agg5['lw'] - agg5['uw']) / agg5['rg'].replace(0, np.nan)
        mkey = (df['timestamp'] // 60).to_numpy()[:cur + 1]
        wb_min = ratio.reindex(mkey).to_numpy()
    if wb_min is None and _wick_bias is not None:
        wb_min = _wick_bias(o[:cur + 1], h[:cur + 1], l[:cur + 1], c[:cur + 1])
    if wb_min is not None:
        wb = wb_min[v0:cur + 1]
        wbm = pd.Series(wb_min).rolling(8, min_periods=4).mean().to_numpy()[v0:cur + 1]
        colr = np.where(wb >= 0, '#2E7D32', '#C62828')
        axw.bar(x, wb, width=0.8, color=colr, alpha=0.75, zorder=3)
        axw.plot(x, wbm, color='#37474F', lw=1.1, alpha=0.9, zorder=4)
        axw.axhspan(-0.12, 0.12, color='#9E9E9E', alpha=0.15, zorder=1)
        axw.axhline(0, color='#616161', lw=0.6, zorder=2)
        wmax = np.nanmax(np.abs(wb)) if np.isfinite(wb).any() else 0.2
        yl = min(1.0, max(0.2, 1.25 * wmax))
        axw.set_ylim(-yl, yl)
        axw.set_title('INTRABAR RATIO · per-1m from 5s wicks (Σlw−Σuw)/Σrng · '
                      'line=8m mean · band=1m-calibrated noise · '
                      'green=buyers defending, red=sellers capping', fontsize=7.5, loc='left')
        axw.set_xlim(v0 - 0.5, cur + 6)
        axw.set_xticks([])
    else:
        axw.text(0.5, 0.5, 'wick-bias module not found', transform=axw.transAxes,
                 ha='center', fontsize=8, color='gray')
        axw.set_xticks([]); axw.set_yticks([])

    # ---- TELESCOPIC sub-panels: intrabar detail. SWITCHABLE resolution panel
    # (owner: "present the 1m AND the telescopic 1s/5s"; later: "start looking
    # at 30s windows, even 15s") PLUS a FIXED always-5s/last-3min panel below
    # it (owner 2026-07-30: "add the 5s at the bottom, only last 3 minutes, so
    # I can see differences in the lengths of the bars" -- side-by-side
    # cross-resolution comparison, not just a toggle) ----
    tres = s.get('tele_res', TELE_RES_DEFAULT)
    # 1h view is inherently multi-day ("look back the past days") -- grid
    # every 6h (4 ticks/day) instead of every bar, else every hourly bar
    # would get its own gridline+label and the panel turns to clutter.
    # Bar count also needs its own budget: TELE_BARS=72 at 1h is only 3
    # days, which cuts off part of the oldest day shown (owner caught this:
    # "show the look back of Monday" -- Monday was only visible from noon
    # onward). 96 bars (4 days) covers a full extra day of buffer.
    tgrid = 6 * 3600 if tres == '1h' else (15 if tres == '1s' else 60)
    # owner 2026-07-30: "can we reduce the 30s read so it's about 10 min
    # worth" -- was 72 bars (36min) at 30s; 10min at 30s = 20 bars exactly.
    # 1s: 72 bars is only 72s and the 60s grid gives ~1 line. Give it 3
    # minutes and a 15s grid so the split-second structure is legible.
    # telescope span in SECONDS (owner 2026-08-04: "zoom in 1s/5s of the last
    # 2 min"); both sub-panels honour it so they stay time-aligned.
    tspan = int(s.get('tele_span', 180))
    tbars = (96 if tres == '1h' else tspan if tres == '1s'
             else 20 if tres == '30s' else TELE_BARS)
    # give the two sub-panels ONE shared time window when their spans match by
    # design (1s xN and 5s xN/5 cover the same seconds), so their gridlines
    # line up vertically; skip it for the 1h/4d panel, whose span is huge.
    span_a = tbars * {'1s': 1, '5s': 5, '15s': 15, '30s': 30, '1h': 3600}.get(tres, 5)
    tw = (ts_cut_now - tspan, ts_cut_now) if span_a == tspan else None
    _draw_tele_panel(ax5a, s, tres, tbars, ts_cut_now, tele_lines, p, grid_s=tgrid,
                     t_window=tw)
    _draw_tele_panel(ax5b, s, '5s', max(2, tspan // 5), ts_cut_now, tele_lines, p,
                     title_prefix='FIXED · ', grid_s=30, t_window=tw)
    fig.tight_layout()
    fig.savefig(PNG)
    plt.close(fig)
    return PNG


def _send(caption='', path=None):
    import requests
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, '.env'))
    tok = os.environ['TELEGRAM_BOT_TOKEN']; chat = os.environ['TELEGRAM_CHAT_ID']
    r = requests.post(f'https://api.telegram.org/bot{tok}/sendPhoto',
                      data={'chat_id': chat, 'caption': caption[:1000]},
                      files={'photo': open(path or PNG, 'rb')}, timeout=30)
    print('sent' if r.ok else f'SEND FAIL {r.text[:120]}')



# ============================================================================
# 1s EXECUTION CORE (owner 2026-08-02: "stop and develop a proper tool to
# execute these instructions"). One engine, one event order, one clock.
#
# Why this exists — the measured failures it retires (all from 2026-08-02):
#   - fill-ordering: _advance committed whole minutes, filling stops from
#     inside frozen futures (stole an owner decision; -10.89 booked wrongly)
#   - seam gap: after a mid-minute halt the rest of that minute was never
#     scanned (alarms fired 44s late)
#   - 5s-start stamps: alarms halted seconds early on straddling bars
#   - manual ratchet latency: the 80/70 protect could not arm inside sub-10s
#     V-moves (~14pt of giveback the protocol was designed to keep)
#   - warnstop popped the sized hard stop (mutual exclusion by design error)
#
# Event order per 1s bar, fixed and deterministic:
#   1. hard stop (wick fill AT the level — house convention)
#   2. MFE update -> protect auto-ratchet (recompute lines EVERY second;
#      new extreme releases a freeze)
#   3. protect warn/hard: warn = wick-touch (freeze fills nothing, so wick
#      sensitivity is safe and catches V-moves); hard = close-based honest
#      fill at that second's close
#   4. alarms / owner regions (with cooldown) / conditional warns / pace
#      milestone — freeze at the exact second
# A halt truncates AT its second. Nothing after it executes. Resume re-enters
# at halt+1, so minute remainders are never skipped.
# ============================================================================
PROTECT_DEFAULTS = dict(warn=0.80, hard=0.70, min_mfe=10.0, arm='region',
                        region=None, prox_pt=3.0)
REGION_COOLDOWN_S = 30


def _eng_sync(s, t):
    """Map the engine clock (absolute 1s ts) onto the legacy frame fields."""
    df = _bars(s['day'])
    b = int(df[df['timestamp'] <= t].index[-1])
    s['cur'] = b
    s['peek_offset'] = 0
    s['halt_ts5'] = int(t)
    _save(s)


def _eng_book(s, px, why, t):
    """Close the open position at px (already the honest fill for `why`) and
    stamp the clock at t in the SAME save.

    History: the replay test double-booked an exit because the book paths
    returned without saving; the audit then found the two-save fix (book,
    then sync) still left a crash window where P&L was booked beside a stale
    resume point, replaying alarm/region spans. One save covers everything.
    The corpus line is written AFTER the save: a missing close is detectable
    (pos None with no close event); a duplicated one silently double-counts
    the training corpus."""
    p = s['pos']
    d = 1 if p['dir'] == 'long' else -1
    pts = (px - p['entry']) * d - FRICTION_PT
    s['pnl_pts'] = round(s.get('pnl_pts', 0.0) + pts, 2)
    s['trades'] = s.get('trades', 0) + 1
    # a milestone deadline can outlive its trade — keep what it needs
    s['last_peak'] = p.get('peak', 0.0)
    s['last_entry'] = p['entry']
    s['pos'] = None
    s.setdefault('exit_marks', []).append([int(t), px])
    dfm = _bars(s['day'])
    s['cur'] = int(dfm[dfm['timestamp'] <= t].index[-1])
    s['peek_offset'] = 0
    s['halt_ts5'] = int(t)
    _save(s)
    _log(s, 'close', reason=why, price=px, pts=round(pts, 2))
    return pts


def _engine_run(s, df, secs, alarms=()):
    """Advance up to `secs` seconds on the 1s tape. Returns (events, halted)."""
    d1 = _bars_tele(s['day'], '1s')
    if d1 is None:
        return ['no 1s data'], False
    p0 = s.get('pos')
    if p0 and (p0.get('stop_reverse') or p0.get('bank_reenter')):
        return ['ENGINE REFUSED: stop_reverse/bank_reenter is armed on the '
                'open position and the 1s engine does not implement reversal '
                'semantics — running would silently book a plain stop '
                'instead. Clear the flag or drive with step/watch.'], False
    # Resume clock. The fallback for a legacy-committed bar is +59 (+peek):
    # the WHOLE minute is committed tape. Resuming at the +55 display cutoff
    # replayed :56-:59 and could fill a fresh stop on pre-entry wicks
    # (audit 3b — 'step; call; stop; run' booked P&L from tape older than
    # the entry).
    t0 = int(s.get('halt_ts5')
             or (int(df['timestamp'].iloc[s['cur']]) + 59
                 + s.get('peek_offset', 0)))
    seg = d1[(d1['timestamp'] > t0) & (d1['timestamp'] <= t0 + secs)]
    if not len(seg):
        msg = [f'no tape after {t0}']
        if s.get('pos'):
            msg.append('*** POSITION STILL OPEN with no tape left — the '
                       'engine never auto-flattens; close with exit --at '
                       'when decided ***')
        return msg, False
    pr = s.get('protect') or {}
    ev = []
    # Region cooldown/inside state PERSISTS in s — a function-local dict
    # re-fired the same region 1s after every halt while price dwelt inside
    # its own density band (the band is BY CONSTRUCTION where price
    # loiters), degrading run to one second per invocation exactly at the
    # owner's decision zones.
    rs = s.setdefault('region_state', {})
    # Alarm semantics: fire when a CLOSE crosses the level relative to the
    # side price was on when the run resumed. Wick-touch + an "already
    # touching" suppression (the previous design) silently swallowed a real
    # break: on 2026-08-04 the first resumed second straddled the owner's
    # rail on the way DOWN, was suppressed, every later bar sat entirely
    # below it, and the rail only fired 2.5 minutes later on the way back up
    # — after a 51pt run he had explicitly asked to be woken for. A close
    # beyond the level is what "the box broke" actually means.
    r1 = seg.iloc[0]
    alarm_side = {}
    for lv in alarms:
        lv = float(lv)
        c1 = float(r1['close'])
        alarm_side[lv] = 1 if c1 > lv else (-1 if c1 < lv else 0)
    ms = pr.get('milestone')          # dict(level=, by_ts=) or None
    for _, r in seg.iterrows():
        t = int(r['timestamp'])
        op_, hi_, lo_, cl = (float(r['open']), float(r['high']),
                             float(r['low']), float(r['close']))
        p = s.get('pos')
        if p:
            d = 1 if p['dir'] == 'long' else -1
            # 0.9 CLOCK RATCHET (owner 2026-08-04, rate chosen after
            # pushback: he proposed +1pt/s, measured down-legs run 0.38pt/s
            # median — a 1pt/s clock beheads the median winner near +6 and
            # would have capped the +26 V-leg near +13; he picked 0.4pt/s =
            # median pace). The stop walks from its ORIGINAL level toward
            # (and through) entry at `rate` pt/s: outrun the clock or the
            # trade is closed. Tightening only, never loosens; computed
            # BEFORE the stop check so the level in force at second t is
            # the level the clock says at t (adverse-first).
            clk = pr.get('clock')
            if clk and p.get('entry_ts') and p.get('stop0') is not None:
                el = (t - int(p['entry_ts'])) - float(clk.get('grace', 0.0))
                if el > 0:
                    ck = p['stop0'] + d * float(clk['rate']) * el
                    cs = p.get('stop')
                    if cs is None or (d > 0 and ck > cs) or (d < 0 and ck < cs):
                        p['stop'] = ck
            # 1. hard stop — chronologically first (adverse wins every
            # intra-second tie: over target, over new-MFE, over protect).
            # Fill AT the level, or at the OPEN when the bar gaps through —
            # filling a gapped stop at the level books a price the tape
            # never printed (audit item 1; 1s gaps are routine in MNQ).
            stp = p.get('stop')
            if stp is not None and ((d > 0 and lo_ <= stp)
                                    or (d < 0 and hi_ >= stp)):
                px = min(op_, float(stp)) if d > 0 else max(op_, float(stp))
                pts = _eng_book(s, px, 'stop', t)
                gap = '' if px == float(stp) else f' (gapped past {stp:.2f})'
                ev.append(f'STOP filled @ {px:.2f}{gap} -> {pts:+.2f}pt')
                return ev, True
            # 1.5 target — resting limit, filled AT the level (a favorable
            # gap-through also fills at the level: understates, i.e.
            # conservative). Stop already had tie priority above — the
            # OPPOSITE of legacy _advance's optimistic target-first, on
            # purpose. Without this check the engine silently ignored
            # call --target entirely (audit: path-dependent semantics).
            tgt = p.get('target')
            if tgt is not None and ((d > 0 and hi_ >= tgt)
                                    or (d < 0 and lo_ <= tgt)):
                pts = _eng_book(s, float(tgt), 'target', t)
                ev.append(f'TARGET filled @ {float(tgt):.2f} -> {pts:+.2f}pt')
                return ev, True
            # 1.6 warnstop — the position's halt-not-exit level (legacy
            # `warnstop` cmd); the engine used to blow through it silently
            ws = p.get('warn_stop')
            if ws is not None and lo_ <= float(ws) <= hi_:
                _eng_sync(s, t)
                _log(s, 'warnstop_hit', price=float(ws), ts1=t)
                ev.append(f'*** WARNSTOP {float(ws):.2f} touched at '
                          f'{_ets(t)} — HALTED, no exit ***')
                return ev, True
            # 1.7 ENTRY-TOUCH WARNING (owner 2026-08-04, after a long that
            # peaked +1.75 then rode to the -10 stop: "any trade that
            # touches entry is automatic warning"). Once the trade has
            # traded away favorably at all, a RETURN to the entry price
            # halts the tape — a warning, not an exit. Fires once per
            # position; tested against the PRIOR second's peak like every
            # other line, so the entry bar itself can never trigger it.
            if (pr.get('entry_warn', True)
                    and not p.get('entry_warned') and p.get('peak', 0.0) > 0
                    and lo_ <= p['entry'] <= hi_):
                p['entry_warned'] = True
                _eng_sync(s, t)
                _log(s, 'entry_touch', price=p['entry'],
                     peak=p.get('peak', 0.0), ts1=t)
                ev.append(f'*** ENTRY TOUCHED {p["entry"]:.2f} at {_ets(t)} '
                          f'— peak was {p.get("peak", 0.0):+.2f}, position '
                          f'open, decision time ***')
                return ev, True
            # 1.8 CONDITIONAL GIVE-BACK WARNING (owner 2026-08-04 "arm +3
            # warning", same pattern as his earlier "a warning at +2p if we
            # go past it and return"): once the trade has EXCEEDED +N, a
            # return to the +N level halts the tape. Warning, not an exit;
            # once per position. Sits between the entry-touch warning
            # (giving back everything) and the percentage ratchet (which
            # only arms at min_mfe).
            gbn = p.get('warn_gb')
            if (gbn and not p.get('warn_gb_fired')
                    and p.get('peak', 0.0) > float(gbn)):
                lvl = p['entry'] + d * float(gbn)
                if lo_ <= lvl <= hi_:
                    p['warn_gb_fired'] = True
                    _eng_sync(s, t)
                    _log(s, 'warn_giveback', level=lvl, pts=float(gbn),
                         peak=p.get('peak', 0.0), ts1=t)
                    ev.append(f'*** +{float(gbn):.0f} WARNING {lvl:.2f} at '
                              f'{_ets(t)} — peak was '
                              f'{p.get("peak", 0.0):+.2f}, position open, '
                              f'decision time ***')
                    return ev, True
            # 2. MFE / auto-ratchet. Lines are checked against the PRIOR
            # second's peak — computing the warn off a peak set by THIS
            # second's own wick makes every wide second self-freeze
            # (caught by inspection before first run).
            pk_prev = p.get('peak', 0.0)
            fav = (hi_ - p['entry']) if d > 0 else (p['entry'] - lo_)
            new_peak_sec = fav > pk_prev
            if new_peak_sec:
                p['peak'] = fav
                if p.get('frozen') is not None:
                    p['frozen'] = None
                    ev.append(f'{ _ets(t) } new MFE {fav:+.2f} — freeze released')
            # 2.5 risk ladder (owner 2026-08-04, after the unarmed-MFE loss:
            # "always arm but leave room to breathe: start at -10, then
            # ratchet at +2; at risk of fake the ratchet region expands to
            # +5 or higher — a risk-based approach"). Once MFE clears the
            # trigger, the resting stop jumps to entry+jump (BE+2 — the
            # measured zero-EV-cost risk control) and never loosens. Silent:
            # a resting-order move, not a decision halt.
            lad = pr.get('ladder')
            if lad and p.get('peak', 0.0) >= float(lad['trigger']):
                # lock: TRAIL a fraction of peak instead of a flat BE+jump
                # (owner 2026-08-04 "lock the 50%", after a trade that
                # peaked +7.75 and kept +2 because the percentage ratchet
                # only arms at min_mfe=10 — the +5..+10 band had no trail).
                lk = lad.get('lock')
                new_stop = (p['entry'] + d * p['peak'] * float(lk) if lk
                            else p['entry'] + d * float(lad.get('jump', 2.0)))
                cs = p.get('stop')
                if (cs is None or (d > 0 and new_stop > cs)
                        or (d < 0 and new_stop < cs)):
                    p['stop'] = new_stop
                    _log(s, 'ladder_stop', stop=new_stop, peak=p['peak'],
                         ts1=t)
                    ev.append(f'{_ets(t)} LADDER: stop -> {new_stop:.2f} '
                              f'(peak {p["peak"]:+.2f})')
            # 3. protect machine (on pk_prev)
            if pr.get('on') and pk_prev >= pr.get('min_mfe', 10.0):
                armed = p.get('prot_armed', False)
                if not armed:
                    mode = pr.get('arm', 'region')
                    if mode == 'always':
                        armed = True
                    elif mode == 'region' and pr.get('region') is not None:
                        tgt = float(pr['region'])
                        if abs(cl - tgt) <= pr.get('prox_pt', 3.0):
                            armed = True
                    if armed:
                        p['prot_armed'] = True
                        ev.append(f'{_ets(t)} PROTECT ARMED (peak '
                                  f'{p["peak"]:+.2f})')
                if armed:
                    peak = pk_prev
                    warn_px = p['entry'] + d * peak * pr.get('warn', 0.80)
                    cur = (cl - p['entry']) * d
                    # A new-MFE second RESETS the structure (owner protocol:
                    # "the warning resets to the new level") — the old 80-line
                    # is dead that second; the NEW line is tested from the
                    # next second. Without this, the 10:20:55 bar re-froze on
                    # the stale line with frozen=10.75 while peak=15.5.
                    if p.get('frozen') is None and not new_peak_sec:
                        touched = (lo_ <= warn_px) if d > 0 else (hi_ >= warn_px)
                        if touched:
                            p['frozen'] = peak
                            _eng_sync(s, t)
                            _log(s, 'prot_warn', price=warn_px, peak=peak)
                            ev.append(f'*** 80 LINE {warn_px:.2f} (peak '
                                      f'{peak:+.2f}) — FROZEN at {_ets(t)}, '
                                      f'position open, decision time ***')
                            ev.append(_gauge_line(s, p, cur, peak, t))
                            al = _actuary_line(s, p, t)
                            if al:
                                ev.append(al)
                            gl = _gbm_line(s, t)
                            if gl:
                                ev.append(gl)
                            return ev, True
                    # frozen-not-None guard: a freeze-release second clears
                    # frozen but keeps prot_hard armed (the ratchet: the 70
                    # goes dormant until the NEXT freeze re-sets its floor).
                    # Without the guard this line computed None*0.7 and the
                    # run died mid-bar in the NORMAL owner flow
                    # freeze -> protect hard -> price recovers (audit #1).
                    elif p.get('prot_hard') and p.get('frozen') is not None:
                        hard_val = p['frozen'] * pr.get('hard', 0.70)
                        if cur <= hard_val:
                            # The trigger is OBSERVED at this close — you
                            # cannot trade a price you just watched print.
                            # Honest fill = next tape second's OPEN (audit:
                            # close-fill was systematically optimistic
                            # exactly in the fast tape where hard exits
                            # fire). Trigger close only if the day ends.
                            nxt = d1[d1['timestamp'] > t]
                            if len(nxt):
                                fpx = float(nxt['open'].iloc[0])
                                ft = int(nxt['timestamp'].iloc[0])
                            else:
                                fpx, ft = cl, t
                            pts = _eng_book(s, fpx, 'prot_hard', ft)
                            ev.append(f'70 HARD: trigger {cl:.2f} at '
                                      f'{_ets(t)} -> fill {fpx:.2f} at '
                                      f'{_ets(ft)} = {pts:+.2f}pt')
                            return ev, True
        # 4. alarms / regions / milestone
        for lv in alarms:
            lv = float(lv)
            side = 1 if cl > lv else (-1 if cl < lv else 0)
            s0 = alarm_side.get(lv, 0)
            if s0 == 0:                      # resumed exactly on the level
                alarm_side[lv] = side
                continue
            if side != 0 and side != s0:     # closed through it — a break
                _eng_sync(s, t)
                _log(s, 'alarm', price=lv, ts1=t, close=cl)
                ev.append(f'*** ALARM {lv:.2f} BROKEN at {_ets(t)} '
                          f'(close {cl:.2f}) — HALTED ***')
                return ev, True
        for lv in [float(x) for x in s.get('owner_lines', [])]:
            k = f'{lv:g}'
            st = rs.get(k) or {'last': 0, 'in': False}
            reg = _level_region(s, lv, t)
            inside = bool(reg and reg[0] <= cl <= reg[1])
            if (inside and not st['in']
                    and t - st['last'] > REGION_COOLDOWN_S):
                st['last'] = t
                st['in'] = True
                rs[k] = st
                _eng_sync(s, t)
                _log(s, 'region_hit', level=lv, price=cl, ts1=t)
                ev.append(f'*** REGION {lv:.2f} entered at {_ets(t)} '
                          f'({cl:.2f}) ***')
                return ev, True
            st['in'] = inside
            rs[k] = st
        if ms and t >= int(ms['by_ts']):
            # pop BEFORE the save — popping after left the milestone on disk
            # and the next run re-fired it (same class as the book-path bug)
            pr.pop('milestone', None)
            late = t > int(ms['by_ts'])
            p = s.get('pos')
            if p:
                # a peak made AFTER the deadline (gap bar past due) must
                # not score a HIT — use the pre-bar peak when late
                pk = pk_prev if late else p.get('peak', 0.0)
                got = pk >= abs(float(ms['level']) - p['entry'])
            elif s.get('last_entry') is not None:
                # trade closed before the deadline — score what it reached
                got = s.get('last_peak', 0.0) >= abs(float(ms['level'])
                                                     - s['last_entry'])
            else:
                got = False
            _eng_sync(s, t)
            _log(s, 'milestone', hit=bool(got), **{k: ms[k] for k in ms})
            ev.append(f'*** MILESTONE {"HIT" if got else "MISSED"} at '
                      f'{_ets(t)} (level {ms["level"]}) ***')
            return ev, True
    _eng_sync(s, int(seg['timestamp'].iloc[-1]))
    return ev + [f'advanced {len(seg)}s, no trigger'], False


def _ets(t):
    return pd.to_datetime(int(t), unit='s', utc=True).tz_convert(
        'America/New_York').strftime('%H:%M:%S')


def _actuary_line(s, p, t):
    """Bayesian-table readout at a freeze (owner overnight order 2026-08-04:
    "no epoch ... Bayesian table"). Says what the corpus SAW in this context
    — posterior, day-clustered interval, N — and refuses to dress up a base
    rate as knowledge: a cell only reads ACTIONABLE if it survived FDR +
    clustered bootstrap. Best-effort; the dojo must never break on it."""
    try:
        import sys as _sys
        bt = os.path.join(REPO, 'research', 'bayes_tables', 'tools')
        if bt not in _sys.path:
            _sys.path.insert(0, bt)
        import actuary
        et = (pd.to_datetime(int(t), unit='s', utc=True)
              .tz_convert('America/New_York'))
        mm = et.hour * 60 + et.minute
        clock_b = ('0930' if mm < 600 else '1000' if mm < 630
                   else '1030' if mm < 720 else '1200' if mm < 840 else '1400')
        d = 1 if p and p['dir'] == 'long' else -1
        out = []
        r = actuary.lookup('stall', 'race',
                           dir_s=('up' if d > 0 else 'dn'), clock_b=clock_b)
        if r:
            out.append(f'stall->new extreme {r.p:.0%} '
                       f'[{r.lo:.0%},{r.hi:.0%}] n={r.n}'
                       + ('' if r.actionable else ' (base)'))
        r2 = actuary.lookup('leg_descent', 'race', clock_b=clock_b)
        if r2:
            out.append(f'descent->new low {r2.p:.0%} n={r2.n}'
                       + ('' if r2.actionable else ' (base)'))
        return 'TABLE ' + ' | '.join(out) if out else ''
    except Exception as e:
        return f'(actuary unavailable: {e})'


def _gbm_line(s, t):
    """Onset readout from the CAUSAL GBM (research/event_onset/models/).

    This is the arming model the pre-registered bake-off selected: the Mamba
    lost 3/3 heads to these 22 hand-made features (RUN1_VERDICT.md). It says
    'an event of this type is forming in the next ~10s' — nothing about
    direction, which the geometry control showed is not there to be had.
    Honest AUCs (causal, fit-2024/score-2025H1): fakeout 0.643, descent
    0.759, chop 0.820, stall 0.654. Best-effort; never breaks the dojo."""
    try:
        import glob as _g
        import joblib
        import sys as _sys
        eo = os.path.join(REPO, 'research', 'event_onset', 'builders')
        if eo not in _sys.path:
            _sys.path.insert(0, eo)
        from build_onset_dataset import _feat_matrix
        d5 = _bars_tele(s['day'], '5s')
        if d5 is None:
            return ''
        ts = d5['timestamp'].to_numpy()
        i = int(np.searchsorted(ts, int(t), side='right')) - 1
        if i < 400:
            return ''
        cols = ('open', 'high', 'low', 'close')
        o, h, l, c = (d5[k].to_numpy() for k in cols)
        v = (d5['volume'].to_numpy() if 'volume' in d5
             else np.ones(len(d5)))
        feat = _feat_matrix(ts, o, h, l, c, v, np.array([i]))
        out = []
        for path in sorted(_g.glob(os.path.join(
                REPO, 'research', 'event_onset', 'models', 'gbm_*_10s.joblib'))):
            b = joblib.load(path)
            name = (os.path.basename(path).replace('gbm_', '')
                    .replace('_10s.joblib', ''))
            X = np.nan_to_num(feat[b['feats']].to_numpy(float), nan=0.0,
                              posinf=0.0, neginf=0.0)
            pr = float(b['model'].predict_proba(b['scaler'].transform(X))[0, 1])
            if pr >= 0.60:                      # only speak when it matters
                out.append(f'{name} {pr:.0%}')
        return ('ONSET(10s) ' + ' | '.join(out)) if out else ''
    except Exception as e:
        return f'(onset unavailable: {e})'


def _gauge_line(s, p, cur, peak, t):
    """p(resume) cockpit readout at a freeze (owner 2026-08-04: 'the
    mechanical part is to calculate the probability of reversal').
    Calibrated on 118k historical giveback events (research/reversal_gauge).
    Honesty: it is a calibrated BASE-RATE gauge, chiefly driven by giveback
    depth — discrimination beyond that sits at the program's 0.57 wall.
    Best-effort: the dojo must never break if the gauge is absent."""
    try:
        import sys as _sys
        rg = os.path.join(REPO, 'research', 'reversal_gauge', 'tools')
        if rg not in _sys.path:
            _sys.path.insert(0, rg)
        import gauge
        mins = (pd.to_datetime(int(t), unit='s', utc=True)
                .tz_convert('America/New_York'))
        mins = mins.hour * 60 + mins.minute - 570
        import math
        feat = dict(giveback_frac=max(0.0, 1.0 - cur / peak) if peak else 0.0,
                    clock_sin=math.sin(2 * math.pi * mins / 360),
                    clock_cos=math.cos(2 * math.pi * mins / 360))
        prob, drivers = gauge.p_resume(feat)
        return ('GAUGE ' + gauge.format_gauge(prob, drivers)
                + ' (calibrated base rate, 118k events)')
    except Exception as e:
        return f'(gauge unavailable: {e})'


def _halt_px(s):
    """Honest default fill while halted: the 1s close AT the frozen instant.
    The full-minute close is up to ~59s of future relative to the truncated
    frame the owner is looking at — the audit flagged every post-halt
    call/exit without --at as a lookahead fill (one 9.75pt misprice shipped
    before --at existed). Returns None when no halt or no 1s data."""
    t = s.get('halt_ts5')
    if not t:
        return None
    d1 = _bars_tele(s['day'], '1s')
    if d1 is None:
        return None
    m = d1[d1['timestamp'] <= int(t)]
    return float(m['close'].iloc[-1]) if len(m) else None


# ---------- commands ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('rest', nargs='*')
    ap.add_argument('--day')
    ap.add_argument('--target', type=float)
    ap.add_argument('--stop', type=float)
    ap.add_argument('--stall', type=int,
                    help='watch: stamp/halt when the OPEN position makes no '
                         'new favourable extreme for this many seconds. '
                         'ATTENTION DEVICE ONLY -- validated 2026-08-02 on '
                         '7,306 trades / 72 sessions: exiting at the stamp '
                         'nets -0.81pt vs the band exit -0.78pt, paired delta '
                         '-0.03 CI [-0.43,+0.36], NOT significant. It is not '
                         'an edge and must never be sold as an exit rule.')
    ap.add_argument('--giveback', type=float,
                    help='watch: halt the first time the OPEN position '
                         'retraces this many points from its running best')
    ap.add_argument('--at', type=float,
                    help='explicit fill price (e.g. an alarm level); '
                         'default is the committed bar close, which '
                         'MISPRICES an entry taken mid-peek at a level')
    ap.add_argument('--send', action='store_true')
    ap.add_argument('--alarm', type=float, nargs='+', default=None,
                    help='halt at a 5s touch of ANY of these levels '
                         '(multi-level: bracket observation while flat); '
                         'no order is opened — the sim just stops so you '
                         'cannot step past a level.')
    ap.add_argument('--until-fill', action='store_true',
                    help='step: stop the moment a target/stop fills, leaving '
                         'cur ON the fill bar instead of running out the full N')
    a = ap.parse_args()

    if a.cmd == 'new':
        days = sorted(f[:-8] for f in os.listdir(DATA) if f.endswith('.parquet'))
        rng = np.random.default_rng(RNG_DAY_SEED or int(time.time()))
        day = a.day or days[rng.integers(len(days))]
        df = _bars(day)
        start = _pick_start(df, rng)          # any hour, liquidity-gated
        s = dict(day=day, cur=start, sigma_w=SIGMA_W0, pos=None, pnl_pts=0.0,
                 trades=0, slice=_next_slice())
        _save(s); _log(s, 'new', start=start)
        _render(s, df)
        cap = f'POCKET DOJO — fogged day (identity hidden). Bar {start}. Your call.'
        # The day identity used to be printed to stdout here, which leaked it
        # to the ASSISTANT on every `new` (owner 2026-08-05: a fogged day must
        # be fogged for both sides). It stays in the state file on disk.
        day_shown = False
        if a.send:
            _send(cap)
        print(f'day={day} (fogged) cur={start} -> {PNG}')
        return

    s = _load()
    if s is None:
        raise SystemExit('no session — run: pocket_dojo.py new')
    df = _bars(s['day'])

    if a.cmd == 'peek':
        # show N more seconds on the telescope panels. BUG FIX (owner
        # 2026-07-30: "if we advance 90s we should advance 1 1m and 30s") --
        # peek_offset used to accumulate unboundedly WITHOUT ever committing a
        # real 1m step, so a stop/target that would have hit during a crossed
        # 1m bar was silently never checked, and the 1m panel went stale. Now
        # every full 60s rolls over into a genuine _advance() (correct
        # stop/target fills, slice bump); only the sub-minute remainder stays
        # as peek_offset.
        secs = int(a.rest[0]) if a.rest else 20
        # If the frame is TRUNCATED at an alarm halt, peek must walk forward
        # from the halt point, not jump straight to the bar's close -- else
        # the first peek hands back the whole rest of the minute we just
        # truncated away. Only once it reaches the bar end do normal
        # peek_offset semantics resume, carrying any leftover seconds.
        halt5 = s.get('halt_ts5')
        if halt5:
            bar_end = int(df['timestamp'].iloc[s['cur']]) + 55
            if int(halt5) + secs < bar_end:
                s['halt_ts5'] = int(halt5) + secs
                _save(s); _log(s, 'peek', secs=secs, from_halt=True)
                _render(s, df)
                cap = f" +{secs}s peek (halt frame -> {pd.to_datetime(s['halt_ts5'], unit='s'):%H:%M:%S})"
                if a.send:
                    _send(cap)
                print(cap); return
            # Engine halts stamp any second, incl. :56-:59 (past the +55
            # legacy display max). Unclamped, bar_end - halt5 goes NEGATIVE
            # and INFLATES secs (audit 3c) — carry the consumed overshoot
            # into peek_offset instead.
            over = int(halt5) - bar_end
            if over > 0:
                s['peek_offset'] = max(s.get('peek_offset', 0), over)
            else:
                secs -= (bar_end - int(halt5))  # spend what's left of this bar
            s.pop('halt_ts5', None)
        total = s.get('peek_offset', 0) + secs
        whole_min, s['peek_offset'] = divmod(total, 60)
        ev = []
        if whole_min:
            s['slice'] = _next_slice()
            ev = _advance(s, df, whole_min)
        # SUB-minute check (owner 2026-07-30: caught a stop touch inside the
        # peeked-but-uncommitted partial bar that whole-minute rollover alone
        # would have missed until the bar fully closed) — check real 5s bars
        # from the last committed close out to the new peek cutoff.
        ts_committed = int(df['timestamp'].iloc[s['cur']]) + 55
        ts_new_cutoff = ts_committed + s['peek_offset']
        sub_ev = _check_5s_fill(s, ts_committed, ts_new_cutoff)
        if sub_ev:
            ev.append(sub_ev)
        _save(s); _log(s, 'peek', secs=secs, rolled_1m=whole_min, remainder=s['peek_offset'])
        _render(s, df)
        cap = (' | '.join(ev) if ev else '') + f" +{secs}s peek (offset {s['peek_offset']}s)"
        bl = _bar_line(s, df)
        if bl:
            cap += '\n' + bl
        osc = _osc_line(s, df)
        if osc:
            cap += '\n' + osc
        if a.send:
            _send(cap)
        print(cap)
    elif a.cmd == 'step':
        n = int(a.rest[0]) if a.rest else 1
        was_flat = s.get('pos') is None
        frm = s['cur']
        s['_prev_peek'] = s.get('peek_offset', 0)   # for the alarm's past-guard
        s['peek_offset'] = 0                   # a real step resets any peek
        h5_pre = s.pop('halt_ts5', None)       # ...and un-truncates a prior halt bar
        s['slice'] = _next_slice()            # each step = a new decision point
        p_ws = s.get('pos') or {}
        alarms = [x for x in [*(a.alarm or []), p_ws.get('warn_stop')]
                  if x is not None]
        # NEVER halt in the past. `step` zeroes peek_offset, so without this the
        # alarm scan re-examines 5s bars the frame had ALREADY revealed and
        # halts at a touch that has already happened -- rewinding visible time
        # and offering a fill at a level we did not act on (owner 2026-08-01,
        # arming the upper region; caught before it produced a fabricated
        # entry). Only bars strictly after the previous frame cutoff count.
        # (The audit found this read halt_ts5 AFTER the pop above — it always
        # fell back and overstated unseen tape by up to ~52s.)
        seen_to = int(h5_pre
                      or (int(df['timestamp'].iloc[frm]) + 55
                          + (s.get('_prev_peek', 0))))
        # A mid-minute halt leaves the REST of that minute unscanned by
        # _advance (it starts at the next 1m bar) — with a position open or
        # alarms armed that skips real stop/target/alarm tape (audit 3a, the
        # exact seam-gap class the engine was built to kill). Consume the
        # remainder through the 1s engine first; an engine halt supersedes
        # the step.
        if h5_pre and (s.get('pos') or alarms):
            mend = (h5_pre // 60) * 60 + 59
            if h5_pre < mend:
                s['halt_ts5'] = h5_pre
                ev0, halted0 = _engine_run(s, df, mend - h5_pre,
                                           alarms=alarms)
                if halted0:
                    _render(s, df)
                    cap = ' | '.join(ev0[-3:])
                    if a.send:
                        _send(cap)
                    print('\n'.join(ev0))
                    return
                s.pop('halt_ts5', None)
                if ev0 and ev0[0] != 'no 1s data':
                    seen_to = mend
        if alarms:
            # ALARM (owner 2026-07-31: "how about we do an alarm limit? we will
            # stop if it triggers the alarm?") -- an ALERT, not an order. Step
            # one bar at a time and halt the instant a real 5s bar touches the
            # level from either side, so a level can never be stepped past
            # unseen. Opens nothing; any position keeps its own stop/target.
            d5a = _bars_tele(s['day'], '5s')
            ev, rang, rlvl = [], None, None
            for _ in range(n):
                ev += _advance(s, df, 1)
                t0 = int(df['timestamp'].iloc[s['cur']])
                w = d5a[(d5a['timestamp'] >= max(t0, seen_to + 1))
                        & (d5a['timestamp'] < t0 + 60)] \
                    if d5a is not None else None
                if w is not None and len(w):
                    for lv in alarms:
                        tag = w[(w['low'] <= lv) & (w['high'] >= lv)]
                        if len(tag):
                            rang, rlvl = int(tag.iloc[0]['timestamp']), lv
                            break
                if rang is not None:
                    break
            if rang is not None:
                tstr = pd.to_datetime(rang, unit='s').strftime('%H:%M:%S')
                kind = ('WARN-STOP' if rlvl == p_ws.get('warn_stop') else 'ALARM')
                held = ' — position still OPEN' if s.get('pos') else ''
                ev.append(f"*** {kind} {rlvl:.2f} touched at {tstr} — HALTED"
                          f"{held}, your call ***")
                s['halt_ts5'] = rang           # truncate the frame AT the touch
                _log(s, 'alarm', price=rlvl, ts5=rang, kind=kind)
        else:
            ev = _advance(s, df, n, stop_on_fill=a.until_fill)
        # decision-binding corpus: stepping while FLAT = the owner SAW these
        # frames and passed — negative examples matter as much as entries.
        # BOTH feature snapshots ride along silently for triangulation:
        # fspace = the ACTUAL (V2 layer) F-space; combiner = the 22-stream vec.
        ts_now = int(df['timestamp'].iloc[s['cur']])
        _log(s, 'step', frm=frm, to=s['cur'],
             passed=bool(was_flat and s.get('pos') is None),
             fspace=_v2_fspace(s['day'], ts_now),
             combiner=_fspace_snapshot(s['day'], ts_now))
        _save(s)
        _render(s, df)
        cap = ' | '.join(ev) if ev else f"advanced {n} bars"
        bl = _bar_line(s, df)
        if bl:
            cap += '\n' + bl
        osc = _osc_line(s, df)
        if osc:
            cap += '\n' + osc
        fsl = _fs_line(s, df)
        if fsl:
            cap += '\n' + fsl
        if a.send:
            _send(cap)
        print(cap)
    elif a.cmd == 'chart':
        _render(s, df)
        fsl = _fs_line(s, df)
        if a.send:
            _send('current frame' + ('\n' + fsl if fsl else ''))
        print(PNG + (('\n' + fsl) if fsl else ''))
    elif a.cmd == 'sigma':
        s['sigma_w'] = int(a.rest[0]); _save(s); _log(s, 'sigma', w=s['sigma_w'])
        _render(s, df)
        if a.send:
            _send(f"σW={s['sigma_w']}")
        print(f"σW={s['sigma_w']}")
    elif a.cmd == 'wakeup':
        # owner 2026-07-30: "like a wakeup timer?" -- threshold (points of
        # high-low range) that makes a big `step N` stop early rather than
        # silently blow through a real move. See WAKEUP_PT_DEFAULT.
        wk = float(a.rest[0]) if a.rest else WAKEUP_PT_DEFAULT
        s['wakeup_pt'] = wk; _save(s); _log(s, 'wakeup', pt=wk)
        if a.send:
            _send(f"wakeup threshold: {wk:.1f}pt")
        print(f"wakeup threshold: {wk:.1f}pt")
    elif a.cmd == 'warnstop':
        # STOP-AS-WARNING (owner 2026-07-31: "if we make the -10pt stop a
        # warning also? since we are doing this to distill and generalize for
        # an ML, it should be able to handle the split second decision").
        # Stored as 'warn_stop', deliberately NOT 'stop', so _advance can never
        # see or act on it -- the step loop halts on it at 5s precision and the
        # HUMAN decides. Every touch becomes a labeled decision point instead
        # of a mechanical event; that is the training signal a hard stop erases.
        p = s.get('pos')
        if not p:
            print('no open position'); return
        px = float(a.rest[0])
        d = 1 if p['dir'] == 'long' else -1
        wrong_side = (d > 0 and px > p['entry']) or (d < 0 and px < p['entry'])
        p.pop('stop', None)                 # a warn-stop REPLACES any hard stop
        p['warn_stop'] = px
        _save(s); _log(s, 'warnstop_set', price=px, wrong_side=wrong_side)
        print(f"WARN-stop armed @ {px:.2f} — halts the sim, does NOT exit"
              + (" -- WARNING: profit side of entry" if wrong_side else ""))
    elif a.cmd == 'stop':
        # attach/modify a stop-loss on the ALREADY-OPEN position (owner
        # 2026-07-30: "we should add stop loss capabilities" -- call already
        # accepted --stop on ENTRY, but there was no way to set one after the
        # fact on a live position, which is the real gap).
        p = s.get('pos')
        if not p:
            print('no open position to stop'); return
        px = float(a.rest[0])
        d = 1 if p['dir'] == 'long' else -1
        # A hard stop SUPERSEDES any warn-stop. Holding both meant the warn
        # fired first (it sat nearer the market) and the hard stop never got a
        # chance -- silently reverting to halt-and-ask when an exit was asked
        # for (owner 2026-08-01, moving from an 80% warning to a 50% stop).
        if p.pop('warn_stop', None) is not None:
            print('(cleared the warn-stop -- a hard stop supersedes it)')
        # "wrong side" means BEYOND THE MARKET in the losing direction. A stop
        # inside the open profit is a legitimate profit-lock, not an error; the
        # old test compared against ENTRY and so flagged every trailing stop.
        cur_px = float(df['close'].iloc[s['cur']])
        wrong_side = (d > 0 and px > cur_px) or (d < 0 and px < cur_px)
        locks = ((d > 0 and p['entry'] < px <= cur_px)
                 or (d < 0 and cur_px <= px < p['entry']))
        p['stop'] = px; p['stop_reverse'] = False
        _save(s); _log(s, 'stop_set', price=px, wrong_side=wrong_side,
                       locks_profit=locks)
        note = ''
        if wrong_side:
            note = (f" -- WARNING: already PAST the market ({cur_px:.2f}); "
                    "this fills immediately on the next bar")
        elif locks:
            lock_pt = abs(p['entry'] - px)
            note = (f" -- profit-lock: guarantees {lock_pt:.2f}pt "
                    f"(${lock_pt * PT_USD:.2f}) if hit")
        print(f"stop set @ {px:.2f}{note}")
    elif a.cmd == 'stopreverse':
        # owner 2026-07-30: "a potential strategy ... stop and reverse, the
        # trick is to add 2 opposite orders instead of 1" -- on STOP hit
        # (adverse move only, never on a target hit), close AND immediately
        # open the OPPOSITE position at the same trigger price, catching a
        # violent reversal as a new trade instead of just a loss. NT8-side
        # equivalent: 2 STOP orders (not limit -- a limit won't trigger on an
        # adverse breakout), one to close + one to open opposite, at the same
        # level.
        p = s.get('pos')
        if not p:
            print('no open position to stopreverse'); return
        px = float(a.rest[0])
        d = 1 if p['dir'] == 'long' else -1
        wrong_side = (d > 0 and px > p['entry']) or (d < 0 and px < p['entry'])
        p['stop'] = px; p['stop_reverse'] = True
        _save(s); _log(s, 'stopreverse_set', price=px, wrong_side=wrong_side)
        print(f"stop-and-reverse armed @ {px:.2f}" + (" -- WARNING: on the PROFIT "
              f"side of entry ({p['entry']:.2f})" if wrong_side else ""))
    elif a.cmd == 'bankreenter':
        # owner 2026-07-30: cascading extension of stop-and-reverse -- "pair
        # it with a 2buy at $10, we bank and reenter with protection." On a
        # TARGET hit (favorable move only, mirrors stopreverse's stop-only
        # trigger), bank the profit AND immediately re-enter the SAME
        # direction at that price with a clean slate (owner sets fresh
        # stop/target after, same as any new entry).
        p = s.get('pos')
        if not p:
            print('no open position to bankreenter'); return
        px = float(a.rest[0])
        d = 1 if p['dir'] == 'long' else -1
        wrong_side = (d > 0 and px < p['entry']) or (d < 0 and px > p['entry'])
        p['target'] = px; p['bank_reenter'] = True
        _save(s); _log(s, 'bankreenter_set', price=px, wrong_side=wrong_side)
        print(f"bank-and-reenter armed @ {px:.2f}" + (" -- WARNING: on the LOSS "
              f"side of entry ({p['entry']:.2f}), this isn't a favorable target"
              if wrong_side else ""))
    elif a.cmd == 'tag':
        # label the current episode in the corpus (owner 2026-08-04: "log as
        # experimental") — the label rides on every subsequent log row via
        # state, so the training corpus can filter episodes by intent.
        s['episode_tag'] = ' '.join(a.rest) if a.rest else None
        _save(s); _log(s, 'episode_tag', tag=s['episode_tag'])
        print('episode tag:', s['episode_tag'])
    elif a.cmd == 'tele':
        # switch the telescope sub-panel's resolution (owner 2026-07-30:
        # "start looking at 30s windows, even 15s, with the 1m in view")
        res = a.rest[0] if a.rest else '5s'
        assert res in ('1s', '5s', '15s', '30s', '1h'), 'tele 1s|5s|15s|30s|1h'
        s['tele_res'] = res
        # optional 2nd arg = span in SECONDS for both sub-panels
        if len(a.rest) > 1:
            sp = int(a.rest[1])
            assert 30 <= sp <= 1800, 'tele span 30..1800 s'
            s['tele_span'] = sp
        _save(s); _log(s, 'tele_res', res=res, span=s.get('tele_span', 180))
        _render(s, df)
        span_txt = f" · last {s.get('tele_span', 180)}s"
        if a.send:
            _send(f"telescope: {res}{span_txt}")
        print(f'telescope resolution: {res}{span_txt}')
    elif a.cmd == 'mainview':
        # switch the MAIN panel's mode (owner 2026-07-30: "I want to see in
        # the main panel the last 4 days" -- '1m' = normal detail view
        # (candles/cubic/sigma-bands/fog), '4d' = macro 1h/4-day telescope
        # view. Independent of `tele` (the second/switchable panel).
        mv = a.rest[0] if a.rest else '1m'
        assert mv in ('1m', '4d'), 'mainview 1m|4d'
        s['main_view'] = mv; _save(s); _log(s, 'main_view', view=mv)
        _render(s, df)
        if a.send:
            _send(f"main view: {mv}")
        print(f'main view: {mv}')
    elif a.cmd == 'prevday':
        back = int(a.rest[0]) if a.rest else 4
        path, cap = _render_prevday(s, back)
        if path is None:
            print(cap); return
        _log(s, 'prevday', back=back)
        if a.send:
            _send(cap, path)
        print(cap)
    elif a.cmd == 'month':
        # month-to-date (owner 2026-08-01: "now let's render the month").
        # Sessions of s['day']'s OWN month up to and including today -- files
        # AFTER today exist on disk and must never be drawn; that would be
        # straight lookahead. Rendered at 15m by default: ~11 sessions of 1m
        # is ~15k bars, which smears into a solid block.
        res = a.rest[0] if a.rest else '15m'
        mon = s['day'][:7]
        allf = sorted(f[:-8] for f in os.listdir(DATA) if f.endswith('.parquet'))
        back = sum(1 for d in allf if d[:7] == mon and d < s['day'])
        if not back:
            print(f'{s["day"]} is the first session of {mon}'); return
        path, cap = _render_prevday(s, back, res=res, label=f'{mon} MONTH-TO-DATE')
        if path is None:
            print(cap); return
        _log(s, 'month', month=mon, sessions=back + 1, res=res)
        if a.send:
            _send(cap, path)
        print(cap)
    elif a.cmd == 'watch':
        # LIVE WATCHER (owner 2026-08-01: "as a live watcher ... we will
        # advance in 1s increments"). Walks forward one SECOND at a time and
        # halts the instant z reaches a band -- the only moment the harvest
        # decision is live. Mid-band there is nothing to decide, so stepping
        # by minutes would blow straight past the actionable instant, which is
        # the same failure the alarm was built to fix.
        budget = int(a.rest[0]) if a.rest else 300
        st0 = _osc_state(s, df)
        armed = st0 is not None and abs(st0[0]) < OSC_BAND   # must LEAVE first
        hit, ev, ws_hit = None, [], None
        # the live watcher must also honour a WARN-STOP (owner 2026-08-01:
        # "we would do a 80% of current profit warning marker, set the watcher
        # and advance 60s or if it triggers the warning"). Previously only the
        # band could halt `watch`, so a giveback marker armed on the position
        # would have been walked straight past.
        pw = s.get('pos') or {}
        ws = pw.get('warn_stop')
        d1w = _bars_tele(s['day'], '1s')
        # ALSO halt on the owner's REFERENCE REGIONS (owner 2026-08-01: "the
        # instruction was wrong, you ran it until it returned to 80% instead of
        # stopping when it started to reach the first level region"). Halting
        # only on sigma bands is blind in exactly the case that matters: enter
        # AT a band and the edge-trigger must LEAVE and re-arm first, so the
        # whole leg toward the next level passes unwatched. Regions are the
        # levels he actually trades to.
        t0w = int(s.get('halt_ts5') or (int(df['timestamp'].iloc[s['cur']]) + 55
                                        + s.get('peek_offset', 0)))
        regions = []
        for L in s.get('owner_lines', []):
            r = _level_region(s, L, t0w)
            if r:
                regions.append((L, r[0], r[1]))
        in_reg = {L for L, lo_, hi_ in regions
                  if d1w is not None and len(d1w[d1w['timestamp'] <= t0w])
                  and lo_ <= float(d1w[d1w['timestamp'] <= t0w]['close'].iloc[-1]) <= hi_}
        reg_hit = None
        # GIVEBACK is a PATH trigger, not a level trigger (owner 2026-08-01:
        # "we should have stopped at 10:11:15"). Neither the sigma band nor a
        # region entry marks that instant -- price was mid-band and had just
        # left the region going up. What marks it is the first real pullback
        # from the position's own running best, and a leg top is a path event.
        gb = a.giveback
        gb_hit = None
        # STALL STAMP (owner 2026-08-01: "just measure pure displacement, the
        # watcher should have stamped the 2 times it stalled"). A stall is not
        # a level and not a retrace -- it is the ABSENCE of new favourable
        # extremes. On the long it marked 10:10:58 (+14.25) and 10:11:14
        # (+17.25, the top); both were actionable and neither band, region nor
        # giveback trigger names them.
        stall_s = a.stall
        stall_hit = None
        best_ext = None      # running favourable extreme
        since_ext = 0        # seconds since it last improved
        pdir = 1 if (pw.get('dir') == 'long') else -1
        best_px = None
        for k in range(budget):
            t_before = int(s.get('halt_ts5')
                           or (int(df['timestamp'].iloc[s['cur']]) + 55
                               + s.get('peek_offset', 0)))
            ev += _peek_1s(s, df)
            if ws is not None and d1w is not None:
                t_now = int(s.get('halt_ts5')
                            or (int(df['timestamp'].iloc[s['cur']]) + 55
                                + s.get('peek_offset', 0)))
                seg = d1w[(d1w['timestamp'] > t_before) & (d1w['timestamp'] <= t_now)]
                if len(seg) and ((seg['low'] <= ws) & (seg['high'] >= ws)).any():
                    ws_hit = (k + 1, t_now)
                    break
            if stall_s and pw and d1w is not None:
                t_now = int(s.get('halt_ts5')
                            or (int(df['timestamp'].iloc[s['cur']]) + 55
                                + s.get('peek_offset', 0)))
                seg = d1w[(d1w['timestamp'] > t_before) & (d1w['timestamp'] <= t_now)]
                if len(seg):
                    edge = (float(seg['high'].max()) if pdir > 0
                            else float(seg['low'].min()))
                    improved = (best_ext is None
                                or (edge - best_ext) * pdir > 0)
                    if improved:
                        best_ext = edge; since_ext = 0
                    else:
                        since_ext += 1
                        if since_ext >= stall_s:
                            stall_hit = (k + 1, best_ext, float(seg['close'].iloc[-1]),
                                         since_ext)
                            break
            if gb is not None and pw and d1w is not None:
                t_now = int(s.get('halt_ts5')
                            or (int(df['timestamp'].iloc[s['cur']]) + 55
                                + s.get('peek_offset', 0)))
                seg = d1w[(d1w['timestamp'] > t_before) & (d1w['timestamp'] <= t_now)]
                if len(seg):
                    edge = float(seg['high'].max()) if pdir > 0 else float(seg['low'].min())
                    best_px = edge if best_px is None else (
                        max(best_px, edge) if pdir > 0 else min(best_px, edge))
                    px_now = float(seg['close'].iloc[-1])
                    if (best_px - px_now) * pdir >= gb:
                        gb_hit = (k + 1, best_px, px_now)
                        break
            if regions and d1w is not None:
                t_now = int(s.get('halt_ts5')
                            or (int(df['timestamp'].iloc[s['cur']]) + 55
                                + s.get('peek_offset', 0)))
                seg = d1w[(d1w['timestamp'] > t_before) & (d1w['timestamp'] <= t_now)]
                if len(seg):
                    px_now = float(seg['close'].iloc[-1])
                    for L, lo_, hi_ in regions:
                        inside = lo_ <= px_now <= hi_
                        if inside and L not in in_reg:      # edge-triggered ENTRY
                            reg_hit = (k + 1, L, lo_, hi_, px_now)
                            break
                        if not inside:
                            in_reg.discard(L)
                    if reg_hit:
                        break
            st = _osc_state(s, df)
            if st is None:
                continue
            if not armed:
                if abs(st[0]) < OSC_BAND:
                    armed = True                  # edge-trigger, never level
                continue
            if abs(st[0]) >= OSC_BAND:
                hit = (k + 1, st)
                break
        _save(s); _render(s, df)
        line = _osc_line(s, df) or ''
        if stall_hit:
            k, best, px_now, secs = stall_hit
            p_ = s['pos']; d_ = 1 if p_['dir'] == 'long' else -1
            cap = (f"*** STALL {secs}s after {k}s [attention only, not an edge] "
                   f"-- no new extreme past "
                   f"{best:.2f} ({(best - p_['entry']) * d_:+.2f}pt). Now "
                   f"{px_now:.2f} ({(px_now - p_['entry']) * d_:+.2f}pt). "
                   f"Your call ***")
            _log(s, 'stall_hit', best=best, price=px_now, stalled_s=secs, secs=k)
        elif gb_hit:
            k, best, px_now = gb_hit
            p_ = s['pos']; d_ = 1 if p_['dir'] == 'long' else -1
            cap = (f"*** GIVEBACK {gb:.2f}pt after {k}s -- best {best:.2f}, "
                   f"now {px_now:.2f}, open P&L {(px_now - p_['entry']) * d_:+.2f}pt. "
                   f"Your call ***")
            _log(s, 'giveback_hit', mfe_px=best, price=px_now, secs=k, thresh=gb)
        elif reg_hit:
            k, L, lo_, hi_, px_now = reg_hit
            cap = (f"*** REGION {L:.2f} ENTERED after {k}s -- price {px_now:.2f} "
                   f"inside {lo_:.2f}-{hi_:.2f}. Your level, your call ***")
            _log(s, 'region_hit', level=L, lo=lo_, hi=hi_, price=px_now, secs=k)
        elif ws_hit:
            k, t_now = ws_hit
            p_ = s['pos']; d_ = 1 if p_['dir'] == 'long' else -1
            cap = (f"*** WARN-STOP {ws:.2f} touched after {k}s -- position "
                   f"still OPEN, your call. Giveback marker hit ***")
            _log(s, 'warn_stop_hit', price=ws, secs=k)
        elif hit:
            k, st = hit
            side = 'HIGH' if st[0] > 0 else 'LOW'
            # say whether this band is FAVOURABLE or ADVERSE for the OPEN
            # position (owner 2026-08-01: "how come you stopped at +1.25?").
            # The trigger is direction-agnostic, so a long gets halted at the
            # LOW band -- an adverse event -- with the same wording as a
            # profit-taking touch. Naming it stops a bad-news halt reading as
            # an opportunity.
            tag = ''
            if s.get('pos'):
                d_ = 1 if s['pos']['dir'] == 'long' else -1
                fav = (st[0] > 0 and d_ > 0) or (st[0] < 0 and d_ < 0)
                tag = ('  [FAVOURABLE for your ' + s['pos']['dir'] + ']'
                       if fav else
                       '  [ADVERSE for your ' + s['pos']['dir']
                       + ' -- this is the wrong end of the range]')
            cap = (f"*** BAND TOUCH ({side}) after {k}s -- z{st[0]:+.2f}, "
                   f"K={st[2]}.{tag} Your call ***")
            _log(s, 'watch_band', z=st[0], K=st[2], secs=k, side=side)
        else:
            cap = f"advanced {budget}s -- no band touch"
        if ev:
            cap = ' | '.join(ev) + '\n' + cap
        cap += '\n' + line
        if a.send:
            _send(cap)
        print(cap)
    elif a.cmd == 'view':
        # main-panel span in 1m bars (owner 2026-08-01: "I got lost in the
        # sense of scale, let's only see the last 20 minutes of tape on 1m").
        # A fixed 45-bar window silently changes what "recent" means as the
        # session moves; making it explicit keeps the read anchored.
        n = int(a.rest[0]) if a.rest else VIEW
        assert 5 <= n <= 400, 'view 5..400 bars'
        s['view_bars'] = n; _save(s); _log(s, 'view', bars=n)
        _render(s, df)
        if a.send:
            _send(f'main view: last {n} 1m bars')
        print(f'main panel: last {n} 1m bars')
    elif a.cmd == 'region':
        cut = int(s.get('halt_ts5')
                  or (int(df['timestamp'].iloc[s['cur']]) + 55
                      + s.get('peek_offset', 0)))
        lv = float(a.rest[0]) if a.rest else None
        levels = [lv] if lv is not None else list(s.get('owner_lines', []))
        if not levels:
            print('no level given and no owner lines set'); return
        for L in levels:
            r = _level_region(s, L, cut)
            if not r:
                print(f'{L:.2f}: too few observations nearby'); continue
            lo_, hi_, n_, hw, skew = r
            print(f'{L:.2f}: region {lo_:.2f}-{hi_:.2f}  ±{hw:.2f}pt  '
                  f'({REGION_MASS:.0%} of {n_} 1s closes within '
                  f'±{REGION_SEARCH_PT:g}pt, last {REGION_LOOKBACK_S // 60}min) '
                  f'| density sits {abs(skew):.2f}pt '
                  f'{"ABOVE" if skew > 0 else "BELOW"} the level')
            _log(s, 'region', level=L, lo=lo_, hi=hi_, n=n_, hw=hw, skew=skew)
    elif a.cmd == 'osc':
        line = _osc_line(s, df)
        if not line:
            print('not enough bars for the oscillation watcher yet'); return
        st = _osc_state(s, df)
        cut = int(s.get('halt_ts5')
                  or (int(df['timestamp'].iloc[s['cur']]) + 55
                      + s.get('peek_offset', 0)))
        recent = [t for t in st[3] if t >= cut - OSC_LOOKBACK_S]
        print(line)
        for t in recent[-8:]:
            print(f"   traverse @ {pd.to_datetime(t, unit='s', utc=True).tz_convert('America/New_York'):%H:%M:%S} ET")
        _log(s, 'osc', z=st[0], K=st[2], band_pt=st[1])
        if a.send:
            _render(s, df); _send(line)
    elif a.cmd == 'protect':
        # protect on [warn hard min_mfe] / protect region PRICE / protect
        # hard / protect off / protect status
        pr = s.get('protect') or dict(PROTECT_DEFAULTS)
        sub = a.rest[0] if a.rest else 'status'
        if sub == 'on':
            pr['on'] = True
            if len(a.rest) > 1: pr['warn'] = float(a.rest[1])
            if len(a.rest) > 2: pr['hard'] = float(a.rest[2])
            if len(a.rest) > 3: pr['min_mfe'] = float(a.rest[3])
        elif sub == 'region':
            pr['region'] = float(a.rest[1]); pr['arm'] = 'region'
        elif sub == 'always':
            pr['arm'] = 'always'
        elif sub == 'hard':
            p = s.get('pos')
            if p:
                p['prot_hard'] = True
                if p.get('frozen'):
                    print(f"70 hard = {p['frozen'] * pr.get('hard', .7):+.2f}pt "
                          f"retention floor (frozen peak {p['frozen']:+.2f})")
            print('70 hard line ENABLED for the open position')
        elif sub == 'entrywarn':
            pr['entry_warn'] = not (len(a.rest) > 1 and a.rest[1] == 'off')
            print(f"entry-touch warning {'ON' if pr['entry_warn'] else 'OFF'}")
        elif sub == 'lock':
            # protect lock FRAC | protect lock off — the ladder TRAILS this
            # fraction of peak once the trigger is cleared
            lad = pr.get('ladder') or dict(trigger=5.0, jump=2.0)
            if len(a.rest) > 1 and a.rest[1] == 'off':
                lad.pop('lock', None)
                print('lock OFF — ladder back to flat BE+%g' %
                      lad.get('jump', 2.0))
            else:
                lad['lock'] = float(a.rest[1])
                print(f"lock ARMED: once MFE >= {lad['trigger']:g}, stop "
                      f"trails {lad['lock']:.0%} of peak")
            pr['ladder'] = lad
        elif sub == 'warnat':
            # protect warnat N | protect warnat off — conditional give-back
            # warning on the OPEN position: exceed +N, then return to it
            p = s.get('pos')
            if not p:
                print('no open position')
            elif len(a.rest) > 1 and a.rest[1] == 'off':
                p.pop('warn_gb', None); p.pop('warn_gb_fired', None)
                print('give-back warning OFF')
            else:
                p['warn_gb'] = float(a.rest[1])
                p.pop('warn_gb_fired', None)
                d0 = 1 if p['dir'] == 'long' else -1
                print(f"give-back warning ARMED at "
                      f"{p['entry'] + d0 * p['warn_gb']:.2f} "
                      f"(+{p['warn_gb']:g}), fires after the trade exceeds it")
        elif sub == 'clock':
            # protect clock RATE [GRACE_S] | protect clock off
            if len(a.rest) > 1 and a.rest[1] == 'off':
                pr.pop('clock', None)
                print('clock OFF')
            else:
                pr['clock'] = dict(rate=float(a.rest[1]),
                                   grace=float(a.rest[2])
                                   if len(a.rest) > 2 else 0.0)
                print(f"clock ARMED: stop walks {pr['clock']['rate']:g}pt/s "
                      f"toward entry after {pr['clock']['grace']:g}s grace")
        elif sub == 'ladder':
            # protect ladder TRIGGER [JUMP] | protect ladder off
            # trigger: MFE pts that arm the jump (owner: 2 normal, 5+ when
            # bracing for a fake). jump: stop lands at entry+jump (BE+2
            # default).
            if len(a.rest) > 1 and a.rest[1] == 'off':
                pr.pop('ladder', None)
                print('ladder OFF')
            else:
                pr['ladder'] = dict(trigger=float(a.rest[1]),
                                    jump=float(a.rest[2])
                                    if len(a.rest) > 2 else 2.0)
                print(f"ladder ARMED: MFE >= {pr['ladder']['trigger']:g} "
                      f"-> stop to entry{pr['ladder']['jump']:+g}")
        elif sub == 'rearm':
            # owner protocol verb (2026-08-03, at the first live engine
            # freeze): release the freeze but keep the 80-machine hot at the
            # STANDING peak — "give it breathing room; anything that pokes
            # <80% stops the tape again". Differs from engine-native extend
            # (freeze holds until a NEW MFE): after rearm the very next
            # wick through the 80 line re-freezes.
            p = s.get('pos')
            if p and p.get('frozen') is not None:
                pk = p.get('peak', 0.0)
                p['frozen'] = None
                _log(s, 'prot_rearm', peak=pk)
                print(f'freeze RELEASED — 80 machine hot at peak {pk:+.2f}; '
                      f'next poke through the line re-freezes')
            else:
                print('nothing frozen')
        elif sub == 'off':
            pr['on'] = False
        elif sub == 'milestone':
            pr['milestone'] = dict(level=float(a.rest[1]),
                                   by_ts=int(a.rest[2]))
        s['protect'] = pr; _save(s); _log(s, 'protect_cfg', **{k: v for k, v
            in pr.items() if k != 'milestone'})
        print('protect:', pr)
    elif a.cmd == 'run':
        secs = int(a.rest[0]) if a.rest else 60
        ev, halted = _engine_run(s, df, secs, alarms=(a.alarm or []))
        _render(s, df)
        cap = ' | '.join(ev[-3:])
        osc = _osc_line(s, df)
        if osc: cap += '\n' + osc
        if a.send: _send(cap)
        print('\n'.join(ev))
    elif a.cmd == 'call':
        # DEFAULT FILL = immediate, at the bar currently shown (owner rule
        # 2026-07-29: "it will always be the next 1s bar you show" — we have
        # no 1s data loaded, so the honest fill is THIS bar's close, not a
        # deferred 1m open; reacting to typing-latency was the bug).
        d = a.rest[0]
        assert d in ('long', 'short')
        # Fill priority: --at (explicit level) > frozen-instant 1s close
        # (when halted) > committed 1m close. Without --at, an entry armed
        # at an alarm level used to fill at the committed 1m close -- 10pt+
        # away mid-peek, silently fabricating P&L (owner 2026-08-01); the
        # audit then flagged the post-halt default as the same lookahead.
        px = (float(a.at) if a.at is not None
              else (_halt_px(s) or float(df['close'].iloc[s['cur']])))
        if s.get('pos') and not s['pos'].get('pending'):
            old = s['pos']; do = 1 if old['dir'] == 'long' else -1
            pts = (px - old['entry']) * do - FRICTION_PT
            s['pnl_pts'] = s.get('pnl_pts', 0.0) + pts
            s['trades'] = s.get('trades', 0) + 1
            _log(s, 'close', reason='reverse', price=px, pts=round(pts, 2))
            # entry_ts/stop0 feed the clock ratchet (it walks the stop from
            # its ORIGINAL level at N pt/s of elapsed tape time)
            s['pos'] = dict(dir=d, pending=False, entry=px, entry_bar=s['cur'],
                            target=a.target, stop=a.stop, stop0=a.stop,
                            entry_ts=int(s.get('halt_ts5')
                                         or df['timestamp'].iloc[s['cur']]))
            _save(s); _log(s, 'fill', dir=d, price=px)
            print(f'REVERSED: closed {old["dir"]} {pts:+.2f}pt, {d} FILLED @ {px:.2f}')
        else:
            # entry_ts/stop0 feed the clock ratchet (it walks the stop from
            # its ORIGINAL level at N pt/s of elapsed tape time)
            s['pos'] = dict(dir=d, pending=False, entry=px, entry_bar=s['cur'],
                            target=a.target, stop=a.stop, stop0=a.stop,
                            entry_ts=int(s.get('halt_ts5')
                                         or df['timestamp'].iloc[s['cur']]))
            s['exit_next'] = False
            _save(s); _log(s, 'call', dir=d, price=px, target=a.target, stop=a.stop)
            print(f'{d} FILLED @ {px:.2f} (target={a.target} stop={a.stop})')
        # protect config outlives trades by design (the ratchet is a standing
        # instruction) — but a NEW position silently inheriting arm=always
        # from an old experiment can freeze/70-exit a trade the owner never
        # asked to protect (audit item 5). Announce it at fill time.
        prb = s.get('protect') or {}
        if prb.get('on'):
            print(f"NOTE: PROTECT ACTIVE on this position (arm="
                  f"{prb.get('arm')}, warn={prb.get('warn')}, hard="
                  f"{prb.get('hard')}, min_mfe={prb.get('min_mfe')}) — "
                  f"'protect off' if unwanted")
    elif a.cmd == 'exit':
        p = s.get('pos')
        if not p or p.get('pending'):
            print('nothing to exit'); return
        # Fill priority: --at (explicit) > frozen-instant 1s close (when
        # halted) > committed bar close. The bar close after a mid-minute
        # halt is up to ~59s of FUTURE — it booked an exit 9.75pt off once
        # (2026-08-02, corrected in state) and the audit flagged the default
        # as a standing lookahead fill.
        px = (float(a.at) if a.at is not None
              else (_halt_px(s) or float(df['close'].iloc[s['cur']])))
        d = 1 if p['dir'] == 'long' else -1
        pts = (px - p['entry']) * d - FRICTION_PT
        s['pnl_pts'] = s.get('pnl_pts', 0.0) + pts
        s['trades'] = s.get('trades', 0) + 1
        s['pos'] = None
        s.setdefault('exit_marks', []).append(
            [int(s.get('halt_ts5') or df['timestamp'].iloc[s['cur']]), px])
        _save(s); _log(s, 'close', reason='manual', price=px, pts=round(pts, 2))
        print(f'EXIT {p["dir"]} @ {px:.2f} -> {pts:+.2f}pt (${pts*PT_USD:+.2f})')
    elif a.cmd == 'line':
        # add owner level(s); 'line clear' wipes them
        if a.rest and a.rest[0] == 'clear':
            s['owner_lines'] = []
        else:
            s.setdefault('owner_lines', []).extend(float(x) for x in a.rest)
        _save(s); _log(s, 'owner_line', lines=s['owner_lines'])
        _render(s, df)
        if a.send:
            _send('owner lines: ' + ', '.join(f'{x:.1f}' for x in s['owner_lines']))
        print('owner lines:', s['owner_lines'])
    elif a.cmd == 'who':
        # attribution toggle: whose decisions are being logged (owner default;
        # 'claude' during self-tests so the 60-owner-leg corpus stays clean)
        w = a.rest[0] if a.rest else 'owner'
        assert w in ('owner', 'claude')
        s['who'] = w; _save(s); _log(s, 'who_set')
        print(f'decisions now attributed to: {w}')
    elif a.cmd == 'progress':
        import sqlite3
        con = sqlite3.connect(os.path.join(REPO, 'research', 'dojo_forge',
                                           'gate_state', 'pocket_dojo.db'))
        legs = con.execute("SELECT COUNT(*) FROM events WHERE event='close' "
                           "AND who='owner'").fetchone()[0]
        notes = con.execute("SELECT COUNT(*) FROM events WHERE event='note' "
                            "AND who='owner'").fetchone()[0]
        days_played = con.execute("SELECT COUNT(DISTINCT day) FROM events "
                                  "WHERE who='owner' AND event='close'").fetchone()[0]
        pnl = con.execute("SELECT COALESCE(SUM(pts),0) FROM events WHERE "
                          "event='close' AND who='owner'").fetchone()[0]
        TARGET = 60
        bar_n = int(min(1.0, legs / TARGET) * 20)
        print(f"OWNER CORPUS: [{'#'*bar_n}{'.'*(20-bar_n)}] {legs}/{TARGET} legs "
              f"· {notes} narrations · {days_played} days · {pnl:+.1f}pt lifetime")
        con.close()
    elif a.cmd == 'fsline':
        # compact one-liner for frame captions (owner: pepper F-space onto the
        # phone frames so the background brain grazes on it)
        print(_fs_line(s, df) or '(no F-space here)')
    elif a.cmd == 'fspace':
        # conscious look at the combiner state for the current slice
        fs = _fspace_snapshot(s['day'], int(df['timestamp'].iloc[s['cur']]))
        if fs is None:
            print('no F-space here (outside RTH window or uncovered day)')
        else:
            _log(s, 'fspace_view', fspace=fs)
            txt = (f"F-SPACE S{s.get('slice','?')}: P_topk={fs['P_topk']} P_any={fs['P_any']} "
                   f"gov={fs['gov_stream']}({fs['gov_dir']:+d}) fires={fs['n_fires']} "
                   f"zz_leg={fs['zz_leg']:+.0f} confirm={fs['zz_confirm']} age={fs['zz_age_min']:.0f}m "
                   f"top={fs['top_streams']}")
            print(txt)
            if a.send:
                _send(txt)
    elif a.cmd == 'verdict':
        # owner grades the most recent CLAUDE decision: agree|partial|disagree
        # [free text why]. The agreement RATE over claude-legs is the
        # distillation-readiness gate (pre-registered threshold; agreement
        # gates transfer, P&L gates edge — never merged).
        v = a.rest[0] if a.rest else None
        assert v in ('agree', 'partial', 'disagree'), 'verdict agree|partial|disagree [why]'
        why = ' '.join(a.rest[1:])
        _log(s, 'verdict', verdict=v, text=why)
        print(f'verdict logged: {v}' + (f' — {why}' if why else ''))
    elif a.cmd == 'note':
        txt = ' '.join(a.rest)
        _log(s, 'note', text=txt)
        print('noted')
    elif a.cmd == 'score':
        out = dict(trades=s.get('trades', 0),
                   pnl_pts=round(s.get('pnl_pts', 0.0), 2),
                   pnl_usd=round(s.get('pnl_pts', 0.0) * PT_USD, 2))
        _log(s, 'score', **out)               # _log adds day/bar itself
        print(json.dumps({'day': s['day'], 'bar': s['cur'], **out}, indent=1))
    else:
        raise SystemExit(f'unknown cmd {a.cmd}')


if __name__ == '__main__':
    main()
