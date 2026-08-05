"""ADVERSARIAL AUDIT of research/event_onset/tools/velocity_legs.py (2026-08-04).

Does NOT modify the target. Replicates it bit-for-bit, then re-measures under
corrected constructions:

  a. entry-point conflation   -- entry at the leg START (t-T) and midpoints
  b. follow-window artifact   -- horizon sweep 30/60/120/300/600s
  c. correlated samples       -- move clustering + DAY-CLUSTERED CIs
  d. sign conventions         -- independent recomputation of dd/run/MAE/MFE
  e. selection/survivorship   -- end-of-mask truncation, dropped days, ties
  f. is 49% a null            -- the direction-free baseline is (1-P(tie))/2,
                                not 50%; expectancy + day-clustered CI
  g. pre-impulse state        -- causal features measured strictly before t-T,
                                vs time-of-day-matched controls

  python research/event_onset/tools/audit_velocity_legs.py

Writes research/event_onset/reports/audit_velocity_legs.json (all numbers) and
the human report is assembled from it into audit_velocity_legs.md.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = '/media/moi/WindowsCode/Bayesian-AI'
BARS = os.path.join(REPO, 'DATA', 'ATLAS', '1s')
OUT = os.path.join(REPO, 'research', 'event_onset', 'reports')
CACHE = ('/tmp/claude-1000/-media-moi-WindowsCode-Bayesian-AI/'
         '3b0d97ff-121f-49a8-a569-f3e509b65820/scratchpad')
RTH0, RTH1 = 9 * 60 + 30, 15 * 60 + 30
GRID = [(10, 30), (10, 60), (15, 30), (15, 60), (20, 60), (10, 15), (20, 30)]
FOLLOW_S = 300
COOLDOWN_S = 60
FRICTION = 0.89                      # repo canonical round-trip, points
HORIZONS = (30, 60, 120, 300, 600)   # seconds
BOOT = 4000
RNG = np.random.default_rng(20260804)


# ---------------------------------------------------------------- data ------
def load_days():
    """Full-day arrays + the RTH slice bounds. Follow windows may cross RTH1,
    which the original could not do (it clipped to the masked array end)."""
    f = os.path.join(CACHE, 'days.npz')
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        return list(z['days'])
    days = []
    paths = [p for p in sorted(glob.glob(os.path.join(BARS, '2025_0[1-6]*.parquet')))
             if len(os.path.basename(p)) == 18]
    for p in tqdm(paths, desc='load'):
        d = pd.read_parquet(p)
        ts = d['timestamp'].to_numpy()
        et = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
        mod = et.hour * 60 + et.minute
        k = np.flatnonzero((mod >= RTH0) & (mod < RTH1))
        if len(k) == 0:
            days.append(dict(day=os.path.basename(p)[:-8], nrth=0))
            continue
        assert k[-1] - k[0] + 1 == len(k), 'RTH mask not contiguous'
        days.append(dict(
            day=os.path.basename(p)[:-8], nrth=len(k), k0=int(k[0]), k1=int(k[-1]),
            ts=ts, o=d['open'].to_numpy(), h=d['high'].to_numpy(),
            l=d['low'].to_numpy(), c=d['close'].to_numpy(),
            v=d['volume'].to_numpy().astype(np.float64),
            tod=(mod.to_numpy() - RTH0).astype(np.float64)))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(f, days=np.array(days, dtype=object))
    return days


# ------------------------------------------------------------ triggers ------
def triggers(day, D, T):
    """Bit-for-bit replication of velocity_legs.day_impulses trigger logic,
    returning MASKED-array indices (i) so both the original and corrected
    measurements can be built off the same event set."""
    if day['nrth'] < 600:
        return np.empty(0, np.int64)
    k0, n = day['k0'], day['nrth']
    c = day['c'][k0:k0 + n]
    ts = day['ts'][k0:k0 + n]
    disp = np.full(n, np.nan)
    disp[T:] = c[T:] - c[:-T]
    cand = np.flatnonzero(np.abs(disp) >= D)
    cand = cand[(cand >= T) & (cand < n - 1)]          # range(T, n-1)
    keep, last = [], -10 ** 9
    for i in cand:
        if ts[i] - last < COOLDOWN_S:
            continue
        keep.append(i)
        last = ts[i]
    return np.array(keep, np.int64)


def measure(day, idx, T):
    """Every measurement variant for one day's trigger set."""
    k0, n = day['k0'], day['nrth']
    C, H, L, TS, V = day['c'], day['h'], day['l'], day['ts'], day['v']
    N = len(C)
    F = k0 + idx                                        # full-array index of t
    dd = np.sign(C[F] - C[F - T]).astype(np.int64)
    dd[dd == 0] = 1
    r = dict(day=[day['day']] * len(F), i=idx, ts=TS[F], dd=dd,
             disp=np.abs(C[F] - C[F - T]), tod=day['tod'][F])

    # ---- ORIGINAL: masked-array, bar-indexed, clipped at RTH end -----------
    j1 = np.minimum(idx + FOLLOW_S, n - 1) + k0
    r['orig_run'] = (C[j1] - C[F]) * dd
    r['orig_trunc'] = (idx + FOLLOW_S > n - 1)
    mae = np.empty(len(F)); mfe = np.empty(len(F))
    for q, (a, b, s) in enumerate(zip(F, j1, dd)):
        sh, sl = H[a:b + 1], L[a:b + 1]
        mae[q] = (C[a] - sl).max() if s > 0 else (sh - C[a]).max()
        mfe[q] = (sh - C[a]).max() if s > 0 else (C[a] - sl).max()
    r['orig_mae'] = np.maximum(mae, 0.0)
    r['orig_mfe'] = np.maximum(mfe, 0.0)

    # ---- CORRECTED: anchor x horizon, true seconds, no RTH-end clip --------
    anchors = dict(start=F - T, mid=F - T // 2, trig=F)
    for aname, A in anchors.items():
        for Hs in HORIZONS:
            J = np.searchsorted(TS, TS[A] + Hs)
            trunc = J >= N
            J = np.minimum(J, N - 1)
            e = C[A]
            run = (C[J] - e) * dd
            m_a = np.empty(len(F)); m_f = np.empty(len(F)); m_a1 = np.empty(len(F))
            for q, (a, b, s, ee) in enumerate(zip(A, J, dd, e)):
                sh, sl = H[a:b + 1], L[a:b + 1]
                m_a[q] = (ee - sl).max() if s > 0 else (sh - ee).max()
                m_f[q] = (sh - ee).max() if s > 0 else (ee - sl).max()
                sh1, sl1 = H[a + 1:b + 1], L[a + 1:b + 1]
                if len(sh1) == 0:
                    m_a1[q] = 0.0
                else:
                    m_a1[q] = (ee - sl1).max() if s > 0 else (sh1 - ee).max()
            p = f'{aname}_{Hs}'
            r[f'run_{p}'] = run
            r[f'mae_{p}'] = np.maximum(m_a, 0.0)
            r[f'mfe_{p}'] = np.maximum(m_f, 0.0)
            r[f'maex_{p}'] = np.maximum(m_a1, 0.0)      # excl. entry bar's own range
            r[f'trunc_{p}'] = trunc
    # realised seconds spanned by the "T-bar" lookback and the 300-bar follow
    r['lookback_secs'] = (TS[F] - TS[F - T]).astype(np.float64)
    r['follow_secs'] = (TS[j1] - TS[F]).astype(np.float64)

    # ---- pre-impulse state, measured strictly at/before t-T ----------------
    r.update(prestate(day, F - T))
    return r


def prestate(day, A):
    """Causal features at anchor A (full-array index). Uses only [A-W, A]."""
    C, H, L, V = day['c'], day['h'], day['l'], day['v']
    o = {}
    for W, tag in ((60, '60'), (300, '300'), (900, '900')):
        rng = np.empty(len(A)); vol = np.empty(len(A)); rv = np.empty(len(A))
        pos = np.empty(len(A))
        for q, a in enumerate(A):
            s = max(0, a - W)
            hh, ll = H[s:a + 1].max(), L[s:a + 1].min()
            rng[q] = hh - ll
            vol[q] = V[s:a + 1].sum()
            d = np.diff(C[s:a + 1])
            rv[q] = d.std() if len(d) > 1 else 0.0
            pos[q] = (C[a] - ll) / (hh - ll) if hh > ll else 0.5
        o[f'pre_rng{tag}'] = rng
        o[f'pre_vol{tag}'] = vol
        o[f'pre_rv{tag}'] = rv
        o[f'pre_pos{tag}'] = pos
    o['pre_compress'] = o['pre_rng60'] / np.maximum(o['pre_rng300'], 1e-9)
    s = np.maximum(A - 300, 0)
    o['pre_ret300'] = C[A] - C[s]
    o['pre_absret300'] = np.abs(o['pre_ret300'])
    o['pre_tod'] = day['tod'][A]
    return o


# ------------------------------------------------------------- stats --------
def day_boot(vals, days, stat=np.mean, n=BOOT):
    """Day-clustered percentile bootstrap: resample DAYS with replacement."""
    vals = np.asarray(vals, float)
    days = np.asarray(days)
    u, inv = np.unique(days, return_inverse=True)
    groups = [vals[inv == q] for q in range(len(u))]
    pick = RNG.integers(0, len(u), size=(n, len(u)))
    out = np.empty(n)
    for b in range(n):
        out[b] = stat(np.concatenate([groups[q] for q in pick[b]]))
    return float(stat(vals)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def auc(x, y):
    """AUC of x separating y==1 from y==0 (rank based, ties handled)."""
    x = np.asarray(x, float); y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cohend(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


# ------------------------------------------------------------ controls ------
def control_pool(days, trig_by_day, T, stride=15, pad=120):
    """Valid non-impulse anchors: full-array indices whose forward window
    [a, a+T+pad] contains no trigger, keyed by time-of-day minute bin."""
    pool = {}
    for d in days:
        if d['nrth'] < 600:
            continue
        k0, n = d['k0'], d['nrth']
        trg = trig_by_day.get(d['day'], np.empty(0, np.int64))
        cand = np.arange(900, n - 900, stride)          # keep 15min of margin
        if len(trg):
            bad = np.zeros(n, bool)
            for i in trg:
                bad[max(0, i - pad - T):min(n, i + pad + 1)] = True
            cand = cand[~bad[cand]]
        for a in cand:
            pool.setdefault(int(d['tod'][k0 + a]), []).append((d['day'], int(k0 + a)))
    return pool


if __name__ == '__main__':
    days = load_days()
    dmap = {d['day']: d for d in days}
    res = {'meta': dict(n_files=len(days),
                        n_days_with_rth=int(sum(d['nrth'] > 0 for d in days)),
                        n_days_ge600=int(sum(d['nrth'] >= 600 for d in days)),
                        rth_bar_coverage=float(np.mean([d['nrth'] / 21600
                                                        for d in days if d['nrth'] > 0])),
                        friction=FRICTION)}
    cells = {}
    for D, T in GRID:
        tb = {}
        for d in days:
            t = triggers(d, D, T)
            if len(t):
                tb[d['day']] = t
        frames = []
        for dy, idx in tqdm(tb.items(), desc=f'D{D}/T{T}', leave=False):
            frames.append(pd.DataFrame(measure(dmap[dy], idx, T)))
        R = pd.concat(frames, ignore_index=True)
        R.to_parquet(os.path.join(CACHE, f'audit_D{D}_T{T}.parquet'), index=False)
        cells[(D, T)] = R
        print(f'D{D}/T{T}: n={len(R):,} days={R["day"].nunique()}')
    np.save(os.path.join(CACHE, 'cells_keys.npy'), np.array(list(cells.keys())))
    json.dump(res['meta'], open(os.path.join(CACHE, 'meta.json'), 'w'), indent=1)
    print('done')
