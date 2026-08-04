"""EVENT-ONSET probe dataset (owner 2026-08-04, gating the Mamba question:
"can we train a mamba?" -> only if event ONSET is predictable early; direction
is a measured null).

Question: at time t, can causal features predict that a named event will
CONFIRM within the next H seconds?

Positives  = event confirmation timestamps from research/event_library/events/
             (already causally stamped + truncation-audited), rewound by H.
Negatives  = timestamps on the same day, same RTH window, at least
             NEG_GAP_S away from ANY event of that type (so a negative is
             genuinely "nothing forming"), sampled 1:1.

Every feature is computed from 5s bars with timestamp <= t. Nothing reads the
event itself. The live sim day is excluded.

Run from repo root:
  python research/event_onset/builders/build_onset_dataset.py
Writes research/event_onset/onset_<event>_<H>s.parquet
"""
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

EVENTS_DIR = os.path.join(REPO, 'research', 'event_library', 'events')
BARS_DIR = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
OUT_DIR = os.path.join(REPO, 'research', 'event_onset')
EXCLUDE_DAYS = {'2024_09_16'}          # live sim day — contaminated
RTH_START_MIN, RTH_END_MIN = 9 * 60 + 30, 15 * 60 + 30
NEG_GAP_S = 300        # a negative must be >=5min from any same-type event
BAR_S = 5
HORIZONS = (5, 10, 30)
EVENTS = ('fakeout_poke', 'stall', 'ultra_chop', 'leg_descent',
          'defended_poke_shelf')
SEED = 20260804        # fixed: the probe must be reproducible


def _feat_matrix(ts, o, h, l, c, v, idx):
    """Causal features at the bars given by `idx` (positions into the arrays).
    Every window looks strictly backwards from the bar at idx."""
    n = len(c)
    out = {}

    def back(k):
        """value k bars ago, clipped at the day start"""
        j = np.maximum(idx - k, 0)
        return j

    for k, name in ((1, '5s'), (6, '30s'), (12, '60s'), (60, '300s')):
        out[f'ret_{name}'] = c[idx] - c[back(k)]
    for k, name in ((6, '30s'), (12, '60s'), (60, '300s')):
        hi = np.array([h[max(i - k, 0):i + 1].max() for i in idx])
        lo = np.array([l[max(i - k, 0):i + 1].min() for i in idx])
        out[f'range_{name}'] = hi - lo
        out[f'dist_hi_{name}'] = hi - c[idx]
        out[f'dist_lo_{name}'] = c[idx] - lo
        rng = np.where(hi - lo > 0, hi - lo, np.nan)
        out[f'pos_in_{name}'] = (c[idx] - lo) / rng
    # realized vol + flip rate over the last 12 bars (60s)
    d1 = np.diff(c, prepend=c[0])
    out['vol_60s'] = np.array([d1[max(i - 12, 0):i + 1].std() for i in idx])
    out['flips_60s'] = np.array([
        int(np.sum(np.diff(np.sign(d1[max(i - 12, 0):i + 1])) != 0))
        for i in idx])
    out['body_ratio'] = np.array([
        (np.abs(c[max(i - 12, 0):i + 1] - o[max(i - 12, 0):i + 1]).sum()
         / max(1e-9, (h[max(i - 12, 0):i + 1] - l[max(i - 12, 0):i + 1]).sum()))
        for i in idx])
    vsum = np.array([v[max(i - 12, 0):i + 1].sum() for i in idx])
    vbase = np.array([v[max(i - 360, 0):i + 1].mean() * 12 for i in idx])
    out['vol_ratio'] = vsum / np.where(vbase > 0, vbase, np.nan)
    et = (pd.to_datetime(ts[idx], unit='s', utc=True)
          .tz_convert('America/New_York'))
    mins = et.hour * 60 + et.minute - RTH_START_MIN
    out['clock_sin'] = np.sin(2 * np.pi * mins / 390)
    out['clock_cos'] = np.cos(2 * np.pi * mins / 390)
    return pd.DataFrame(out)


def build(event, horizon, rng):
    ev = pd.read_parquet(os.path.join(EVENTS_DIR, f'{event}.parquet'))
    ev = ev[~ev['day'].isin(EXCLUDE_DAYS)]
    rows = []
    for day, grp in tqdm(ev.groupby('day'), desc=f'{event} H={horizon}s',
                         leave=False):
        path = os.path.join(BARS_DIR, f'{day}.parquet')
        if not os.path.exists(path):
            continue
        d = pd.read_parquet(path)
        ts = d['timestamp'].to_numpy()
        o, h = d['open'].to_numpy(), d['high'].to_numpy()
        l, c = d['low'].to_numpy(), d['close'].to_numpy()
        v = d['volume'].to_numpy() if 'volume' in d else np.ones(len(d))
        et = (pd.to_datetime(ts, unit='s', utc=True)
              .tz_convert('America/New_York'))
        mod = et.hour * 60 + et.minute
        rth = np.flatnonzero((mod >= RTH_START_MIN) & (mod < RTH_END_MIN))
        if len(rth) < 400:
            continue
        ev_ts = np.sort(grp['ts'].to_numpy())
        # POSITIVES: the bar at or before (event_ts - horizon)
        pos_idx = np.searchsorted(ts, ev_ts - horizon, side='right') - 1
        pos_idx = pos_idx[(pos_idx > 360) & (pos_idx < len(ts) - 1)]
        pos_idx = np.array([i for i in pos_idx
                            if RTH_START_MIN <= mod[i] < RTH_END_MIN])
        if not len(pos_idx):
            continue
        # NEGATIVES: RTH bars >= NEG_GAP_S from every same-type event
        cand = rth[(rth > 360) & (rth < len(ts) - 1)]
        if len(ev_ts):
            near = np.searchsorted(ev_ts, ts[cand])
            lo_gap = np.where(near > 0, ts[cand] - ev_ts[np.maximum(near - 1, 0)],
                              10 ** 9)
            hi_gap = np.where(near < len(ev_ts),
                              ev_ts[np.minimum(near, len(ev_ts) - 1)] - ts[cand],
                              10 ** 9)
            cand = cand[np.minimum(lo_gap, hi_gap) >= NEG_GAP_S]
        if not len(cand):
            continue
        take = min(len(pos_idx), len(cand))
        neg_idx = rng.choice(cand, size=take, replace=False)
        idx = np.concatenate([pos_idx[:take], neg_idx])
        y = np.concatenate([np.ones(take, int), np.zeros(take, int)])
        f = _feat_matrix(ts, o, h, l, c, v, idx)
        f['y'] = y
        f['day'] = day
        f['ts'] = ts[idx]
        rows.append(f)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    p = os.path.join(OUT_DIR, f'onset_{event}_{horizon}s.parquet')
    out.to_parquet(p, index=False)
    return p, len(out), float(out['y'].mean())


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for event in EVENTS:
        for hz in HORIZONS:
            r = build(event, hz, rng)
            if r:
                print(f'{event} H={hz}s -> {r[1]} rows, base {r[2]:.3f} '
                      f'({os.path.basename(r[0])})')
            else:
                print(f'{event} H={hz}s -> NO DATA')
