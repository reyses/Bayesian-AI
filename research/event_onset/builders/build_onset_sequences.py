"""ONSET MAMBA — sequence dataset builder (spec §11 step 1).

Stores PER-DAY 1s feature and label arrays; windows are sliced in the
dataloader. Materialising 300-step windows for every RTH second would cost
~68 GB and buy nothing — the same bars would be copied 300 times.

Features (8 per second, all causal, NO absolute price — MNQ ran 16k->28k
across this corpus and a level is just a date stamp):
    0 ret_ticks        close - prev_close
    1 upper_wick       high - max(open, close)
    2 lower_wick       min(open, close) - low
    3 body             close - open
    4 range            high - low
    5 vol_z            log1p(volume), z-scored on a TRAILING 600s window
    6 clock_sin        seconds since 09:30
    7 clock_cos

Labels (9 per second): 3 event types x 3 horizons.
    y[k*3 + h] = 1 if an event of type k confirms in (t, t+H_h]

Warmup: WINDOW seconds before 09:30 are kept so the first RTH second has a
full window; they are marked non-sampleable in the mask.

Run from repo root:
  python research/event_onset/builders/build_onset_sequences.py
Writes research/event_onset/seq/<day>.npz + seq/manifest.parquet
"""
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
BARS = os.path.join(REPO, 'DATA', 'ATLAS', '1s')
EVENTS = os.path.join(REPO, 'research', 'event_library', 'events')
OUT = os.path.join(REPO, 'research', 'event_onset', 'seq')
EXCLUDE = {'2024_09_16'}
HEADS = ('fakeout_poke', 'leg_descent', 'ultra_chop')   # spec §1, post-ablation
HORIZONS = (5, 10, 30)
WINDOW = 300
TICK = 0.25
RTH0, RTH1 = 9 * 60 + 30, 15 * 60 + 30
VOL_Z_WIN = 600


def load_event_ts():
    out = {}
    for k in HEADS:
        d = pd.read_parquet(os.path.join(EVENTS, f'{k}.parquet'),
                            columns=['day', 'ts'])
        out[k] = {day: np.sort(g['ts'].to_numpy())
                  for day, g in d.groupby('day')}
    return out


def build_day(path, ev_ts):
    day = os.path.basename(path).replace('.parquet', '')
    if day in EXCLUDE:
        return None
    d = pd.read_parquet(path)
    ts = d['timestamp'].to_numpy()
    et = (pd.to_datetime(ts, unit='s', utc=True)
          .tz_convert('America/New_York'))
    mod = et.hour * 60 + et.minute
    keep = np.flatnonzero((mod >= RTH0 - WINDOW // 60 - 1) & (mod < RTH1))
    if len(keep) < WINDOW + 600:
        return None
    d = d.iloc[keep]
    ts = ts[keep]
    mod = mod[keep]
    o = d['open'].to_numpy(np.float32)
    h = d['high'].to_numpy(np.float32)
    lo = d['low'].to_numpy(np.float32)
    c = d['close'].to_numpy(np.float32)
    v = (d['volume'].to_numpy(np.float32) if 'volume' in d
         else np.ones(len(d), np.float32))

    f = np.zeros((len(c), 8), np.float32)
    f[1:, 0] = np.diff(c) / TICK
    f[:, 1] = (h - np.maximum(o, c)) / TICK
    f[:, 2] = (np.minimum(o, c) - lo) / TICK
    f[:, 3] = (c - o) / TICK
    f[:, 4] = (h - lo) / TICK
    lv = np.log1p(v)
    s = pd.Series(lv)
    mu = s.rolling(VOL_Z_WIN, min_periods=30).mean().to_numpy()
    sd = s.rolling(VOL_Z_WIN, min_periods=30).std().to_numpy()
    f[:, 5] = np.nan_to_num((lv - mu) / np.where(sd > 1e-6, sd, 1.0))
    secs = (et.hour[keep] * 3600 + et.minute[keep] * 60
            + et.second[keep]).to_numpy() - RTH0 * 60
    f[:, 6] = np.sin(2 * np.pi * secs / (6.5 * 3600))
    f[:, 7] = np.cos(2 * np.pi * secs / (6.5 * 3600))

    y = np.zeros((len(c), len(HEADS) * len(HORIZONS)), np.int8)
    for ki, k in enumerate(HEADS):
        e = ev_ts[k].get(day)
        if e is None or not len(e):
            continue
        for hi, H in enumerate(HORIZONS):
            # event confirms in (t, t+H]  <=>  searchsorted window non-empty
            left = np.searchsorted(e, ts, side='right')
            right = np.searchsorted(e, ts + H, side='right')
            y[:, ki * 3 + hi] = (right > left).astype(np.int8)

    # sampleable = full window behind it AND inside RTH
    mask = np.zeros(len(c), bool)
    mask[WINDOW:] = True
    mask &= (mod >= RTH0) & (mod < RTH1)
    if mask.sum() < 100:
        return None
    np.savez_compressed(os.path.join(OUT, f'{day}.npz'),
                        f=f.astype(np.float16), y=y,
                        mask=mask, ts=ts.astype(np.int64))
    return dict(day=day, rows=len(c), sampleable=int(mask.sum()),
                pos_rate=float(y[mask].mean()))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    ev_ts = load_event_ts()
    rows = []
    for p in tqdm(sorted(glob.glob(os.path.join(BARS, '*.parquet'))),
                  desc='days'):
        r = build_day(p, ev_ts)
        if r:
            rows.append(r)
    man = pd.DataFrame(rows)
    man.to_parquet(os.path.join(OUT, 'manifest.parquet'), index=False)
    sz = sum(os.path.getsize(f) for f in glob.glob(os.path.join(OUT, '*.npz')))
    print(f'{len(man)} days, {man["sampleable"].sum():,} sampleable seconds, '
          f'mean label rate {man["pos_rate"].mean():.4f}, {sz/1e6:.0f} MB')
