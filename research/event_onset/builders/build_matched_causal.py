"""Matched-negative probe datasets with GENUINELY causal features.

The `--causal` flag in fit_matched.py was WRONG: it shifted feature rows by
one within a day, but matched rows are (event-H, event-H-300) pairs, not a
contiguous time series — so the shift pulled features from an unrelated
event. It collapsed every AUC toward 0.55-0.62 and was measuring nothing.

Correct fix: recompute the SAME rows with the feature index moved one 5s bar
back, so a row stamped t uses bars <= t-5 and cannot contain the bar-OPEN
+5s of future.

  python research/event_onset/builders/build_matched_causal.py
"""
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_onset_dataset import (_feat_matrix, EVENTS_DIR, BARS_DIR, OUT_DIR,
                                 EXCLUDE_DAYS, RTH_START_MIN, RTH_END_MIN)
from build_onset_matched import LAG_S

EVENTS = ('fakeout_poke', 'leg_descent', 'ultra_chop', 'stall')
HORIZON = 10
SHIFT = 1          # bars back (5s) — kills the bar-OPEN lookahead


def build(event, horizon=HORIZON):
    ev = pd.read_parquet(os.path.join(EVENTS_DIR, f'{event}.parquet'))
    ev = ev[~ev['day'].isin(EXCLUDE_DAYS)]
    rows = []
    for day, grp in tqdm(ev.groupby('day'), desc=f'{event} causal', leave=False):
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
        ev_ts = np.sort(grp['ts'].to_numpy())
        pos_i, neg_i = [], []
        for e in ev_ts:
            pt, nt = e - horizon, e - horizon - LAG_S
            if np.any((ev_ts > nt) & (ev_ts <= nt + horizon)):
                continue
            pi = np.searchsorted(ts, pt, side='right') - 1 - SHIFT
            ni = np.searchsorted(ts, nt, side='right') - 1 - SHIFT
            if pi <= 360 or ni <= 360 or pi >= len(ts) - 1:
                continue
            if not (RTH_START_MIN <= mod[pi] < RTH_END_MIN
                    and RTH_START_MIN <= mod[ni] < RTH_END_MIN):
                continue
            pos_i.append(pi)
            neg_i.append(ni)
        if not pos_i:
            continue
        idx = np.array(pos_i + neg_i)
        f = _feat_matrix(ts, o, h, l, c, v, idx)
        f['y'] = np.concatenate([np.ones(len(pos_i), int),
                                 np.zeros(len(neg_i), int)])
        f['day'] = day
        f['ts'] = ts[idx]
        rows.append(f)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    p = os.path.join(OUT_DIR, f'causal_{event}_{horizon}s.parquet')
    out.to_parquet(p, index=False)
    return p, len(out)


if __name__ == '__main__':
    for e in EVENTS:
        r = build(e)
        print(f'{e} -> {r[1]} rows' if r else f'{e} -> NO DATA')
