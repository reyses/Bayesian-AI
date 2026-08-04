"""EVENT-ONSET probe v2 — HARD (matched) negatives.

Why this exists: v1 (build_onset_dataset.py) drew negatives from stretches
>=5min away from ANY same-type event and scored AUC up to 0.9965. That is
too good, and the reason is design, not signal: for a frequent event, "no
event within 5 minutes" selects an unusually QUIET regime, so the classifier
can win by answering "is the tape active?" — a question with no trading
value.

v2 asks the question that actually matters:
    at T-H seconds vs T-H-LAG seconds on the SAME DAY, in the SAME regime,
    can you tell that the event is about to confirm?
Negative = the same event rewound a further LAG seconds (default 300).
Rejected if another same-type event confirms inside the negative's own
horizon window (it would be a positive in disguise).

Same causal feature builder as v1 — imported, not copied.

Run from repo root:
  python research/event_onset/builders/build_onset_matched.py
Writes research/event_onset/matched_<event>_<H>s.parquet
"""
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'research', 'event_onset', 'builders'))
from build_onset_dataset import (          # noqa: E402
    _feat_matrix, EVENTS_DIR, BARS_DIR, OUT_DIR, EXCLUDE_DAYS,
    RTH_START_MIN, RTH_END_MIN, HORIZONS, EVENTS)

LAG_S = 300            # how far back the matched negative sits


def build(event, horizon):
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
        ev_ts = np.sort(grp['ts'].to_numpy())
        pos_t, neg_t = [], []
        for e in ev_ts:
            pt, nt = e - horizon, e - horizon - LAG_S
            # the negative must not itself be an event about to confirm
            if np.any((ev_ts > nt) & (ev_ts <= nt + horizon)):
                continue
            pi = np.searchsorted(ts, pt, side='right') - 1
            ni = np.searchsorted(ts, nt, side='right') - 1
            if pi <= 360 or ni <= 360 or pi >= len(ts) - 1:
                continue
            if not (RTH_START_MIN <= mod[pi] < RTH_END_MIN
                    and RTH_START_MIN <= mod[ni] < RTH_END_MIN):
                continue
            pos_t.append(pi)
            neg_t.append(ni)
        if not pos_t:
            continue
        idx = np.array(pos_t + neg_t)
        y = np.concatenate([np.ones(len(pos_t), int), np.zeros(len(neg_t), int)])
        f = _feat_matrix(ts, o, h, l, c, v, idx)
        f['y'] = y
        f['day'] = day
        f['ts'] = ts[idx]
        rows.append(f)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    p = os.path.join(OUT_DIR, f'matched_{event}_{horizon}s.parquet')
    out.to_parquet(p, index=False)
    return p, len(out), float(out['y'].mean())


if __name__ == '__main__':
    for event in EVENTS:
        for hz in HORIZONS:
            r = build(event, hz)
            print(f'{event} H={hz}s -> ' +
                  (f'{r[1]} rows, base {r[2]:.3f}' if r else 'NO DATA'))
