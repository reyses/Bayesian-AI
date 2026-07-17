"""phit_feed.py -- causal knowability feed for the anti-freeze reward.

Sources the calibrated per-fire probabilities from
``research/nt8_catalog/reports/econ_drift_rows.parquet`` (the pooled detector
fires with calibrated decile labels). Decile 9 == "act-as-is" high-P fires;
decile 0 == "act-inverted" fires (the detector is confidently WRONG there, so
acting inverted is high-confidence). Every other decile is treated as noise
(no live signal).

``live_signal(ts) -> (c_t, dir)``
    Among decile-{0,9} fires with ``fire_ts <= ts`` and within the trailing
    ``WINDOW_S`` seconds, take the MOST RECENT (tie-broken by confidence) and
    return:
        decile 9 : c_t = P,      dir = +1 if is_long else -1
        decile 0 : c_t = 1 - P,  dir = -1 if is_long else +1   (inverted)
    No qualifying fire -> ``(0.0, 0)``.

STRICTLY CAUSAL: only fires whose ``fire_ts <= ts`` are ever considered, so the
feed can be queried live bar-by-bar with no lookahead.

Empirically (train split): decile-9 P>=0.795 and decile-0 (1-P)>=0.637, so any
extreme fire clears ``theta_c = 0.5`` -- the gate effectively asks "is there a
recent calibrated extreme fire", and c_t modulates the magnitude.

Side effect: ``self.last_fire_ts`` is set to the driving fire's ts (or ``None``)
on every call, so the caller can build a stable per-swing id (regret is capped
once per swing = once per distinct driving fire). The feed is meant to be
queried once per bar in monotonically increasing ts order.
"""
import os
import numpy as np
import pandas as pd

# Trailing causal lookback for a live signal (seconds). One extreme fire lands
# roughly every ~90s (train split), so 120s keeps ~1 fire live at a time.
WINDOW_S = 120

# Decile ends treated as actionable: 9 = act-as-is, 0 = act-inverted.
_DECILE_ACT_AS_IS = 9
_DECILE_ACT_INVERTED = 0

_DEFAULT_PARQUET = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')),
    'research', 'nt8_catalog', 'reports', 'econ_drift_rows.parquet')


class PhitFeed:
    """Causal decile-9/0 knowability feed for a set of trading days."""

    def __init__(self, days, parquet_path=None):
        parquet_path = parquet_path or _DEFAULT_PARQUET
        df = pd.read_parquet(
            parquet_path, columns=['ts', 'day', 'is_long', 'P', 'decile'])
        day_set = set(days)
        df = df[df['day'].isin(day_set)
                & df['decile'].isin([_DECILE_ACT_AS_IS, _DECILE_ACT_INVERTED])]

        P = df['P'].to_numpy(dtype=np.float64)
        dec = df['decile'].to_numpy()
        is_long = df['is_long'].to_numpy()

        # decile 9 -> c_t = P (act as-is); decile 0 -> c_t = 1 - P (act inverted)
        c_t = np.where(dec == _DECILE_ACT_AS_IS, P, 1.0 - P)
        base_dir = np.where(is_long, 1, -1)
        # decile 9 keeps the detector side; decile 0 inverts it.
        direction = np.where(dec == _DECILE_ACT_AS_IS, base_dir, -base_dir)

        ts = df['ts'].to_numpy(dtype=np.int64)
        order = np.argsort(ts, kind='stable')
        self.fire_ts = ts[order]
        self.c_t = c_t[order].astype(np.float64)
        self.dir = direction[order].astype(np.int64)

        self.last_fire_ts = None
        self.n_fires = int(self.fire_ts.shape[0])

    def live_signal(self, ts):
        """Return (c_t, dir) for the most recent causal extreme fire, else (0.0, 0)."""
        ts = int(ts)
        lo = np.searchsorted(self.fire_ts, ts - WINDOW_S, side='left')
        hi = np.searchsorted(self.fire_ts, ts, side='right')  # fire_ts <= ts
        if hi <= lo:
            self.last_fire_ts = None
            return 0.0, 0

        win_ts = self.fire_ts[lo:hi]
        max_ts = win_ts[-1]  # sorted ascending -> most recent is last
        # Sub-slice of the window sharing the most-recent ts; tie-break on c_t.
        j0 = lo + int(np.searchsorted(win_ts, max_ts, side='left'))
        k = j0 + int(np.argmax(self.c_t[j0:hi]))
        self.last_fire_ts = int(self.fire_ts[k])
        return float(self.c_t[k]), int(self.dir[k])


def _self_test():
    """Tiny offline check against a hand-built feed (no parquet required)."""
    feed = PhitFeed.__new__(PhitFeed)
    #            t=100(d9,P.8,long) 150(d0,P.2,long) 150(d9,P.9,short)
    feed.fire_ts = np.array([100, 150, 150, 400], dtype=np.int64)
    feed.c_t = np.array([0.80, 0.80, 0.90, 0.70], dtype=np.float64)  # d0: 1-.2=.8
    feed.dir = np.array([1, -1, -1, 1], dtype=np.int64)
    feed.last_fire_ts = None
    feed.n_fires = 4

    # Just after t=100: the d9 long is live.
    assert feed.live_signal(110) == (0.80, 1), feed.live_signal(110)
    # At t=160: two fires at t=150 (c_t .80 vs .90) -> tie-break to .90 short.
    assert feed.live_signal(160) == (0.90, -1), feed.live_signal(160)
    # At t=90: nothing has fired yet -> no signal.
    assert feed.live_signal(90) == (0.0, 0), feed.live_signal(90)
    # At t=300: the t=150 fires are >120s stale -> no signal.
    assert feed.live_signal(300) == (0.0, 0), feed.live_signal(300)
    # last_fire_ts side effect tracks the driving fire.
    feed.live_signal(410)
    assert feed.last_fire_ts == 400, feed.last_fire_ts
    feed.live_signal(300)
    assert feed.last_fire_ts is None, feed.last_fire_ts
    print("[PASS] phit_feed offline self-test")


if __name__ == '__main__':
    _self_test()
