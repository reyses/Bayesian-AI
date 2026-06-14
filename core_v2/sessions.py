"""Canonical session-day boundary for CME equity-index futures (MNQ).

THE SINGLE SOURCE OF TRUTH for what a trading "day" is, project-wide. Ingestion
partitioning, the NT8 dumper (mirrored in C#), feature building, and EVERY
analysis day-grouping (day-block bootstrap, $/day, Day WR) must use this — a
mixed UTC-day / session-day state silently desyncs IS vs OOS and the metrics.

WHY: the CME Globex daily maintenance halt is 16:00-17:00 America/Chicago; the
session reopens at 17:00 CT. A UTC calendar day cuts straight through this (the
tail of one session + the halt + the head of the next), so a UTC "day" both
contains a mid-day gap AND mixes two sessions. We instead define the trading day
by the session reopen: a session runs 17:00 CT -> 17:00 CT and is LABELLED by
its CLOSING Chicago calendar date, so the Sunday-evening reopen belongs to
"Monday". The maintenance halt then always sits at a day EDGE, never the middle.

DST-AWARE: anchored to America/Chicago wall-clock, so the boundary is 22:00 UTC
in summer (CDT) and 23:00 UTC in winter (CST) automatically. Uses pandas tz
(bundled) rather than zoneinfo, so it works on Windows without a system tzdata.
"""
import numpy as np
import pandas as pd

CHICAGO_TZ = 'America/Chicago'
SESSION_REOPEN_HOUR_CT = 17   # 17:00 CT = CME equity-index reopen after maintenance


def session_day(ts):
    """Session-day label 'YYYY_MM_DD' for one UTC unix-epoch-seconds timestamp.

    The session 17:00 CT (prior day) -> 16:00 CT is labelled by its close date,
    so a 17:00-CT-or-later bar rolls into the NEXT calendar day's label."""
    t = pd.Timestamp(int(ts), unit='s', tz='UTC').tz_convert(CHICAGO_TZ)
    label = t.normalize().tz_localize(None)            # naive CT midnight of the bar's date
    if t.hour >= SESSION_REOPEN_HOUR_CT:
        label = label + pd.Timedelta(days=1)
    return label.strftime('%Y_%m_%d')


def session_day_array(ts_seconds):
    """Vectorized session_day for an array/Series of UTC epoch seconds.

    Returns a numpy array of 'YYYY_MM_DD' strings. Use this in pandas pipelines;
    it is the array twin of session_day() and gives identical labels."""
    idx = pd.to_datetime(np.asarray(ts_seconds, dtype='int64'), unit='s', utc=True).tz_convert(CHICAGO_TZ)
    base = idx.tz_localize(None).normalize()           # naive CT midnight per bar
    bump = pd.to_timedelta((idx.hour >= SESSION_REOPEN_HOUR_CT).astype('int64'), unit='D')
    return (base + bump).strftime('%Y_%m_%d').to_numpy()
