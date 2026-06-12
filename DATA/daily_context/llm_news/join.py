"""Lookahead-safe joins from per-release LLM scores -> per-day intensity columns.

Two columns are produced (in two phases per the project plan):

1. news_intensity_today  -- max intensity of releases with release_ts_et
                            STRICTLY BEFORE 09:30 ET on the target date.
                            (08:30 ET CPI/NFP prints land here; 14:00 ET
                            FOMC releases do not -- they fire DURING the
                            trading day, so for "today" they are unsafe.)

2. news_intensity_prior  -- max intensity of releases with release_ts_et
                            AT OR AFTER 09:30 ET on the PRIOR trading day.
                            (Yesterday's 14:00 ET FOMC lands here.)

Both columns default to 0.0 (neutral) on dates with no qualifying release.

`release_ts_et` is stored in the news_scores parquet as a tz-aware
Timestamp in America/New_York. The 09:30 ET cutoff is applied in that
zone -- callers do not need to handle DST.
"""
from __future__ import annotations
from datetime import datetime, time

import numpy as np
import pandas as pd

CUTOFF = time(9, 30)


def _ts_to_et_date_and_time(ts: pd.Timestamp) -> tuple[datetime, time]:
    """Return (date, time-of-day) in America/New_York for a tz-aware Timestamp."""
    if ts.tzinfo is None:
        raise ValueError(f'release_ts_et must be tz-aware, got naive: {ts}')
    et = ts.tz_convert('America/New_York')
    return et.date(), et.time()


def compute_news_intensity_today(news_scores: pd.DataFrame,
                                  dates: pd.Series) -> pd.Series:
    """For each date in `dates`, return max intensity of releases on that
    same calendar date whose release_ts_et < 09:30 ET.

    Args:
        news_scores: DataFrame with columns ['release_ts_et', 'intensity'].
                     release_ts_et must be tz-aware (ET).
        dates: pd.Series of datetime.date (calendar dates).

    Returns:
        pd.Series aligned to `dates`, dtype float, default 0.0.
    """
    if len(news_scores) == 0:
        return pd.Series(np.zeros(len(dates), dtype=np.float32), index=dates.index)

    by_date: dict = {}
    for _, row in news_scores.iterrows():
        et_date, et_time = _ts_to_et_date_and_time(row['release_ts_et'])
        if et_time >= CUTOFF:
            continue
        cur = by_date.get(et_date, 0.0)
        if float(row['intensity']) > cur:
            by_date[et_date] = float(row['intensity'])

    out = np.zeros(len(dates), dtype=np.float32)
    for i, d in enumerate(dates):
        out[i] = by_date.get(d, 0.0)
    return pd.Series(out, index=dates.index)


def compute_news_intensity_prior(news_scores: pd.DataFrame,
                                  dates: pd.Series,
                                  trading_dates_sorted: list) -> pd.Series:
    """For each date in `dates`, return max intensity of releases on the
    PRIOR trading date whose release_ts_et >= 09:30 ET (i.e. afternoon-of-yesterday).

    Args:
        news_scores: DataFrame with columns ['release_ts_et', 'intensity'].
        dates: pd.Series of datetime.date for target days.
        trading_dates_sorted: list of all trading dates (datetime.date),
                              ascending, used to identify "prior trading day"
                              (skips weekends/holidays).

    Returns:
        pd.Series aligned to `dates`, dtype float, default 0.0.
    """
    if len(news_scores) == 0:
        return pd.Series(np.zeros(len(dates), dtype=np.float32), index=dates.index)

    by_date: dict = {}
    for _, row in news_scores.iterrows():
        et_date, et_time = _ts_to_et_date_and_time(row['release_ts_et'])
        if et_time < CUTOFF:
            continue
        cur = by_date.get(et_date, 0.0)
        if float(row['intensity']) > cur:
            by_date[et_date] = float(row['intensity'])

    date_to_idx = {d: i for i, d in enumerate(trading_dates_sorted)}
    out = np.zeros(len(dates), dtype=np.float32)
    for i, d in enumerate(dates):
        idx = date_to_idx.get(d)
        if idx is None or idx == 0:
            continue
        prior = trading_dates_sorted[idx - 1]
        out[i] = by_date.get(prior, 0.0)
    return pd.Series(out, index=dates.index)
