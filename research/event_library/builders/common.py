"""Shared infrastructure for the EVENT LIBRARY (v0).

Owner-ratified architecture (2026-08): "identify specific events, read the
fuzzy events" — each owner-named tape state gets (a) a CAUSAL detector and
(b) its own cohort outcome table. This module holds the data plumbing and
statistics helpers shared by all detectors.

HARD RULE — live-day guard: day 2024_09_16 is the live sim day and is
hindsight-contaminated (a prior tool leaked its future mid-decision). It is
excluded from EVERY table and fit. It is used ONLY by
tools/anchor_check.py to sanity-fire detectors at their calibration anchors.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ATLAS_DIR = os.path.join(REPO_ROOT, "DATA", "ATLAS")
EVENTS_DIR = os.path.join(REPO_ROOT, "research", "event_library", "events")
REPORTS_DIR = os.path.join(REPO_ROOT, "research", "event_library", "reports")

ET_TZ = "America/New_York"

# Live sim day — hindsight-contaminated; excluded from all tables/fits.
EXCLUDED_DAYS = {"2024_09_16"}

# ET minutes-of-day (fractional, seconds included).
# Day files START 18:00 the PREVIOUS evening — every time-of-day mask MUST be
# bounded on both sides or the first match lands on prior-evening bars
# (the audit-fix class of bug in reversal_gauge _flush_confirm_ts).
RTH_START_MIN = 9 * 60 + 30.0    # 09:30 inclusive
RTH_EVENT_END_MIN = 15 * 60 + 30.0  # 15:30 — last confirm time for intraday events
SESSION_END_MIN = 16 * 60.0      # 16:00 — outcome-measurement bound (RTH close)

WILSON_Z = 1.96                  # 95% two-sided normal quantile


def day_list() -> list[str]:
    """Canonical corpus: the 5s per-day files, minus the excluded live day."""
    files = sorted(os.listdir(os.path.join(ATLAS_DIR, "5s")))
    days = [f[:-8] for f in files if f.endswith(".parquet")]
    return [d for d in days if d not in EXCLUDED_DAYS]


def load_day(tf: str, day: str) -> dict | None:
    """Load one per-day ATLAS file into numpy arrays + ET minute-of-day."""
    path = os.path.join(ATLAS_DIR, tf, f"{day}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    ts = df["timestamp"].to_numpy(np.int64)
    eti = pd.DatetimeIndex(pd.to_datetime(ts, unit="s", utc=True)).tz_convert(ET_TZ)
    mod = (eti.hour.to_numpy() * 60.0 + eti.minute.to_numpy()
           + eti.second.to_numpy() / 60.0)
    return {
        "day": day,
        "ts": ts,
        "open": df["open"].to_numpy(np.float64),
        "high": df["high"].to_numpy(np.float64),
        "low": df["low"].to_numpy(np.float64),
        "close": df["close"].to_numpy(np.float64),
        "mod": mod,
        "hms": np.array(eti.strftime("%H:%M:%S")),
        "n": len(df),
    }


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Wilson 95% CI for a proportion: (p_hat, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = WILSON_Z
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, ctr - hw), min(1.0, ctr + hw)


def wilson_str(k: int, n: int, pct: bool = True) -> str:
    p, lo, hi = wilson(k, n)
    if n == 0:
        return "n=0"
    if pct:
        s = f"{p:.1%} [{lo:.1%}, {hi:.1%}] (k={k}, n={n})"
    else:
        s = f"{p:.3f} [{lo:.3f}, {hi:.3f}] (k={k}, n={n})"
    if n < 30:
        s += " UNDERPOWERED"
    return s


def quart_str(x, unit: str = "") -> str:
    """median [q25, q75] (n=...) string for a magnitude distribution."""
    x = pd.Series(x).dropna()
    n = len(x)
    if n == 0:
        return "n=0"
    s = (f"median {x.median():+.1f}{unit} "
         f"[q25 {x.quantile(0.25):+.1f}, q75 {x.quantile(0.75):+.1f}] (n={n})")
    if n < 30:
        s += " UNDERPOWERED"
    return s
