"""Extract 'freeze' events (25% giveback inside an active zigzag leg) from ATLAS bars.

Writes research/reversal_gauge/events.parquet (one row per event, max one event
per leg, RTH 09:30-15:30 ET only).

Causality contract: every feature uses ONLY bars <= event ts. A leg is 'active'
from the bar its origin pivot was CONFIRMED (price moved REVERSAL_PT off the
origin close); events are never labeled back in time. day_class is likewise
causal: an event on a flushV day is tagged 'flushV' only if the flush window
had closed AND the recovery had printed at or before the event bar.

All price conditions (zigzag, peak, giveback, labels) are on bar CLOSES for
internal consistency with the close-based zigzag; bar highs/lows are used only
where intrabar reach is the point (worn-level touches, flush low / recovery).

Run from repo root:
    python research/reversal_gauge/builders/extract_freeze_events.py
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ATLAS_DIR = os.path.join(REPO_ROOT, "DATA", "ATLAS")
OUT_PATH = os.path.join(REPO_ROOT, "research", "reversal_gauge", "events.parquet")

ET_TZ = "America/New_York"
TARGET_TF_S = 5                    # preferred bar size (seconds)

REVERSAL_PT = 8.0                  # zigzag reversal threshold (points, MNQ)
MIN_MFE_PT = 8.0                   # event gate: leg MFE must be at least this
EVENT_GIVEBACK_FRAC = 0.25         # event fires at first giveback >= this
RESUME_EXCEED_PT = 0.5             # label_resume: exceed peak_px by this (dir-adjusted)
RETENTION_FAIL_FRAC = 0.50         # label_resume=0 when retention drops below this
LABEL_WINDOW_S = 45 * 60           # label scan horizon after the event bar

REPOKE_TOL_PT = 2.0                # peak within this of a prior same-dir leg extreme
REPOKE_WINDOW_S = 90 * 60          # ...whose extreme printed within this of event ts
WORN_ROUND_PT = 5.0                # worn level = peak rounded to nearest 5pt
WORN_WINDOW_S = 120 * 60           # touch-count lookback from event ts

FLUSH_MIN_PT = 60.0                # flushV: 09:30 open minus min low 09:30-09:50
FLUSH_RECOVERY_FRAC = 0.60         # ...recovered at least this fraction by 10:20

# ET minutes-of-day (fractional; seconds included)
RTH_START_MIN = 9 * 60 + 30.0      # 09:30 inclusive
RTH_END_MIN = 15 * 60 + 30.0       # 15:30 exclusive
FLUSH_WIN_END_MIN = 9 * 60 + 50.0  # flush window is [09:30, 09:50)
RECOVERY_DEADLINE_MIN = 10 * 60 + 20.0  # recovery must print at/before 10:20:00
CLOCK_BUCKET_MIN = 30              # half-hour clock buckets

EXCLUDED_DAYS = {"2024_09_16"}     # live/contaminated day

COLUMNS = [
    "day", "ts", "dir", "anchor_px", "peak_px", "mfe_pt", "giveback_frac",
    "t2peak_s", "pace_pts_s", "spike_score", "repoke", "worn_touches",
    "day_class", "clock_bucket", "label_resume", "label_resolved",
]


def _tf_seconds(name: str) -> int | None:
    """'5s' -> 5, '1m' -> 60, '1h' -> 3600, '1D' -> 86400; None if not a tf dir."""
    m = re.fullmatch(r"(\d+)([smhD])", name)
    if m is None:
        return None
    mult = {"s": 1, "m": 60, "h": 3600, "D": 86400}[m.group(2)]
    return int(m.group(1)) * mult


def resolve_data_dir() -> tuple[str, int]:
    """Prefer DATA/ATLAS/5s; else the finest tf dir <= 5s; else fall back to 1m."""
    preferred = os.path.join(ATLAS_DIR, f"{TARGET_TF_S}s")
    if os.path.isdir(preferred):
        return preferred, TARGET_TF_S
    candidates = []
    for name in sorted(os.listdir(ATLAS_DIR)):
        sec = _tf_seconds(name)
        if sec is not None and sec <= TARGET_TF_S and os.path.isdir(os.path.join(ATLAS_DIR, name)):
            candidates.append((sec, name))
    if candidates:
        sec, name = max(candidates)  # coarsest among those <= target = finest usable
        return os.path.join(ATLAS_DIR, name), sec
    one_min = os.path.join(ATLAS_DIR, "1m")
    if os.path.isdir(one_min):
        return one_min, 60
    raise FileNotFoundError(f"No usable timeframe dir under {ATLAS_DIR}")


def _flush_confirm_ts(ts: np.ndarray, opn: np.ndarray, high: np.ndarray,
                      low: np.ndarray, mod: np.ndarray) -> int | None:
    """Return the epoch ts at which the flushV classification becomes knowable,
    or None if the day is not flushV.

    flushV: 09:30 open minus min low over [09:30, 09:50) >= FLUSH_MIN_PT, AND a
    bar HIGH after the min-low bar reaches low + FLUSH_RECOVERY_FRAC * flush at
    or before 10:20:00 ET. Knowable only once BOTH the flush window has closed
    (first bar at/after 09:50) and the recovery bar has printed.
    """
    # AUDIT FIX (2026-08-04): day files span 18:00-prev-evening onward, so an
    # unbounded `mod >= X` first-match lands on PRIOR-EVENING bars (mod >=
    # 1080). open_px was the previous evening's open (wrong flushV set on
    # 167/600 days) and the window-closed guard was dead (992 events tagged
    # flushV with lookahead). Both selectors now bound mod to the morning.
    win = np.flatnonzero((mod >= RTH_START_MIN) & (mod < FLUSH_WIN_END_MIN))
    if win.size == 0:
        return None
    open_px = opn[win[0]]
    rel_min = int(np.argmin(low[win]))   # argmin -> first occurrence of the min
    flush_low = low[win[rel_min]]
    t_low_idx = win[rel_min]
    flush_pt = open_px - flush_low
    if flush_pt < FLUSH_MIN_PT:
        return None
    target = flush_low + FLUSH_RECOVERY_FRAC * flush_pt
    idx = np.arange(ts.size)
    rec = np.flatnonzero((idx > t_low_idx) & (mod <= RECOVERY_DEADLINE_MIN) & (high >= target))
    if rec.size == 0:
        return None
    closed = np.flatnonzero((mod >= FLUSH_WIN_END_MIN)
                            & (mod < RECOVERY_DEADLINE_MIN + 60))
    if closed.size == 0:
        return None
    return int(max(ts[rec[0]], ts[closed[0]]))


def _build_event(day, i, d, anchor_px, anchor_i, peak_px, peak_i,
                 ts, close, high, low, hour, minute,
                 prior_extremes, flush_ts):
    t = int(ts[i])
    mfe = d * (peak_px - anchor_px)
    giveback = d * (peak_px - close[i]) / mfe
    t2peak = int(ts[peak_i] - ts[anchor_i])
    # peak bar strictly follows the anchor bar, so t2peak > 0; guard for data gaps
    pace = mfe / max(t2peak, 1)

    # spike: largest single-bar favorable close-to-close move in the leg so far
    deltas = d * np.diff(close[anchor_i:i + 1])
    spike = float(max(deltas.max(), 0.0)) / mfe

    repoke = 0
    for ld, lpx, lts in reversed(prior_extremes):
        if t - lts > REPOKE_WINDOW_S:
            break  # list is chronological; everything earlier is out of window
        if ld == d and abs(peak_px - lpx) <= REPOKE_TOL_PT:
            repoke = 1
            break

    # worn touches: bars BEFORE the current leg (idx <= anchor pivot bar) whose
    # intrabar range touched the 5pt-rounded peak level, within the lookback
    level = round(peak_px / WORN_ROUND_PT) * WORN_ROUND_PT
    lo_i = int(np.searchsorted(ts, t - WORN_WINDOW_S, side="left"))
    hi_i = anchor_i + 1
    if hi_i > lo_i:
        seg_lo = low[lo_i:hi_i]
        seg_hi = high[lo_i:hi_i]
        worn = int(np.sum((seg_lo <= level) & (level <= seg_hi)))
    else:
        worn = 0

    day_class = "flushV" if (flush_ts is not None and t >= flush_ts) else "other"
    clock_bucket = f"{int(hour[i]):02d}:{(int(minute[i]) // CLOCK_BUCKET_MIN) * CLOCK_BUCKET_MIN:02d}"

    # label: forward scan, closes only; the two conditions are mutually
    # exclusive on a single close (peak + 0.5 > anchor + 0.5*mfe since mfe >= 8)
    resume, resolved = 0, 0
    end_t = t + LABEL_WINDOW_S
    j = i + 1
    n = ts.size
    while j < n and ts[j] <= end_t:
        cj = close[j]
        if d * (cj - peak_px) >= RESUME_EXCEED_PT:
            resume, resolved = 1, 1
            break
        if d * (cj - anchor_px) / mfe < RETENTION_FAIL_FRAC:
            resume, resolved = 0, 1
            break
        j += 1

    return {
        "day": day, "ts": t, "dir": int(d),
        "anchor_px": float(anchor_px), "peak_px": float(peak_px),
        "mfe_pt": float(mfe), "giveback_frac": float(giveback),
        "t2peak_s": t2peak, "pace_pts_s": float(pace),
        "spike_score": spike, "repoke": repoke, "worn_touches": worn,
        "day_class": day_class, "clock_bucket": clock_bucket,
        "label_resume": resume, "label_resolved": resolved,
    }


def process_day(day: str, df: pd.DataFrame) -> list[dict]:
    if len(df) < 2:
        return []
    ts = df["timestamp"].to_numpy(np.int64)
    opn = df["open"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)

    eti = pd.DatetimeIndex(pd.to_datetime(ts, unit="s", utc=True)).tz_convert(ET_TZ)
    hour = eti.hour.to_numpy()
    minute = eti.minute.to_numpy()
    mod = hour * 60.0 + minute + eti.second.to_numpy() / 60.0

    flush_ts = _flush_confirm_ts(ts, opn, high, low, mod)

    events: list[dict] = []
    prior_extremes: list[tuple[int, float, int]] = []  # (leg_dir, extreme_px, extreme_ts)

    d = 0  # 0 = no confirmed leg yet
    run_min_px, run_min_i = close[0], 0
    run_max_px, run_max_i = close[0], 0
    anchor_px, anchor_i = 0.0, 0
    peak_px, peak_i = 0.0, 0
    event_fired = False

    for i in range(ts.size):
        c = close[i]

        if d == 0:
            if c < run_min_px:
                run_min_px, run_min_i = c, i
            if c > run_max_px:
                run_max_px, run_max_i = c, i
            up_conf = c - run_min_px >= REVERSAL_PT
            dn_conf = run_max_px - c >= REVERSAL_PT
            if up_conf and dn_conf:
                # wide pre-leg range: both confirmable on one bar; the more
                # recent extreme is the origin pivot
                up_conf = run_min_i > run_max_i
                dn_conf = not up_conf
            if up_conf or dn_conf:
                d = 1 if up_conf else -1
                anchor_px, anchor_i = (run_min_px, run_min_i) if up_conf else (run_max_px, run_max_i)
                # origin pivot doubles as the extreme of the (implicit) prior leg
                prior_extremes.append((-d, float(anchor_px), int(ts[anchor_i])))
                peak_px, peak_i = c, i
                event_fired = False
            continue

        # active leg: leg confirmed => MFE already >= REVERSAL_PT from the anchor
        if d * (c - peak_px) > 0:
            peak_px, peak_i = c, i
        mfe = d * (peak_px - anchor_px)

        if (not event_fired) and RTH_START_MIN <= mod[i] < RTH_END_MIN and mfe >= MIN_MFE_PT:
            if d * (peak_px - c) / mfe >= EVENT_GIVEBACK_FRAC:
                event_fired = True
                events.append(_build_event(
                    day, i, d, anchor_px, anchor_i, peak_px, peak_i,
                    ts, close, high, low, hour, minute,
                    prior_extremes, flush_ts))

        # event check precedes the reversal check so a single collapse bar can
        # both fire the event and terminate the leg
        if d * (peak_px - c) >= REVERSAL_PT:
            prior_extremes.append((d, float(peak_px), int(ts[peak_i])))
            anchor_px, anchor_i = peak_px, peak_i
            d = -d
            peak_px, peak_i = c, i
            event_fired = False

    return events


def main() -> None:
    data_dir, tf_s = resolve_data_dir()
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    print(f"data: {data_dir} ({len(files)} files, {tf_s}s bars)")

    rows: list[dict] = []
    for path in tqdm(files, desc="days"):
        day = os.path.splitext(os.path.basename(path))[0]
        if day in EXCLUDED_DAYS:
            continue
        df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
        df = df.drop_duplicates("timestamp").sort_values("timestamp")
        rows.extend(process_day(day, df))

    out = pd.DataFrame(rows, columns=COLUMNS)
    for col in ("ts", "t2peak_s"):
        out[col] = out[col].astype(np.int64)
    for col in ("dir", "repoke", "worn_touches", "label_resume", "label_resolved"):
        out[col] = out[col].astype(np.int64)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    n = len(out)
    resolve_rate = out["label_resolved"].mean() if n else float("nan")
    resolved = out[out["label_resolved"] == 1]
    base_rate = resolved["label_resume"].mean() if len(resolved) else float("nan")
    print(f"wrote {OUT_PATH}")
    print(f"n events: {n} | resolve rate: {resolve_rate:.3f} | "
          f"base rate p(resume|resolved): {base_rate:.3f} (n resolved: {len(resolved)})")


if __name__ == "__main__":
    main()
