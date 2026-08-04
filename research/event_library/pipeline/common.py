"""Shared infrastructure for the EVENT LIBRARY: day loading, clock handling,
the canonical streaming zigzag, and the statistics helpers every table uses.

Design rules enforced here (they are the failure modes this package is built
around):

1. DAY FILES START THE PREVIOUS EVENING. `DATA/ATLAS/*/YYYY_MM_DD.parquet`
   spans 18:00 ET of the PRIOR calendar day through ~17:00 ET of the named
   day. Any unbounded `minute_of_day >= X` mask therefore matches PRIOR-EVENING
   bars first (mod >= 1080). This is the audit bug that corrupted flushV
   labelling on 167/600 days in `research/reversal_gauge/builders/
   extract_freeze_events.py` (see its AUDIT FIX comment). Every window in this
   package is BOUNDED on both sides via `rth_mask()` / explicit lo<=mod<hi.

2. STRICT CAUSALITY. An event is stamped at the bar where every defining
   condition is OBSERVABLE from bars <= that bar. Outcomes are measured
   strictly AFTER the stamp. Detectors live in `detectors.py`, outcome
   measurement lives in `outcomes.py`; the split is deliberate so the
   causality boundary is a module boundary.

3. LIVE-DAY GUARD. 2024_09_16 is the pocket-dojo live-sim day and is
   hindsight-contaminated. It is excluded from every parquet and every table.
   It is used ONLY by `tools/anchor_fire.py` for detector calibration.
"""
from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ATLAS_DIR = os.path.join(REPO_ROOT, "DATA", "ATLAS")
EVENTS_DIR = os.path.join(REPO_ROOT, "research", "event_library", "events")
REPORTS_DIR = os.path.join(REPO_ROOT, "research", "event_library", "reports")

ET_TZ = "America/New_York"

# MNQ contract spec (CLAUDE.md)
TICK_PT = 0.25

# LIVE-DAY GUARD — pocket-dojo sim day, hindsight-contaminated. Excluded from
# every table and fit; used only for anchor calibration.
LIVE_DAY = "2024_09_16"
EXCLUDED_DAYS = {LIVE_DAY}

# ET minutes-of-day (fractional; seconds included). RTH per repo convention
# (research/reversal_gauge/builders/extract_freeze_events.py).
RTH_START_MIN = 9 * 60 + 30.0      # 09:30 inclusive
RTH_END_MIN = 15 * 60 + 30.0       # 15:30 exclusive — detection window
# Outcome scans may run to the cash close but never into the evening session
# of the SAME file (which belongs to the next trading day's tape).
OUTCOME_END_MIN = 16 * 60 + 0.0    # 16:00

# Repo-canonical leg definition: an 8.0pt close-to-close reversal terminates a
# leg. Same value as research/reversal_gauge REVERSAL_PT, so "leg" means the
# same thing in this package as in the gauge.
ZZ_REVERSAL_PT = 8.0

# Prior same-direction extremes stay "live" as reference levels for 90 min
# (research/reversal_gauge REPOKE_WINDOW_S).
EXTREME_MEMORY_S = 90 * 60


# ---------------------------------------------------------------------------
# day loading
# ---------------------------------------------------------------------------
def day_list(timeframes=("1s", "5s", "1m"), include_live=False) -> list[str]:
    """Canonical day list = intersection of the given timeframe dirs.

    Intersecting keeps N comparable across events (the 1s dir carries 6 extra
    non-standard files: *_BROWN / *_FOUR contract variants).
    """
    sets = []
    for tf in timeframes:
        paths = glob.glob(os.path.join(ATLAS_DIR, tf, "*.parquet"))
        sets.append({os.path.basename(p)[:-len(".parquet")] for p in paths})
    days = set.intersection(*sets) if sets else set()
    if not include_live:
        days -= EXCLUDED_DAYS
    return sorted(days)


class Day:
    """One day's bars at one timeframe, with ET clock precomputed.

    Attributes are plain numpy arrays so detectors can run tight loops.
    `mod` is fractional ET minutes-of-day; PRIOR-EVENING bars carry mod >= 1080
    (18:00) — see rule 1 in the module docstring.
    """

    __slots__ = ("day", "tf", "ts", "open", "high", "low", "close", "volume",
                 "mod", "n")

    def __init__(self, day: str, tf: str, df: pd.DataFrame):
        self.day = day
        self.tf = tf
        self.ts = df["timestamp"].to_numpy(np.int64)
        self.open = df["open"].to_numpy(np.float64)
        self.high = df["high"].to_numpy(np.float64)
        self.low = df["low"].to_numpy(np.float64)
        self.close = df["close"].to_numpy(np.float64)
        self.volume = (df["volume"].to_numpy(np.float64)
                       if "volume" in df.columns else np.zeros(len(df)))
        eti = pd.DatetimeIndex(
            pd.to_datetime(self.ts, unit="s", utc=True)).tz_convert(ET_TZ)
        self.mod = (eti.hour.to_numpy() * 60.0 + eti.minute.to_numpy()
                    + eti.second.to_numpy() / 60.0)
        self.n = self.ts.size

    def rth_mask(self, lo=RTH_START_MIN, hi=RTH_END_MIN) -> np.ndarray:
        """BOUNDED both sides — never an open-ended `mod >= lo`."""
        return (self.mod >= lo) & (self.mod < hi)

    def et_str(self, i: int) -> str:
        return (pd.Timestamp(int(self.ts[i]), unit="s", tz="UTC")
                .tz_convert(ET_TZ).strftime("%H:%M:%S"))


def load_day(day: str, tf: str) -> Day | None:
    path = os.path.join(ATLAS_DIR, tf, f"{day}.parquet")
    if not os.path.isfile(path):
        return None
    df = pd.read_parquet(
        path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    if len(df) < 2:
        return None
    return Day(day, tf, df)


# ---------------------------------------------------------------------------
# streaming zigzag (close-based, causal)
# ---------------------------------------------------------------------------
class ZigZag:
    """Close-based streaming zigzag, identical in logic to the one in
    research/reversal_gauge/builders/extract_freeze_events.py.

    A leg is ACTIVE from the bar its origin pivot was CONFIRMED (price moved
    `reversal_pt` off the origin close) — never back-dated. `step(i, c)`
    returns a dict on the bar where a pivot is confirmed, else None.

    Public state during an active leg:
      d          leg direction (+1 up / -1 down; 0 = not yet bootstrapped)
      anchor_px  origin pivot close, anchor_i its bar
      peak_px    running favourable extreme close, peak_i its bar
    """

    __slots__ = ("rev", "d", "run_min_px", "run_min_i", "run_max_px",
                 "run_max_i", "anchor_px", "anchor_i", "peak_px", "peak_i")

    def __init__(self, reversal_pt: float = ZZ_REVERSAL_PT):
        self.rev = float(reversal_pt)
        self.d = 0
        self.run_min_px = self.run_max_px = math.nan
        self.run_min_i = self.run_max_i = 0
        self.anchor_px = self.peak_px = math.nan
        self.anchor_i = self.peak_i = 0

    def step(self, i: int, c: float):
        if self.d == 0:
            if math.isnan(self.run_min_px):
                self.run_min_px = self.run_max_px = c
                self.run_min_i = self.run_max_i = i
            if c < self.run_min_px:
                self.run_min_px, self.run_min_i = c, i
            if c > self.run_max_px:
                self.run_max_px, self.run_max_i = c, i
            up = c - self.run_min_px >= self.rev
            dn = self.run_max_px - c >= self.rev
            if up and dn:
                # wide pre-leg range: the more recent extreme is the origin
                up = self.run_min_i > self.run_max_i
                dn = not up
            if up or dn:
                self.d = 1 if up else -1
                if up:
                    self.anchor_px, self.anchor_i = self.run_min_px, self.run_min_i
                else:
                    self.anchor_px, self.anchor_i = self.run_max_px, self.run_max_i
                self.peak_px, self.peak_i = c, i
                return dict(kind="bootstrap", d=self.d, pivot_px=self.anchor_px,
                            pivot_i=self.anchor_i, confirm_i=i)
            return None

        if self.d * (c - self.peak_px) > 0:
            self.peak_px, self.peak_i = c, i

        if self.d * (self.peak_px - c) >= self.rev:
            out = dict(kind="pivot", d=-self.d, pivot_px=self.peak_px,
                       pivot_i=self.peak_i, confirm_i=i,
                       prev_anchor_px=self.anchor_px, prev_anchor_i=self.anchor_i)
            self.anchor_px, self.anchor_i = self.peak_px, self.peak_i
            self.d = -self.d
            self.peak_px, self.peak_i = c, i
            return out
        return None

    @property
    def mfe(self) -> float:
        if self.d == 0:
            return 0.0
        return self.d * (self.peak_px - self.anchor_px)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
Z95 = 1.96                      # 95% normal quantile
BOOT_N = 4000                   # bootstrap resamples (CLAUDE.md metric spec)
UNDERPOWERED_N = 30             # below this N a table must be flagged


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """(p_hat, lo, hi) — Wilson score interval for a proportion."""
    if n <= 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, ctr - hw), min(1.0, ctr + hw)


def pct_ci(k: int, n: int, label: str = "") -> str:
    """'53% [45%, 61%] (n=210)' — never a bare point estimate."""
    if n <= 0:
        return f"{label}n/a (n=0)"
    p, lo, hi = wilson(k, n)
    flag = "  **UNDERPOWERED**" if n < UNDERPOWERED_N else ""
    return f"{label}{p:.1%} [{lo:.1%}, {hi:.1%}] (n={n}){flag}"


def quart(x, unit: str = "pt") -> str:
    """'median +12.5pt [q25 +3.0, q75 +28.2] (n=310)'."""
    a = np.asarray([v for v in np.asarray(x, dtype=float) if np.isfinite(v)])
    if a.size == 0:
        return "n/a (n=0)"
    flag = "  **UNDERPOWERED**" if a.size < UNDERPOWERED_N else ""
    return (f"median {np.median(a):+.2f}{unit} [q25 {np.percentile(a, 25):+.2f}, "
            f"q75 {np.percentile(a, 75):+.2f}] (n={a.size}){flag}")


def boot_median_ci(x, n_boot: int = BOOT_N, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray([v for v in np.asarray(x, dtype=float) if np.isfinite(v)])
    if a.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    meds = np.median(a[idx], axis=1)
    return float(np.median(a)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def prop_delta_ci(k1: int, n1: int, k2: int, n2: int,
                  n_boot: int = BOOT_N, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI on p2 - p1 (independent binomials). CLAUDE.md: a delta
    whose 95% CI includes 0 is NOT significant and must be said so."""
    if n1 <= 0 or n2 <= 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    b1 = rng.binomial(n1, k1 / n1, n_boot) / n1
    b2 = rng.binomial(n2, k2 / n2, n_boot) / n2
    d = b2 - b1
    return (k2 / n2 - k1 / n1, float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


def sig_note(lo: float, hi: float) -> str:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "n/a"
    return "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant (CI includes 0)"


# ---------------------------------------------------------------------------
# forward-scan helpers (outcome side of the causality boundary)
# ---------------------------------------------------------------------------
def idx_at_or_before(ts: np.ndarray, t: int, lo: int = 0) -> int | None:
    """Last bar index with ts <= t, searching from `lo`. None if none exists."""
    j = int(np.searchsorted(ts, t, side="right")) - 1
    return j if j >= lo else None


def forward_slice(d: Day, i0: int, horizon_s: int) -> tuple[int, int]:
    """[i0+1, i_end) bar range: forward `horizon_s` seconds, hard-clipped at
    OUTCOME_END_MIN so no outcome ever reads the evening session."""
    t_end = int(d.ts[i0]) + int(horizon_s)
    j = int(np.searchsorted(d.ts, t_end, side="right"))
    # clip at the cash close
    after = np.flatnonzero(d.mod[i0 + 1:] >= OUTCOME_END_MIN)
    if after.size:
        j = min(j, i0 + 1 + int(after[0]))
    return i0 + 1, max(i0 + 1, j)
