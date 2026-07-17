"""
Exit Dojo -- episode builder (research/exit_dojo/builders/episode_builder.py)

Builds "episodes": frame-by-frame text replays of real historical trade entries for
an LLM agent to play HOLD/EXIT decisions against. This is hypothesis GENERATION for
exit rules -- NOT a live decider, NOT a sealed-test result. See ../README.md for the
full leakage caveat.

INPUTS (all pre-existing, read-only; never written by this script):
  - research/nt8_catalog/reports/econ_drift_rows.parquet
        candidate ENTRY fires: ts, day, det, is_long, P, decile, split, drift_Xm, trunc_Xm
  - research/nt8_catalog/reports/signal_rows_{EXITKMDR,TURNCLIMAX,TURNHA,PROPTURNP}.parquet
        AUX fire streams (ts, is_long, day, ...) -> the "fires last 3m" frame field
  - DATA/ATLAS/5s/<day>.parquet
        causal 5s OHLCV stream (the only price source used to build frames)
  - DATA/ai_cusp_picks/ai_picks_YYYY-MM-DD_multi.json
        hindsight "oracle" trade labels (entry_ts/exit_ts/direction). Used ONLY for
        (a) episode SELECTION stratification and (b) the ground-truth sidecar.
        NEVER copied into a packet (episodes/ep_NN.md).

SIGN CONVENTION (verified empirically against DATA/ATLAS/5s -- see README.md):
econ_drift_rows.drift_Xm = sign * (close_future - close_now) where sign = +1 for
is_long=True, -1 for is_long=False -- i.e. FAVORABLE-SIGNED: positive always means
"good for the position taken". Every point-delta emitted into a frame (px-from-entry
AND the 1m-bar O/H/L/C) uses this SAME convention, for internal consistency.

LEG GEOMETRY ("leg: age Xm, amp Ypts, giveback Z%") mirrors the running-pivot /
running-extreme / amplitude / giveback state machine in
research/nt8_catalog/tools/dossier_signal_pipeline.py::_propturn_core (lines
1389-1434), reusing its frozen STATIC constants (PROPTURN_R/S_MIN/A_MIN, lines
1376-1378). `track_leg_state()` below is a plain-Python re-implementation (the
original is numba-jitted and only returns emitted fires; we need a per-bar
snapshot for descriptive display). Giveback CAN exceed 100% for sub-A_min-amplitude
legs (the original tracker's "escape clause" territory) -- that is faithful
behavior, not a bug; see the docstring on track_leg_state().

ER10 = Kaufman efficiency ratio on 1-minute closes, N=10 (dossier_signal_pipeline.py
gen_ctx_er / _pp_arrays; ER_N=10 at lines 912 / 1532). vol(5m) = std of the last 60
5s closes, ddof=1 (same definition as `tvol` in _pp_arrays, line 1538).

Run: python research/exit_dojo/builders/episode_builder.py [--seed N]
"""
import os
import sys
import json
import string
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- paths -----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DOJO_ROOT = os.path.abspath(os.path.join(HERE, '..'))
NT8_REPORTS_DIR = os.path.join(ROOT, 'research', 'nt8_catalog', 'reports')
ECON_DRIFT_PATH = os.path.join(NT8_REPORTS_DIR, 'econ_drift_rows.parquet')
ATLAS_5S_DIR = os.path.join(ROOT, 'DATA', 'ATLAS', '5s')
AI_CUSP_DIR = os.path.join(ROOT, 'DATA', 'ai_cusp_picks')
EPISODES_DIR = os.path.join(DOJO_ROOT, 'episodes')
TRUTH_DIR = os.path.join(EPISODES_DIR, 'truth')
REPORTS_DIR = os.path.join(DOJO_ROOT, 'reports')

AUX_TAGS = {                          # aux fire stream -> short display tag
    'EXITKMDR': 'KMDR',
    'TURNCLIMAX': 'CLIMAX',
    'TURNHA': 'HA',
    'PROPTURNP': 'PROPP',
}

# ---- declared constants (house rule: no bare magic numbers) -----------------------
RTH0 = pd.Timestamp('08:30').time()          # dossier_signal_pipeline.py:36
RTH1 = pd.Timestamp('15:15').time()          # dossier_signal_pipeline.py:36 ("session end")
PROPTURN_R = 0.05                            # dossier_signal_pipeline.py:1376 (retrace frac)
PROPTURN_S_MIN = 3.0                         # dossier_signal_pipeline.py:1377 (stall gate, min)
PROPTURN_A_MIN = 15.0                        # dossier_signal_pipeline.py:1378 (min leg amp, pts)
ER_N = 10                                    # dossier_signal_pipeline.py:912 (Kaufman ER window)
VOL_WINDOW_5S_BARS = 60                      # 60 x 5s = 5 min, matches `tvol` (:1538)
AUX_LOOKBACK_S = 180                         # "fires last 3m"

MAX_WINDOW_MIN = 40                          # spec: frames to min(40min, session end)
MIN_WINDOW_FOR_SELECTION = 20                # pilot-selection floor: fits all 4 bucket rules
CHOP_TOL_PTS = 4.0                           # DECLARED DEVIATION from spec's "+-3 pts": an
                                              # empirical sweep (see README.md "Declared
                                              # deviations") found only 1/282 decile-9 test days
                                              # has ANY 15-min stretch within +-3pts (2025_11_27);
                                              # +-4pts unlocks 9 days (+-5=26, +-6=50) -- decile-9
                                              # (highest-confidence) fires essentially never stay
                                              # this flat. +-4pts is the minimal widening that gives
                                              # enough days to pick 2 diverse chop episodes from.
CHOP_MIN_MIN = 15                            # "...for 15+ min"
WINNER_MIN_MIN = 20                          # "label persisted >=20 min"
MIDFLIP_LO_MIN, MIDFLIP_HI_MIN = 5, 15       # "label flipped 5-15 min in"
INSTANTFAIL_MAX_MIN = 5                      # "flip <5 min"
MIN_ORACLE_FOR_RATIO = 0.5                   # pts; scorer's own guard, kept here for one source

SELECTION_SEED = 20260713                    # fixed arbitrary seed; rerun with --seed to explore
BUCKET_TARGETS = {'winner': 3, 'midflip': 3, 'instantfail': 2, 'chop': 2}   # spec: 3/3/2/2 = 10

DECISION_CONTRACT = (
    "You are drilling exits on a real historical trade replay. Frames are strictly "
    "chronological. Process them IN ORDER. For EACH frame output exactly one line: "
    "`t=<min>: HOLD|EXIT — <one-line reason>` committing your decision using ONLY "
    "frames up to that t. After your first EXIT, output nothing further except a "
    "final line `SUMMARY: <2-3 sentences: what signature drove your exit, what you'd "
    "watch next time>`. If you never exit, end with the same SUMMARY line. Do not "
    "reference frames after your exit point anywhere."
)


@dataclass
class DayData:
    day: str
    ts5: np.ndarray
    o5: np.ndarray
    h5: np.ndarray
    l5: np.ndarray
    c5: np.ndarray
    session_end: Optional[int]
    oracle_ivals: List[Tuple[float, float, bool]]


# ---- generic helpers ---------------------------------------------------------------
def asof_idx(ts_arr: np.ndarray, t: int) -> int:
    """Index of the last bar with ts <= t (causal 'as-of' lookup); clamped to a valid index."""
    idx = np.searchsorted(ts_arr, t, side='right') - 1
    return int(np.clip(idx, 0, len(ts_arr) - 1))


def compute_session_end(ts5: np.ndarray) -> Optional[int]:
    """Unix ts of the last RTH (08:30-15:15 America/Chicago) 5s bar in this day's file."""
    dt = pd.to_datetime(ts5, unit='s', utc=True).tz_convert('America/Chicago')
    tt = dt.time
    rth_mask = (tt >= RTH0) & (tt <= RTH1)
    if not rth_mask.any():
        return None
    return int(ts5[rth_mask].max())


def load_oracle_intervals(day: str) -> List[Tuple[float, float, bool]]:
    """ai_cusp_picks hindsight trades for `day`, as sorted (entry_ts, exit_ts, is_long)."""
    fn = f"ai_picks_{day.replace('_', '-')}_multi.json"
    path = os.path.join(AI_CUSP_DIR, fn)
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    ivals = [(float(t['entry_ts']), float(t['exit_ts']), t['direction'] == 'LONG')
             for t in data.get('trades', [])]
    ivals.sort(key=lambda x: x[0])
    return ivals


def oracle_direction_at(ivals: List[Tuple[float, float, bool]], t: float) -> Optional[bool]:
    """Hindsight-optimal direction active at time t; None if t falls in a flat/gap."""
    for e0, e1, is_long in ivals:
        if e0 <= t <= e1:
            return is_long
    return None


def label_flip_minute(ivals, entry_ts: int, is_long0: bool, window_minutes: int) -> Optional[int]:
    """First minute (0..window_minutes) at which the oracle's active direction is the
    OPPOSITE of is_long0 (a genuine contradicting hindsight trade). None = never flips
    (persists, or stays in agreement/gap) within the window."""
    for m in range(0, window_minutes + 1):
        od = oracle_direction_at(ivals, entry_ts + m * 60)
        if od is not None and od != is_long0:
            return m
    return None


def clean0(x: float) -> float:
    """Collapses IEEE-754 negative-zero (e.g. sign=-1.0 * 0.0 == -0.0) to plain 0.0 so
    formatted output never shows a confusing '-0.00'."""
    return 0.0 if abs(x) < 1e-9 else x


def signed_drift_path(ts5, c5, entry_ts: int, is_long0: bool, window_minutes: int):
    """Favorable-signed point drift at each minute 0..window_minutes, plus entry price."""
    sign = 1.0 if is_long0 else -1.0
    entry_idx = asof_idx(ts5, entry_ts)
    entry_price = float(c5[entry_idx])
    path = []
    for m in range(0, window_minutes + 1):
        idx = asof_idx(ts5, entry_ts + m * 60)
        path.append(clean0(sign * (float(c5[idx]) - entry_price)))
    return path, entry_price


def is_chop(drift_path: List[float], minutes: int = CHOP_MIN_MIN, tol: float = CHOP_TOL_PTS) -> bool:
    """True if drift stays within +-tol pts for ANY 15+-consecutive-minute stretch
    inside the episode window (not required to start at t=0 -- a decile-9/high-P
    entry that moves early then stalls is just as legitimately "chop" for exit-drilling
    purposes as one that is flat from the first minute; empirically, requiring
    flatness from t=0 made 'chop' almost unfindable in the decile-9 pool, see
    README.md). A sliding window over [0, window_minutes]."""
    n = len(drift_path)
    if n <= minutes:
        return False
    for s in range(0, n - minutes):
        if all(abs(v) <= tol for v in drift_path[s:s + minutes + 1]):
            return True
    return False


BUCKET_PRIORITY = ('chop', 'instantfail', 'midflip', 'winner')   # scarcest-first tie-break


def natural_buckets(label_end_min: Optional[int], chop: bool) -> List[str]:
    """All pilot buckets a candidate qualifies for on its own merits (quota-independent;
    chop and the label-timing buckets are computed independently and CAN co-occur --
    e.g. flat price with a persisting label -- so a candidate may match more than one)."""
    buckets = []
    if chop:
        buckets.append('chop')
    if label_end_min is not None and label_end_min < INSTANTFAIL_MAX_MIN:
        buckets.append('instantfail')
    if label_end_min is not None and MIDFLIP_LO_MIN <= label_end_min <= MIDFLIP_HI_MIN:
        buckets.append('midflip')
    if label_end_min is None or label_end_min >= WINNER_MIN_MIN:
        buckets.append('winner')
    return buckets


# ---- leg geometry (PROP-TURN-P approach; see module docstring) --------------------
def track_leg_state(ts5: np.ndarray, c5: np.ndarray):
    """Per-bar (piv_ts, amplitude, giveback_fraction) snapshots of the running-pivot
    leg tracker, mirroring dossier_signal_pipeline.py::_propturn_core's state machine
    verbatim (P0/d/hi_v/lo_v/hi_ts/lo_ts update every bar; STATIC frozen constants;
    no RTH/emission gating here since we want continuous descriptive state, not
    fires). giveback_fraction = retrace / amplitude and CAN exceed 1.0 for legs whose
    amplitude is below PROPTURN_A_MIN (the original's "escape clause" territory,
    where a full A_MIN countermove -- not a fraction of a small amplitude -- is
    needed to re-designate the leg); this is faithful to the source, not a bug."""
    n = len(c5)
    piv_ts = np.empty(n, dtype=np.int64)
    amp = np.empty(n, dtype=np.float64)
    giveback = np.empty(n, dtype=np.float64)
    s_sec = PROPTURN_S_MIN * 60.0

    P0 = c5[0]
    d = 0                      # 0 unseeded (treated as an up-leg watch), +1 up, -1 down
    hi_v = lo_v = c5[0]
    hi_ts = lo_ts = piv = int(ts5[0])

    for i in range(n):
        x = c5[i]
        t = int(ts5[i])
        if i > 0:
            if x > hi_v:
                hi_v = x
                hi_ts = t
            if x < lo_v:
                lo_v = x
                lo_ts = t
            fired = False
            if d >= 0:
                A = hi_v - P0
                retr = hi_v - x
                if (((A >= PROPTURN_A_MIN and retr >= PROPTURN_R * A) or
                     (A < PROPTURN_A_MIN and retr >= PROPTURN_A_MIN)) and
                        (s_sec <= 0.0 or (t - hi_ts) >= s_sec)):
                    P0 = hi_v
                    d = -1
                    lo_v = x
                    lo_ts = t
                    piv = t
                    fired = True
            if (not fired) and d <= 0:
                A = P0 - lo_v
                retr = x - lo_v
                if (((A >= PROPTURN_A_MIN and retr >= PROPTURN_R * A) or
                     (A < PROPTURN_A_MIN and retr >= PROPTURN_A_MIN)) and
                        (s_sec <= 0.0 or (t - lo_ts) >= s_sec)):
                    P0 = lo_v
                    d = 1
                    hi_v = x
                    hi_ts = t
                    piv = t
                    fired = True
        if d >= 0:
            A_cur = hi_v - P0
            retr_cur = hi_v - x
        else:
            A_cur = P0 - lo_v
            retr_cur = x - lo_v
        piv_ts[i] = piv
        amp[i] = A_cur
        giveback[i] = (retr_cur / A_cur) if A_cur > 0 else 0.0

    return piv_ts, amp, giveback


# ---- 1m bucketing / ER10 -----------------------------------------------------------
def build_1m_ohlc(ts5, o5, h5, l5, c5) -> pd.DataFrame:
    """1-minute OHLC bars from the 5s stream, indexed by integer bucket id (ts//60)."""
    bucket = ts5 // 60
    idx = pd.Index(bucket, name='bucket')
    df = pd.DataFrame({'o': o5, 'h': h5, 'l': l5, 'c': c5}, index=idx)
    g = df.groupby(level=0)
    return pd.DataFrame({'o': g['o'].first(), 'h': g['h'].max(),
                          'l': g['l'].min(), 'c': g['c'].last()})


def compute_er10_series(c1: pd.Series, n: int = ER_N) -> pd.Series:
    """Kaufman efficiency ratio on a 1-minute close series (dossier_signal_pipeline.py
    gen_ctx_er / _pp_arrays convention): |c[k]-c[k-n]| / sum(|diff(c)|, last n)."""
    dc = c1.diff().abs()
    denom = dc.rolling(n, min_periods=n).sum()
    net = (c1 - c1.shift(n)).abs()
    return net / denom.replace(0, np.nan)


# ---- data loading -------------------------------------------------------------------
def load_econ_drift() -> pd.DataFrame:
    df = pd.read_parquet(ECON_DRIFT_PATH)
    df = df[(df['split'] == 'test') & (df['decile'] == 9) &
            (df['day'].str[:4].isin(['2025', '2026']))]
    df = df.drop_duplicates(subset=['day', 'ts', 'is_long'], keep='first').reset_index(drop=True)
    return df


def load_day_data(day: str) -> Optional[DayData]:
    path = os.path.join(ATLAS_5S_DIR, f'{day}.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    ts5 = df['timestamp'].values.astype(np.int64)
    session_end = compute_session_end(ts5)
    if session_end is None:
        return None
    return DayData(day=day, ts5=ts5,
                   o5=df['open'].values.astype(np.float64),
                   h5=df['high'].values.astype(np.float64),
                   l5=df['low'].values.astype(np.float64),
                   c5=df['close'].values.astype(np.float64),
                   session_end=session_end,
                   oracle_ivals=load_oracle_intervals(day))


def load_aux_data() -> Dict[str, pd.DataFrame]:
    out = {}
    for name in AUX_TAGS:
        path = os.path.join(NT8_REPORTS_DIR, f'signal_rows_{name}.parquet')
        out[name] = pd.read_parquet(path, columns=['ts', 'is_long', 'day'])
    return out


# ---- selection ----------------------------------------------------------------------
def select_episodes(seed: int = SELECTION_SEED):
    """Scans candidate days in a seeded-random order and takes AT MOST ONE episode per
    REAL day (diversity constraint, not in the literal spec but necessary: a single
    high-fire day can otherwise fill most buckets by itself, defeating the point of a
    2025-26 stratified pilot -- see README.md)."""
    econ = load_econ_drift().sample(frac=1, random_state=seed).reset_index(drop=True)
    day_groups = {d: g for d, g in econ.groupby('day', sort=False)}
    days = np.array(list(day_groups.keys()))
    rng = np.random.default_rng(seed)
    days = rng.permutation(days)

    bucket_counts = {k: 0 for k in BUCKET_TARGETS}
    total_target = sum(BUCKET_TARGETS.values())
    selected = []
    day_cache: Dict[str, Optional[DayData]] = {}
    scanned_days = 0
    scanned_rows = 0

    pbar = tqdm(days, desc='scanning days for pilot selection')
    for day in pbar:
        if sum(bucket_counts.values()) >= total_target:
            break
        dd = day_cache.get(day)
        if dd is None and day not in day_cache:
            dd = load_day_data(day)
            day_cache[day] = dd
        if dd is None:
            continue
        scanned_days += 1
        pbar.set_postfix(bucket_counts)

        # Scan the FULL day (not stopping at the first hit) so a day's rarer-bucket
        # candidate isn't shadowed by a more-common one earlier in row order; record
        # only the FIRST candidate found per bucket (still >=1 per day is plenty).
        open_buckets = {k for k, v in bucket_counts.items() if v < BUCKET_TARGETS[k]}
        day_matches: Dict[str, dict] = {}
        for r in day_groups[day].itertuples(index=False):
            scanned_rows += 1
            if set(day_matches) >= open_buckets:
                break   # this day already covers every still-open bucket
            entry_ts = int(r.ts)
            is_long0 = bool(r.is_long)
            window_minutes = min(MAX_WINDOW_MIN, (dd.session_end - entry_ts) // 60)
            if window_minutes < MIN_WINDOW_FOR_SELECTION:
                continue
            drift_path, entry_price = signed_drift_path(dd.ts5, dd.c5, entry_ts, is_long0, window_minutes)
            chop = is_chop(drift_path)
            lem = label_flip_minute(dd.oracle_ivals, entry_ts, is_long0, window_minutes)
            for bucket in natural_buckets(lem, chop):
                if bucket in open_buckets and bucket not in day_matches:
                    oracle_min = lem if lem is not None else window_minutes
                    day_matches[bucket] = dict(
                        day=day, ts=entry_ts, is_long=is_long0, P=float(r.P), det=r.det,
                        type=bucket, window_minutes=int(window_minutes), label_end_minute=lem,
                        oracle_capture=float(drift_path[oracle_min]), oracle_minute=oracle_min,
                        per_minute_forward_drift=drift_path, entry_price=entry_price, chop_flag=chop,
                    )

        # diversity constraint: at most ONE selected episode per real day -- assign
        # this day's single slot to whichever open bucket is scarcest (BUCKET_PRIORITY).
        for bucket in BUCKET_PRIORITY:
            if bucket in day_matches:
                selected.append(day_matches[bucket])
                bucket_counts[bucket] += 1
                break
        if sum(bucket_counts.values()) >= total_target:
            break

    missing = {k: BUCKET_TARGETS[k] - v for k, v in bucket_counts.items() if v < BUCKET_TARGETS[k]}
    if missing:
        raise RuntimeError(
            f'could not fill all buckets after scanning {scanned_days} days / {scanned_rows} '
            f'candidate rows; still short: {missing}. Consider relaxing MIN_WINDOW_FOR_SELECTION '
            f'or scanning more days (pool exhausted at {len(days)} days).')

    ordered = []
    for b in BUCKET_TARGETS:
        ordered.extend(s for s in selected if s['type'] == b)
    return ordered, dict(scanned_days=scanned_days, scanned_rows=scanned_rows, seed=seed)


# ---- frame construction ---------------------------------------------------------------
def aux_fires_text(day_aux: Dict[str, pd.DataFrame], frame_ts: int, is_long0: bool) -> str:
    parts = []
    for name, tag in AUX_TAGS.items():
        sub = day_aux[name]
        m = sub[(sub['ts'] <= frame_ts) & (sub['ts'] >= frame_ts - AUX_LOOKBACK_S)]
        if len(m) == 0:
            continue
        rr = m.loc[m['ts'].idxmax()]
        mins_ago = int(round((frame_ts - rr['ts']) / 60.0))
        rel = 'with' if bool(rr['is_long']) == is_long0 else 'against'
        parts.append(f"{tag}-{rel}({mins_ago}m ago)")
    if not parts:
        return "none"
    txt = ", ".join(parts)
    if len(parts) < len(AUX_TAGS):
        txt += ", none-else"
    return txt


def build_frames_for_episode(sel: dict, aux_data: Dict[str, pd.DataFrame], dd: DayData) -> List[str]:
    ts5, o5, h5, l5, c5 = dd.ts5, dd.o5, dd.h5, dd.l5, dd.c5
    piv_ts_arr, amp_arr, giveback_arr = track_leg_state(ts5, c5)
    ohlc1m = build_1m_ohlc(ts5, o5, h5, l5, c5)
    er10 = compute_er10_series(ohlc1m['c'])
    sign = 1.0 if sel['is_long'] else -1.0
    entry_ts = sel['ts']
    day_aux = {name: df[df['day'] == sel['day']] for name, df in aux_data.items()}

    lines = []
    for m in range(0, sel['window_minutes'] + 1):
        frame_ts = entry_ts + m * 60
        idx = asof_idx(ts5, frame_ts)
        px = sel['per_minute_forward_drift'][m]

        cur_bucket = int(ts5[idx] // 60)
        closed_bucket = cur_bucket - 1
        if closed_bucket in ohlc1m.index and (closed_bucket - 1) in ohlc1m.index:
            row = ohlc1m.loc[closed_bucket]
            prior_c = ohlc1m.loc[closed_bucket - 1, 'c']
            bo, bh, bl, bc = (clean0(sign * (row['o'] - prior_c)), clean0(sign * (row['h'] - prior_c)),
                              clean0(sign * (row['l'] - prior_c)), clean0(sign * (row['c'] - prior_c)))
            bar_txt = f"O{bo:+.2f} H{bh:+.2f} L{bl:+.2f} C{bc:+.2f}"
        else:
            bar_txt = "n/a (insufficient bucket history)"

        leg_age_min = (int(ts5[idx]) - int(piv_ts_arr[idx])) / 60.0
        amp = float(amp_arr[idx])
        giveback = float(giveback_arr[idx])

        vol_lo = max(0, idx - VOL_WINDOW_5S_BARS + 1)
        vol_win = c5[vol_lo: idx + 1]
        vol5m = float(np.std(vol_win, ddof=1)) if len(vol_win) >= 2 else float('nan')

        er_val = er10.get(closed_bucket, np.nan)
        er_txt = f"{er_val:.2f}" if np.isfinite(er_val) else "n/a"

        fires_txt = aux_fires_text(day_aux, frame_ts, sel['is_long'])

        line = (f"[t={m}m] px:{clean0(px):+.2f}pts from entry | 1m bar: {bar_txt} (rel prior close) | "
                f"leg: age {leg_age_min:.0f}m, amp {amp:.1f}pts, giveback {giveback * 100:.0f}% | "
                f"vol(5m): {vol5m:.1f}pts | ER10: {er_txt} | fires last 3m: {fires_txt} | "
                f"entryP: {sel['P']:.2f}")
        lines.append(line)
    return lines


# ---- output writers -------------------------------------------------------------------
def anonymize_days(selected: List[dict]) -> Dict[str, str]:
    mapping = {}
    for s in selected:
        if s['day'] not in mapping:
            mapping[s['day']] = f"Day {string.ascii_uppercase[len(mapping)]}"
    return mapping


def write_episode(ep_num: int, sel: dict, frame_lines: List[str], day_label: str) -> Tuple[str, str]:
    ep_id = f"ep_{ep_num:02d}"
    direction = "LONG" if sel['is_long'] else "SHORT"
    md = [
        f"# Episode {ep_num:02d}",
        "",
        f"**Direction:** {direction}  ",
        f"**Entry P:** {sel['P']:.2f}  ",
        f"**Day:** {day_label}  ",
        "**All times below are minutes elapsed since entry (t=0m = entry).**",
        "",
        "**Convention:** every point-delta below (px and the 1m-bar O/H/L/C) is "
        "FAVORABLE-SIGNED -- positive always means good for this position, negative "
        "always bad, regardless of LONG/SHORT.",
        "",
        "---",
        "",
        "## Decision contract",
        "",
        DECISION_CONTRACT,
        "",
        "---",
        "",
        "## Frames",
        "",
        *frame_lines,
        "",
    ]
    md_path = os.path.join(EPISODES_DIR, f"{ep_id}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))

    truth = dict(
        episode_id=ep_id, type=sel['type'], is_long=sel['is_long'], entry_ts=sel['ts'],
        entry_price=sel['entry_price'], det=sel['det'], P=sel['P'],
        window_minutes=sel['window_minutes'], label_end_minute=sel['label_end_minute'],
        oracle_capture=sel['oracle_capture'], oracle_minute=sel['oracle_minute'],
        per_minute_forward_drift=sel['per_minute_forward_drift'],
        chop_flag=sel['chop_flag'], real_day=sel['day'],
    )
    truth_path = os.path.join(TRUTH_DIR, f"{ep_id}.json")
    with open(truth_path, 'w', encoding='utf-8') as f:
        json.dump(truth, f, indent=2)
    return md_path, truth_path


def write_selection_table(selected: List[dict], day_label_map: Dict[str, str], scan_stats: dict) -> str:
    lines = [
        "# Exit Dojo -- pilot selection table",
        "",
        f"Seed={scan_stats['seed']}; scanned {scan_stats['scanned_days']} days / "
        f"{scan_stats['scanned_rows']} candidate rows (test-split, decile-9, 2025-26) "
        f"to fill the {sum(BUCKET_TARGETS.values())}-episode stratified pilot "
        f"({', '.join(f'{v} {k}' for k, v in BUCKET_TARGETS.items())}).",
        "",
        "| ep | type | anon day | real day | det | entry ts (UTC) | is_long | P | "
        "window (min) | label_end_min | oracle capture (pts) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(selected, start=1):
        lines.append(
            f"| ep_{i:02d} | {s['type']} | {day_label_map[s['day']]} | {s['day']} | {s['det']} | "
            f"{s['ts']} | {'LONG' if s['is_long'] else 'SHORT'} | {s['P']:.3f} | "
            f"{s['window_minutes']} | {s['label_end_minute']} | {s['oracle_capture']:+.2f} |"
        )
    path = os.path.join(REPORTS_DIR, 'selection_table.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    return path


# ---- main ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=SELECTION_SEED)
    args = ap.parse_args()

    os.makedirs(EPISODES_DIR, exist_ok=True)
    os.makedirs(TRUTH_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f'selecting {sum(BUCKET_TARGETS.values())} episodes ({BUCKET_TARGETS}) seed={args.seed} ...')
    selected, scan_stats = select_episodes(seed=args.seed)
    day_label_map = anonymize_days(selected)

    table_path = write_selection_table(selected, day_label_map, scan_stats)
    print(f'\nwrote {table_path}\n')
    print(f"{'ep':6s} {'type':12s} {'day':7s} {'det':10s} {'entry_ts':12s} {'dir':6s} {'P':6s} "
          f"{'win':4s} {'lbl_end':8s} {'oracle_cap':10s}")
    for i, s in enumerate(selected, start=1):
        print(f"ep_{i:02d}  {s['type']:12s} {day_label_map[s['day']]:7s} {s['det']:10s} "
              f"{s['ts']:<12d} {'LONG' if s['is_long'] else 'SHORT':6s} {s['P']:.3f}  "
              f"{s['window_minutes']:<4d} {str(s['label_end_minute']):8s} {s['oracle_capture']:+.2f}")

    aux_data = load_aux_data()
    day_cache: Dict[str, DayData] = {}
    print('\nbuilding frame packets ...')
    for i, sel in enumerate(tqdm(selected, desc='episodes'), start=1):
        dd = day_cache.get(sel['day'])
        if dd is None:
            dd = load_day_data(sel['day'])
            day_cache[sel['day']] = dd
        frame_lines = build_frames_for_episode(sel, aux_data, dd)
        md_path, truth_path = write_episode(i, sel, frame_lines, day_label_map[sel['day']])
        tqdm.write(f'  ep_{i:02d} ({sel["type"]}): {md_path}')

    print(f'\ndone. {len(selected)} episodes written to {EPISODES_DIR}')
    print(f'ground-truth sidecars written to {TRUTH_DIR}')


if __name__ == '__main__':
    main()
