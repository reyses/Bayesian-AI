"""Regret analysis with automated 1m-CRM-peak oracle.

Per user 2026-05-12 / 2026-05-14: replace human picking with automated oracle —
  1. Detect every local maximum of RAW PRICE (bar high) → oracle SHORT entry
  2. Detect every local minimum of RAW PRICE (bar low)  → oracle LONG entry
     (the opportunity is defined by price truth; the CRMs/anchors are the
      analysis layer — the state vector — not the detection series)
  3. From each entry, measure forward MFE over NEXT 60 MINUTES
     (1-hour trade constraint matches user's typical hold time)
  4. Snapshot indicator state at entry
  Base TF defaults to 5s for entry-pricing + forward-MFE-timing precision.
  5. Build empirical P(state) tables to ask:
       P(state X at oracle entries) / P(state X at random bars)
       → lift > 1 means state X discriminates oracle entries

Session-aware (per user 2026-05-14): designed to run as ONE continuous pass
over the full IS (2025-01-01 → 2025-12-31). The 1m stream is segmented into
Globex sessions by empirical gap detection (the ~61-min maintenance halt at
5-6 PM ET — UTC time DST-shifts, so it is never hardcoded). Neither the
extrema-detection window nor the forward-MFE window may cross a halt or
weekend gap. End-of-session extrema are kept and tagged (full_window=0,
available_fwd_min<60), never dropped — blender-first.

Run the full IS in one pass:
    python tools/regret_1m_oracle.py --start 2025-01-01 --end 2025-12-31 --name IS_full

Output structure:
    reports/findings/regret_oracle/
        oracle_entries_<run>.csv       one row per oracle entry; adds session_id,
                                       session_date, tod_minutes, full_window,
                                       available_fwd_min, mfe_velocity, volume, bar_range
        lift_table_<run>.csv           P(state|oracle) / P(state|random)

State vector at each oracle entry (mirrors the 4-layer framework):
    LAYER 1 — STRUCTURAL POSITION
        z_1h_high, z_1h_low, near_1h_Mh, near_1h_Ml,
        15m_above_1h_Mh, 15m_below_1h_Ml
    LAYER 2 — ALIGNMENT (stack order)
        crm_stack_order, align_state
    LAYER 3 — VELOCITY
        slope_15s_3m, slope_15m_5m, slope_15m_decel, fan_change_10m
    LAYER 4 — DISTANCE / EXTENSION
        z_15s, z_1m, z_15m, dist_15s_1m, fan_width
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cusp_marker import compute_anchor
from tools.sim_decay_rules import load_1m_bars


OUT_DIR = Path('reports/findings/regret_oracle')
TICK = 0.25
TICK_DOLLAR = 0.50
FORWARD_MINUTES = 60   # 60-min trade constraint (the 1-hour lapse premise)
FORWARD_BARS = 60      # legacy module constant — 1m-based importers (e.g.
                       # train_mfe_regressor) rely on this. run() derives its
                       # own forward-bar count from --tf; do not use here.
TF_SECONDS = {'5s': 5, '15s': 15, '30s': 30, '1m': 60, '5m': 300}
# A "session" = Globex maintenance-halt to maintenance-halt. The halt is a
# clean ~61-min gap in the bar stream (5-6 PM ET; UTC time DST-shifts, so we
# detect it empirically rather than hardcoding). The weekend gap (~48h) is
# also caught. Any gap > this threshold starts a new session.
SESSION_GAP_S = 30 * 60


def load_tf_bars(start_ts: float, end_ts: float, tf: str) -> pd.DataFrame:
    """Load OHLCV bars of any timeframe across [start_ts, end_ts] from the
    daily ATLAS parquets at DATA/ATLAS/<tf>/YYYY_MM_DD.parquet."""
    from datetime import timedelta
    dt_s = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    dt_e = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    parts = []
    cur = datetime(dt_s.year, dt_s.month, dt_s.day, tzinfo=timezone.utc)
    end_cap = datetime(dt_e.year, dt_e.month, dt_e.day, tzinfo=timezone.utc)
    while cur <= end_cap:
        path = f'DATA/ATLAS/{tf}/{cur.strftime("%Y_%m_%d")}.parquet'
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.copy()
                df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
            parts.append(df)
        cur = cur + timedelta(days=1)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].reset_index(drop=True)
    return df


def build_session_info(ts: np.ndarray, gap_threshold_s: int = SESSION_GAP_S):
    """Segment the 1m stream into Globex sessions by gap detection.
    Returns per-bar arrays: session_id, session_end_idx (last bar idx of the
    bar's session), session_first_ts, session_date (UTC date of the session's
    LAST bar = the 'trade date'), and the total session count."""
    n = len(ts)
    gaps = np.diff(ts)
    breaks = np.where(gaps > gap_threshold_s)[0]   # session ends AT index b
    bounds = []
    start = 0
    for b in breaks:
        bounds.append((start, int(b)))
        start = int(b) + 1
    bounds.append((start, n - 1))

    session_id       = np.zeros(n, dtype=int)
    session_end_idx  = np.zeros(n, dtype=int)
    session_first_ts = np.zeros(n, dtype=np.int64)
    session_date     = np.empty(n, dtype=object)
    for sid, (s, e) in enumerate(bounds):
        session_id[s:e + 1]       = sid
        session_end_idx[s:e + 1]  = e
        session_first_ts[s:e + 1] = ts[s]
        session_date[s:e + 1] = datetime.fromtimestamp(
            ts[e], tz=timezone.utc).strftime('%Y-%m-%d')
    return session_id, session_end_idx, session_first_ts, session_date, len(bounds)


def detect_local_extrema(series: np.ndarray, session_id: np.ndarray,
                                  window: int = 15, min_sep: int = 10) -> tuple:
    """Detect local maxima and minima of a series using a centered window.
    A bar is a local max if it's the MAXIMUM within [t-window, t+window].
    Returns (max_indices, min_indices) sorted by index.
    Enforces min_sep between consecutive extrema (avoid clustered detections).

    Session-aware: the centered detection window must not span a session
    boundary (a halt/weekend gap), and the min_sep counter resets at each
    session open (extrema across a 61-min gap are not 'clustered')."""
    n = len(series)
    max_idx = []
    min_idx = []
    last_max = -min_sep - 1
    last_min = -min_sep - 1
    for i in range(window, n - window):
        # Reset separation counters at session open
        if session_id[i] != session_id[i - 1]:
            last_max = i - min_sep - 1
            last_min = i - min_sep - 1
        # Skip if the centered window straddles a session boundary
        if session_id[i - window] != session_id[i + window]:
            continue
        if np.isnan(series[i]):
            continue
        win = series[i - window : i + window + 1]
        if np.isnan(win).any():
            continue
        if series[i] == win.max() and i - last_max >= min_sep:
            max_idx.append(i)
            last_max = i
        elif series[i] == win.min() and i - last_min >= min_sep:
            min_idx.append(i)
            last_min = i
    return np.array(max_idx, dtype=int), np.array(min_idx, dtype=int)


def forward_mfe(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                     entry_idx: int, entry_price: float, direction: str,
                     n_fwd: int, session_end_idx: int) -> tuple:
    """From entry, find MFE within next n_fwd bars in trade direction — but
    never past the entry's session end (a trade cannot be held across the
    maintenance halt). entry_price is the raw-price extreme (the bar HIGH for
    a SHORT peak, the bar LOW for a LONG trough). Returns
    (mfe_ticks, time_to_mfe_bars, exit_idx, exit_price, available_fwd_bars)."""
    end_idx = min(len(close) - 1, entry_idx + n_fwd, int(session_end_idx))
    available_fwd = end_idx - entry_idx
    if end_idx <= entry_idx:
        return 0.0, 0, entry_idx, entry_price, 0
    fwd_high = high[entry_idx + 1 : end_idx + 1]
    fwd_low = low[entry_idx + 1 : end_idx + 1]
    if direction == 'SHORT':
        # MFE = max excursion DOWN (entry - low)
        if len(fwd_low) == 0 or np.isnan(fwd_low).all():
            return 0.0, 0, entry_idx, entry_price, available_fwd
        mfe_idx_local = int(np.nanargmin(fwd_low))
        mfe_price = float(fwd_low[mfe_idx_local])
        mfe_ticks = (entry_price - mfe_price) / TICK
    else:  # LONG
        if len(fwd_high) == 0 or np.isnan(fwd_high).all():
            return 0.0, 0, entry_idx, entry_price, available_fwd
        mfe_idx_local = int(np.nanargmax(fwd_high))
        mfe_price = float(fwd_high[mfe_idx_local])
        mfe_ticks = (mfe_price - entry_price) / TICK
    exit_idx = entry_idx + 1 + mfe_idx_local
    time_to_mfe_bars = mfe_idx_local + 1
    return mfe_ticks, time_to_mfe_bars, exit_idx, mfe_price, available_fwd


# ── State vector extraction at oracle entry ────────────────────────────────

def extract_state_vector(idx: int, close: np.ndarray,
                                M_15s: np.ndarray, S_15s: np.ndarray,
                                M_1m: np.ndarray, S_1m: np.ndarray,
                                M_15m: np.ndarray, S_15m: np.ndarray,
                                Mh: np.ndarray, Sh: np.ndarray,
                                Ml: np.ndarray, Sl: np.ndarray,
                                Mc: np.ndarray, Sc: np.ndarray,
                                tf_seconds: int = 60) -> dict:
    """Compute full state vector at bar idx. None for unavailable values.
    tf_seconds = the base-bar TF (default 60 = 1m, so existing 1m importers
    are unaffected); slope lookbacks are specified in minutes and converted
    to bars for the current TF, normalized per-minute (TF-invariant)."""
    c = close[idx]
    def z(num_arr, denom_arr):
        if denom_arr is None or denom_arr[idx] <= 0 or np.isnan(denom_arr[idx]):
            return None
        return float((c - num_arr[idx]) / denom_arr[idx])

    state = {
        'z_15s':      None if np.isnan(M_15s[idx]) else round(z(M_15s, S_15s), 3),
        'z_1m':       None if np.isnan(M_1m[idx]) else round(z(M_1m, S_1m), 3),
        'z_15m':      None if np.isnan(M_15m[idx]) else round(z(M_15m, S_15m), 3),
        'z_1h_high':  None if np.isnan(Mh[idx]) else round(z(Mh, Sh), 3),
        'z_1h_low':   None if np.isnan(Ml[idx]) else round(z(Ml, Sl), 3),
    }

    # Distances (CRM-to-CRM in σ units)
    if not (np.isnan(M_15s[idx]) or np.isnan(M_1m[idx])) and S_1m[idx] > 0:
        state['dist_15s_1m'] = round(float((M_15s[idx] - M_1m[idx]) / S_1m[idx]), 3)
    else:
        state['dist_15s_1m'] = None
    if not (np.isnan(M_1m[idx]) or np.isnan(M_15m[idx])) and S_15m[idx] > 0:
        state['dist_1m_15m'] = round(float((M_1m[idx] - M_15m[idx]) / S_15m[idx]), 3)
    else:
        state['dist_1m_15m'] = None
    if not (np.isnan(M_15s[idx]) or np.isnan(M_15m[idx])) and S_15m[idx] > 0:
        state['dist_15s_15m'] = round(float((M_15s[idx] - M_15m[idx]) / S_15m[idx]), 3)
    else:
        state['dist_15s_15m'] = None
    if all(state[k] is not None for k in ('dist_15s_1m', 'dist_1m_15m', 'dist_15s_15m')):
        state['fan_width'] = round(abs(state['dist_15s_1m']) + abs(state['dist_1m_15m']) +
                                                abs(state['dist_15s_15m']), 3)
    else:
        state['fan_width'] = None

    # Stack order (top-to-bottom by price)
    stack_vals = []
    if not np.isnan(M_15s[idx]): stack_vals.append(('15s', float(M_15s[idx])))
    if not np.isnan(M_1m[idx]):  stack_vals.append(('1m',  float(M_1m[idx])))
    if not np.isnan(M_15m[idx]): stack_vals.append(('15m', float(M_15m[idx])))
    stack_vals.sort(key=lambda v: -v[1])
    state['crm_stack'] = '>'.join(s[0] for s in stack_vals) if stack_vals else 'NA'

    # Slopes — lookback given in MINUTES, converted to bars for the current
    # base TF, normalized per-minute so the feature is TF-invariant.
    def slope(arr, lb_min):
        lb = max(1, round(lb_min * 60 / tf_seconds))
        if idx < lb or np.isnan(arr[idx]) or np.isnan(arr[idx - lb]):
            return None
        return round(float((arr[idx] - arr[idx - lb]) / lb_min), 3)
    state['slope_15s_3m']  = slope(M_15s, 3)
    state['slope_15s_10m'] = slope(M_15s, 10)
    state['slope_1m_10m']  = slope(M_1m,  10)
    state['slope_15m_5m']  = slope(M_15m, 5)
    state['slope_15m_15m'] = slope(M_15m, 15)

    # 15m × 1h rail relationship (the key structural feature)
    if not np.isnan(M_15m[idx]) and not np.isnan(Mh[idx]) and Sh[idx] > 0:
        d_15m_hi = (M_15m[idx] - Mh[idx]) / Sh[idx]
        state['dist_15m_to_Mh'] = round(float(d_15m_hi), 3)
        state['15m_above_Mh'] = int(d_15m_hi > 0)
        state['15m_near_Mh']  = int(abs(d_15m_hi) <= 0.5)
    else:
        state['dist_15m_to_Mh'] = None
        state['15m_above_Mh'] = 0
        state['15m_near_Mh']  = 0
    if not np.isnan(M_15m[idx]) and not np.isnan(Ml[idx]) and Sl[idx] > 0:
        d_15m_lo = (M_15m[idx] - Ml[idx]) / Sl[idx]
        state['dist_15m_to_Ml'] = round(float(d_15m_lo), 3)
        state['15m_below_Ml'] = int(d_15m_lo < 0)
        state['15m_near_Ml']  = int(abs(d_15m_lo) <= 0.5)
    else:
        state['dist_15m_to_Ml'] = None
        state['15m_below_Ml'] = 0
        state['15m_near_Ml']  = 0

    return state


# ── Discretize state for empirical P-table ─────────────────────────────────

def discretize_state(state: dict, direction: str) -> dict:
    """Convert continuous state to categorical bins for empirical P lookups."""
    out = {'direction': direction}
    # Stack order (categorical already)
    out['stack'] = state.get('crm_stack', 'NA')
    # z_15m bin
    z15m = state.get('z_15m')
    out['z_15m_bin'] = (
        'ext_dn' if z15m is not None and z15m <= -1.5 else
        'mod_dn' if z15m is not None and z15m <= -0.5 else
        'mid'    if z15m is not None and z15m <= 0.5 else
        'mod_up' if z15m is not None and z15m <= 1.5 else
        'ext_up' if z15m is not None else 'NA'
    )
    # 15m × 1h rail position
    d_hi = state.get('dist_15m_to_Mh')
    d_lo = state.get('dist_15m_to_Ml')
    if d_hi is None or d_lo is None:
        out['rail_position'] = 'NA'
    elif d_hi > 0:
        out['rail_position'] = 'above_Mh'
    elif abs(d_hi) <= 0.5:
        out['rail_position'] = 'near_Mh'
    elif d_lo < 0:
        out['rail_position'] = 'below_Ml'
    elif abs(d_lo) <= 0.5:
        out['rail_position'] = 'near_Ml'
    else:
        out['rail_position'] = 'mid_band'
    # Fan width bin
    fw = state.get('fan_width')
    out['fan_bin'] = (
        'compressed' if fw is not None and fw < 1.5 else
        'normal'     if fw is not None and fw < 3.0 else
        'wide'       if fw is not None and fw < 5.0 else
        'extreme'    if fw is not None else 'NA'
    )
    # Slope_15m direction
    s15m = state.get('slope_15m_5m')
    out['slope_15m_sign'] = (
        'down' if s15m is not None and s15m < -0.1 else
        'up'   if s15m is not None and s15m > 0.1  else
        'flat' if s15m is not None else 'NA'
    )
    return out


# ── Run pipeline ────────────────────────────────────────────────────────────

def run(t_start: float, t_end: float, run_name: str, tf: str = '5s',
          peak_window_min: float = 15, min_sep_min: float = 10):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tf_seconds = TF_SECONDS.get(tf)
    if tf_seconds is None:
        raise ValueError(f'Unsupported --tf {tf!r}; choose from {list(TF_SECONDS)}')
    fwd_bars    = int(FORWARD_MINUTES * 60 / tf_seconds)
    peak_window = max(1, int(peak_window_min * 60 / tf_seconds))
    min_sep     = max(1, int(min_sep_min  * 60 / tf_seconds))
    print(f'\n=== Oracle regret analysis — {run_name} (tf={tf}) ===')
    print(f'Range: {datetime.fromtimestamp(t_start, tz=timezone.utc)} -> '
              f'{datetime.fromtimestamp(t_end, tz=timezone.utc)}')
    print(f'  tf={tf} ({tf_seconds}s/bar)  fwd={FORWARD_MINUTES}min={fwd_bars}bars  '
          f'peak_window=±{peak_window_min}min={peak_window}bars  '
          f'min_sep={min_sep_min}min={min_sep}bars')

    df = load_tf_bars(t_start, t_end, tf)
    if df.empty:
        print('No data'); return
    ts    = df['timestamp'].values.astype(np.int64)
    close = df['close'].values.astype(float)
    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)
    print(f'Loaded {len(df)} {tf} bars')

    # Segment into Globex sessions (halt-to-halt) — empirical gap detection,
    # DST-proof (no hardcoded UTC offset).
    session_id, session_end_idx, session_first_ts, session_date, n_sessions = \
        build_session_info(ts)
    print(f'  Segmented into {n_sessions} sessions (halt-to-halt, '
          f'gap > {SESSION_GAP_S // 60}min)')
    bar_range = high - low
    volume = (df['volume'].values.astype(float) if 'volume' in df.columns
              else np.full(len(df), np.nan))

    print('Computing CRMs + anchors...')
    M_15s, S_15s = compute_anchor('15s', ts, t_start, t_end, window=20, column='close')
    M_1m,  S_1m  = compute_anchor('1m',  ts, t_start, t_end, window=15, column='close')
    M_15m, S_15m = compute_anchor('15m', ts, t_start, t_end, window=12, column='close')
    Mh,    Sh    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='high')
    Ml,    Sl    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='low')
    Mc,    Sc    = compute_anchor('1h',  ts, t_start, t_end, window=12, column='close')

    # Oracle entry detection on RAW PRICE (per user 2026-05-14): peaks on the
    # bar HIGH, troughs on the bar LOW. The opportunity is defined by price
    # truth; the CRMs/anchors are the analysis layer (the state vector).
    print(f'Detecting raw-price local extrema '
          f'(window=±{peak_window} bars, min_sep={min_sep} bars)...')
    peaks,   _ = detect_local_extrema(high, session_id,
                                      window=peak_window, min_sep=min_sep)
    _, troughs = detect_local_extrema(low,  session_id,
                                      window=peak_window, min_sep=min_sep)
    print(f'  price peaks: {len(peaks)}    price troughs: {len(troughs)}')

    # Build oracle entries
    def _build_entry(idx: int, direction: str) -> dict:
        idx = int(idx)
        # Entry at the raw-price extreme: bar HIGH for a SHORT peak, bar LOW
        # for a LONG trough — the truest oracle entry.
        entry_price = float(high[idx]) if direction == 'SHORT' else float(low[idx])
        mfe, ttm_bars, _exit_idx, exit_px, avail = forward_mfe(
            close, high, low, idx, entry_price, direction, fwd_bars,
            session_end_idx[idx])
        state = extract_state_vector(idx, close, M_15s, S_15s, M_1m, S_1m,
                                                    M_15m, S_15m, Mh, Sh, Ml, Sl, Mc, Sc,
                                                    tf_seconds=tf_seconds)
        disc = discretize_state(state, direction)
        mfe_dollars = round(mfe * TICK_DOLLAR, 2)
        ttm_min   = ttm_bars * tf_seconds / 60.0
        avail_min = avail * tf_seconds / 60.0
        vol = volume[idx]
        return {
            'oracle_idx': idx, 'oracle_ts': int(ts[idx]),
            'oracle_utc': datetime.fromtimestamp(ts[idx], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': int(session_id[idx]),
            'session_date': session_date[idx],
            'tod_minutes': int((ts[idx] - session_first_ts[idx]) // 60),
            'direction': direction, 'entry_price': round(entry_price, 2),
            'mfe_ticks': round(mfe, 1), 'mfe_dollars': mfe_dollars,
            'time_to_mfe_min': round(ttm_min, 2), 'exit_price': round(float(exit_px), 2),
            'mfe_velocity': round(mfe_dollars / max(ttm_min, tf_seconds / 60.0), 3),
            'full_window': int(avail >= fwd_bars),
            'available_fwd_min': round(avail_min, 1),
            'volume': None if np.isnan(vol) else round(float(vol), 1),
            'bar_range': round(float(bar_range[idx]), 2),
            **state, **{f'd_{k}': v for k, v in disc.items()},
        }

    entries = [_build_entry(idx, 'SHORT') for idx in peaks]
    entries += [_build_entry(idx, 'LONG') for idx in troughs]
    entries.sort(key=lambda e: e['oracle_idx'])
    print(f'Total oracle entries: {len(entries)}')
    n_full = sum(e['full_window'] for e in entries)
    print(f'  full 60-min window: {n_full}   '
          f'truncated near halt: {len(entries) - n_full}')

    # Save oracle entries
    ent_path = OUT_DIR / f'oracle_entries_{run_name}.csv'
    if entries:
        cols = list(entries[0].keys())
        with open(ent_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for e in entries:
                w.writerow(e)
        print(f'Wrote oracle entries: {ent_path}')

    # Compute random-bar state distribution as baseline
    print('\nBuilding random-bar baseline state distribution...')
    random_states = []
    sample_idxs = np.random.RandomState(42).choice(
        np.arange(peak_window, len(ts) - fwd_bars), size=min(5000, len(ts) // 5),
        replace=False)
    for idx in sample_idxs:
        state = extract_state_vector(int(idx), close, M_15s, S_15s, M_1m, S_1m,
                                                    M_15m, S_15m, Mh, Sh, Ml, Sl, Mc, Sc,
                                                    tf_seconds=tf_seconds)
        # Random direction (50/50) — to measure baseline without direction bias
        for direction in ('SHORT', 'LONG'):
            disc = discretize_state(state, direction)
            random_states.append(disc)

    # Build P-tables: P(state_bin | random)  vs  P(state_bin | oracle)
    from collections import Counter
    print('\nComputing empirical probabilities + lift...')
    state_keys = ['stack', 'z_15m_bin', 'rail_position', 'fan_bin', 'slope_15m_sign']

    lift_rows = []
    for key in state_keys:
        # Distinct values
        oracle_counts = Counter(e[f'd_{key}'] for e in entries)
        random_counts = Counter(r[key] for r in random_states)
        n_oracle = len(entries)
        n_random = len(random_states)
        all_vals = set(oracle_counts) | set(random_counts)
        for val in sorted(all_vals):
            p_o = oracle_counts.get(val, 0) / max(1, n_oracle)
            p_r = random_counts.get(val, 0) / max(1, n_random)
            lift = (p_o / p_r) if p_r > 0 else float('inf')
            lift_rows.append({
                'feature': key, 'value': val,
                'n_oracle': oracle_counts.get(val, 0),
                'n_random': random_counts.get(val, 0),
                'p_oracle': round(p_o, 4),
                'p_random': round(p_r, 4),
                'lift':     round(lift, 2) if lift != float('inf') else 'inf',
            })

    lift_path = OUT_DIR / f'lift_table_{run_name}.csv'
    with open(lift_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(lift_rows[0].keys()))
        w.writeheader()
        for r in lift_rows:
            w.writerow(r)
    print(f'Wrote lift table: {lift_path}')

    # Top lift summary
    print(f'\n=== Top features by lift (P_oracle / P_random) ===')
    sorted_rows = sorted([r for r in lift_rows if r['lift'] != 'inf' and r['n_oracle'] >= 20],
                                  key=lambda r: r['lift'], reverse=True)
    for r in sorted_rows[:15]:
        print(f"  {r['feature']:<18} {r['value']:<14}  P_o={r['p_oracle']:.3f}  "
                  f"P_r={r['p_random']:.3f}  lift={r['lift']:.2f}  "
                  f"(n_oracle={r['n_oracle']})")

    # MFE distribution
    mfe_dollars = np.array([e['mfe_dollars'] for e in entries])
    print(f'\n=== Oracle MFE distribution (60-min forward) ===')
    print(f'  n entries:   {len(entries)}')
    print(f'  mean MFE:    ${mfe_dollars.mean():.0f}')
    print(f'  median MFE:  ${np.median(mfe_dollars):.0f}')
    print(f'  q25:         ${np.percentile(mfe_dollars, 25):.0f}')
    print(f'  q75:         ${np.percentile(mfe_dollars, 75):.0f}')
    print(f'  total $:     ${mfe_dollars.sum():.0f}')

    return entries, lift_rows


def _ts(d):
    return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', help='YYYY-MM-DD')
    ap.add_argument('--end',   help='YYYY-MM-DD')
    ap.add_argument('--date',  help='Single day YYYY-MM-DD')
    ap.add_argument('--tf', default='5s', choices=list(TF_SECONDS),
                       help='Base bar timeframe for the oracle (default 5s)')
    ap.add_argument('--peak-window-min', type=float, default=15,
                       help='±N MINUTES centered window for raw-price local extrema')
    ap.add_argument('--min-sep-min', type=float, default=10,
                       help='Min separation between consecutive extrema, in MINUTES')
    ap.add_argument('--name', help='Run name (auto)')
    args = ap.parse_args()

    if args.date:
        t_start = _ts(args.date)
        t_end = t_start + 86400
        name = args.name or args.date
    elif args.start and args.end:
        t_start = _ts(args.start)
        t_end = _ts(args.end) + 86400
        name = args.name or f'{args.start}_{args.end}'
    else:
        ap.error('Provide --date OR --start+--end')

    run(t_start, t_end, name, tf=args.tf,
         peak_window_min=args.peak_window_min,
         min_sep_min=args.min_sep_min)


if __name__ == '__main__':
    main()
