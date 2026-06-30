#!/usr/bin/env python
"""
Cusp Marker — peak-style single-click marker with multi-scale anchor overlays.

Single click = mark a cusp at the snapped bar extreme. Direction auto-detected
from surrounding price action (rising → SHORT peak; falling → LONG trough).
Forward MFE/MAE measured over a configurable forward window (default 60 min)
from the marked bar at 1s resolution.

Overlays:
    15s CRM  (teal)    — tight tactical anchor
    1m  CRM  (blue)    — short-horizon trade anchor
    15m CRM  (purple)  — medium context
    1h M_high + envelope (green ± 2σ, +2σ to +3σ rally trigger)
    1h M_low  + envelope (red   ± 2σ, −2σ to −3σ crash trigger)

Each pick captures: timestamp, snapped price (H or L), auto-direction,
forward MFE/MAE, AND a snapshot of every anchor's M / S / z at the entry bar.

Usage:
    python tools/cusp_marker.py --date 2025-06-06
    python tools/cusp_marker.py --date 2025-06-06:2025-06-13   # date range
    python tools/cusp_marker.py --date 2025-06-06 --days 5 --tf 1m
    python tools/cusp_marker.py --date 2025-06-06 --fwd-mins 30

Keys:
    Click           drop a pick (snap to bar H or L, dir auto)
    Click on pick   remove that pick
    L / S           flip direction of last pick
    D               delete last pick
    1 / 2 / 3 / 4   toggle 15s / 1m / 15m / 1h HL overlays
    5               toggle loaded trades overlay (from --load-trades)
    P               screenshot → examples/ (no dialog)
    Q               save + quit
    Arrows          pan (left/right), zoom (up/down)

Load trades for visual inspection:
    python tools/cusp_marker.py --date 2025-09-08 --load-trades reports/findings/decay_sim/v6_v6_OOS_trades.csv
    python tools/cusp_marker.py --date 2026-02-12 --load-trades training_iso_v2/output/oos_FADE_CALM.pkl

Visualize regret/oracle entries (where the oracle WANTS to trade):
    python tools/cusp_marker.py --date 2025-07-08 --days 1 --load-trades reports/findings/regret_oracle/oracle_entries_jul2025.csv
    python tools/cusp_marker.py --date 2025-07-01 --days 31 --load-trades reports/findings/regret_oracle/oracle_entries_jul2025.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from archive.tools.research.data import load_atlas_tf

plt.switch_backend('TkAgg')

TICK_SIZE = 0.25
TICK_VALUE = 0.50
PICKS_DIR = 'DATA/cusp_picks'
SETTINGS_PATH = 'DATA/cusp_picks/marker_settings.json'
# Screenshots (P key + toolbar Save button) auto-direct here
EXAMPLES_DIR = str(Path(__file__).resolve().parent.parent / 'examples')
matplotlib.rcParams['savefig.directory'] = EXAMPLES_DIR


# ── Trade-loading from CSV / JSON / pickle ─────────────────────────────────

def load_trades(path: str) -> list:
    """Load trades from CSV / JSON / pickle. Returns list of dicts with:
      side, entry_ts, exit_ts, entry_price, exit_price, pnl_dollars, label.
    Tolerates schema variations; missing fields → None."""
    import pickle, csv
    if not path or not os.path.exists(path):
        print(f'  [load_trades] path not found: {path}')
        return []
    ext = Path(path).suffix.lower()
    rows = []
    try:
        if ext == '.csv':
            with open(path) as f:
                r = csv.DictReader(f)
                rows = list(r)
        elif ext == '.json':
            with open(path) as f:
                data = json.load(f)
            rows = data.get('picks', data.get('trades', data if isinstance(data, list) else []))
        elif ext in ('.pkl', '.pickle'):
            with open(path, 'rb') as f:
                rows = pickle.load(f)
        else:
            print(f'  [load_trades] unknown extension: {ext}')
            return []
    except Exception as e:
        print(f'  [load_trades] error reading {path}: {e}')
        return []

    out = []
    for r in rows:
        if not isinstance(r, dict):
            # Try to convert pickle objects with attribute access
            r = {k: getattr(r, k, None) for k in
                     ('side', 'entry_ts', 'exit_ts', 'entry_price', 'exit_price',
                       'pnl_dollars', 'realized_pnl_dollars', 'direction',
                       'entry_utc', 'exit_utc', 'change_dollars', 'mfe_dollars')}
        # Side / direction
        side = r.get('side') or r.get('direction') or 'LONG'
        side = str(side).upper()
        # Timestamps — accept ts (seconds) OR utc string
        def _to_ts(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            if isinstance(v, (int, float)):
                return float(v)
            # Numeric string = epoch seconds (e.g. CSV oracle_ts column)
            try:
                fv = float(v)
                if fv > 1e8:    # plausible epoch-seconds (≈ year 1973+)
                    return fv
            except (TypeError, ValueError):
                pass
            try:
                return datetime.strptime(str(v), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                pass
            try:
                return pd.Timestamp(v).timestamp()
            except Exception:
                return None
        # Accept also single-bar pick format (timestamp, price + forward MFE/MAE)
        # and the regret/oracle schema (oracle_ts/oracle_utc + time_to_mfe_min).
        ets = _to_ts(r.get('entry_ts') or r.get('entry_utc')
                          or r.get('ts_start') or r.get('timestamp')
                          or r.get('oracle_ts') or r.get('oracle_utc'))
        xts = _to_ts(r.get('exit_ts') or r.get('exit_utc') or r.get('ts_end'))
        # Oracle/regret rows have no exit_ts — derive it from forward-MFE timing
        if xts is None and ets is not None:
            ttm = r.get('time_to_mfe_min') or r.get('time_to_mfe_mins')
            if ttm is not None:
                try:
                    xts = ets + float(ttm) * 60.0
                except (TypeError, ValueError):
                    pass
        epx = r.get('entry_price') or r.get('entry_px') or r.get('price')
        xpx = r.get('exit_price') or r.get('exit_px')
        # PnL: prefer explicit dollar fields; else compute from entry/exit + direction
        pnl = (r.get('pnl_dollars') or r.get('realized_pnl_dollars')
                  or r.get('change_dollars'))
        if pnl is None and epx is not None and xpx is not None:
            sign = +1 if side == 'LONG' else -1
            pnl = sign * (float(xpx) - float(epx)) / TICK_SIZE * TICK_VALUE
        if pnl is None:
            # Last resort: a `pnl` field (may be ticks for iso ClosedTrade — convert)
            raw = r.get('pnl') or r.get('mfe_dollars')
            if raw is not None:
                # If iso ClosedTrade (no _dollars suffix anywhere), assume ticks
                pnl = float(raw) * TICK_VALUE if r.get('bars_held') is not None else float(raw)
            else:
                pnl = 0.0
        label = r.get('reason') or r.get('exit_reason') or r.get('tier') or ''
        if ets is None or epx is None:
            continue
        try:
            out.append({
                'side': side, 'entry_ts': float(ets),
                'exit_ts': float(xts) if xts is not None else None,
                'entry_price': float(epx),
                'exit_price': float(xpx) if xpx is not None else None,
                'pnl_dollars': float(pnl) if pnl is not None else 0.0,
                'label': str(label),
            })
        except (TypeError, ValueError):
            continue
    print(f'  [load_trades] {path}: {len(out)} trades loaded')
    return out


def load_settings() -> dict:
    """Load persisted window geometry + overlay visibility + defaults."""
    if not os.path.exists(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d: dict):
    """Persist settings to disk (best-effort; ignore errors)."""
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception as e:
        print(f'  [warn] could not save settings: {e}')


# ── 1s loader (daily files, e.g. DATA/ATLAS/1s/2025_06_06.parquet) ──────────

def load_1s_window(ts_start: float, ts_end: float, cache: dict = None) -> pd.DataFrame:
    """Load 1s bars across [ts_start, ts_end], reading daily parquets. Caches
    daily frames in `cache` (caller-provided dict) to avoid re-reads."""
    if cache is None:
        cache = {}
    dt_start = datetime.utcfromtimestamp(ts_start).replace(tzinfo=timezone.utc)
    dt_end = datetime.utcfromtimestamp(ts_end).replace(tzinfo=timezone.utc)
    frames = []
    cur = datetime(dt_start.year, dt_start.month, dt_start.day, tzinfo=timezone.utc)
    end_cap = datetime(dt_end.year, dt_end.month, dt_end.day, tzinfo=timezone.utc)
    while cur <= end_cap:
        key = cur.strftime('%Y_%m_%d')
        if key not in cache:
            path = f'DATA/ATLAS/1s/{key}.parquet'
            cache[key] = pd.read_parquet(path) if os.path.exists(path) else None
        df = cache[key]
        if df is not None:
            mask = (df['timestamp'] >= ts_start) & (df['timestamp'] <= ts_end)
            frames.append(df[mask])
        cur = cur + pd.Timedelta(days=1).to_pytimedelta()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values('timestamp').reset_index(drop=True)


# ── Anchor (rolling regression of OHLCV at higher TFs) ──────────────────────

def _load_tf_ohlcv_global(tf: str, t_start: float, t_end: float) -> pd.DataFrame:
    period_s_map = {'15s': 15, '30s': 30, '1m': 60, '5m': 300,
                       '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400}
    if tf not in period_s_map:
        return pd.DataFrame()
    PRE_ROLL_S = 86400
    start_dt = datetime.fromtimestamp(t_start - PRE_ROLL_S, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(t_end, tz=timezone.utc)

    daily_dfs = []
    cur = datetime(start_dt.year, start_dt.month, start_dt.day, tzinfo=timezone.utc)
    while cur < end_dt + pd.Timedelta(days=1).to_pytimedelta():
        day_str = cur.strftime('%Y_%m_%d')
        path = f'DATA/ATLAS/{tf}/{day_str}.parquet'
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.copy()
                df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
            daily_dfs.append(df)
        cur = cur + pd.Timedelta(days=1).to_pytimedelta()

    if not daily_dfs:
        return pd.DataFrame()
    df = pd.concat(daily_dfs, ignore_index=True)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def compute_anchor(tf: str, target_ts: np.ndarray, t_start: float, t_end: float,
                       window: int, column: str = 'close'):
    period_s_map = {'15s': 15, '30s': 30, '1m': 60, '5m': 300,
                       '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400}
    if tf not in period_s_map:
        return None, None
    period_s = period_s_map[tf]
    oh = _load_tf_ohlcv_global(tf, t_start, t_end)
    if oh.empty:
        return None, None
    M = oh[column].rolling(window, min_periods=2).mean().values
    S = oh[column].rolling(window, min_periods=2).std().values
    tf_ts = oh['timestamp'].values.astype(np.int64)
    target = target_ts.astype(np.int64) - period_s
    idx = np.searchsorted(tf_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(tf_ts) - 1)
    return M[idx], S[idx]


# ── Marker UI (peak-style) ──────────────────────────────────────────────────

import pickle
from tools.viz.core.feature_utils import compute_primitive_arrays
from tools.viz.core.cubic_utils import find_raw_turns

class CuspMarker:
    """Peak-style single-click marker. Snaps to bar H or L, auto-detects
    direction from local trend, captures anchor snapshot, measures forward
    MFE/MAE over a fixed horizon at 1s resolution."""

    def __init__(self, df, date_str, tf='1m', anchors=None, fwd_mins=60,
                  loaded_trades=None, cubic_n=20):
        self.df = df
        self.date_str = date_str
        self.tf = tf
        self.anchors = anchors or {}
        self.fwd_mins = fwd_mins
        self.cubic_n = cubic_n
        self._cache_1s = {}
        # Loaded trades from --load-trades (read-only overlay, distinct from user picks)
        self.loaded_trades = loaded_trades or []
        self._loaded_artists = []

        self.close = df['close'].values.astype(float)
        self.high = df['high'].values.astype(float)
        self.low = df['low'].values.astype(float)
        self.timestamps = df['timestamp'].values.astype(float)
        self.dt_stamps = [datetime.fromtimestamp(t, tz=timezone.utc) for t in self.timestamps]

        # Load candidate filter classifier
        self._classifier_model = None
        self._classifier_scaler = None
        self._classifier_features = None
        self._load_classifier()
        
        # Pre-compute cubic candidates
        self.cubic_turns = []
        self.cubic_curve = None
        self.cubic_features = None
        self._compute_cubic_candidates()

        self.picks = []
        self._pick_artists = []   # one per pick: (scatter, label) tuple
        # Auto-load existing picks for this date (persistence across launches)
        self._autoload_existing_picks()

        # Load persisted overlay visibility (default: all on)
        self._settings = load_settings()
        self._last_geometry = None
        ov = self._settings.get('overlays', {})
        self.show_15s = ov.get('15s', True)
        self.show_1m = ov.get('1m', True)
        self.show_15m = ov.get('15m', True)
        self.show_1h_hl = ov.get('1h_hl', True)
        self.show_loaded = ov.get('loaded', True)
        self.show_cubic = ov.get('cubic', True)
        self._overlay_artists = {'15s': [], '1m': [], '15m': [], '1h_hl': [], 'cubic': []}

        # 2-click state machine
        self._pending_entry_x = None
        self._pending_entry_y = None
        self._pending_vline = None

    # ── Candidate Classifier & Cubic Generation ────────────────────────────

    def _load_classifier(self):
        path = 'DATA/cusp_picks/model.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self._classifier_model = data['model']
                self._classifier_scaler = data.get('scaler')
                self._classifier_features = data['features']
            print(f"  [load_classifier] restored candidate filter (trained on {len(self._classifier_features)} features)")
        else:
            print(f"  [load_classifier] no model found at {path}, candidates will not be scored.")

    def _compute_cubic_candidates(self):
        print(f"  [cubic] computing candidates for {self.date_str} (N={self.cubic_n})...")
        turns, price_smooth, slope, curv = find_raw_turns(self.close, self.cubic_n)
        self.cubic_curve = price_smooth
        self.cubic_turns = turns
        
        # Only compute primitive arrays if we have a model to score them
        if self._classifier_model:
            self.cubic_features = compute_primitive_arrays(self.close, self.anchors)
        else:
            self.cubic_features = None

    # ── Auto-load existing picks for this date ─────────────────────────────

    def _autoload_existing_picks(self):
        """Load previously-saved picks for this date_str + TF so the user's
        marks persist across re-launches. Remaps bar_index by timestamp
        (current chart may have different range than save-time)."""
        date_key = self.date_str.split()[0].replace(':', '_to_')
        path = os.path.join(PICKS_DIR, f'picks_{date_key}_multi.json')
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f'  [autoload] failed to read {path}: {e}')
            return
        prior = [p for p in data.get('picks', [])
                     if p.get('timeframe', '1m') == self.tf]
        if not prior:
            return

        # Map each prior pick's timestamp to current chart's bar_index
        ts_array = self.timestamps.astype(np.int64)
        ts_lo, ts_hi = int(ts_array[0]), int(ts_array[-1])
        loaded = []
        for p in prior:
            pts = int(p.get('timestamp', 0))
            if pts < ts_lo or pts > ts_hi:
                continue
            new_idx = int(np.searchsorted(ts_array, pts))
            new_idx = max(0, min(new_idx, len(ts_array) - 1))
            # If exact match not found, snap to nearest
            if ts_array[new_idx] != pts:
                # Try nearest neighbor
                if new_idx > 0 and abs(ts_array[new_idx - 1] - pts) < abs(ts_array[new_idx] - pts):
                    new_idx = new_idx - 1
            p['bar_index'] = new_idx
            loaded.append(p)
        self.picks = loaded
        print(f'  [autoload] restored {len(loaded)} prior picks from {path}')

    # ── Forward MFE/MAE at 1s resolution ───────────────────────────────────

    def _measure_forward(self, ts_pick, direction):
        """Measure MFE/MAE from ts_pick forward by fwd_mins minutes at 1s."""
        ts_end = ts_pick + self.fwd_mins * 60.0
        df_1s = load_1s_window(ts_pick, ts_end, self._cache_1s)
        if len(df_1s) < 5:
            return 0.0, 0.0, 0.0, 0.0
        p = df_1s['close'].values.astype(float)
        ts = df_1s['timestamp'].values.astype(float)
        entry = p[0]
        if direction == 'LONG':
            fav = (p - entry) / TICK_SIZE
            adv = (entry - p) / TICK_SIZE
        else:   # SHORT
            fav = (entry - p) / TICK_SIZE
            adv = (p - entry) / TICK_SIZE
        mfe_idx = int(np.argmax(fav))
        mfe = float(fav[mfe_idx])
        mae = float(np.max(adv[:mfe_idx + 1])) if mfe_idx > 0 else 0.0
        time_to_mfe = float(ts[mfe_idx] - ts[0]) / 60.0
        end_pnl = float(fav[-1])
        return mfe, mae, time_to_mfe, end_pnl

    # ── Direction auto-detect: STRUCTURAL PRIORITY (per user 2026-05-11) ──
    # The previous version used local trend (rising→SHORT, falling→LONG),
    # which gave OPPOSITE direction from what structural position demanded.
    # User example: P1 was at z_1h_high=+0.65 (above rail) — local trend said
    # LONG, but rubber-band wanted SHORT. Flipping made it a winner.
    #
    # New rule (priority order):
    #   1. STRUCTURAL: if at/past 1h rail → fade direction
    #   2. MEDIUM EXTENSION: if z_15m strongly ±, fade direction
    #   3. FAST BIAS: if z_15s clearly ±, bounce direction
    #   4. LOCAL TREND: fallback to peak/trough cusp detection

    def _detect_direction(self, idx, lookback=5, lookahead=5):
        # PRIORITY 1: structural extension at 1h HL
        a = self.anchors
        if 'Mh_1h' in a and 'Sh_1h' in a:
            try:
                mh, sh = float(a['Mh_1h'][idx]), float(a['Sh_1h'][idx])
                ml, sl = float(a['Ml_1h'][idx]), float(a['Sl_1h'][idx])
                if sh > 0 and not np.isnan(mh):
                    z_1h_hi = (self.close[idx] - mh) / sh
                    if z_1h_hi >= +0.3:
                        return 'SHORT'   # above upper rail → rubber band down
                if sl > 0 and not np.isnan(ml):
                    z_1h_lo = (self.close[idx] - ml) / sl
                    if z_1h_lo <= -0.3:
                        return 'LONG'    # below lower rail → rubber band up
            except (IndexError, TypeError, ValueError):
                pass

        # PRIORITY 2: 15m extension
        if 'M_15m' in a and 'S_15m' in a:
            try:
                m, s = float(a['M_15m'][idx]), float(a['S_15m'][idx])
                if s > 0 and not np.isnan(m):
                    z_15m = (self.close[idx] - m) / s
                    if z_15m >= +1.5:
                        return 'SHORT'
                    if z_15m <= -1.5:
                        return 'LONG'
            except (IndexError, TypeError, ValueError):
                pass

        # PRIORITY 3: fast extension (1m, 15s) — small bias toward bounce
        if 'M_1m' in a and 'S_1m' in a:
            try:
                m, s = float(a['M_1m'][idx]), float(a['S_1m'][idx])
                if s > 0 and not np.isnan(m):
                    z_1m = (self.close[idx] - m) / s
                    if z_1m >= +1.5:
                        return 'SHORT'
                    if z_1m <= -1.5:
                        return 'LONG'
            except (IndexError, TypeError, ValueError):
                pass

        # PRIORITY 4: local trend cusp (peak / trough) — original behavior
        start = max(0, idx - lookback)
        end = min(len(self.close) - 1, idx + lookahead)
        before = self.close[start:idx]
        after = self.close[idx:end + 1]
        if len(before) < 2 or len(after) < 2:
            return 'UNKNOWN'
        tb = before[-1] - before[0]
        ta = after[-1] - after[0]
        if tb > 0 and ta < 0:
            return 'SHORT'
        if tb < 0 and ta > 0:
            return 'LONG'
        if tb > 0:
            return 'SHORT'
        if tb < 0:
            return 'LONG'
        return 'UNKNOWN'

    # ── Anchor snapshot at a bar ───────────────────────────────────────────

    def _snapshot_anchors(self, idx):
        snap = {}
        a = self.anchors
        c = self.close[idx]
        if 'M_15s' in a:
            m, s = float(a['M_15s'][idx]), float(a['S_15s'][idx])
            snap['M_15s'] = round(m, 3)
            snap['S_15s'] = round(s, 4)
            snap['z_15s_crm'] = round((c - m) / s, 3) if s > 0 else None
        if 'M_1m' in a:
            m, s = float(a['M_1m'][idx]), float(a['S_1m'][idx])
            snap['M_1m'] = round(m, 3)
            snap['S_1m'] = round(s, 4)
            snap['z_1m_crm'] = round((c - m) / s, 3) if s > 0 else None
        if 'M_15m' in a:
            m, s = float(a['M_15m'][idx]), float(a['S_15m'][idx])
            snap['M_15m'] = round(m, 3)
            snap['S_15m'] = round(s, 4)
            snap['z_15m_crm'] = round((c - m) / s, 3) if s > 0 else None
        if 'Mh_1h' in a:
            mh, sh = float(a['Mh_1h'][idx]), float(a['Sh_1h'][idx])
            snap['Mh_1h'] = round(mh, 3)
            snap['Sh_1h'] = round(sh, 4)
            snap['z_1h_high'] = round((c - mh) / sh, 3) if sh > 0 else None
        if 'Ml_1h' in a:
            ml, sl = float(a['Ml_1h'][idx]), float(a['Sl_1h'][idx])
            snap['Ml_1h'] = round(ml, 3)
            snap['Sl_1h'] = round(sl, 4)
            snap['z_1h_low'] = round((c - ml) / sl, 3) if sl > 0 else None
        if 'Mc_1h' in a:
            mc, sc = float(a['Mc_1h'][idx]), float(a['Sc_1h'][idx])
            snap['Mc_1h'] = round(mc, 3)
            snap['Sc_1h'] = round(sc, 4)
            snap['z_1h_close'] = round((c - mc) / sc, 3) if sc > 0 else None
        return snap

    # ── Drawing a pick ─────────────────────────────────────────────────────

    def _draw_pick(self, pick, idx):
        color = '#00C853' if pick['direction'] == 'LONG' else '#FF1744'
        marker = '^' if pick['direction'] == 'LONG' else 'v'
        start_dt = self.dt_stamps[pick['bar_index']]
        
        sc = self.ax.scatter([start_dt], [pick['price']],
                                 color=color, s=200, zorder=10, marker=marker,
                                 edgecolors='black', linewidth=1.5)
        
        # Highlight the forward measurement window (fwd_mins)
        fwd_mins = pick.get('fwd_mins', self.fwd_mins)
        end_dt = start_dt + pd.Timedelta(minutes=fwd_mins)
        span = self.ax.axvspan(start_dt, end_dt, color=color, alpha=0.1, zorder=1)

        end_pnl = pick.get('end_pnl_ticks', 0.0)
        label = self.ax.text(
            start_dt, pick['price'] + (3 if pick['direction'] == 'LONG' else -3),
            f"P{idx+1} {pick['direction'][0]}\nMFE:+{pick['mfe_ticks']:.0f} MAE:-{pick['mae_ticks']:.0f} End:{end_pnl:+.0f}",
            fontsize=8, ha='center',
            va='bottom' if pick['direction'] == 'LONG' else 'top',
            fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.85, edgecolor=color))
                          
        self._pick_artists.append((sc, span, label))

    def _redraw_all_picks(self):
        for artists in self._pick_artists:
            for art in artists:
                art.remove()
        self._pick_artists = []
        for i, p in enumerate(self.picks):
            self._draw_pick(p, i)
        self.fig.canvas.draw_idle()

    # ── Title bar ──────────────────────────────────────────────────────────

    def _update_title(self):
        n = len(self.picks)
        n_long = sum(1 for p in self.picks if p['direction'] == 'LONG')
        n_short = n - n_long
        total_mfe = sum(p['mfe_dollars'] for p in self.picks)
        total_mae = sum(p['mae_dollars'] for p in self.picks)
        loaded_str = (f" Loaded={'On' if self.show_loaded else 'Off'}({len(self.loaded_trades)})"
                            if self.loaded_trades else "")
        overlays = (f"15s={'On' if self.show_15s else 'Off'} "
                         f"1m={'On' if self.show_1m else 'Off'} "
                         f"15m={'On' if self.show_15m else 'Off'} "
                         f"1hHL={'On' if self.show_1h_hl else 'Off'} "
                         f"Cubic={'On' if self.show_cubic else 'Off'}{loaded_str}")
        self.ax.set_title(
            f"{self.date_str} | {self.tf} | {len(self.df)} bars   ({overlays})\n"
            f"Picks: {n} (L:{n_long} S:{n_short}) | Forward MFE: ${total_mfe:.0f} | MAE: ${total_mae:.0f}\n"
            f"Click=mark+snap, drag=range, click-on-pick=rm, L/S=flip, D=del, 1/2/3/C=toggle, Q=save+quit, fwd={self.fwd_mins}m",
            fontsize=10, fontweight='bold'
        )
        self.fig.canvas.draw_idle()

    # ── Event handlers ─────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        toolbar = self.fig.canvas.toolbar
        if toolbar and hasattr(toolbar, 'mode') and toolbar.mode:
            return  # toolbar zoom/pan active

        import matplotlib.dates as mdates
        if event.xdata is None:
            return
        bar_nums = mdates.date2num(self.dt_stamps)
        idx = int(np.argmin(np.abs(bar_nums - event.xdata)))

        # State 0: No pending entry
        if self._pending_entry_x is None:
            # Click on existing pick → remove
            for pi, existing in enumerate(self.picks):
                if existing['bar_index'] == idx:
                    self.picks.pop(pi)
                    artists = self._pick_artists.pop(pi)
                    for art in artists:
                        art.remove()
                    print(f"  Removed P{pi+1} @ {existing['time_utc']}")
                    self._redraw_all_picks()
                    self._update_title()
                    return
            
            # Start new pick
            self._pending_entry_x = event.xdata
            self._pending_entry_y = event.ydata
            self._pending_vline = self.ax.axvline(self.dt_stamps[idx], color='gray', linestyle='--', zorder=5)
            self.fig.canvas.draw_idle()
            return

        # State 1: We have a pending entry, so this click is the exit!
        idx_start = int(np.argmin(np.abs(bar_nums - self._pending_entry_x)))
        idx_end = idx
        
        # clear pending state
        if self._pending_vline:
            self._pending_vline.remove()
            self._pending_vline = None
            
        if idx_start > idx_end:
            idx_start, idx_end = idx_end, idx_start
            
        ts_start = float(self.timestamps[idx_start])
        ts_end = float(self.timestamps[idx_end])
        fwd_mins = (ts_end - ts_start) / 60.0

        idx = idx_start
        click_y = self._pending_entry_y if self._pending_entry_y is not None else self.close[idx]
        self._pending_entry_x = None
        self._pending_entry_y = None
        
        if abs(click_y - self.high[idx]) < abs(click_y - self.low[idx]):
            snap_price = float(self.high[idx])
            snap_label = 'H'
        else:
            snap_price = float(self.low[idx])
            snap_label = 'L'

        direction = self._detect_direction(idx)
        if direction == 'UNKNOWN':
            # Fall back: snap-H → SHORT, snap-L → LONG
            direction = 'SHORT' if snap_label == 'H' else 'LONG'

        ts_pick = float(self.timestamps[idx])
        old_fwd = self.fwd_mins
        self.fwd_mins = fwd_mins
        mfe, mae, ttm, end_pnl = self._measure_forward(ts_pick, direction)
        self.fwd_mins = old_fwd

        # Provenance tracking against cubic candidates
        provenance = 'human_fresh'
        if self.cubic_turns:
            for cand in self.cubic_turns:
                c_idx = cand['index']
                if abs(c_idx - idx) <= 3:
                    cand_dir = 'SHORT' if cand['type'] == 'top' else 'LONG'
                    if cand_dir == direction:
                        provenance = 'auto_accepted'
                    else:
                        provenance = 'auto_corrected'
                    break

        pick = {
            'pick_id': len(self.picks),
            'timeframe': self.tf,
            'bar_index': idx,
            'timestamp': ts_pick,
            'time_utc': self.dt_stamps[idx].strftime('%Y-%m-%d %H:%M:%S'),
            'direction': direction,
            'price': round(snap_price, 2),
            'snap': snap_label,
            'close': round(float(self.close[idx]), 2),
            'high': round(float(self.high[idx]), 2),
            'low': round(float(self.low[idx]), 2),
            'fwd_mins': fwd_mins,
            'mfe_ticks': round(mfe, 1),
            'mae_ticks': round(mae, 1),
            'end_pnl_ticks': round(end_pnl, 1),
            'mfe_dollars': round(mfe * TICK_VALUE, 2),
            'mae_dollars': round(mae * TICK_VALUE, 2),
            'time_to_mfe_mins': round(ttm, 1),
            'provenance': provenance,
            'anchors': self._snapshot_anchors(idx),
        }
        self.picks.append(pick)
        self._draw_pick(pick, len(self.picks) - 1)
        rr = mfe / mae if mae > 0 else float('inf')
        a = pick['anchors']
        print(f"  P{pick['pick_id']+1} @ {pick['time_utc']} {direction} {snap_label} "
                  f"{snap_price:.2f}  MFE:${mfe*TICK_VALUE:.0f}/MAE:${mae*TICK_VALUE:.0f}/"
                  f"RR:1:{rr:.1f}  z15s={a.get('z_15s_crm','?')} z1m={a.get('z_1m_crm','?')} "
                  f"z15m={a.get('z_15m_crm','?')} z1h+={a.get('z_1h_high','?')} "
                  f"z1h-={a.get('z_1h_low','?')}")
        self._update_title()

    def _toggle(self, layer):
        if layer == '15s':
            self.show_15s = not self.show_15s
            for art in self._overlay_artists['15s']:
                art.set_visible(self.show_15s)
        elif layer == '1m':
            self.show_1m = not self.show_1m
            for art in self._overlay_artists['1m']:
                art.set_visible(self.show_1m)
        elif layer == '15m':
            self.show_15m = not self.show_15m
            for art in self._overlay_artists['15m']:
                art.set_visible(self.show_15m)
        elif layer == '1h_hl':
            self.show_1h_hl = not self.show_1h_hl
            for art in self._overlay_artists['1h_hl']:
                art.set_visible(self.show_1h_hl)
        elif layer == 'cubic':
            self.show_cubic = not self.show_cubic
            for art in self._overlay_artists['cubic']:
                art.set_visible(self.show_cubic)
        self._update_title()

    def _flip_last(self):
        if not self.picks:
            return
        p = self.picks[-1]
        p['direction'] = 'SHORT' if p['direction'] == 'LONG' else 'LONG'
        
        old_fwd = self.fwd_mins
        self.fwd_mins = p.get('fwd_mins', self.fwd_mins)
        mfe, mae, ttm, end_pnl = self._measure_forward(p['timestamp'], p['direction'])
        self.fwd_mins = old_fwd
        
        p['mfe_ticks'] = round(mfe, 1)
        p['mae_ticks'] = round(mae, 1)
        p['end_pnl_ticks'] = round(end_pnl, 1)
        p['mfe_dollars'] = round(mfe * TICK_VALUE, 2)
        p['mae_dollars'] = round(mae * TICK_VALUE, 2)
        p['time_to_mfe_mins'] = round(ttm, 1)
        self._redraw_all_picks()
        self._update_title()
        print(f"  P{p['pick_id']+1}: flipped to {p['direction']}")

    def _delete_last(self):
        if not self.picks:
            return
        p = self.picks.pop()
        artists = self._pick_artists.pop()
        for art in artists:
            art.remove()
        self._redraw_all_picks()
        self._update_title()
        print(f"  Deleted P{p['pick_id']+1}")

    def _on_key(self, event):
        if event.key == '1':
            self._toggle('15s')
        elif event.key == '2':
            self._toggle('1m')
        elif event.key == '3':
            self._toggle('15m')
        elif event.key == '4':
            self._toggle('1h_hl')
        elif event.key == '5':
            self._toggle_loaded()
        elif event.key in ('c', 'C'):
            self._toggle('cubic')
        elif event.key in ('l', 'L', 's', 'S'):
            self._flip_last()
        elif event.key in ('d', 'D'):
            self._delete_last()
        elif event.key in ('p', 'P'):
            self._screenshot()
        elif event.key == 'left':
            self._pan(-1)
        elif event.key == 'right':
            self._pan(1)
        elif event.key == 'up':
            self._zoom(0.5)
        elif event.key == 'down':
            self._zoom(2.0)
        elif event.key in ('q', 'Q'):
            self._save()
            self._persist_settings()
            plt.close(self.fig)

    # ── Settings persistence (window geometry + overlay flags) ────────────

    def _capture_settings(self) -> dict:
        """Snapshot current window geometry + overlay flags for next launch."""
        out = {
            'overlays': {
                '15s': self.show_15s, '1m': self.show_1m,
                '15m': self.show_15m, '1h_hl': self.show_1h_hl,
                'loaded': self.show_loaded,
                'cubic': self.show_cubic,
            },
            'fwd_mins': self.fwd_mins,
            'tf': self.tf,
        }
        # Window geometry (TkAgg-specific; safe-fallback on other backends)
        try:
            backend = plt.get_backend().lower()
            mgr = self.fig.canvas.manager
            geom = None
            if 'tk' in backend:
                geom = mgr.window.geometry()
            elif 'qt' in backend:
                geom = mgr.window.geometry().getRect()
            if geom:
                out['window_geometry'] = geom
                self._last_geometry = geom
        except Exception:
            if self._last_geometry:
                out['window_geometry'] = self._last_geometry
        return out

    def _persist_settings(self):
        save_settings(self._capture_settings())

    def _apply_window_geometry(self):
        """Restore window position+size from last session if persisted."""
        geom = self._settings.get('window_geometry')
        if not geom or geom == '?':
            return
        try:
            backend = plt.get_backend().lower()
            mgr = self.fig.canvas.manager
            if 'tk' in backend and isinstance(geom, str):
                mgr.window.geometry(geom)
            elif 'qt' in backend and isinstance(geom, (list, tuple)) and len(geom) == 4:
                mgr.window.setGeometry(*geom)
        except Exception as e:
            print(f"  [warn] failed to restore window geometry: {e}")

    def _apply_overlay_visibility(self):
        """After overlays are drawn, hide any whose persisted flag is off."""
        for art in self._overlay_artists['15s']:
            art.set_visible(self.show_15s)
        for art in self._overlay_artists['1m']:
            art.set_visible(self.show_1m)
        for art in self._overlay_artists['15m']:
            art.set_visible(self.show_15m)
        for art in self._overlay_artists['1h_hl']:
            art.set_visible(self.show_1h_hl)
        for art in self._overlay_artists['cubic']:
            art.set_visible(self.show_cubic)

    # ── Save ───────────────────────────────────────────────────────────────

    def _auto_correct_orientations(self):
        """Automatically correct orientation of losing picks before saving."""
        if not self.picks:
            return 0
        flipped_count = 0
        for p in self.picks:
            if p.get('end_pnl_ticks', 0) < 0:
                p['direction'] = 'SHORT' if p['direction'] == 'LONG' else 'LONG'
                
                old_fwd = self.fwd_mins
                self.fwd_mins = p.get('fwd_mins', self.fwd_mins)
                mfe, mae, ttm, end_pnl = self._measure_forward(p['timestamp'], p['direction'])
                self.fwd_mins = old_fwd
                
                p['mfe_ticks'] = round(mfe, 1)
                p['mae_ticks'] = round(mae, 1)
                p['end_pnl_ticks'] = round(end_pnl, 1)
                p['mfe_dollars'] = round(mfe * TICK_VALUE, 2)
                p['mae_dollars'] = round(mae * TICK_VALUE, 2)
                p['time_to_mfe_mins'] = round(ttm, 1)
                flipped_count += 1
                
        if flipped_count > 0:
            print(f"\n  [auto-correct] Flipped {flipped_count} incorrect orientations to positive PnL.")
            self._redraw_all_picks()
            self._update_title()
            
        return flipped_count

    def _save(self):
        if not self.picks:
            print("\n  No picks marked. Nothing saved.")
            return
            
        self._auto_correct_orientations()

        os.makedirs(PICKS_DIR, exist_ok=True)
        date_key = self.date_str.split()[0].replace(':', '_to_')
        canonical_path = os.path.join(PICKS_DIR, f'picks_{date_key}_multi.json')
        existing = []
        existing_tfs = set()
        if os.path.exists(canonical_path):
            with open(canonical_path) as f:
                data = json.load(f)
            existing = data.get('picks', [])
            existing_tfs = set(data.get('marked_timeframes', []))
            existing = [s for s in existing if s.get('timeframe', '1m') != self.tf]
            print(f"  Merging: {len(existing)} prior picks from {existing_tfs}")
        all_picks = existing + self.picks
        all_tfs = existing_tfs | {self.tf}
        for i, p in enumerate(all_picks):
            p['pick_id'] = i

        ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(canonical_path, 'w') as f:
            json.dump({'date_range': date_key,
                          'marked_timeframes': sorted(all_tfs),
                          'created': ts_tag, 'n_picks': len(all_picks),
                          'fwd_mins': self.fwd_mins,
                          'cubic_n': self.cubic_n,
                          'picks': all_picks}, f, indent=2)
        snap_path = os.path.join(PICKS_DIR, f'picks_{date_key}_{self.tf}_{ts_tag}.json')
        with open(snap_path, 'w') as f:
            json.dump({'date_range': date_key, 'timeframe': self.tf,
                          'created': ts_tag, 'fwd_mins': self.fwd_mins,
                          'n_picks': len(self.picks),
                          'picks': self.picks}, f, indent=2)
        print(f"\n  Saved {len(self.picks)} picks @ {self.tf} → {canonical_path}")
        print(f"  Snapshot:                                 {snap_path}")

    # ── Screenshot (auto-direct to examples/) ─────────────────────────────

    def _screenshot(self):
        """Save the current figure straight to examples/ — no file dialog."""
        os.makedirs(EXAMPLES_DIR, exist_ok=True)
        date_key = self.date_str.split()[0].replace(':', '_to_')
        ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(EXAMPLES_DIR, f'cusp_{date_key}_{self.tf}_{ts_tag}.png')
        self.fig.savefig(path, dpi=130, bbox_inches='tight')
        print(f'  [screenshot] saved -> {path}')

    # ── Pan / zoom ─────────────────────────────────────────────────────────

    def _pan(self, direction):
        import matplotlib.dates as mdates
        xlim = self.ax.get_xlim()
        window = xlim[1] - xlim[0]
        shift = window * 0.5 * direction
        x_min = mdates.date2num(self.dt_stamps[0])
        x_max = mdates.date2num(self.dt_stamps[-1])
        new_left = xlim[0] + shift
        new_right = xlim[1] + shift
        if new_left < x_min:
            new_left = x_min
            new_right = x_min + window
        if new_right > x_max:
            new_right = x_max
            new_left = max(x_min, x_max - window)
        self.ax.set_xlim(new_left, new_right)
        self._autofit_y()
        self._train_and_update_classifier()
        self.fig.canvas.draw_idle()

    def _zoom(self, factor):
        import matplotlib.dates as mdates
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        half_w = (xlim[1] - xlim[0]) / 2 * factor
        x_min = mdates.date2num(self.dt_stamps[0])
        x_max = mdates.date2num(self.dt_stamps[-1])
        new_left = max(x_min, center - half_w)
        new_right = min(x_max, center + half_w)
        self.ax.set_xlim(new_left, new_right)
        self._autofit_y()
        self.fig.canvas.draw_idle()

    def _autofit_y(self):
        import matplotlib.dates as mdates
        xlim = self.ax.get_xlim()
        bar_nums = mdates.date2num(self.dt_stamps)
        mask = (bar_nums >= xlim[0]) & (bar_nums <= xlim[1])
        if not mask.any():
            return
        vis_low = float(self.low[mask].min())
        vis_high = float(self.high[mask].max())
        # Include 1h HL envelopes if shown
        a = self.anchors
        if self.show_1h_hl and 'Mh_1h' in a:
            band_hi = (a['Mh_1h'] + 3 * a['Sh_1h'])[mask]
            band_lo = (a['Ml_1h'] - 3 * a['Sl_1h'])[mask]
            if np.any(~np.isnan(band_hi)):
                vis_high = max(vis_high, float(np.nanmax(band_hi)))
            if np.any(~np.isnan(band_lo)):
                vis_low = min(vis_low, float(np.nanmin(band_lo)))
        pad = (vis_high - vis_low) * 0.05
        self.ax.set_ylim(vis_low - pad, vis_high + pad)

    # ── Overlays ───────────────────────────────────────────────────────────

    def _draw_loaded_trades(self):
        """Render trades from --load-trades as a read-only overlay.
        Distinct style: square markers, entry→exit line, PnL color-coded."""
        if not self.loaded_trades:
            return
        ts0 = float(self.timestamps[0])
        ts1 = float(self.timestamps[-1])
        for t in self.loaded_trades:
            if t['entry_ts'] < ts0 or t['entry_ts'] > ts1:
                continue
            # Find chart-time positions
            ets = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc)
            xts = (datetime.fromtimestamp(t['exit_ts'], tz=timezone.utc)
                       if t['exit_ts'] else None)
            pnl = t['pnl_dollars']
            color = '#2962FF' if pnl >= 0 else '#D32F2F'
            marker_e = 's'   # square = loaded
            # Entry marker
            sc = self.ax.scatter([ets], [t['entry_price']], color=color,
                                          marker=marker_e, s=120, zorder=8,
                                          edgecolors='black', linewidth=1.2, alpha=0.85)
            self._loaded_artists.append(sc)
            # Exit marker + line
            if xts and t['exit_price'] is not None:
                sc2 = self.ax.scatter([xts], [t['exit_price']], color='black',
                                                marker='X', s=80, zorder=8, alpha=0.85)
                self._loaded_artists.append(sc2)
                ln = self.ax.plot([ets, xts], [t['entry_price'], t['exit_price']],
                                            color=color, lw=1.0, ls=':', alpha=0.55, zorder=7)[0]
                self._loaded_artists.append(ln)
            # Small label
            side_char = t['side'][0] if t['side'] else '?'
            label = self.ax.text(ets, t['entry_price'],
                                          f'  {side_char}${pnl:+.0f}',
                                          fontsize=7, color=color, alpha=0.85,
                                          va='center', ha='left', zorder=9)
            self._loaded_artists.append(label)

        # Apply persisted visibility
        for art in self._loaded_artists:
            art.set_visible(self.show_loaded)

    def _toggle_loaded(self):
        self.show_loaded = not self.show_loaded
        for art in self._loaded_artists:
            art.set_visible(self.show_loaded)
        self._update_title()
        self.fig.canvas.draw_idle()

    def _draw_overlays(self):
        a = self.anchors
        if 'M_15s' in a:
            ln, = self.ax.plot(self.dt_stamps, a['M_15s'], color='#00897B',
                                  lw=0.8, ls='-', alpha=0.75, label='15s CRM')
            self._overlay_artists['15s'].append(ln)
        if 'M_1m' in a:
            ln, = self.ax.plot(self.dt_stamps, a['M_1m'], color='#1565C0',
                                  lw=1.0, ls='-', alpha=0.85, label='1m CRM')
            self._overlay_artists['1m'].append(ln)
        if 'M_15m' in a:
            ln, = self.ax.plot(self.dt_stamps, a['M_15m'], color='#5E35B1',
                                  lw=1.4, ls='-', alpha=0.85, label='15m CRM')
            self._overlay_artists['15m'].append(ln)
        if 'Mh_1h' in a:
            mh, sh = a['Mh_1h'], a['Sh_1h']
            ml, sl = a['Ml_1h'], a['Sl_1h']
            # FAR-SIDE-ONLY rendering (per user 2026-05-11): show extension
            # zones outside the rails, not the inside-band fills.
            for art in [
                self.ax.plot(self.dt_stamps, mh, color='#43A047', lw=1.2,
                                  ls='-', alpha=0.85, label='1h M_high')[0],
                self.ax.plot(self.dt_stamps, ml, color='#E53935', lw=1.2,
                                  ls='-', alpha=0.85, label='1h M_low')[0],
                # FAR side of M_high: above the rail (rally extension)
                self.ax.fill_between(self.dt_stamps, mh, mh + 2*sh,
                                            color='#43A047', alpha=0.05,
                                            label='1h M_high → +2σ'),
                self.ax.fill_between(self.dt_stamps, mh + 2*sh, mh + 3*sh,
                                            color='#43A047', alpha=0.13,
                                            label='1h +2σ to +3σ (rally trig)'),
                # FAR side of M_low: below the rail (crash extension)
                self.ax.fill_between(self.dt_stamps, ml - 2*sl, ml,
                                            color='#E53935', alpha=0.05,
                                            label='1h M_low → −2σ'),
                self.ax.fill_between(self.dt_stamps, ml - 3*sl, ml - 2*sl,
                                            color='#E53935', alpha=0.13,
                                            label='1h −3σ to −2σ (crash trig)'),
                # Outer-edge dashed lines (3σ extension cap)
                self.ax.plot(self.dt_stamps, mh + 3*sh, color='#1B5E20',
                                  lw=0.7, ls='--', alpha=0.6)[0],
                self.ax.plot(self.dt_stamps, ml - 3*sl, color='#B71C1C',
                                  lw=0.7, ls='--', alpha=0.6)[0],
            ]:
                self._overlay_artists['1h_hl'].append(art)
                
        self._draw_cubic_overlay()

    def _redraw_cubic_overlay(self):
        for art in self._overlay_artists.get('cubic', []):
            art.remove()
        self._overlay_artists['cubic'] = []
        self._draw_cubic_overlay()
        for art in self._overlay_artists['cubic']:
            art.set_visible(self.show_cubic)
        
    def _train_and_update_classifier(self):
        if not self.picks or self.cubic_features is None or not self.cubic_turns:
            return
            
        import numpy as np
        labels = np.zeros(len(self.cubic_turns))
        pick_indices = [p['bar_index'] for p in self.picks]
        pick_dirs = [p['direction'] for p in self.picks]
        
        for i, cand in enumerate(self.cubic_turns):
            cand_idx = cand['index']
            cand_type = cand['type']
            
            best_dist = 9999
            for p_idx, p_dir in zip(pick_indices, pick_dirs):
                # match type: top matches SHORT, bottom matches LONG
                if cand_type == 'top' and p_dir != 'SHORT': continue
                if cand_type == 'bottom' and p_dir != 'LONG': continue
                
                dist = abs(cand_idx - p_idx)
                if dist < best_dist:
                    best_dist = dist
                    
            # A candidate is considered a "positive" class if it's within 3 bars of a human pick
            if best_dist <= 3:
                labels[i] = 1
                
        feature_names = list(self.cubic_features.keys())
        X = []
        for cand in self.cubic_turns:
            idx = cand['index']
            row = [self.cubic_features[f][idx] for f in feature_names]
            X.append(row)
            
        X = np.array(X)
        y = np.array(labels)
        
        if len(np.unique(y)) < 2:
            return
            
        from sklearn.ensemble import RandomForestClassifier
        self._classifier_model = RandomForestClassifier(n_estimators=30, max_depth=3, class_weight='balanced', random_state=42)
        np.nan_to_num(X, copy=False)
        self._classifier_model.fit(X, y)
        self._classifier_features = feature_names
        self._classifier_scaler = None
        
        self._redraw_cubic_overlay()

    def _draw_cubic_overlay(self):
        """Render the centered cubic regression overlay and candidate markers."""
        if self.cubic_curve is None:
            return

        # The orange smoothed price line
        ln, = self.ax.plot(self.dt_stamps, self.cubic_curve, color='#FF9800',
                              lw=1.5, ls='-', alpha=0.9, label=f'Cubic N={self.cubic_n}')
        self._overlay_artists['cubic'].append(ln)

        # Draw candidate markers
        for cand in self.cubic_turns:
            idx = cand['index']
            cand_type = cand['type']
            price = self.cubic_curve[idx]
            
            color = '#D32F2F' if cand_type == 'top' else '#2E7D32'
            marker = 'v' if cand_type == 'top' else '^'
            prob = 1.0 # default to fully visible if no model
            
            if self._classifier_model and self.cubic_features is not None:
                try:
                    feat_array = [self.cubic_features.get(f, np.zeros_like(self.close))[idx] for f in self._classifier_features]
                    if np.any(np.isnan(feat_array)):
                        prob = 0.0
                    else:
                        X = np.array(feat_array).reshape(1, -1)
                        if self._classifier_scaler:
                            X = self._classifier_scaler.transform(X)
                        prob = self._classifier_model.predict_proba(X)[0, 1]
                except Exception as e:
                    prob = 0.0

            if prob > 0.5:
                size = 180
                alpha = 1.0
                zorder = 7
            else:
                continue

            sc = self.ax.scatter([self.dt_stamps[idx]], [price],
                                     color=color, marker=marker, s=size, 
                                     alpha=alpha, zorder=zorder, edgecolors='black', linewidth=0.8)
            self._overlay_artists['cubic'].append(sc)

    def run(self):
        import matplotlib.dates as mdates
        from matplotlib.widgets import Cursor, Button

        self.fig, self.ax = plt.subplots(1, 1, figsize=(22, 9))
        # Extra bottom space for two rows of touch-friendly buttons
        self.fig.subplots_adjust(bottom=0.18)
        self.ax.set_facecolor('#FAFAFA')

        # Price (close + H/L band)
        self.ax.plot(self.dt_stamps, self.close, color='#212121',
                        linewidth=1.0, alpha=0.9, label='Close')
        self.ax.fill_between(self.dt_stamps, self.low, self.high,
                                alpha=0.06, color='#212121')

        self._draw_overlays()
        self._draw_loaded_trades()
        # Apply persisted overlay visibility AFTER artists exist
        self._apply_overlay_visibility()
        # Render autoloaded picks (from disk via _autoload_existing_picks)
        for i, p in enumerate(self.picks):
            self._draw_pick(p, i)

        self.ax.set_ylabel('Price', fontsize=11)
        self.ax.grid(True, alpha=0.2)
        self.ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.85)

        # Day-divider lines
        seen = set()
        for i, dt_s in enumerate(self.dt_stamps):
            day_str = dt_s.strftime('%Y-%m-%d')
            if day_str not in seen:
                seen.add(day_str)
                if i > 0:
                    self.ax.axvline(x=dt_s, color='#FF9800', linewidth=1.2,
                                          linestyle='-', alpha=0.4)

        # Initial zoom: if picks were autoloaded, span them all + 1h padding.
        # Otherwise default to first 4 hours of the day.
        n_bars = len(self.dt_stamps)
        bars_per_hour = {'15s': 240, '30s': 120, '1m': 60, '5m': 12,
                              '15m': 4, '30m': 2, '1h': 1}.get(self.tf, 60)
        if self.picks:
            pick_idxs = [p['bar_index'] for p in self.picks]
            pad = bars_per_hour     # 1 hour of padding each side
            left_idx = max(0, min(pick_idxs) - pad)
            right_idx = min(n_bars - 1, max(pick_idxs) + pad)
            # Ensure window is at least 4 hours wide for context
            min_width = bars_per_hour * 4
            if right_idx - left_idx < min_width:
                center = (left_idx + right_idx) // 2
                left_idx = max(0, center - min_width // 2)
                right_idx = min(n_bars - 1, center + min_width // 2)
            self.ax.set_xlim(mdates.date2num(self.dt_stamps[left_idx]),
                                  mdates.date2num(self.dt_stamps[right_idx]))
        else:
            zoom_end = min(bars_per_hour * 4, n_bars - 1)
            self.ax.set_xlim(mdates.date2num(self.dt_stamps[0]),
                                  mdates.date2num(self.dt_stamps[zoom_end]))
        self._autofit_y()

        self.ax.format_coord = lambda x, y: f'Price: {y:.2f}'
        self.cursor = Cursor(self.ax, useblit=True, color='gray',
                                  linewidth=0.5, linestyle='--')

        # ── Bottom buttons (touch-friendly: large, two rows) ─────────────
        btn_color = '#E0E0E0'
        btn_overlay = '#C8E6C9'    # green tint for overlay toggles
        btn_action = '#FFE0B2'     # orange tint for destructive/save actions

        # Row 1 (lower) — navigation: pan, zoom, fit
        # 5 buttons across width: each ~0.16 wide, height 0.05
        ax_left  = self.fig.add_axes([0.06, 0.02, 0.16, 0.05])
        ax_right = self.fig.add_axes([0.23, 0.02, 0.16, 0.05])
        ax_zin   = self.fig.add_axes([0.41, 0.02, 0.12, 0.05])
        ax_zout  = self.fig.add_axes([0.54, 0.02, 0.12, 0.05])
        ax_fit   = self.fig.add_axes([0.67, 0.02, 0.12, 0.05])
        ax_save  = self.fig.add_axes([0.80, 0.02, 0.14, 0.05])

        self.btn_left  = Button(ax_left,  '<<  Pan',     color=btn_color)
        self.btn_right = Button(ax_right, 'Pan  >>',     color=btn_color)
        self.btn_zin   = Button(ax_zin,   '+ Zoom',      color=btn_color)
        self.btn_zout  = Button(ax_zout,  '- Zoom',      color=btn_color)
        self.btn_fit   = Button(ax_fit,   'Fit All',     color=btn_color)
        self.btn_save  = Button(ax_save,  'Save & Quit', color=btn_action)

        self.btn_left.on_clicked( lambda e: self._pan(-1))
        self.btn_right.on_clicked(lambda e: self._pan(1))
        self.btn_zin.on_clicked(  lambda e: self._zoom(0.5))
        self.btn_zout.on_clicked( lambda e: self._zoom(2.0))
        self.btn_fit.on_clicked(  lambda e: (
            self.ax.set_xlim(mdates.date2num(self.dt_stamps[0]),
                                  mdates.date2num(self.dt_stamps[-1])),
            self._autofit_y(),
            self.fig.canvas.draw_idle()
        ))
        self.btn_save.on_clicked( lambda e: (self._save(), self._persist_settings(), plt.close(self.fig)))

        # Row 2 (upper) — overlay toggles + edit picks
        # 6 buttons across width: 4 toggles + flip-last + del-last
        ax_t15s  = self.fig.add_axes([0.06, 0.09, 0.10, 0.045])
        ax_t1m   = self.fig.add_axes([0.17, 0.09, 0.10, 0.045])
        ax_t15m  = self.fig.add_axes([0.28, 0.09, 0.10, 0.045])
        ax_t1h   = self.fig.add_axes([0.39, 0.09, 0.12, 0.045])
        ax_flip  = self.fig.add_axes([0.55, 0.09, 0.14, 0.045])
        ax_del   = self.fig.add_axes([0.70, 0.09, 0.14, 0.045])

        self.btn_t15s = Button(ax_t15s, '15s CRM',     color=btn_overlay)
        self.btn_t1m  = Button(ax_t1m,  '1m CRM',      color=btn_overlay)
        self.btn_t15m = Button(ax_t15m, '15m CRM',     color=btn_overlay)
        self.btn_t1h  = Button(ax_t1h,  '1h HL bands', color=btn_overlay)
        self.btn_flip = Button(ax_flip, 'Flip Last',   color=btn_action)
        self.btn_del  = Button(ax_del,  'Del Last',    color=btn_action)

        self.btn_t15s.on_clicked(lambda e: self._toggle('15s'))
        self.btn_t1m.on_clicked( lambda e: self._toggle('1m'))
        self.btn_t15m.on_clicked(lambda e: self._toggle('15m'))
        self.btn_t1h.on_clicked( lambda e: self._toggle('1h_hl'))
        self.btn_flip.on_clicked(lambda e: self._flip_last())
        self.btn_del.on_clicked( lambda e: self._delete_last())

        self._update_title()

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        # Persist settings AND save picks on any close path (X button, Q, Save & Quit)
        self.fig.canvas.mpl_connect('close_event', lambda e: (self._save(), self._persist_settings()))

        print(f"\n  Cusp Marker ready -- {len(self.df)} bars loaded, fwd window {self.fwd_mins}m")
        print(f"  Click = mark + snap to H/L; auto-direction from local trend")
        print(f"  L/S=flip last, D=del last, 1/2/3/4=toggle (15s/1m/15m/1h), P=screenshot, Q=save+quit")
        print(f"  Arrow keys / bottom buttons: pan left/right + zoom in/out\n")
        if self._settings:
            saved_geom = self._settings.get('window_geometry', '?')
            saved_ov = self._settings.get('overlays', {})
            print(f"  [settings restored] geom={saved_geom}  overlays={saved_ov}\n")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout(rect=[0, 0.16, 1, 1])

        # Apply restored geometry AFTER show() has created the window
        # We use a short timer so that matplotlib's default sizing doesn't override it.
        backend = plt.get_backend().lower()
        if 'tk' in backend:
            self.fig.canvas.manager.window.after(100, self._apply_window_geometry)
        else:
            geom_timer = self.fig.canvas.new_timer(interval=100)
            geom_timer.single_shot = True
            geom_timer.add_callback(self._apply_window_geometry)
            geom_timer.start()

        # Continuously poll geometry so it's captured before the window is destroyed
        # by an OS 'X' close event.
        def _poll_geometry():
            try:
                mgr = self.fig.canvas.manager
                if 'tk' in backend:
                    self._last_geometry = mgr.window.geometry()
                elif 'qt' in backend:
                    self._last_geometry = mgr.window.geometry().getRect()
            except Exception:
                pass
                
        self._poll_timer = self.fig.canvas.new_timer(interval=1000)
        self._poll_timer.add_callback(_poll_geometry)
        self._poll_timer.start()

        plt.show(block=True)


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_date_range(s: str) -> tuple[str, str]:
    """Accepts 'YYYY-MM-DD' or 'YYYY-MM-DD:YYYY-MM-DD'."""
    if ':' in s:
        a, b = s.split(':', 1)
        return a.strip(), b.strip()
    return s, s


def main():
    parser = argparse.ArgumentParser(
        description='Cusp Marker (peak-style + multi-anchor overlays)')
    parser.add_argument('--data-dir', default='DATA/ATLAS')
    parser.add_argument('--date', required=True,
                                help='YYYY-MM-DD  or  YYYY-MM-DD:YYYY-MM-DD (range)')
    parser.add_argument('--days', type=int, default=5,
                                help='If --date is single, load N consecutive calendar days '
                                          '(default 5 for continuous panning)')
    parser.add_argument('--tf', default='1m')
    parser.add_argument('--fwd-mins', type=float, default=60,
                                help='Forward MFE/MAE measurement window in minutes (default 60)')
    parser.add_argument('--burn-hours', type=float, default=0)
    parser.add_argument('--load-trades', type=str, default=None,
                                help='Load trades from CSV / JSON / pickle and overlay '
                                          'on chart (read-only). e.g. v6 sim output, iso pickle.')
    parser.add_argument('--cubic-n', type=int, default=20,
                                help='Cubic regression window size for candidate overlay')
    args = parser.parse_args()

    import pytz
    est = pytz.timezone('US/Eastern')

    start_str, end_str = _parse_date_range(args.date)
    
    start_dt_cal = datetime.strptime(start_str, '%Y-%m-%d')
    end_dt_cal = datetime.strptime(end_str, '%Y-%m-%d')
    
    if start_str == end_str and args.days > 1:
        end_dt_cal = start_dt_cal + pd.Timedelta(days=args.days - 1).to_pytimedelta()

    # The trading session for a given trade date starts at 18:00 EST on the PREVIOUS day,
    # and ends at 17:00 EST on the CURRENT day. This captures the Globex open exactly after the maintenance window.
    session_start = est.localize(datetime(
        start_dt_cal.year, start_dt_cal.month, start_dt_cal.day, 18, 0, 0
    )) - pd.Timedelta(days=1)
    
    session_end = est.localize(datetime(
        end_dt_cal.year, end_dt_cal.month, end_dt_cal.day, 17, 0, 0
    ))

    t_start = session_start.timestamp()
    t_end = session_end.timestamp()

    date_label = (f"{start_str}" if start_str == end_str
                       else f"{start_str}:{end_str}")
    print(f"Cusp Marker -- {date_label} ({args.tf})")

    print("Loading data...")
    df = load_atlas_tf(args.data_dir, args.tf)
    if df.empty:
        print(f"ERROR: No {args.tf} data")
        sys.exit(1)
    mask = (df['timestamp'] >= t_start) & (df['timestamp'] < t_end)
    df_day = df[mask].reset_index(drop=True)

    if args.burn_hours > 0:
        burn_until = t_start + args.burn_hours * 3600
        burn_mask = df_day['timestamp'] >= burn_until
        n_burned = (~burn_mask).sum()
        df_day = df_day[burn_mask].reset_index(drop=True)
        print(f"  Burned first {args.burn_hours}h ({n_burned} bars skipped)")

    if df_day.empty:
        print(f"ERROR: No data for {date_label}")
        sys.exit(1)

    ts_first = datetime.fromtimestamp(float(df_day['timestamp'].iloc[0]), tz=timezone.utc)
    ts_last = datetime.fromtimestamp(float(df_day['timestamp'].iloc[-1]), tz=timezone.utc)
    span_hours = (float(df_day['timestamp'].iloc[-1]) - float(df_day['timestamp'].iloc[0])) / 3600
    print(f"  {len(df_day)} bars | {ts_first.strftime('%a %b %d %H:%M')} -> "
              f"{ts_last.strftime('%a %b %d %H:%M')} UTC ({span_hours:.1f}h)")

    print("Computing anchors (15s CRM, 1m CRM, 15m CRM, 1h HL RM)...")
    target_ts = df_day['timestamp'].values.astype(np.int64)
    # 15s window=20 → 5-min lookback (tight tactical anchor)
    # 1m  window=15 → 15-min lookback (matches L3_1m N_BASE=15)
    # 15m window=12 → 3-hr lookback (medium context)
    # 1h  window=12 → 12-hr lookback (slow HL envelope)
    M_15s, S_15s = compute_anchor('15s', target_ts, t_start, t_end, window=20, column='close')
    M_1m,  S_1m  = compute_anchor('1m',  target_ts, t_start, t_end, window=15, column='close')
    M_15m, S_15m = compute_anchor('15m', target_ts, t_start, t_end, window=12, column='close')
    Mh_1h, Sh_1h = compute_anchor('1h',  target_ts, t_start, t_end, window=12, column='high')
    Ml_1h, Sl_1h = compute_anchor('1h',  target_ts, t_start, t_end, window=12, column='low')
    Mc_1h, Sc_1h = compute_anchor('1h',  target_ts, t_start, t_end, window=12, column='close')

    anchors = {}
    if M_15s is not None:
        anchors['M_15s'], anchors['S_15s'] = M_15s, S_15s
    if M_1m is not None:
        anchors['M_1m'], anchors['S_1m'] = M_1m, S_1m
    if M_15m is not None:
        anchors['M_15m'], anchors['S_15m'] = M_15m, S_15m
    if Mh_1h is not None:
        anchors['Mh_1h'], anchors['Sh_1h'] = Mh_1h, Sh_1h
    if Ml_1h is not None:
        anchors['Ml_1h'], anchors['Sl_1h'] = Ml_1h, Sl_1h
    if Mc_1h is not None:
        anchors['Mc_1h'], anchors['Sc_1h'] = Mc_1h, Sc_1h
    print(f"  Anchors loaded: {list(anchors.keys())}")

    # Load trades to overlay (optional)
    loaded_trades = []
    if args.load_trades:
        print(f'Loading trades: {args.load_trades}')
        loaded_trades = load_trades(args.load_trades)
        # Filter to current chart's time window
        if loaded_trades:
            ts0 = float(df_day['timestamp'].iloc[0])
            ts1 = float(df_day['timestamp'].iloc[-1])
            in_window = [t for t in loaded_trades
                              if ts0 <= t['entry_ts'] <= ts1]
            print(f'  {len(in_window)}/{len(loaded_trades)} trades in chart window')
            loaded_trades = in_window

    marker = CuspMarker(df_day, date_label, tf=args.tf, anchors=anchors,
                              fwd_mins=args.fwd_mins, loaded_trades=loaded_trades,
                              cubic_n=args.cubic_n)
    marker.run()


if __name__ == '__main__':
    main()
