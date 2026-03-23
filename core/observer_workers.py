"""Observer Workers — trend/peak/session monitors that LOG only, no gate influence.

Three worker sets that observe the market at different TF scales:
  1. TrendObserver (1m): detects swing direction using 1m worker state
  2. PeakObserver (15s): tracks peak detections, alignment with trend
  3. SessionObserver (1h): session regime (rally/selloff/chop/reversal)

All workers:
  - update() every bar with the current TBN worker states
  - log to CSV: reports/observers/{worker}_log.csv
  - recommend() returns what they WOULD do (no actual blocking)
  - checkpoint() saves state periodically (monthly/weekly)

Integration:
  - AdvanceEngine calls observer_hub.update(bar_index, timestamp, state, tbn)
  - AdvanceEngine calls observer_hub.on_peak(bar_index, timestamp, direction, entered)
  - AdvanceEngine calls observer_hub.on_trade_exit(bar_index, timestamp, trade_result)
  - Observer hub writes CSV per worker + combined event log

Usage:
    hub = ObserverHub(output_dir='reports/observers')
    hub.update(bar_index, timestamp, state, belief_network)
    hub.on_peak(bar_index, timestamp, 'LONG', entered=True)
    hub.checkpoint('2025_03')  # save state
"""

import csv
import os
import pickle
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ── Trend Observer (1m) ───────────────────────────────────────────────

@dataclass
class TrendState:
    direction: str = 'FLAT'          # LONG, SHORT, FLAT
    strength: float = 0.0            # 0-1 confidence
    duration_bars: int = 0           # bars in current direction
    fm_1m: float = 0.0               # 1m F_momentum
    vol_1m: float = 0.0              # 1m volume_delta
    dmi_diff_1m: float = 0.0         # 1m DMI+ - DMI-
    z_1m: float = 0.0                # 1m z_score
    adx_1m: float = 0.0              # 1m ADX


class TrendObserver:
    """Detects 1m swing direction from TBN worker state."""

    def __init__(self):
        self.state = TrendState()
        self._prev_dir = 'FLAT'
        self._dir_bars = 0
        self._fm_buffer = deque(maxlen=10)  # 10 bars of 1m fm for slope
        self._vol_buffer = deque(maxlen=10)

    def update(self, tbn) -> TrendState:
        """Read 1m worker state and classify trend direction."""
        w_1m = tbn.workers.get(60)
        if w_1m is None:
            return self.state

        idx = w_1m._last_tf_bar_idx
        if idx < 0 or idx >= len(w_1m._states):
            return self.state

        raw = w_1m._states[idx]
        ms = raw['state'] if isinstance(raw, dict) and 'state' in raw else raw

        fm = getattr(ms, 'F_momentum', 0.0)
        vol = getattr(ms, 'volume_delta', 0.0)
        dmi_p = getattr(ms, 'dmi_plus', 0.0)
        dmi_m = getattr(ms, 'dmi_minus', 0.0)
        dmi_diff = dmi_p - dmi_m
        z = getattr(ms, 'z_score', 0.0)
        adx = getattr(ms, 'adx_strength', 0.0)

        self._fm_buffer.append(fm)
        self._vol_buffer.append(vol)

        # Direction from momentum + volume agreement
        fm_sign = np.sign(fm) if abs(fm) > 1.0 else 0
        vol_sign = np.sign(vol) if abs(vol) > 5.0 else 0

        if fm_sign > 0 and vol_sign > 0:
            direction = 'LONG'
        elif fm_sign < 0 and vol_sign < 0:
            direction = 'SHORT'
        elif abs(dmi_diff) > 10:
            direction = 'LONG' if dmi_diff > 0 else 'SHORT'
        else:
            direction = 'FLAT'

        # Strength: combine |fm|, |vol|, |dmi_diff|, adx
        strength = min(1.0, (abs(fm) / 50 + abs(vol) / 100 + abs(dmi_diff) / 30 + adx / 40) / 4)

        # Track duration
        if direction == self._prev_dir:
            self._dir_bars += 1
        else:
            self._dir_bars = 1
            self._prev_dir = direction

        self.state = TrendState(
            direction=direction,
            strength=strength,
            duration_bars=self._dir_bars,
            fm_1m=fm,
            vol_1m=vol,
            dmi_diff_1m=dmi_diff,
            z_1m=z,
            adx_1m=adx,
        )
        return self.state

    def recommend(self, peak_direction: str) -> tuple:
        """What would the trend worker recommend for this peak?
        Returns (approve: bool, confidence: float, reason: str).
        """
        s = self.state
        if s.direction == 'FLAT':
            return True, 0.3, 'trend_flat'

        aligned = (peak_direction == s.direction)
        if aligned:
            return True, s.strength, f'trend_aligned({s.direction} str={s.strength:.2f})'
        else:
            return False, s.strength, f'trend_counter({s.direction} vs {peak_direction})'


# ── Peak Observer (15s) ───────────────────────────────────────────────

@dataclass
class PeakEvent:
    bar_index: int = 0
    timestamp: float = 0.0
    direction: str = ''
    entered: bool = False
    trend_aligned: bool = False
    trend_direction: str = 'FLAT'
    trend_strength: float = 0.0
    fm_15s: float = 0.0
    vol_15s: float = 0.0


class PeakObserver:
    """Tracks all peak detections and their relationship to the trend."""

    def __init__(self):
        self.events = []
        self.stats = {
            'total': 0, 'entered': 0, 'skipped': 0,
            'aligned': 0, 'counter': 0, 'flat': 0,
            'aligned_entered': 0, 'counter_entered': 0,
        }

    def on_peak(self, bar_index: int, timestamp: float, direction: str,
                entered: bool, trend_state: TrendState,
                fm_15s: float = 0.0, vol_15s: float = 0.0):
        """Record a peak detection."""
        aligned = (direction == trend_state.direction)
        flat = (trend_state.direction == 'FLAT')

        event = PeakEvent(
            bar_index=bar_index, timestamp=timestamp,
            direction=direction, entered=entered,
            trend_aligned=aligned if not flat else False,
            trend_direction=trend_state.direction,
            trend_strength=trend_state.strength,
            fm_15s=fm_15s, vol_15s=vol_15s,
        )
        self.events.append(event)

        self.stats['total'] += 1
        if entered:
            self.stats['entered'] += 1
        else:
            self.stats['skipped'] += 1

        if flat:
            self.stats['flat'] += 1
        elif aligned:
            self.stats['aligned'] += 1
            if entered:
                self.stats['aligned_entered'] += 1
        else:
            self.stats['counter'] += 1
            if entered:
                self.stats['counter_entered'] += 1


# ── Session Observer (1h) ─────────────────────────────────────────────

@dataclass
class SessionState:
    regime: str = 'UNKNOWN'      # RALLY, SELLOFF, CHOP, REVERSAL
    direction: str = 'FLAT'       # LONG, SHORT, FLAT
    strength: float = 0.0
    hour_pnl: float = 0.0        # running PnL this hour
    session_pnl: float = 0.0     # running PnL this session


class SessionObserver:
    """Tracks 1h session regime from TBN 1h/4h workers."""

    def __init__(self):
        self.state = SessionState()
        self._hour_trades = 0
        self._hour_pnl = 0.0
        self._session_pnl = 0.0

    def update(self, tbn) -> SessionState:
        """Read 1h worker state and classify session regime."""
        w_1h = tbn.workers.get(3600)
        if w_1h is None:
            return self.state

        idx = w_1h._last_tf_bar_idx
        if idx < 0 or idx >= len(w_1h._states):
            return self.state

        raw = w_1h._states[idx]
        ms = raw['state'] if isinstance(raw, dict) and 'state' in raw else raw

        fm = getattr(ms, 'F_momentum', 0.0)
        dmi_p = getattr(ms, 'dmi_plus', 0.0)
        dmi_m = getattr(ms, 'dmi_minus', 0.0)
        dmi_diff = dmi_p - dmi_m
        adx = getattr(ms, 'adx_strength', 0.0)
        z = getattr(ms, 'z_score', 0.0)

        # Session regime
        if adx > 25 and abs(dmi_diff) > 10:
            if dmi_diff > 0:
                regime = 'RALLY'
                direction = 'LONG'
            else:
                regime = 'SELLOFF'
                direction = 'SHORT'
        elif adx < 15:
            regime = 'CHOP'
            direction = 'FLAT'
        else:
            regime = 'REVERSAL'
            direction = 'LONG' if fm > 0 else 'SHORT' if fm < 0 else 'FLAT'

        strength = min(1.0, adx / 40)

        self.state = SessionState(
            regime=regime, direction=direction, strength=strength,
            hour_pnl=self._hour_pnl, session_pnl=self._session_pnl,
        )
        return self.state

    def on_trade_result(self, pnl: float):
        """Track session PnL."""
        self._hour_pnl += pnl
        self._session_pnl += pnl
        self._hour_trades += 1

    def reset_hour(self):
        self._hour_pnl = 0.0
        self._hour_trades = 0


# ── Observer Hub ──────────────────────────────────────────────────────

class ObserverHub:
    """Coordinates all observer workers and manages logging."""

    def __init__(self, output_dir: str = 'reports/observers', enabled: bool = True):
        self.enabled = enabled
        self.output_dir = output_dir
        self.trend = TrendObserver()
        self.peak = PeakObserver()
        self.session = SessionObserver()

        self._bar_log = []       # per-bar trend/session state
        self._bar_log_interval = 4  # log every 4 bars (1 min at 15s)
        self._bar_count = 0
        self._checkpoint_data = {}

        if enabled:
            os.makedirs(output_dir, exist_ok=True)

    def update(self, bar_index: int, timestamp: float, state, tbn):
        """Called every 15s bar. Updates trend + session observers."""
        if not self.enabled:
            return

        trend_state = self.trend.update(tbn)
        session_state = self.session.update(tbn)

        self._bar_count += 1
        if self._bar_count % self._bar_log_interval == 0:
            self._bar_log.append({
                'bar': bar_index,
                'ts': timestamp,
                'trend_dir': trend_state.direction,
                'trend_str': round(trend_state.strength, 3),
                'trend_dur': trend_state.duration_bars,
                'fm_1m': round(trend_state.fm_1m, 2),
                'vol_1m': round(trend_state.vol_1m, 2),
                'dmi_1m': round(trend_state.dmi_diff_1m, 2),
                'adx_1m': round(trend_state.adx_1m, 2),
                'session': session_state.regime,
                'session_dir': session_state.direction,
            })

    def on_peak(self, bar_index: int, timestamp: float, direction: str,
                entered: bool, fm_15s: float = 0.0, vol_15s: float = 0.0):
        """Called when peak detected (entered or skipped)."""
        if not self.enabled:
            return
        self.peak.on_peak(
            bar_index, timestamp, direction, entered,
            self.trend.state, fm_15s, vol_15s,
        )

    def on_trade_exit(self, pnl: float):
        """Called when trade exits."""
        if not self.enabled:
            return
        self.session.on_trade_result(pnl)

    def get_trend_recommendation(self, peak_direction: str) -> tuple:
        """Ask trend worker if this peak is aligned. Observer only — no blocking."""
        return self.trend.recommend(peak_direction)

    def checkpoint(self, label: str):
        """Save observer state for resume. Called at month/week boundaries."""
        if not self.enabled:
            return
        path = os.path.join(self.output_dir, f'checkpoint_{label}.pkl')
        data = {
            'bar_log': self._bar_log.copy(),
            'peak_events': self.peak.events.copy(),
            'peak_stats': self.peak.stats.copy(),
            'trend_state': self.trend.state,
            'session_state': self.session.state,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f'  [OBSERVER] Checkpoint: {path} '
              f'({len(self._bar_log)} bars, {len(self.peak.events)} peaks)')

    def flush(self):
        """Write all logs to CSV files."""
        if not self.enabled:
            return

        # Bar log (trend + session state every minute)
        if self._bar_log:
            path = os.path.join(self.output_dir, 'trend_session_log.csv')
            keys = self._bar_log[0].keys()
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self._bar_log)
            print(f'  [OBSERVER] Trend/session log: {path} ({len(self._bar_log)} entries)')

        # Peak events
        if self.peak.events:
            path = os.path.join(self.output_dir, 'peak_events.csv')
            rows = []
            for e in self.peak.events:
                rows.append({
                    'bar': e.bar_index, 'ts': e.timestamp,
                    'direction': e.direction, 'entered': e.entered,
                    'trend_aligned': e.trend_aligned,
                    'trend_dir': e.trend_direction,
                    'trend_str': round(e.trend_strength, 3),
                    'fm_15s': round(e.fm_15s, 2),
                    'vol_15s': round(e.vol_15s, 2),
                })
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            print(f'  [OBSERVER] Peak events: {path} ({len(rows)} events)')

        # Summary stats
        path = os.path.join(self.output_dir, 'observer_summary.txt')
        with open(path, 'w') as f:
            f.write('OBSERVER WORKER SUMMARY\n')
            f.write('=' * 50 + '\n\n')
            f.write(f'Total bars logged: {len(self._bar_log)}\n')
            f.write(f'Total peak events: {len(self.peak.events)}\n\n')
            s = self.peak.stats
            f.write(f'Peak Stats:\n')
            f.write(f'  Total: {s["total"]}\n')
            f.write(f'  Entered: {s["entered"]} ({s["entered"]/max(s["total"],1)*100:.1f}%)\n')
            f.write(f'  Skipped: {s["skipped"]}\n')
            f.write(f'  Trend-aligned: {s["aligned"]} '
                    f'(entered: {s["aligned_entered"]})\n')
            f.write(f'  Counter-trend: {s["counter"]} '
                    f'(entered: {s["counter_entered"]})\n')
            f.write(f'  Flat/no trend: {s["flat"]}\n')
        print(f'  [OBSERVER] Summary: {path}')
