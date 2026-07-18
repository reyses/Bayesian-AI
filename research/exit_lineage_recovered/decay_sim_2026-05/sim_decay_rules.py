"""Walk-forward simulator for the two-rule decay strategy extracted from
human cusp picks (2026-05-10).

The strategy has TWO complementary rules around the 1h_high rail:

  RALLY_LONG: 15s CRM rising AND below the rail → ride up to the rail
  DECAY_SHORT: 15s CRM overshot above the rail → fade back into the band

Walks bar-by-bar at 1m, computes anchors at each bar, evaluates entry/exit
state machine, accounts realized PnL at 1m close.

Usage:
    # Run on the 4 marked days (capture ratio vs oracle)
    python tools/sim_decay_rules.py --days 2025-06-06,2025-09-08,2025-09-09,2025-09-10

    # Run on a clean IS sample (un-marked validation)
    python tools/sim_decay_rules.py --start 2025-07-01 --end 2025-07-31

    # OOS validation
    python tools/sim_decay_rules.py --start 2026-01-01 --end 2026-02-29

Output:
    reports/findings/decay_sim/<run_name>_trades.csv
    reports/findings/decay_sim/<run_name>_daily.csv
    reports/findings/decay_sim/<run_name>_summary.txt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.cusp_marker import compute_anchor, load_1s_window  # reuse
from tools.phase_gate import phases_at_1m


TICK = 0.25
DOL = 0.50
OUT_DIR = Path('reports/findings/decay_sim')


# ── Strategy thresholds (from cusp_picks analysis) ──────────────────────────

@dataclass
class Thresholds:
    """v4: 15s CRM slope SIGN-FLIP detection (the actual user signal).
    Two slope windows per bar: long (10-min) shows recent direction; short
    (3-min) shows current direction. A sign flip between them = cusp.
    Three entry types — short_cusp, long_cusp, long_trend — match the 3
    patterns visible in the user's 36 picks."""
    # Slope thresholds for sign-flip detection
    slope_long_was_rising:   float = 0.30   # SHORT cusp: long was at least +0.3
    slope_short_now_falling: float = -0.10  # SHORT cusp: short slope <= -0.1
    slope_long_was_falling:  float = -0.30  # LONG cusp:  long was at most -0.3
    slope_short_now_rising:  float = 0.10   # LONG cusp:  short slope >= +0.1

    # Long-trend (continuation) — 8 of 18 LONG picks were already trending
    trend_min_slope_long:    float = 0.30   # 10-min slope sustained
    trend_max_pos_in_band:   float = 0.80   # not already at the ceiling

    # Position-in-band filter (room to revert)
    short_min_pos_in_band:   float = 0.50   # SHORT cusp: in upper half
    long_max_pos_in_band:    float = 0.50   # LONG cusp:  in lower half

    # LONG exit — target the upper rail
    long_target_z_1h_high:   float = 1.5
    long_extension_cap:      float = 3.0

    # SHORT exit
    short_max_pos_in_band:   float = 0.30
    short_target_z_15m:      float = -1.0

    # Slope lookbacks (bars at 1m grid)
    slope_long_lb_bars:      int = 10   # 10-min window
    slope_short_lb_bars:     int = 3    # 3-min window
    slope_15m_lb_bars:       int = 15

    # Risk + cadence
    cooldown_min:            int = 30
    max_hold_min:            int = 120
    hard_stop_ticks:         float = 60.0

THRESH = Thresholds()


# ── Bar loading ──────────────────────────────────────────────────────────────

def load_1m_bars(start_ts: float, end_ts: float) -> pd.DataFrame:
    """Load 1m bars across [start_ts, end_ts] from daily ATLAS parquets."""
    dt_s = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    dt_e = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    parts = []
    cur = datetime(dt_s.year, dt_s.month, dt_s.day, tzinfo=timezone.utc)
    end_cap = datetime(dt_e.year, dt_e.month, dt_e.day, tzinfo=timezone.utc)
    while cur <= end_cap:
        day_str = cur.strftime('%Y_%m_%d')
        path = f'DATA/ATLAS/1m/{day_str}.parquet'
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


def compute_anchor_set(target_ts: np.ndarray, t_start: float, t_end: float) -> dict:
    """Compute the four anchors used by the rules. Returns dict of arrays."""
    M_15s, S_15s = compute_anchor('15s', target_ts, t_start, t_end, window=20, column='close')
    M_15m, S_15m = compute_anchor('15m', target_ts, t_start, t_end, window=12, column='close')
    Mh_1h, Sh_1h = compute_anchor('1h',  target_ts, t_start, t_end, window=12, column='high')
    Ml_1h, Sl_1h = compute_anchor('1h',  target_ts, t_start, t_end, window=12, column='low')
    return {
        'M_15s': M_15s, 'S_15s': S_15s,
        'M_15m': M_15m, 'S_15m': S_15m,
        'Mh_1h': Mh_1h, 'Sh_1h': Sh_1h,
        'Ml_1h': Ml_1h, 'Sl_1h': Sl_1h,
    }


# ── Feature engineering at each bar ─────────────────────────────────────────

@dataclass
class BarFeatures:
    ts: float
    close: float
    pos_in_band: float
    sig_above_1h_low: float
    z_1h_high: float              # (close - Mh_1h) / Sh_1h
    z_1h_low: float               # (close - Ml_1h) / Sl_1h
    z_15m_crm: float
    slope_15s_long: float
    slope_15s_short: float
    slope_15m_per_min: float


def features_at(idx: int, close: np.ndarray, ts: np.ndarray, a: dict,
                     slope_lb_15m: int = None) -> Optional[BarFeatures]:
    if slope_lb_15m is None:
        slope_lb_15m = THRESH.slope_15m_lb_bars
    slope_lb_long = THRESH.slope_long_lb_bars
    slope_lb_short = THRESH.slope_short_lb_bars
    """Compute BarFeatures at bar idx. Returns None if any anchor is NaN."""
    c = close[idx]
    M_15s = a['M_15s'][idx];  S_15s = a['S_15s'][idx]
    M_15m = a['M_15m'][idx];  S_15m = a['S_15m'][idx]
    Mh = a['Mh_1h'][idx];     Sh = a['Sh_1h'][idx]
    Ml = a['Ml_1h'][idx];     Sl = a['Sl_1h'][idx]
    if any(np.isnan(v) for v in (M_15s, M_15m, Mh, Ml, Sh, Sl, S_15m)):
        return None
    if Sh <= 0 or Sl <= 0 or S_15m <= 0:
        return None
    band_w = Mh - Ml
    if band_w <= 0:
        return None

    pos = (M_15s - Ml) / band_w
    sig_above_lo = (M_15s - Ml) / Sl
    z_hi = (c - Mh) / Sh
    z_lo = (c - Ml) / Sl
    z_15m = (c - M_15m) / S_15m

    # Slopes — long (10-min) for recent direction, short (3-min) for current
    if idx >= slope_lb_long and not np.isnan(a['M_15s'][idx - slope_lb_long]):
        slope_long = (M_15s - a['M_15s'][idx - slope_lb_long]) / slope_lb_long
    else:
        slope_long = 0.0
    if idx >= slope_lb_short and not np.isnan(a['M_15s'][idx - slope_lb_short]):
        slope_short = (M_15s - a['M_15s'][idx - slope_lb_short]) / slope_lb_short
    else:
        slope_short = 0.0
    if idx >= slope_lb_15m and not np.isnan(a['M_15m'][idx - slope_lb_15m]):
        slope_15m = (M_15m - a['M_15m'][idx - slope_lb_15m]) / slope_lb_15m
    else:
        slope_15m = 0.0

    return BarFeatures(
        ts=ts[idx], close=c,
        pos_in_band=pos, sig_above_1h_low=sig_above_lo,
        z_1h_high=z_hi, z_1h_low=z_lo, z_15m_crm=z_15m,
        slope_15s_long=slope_long, slope_15s_short=slope_short,
        slope_15m_per_min=slope_15m,
    )


# ── Entry/exit rules ─────────────────────────────────────────────────────────

def p_revert_short(z_1h_high: float, slope_15m: float) -> float:
    """Empirical P(reversion DOWN first by 20t | z_1h_high, slope_15m).
    From IS Apr-Oct 2025, n=190k 1m bars. See p_revert_table.json."""
    if z_1h_high >= 1.0 and z_1h_high < 2.0:
        if slope_15m <= -0.1 and slope_15m > -0.5:
            return 0.55                         # the prime SHORT fade cell
        if -0.1 < slope_15m < 0.1:
            return 0.475
    if z_1h_high >= 2.0 and z_1h_high < 3.0:
        if slope_15m <= -0.1:
            return 0.55                         # cautious — n=33 + neighbor evidence
        if -0.1 < slope_15m < 0.1:
            return 0.40
    if z_1h_high >= 3.0:
        if -0.1 < slope_15m < 0.1:
            return 0.31                         # +3σ FLAT 15m: continuation dominant
        return 0.40
    return 0.46                                 # base rate; below +1σ, no fade edge


def p_revert_long(z_1h_low: float, slope_15m: float) -> float:
    """Empirical P(reversion UP first by 20t | z_1h_low, slope_15m).
    The bounce side is stronger and works at lower extension levels."""
    if z_1h_low <= -3.0:
        if slope_15m <= -0.5:
            return 0.71                         # strongest cell
        if slope_15m <= -0.1:
            return 0.64
        return 0.55
    if z_1h_low <= -2.0:
        if slope_15m <= -0.5:
            return 0.64
        if slope_15m <= -0.1:
            return 0.56
        return 0.52
    if z_1h_low <= -1.0:
        if slope_15m <= -0.5:
            return 0.63                         # below low rail with hard 15m down
        if slope_15m <= -0.1:
            return 0.55
    return 0.50


P_REVERT_FIRE = 0.55   # Level-1 gate: P_revert must be >= this to fire


def short_cusp_entry(f: BarFeatures) -> bool:
    """SHORT cusp: 15s CRM slope sign-flip + Level-1 P_revert gate."""
    if not (f.slope_15s_long >= THRESH.slope_long_was_rising
                and f.slope_15s_short <= THRESH.slope_short_now_falling
                and f.pos_in_band >= THRESH.short_min_pos_in_band):
        return False
    return p_revert_short(f.z_1h_high, f.slope_15m_per_min) >= P_REVERT_FIRE


def long_cusp_entry(f: BarFeatures) -> bool:
    """LONG cusp: 15s CRM slope sign-flip + Level-1 P_revert gate.
    Uses z_1h_low (price vs 1h_low rail) for the empirical probability."""
    if not (f.slope_15s_long <= THRESH.slope_long_was_falling
                and f.slope_15s_short >= THRESH.slope_short_now_rising
                and f.pos_in_band <= THRESH.long_max_pos_in_band):
        return False
    return p_revert_long(f.z_1h_low, f.slope_15m_per_min) >= P_REVERT_FIRE


def long_trend_entry(f: BarFeatures) -> bool:
    """LONG continuation — no probability gate (trend has different signature)."""
    return (f.slope_15s_long >= THRESH.trend_min_slope_long
                and f.slope_15s_short >= THRESH.trend_min_slope_long
                and f.pos_in_band < THRESH.trend_max_pos_in_band
                and f.z_1h_high < 0)


def long_exit(f: BarFeatures, entry_z_hi: float,
                  pnl_ticks: float) -> tuple[bool, str]:
    if pnl_ticks <= -THRESH.hard_stop_ticks:
        return True, 'hard_stop'
    if f.z_1h_high >= THRESH.long_target_z_1h_high:
        return True, 'target_1h_rail'
    if (f.z_1h_high >= THRESH.long_extension_cap
            and f.slope_15s_short <= 0):
        return True, 'extension_cap'
    return False, ''


def short_exit(f: BarFeatures, entry_pos: float,
                   pnl_ticks: float) -> tuple[bool, str]:
    if pnl_ticks <= -THRESH.hard_stop_ticks:
        return True, 'hard_stop'
    if f.pos_in_band <= THRESH.short_max_pos_in_band:
        return True, 'back_in_band'
    if f.z_15m_crm <= THRESH.short_target_z_15m:
        return True, 'crossed_15m_mean'
    return False, ''


# ── State machine simulator ──────────────────────────────────────────────────

@dataclass
class Trade:
    side: str
    entry_ts: float
    entry_price: float
    entry_features: BarFeatures
    exit_ts: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ''
    peak_pnl_ticks: float = 0.0
    worst_pnl_ticks: float = 0.0
    duration_min: float = 0.0

    @property
    def realized_pnl_dollars(self) -> float:
        if self.exit_ts == 0:
            return 0.0
        if self.side == 'LONG':
            ticks = (self.exit_price - self.entry_price) / TICK
        else:
            ticks = (self.entry_price - self.exit_price) / TICK
        return ticks * DOL


def _build_phase_array(ts: np.ndarray) -> np.ndarray:
    """Compute phase per 1m bar by sampling the per-day phase array. Returns
    'NORMAL' for any day whose data is missing (permissive)."""
    from collections import defaultdict
    by_day = defaultdict(list)
    for i, t in enumerate(ts):
        day = datetime.fromtimestamp(int(t), tz=timezone.utc).strftime('%Y_%m_%d')
        by_day[day].append(i)
    phases = np.array(['NORMAL'] * len(ts), dtype=object)
    for day, indices in by_day.items():
        idx_arr = np.array(indices)
        ts_arr = ts[idx_arr].astype(np.int64)
        day_phases = phases_at_1m(day, ts_arr)
        for k, p in zip(indices, day_phases):
            phases[k] = p
    return phases


def simulate(close: np.ndarray, ts: np.ndarray, a: dict,
                phase_gate: set = None) -> list:
    """Walk bar-by-bar with TRANSITION-based entries.
    Optional phase_gate: set of phase labels where entries are ALLOWED.
    Other bars stay flat (existing trades manage normally)."""
    trades = []
    open_trade: Optional[Trade] = None
    cooldown_until_ts = 0.0
    phases = _build_phase_array(ts) if phase_gate is not None else None

    for i in range(len(close)):
        f = features_at(i, close, ts, a)
        if f is None:
            continue

        # ── Manage open trade
        if open_trade is not None:
            if open_trade.side == 'LONG':
                pnl_ticks = (f.close - open_trade.entry_price) / TICK
            else:
                pnl_ticks = (open_trade.entry_price - f.close) / TICK
            open_trade.peak_pnl_ticks = max(open_trade.peak_pnl_ticks, pnl_ticks)
            open_trade.worst_pnl_ticks = min(open_trade.worst_pnl_ticks, pnl_ticks)

            should_exit, reason = False, ''
            if open_trade.side == 'LONG':
                should_exit, reason = long_exit(
                    f, open_trade.entry_features.z_1h_high, pnl_ticks)
            else:
                should_exit, reason = short_exit(
                    f, open_trade.entry_features.pos_in_band, pnl_ticks)

            dur_min = (f.ts - open_trade.entry_ts) / 60.0
            if not should_exit and dur_min >= THRESH.max_hold_min:
                should_exit, reason = True, 'time_stop'

            if should_exit:
                open_trade.exit_ts = f.ts
                open_trade.exit_price = f.close
                open_trade.exit_reason = reason
                open_trade.duration_min = dur_min
                trades.append(open_trade)
                cooldown_until_ts = f.ts + THRESH.cooldown_min * 60
                open_trade = None
            else:
                continue

        # ── Try to open (cooldown + phase gate)
        if f.ts < cooldown_until_ts:
            continue
        if phase_gate is not None and phases[i] not in phase_gate:
            continue

        # Try entries in priority order: cusps first (rare), trend last (common)
        if short_cusp_entry(f):
            open_trade = Trade(side='SHORT', entry_ts=f.ts,
                                  entry_price=f.close, entry_features=f,
                                  exit_reason='', peak_pnl_ticks=0, worst_pnl_ticks=0,
                                  exit_ts=0, exit_price=0, duration_min=0)
            open_trade.entry_reason = 'short_cusp'
        elif long_cusp_entry(f):
            open_trade = Trade(side='LONG', entry_ts=f.ts,
                                  entry_price=f.close, entry_features=f)
            open_trade.entry_reason = 'long_cusp'
        elif long_trend_entry(f):
            open_trade = Trade(side='LONG', entry_ts=f.ts,
                                  entry_price=f.close, entry_features=f)
            open_trade.entry_reason = 'long_trend'

    # Force-close anything still open at end-of-window with last bar price
    if open_trade is not None and len(close) > 0:
        last_f = features_at(len(close) - 1, close, ts, a)
        if last_f is not None:
            open_trade.exit_ts = last_f.ts
            open_trade.exit_price = last_f.close
            open_trade.exit_reason = 'eod_force_close'
            open_trade.duration_min = (last_f.ts - open_trade.entry_ts) / 60.0
            trades.append(open_trade)

    return trades


# ── Reporting ────────────────────────────────────────────────────────────────

def write_outputs(trades: list, run_name: str, oracle_pnl: float = 0.0):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trade_path = OUT_DIR / f'{run_name}_trades.csv'
    daily_path = OUT_DIR / f'{run_name}_daily.csv'
    summary_path = OUT_DIR / f'{run_name}_summary.txt'

    # Per-trade CSV
    with open(trade_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['side', 'entry_ts', 'entry_utc', 'entry_price',
                       'exit_ts', 'exit_utc', 'exit_price', 'exit_reason',
                       'duration_min', 'realized_pnl_dollars',
                       'peak_pnl_ticks', 'worst_pnl_ticks',
                       'entry_pos_in_band', 'entry_z_1h_high', 'entry_z_15m_crm',
                       'entry_slope_15s_long', 'entry_slope_15s_short',
                       'entry_slope_15m', 'entry_sig_above_1h_low'])
        for t in trades:
            ef = t.entry_features
            w.writerow([
                t.side, t.entry_ts,
                datetime.fromtimestamp(t.entry_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                t.entry_price, t.exit_ts,
                datetime.fromtimestamp(t.exit_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                t.exit_price, t.exit_reason,
                round(t.duration_min, 1),
                round(t.realized_pnl_dollars, 2),
                round(t.peak_pnl_ticks, 1), round(t.worst_pnl_ticks, 1),
                round(ef.pos_in_band, 3), round(ef.z_1h_high, 3),
                round(ef.z_15m_crm, 3), round(ef.slope_15s_long, 3),
                round(ef.slope_15s_short, 3), round(ef.slope_15m_per_min, 3),
                round(ef.sig_above_1h_low, 3),
            ])

    # Daily PnL
    from collections import defaultdict
    daily = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'n_long': 0, 'n_short': 0})
    for t in trades:
        day = datetime.fromtimestamp(t.entry_ts, tz=timezone.utc).strftime('%Y-%m-%d')
        daily[day]['n'] += 1
        daily[day]['pnl'] += t.realized_pnl_dollars
        if t.side == 'LONG':
            daily[day]['n_long'] += 1
        else:
            daily[day]['n_short'] += 1
    with open(daily_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'n_trades', 'n_long', 'n_short', 'pnl_dollars'])
        for day in sorted(daily.keys()):
            d = daily[day]
            w.writerow([day, d['n'], d['n_long'], d['n_short'], round(d['pnl'], 2)])

    # Summary
    n = len(trades)
    if n == 0:
        summary = f'{run_name}: NO TRADES FIRED\n'
    else:
        pnls = np.array([t.realized_pnl_dollars for t in trades])
        durs = np.array([t.duration_min for t in trades])
        longs = [t for t in trades if t.side == 'LONG']
        shorts = [t for t in trades if t.side == 'SHORT']
        n_win = (pnls > 0).sum()
        n_lose = (pnls < 0).sum()
        win = pnls[pnls > 0].sum()
        lose = -pnls[pnls < 0].sum()
        pf_wr = (win / lose - 1) if lose > 0 else float('inf')
        n_days = len(daily)

        # PF-WR per CLAUDE.md
        # Reason breakdown
        from collections import Counter
        reasons = Counter(t.exit_reason for t in trades)

        summary = (
            f'=== {run_name} ===\n'
            f'n_trades       : {n}  (LONG {len(longs)} / SHORT {len(shorts)})\n'
            f'n_days_active  : {n_days}\n'
            f'\n'
            f'Total PnL      : ${pnls.sum():.0f}\n'
            f'$/day          : ${pnls.sum()/n_days:.1f}\n'
            f'Mean per trade : ${pnls.mean():.2f}\n'
            f'Median per tr  : ${np.median(pnls):.2f}\n'
            f'\n'
            f'Win rate (cnt) : {100*n_win/n:.1f}% ({n_win} W / {n_lose} L)\n'
            f'PF-WR (CLAUDE) : {pf_wr:+.3f}\n'
            f'Total wins     : ${win:.0f}\n'
            f'Total losses   : ${lose:.0f}\n'
            f'\n'
            f'Mean duration  : {durs.mean():.0f} min\n'
            f'Median duration: {np.median(durs):.0f} min\n'
            f'\n'
            f'Exit reasons   : {dict(reasons)}\n'
        )
        if oracle_pnl > 0:
            capture = 100 * pnls.sum() / oracle_pnl
            summary += f'\nORACLE PnL     : ${oracle_pnl:.0f}\n'
            summary += f'CAPTURE RATIO  : {capture:.0f}% (sim ${pnls.sum():.0f} / oracle ${oracle_pnl:.0f})\n'

    print(summary)
    with open(summary_path, 'w') as f:
        f.write(summary)
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def _ts_of(d: str) -> float:
    return datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', help='YYYY-MM-DD start (range mode)')
    ap.add_argument('--end', help='YYYY-MM-DD end (range mode)')
    ap.add_argument('--days', help='Comma-separated YYYY-MM-DD list (specific days)')
    ap.add_argument('--name', help='Run name (default auto)')
    ap.add_argument('--oracle', type=float, default=0,
                       help='Optional oracle PnL for capture-ratio reporting')
    ap.add_argument('--phase-gate', choices=['none', 'strict', 'extended'], default='none',
                       help='Phase gate: strict=NORMAL+STABILIZING, '
                              'extended=all except DIRECTIONAL')
    args = ap.parse_args()

    if args.days:
        day_list = [d.strip() for d in args.days.split(',')]
        # Build a contiguous time range covering all days
        sorted_days = sorted(day_list)
        t_start = _ts_of(sorted_days[0])
        t_end = _ts_of(sorted_days[-1]) + 86400
        run_name = args.name or f'days_{sorted_days[0]}_to_{sorted_days[-1]}'
    elif args.start and args.end:
        t_start = _ts_of(args.start)
        t_end = _ts_of(args.end) + 86400
        run_name = args.name or f'range_{args.start}_to_{args.end}'
    else:
        ap.error('Provide --days OR --start+--end')

    print(f'Loading 1m bars {datetime.fromtimestamp(t_start, tz=timezone.utc)} '
              f'→ {datetime.fromtimestamp(t_end, tz=timezone.utc)}...')
    df = load_1m_bars(t_start, t_end)
    if df.empty:
        print('No data')
        return

    # If --days specified, filter to those specific calendar days only
    if args.days:
        keep_mask = pd.Series(False, index=df.index)
        for d in day_list:
            day_start = _ts_of(d)
            day_end = day_start + 86400
            keep_mask |= (df['timestamp'] >= day_start) & (df['timestamp'] < day_end)
        df = df[keep_mask].reset_index(drop=True)

    print(f'  {len(df)} bars loaded')
    if len(df) == 0:
        return

    close = df['close'].values.astype(float)
    ts = df['timestamp'].values.astype(np.int64)

    print('Computing anchors...')
    a = compute_anchor_set(ts, t_start, t_end)
    print(f'  Anchors ready (M_15s, M_15m, Mh_1h, Ml_1h)')

    if args.phase_gate == 'strict':
        gate = {'NORMAL', 'STABILIZING'}
    elif args.phase_gate == 'extended':
        gate = {'NORMAL', 'STABILIZING', 'FLATTENED', 'PIVOT_CANDIDATE'}
    else:
        gate = None
    print(f'Simulating (phase_gate={args.phase_gate})...')
    trades = simulate(close, ts, a, phase_gate=gate)
    print(f'  {len(trades)} trades closed')

    write_outputs(trades, run_name, oracle_pnl=args.oracle)


if __name__ == '__main__':
    main()
