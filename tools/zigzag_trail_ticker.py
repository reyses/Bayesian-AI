"""
zigzag_trail_ticker.py -- True bar-by-bar v1.3-RC trail simulator.
==================================================================

Honest forward pass for v1.3-RC: walks 1s data per day, aggregates to 1m,
detects pivot confirmations, tracks per-trade peak_pnl on 1m closes (matching
NT8 v1.3-RC `Calculate.OnBarClose`), checks trail breach on 1m close, exits
at next 1s tick (matching NT8 market-order fill semantics).

Replaces the analytical approximation in `tools/forward_pass_v15rc.py` (which
overstates trail by 1.5-3x because it uses idealized MFE-eff_dist).

Pattern lifted from `tools/nightmare_ticker.py` (the existing live-system bar
walker with peak_pnl tracking and giveback-style exits). Same 1s tick loop,
same 1m bar aggregation, same numpy-array hot path. Strategy logic swapped:
zigzag pivot entry + opposite-pivot exit + max(fixed, pct·HWM) trail.

Per-trade economics computed at:
  Entry:  open of 1s tick AFTER 1m pivot confirmation bar closes.
  Exit:   1m bar where trail breaches (open of next 1s tick), OR
          opposite pivot's confirm bar+1, OR EOD force-close.

Usage:
    python tools/zigzag_trail_ticker.py 2026-03-20
    python tools/zigzag_trail_ticker.py 2026-02-09 --r 30 --trail-pct 0.10
    python tools/zigzag_trail_ticker.py all --r 30
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.zigzag_backtest import zigzag_pivots_with_confirmation

# ── MNQ contract constants ────────────────────────────────────────────────
TICK_SIZE = 0.25
DOLLAR_PER_POINT = 2.0          # MNQ
COMMISSION_PER_SIDE = 1.0        # $1 each side per contract

# ── v1.3-RC trail defaults (must mirror NT8_ZigzagRunner.cs) ──────────────
DEFAULT_R                       = 30.0
DEFAULT_TRAIL_ACTIVATE_PTS      = 10.0
DEFAULT_TRAIL_DIST_PTS          = 5.0
DEFAULT_TRAIL_PCT               = 0.10

# ── EOD / cutoff (UTC, matches NT8 strategy defaults) ─────────────────────
EOD_HOUR_UTC, EOD_MIN_UTC       = 20, 55
ENTRY_CUTOFF_HOUR_UTC, ENTRY_CUTOFF_MIN_UTC = 20, 30

ATLAS_ROOT = 'DATA/ATLAS'


# ── Per-day simulator ─────────────────────────────────────────────────────

def _regime_at_ts(pivots_1h_with_close_ts: list, ts_query: int) -> int:
    """Latest 1h regime at ts_query. +1 LONG, -1 SHORT, 0 unknown."""
    if not pivots_1h_with_close_ts:
        return 0
    # Linear scan (small N, called per pivot — fine).
    regime = 0
    for confirm_ts, kind in pivots_1h_with_close_ts:
        if confirm_ts <= ts_query:
            regime = +1 if kind == 'low' else -1
        else:
            break
    return regime


def _envelope_band(z: float) -> str:
    a = abs(z)
    if   a < 0.5: return 'inside'
    elif a < 1.0: return 'near'
    elif a < 1.5: return 'edge'
    elif a < 2.0: return 'outside'
    else:         return 'extreme'


def _filter_keep(regime: int, direction: int, band: str) -> bool:
    """v1.5-RC regime+envelope filter:
        SKIP regime AGAINST + extreme  (fakeout-at-exhaustion)
        SKIP regime WITH    + inside   (noise fakeout)"""
    if regime == 0:
        return True
    with_reg = (regime == direction)
    if (not with_reg) and band == 'extreme': return False
    if with_reg and band == 'inside':        return False
    return True


def simulate_day(day_label: str, r: float,
                 trail_activate: float, trail_dist: float, trail_pct: float,
                 atlas_root: str = ATLAS_ROOT,
                 use_filter: bool = False,
                 r_1h: float = 75.0,
                 sl_pts: float = 0.0) -> tuple[list[dict], dict]:
    """Walk a single day's 1s data with v1.3-RC trail, return (trades, summary).

    Trade dict keys: entry_ts, entry_price, exit_ts, exit_price, direction,
                     pnl_pts, pnl_usd, mfe_pts, mae_pts, leg_min, exit_reason,
                     pivot_kind, peak_at_exit_pts.

    If use_filter=True, applies the v1.5-RC regime+envelope filter at entry:
    skips trades when 1h regime is AGAINST + envelope extreme, or
    1h regime is WITH + envelope inside.

    If sl_pts > 0, applies a hard intra-bar stop loss at entry_price ± sl_pts
    (= v1.6-RC catastrophic backstop). SL is checked on every 1s tick (intra-bar
    semantics, mirroring how a real stop order fires in NT8). When SL triggers,
    exit price = entry_price ± sl_pts (assuming sufficient liquidity at the
    stop level; conservative since stop orders may slip in thin moments).
    """
    p_1m = os.path.join(atlas_root, '1m', f'{day_label}.parquet')
    p_1s = os.path.join(atlas_root, '1s', f'{day_label}.parquet')
    if not (os.path.exists(p_1m) and os.path.exists(p_1s)):
        return [], dict(day=day_label, n_trades=0, n_skipped=0,
                        n_trail=0, n_pivot=0, n_eod=0, total_usd=0.0)

    df_1m = pd.read_parquet(p_1m).sort_values('timestamp').reset_index(drop=True)
    df_1s = pd.read_parquet(p_1s).sort_values('timestamp').reset_index(drop=True)
    if len(df_1m) < 2 or len(df_1s) < 60:
        return [], dict(day=day_label, n_trades=0, n_skipped=0,
                        n_trail=0, n_pivot=0, n_eod=0, total_usd=0.0)

    closes_1m = df_1m['close'].values.astype(np.float64)
    ts_1m     = df_1m['timestamp'].values.astype(np.int64)

    pivots = zigzag_pivots_with_confirmation(closes_1m, r)
    if len(pivots) < 2:
        return [], dict(day=day_label, n_trades=0, n_skipped=0,
                        n_trail=0, n_pivot=0, n_eod=0, total_usd=0.0)

    # Map confirm bar index -> (kind, ext_price). Each pivot confirms at
    # 1m bar `confirm_idx`; entry for that pivot's trade fills on the bar
    # AFTER (= confirm_idx + 1).
    pivot_by_confirm: dict[int, tuple[str, float]] = {}
    for ext_idx, ext_price, kind, confirm_idx in pivots:
        pivot_by_confirm[int(confirm_idx)] = (kind, float(ext_price))

    # ── Filter setup: pre-compute 1h pivots and load FEATURES_5s ──────────
    pivots_1h_close_ts: list[tuple[int, str]] = []   # (confirm_close_ts, kind)
    feat_ts: np.ndarray | None = None
    feat_z_1h: np.ndarray | None = None
    if use_filter:
        p_1h = os.path.join(atlas_root, '1h', f'{day_label}.parquet')
        p_feat = os.path.join(atlas_root, 'FEATURES_5s', f'{day_label}.parquet')
        if os.path.exists(p_1h) and os.path.exists(p_feat):
            df_1h = pd.read_parquet(p_1h).sort_values('timestamp').reset_index(drop=True)
            closes_1h = df_1h['close'].values.astype(np.float64)
            ts_1h_arr = df_1h['timestamp'].values.astype(np.int64)
            pivs_1h = zigzag_pivots_with_confirmation(closes_1h, r_1h)
            for _ext_idx_h, _ext_p_h, kind_h, confirm_idx_h in pivs_1h:
                if int(confirm_idx_h) < len(ts_1h_arr):
                    close_ts_h = int(ts_1h_arr[confirm_idx_h]) + 3600
                    pivots_1h_close_ts.append((close_ts_h, kind_h))
            pivots_1h_close_ts.sort()

            df_feat = pd.read_parquet(p_feat).sort_values('timestamp').reset_index(drop=True)
            if '1h_z_se' in df_feat.columns:
                feat_ts = df_feat['timestamp'].values.astype(np.int64)
                feat_z_1h = df_feat['1h_z_se'].values.astype(np.float64)
                feat_z_1h = np.nan_to_num(feat_z_1h, nan=0.0, posinf=0.0, neginf=0.0)
        # If 1h or features missing, filter falls back to "always keep".

    # 1s arrays for tick loop
    _ts_1s = df_1s['timestamp'].values.astype(np.int64)
    _o_1s  = df_1s['open'].values.astype(np.float64)
    _h_1s  = df_1s['high'].values.astype(np.float64)
    _l_1s  = df_1s['low'].values.astype(np.float64)
    _c_1s  = df_1s['close'].values.astype(np.float64)
    n_ticks = len(_ts_1s)

    # ── Trade state ──────────────────────────────────────────────────────
    trades: list[dict] = []
    n_skipped = n_trail = n_pivot = n_eod = n_sl = 0

    in_pos = False
    direction = 0   # +1 long, -1 short
    entry_price = 0.0
    entry_ts = 0
    peak_pnl_pts = 0.0    # HWM in points (MFE)
    peak_close_pts = 0.0  # HWM tracked from 1m closes (NT8 v1.3-RC behavior)
    peak_close_price = 0.0  # actual price at the peak-close
    worst_pnl_pts = 0.0   # MAE in points (negative tracker)
    bars_held_1m = 0
    trail_armed = False
    pending_entry: tuple[str, float] | None = None  # set when 1m close fires a pivot

    # 1m aggregator state
    cur_minute_idx = -1
    agg_open = 0.0
    agg_high = -1e18
    agg_low  = 1e18
    agg_close = 0.0
    agg_ts = 0

    # Index into ts_1m so we can look up which pivot (if any) confirmed at this minute close.
    def _confirm_lookup(close_ts: int) -> tuple[str, float] | None:
        """Find the 1m bar whose ts_1m[i]+60 == close_ts (i.e. this minute's close).
        If that bar is in pivot_by_confirm, return the pivot info."""
        # Each 1m bar in df_1m closes at ts_1m[i] + 60. close_ts of an aggregated
        # minute equals (minute_index + 1) * 60.
        # Bar index = where ts_1m == close_ts - 60.
        bar_start_ts = close_ts - 60
        # Binary search
        idx = np.searchsorted(ts_1m, bar_start_ts, side='left')
        if idx < len(ts_1m) and int(ts_1m[idx]) == int(bar_start_ts):
            if int(idx) in pivot_by_confirm:
                return pivot_by_confirm[int(idx)]
        return None

    # ── Tick loop ────────────────────────────────────────────────────────
    for i in range(n_ticks):
        tick_ts = int(_ts_1s[i])
        tick_o  = _o_1s[i]
        tick_h  = _h_1s[i]
        tick_l  = _l_1s[i]
        tick_c  = _c_1s[i]
        tick_minute_idx = tick_ts // 60

        # Initialize aggregator on first tick
        if cur_minute_idx == -1:
            cur_minute_idx = tick_minute_idx
            agg_open = tick_o
            agg_high = tick_h
            agg_low  = tick_l
            agg_close = tick_c
            agg_ts = tick_ts
        else:
            agg_high = max(agg_high, tick_h)
            agg_low  = min(agg_low, tick_l)
            agg_close = tick_c

        # ── If we have a pending entry from the previous 1m close, fill on
        #    THIS tick (open price of first 1s after the 1m close).
        if pending_entry is not None and not in_pos:
            kind, _ext_price = pending_entry
            direction = +1 if kind == 'low' else -1
            entry_price = tick_o
            entry_ts = tick_ts
            peak_pnl_pts = 0.0
            peak_close_pts = 0.0
            peak_close_price = entry_price
            worst_pnl_pts = 0.0
            bars_held_1m = 0
            trail_armed = False
            in_pos = True
            pending_entry = None

        # ── In-position: track 1s MAE / MFE for the trade record (not for trail)
        if in_pos:
            if direction > 0:
                tick_pnl_pts = tick_c - entry_price
                tick_worst = tick_l - entry_price   # most adverse 1s extreme this tick
            else:
                tick_pnl_pts = entry_price - tick_c
                tick_worst = entry_price - tick_h
            if tick_pnl_pts > peak_pnl_pts:
                peak_pnl_pts = tick_pnl_pts
            if tick_worst < worst_pnl_pts:
                worst_pnl_pts = tick_worst

            # ── HARD STOP-LOSS check (intra-bar, fires on tick low/high) ─────
            # SL price = entry_price ± sl_pts. Check 1s bar's low (long) or high (short).
            if sl_pts > 0:
                if direction > 0:
                    sl_price = entry_price - sl_pts
                    sl_breached = (tick_l <= sl_price)
                else:
                    sl_price = entry_price + sl_pts
                    sl_breached = (tick_h >= sl_price)
                if sl_breached:
                    # Conservative fill: assume SL fills at the SL trigger price
                    # (in reality, slippage in fast moves can fill below SL — modeled
                    # as fixed sl_pts loss for a clean estimate).
                    exit_price_sl = sl_price
                    pnl_pts = direction * (exit_price_sl - entry_price)
                    pnl_usd = pnl_pts * DOLLAR_PER_POINT - 2.0 * COMMISSION_PER_SIDE
                    trades.append({
                        'day': day_label,
                        'entry_ts': entry_ts, 'entry_price': entry_price,
                        'exit_ts': tick_ts,    'exit_price': float(exit_price_sl),
                        'direction': direction,
                        'pnl_pts': float(pnl_pts), 'pnl_usd': float(pnl_usd),
                        'mfe_pts': float(peak_pnl_pts),
                        'mae_pts': float(-worst_pnl_pts),
                        'peak_close_pts': float(peak_close_pts),
                        'leg_min': (tick_ts - entry_ts) / 60.0,
                        'exit_reason': 'sl',
                    })
                    n_sl += 1
                    in_pos = False
                    trail_armed = False
                    # Continue tick loop — flat now, won't re-enter until next pivot
                    continue

        # ── 1m bar boundary? ──────────────────────────────────────────────
        if tick_minute_idx == cur_minute_idx:
            continue

        # New minute started — process the close of the just-completed minute.
        completed_close_ts = (cur_minute_idx + 1) * 60
        completed_close_price = agg_close

        # Reset aggregator for the new minute (current tick starts it)
        cur_minute_idx = tick_minute_idx
        agg_open = tick_o
        agg_high = tick_h
        agg_low  = tick_l
        agg_close = tick_c
        agg_ts = tick_ts

        # ── 1) On 1m close: update HWM-of-closes for trail logic ───────────
        if in_pos:
            bars_held_1m += 1
            if direction > 0:
                close_pnl_pts = completed_close_price - entry_price
            else:
                close_pnl_pts = entry_price - completed_close_price
            if close_pnl_pts > peak_close_pts:
                peak_close_pts = close_pnl_pts
                peak_close_price = completed_close_price

            # ── 2) Trail check (mirrors v1.3-RC OnBarClose logic) ─────────
            if trail_activate > 0 and not trail_armed and peak_close_pts >= trail_activate:
                trail_armed = True

            exit_reason = None
            exit_price = None
            if trail_armed:
                eff_dist = max(trail_dist, trail_pct * peak_close_pts)
                if direction > 0:
                    stop_px = peak_close_price - eff_dist
                    breached = completed_close_price <= stop_px
                else:
                    stop_px = peak_close_price + eff_dist
                    breached = completed_close_price >= stop_px
                if breached:
                    # Exit at next 1s tick open (= the current tick we're processing).
                    exit_price = float(tick_o)
                    exit_reason = 'trail'

            # ── 3) EOD force-close ────────────────────────────────────────
            if exit_reason is None:
                close_dt = datetime.fromtimestamp(completed_close_ts, tz=timezone.utc)
                mins_of_day = close_dt.hour * 60 + close_dt.minute
                if mins_of_day >= EOD_HOUR_UTC * 60 + EOD_MIN_UTC:
                    exit_price = float(tick_o)
                    exit_reason = 'eod'

            # If exit triggered, close the trade
            if exit_reason is not None:
                pnl_pts = direction * (exit_price - entry_price)
                pnl_usd = pnl_pts * DOLLAR_PER_POINT - 2.0 * COMMISSION_PER_SIDE
                trades.append({
                    'day': day_label,
                    'entry_ts': entry_ts, 'entry_price': entry_price,
                    'exit_ts': tick_ts,    'exit_price': exit_price,
                    'direction': direction,
                    'pnl_pts': float(pnl_pts), 'pnl_usd': float(pnl_usd),
                    'mfe_pts': float(peak_pnl_pts),
                    'mae_pts': float(-worst_pnl_pts),
                    'peak_close_pts': float(peak_close_pts),
                    'leg_min': (tick_ts - entry_ts) / 60.0,
                    'exit_reason': exit_reason,
                })
                if exit_reason == 'trail': n_trail += 1
                else:                       n_eod += 1
                in_pos = False
                trail_armed = False

        # ── 4) Pivot confirm at this 1m close? Set up entry for NEXT tick ──
        pivot_info = _confirm_lookup(completed_close_ts)
        if pivot_info is not None:
            new_kind, new_ext_price = pivot_info

            # If currently in position: exit on opposite pivot (= reverse).
            if in_pos:
                # Opposite pivot if direction LONG and new pivot is high (= want SHORT next),
                # or direction SHORT and new pivot is low (= want LONG next).
                opposite = ((direction > 0 and new_kind == 'high') or
                            (direction < 0 and new_kind == 'low'))
                if opposite:
                    exit_price = float(tick_o)
                    pnl_pts = direction * (exit_price - entry_price)
                    pnl_usd = pnl_pts * DOLLAR_PER_POINT - 2.0 * COMMISSION_PER_SIDE
                    trades.append({
                        'day': day_label,
                        'entry_ts': entry_ts, 'entry_price': entry_price,
                        'exit_ts': tick_ts,    'exit_price': exit_price,
                        'direction': direction,
                        'pnl_pts': float(pnl_pts), 'pnl_usd': float(pnl_usd),
                        'mfe_pts': float(peak_pnl_pts),
                        'peak_close_pts': float(peak_close_pts),
                        'leg_min': (tick_ts - entry_ts) / 60.0,
                        'exit_reason': 'pivot',
                    })
                    n_pivot += 1
                    in_pos = False
                    trail_armed = False

            # If past entry cutoff, do NOT open a new position
            if not in_pos:
                close_dt = datetime.fromtimestamp(completed_close_ts, tz=timezone.utc)
                mins_of_day = close_dt.hour * 60 + close_dt.minute
                past_cutoff = mins_of_day >= ENTRY_CUTOFF_HOUR_UTC * 60 + ENTRY_CUTOFF_MIN_UTC
                if past_cutoff:
                    n_skipped += 1
                elif use_filter and pivots_1h_close_ts and feat_ts is not None:
                    # v1.5-RC regime+envelope filter
                    new_dir = +1 if new_kind == 'low' else -1
                    regime = _regime_at_ts(pivots_1h_close_ts, completed_close_ts)
                    fz_idx = int(np.searchsorted(feat_ts, completed_close_ts, side='right')) - 1
                    z_1h = float(feat_z_1h[fz_idx]) if 0 <= fz_idx < len(feat_z_1h) else 0.0
                    band = _envelope_band(z_1h)
                    if _filter_keep(regime, new_dir, band):
                        pending_entry = (new_kind, new_ext_price)
                    else:
                        n_skipped += 1
                else:
                    # Queue entry for next 1s tick (no filter, or filter data missing)
                    pending_entry = (new_kind, new_ext_price)

    # End of day: close any open position at last 1s close
    if in_pos:
        last_tick_ts = int(_ts_1s[-1])
        last_tick_price = float(_c_1s[-1])
        pnl_pts = direction * (last_tick_price - entry_price)
        pnl_usd = pnl_pts * DOLLAR_PER_POINT - 2.0 * COMMISSION_PER_SIDE
        trades.append({
            'day': day_label,
            'entry_ts': entry_ts, 'entry_price': entry_price,
            'exit_ts': last_tick_ts, 'exit_price': last_tick_price,
            'direction': direction,
            'pnl_pts': float(pnl_pts), 'pnl_usd': float(pnl_usd),
            'mfe_pts': float(peak_pnl_pts),
            'peak_close_pts': float(peak_close_pts),
            'leg_min': (last_tick_ts - entry_ts) / 60.0,
            'exit_reason': 'eod_final',
        })
        n_eod += 1

    summary = dict(
        day=day_label,
        n_trades=len(trades),
        n_skipped=n_skipped,
        n_trail=n_trail, n_pivot=n_pivot, n_eod=n_eod, n_sl=n_sl,
        total_usd=sum(t['pnl_usd'] for t in trades),
    )
    return trades, summary


# ── Driver ────────────────────────────────────────────────────────────────

def is_2025(d): return d.startswith('2025_')
def is_2026(d): return d.startswith('2026_')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('day', nargs='?', default='all',
                    help='Date YYYY-MM-DD, comma-separated, or "all"')
    ap.add_argument('--atlas', default=ATLAS_ROOT)
    ap.add_argument('--r', type=float, default=DEFAULT_R)
    ap.add_argument('--trail-activate', type=float, default=DEFAULT_TRAIL_ACTIVATE_PTS)
    ap.add_argument('--trail-dist',     type=float, default=DEFAULT_TRAIL_DIST_PTS)
    ap.add_argument('--trail-pct',      type=float, default=DEFAULT_TRAIL_PCT)
    ap.add_argument('--out',            default='reports/findings/zigzag_trail_ticker_trades.csv')
    ap.add_argument('--filter', action='store_true',
                    help='Enable regime+envelope filter (skip AGAINST+extreme + WITH+inside)')
    ap.add_argument('--r-1h',  type=float, default=75.0, help='1h zigzag R for regime')
    ap.add_argument('--sl-pts', type=float, default=0.0,
                    help='Hard intra-bar stop loss in points (0 = disabled). '
                         'Mirrors NT8 SetStopLoss(): fires on 1s tick when low/high '
                         'touches entry_price - sl_pts (long) or entry_price + sl_pts (short).')
    args = ap.parse_args()

    # Resolve day list
    if args.day == 'all':
        files = sorted(glob.glob(os.path.join(args.atlas, '1m', '*.parquet')))
        days = [os.path.splitext(os.path.basename(p))[0] for p in files]
    elif ',' in args.day:
        days = [d.replace('-', '_') for d in args.day.split(',')]
    else:
        days = [args.day.replace('-', '_')]

    print('=' * 110)
    print(f'V1.3-RC BAR-BY-BAR TRAIL TICKER  --  R={args.r:g}, '
          f'trail activate={args.trail_activate}pt, dist={args.trail_dist}pt, pct={args.trail_pct}')
    print(f'Days: {len(days)}.  EOD={EOD_HOUR_UTC:02d}:{EOD_MIN_UTC:02d} UTC, '
          f'cutoff={ENTRY_CUTOFF_HOUR_UTC:02d}:{ENTRY_CUTOFF_MIN_UTC:02d} UTC')
    print('=' * 110)

    all_trades: list[dict] = []
    summaries: list[dict] = []
    iterator = days if len(days) == 1 else tqdm(days, desc='days')
    for d in iterator:
        trades, summary = simulate_day(
            d, args.r, args.trail_activate, args.trail_dist, args.trail_pct,
            atlas_root=args.atlas, use_filter=args.filter, r_1h=args.r_1h,
            sl_pts=args.sl_pts)
        all_trades.extend(trades)
        summaries.append(summary)

    if not all_trades:
        print('No trades produced.')
        return

    df_t = pd.DataFrame(all_trades)
    df_s = pd.DataFrame(summaries)

    # Daily P&L splits
    daily = df_t.groupby('day')['pnl_usd'].sum()
    is_d  = daily[daily.index.map(is_2025)]
    oos_d = daily[daily.index.map(is_2026)]

    def rep(label, s):
        if len(s) == 0:
            print(f'{label}: 0 days')
            return
        arr = s.values
        print(f'{label}: {len(arr)} days  '
              f'mean ${arr.mean():+8.2f}/day  '
              f'dWR {(arr>0).mean()*100:5.1f}%  '
              f'best ${arr.max():+7.0f}  worst ${arr.min():+7.0f}  '
              f'std ${arr.std():6.0f}')

    print()
    rep('IS  2025', is_d)
    rep('OOS 2026', oos_d)

    # Exit-reason breakdown
    print(f'\nExit reasons across {len(df_t):,} trades:')
    for reason, sub in df_t.groupby('exit_reason'):
        wr = (sub['pnl_pts'] > 0).mean() * 100
        mean_usd = sub['pnl_usd'].mean()
        print(f'  {reason:>12}  N={len(sub):>5,}  '
              f'WR {wr:5.1f}%  mean ${mean_usd:+7.2f}/trade  '
              f'mean MFE {sub["mfe_pts"].mean()*DOLLAR_PER_POINT:+7.2f} USD')

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_t.to_csv(args.out, index=False)
    print(f'\nWrote {len(df_t):,} trades -> {args.out}')


if __name__ == '__main__':
    main()
