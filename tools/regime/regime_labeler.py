#!/usr/bin/env python
"""
Regime Labeler — Step through I-MR regime segments one at a time.

Shows each regime segment zoomed in with 1s detail. For each segment:
  - View price action, MFE/MAE, direction, duration
  - Press Y = mark as SEED, N = skip, Q = quit and save

Seeds are saved to data/regime_seeds/ as ground truth for 192D classifier.

Usage:
    python tools/regime_labeler.py --date 2025-07-14      # label one day
    python tools/regime_labeler.py --week 2025-07-14      # label one week
    python tools/regime_labeler.py --seed 42              # random week
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf
from tools.research.imr import compute_price_imr, detect_regimes
from tools.golden_path import load_1s_index, load_1s_window

# Force interactive backend AFTER imports (tools/research/__init__.py sets Agg via plots.py)
plt.switch_backend('TkAgg')

TICK_SIZE = 0.25
TICK_VALUE = 0.50
SEEDS_DIR = 'data/regime_seeds'
LOOKBACK_BARS = 10  # bars before regime start = the "setup" context


class RegimeLabeler:
    """Step-through regime labeler — one segment at a time."""

    def __init__(self, df_1m, index_1s, context_days=21, analysis_days=7,
                 min_regime_bars=8, label=""):
        self.df_1m = df_1m
        self.index_1s = index_1s
        self.label = label

        # Run I-MR
        print("Computing I-MR regimes...")
        self.price_imr = compute_price_imr(df_1m, context_days=context_days,
                                           analysis_days=analysis_days)
        self.regime_ids, self.regime_meta = detect_regimes(
            self.price_imr, min_regime_bars=min_regime_bars)

        if not self.regime_meta:
            print("No regimes found.")
            sys.exit(1)

        self.close = self.price_imr['close']
        self.timestamps = self.price_imr['timestamps']
        self.highs = df_1m['high'].values.astype(float)
        self.lows = df_1m['low'].values.astype(float)

        # Measure each regime with 1s data
        self._measure_regimes()

        # Labeling state
        self.labels = {}  # regime_id -> 'seed' | 'skip'
        self.current_idx = 0

        print(f"\n  Found {len(self.regime_meta)} regimes to review.")

    def _measure_regimes(self):
        """Compute 1s-resolution MFE/MAE for each regime."""
        cache = {}
        for rm in self.regime_meta:
            si = rm['start_idx']
            ei = rm['end_idx']
            entry = self.close[si]
            exit_p = self.close[ei]
            ts_s = float(self.timestamps[si])
            ts_e = float(self.timestamps[ei])

            change = exit_p - entry
            change_ticks = change / TICK_SIZE
            direction = 'LONG' if change > 0 else ('SHORT' if change < 0 else 'FLAT')

            # Load 1s data for this regime
            rm['_1s_prices'] = None
            rm['_1s_timestamps'] = None
            mfe_1s = mae_1s = 0.0
            time_to_mfe = 0.0

            try:
                df_1s = load_1s_window(self.index_1s, ts_s, ts_e, cache)
                if len(df_1s) >= 5:
                    p1s = df_1s['close'].values.astype(float)
                    ts1s = df_1s['timestamp'].values.astype(float)
                    rm['_1s_prices'] = p1s
                    rm['_1s_timestamps'] = ts1s

                    if direction == 'LONG':
                        fav = (p1s - entry) / TICK_SIZE
                        adv = (entry - p1s) / TICK_SIZE
                    elif direction == 'SHORT':
                        fav = (entry - p1s) / TICK_SIZE
                        adv = (p1s - entry) / TICK_SIZE
                    else:
                        fav = np.abs(p1s - entry) / TICK_SIZE
                        adv = np.zeros(len(p1s))

                    mfe_idx = int(np.argmax(fav))
                    mfe_1s = float(fav[mfe_idx])
                    mae_1s = float(np.max(adv[:mfe_idx + 1])) if mfe_idx > 0 else 0.0
                    time_to_mfe = float(ts1s[mfe_idx] - ts1s[0])
            except Exception:
                pass

            rm['direction'] = direction
            rm['change_ticks'] = round(change_ticks, 1)
            rm['change_dollars'] = round(change_ticks * TICK_VALUE, 2)
            rm['mfe_ticks'] = round(mfe_1s, 1)
            rm['mae_ticks'] = round(mae_1s, 1)
            rm['mfe_dollars'] = round(mfe_1s * TICK_VALUE, 2)
            rm['mae_dollars'] = round(mae_1s * TICK_VALUE, 2)
            rm['duration_mins'] = round((ts_e - ts_s) / 60, 1)
            rm['time_to_mfe_mins'] = round(time_to_mfe / 60, 1)
            rm['ts_start'] = ts_s
            rm['ts_end'] = ts_e
            rm['entry_price'] = round(entry, 2)
            rm['exit_price'] = round(exit_p, 2)

    def _plot_segment(self, rm):
        """Plot a single regime segment with lookback context."""
        plt.close('all')

        si = rm['start_idx']
        ei = rm['end_idx']
        rid = rm['regime_id']

        # Lookback: LOOKBACK_BARS before regime = the setup zone
        lb_start = max(0, si - LOOKBACK_BARS)
        # Also show a few bars after regime end
        plot_end = min(len(self.close), ei + 5)

        ts_slice = self.timestamps[lb_start:plot_end]
        close_slice = self.close[lb_start:plot_end]
        high_slice = self.highs[lb_start:plot_end]
        low_slice = self.lows[lb_start:plot_end]

        dt_stamps = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts_slice]

        fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                                 gridspec_kw={'height_ratios': [3, 1]})
        self.fig = fig

        # ── Panel 1: Price with lookback + regime ──
        ax = axes[0]
        ax.set_facecolor('#FAFAFA')

        # --- LOOKBACK ZONE (blue background, 10 bars before regime) ---
        lb_offset = 0
        lb_len = si - lb_start  # number of lookback bars
        if lb_len > 0:
            lb_dt = dt_stamps[lb_offset:lb_offset + lb_len]
            lb_close = close_slice[lb_offset:lb_offset + lb_len]
            ax.plot(lb_dt, lb_close, color='#1565C0', linewidth=2, alpha=0.7)
            ax.axvspan(lb_dt[0], lb_dt[-1], alpha=0.08, color='#1565C0',
                       label=f'LOOKBACK ({lb_len} bars)')

        # --- REGIME (green/red) ---
        regime_offset = si - lb_start
        regime_len = ei - si + 1
        regime_dt = dt_stamps[regime_offset:regime_offset + regime_len]
        regime_close = close_slice[regime_offset:regime_offset + regime_len]

        color = '#00C853' if rm['direction'] == 'LONG' else '#FF1744' if rm['direction'] == 'SHORT' else '#9E9E9E'
        ax.plot(regime_dt, regime_close, color=color, linewidth=2.5, alpha=0.9)
        if len(regime_dt) >= 2:
            ax.axvspan(regime_dt[0], regime_dt[-1], alpha=0.1, color=color,
                       label=f'REGIME ({rm["direction"]})')

        # --- After bars (gray) ---
        after_offset = regime_offset + regime_len
        if after_offset < len(dt_stamps):
            ax.plot(dt_stamps[after_offset:], close_slice[after_offset:],
                    color='#BDBDBD', linewidth=1, alpha=0.5)

        # Boundary line between lookback and regime
        if lb_len > 0 and len(regime_dt) > 0:
            ax.axvline(x=regime_dt[0], color='black', linewidth=2, linestyle='-',
                       alpha=0.6, label='REGIME START')

        # Entry/exit markers
        ax.scatter([regime_dt[0]], [regime_close[0]], color='blue', s=120,
                   zorder=5, marker='^' if rm['direction'] == 'LONG' else 'v',
                   label=f"Entry: {rm['entry_price']}")
        ax.scatter([regime_dt[-1]], [regime_close[-1]], color='red', s=120,
                   zorder=5, marker='s', label=f"Exit: {rm['exit_price']}")

        # High/low range for regime
        regime_hi = high_slice[regime_offset:regime_offset + regime_len]
        regime_lo = low_slice[regime_offset:regime_offset + regime_len]
        ax.fill_between(regime_dt, regime_lo, regime_hi, alpha=0.08, color=color)

        # SL level ($10 = 20 ticks from entry)
        sl_price = rm['entry_price'] - 20 * TICK_SIZE if rm['direction'] == 'LONG' else rm['entry_price'] + 20 * TICK_SIZE
        ax.axhline(y=sl_price, color='red', linestyle=':', alpha=0.5, label=f'SL $10: {sl_price:.2f}')

        ax.legend(fontsize=8, loc='upper left', ncol=2)
        ax.set_ylabel('Price', fontsize=11)
        ax.grid(True, alpha=0.15)

        # Title with all metrics
        n_total = len(self.regime_meta)
        n_done = len(self.labels)
        n_seeds = sum(1 for v in self.labels.values() if v == 'seed')

        survived = "YES" if rm['mae_ticks'] <= 20 else "NO"
        rr = rm['mfe_dollars'] / rm['mae_dollars'] if rm['mae_dollars'] > 0 else float('inf')
        rr_str = f"1:{rr:.1f}" if rr < 100 else "1:inf"

        ax.set_title(
            f"Regime R{rid} ({self.current_idx + 1}/{n_total}) -- {self.label}\n"
            f"{rm['direction']} | {rm['duration_mins']:.0f} min | {rm['n_bars']} bars | "
            f"Change: {rm['change_ticks']:+.0f}t (${rm['change_dollars']:+.0f})\n"
            f"MFE: ${rm['mfe_dollars']:.0f} ({rm['mfe_ticks']:.0f}t) | "
            f"MAE: ${rm['mae_dollars']:.0f} ({rm['mae_ticks']:.0f}t) | "
            f"R:R: {rr_str} | "
            f"$10 SL survives: {survived} | "
            f"Time to MFE: {rm['time_to_mfe_mins']:.1f}m\n"
            f"[Seeds so far: {n_seeds} | Reviewed: {n_done}/{n_total}] "
            f"Press Y=SEED, N=SKIP, Q=SAVE+QUIT",
            fontsize=10, fontweight='bold'
        )

        # ── Panel 2: 1s resolution (if available) ──
        ax2 = axes[1]
        ax2.set_facecolor('#FAFAFA')

        if rm['_1s_prices'] is not None:
            p1s = rm['_1s_prices']
            ts1s = rm['_1s_timestamps']
            dt1s = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts1s]

            # PnL path in ticks
            if rm['direction'] == 'LONG':
                pnl = (p1s - rm['entry_price']) / TICK_SIZE
            elif rm['direction'] == 'SHORT':
                pnl = (rm['entry_price'] - p1s) / TICK_SIZE
            else:
                pnl = (p1s - rm['entry_price']) / TICK_SIZE

            pnl_colors = ['#00C853' if v >= 0 else '#FF1744' for v in pnl]
            ax2.fill_between(dt1s, 0, pnl, where=pnl >= 0, alpha=0.3, color='#00C853')
            ax2.fill_between(dt1s, 0, pnl, where=pnl < 0, alpha=0.3, color='#FF1744')
            ax2.plot(dt1s, pnl, color='black', linewidth=0.8, alpha=0.7)
            ax2.axhline(y=0, color='black', linewidth=0.5)
            ax2.axhline(y=-20, color='red', linestyle=':', alpha=0.5, label='SL (-20t)')
            ax2.axhline(y=rm['mfe_ticks'], color='green', linestyle='--', alpha=0.5,
                        label=f'MFE ({rm["mfe_ticks"]:.0f}t)')

            ax2.set_ylabel('PnL (ticks)', fontsize=10)
            ax2.set_title('1-Second Resolution PnL Path', fontsize=10)
            ax2.legend(fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No 1s data available', ha='center', va='center',
                     fontsize=14, color='gray', transform=ax2.transAxes)

        ax2.grid(True, alpha=0.15)

        plt.tight_layout()

        # Connect key handler
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        plt.show(block=True)

    def _on_key(self, event):
        """Handle key press — Y=seed, N=skip, Q=save+quit."""
        if event.key in ('y', 'Y'):
            rid = self.regime_meta[self.current_idx]['regime_id']
            self.labels[rid] = 'seed'
            print(f"  R{rid}: SEED")
            plt.close(self.fig)
        elif event.key in ('n', 'N'):
            rid = self.regime_meta[self.current_idx]['regime_id']
            self.labels[rid] = 'skip'
            print(f"  R{rid}: skip")
            plt.close(self.fig)
        elif event.key in ('q', 'Q'):
            self.labels['__quit'] = True
            plt.close(self.fig)

    def run(self):
        """Step through all regimes."""
        print(f"\n  Starting labeling: {len(self.regime_meta)} regimes")
        print(f"  Y = mark as SEED, N = skip, Q = save and quit\n")

        for i, rm in enumerate(self.regime_meta):
            self.current_idx = i

            if '__quit' in self.labels:
                break

            self._plot_segment(rm)

        self._save()

    def _save(self):
        """Save labeled seeds."""
        seeds = []
        for rm in self.regime_meta:
            rid = rm['regime_id']
            if self.labels.get(rid) == 'seed':
                si = rm['start_idx']
                # Lookback: 10 bars before regime start = setup context
                lb_start_idx = max(0, si - LOOKBACK_BARS)
                lb_timestamps = [float(self.timestamps[j])
                                 for j in range(lb_start_idx, si)]

                seeds.append({
                    'regime_id': rid,
                    'direction': rm['direction'],
                    'ts_start': rm['ts_start'],
                    'ts_end': rm['ts_end'],
                    'entry_price': rm['entry_price'],
                    'exit_price': rm['exit_price'],
                    'change_ticks': rm['change_ticks'],
                    'change_dollars': rm['change_dollars'],
                    'mfe_ticks': rm['mfe_ticks'],
                    'mae_ticks': rm['mae_ticks'],
                    'mfe_dollars': rm['mfe_dollars'],
                    'mae_dollars': rm['mae_dollars'],
                    'duration_mins': rm['duration_mins'],
                    'time_to_mfe_mins': rm['time_to_mfe_mins'],
                    'n_bars': rm['n_bars'],
                    # Setup context: 10 bars before regime start
                    'lookback_bars': LOOKBACK_BARS,
                    'lookback_timestamps': lb_timestamps,
                    'lookback_start_idx': lb_start_idx,
                    'regime_start_idx': si,
                })

        if not seeds:
            print("\n  No seeds marked. Nothing saved.")
            return

        os.makedirs(SEEDS_DIR, exist_ok=True)
        ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        seed_path = os.path.join(SEEDS_DIR, f'seeds_{ts_tag}.json')

        with open(seed_path, 'w') as f:
            json.dump({
                'label': self.label,
                'created': ts_tag,
                'n_seeds': len(seeds),
                'n_reviewed': len(self.labels) - (1 if '__quit' in self.labels else 0),
                'n_total_regimes': len(self.regime_meta),
                'seeds': seeds,
            }, f, indent=2)

        csv_path = seed_path.replace('.json', '.csv')
        pd.DataFrame(seeds).to_csv(csv_path, index=False)

        n_reviewed = len(self.labels) - (1 if '__quit' in self.labels else 0)
        n_seeds = len(seeds)
        n_skipped = n_reviewed - n_seeds

        print(f"\n{'='*60}")
        print(f"  LABELING COMPLETE")
        print(f"{'='*60}")
        print(f"  Reviewed:  {n_reviewed}/{len(self.regime_meta)}")
        print(f"  Seeds:     {n_seeds}")
        print(f"  Skipped:   {n_skipped}")
        print(f"  Saved:     {seed_path}")
        print(f"  CSV:       {csv_path}")

        if seeds:
            total_mfe = sum(s['mfe_dollars'] for s in seeds)
            avg_mfe = total_mfe / len(seeds)
            print(f"\n  Seed stats:")
            print(f"    Avg MFE:   ${avg_mfe:.2f}")
            print(f"    Total MFE: ${total_mfe:.2f}")
            print(f"    LONG:      {sum(1 for s in seeds if s['direction']=='LONG')}")
            print(f"    SHORT:     {sum(1 for s in seeds if s['direction']=='SHORT')}")


def main():
    parser = argparse.ArgumentParser(description='Regime Labeler (step-through)')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS root directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random week seed')
    parser.add_argument('--date', default=None,
                        help='Specific day (YYYY-MM-DD)')
    parser.add_argument('--week', default=None,
                        help='Week starting at (YYYY-MM-DD)')
    parser.add_argument('--min-regime-bars', type=int, default=8,
                        help='Min bars per regime (default 8)')
    args = parser.parse_args()

    print("Regime Labeler (step-through)")
    print(f"  Data: {args.data_dir}")

    # Load data
    print("\nLoading 1m data...")
    df_1m = load_atlas_tf(args.data_dir, '1m')
    if df_1m.empty:
        print("ERROR: No 1m data")
        sys.exit(1)

    print("Loading 1s index...")
    index_1s = load_1s_index(args.data_dir)

    # Time window
    if args.date:
        dt = datetime.strptime(args.date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        t_start = dt.timestamp() - 21 * 86400
        t_end = dt.timestamp() + 86400
        mask = (df_1m['timestamp'] >= t_start) & (df_1m['timestamp'] <= t_end)
        df_window = df_1m[mask].reset_index(drop=True)
        label = f"Day: {args.date}"
        analysis_days = 1

    elif args.week:
        dt = datetime.strptime(args.week, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        t_start = dt.timestamp() - 21 * 86400
        t_end = dt.timestamp() + 7 * 86400
        mask = (df_1m['timestamp'] >= t_start) & (df_1m['timestamp'] <= t_end)
        df_window = df_1m[mask].reset_index(drop=True)
        label = f"Week: {args.week}"
        analysis_days = 7

    elif args.seed is not None:
        from tools.imr_golden_path import pick_random_week
        ctx_start, week_start, week_end = pick_random_week(df_1m, seed=args.seed)
        mask = (df_1m['timestamp'] >= ctx_start) & (df_1m['timestamp'] <= week_end)
        df_window = df_1m[mask].reset_index(drop=True)
        ws = datetime.fromtimestamp(week_start, tz=timezone.utc)
        label = f"Week: {ws:%Y-%m-%d} (seed={args.seed})"
        analysis_days = 7
    else:
        from tools.imr_golden_path import pick_random_week
        ctx_start, week_start, week_end = pick_random_week(df_1m, seed=42)
        mask = (df_1m['timestamp'] >= ctx_start) & (df_1m['timestamp'] <= week_end)
        df_window = df_1m[mask].reset_index(drop=True)
        ws = datetime.fromtimestamp(week_start, tz=timezone.utc)
        label = f"Week: {ws:%Y-%m-%d} (seed=42)"
        analysis_days = 7

    print(f"\n  Window: {len(df_window)} bars, {label}")

    labeler = RegimeLabeler(df_window, index_1s,
                            context_days=21, analysis_days=analysis_days,
                            min_regime_bars=args.min_regime_bars,
                            label=label)
    labeler.run()


if __name__ == '__main__':
    main()
