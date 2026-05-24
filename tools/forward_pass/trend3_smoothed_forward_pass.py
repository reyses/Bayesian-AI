"""DMI-smoothed trend3 forward pass on NT8 OOS.

Tests regime-flip entries with DMI-like smoothing:
  - EMA period × margin × ADX floor sweep (controls flip frequency)
  - Fire mode: 'on_flip' (only on regime change) or 'regime' (every bar of regime)
  - TP/SL via tick-exact engine

Per user 2026-05-17: filter blips by requiring sustained prior agreement
before flipping the regime.
"""
from __future__ import annotations
import os
import sys
import subprocess
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from training.utils.ticker import MultiDayV2Ticker
from core_v2.exits import TimeStop
from core_v2.exits_tick_exact import TickExactTP, TickExactSL
from training.pipelines.iso_orchestrator import IsoOrchestrator
from training.strategies import Trend3SmoothedStrategy
from training.utils.v2_cols import swing_noise_w, reversion_prob_w, z_se_w
from training.pipelines.v2_native_iso import _histogram_mode, _pf_trade_wr


RAW_CACHE = 'reports/findings/regret_oracle/trend3_cache_OOS_NT8.parquet'
NT8_ATLAS_ROOT = 'DATA/ATLAS_NT8'
NT8_FEATURES_ROOT = 'DATA/ATLAS_NT8/FEATURES_5s_v2'
COST = 4.0
MAX_HOLD = 120   # 10 min
OUT_DIR = Path('reports/findings/regret_oracle')


def _entry_extras(state):
    return {
        'entry_swing_noise': state.get(swing_noise_w('1m')),
        'entry_reversion_prob': state.get(reversion_prob_w('1m')),
        'entry_z_se': state.get(z_se_w('1m')),
    }


def bootstrap_ci_mean(arr, n_boot=4000, seed=42):
    arr = np.asarray(arr)
    if len(arr) < 2:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        boots[i] = arr[rng.integers(0, n, n)].mean()
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def build_smoothed_cache(ema_p, adx_p, margin, adx_floor):
    """Re-run precompute_trend3_smoothed with these params; return out path."""
    out = OUT_DIR / f'trend3_smoothed_E{ema_p}_A{adx_p}_M{margin:.2f}_F{int(adx_floor)}.parquet'
    if not out.exists():
        subprocess.check_call([
            sys.executable, 'tools/precompute_trend3_smoothed.py',
            '--input', RAW_CACHE,
            '--out', str(out),
            '--ema-period', str(ema_p),
            '--adx-period', str(adx_p),
            '--margin', f'{margin}',
            '--adx-floor', f'{adx_floor}',
        ])
    return str(out)


def run_one(days, cache_path, mode, adx_floor_runtime, tp, sl):
    strat = Trend3SmoothedStrategy(smoothed_cache=cache_path,
                                    mode=mode, adx_floor=adx_floor_runtime,
                                    fire_cadence='1m')
    exits = [
        TickExactSL(usd=-abs(sl), slippage_ticks=1.0),
        TickExactTP(usd=abs(tp), slippage_ticks=0.0),
        TimeStop(max_bars=MAX_HOLD),
    ]
    orch = IsoOrchestrator(strategies=[strat], exits=exits,
                            entry_extras_hook=_entry_extras)
    multi = MultiDayV2Ticker(days=days, atlas_root=NT8_ATLAS_ROOT,
                              features_root=NT8_FEATURES_ROOT)
    per_tier = orch.run(multi)
    trades = per_tier.get('TREND3_SMOOTHED', [])
    if not trades:
        return None

    gross = np.array([t.pnl for t in trades], dtype=np.float64)
    mae   = np.array([t.trough_pnl for t in trades], dtype=np.float64)
    pnls  = gross - COST
    day_pnl = defaultdict(float)
    for t, p in zip(trades, pnls):
        day_pnl[t.entry_day] += p
    active = list(day_pnl.values())
    mean_d, lo_d, hi_d = bootstrap_ci_mean(active)
    win_days = sum(1 for v in active if v > 0)
    n_total = len(days); n_active = len(active)
    reasons = Counter(t.exit_reason for t in trades)
    return {
        'n_trades': int(len(trades)),
        'trades_per_day': float(len(trades) / max(n_total, 1)),
        'mean_per_trade_net': float(pnls.mean()),
        'mean_per_day_net': mean_d,
        'pnl_day_ci_lo': lo_d, 'pnl_day_ci_hi': hi_d,
        'mode_per_day': _histogram_mode(np.array(active), bin_width=25.0),
        'median_per_day': float(np.median(active)),
        'day_wr_active': win_days / max(n_active, 1),
        'pf_trade_wr_net': _pf_trade_wr(pnls),
        'mean_mae': float(mae.mean()),
        'tp_pct': float(((reasons.get('take_profit_exact', 0) +
                          reasons.get('take_profit', 0)) / len(trades)) * 100),
        'sl_pct': float(((reasons.get('hard_stop_exact', 0) +
                          reasons.get('hard_stop', 0)) / len(trades)) * 100),
    }


def main():
    # Discover days from raw cache (matches OOS coverage)
    cache_df = pd.read_parquet(RAW_CACHE)
    days = sorted(cache_df['day'].unique().tolist())
    print(f'NT8 OOS days: {len(days)}')

    # DMI-style smoothing grid
    SMOOTH_GRID = [
        # (ema_p, adx_p, margin, adx_floor_build)
        (5,  10, 0.05, 15),    # baseline
        (5,  10, 0.10, 20),    # tighter
        (10, 14, 0.10, 20),
        (10, 14, 0.15, 25),    # very strict
        (15, 14, 0.15, 25),    # slowest
    ]
    # TP/SL — keep focused
    TP_SL = [(20, 10), (30, 15), (40, 20), (60, 30)]
    MODE = 'on_flip'   # fire only on confirmed regime change

    rows = []
    combo = 0
    total = len(SMOOTH_GRID) * len(TP_SL)
    for (ep, ap, mg, af) in SMOOTH_GRID:
        cache = build_smoothed_cache(ep, ap, mg, af)
        for (tp, sl) in TP_SL:
            combo += 1
            print(f'\n[{combo}/{total}] EMA={ep} ADX={ap} margin={mg} floor={af}  TP={tp} SL={sl}')
            r = run_one(days, cache, MODE, 0.0, tp, sl)
            if r is None:
                print('  (no trades)')
                continue
            r.update(dict(ema_p=ep, adx_p=ap, margin=mg, adx_floor=af,
                          tp_usd=tp, sl_usd=sl, mode=MODE))
            print(f'  n={r["n_trades"]} /day={r["trades_per_day"]:.2f}  '
                  f'$/t={r["mean_per_trade_net"]:+.2f}  '
                  f'$/day={r["mean_per_day_net"]:+.1f} CI [{r["pnl_day_ci_lo"]:+.0f},{r["pnl_day_ci_hi"]:+.0f}]  '
                  f'DayWR={r["day_wr_active"]:.3f}  TP%={r["tp_pct"]:.0f} SL%={r["sl_pct"]:.0f}')
            rows.append(r)

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'trend3_smoothed_forward_pass.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nWrote: {out_csv}')

    cols = ['ema_p','adx_p','margin','adx_floor','tp_usd','sl_usd',
            'n_trades','trades_per_day','mean_per_trade_net','mean_per_day_net',
            'pnl_day_ci_lo','pnl_day_ci_hi','day_wr_active','tp_pct','sl_pct']
    print('\n=== TOP 10 BY $/day NET ===')
    print(df.sort_values('mean_per_day_net', ascending=False).head(10)[cols].to_string(index=False))

    sig = df[df['pnl_day_ci_lo'] > 0]
    print(f'\n=== POSITIVE WITH SIGNIFICANT CI (lo > 0): {len(sig)} configs ===')
    if len(sig) > 0:
        print(sig.sort_values('mean_per_day_net', ascending=False)[cols].to_string(index=False))


if __name__ == '__main__':
    main()
