"""KPI grid for GoldenCombinedStrategy through tick-exact engine on OOS 2026.

User KPI: $100/day NET, low MAE, high Day WR.

Grid:
  T_timing: 0.70, 0.80, 0.85, 0.90, 0.95
  T_dir   : 0.55, 0.65, 0.75, 0.85
  TP/SL pairs: (20,20), (30,20), (40,20), (60,30)
  Cost: $4/trade
  Time stop: 30 min
  Tick-exact exits (no intrabar overshoot)
"""
from __future__ import annotations
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from training_iso_v2.ticker import MultiDayV2Ticker
from training_iso_v2.exits import TimeStop
from training_iso_v2.exits_tick_exact import TickExactTP, TickExactSL
from training_iso_v2.iso_orchestrator import IsoOrchestrator
from training_iso_v2.strategies import GoldenCombinedStrategy
from training_iso_v2.v2_cols import swing_noise_w, reversion_prob_w, z_se_w
from training_iso_v2.run_iso import _resolve_days, _histogram_mode, _pf_trade_wr


# User insight 2026-05-17: direction is easier at 1m than 5s — trust the
# 89% direction acc by loosening T_dir, and tighten T_timing for higher
# precision on the rare-pivot-moment classification.
T_TIMING = [0.80, 0.85, 0.88, 0.91, 0.94]
T_DIR = [0.50, 0.55, 0.60]   # loose — let direction classifier do the work
TP_SL = [(30, 15), (30, 20), (40, 20), (60, 30), (40, 15)]
COST = 4.0
MAX_HOLD = 120   # 10 min
TIMING_CACHE_OOS = 'reports/findings/regret_oracle/zz_timing_cache_OOS_NT8_atr4_gbm.parquet'
TIMING_CACHE_IS = 'reports/findings/regret_oracle/zz_timing_cache_IS_atr4_gbm.parquet'
TIMING_CACHE = TIMING_CACHE_OOS   # default; overridden by --target
OUT_DIR = Path('reports/findings/regret_oracle')

# NT8 OOS lives under DATA/ATLAS_NT8; need to pass atlas_root through to ticker
NT8_ATLAS_ROOT = 'DATA/ATLAS_NT8'
NT8_FEATURES_ROOT = 'DATA/ATLAS_NT8/FEATURES_5s_v2'


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


def run_one(days, t_timing, t_dir, tp, sl, atlas_root=None, features_root=None):
    strat = GoldenCombinedStrategy(timing_cache=TIMING_CACHE,
                                    t_timing=t_timing, t_dir=t_dir,
                                    fire_cadence='1m')
    exits = [
        TickExactSL(usd=-abs(sl), slippage_ticks=1.0),
        TickExactTP(usd=abs(tp), slippage_ticks=0.0),
        TimeStop(max_bars=MAX_HOLD),
    ]
    orch = IsoOrchestrator(
        strategies=[strat],
        exits=exits,
        entry_extras_hook=_entry_extras,
    )
    ticker_kwargs = {}
    if atlas_root: ticker_kwargs['atlas_root'] = atlas_root
    if features_root: ticker_kwargs['features_root'] = features_root
    multi = MultiDayV2Ticker(days=days, **ticker_kwargs)
    per_tier = orch.run(multi)
    trades = per_tier.get('GOLDEN_COMBINED', [])
    if not trades:
        return None
    gross = np.array([t.pnl for t in trades], dtype=np.float64)
    mae = np.array([t.trough_pnl for t in trades], dtype=np.float64)
    peak = np.array([t.peak_pnl for t in trades], dtype=np.float64)
    pnls = gross - COST
    day_pnl = defaultdict(float)
    for t, p in zip(trades, pnls):
        day_pnl[t.entry_day] += p
    active = list(day_pnl.values())
    mean_d, lo_d, hi_d = bootstrap_ci_mean(active)
    win_days = sum(1 for v in active if v > 0)
    n_total = len(days); n_active = len(active)
    reasons = Counter(t.exit_reason for t in trades)
    return {
        't_timing': t_timing, 't_dir': t_dir, 'tp_usd': tp, 'sl_usd': sl,
        'n_trades': int(len(trades)),
        'trades_per_day': float(len(trades) / max(n_total, 1)),
        'mean_per_trade_net': float(pnls.mean()),
        'mode_per_trade': _histogram_mode(pnls, bin_width=2.0),
        'mean_per_day_net': mean_d,
        'pnl_day_ci_lo': lo_d, 'pnl_day_ci_hi': hi_d,
        'mode_per_day': _histogram_mode(np.array(active), bin_width=25.0),
        'median_per_day': float(np.median(active)),
        'day_wr_active': win_days / max(n_active, 1),
        'day_wr_total': win_days / max(n_total, 1),
        'pf_trade_wr_net': _pf_trade_wr(pnls),
        'mean_mae': float(mae.mean()),
        'tp_pct': float(((reasons.get('take_profit_exact', 0) + reasons.get('take_profit', 0)) / len(trades)) * 100),
        'sl_pct': float(((reasons.get('hard_stop_exact', 0) + reasons.get('hard_stop', 0)) / len(trades)) * 100),
    }


def main():
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument('--target', choices=['is','oos'], default='oos')
    p.add_argument('--label', default='', help='Output csv suffix')
    args = p.parse_args()
    global TIMING_CACHE
    TIMING_CACHE = TIMING_CACHE_IS if args.target == 'is' else TIMING_CACHE_OOS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.target == 'oos':
        # NT8 OOS — explicit day list from the cache (32 days from 2026_03_20+)
        import pandas as _pd
        days = sorted(_pd.read_parquet(TIMING_CACHE)['day'].unique().tolist())
        atlas_root = NT8_ATLAS_ROOT
        features_root = NT8_FEATURES_ROOT
    else:
        days = _resolve_days('is')
        atlas_root = None
        features_root = None
    print(f'{args.target.upper()} days: {len(days)}   cache: {TIMING_CACHE}')
    print(f'  atlas_root: {atlas_root or "default DATA/ATLAS"}')
    combos = [(t_t, t_d, tp, sl)
              for t_t in T_TIMING
              for t_d in T_DIR
              for (tp, sl) in TP_SL]
    print(f'Grid: {len(combos)} combos')

    rows = []
    for i, (t_t, t_d, tp, sl) in enumerate(combos):
        print(f'\n[{i+1}/{len(combos)}] T_timing={t_t} T_dir={t_d} TP={tp} SL={sl}')
        r = run_one(days, t_t, t_d, tp, sl,
                     atlas_root=atlas_root, features_root=features_root)
        if r is None:
            print('  (no trades)')
            continue
        print(f'  n={r["n_trades"]} /day={r["trades_per_day"]:.2f}  '
              f'$/t={r["mean_per_trade_net"]:+.2f}  '
              f'$/day={r["mean_per_day_net"]:+.1f} CI [{r["pnl_day_ci_lo"]:+.0f},{r["pnl_day_ci_hi"]:+.0f}]  '
              f'DayWR={r["day_wr_active"]:.3f}  MAE=${r["mean_mae"]:.1f}')
        print(f'  TP%={r["tp_pct"]:.0f}  SL%={r["sl_pct"]:.0f}')
        rows.append(r)

    df = pd.DataFrame(rows)
    suffix = f'_{args.label}' if args.label else f'_{args.target.upper()}'
    out_csv = OUT_DIR / f'golden_combined_kpi_ZIGZAG{suffix}.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nWrote: {out_csv}')

    print(f'\n=== TOP 10 BY $/day NET ===')
    cols = ['t_timing','t_dir','tp_usd','sl_usd','n_trades','trades_per_day',
            'mean_per_trade_net','mean_per_day_net','pnl_day_ci_lo','pnl_day_ci_hi',
            'day_wr_active','mean_mae','tp_pct','sl_pct']
    print(df.sort_values('mean_per_day_net', ascending=False).head(10)[cols].to_string(index=False))

    print(f'\n=== POSITIVE PNL configs (CI lower bound > 0) ===')
    sig = df[df['pnl_day_ci_lo'] > 0]
    if len(sig) > 0:
        print(sig.sort_values('mean_per_day_net', ascending=False)[cols].to_string(index=False))
    else:
        print('(none with CI > 0)')

    print(f'\n=== Top by DayWR (descending) ===')
    print(df.sort_values('day_wr_active', ascending=False).head(10)[cols].to_string(index=False))


if __name__ == '__main__':
    main()
