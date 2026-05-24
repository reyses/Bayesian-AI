"""Forward-pass KPI grid for Trend3Strategy on NT8 OOS.

The 3-class direction classifier predicts LONG/SHORT/NEUTRAL at every 1m bar.
This forward pass fires:
    LONG  when P(LONG)  > t_long  and P(LONG)  > P(SHORT)
    SHORT when P(SHORT) > t_short and P(SHORT) > P(LONG)
plus an optional directional strength gate (max(p_long,p_short) - p_neutral >= t_strength).

Through the tick-exact engine with TP/SL/TimeStop exits.

KPI: $/day NET, CI, Day WR, TP%/SL%, MAE — per CLAUDE.md mandates.
"""
from __future__ import annotations
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from training.utils.ticker import MultiDayV2Ticker
from core_v2.exits import TimeStop
from core_v2.exits_tick_exact import TickExactTP, TickExactSL
from training.pipelines.iso_orchestrator import IsoOrchestrator
from training.strategies import Trend3Strategy
from training.utils.v2_cols import swing_noise_w, reversion_prob_w, z_se_w
from training.pipelines.v2_native_iso import _histogram_mode, _pf_trade_wr


TREND3_CACHE_OOS = 'reports/findings/regret_oracle/trend3_cache_OOS_NT8.parquet'
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


def run_one(days, t_long, t_short, t_strength, tp, sl):
    strat = Trend3Strategy(trend3_cache=TREND3_CACHE_OOS,
                           t_long=t_long, t_short=t_short,
                           t_strength=t_strength,
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
    trades = per_tier.get('TREND3', [])
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
        't_long': t_long, 't_short': t_short, 't_strength': t_strength,
        'tp_usd': tp, 'sl_usd': sl,
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
        'ts_pct': float((reasons.get('time_stop', 0) / len(trades)) * 100),
    }


def main():
    # Days from cache
    cache_df = pd.read_parquet(TREND3_CACHE_OOS)
    days = sorted(cache_df['day'].unique().tolist())
    print(f'NT8 OOS days: {len(days)}')

    # Grid — symmetric thresholds; sweep TP/SL too
    T_LONG  = [0.40, 0.50, 0.60, 0.70, 0.80]
    T_STRENGTH = [0.0, 0.10, 0.20]
    TP_SL = [(20, 10), (30, 15), (30, 20), (40, 20), (60, 30)]

    combos = [(tl, tl, ts, tp, sl)
              for tl in T_LONG
              for ts in T_STRENGTH
              for (tp, sl) in TP_SL]
    print(f'Grid: {len(combos)} combos')

    rows = []
    for i, (tl, tsh, tst, tp, sl) in enumerate(combos):
        print(f'\n[{i+1}/{len(combos)}] t_long=t_short={tl}  t_strength={tst}  '
              f'TP={tp}  SL={sl}')
        r = run_one(days, tl, tsh, tst, tp, sl)
        if r is None:
            print('  (no trades)')
            continue
        print(f'  n={r["n_trades"]} /day={r["trades_per_day"]:.2f}  '
              f'$/t={r["mean_per_trade_net"]:+.2f}  '
              f'$/day={r["mean_per_day_net"]:+.1f} CI [{r["pnl_day_ci_lo"]:+.0f},{r["pnl_day_ci_hi"]:+.0f}]  '
              f'DayWR={r["day_wr_active"]:.3f}  MAE=${r["mean_mae"]:.1f}  '
              f'TP%={r["tp_pct"]:.0f} SL%={r["sl_pct"]:.0f}')
        rows.append(r)

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'trend3_forward_pass_NT8.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nWrote: {out_csv}')

    cols = ['t_long','t_strength','tp_usd','sl_usd','n_trades','trades_per_day',
            'mean_per_trade_net','mean_per_day_net','pnl_day_ci_lo','pnl_day_ci_hi',
            'day_wr_active','mean_mae','tp_pct','sl_pct']
    print('\n=== TOP 15 BY $/day NET ===')
    print(df.sort_values('mean_per_day_net', ascending=False).head(15)[cols].to_string(index=False))

    print('\n=== POSITIVE $/day with CI lower > 0 (significant) ===')
    sig = df[df['pnl_day_ci_lo'] > 0]
    if len(sig) > 0:
        print(sig.sort_values('mean_per_day_net', ascending=False)[cols].to_string(index=False))
    else:
        print('(none)')

    print('\n=== TOP 10 BY Day WR ===')
    print(df.sort_values('day_wr_active', ascending=False).head(10)[cols].to_string(index=False))


if __name__ == '__main__':
    main()
