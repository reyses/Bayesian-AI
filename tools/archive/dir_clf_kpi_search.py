"""Autonomous KPI-driven grid search on direction-classifier strategy.

KPIs (per user 2026-05-16):
  1. PRIMARY:    $100/day NET (mean, OOS)
  2. SECONDARY:  low MAE (max adverse excursion per trade)
  3. TERTIARY:   high Day WR (positive PnL days)

Grid:
  TP/SL pairs ($): (20,20), (30,20), (40,20), (40,30), (60,30), (60,40)
  Confidence thresholds: 0.65, 0.75, 0.85
  Fire cadences: 5m, 15m
  Cost per trade: $4 round-trip (commission + slippage)
  Time stop: 30 min

For each combo, runs OOS through training_iso_v2 engine with clean exit suite
(TP + SL + TimeStop only — no thesis exits).

Composite score:
  score = $/day_net + 50 * (Day WR - 0.5) - 0.5 * mean_MAE_abs
        (rewards $/day, bonuses for >50% DayWR, penalty for big MAE)

Output:
  reports/findings/regret_oracle/dir_clf_kpi_search.csv  (full grid)
  reports/findings/regret_oracle/dir_clf_kpi_top.md      (top 10 ranked)
"""
from __future__ import annotations
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from training.utils.ticker import MultiDayV2Ticker
from core_v2.exits import HardStop, TakeProfit, TimeStop
from core_v2.exits_tick_exact import TickExactTP, TickExactSL
from training.pipelines.iso_orchestrator import IsoOrchestrator
from training.strategies import DirectionClassifierStrategy
from training.utils.v2_cols import swing_noise_w, reversion_prob_w, z_se_w
from training.pipelines.v2_native_iso import _resolve_days, _histogram_mode, _pf_trade_wr


# Grid — asymmetric R/R variants, higher confidence thresholds
TP_SL_PAIRS = [
    (10, 5),    # very tight, 2:1
    (15, 10),   # 1.5:1
    (20, 10),   # 2:1
    (30, 10),   # 3:1
    (40, 10),   # 4:1
    (20, 5),    # 4:1, tight
]
THRESHOLDS = [0.85, 0.90, 0.95]
CADENCES = ['5m', '15m']
COST_PER_TRADE = 4.0       # $ round-trip
MAX_HOLD_BARS = 360         # 30 min
OUT_DIR = Path('reports/findings/regret_oracle')

# KPI weights
TARGET_DAY = 100.0
DAY_WR_BONUS = 50.0    # per (DayWR - 0.5)
MAE_PENALTY = 0.5      # per $ of mean adverse excursion (abs value)


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


def run_one(days, thr, cadence, tp, sl, cost, max_hold, tick_exact: bool = True,
            slippage_ticks: float = 1.0):
    strat = DirectionClassifierStrategy(threshold=thr, fire_cadence=cadence)
    if tick_exact:
        # Tick-exact: closes at exactly TP/SL price (intrabar via OHLC high/low),
        # with slippage of 1 tick against us on adverse fills (SL).
        exits = [
            TickExactSL(usd=-abs(sl), slippage_ticks=slippage_ticks),
            TickExactTP(usd=abs(tp), slippage_ticks=0.0),
            TimeStop(max_bars=max_hold),
        ]
    else:
        exits = [HardStop(usd=-abs(sl)), TakeProfit(usd=abs(tp)),
                 TimeStop(max_bars=max_hold)]
    orch = IsoOrchestrator(
        strategies=[strat],
        exits=exits,
        entry_extras_hook=_entry_extras,
    )
    multi = MultiDayV2Ticker(days=days)
    per_tier = orch.run(multi)
    trades = per_tier.get('DIRECTION_CLF', [])
    if not trades:
        return None
    gross = np.array([t.pnl for t in trades], dtype=np.float64)
    mae = np.array([t.trough_pnl for t in trades], dtype=np.float64)
    peak = np.array([t.peak_pnl for t in trades], dtype=np.float64)
    pnls = gross - cost
    day_pnl = defaultdict(float)
    for t, p in zip(trades, pnls):
        day_pnl[t.entry_day] += p
    active = list(day_pnl.values())
    mean_d, lo_d, hi_d = bootstrap_ci_mean(active)
    win_days = sum(1 for v in active if v > 0)
    n_total = len(days)
    n_active = len(active)
    day_wr = win_days / max(n_active, 1)

    reasons = Counter(t.exit_reason for t in trades)
    mean_mae_abs = float(abs(mae.mean()))
    return {
        'threshold': thr, 'cadence': cadence, 'tp_usd': tp, 'sl_usd': sl,
        'n_trades': int(len(trades)),
        'trades_per_day': float(len(trades) / max(n_total, 1)),
        'n_active_days': n_active, 'n_total_days': n_total,
        'mean_per_trade_gross': float(gross.mean()),
        'mean_per_trade_net': float(pnls.mean()),
        'mode_per_trade': _histogram_mode(pnls, bin_width=2.0),
        'mean_per_day_net': mean_d,
        'pnl_day_ci_lo': lo_d, 'pnl_day_ci_hi': hi_d,
        'mode_per_day': _histogram_mode(np.array(active), bin_width=25.0),
        'median_per_day': float(np.median(active)),
        'day_wr_active': day_wr, 'day_wr_total': win_days / max(n_total, 1),
        'pf_trade_wr_net': _pf_trade_wr(pnls),
        'mean_mae': float(mae.mean()),    # negative
        'mean_mae_abs': mean_mae_abs,
        'median_mae': float(np.median(mae)),
        'worst10_mae': float(np.percentile(mae, 10)),  # 10th pct = worst 10%
        'mean_peak': float(peak.mean()),
        'tp_pct': float(((reasons.get('take_profit', 0) + reasons.get('take_profit_exact', 0)) / len(trades)) * 100),
        'sl_pct': float(((reasons.get('hard_stop', 0) + reasons.get('hard_stop_exact', 0)) / len(trades)) * 100),
        'ts_pct': float((reasons.get('time_stop', 0) / len(trades)) * 100),
        'eod_pct': float((reasons.get('eod_close', 0) / len(trades)) * 100),
        'composite_score': mean_d + DAY_WR_BONUS * (day_wr - 0.5) - MAE_PENALTY * mean_mae_abs,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    days = _resolve_days('oos')
    print(f'OOS days: {len(days)}')
    print(f'Grid: {len(TP_SL_PAIRS)} TP/SL × {len(THRESHOLDS)} thresholds × '
          f'{len(CADENCES)} cadences = {len(TP_SL_PAIRS)*len(THRESHOLDS)*len(CADENCES)} combos')

    all_rows = []
    combos = [(tp, sl, thr, cad)
              for (tp, sl) in TP_SL_PAIRS
              for thr in THRESHOLDS
              for cad in CADENCES]
    for i, (tp, sl, thr, cad) in enumerate(combos):
        print(f'\n[{i+1}/{len(combos)}] TP={tp} SL={sl} T={thr} cad={cad}')
        row = run_one(days, thr, cad, tp, sl, COST_PER_TRADE, MAX_HOLD_BARS)
        if row is None:
            print(f'  (no trades)')
            continue
        print(f'  n={row["n_trades"]}  /day={row["trades_per_day"]:.1f}  '
              f'$/day=${row["mean_per_day_net"]:+.1f} CI [{row["pnl_day_ci_lo"]:+.0f},'
              f'{row["pnl_day_ci_hi"]:+.0f}]  $/t=${row["mean_per_trade_net"]:+.2f}  '
              f'DayWR={row["day_wr_active"]:.3f}  MAE=${row["mean_mae"]:.1f}  '
              f'score={row["composite_score"]:.1f}')
        print(f'  TP%={row["tp_pct"]:.0f}  SL%={row["sl_pct"]:.0f}  '
              f'TS%={row["ts_pct"]:.0f}  EOD%={row["eod_pct"]:.0f}')
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / 'dir_clf_kpi_search_v2_asymR.csv', index=False)
    print(f'\nWrote: {OUT_DIR / "dir_clf_kpi_search_v2_asymR.csv"}')

    # Top 10 by composite score
    top = df.sort_values('composite_score', ascending=False).head(10)
    print(f'\n=== TOP 10 BY COMPOSITE SCORE (target $100/day, low MAE, high DayWR) ===')
    cols = ['threshold','cadence','tp_usd','sl_usd','n_trades','trades_per_day',
            'mean_per_trade_net','mean_per_day_net','pnl_day_ci_lo','pnl_day_ci_hi',
            'day_wr_active','mean_mae','tp_pct','sl_pct','composite_score']
    print(top[cols].to_string(index=False))

    # Also: top 10 by $/day alone, low-MAE constrained
    hi_dayWR = df[df['day_wr_active'] >= 0.6].sort_values('mean_per_day_net', ascending=False).head(10)
    print(f'\n=== TOP 10 BY $/DAY among DayWR >= 0.6 ===')
    print(hi_dayWR[cols].to_string(index=False))

    md_lines = ['# Direction-classifier KPI search — OOS 2026',
                f'\n**Grid**: {len(combos)} combos.  Cost: ${COST_PER_TRADE}/trade. '
                f'Time stop: {MAX_HOLD_BARS} bars (30min).',
                f'\nKPIs: $100/day target, low MAE, high Day WR.',
                f'\n## Top 10 by composite score\n',
                '```',
                top[cols].to_string(index=False),
                '```',
                f'\n## Top 10 by $/day among DayWR >= 0.6\n',
                '```',
                hi_dayWR[cols].to_string(index=False) if len(hi_dayWR) else '(none)',
                '```',
                ]
    (OUT_DIR / 'dir_clf_kpi_top.md').write_text('\n'.join(md_lines))
    print(f'\nWrote: {OUT_DIR / "dir_clf_kpi_top.md"}')


if __name__ == '__main__':
    main()
