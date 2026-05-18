"""Direction-classifier threshold sweep through the V2 iso engine.

For each (threshold, fire_cadence) the strategy is rebuilt and the full
day-stream is replayed through engine + default_exit_suite. This is the
HONEST forward pass — realistic SL/TP/timeout/zse exits, no oracle MFE
assumption.

Output:
  reports/findings/regret_oracle/dir_clf_sweep_{target}.csv
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from training_iso_v2.ticker import MultiDayV2Ticker
from training_iso_v2.exits import default_exit_suite, HardStop, TakeProfit, TimeStop
from training_iso_v2.iso_orchestrator import IsoOrchestrator
from training_iso_v2.strategies import DirectionClassifierStrategy
from training_iso_v2.v2_cols import swing_noise_w, reversion_prob_w, z_se_w
from training_iso_v2.run_iso import _resolve_days, _histogram_mode, _pf_trade_wr


def clean_exit_suite(tp_usd: float = 20.0, sl_usd: float = 20.0,
                     max_hold_bars: int = 360):
    """Clean exit suite for direction-only strategies.

    Just three rules — no thesis exits, no giveback. Order matters:
    SL checks before TP so adverse moves cap first; TimeStop is the
    longest-leash backstop.
    """
    return [
        HardStop(usd=-abs(sl_usd)),
        TakeProfit(usd=abs(tp_usd)),
        TimeStop(max_bars=max_hold_bars),
    ]


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


def run_one(target, thresholds, cadences, days, tp_usd: float = 20.0,
            sl_usd: float = 20.0, max_hold_bars: int = 360,
            exit_kind: str = 'clean', cost_per_trade: float = 4.0):
    rows = []
    for cadence in cadences:
        for thr in thresholds:
            strat = DirectionClassifierStrategy(threshold=thr, fire_cadence=cadence)
            if exit_kind == 'clean':
                exits = clean_exit_suite(tp_usd=tp_usd, sl_usd=sl_usd,
                                         max_hold_bars=max_hold_bars)
            else:
                exits = default_exit_suite()
            orch = IsoOrchestrator(
                strategies=[strat],
                exits=exits,
                entry_extras_hook=_entry_extras,
            )
            multi = MultiDayV2Ticker(days=days)
            per_tier = orch.run(tqdm(multi, desc=f'{target} cad={cadence} T={thr:.2f}',
                                      total=20000))
            trades = per_tier.get('DIRECTION_CLF', [])
            n = len(trades)
            if n == 0:
                rows.append({
                    'target': target, 'cadence': cadence, 'threshold': thr,
                    'n_trades': 0, 'n_active_days': 0, 'n_total_days': len(days),
                    'mean_per_trade': 0.0, 'mode_per_trade': 0.0,
                    'mean_per_day': 0.0, 'pnl_day_ci_lo': 0.0, 'pnl_day_ci_hi': 0.0,
                    'mode_per_day': 0.0, 'median_per_day': 0.0,
                    'day_wr_active': 0.0, 'day_wr_total': 0.0,
                    'pf_trade_wr': 0.0,
                })
                continue
            # Gross and NET (after commission/slippage) PnL
            gross_pnls = np.array([t.pnl for t in trades], dtype=np.float64)
            mae_per_trade = np.array([t.trough_pnl for t in trades], dtype=np.float64)
            peak_per_trade = np.array([t.peak_pnl for t in trades], dtype=np.float64)
            pnls = gross_pnls - cost_per_trade   # round-trip cost subtracted
            day_pnl = defaultdict(float)
            day_pnl_gross = defaultdict(float)
            for t, p, gp in zip(trades, pnls, gross_pnls):
                day_pnl[t.entry_day] += p
                day_pnl_gross[t.entry_day] += gp
            active_days = list(day_pnl.values())
            mean_d, lo_d, hi_d = bootstrap_ci_mean(active_days)
            mode_per_trade = _histogram_mode(pnls, bin_width=2.0)
            mode_per_day = _histogram_mode(np.array(active_days), bin_width=25.0)
            n_active = len(active_days)
            n_total = len(days)
            win_days = sum(1 for v in active_days if v > 0)
            row = {
                'target': target, 'cadence': cadence, 'threshold': thr,
                'n_trades': int(n), 'n_active_days': int(n_active),
                'n_total_days': int(n_total),
                'mean_per_trade_gross': float(gross_pnls.mean()),
                'mean_per_trade_net': float(pnls.mean()),
                'mode_per_trade': mode_per_trade,
                'mean_per_day': mean_d,
                'pnl_day_ci_lo': lo_d, 'pnl_day_ci_hi': hi_d,
                'mode_per_day': mode_per_day,
                'median_per_day': float(np.median(active_days)),
                'day_wr_active': float(win_days / max(n_active, 1)),
                'day_wr_total': float(win_days / max(n_total, 1)),
                'pf_trade_wr_net': _pf_trade_wr(pnls),
                'pf_trade_wr_gross': _pf_trade_wr(gross_pnls),
                'total_pnl_net': float(pnls.sum()),
                'total_pnl_gross': float(gross_pnls.sum()),
                'cost_per_trade': cost_per_trade,
                'tp_usd': tp_usd, 'sl_usd': sl_usd, 'exit_kind': exit_kind,
                'mean_mae': float(mae_per_trade.mean()),
                'median_mae': float(np.median(mae_per_trade)),
                'p90_mae': float(np.percentile(mae_per_trade, 10)),  # 10th percentile = worst 10%
                'mean_peak': float(peak_per_trade.mean()),
            }
            # Reason breakdown
            from collections import Counter
            reasons = Counter(t.exit_reason for t in trades)
            row['reason_breakdown'] = dict(reasons)
            print(f'\n  cadence={cadence} T={thr:.2f}  n={n:5d}  trades/day={n/max(n_total,1):.1f}')
            print(f'    $/trade GROSS=${row["mean_per_trade_gross"]:+6.2f}  '
                  f'NET=${row["mean_per_trade_net"]:+6.2f} (cost=${cost_per_trade})')
            print(f'    $/day NET=${mean_d:+8.1f} CI [{lo_d:+.0f},{hi_d:+.0f}]  '
                  f'mode=${mode_per_day:+.0f}  '
                  f'PF_WR_net={row["pf_trade_wr_net"]:+.3f}  '
                  f'DayWR={row["day_wr_active"]:.3f}')
            print(f'    MAE mean=${row["mean_mae"]:+.1f}  median=${row["median_mae"]:+.1f}  '
                  f'worst10%=${row["p90_mae"]:+.1f}  peak mean=${row["mean_peak"]:+.1f}')
            print(f'    exit reasons: {dict(reasons)}')
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thresholds', nargs='+', type=float,
                    default=[0.55, 0.65, 0.75, 0.85])
    ap.add_argument('--cadences', nargs='+',
                    default=['1m'],
                    help='Bar cadences: 5s, 15s, 1m, 5m, 15m')
    ap.add_argument('--targets', nargs='+', default=['oos'],
                    help='is, oos, or both')
    ap.add_argument('--out-dir', default='reports/findings/regret_oracle')
    ap.add_argument('--tp', type=float, default=20.0, help='TP in $')
    ap.add_argument('--sl', type=float, default=20.0, help='SL in $ (will be made negative)')
    ap.add_argument('--max-hold-bars', type=int, default=360,
                    help='Time stop in 5s bars (360 = 30 min)')
    ap.add_argument('--exit-kind', choices=['clean', 'default'], default='clean',
                    help='clean=TP+SL+TimeStop only; default=full thesis suite')
    ap.add_argument('--cost', type=float, default=4.0,
                    help='Round-trip cost per trade ($, commission + slippage)')
    ap.add_argument('--label', default='', help='Optional label for output csv')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f'_{args.label}' if args.label else f'_TP{int(args.tp)}_SL{int(args.sl)}_{args.exit_kind}'
    all_rows = []
    for target in args.targets:
        days = _resolve_days(target)
        print(f'\n=== Target {target}: {len(days)} days   '
              f'exit={args.exit_kind} TP=${args.tp} SL=${args.sl} '
              f'max_hold={args.max_hold_bars} bars ===')
        rows = run_one(target, args.thresholds, args.cadences, days,
                       tp_usd=args.tp, sl_usd=args.sl,
                       max_hold_bars=args.max_hold_bars,
                       exit_kind=args.exit_kind,
                       cost_per_trade=args.cost)
        all_rows.extend(rows)
        df = pd.DataFrame(rows)
        out_path = out_dir / f'dir_clf_sweep_{target}{suffix}.csv'
        df.to_csv(out_path, index=False)
        print(f'\nWrote: {out_path}')

    combo = pd.DataFrame(all_rows)
    combo_path = out_dir / f'dir_clf_sweep_combined{suffix}.csv'
    combo.to_csv(combo_path, index=False)
    print(f'Wrote: {combo_path}')


if __name__ == '__main__':
    main()
