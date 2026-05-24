"""
Forward pass — RM zigzag pivots with Cycle 1's corrected direction rule.

Cycle 2 of RM Pivot research. Ref:
  research/rm_pivot/cycle_02.md
  research/rm_pivot/cycle_01.md (signal verification)

Rules (v1 — simplest honest test):
  - Entry: confirmed RM zigzag pivot at R=$4 (60-bar rolling OLS on 1m closes)
  - Direction: LOW pivot → LONG · HIGH pivot → SHORT  (trade the pivot)
  - Exit: next RM pivot confirms (opposite type by definition)
  - No stop-loss, no take-profit, no max-hold, one position at a time
  - EOD force-close at 20:55 UTC
  - Trade price = 1m close at pivot confirmation bar (1s slippage deferred to Control)

Outputs:
  training_RM_physics/output/trades/rm_is.pkl   (trades list, day-tagged)
  training_RM_physics/output/trades/rm_oos.pkl
  research/rm_pivot/findings/YYYY-MM-DD_fwd_pass_rm_pivot.md  (report)
  research/rm_pivot/findings/YYYY-MM-DD_fwd_pass_rm_pivot.png (chart)

Reproduction:
  python tools/forward_pass_rm_pivot.py
"""
import os
import sys
import glob
import pickle
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Paths ──────────────────────────────────────────────────────────────
ATLAS_1M_DIR = 'DATA/ATLAS/1m'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
OUTPUT_DIR = 'training_RM_physics/output/trades'
FINDINGS_DIR = 'research/rm_pivot/findings'

# ── Constants (Cycle 1 standardized) ───────────────────────────────────
REG_WINDOW = 60
R_DOLLARS = 4.0
DOLLAR_PER_POINT = 2.0
R_POINTS = R_DOLLARS / DOLLAR_PER_POINT
EOD_UTC_SEC = 20 * 3600 + 55 * 60   # 20:55 UTC (5 min before MNQ maint)
RES_COL = '1m_z_se'

# ── Trade-profitability floor (per user instruction) ───────────────────
NET_FLOOR_DOLLARS = 10.0  # trades below this are "noise", above are "real"


# ══════════════════════════════════════════════════════════════════════
# Math
# ══════════════════════════════════════════════════════════════════════

def rolling_rm(closes, window=REG_WINDOW):
    n = len(closes)
    rm = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)
    xm = x.mean()
    dx = x - xm
    denom = float((dx * dx).sum())
    if denom < 1e-12:
        return rm
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        ym = y.mean()
        slope = float((dx * (y - ym)).sum() / denom)
        intercept = ym - slope * xm
        rm[i] = intercept + slope * (window - 1)
    return rm


def zigzag_confirmations(series, r_points):
    """Live-safe zigzag. Returns [(confirm_idx, pivot_type), ...]."""
    out = []
    leg_dir = None
    extreme_val = None
    for i in range(len(series)):
        v = series[i]
        if np.isnan(v):
            continue
        if extreme_val is None:
            extreme_val = v
            continue
        if leg_dir is None:
            if v - extreme_val >= r_points:
                leg_dir = 'up'
                extreme_val = v
            elif extreme_val - v >= r_points:
                leg_dir = 'down'
                extreme_val = v
            else:
                extreme_val = v
        elif leg_dir == 'up':
            if v > extreme_val:
                extreme_val = v
            elif extreme_val - v >= r_points:
                out.append((i, 'HIGH'))
                leg_dir = 'down'
                extreme_val = v
        else:
            if v < extreme_val:
                extreme_val = v
            elif v - extreme_val >= r_points:
                out.append((i, 'LOW'))
                leg_dir = 'up'
                extreme_val = v
    return out


def seconds_past_midnight(ts):
    return int(ts) % 86400


# ══════════════════════════════════════════════════════════════════════
# Per-day simulator
# ══════════════════════════════════════════════════════════════════════

def simulate_day(day_path):
    day = os.path.basename(day_path).replace('.parquet', '')
    feat_path = os.path.join(FEATURES_5S_DIR, f'{day}.parquet')
    if not os.path.exists(feat_path):
        return [], day

    df_min = (pd.read_parquet(day_path)
              .sort_values('timestamp').reset_index(drop=True))
    closes = df_min['close'].values.astype(np.float64)
    ts_1m = df_min['timestamp'].values.astype(np.int64)

    df_feat = (pd.read_parquet(feat_path)
               .sort_values('timestamp').reset_index(drop=True))
    if RES_COL in df_feat.columns:
        res_ts = df_feat['timestamp'].values.astype(np.int64)
        res_vals = df_feat[RES_COL].values.astype(np.float64)
    else:
        res_ts = np.array([], dtype=np.int64)
        res_vals = np.array([], dtype=np.float64)

    rm = rolling_rm(closes, REG_WINDOW)
    confs = zigzag_confirmations(rm, R_POINTS)

    trades = []
    open_pos = None   # dict when a position is open, else None

    for conf_idx, ptype in confs:
        ts = int(ts_1m[conf_idx])
        # Block new entries at/after EOD
        if seconds_past_midnight(ts) >= EOD_UTC_SEC:
            # If there's an open position, force-close on this pivot bar
            # (EOD rule takes precedence even over the normal exit)
            if open_pos is not None:
                _close(open_pos, closes[conf_idx], ts, 'EOD',
                        res_ts, res_vals, trades, day)
                open_pos = None
            break

        # If holding, this pivot is our exit signal
        if open_pos is not None:
            # Sanity: the exit pivot should be opposite type of entry
            # (zigzag alternates), so LONG exits on HIGH, SHORT on LOW.
            exit_reason = 'RM_PIVOT' if (
                (open_pos['direction'] == 'long' and ptype == 'HIGH') or
                (open_pos['direction'] == 'short' and ptype == 'LOW')
            ) else 'RM_PIVOT_OFF'   # shouldn't happen with alternating zigzag
            _close(open_pos, closes[conf_idx], ts, exit_reason,
                    res_ts, res_vals, trades, day)
            open_pos = None
            # Do NOT also open on the same pivot — v1 waits for next one
            # so we never chain the same bar. (Simplification; tune later.)
            continue

        # No position: enter in pivot direction
        direction = 'long' if ptype == 'LOW' else 'short'
        residual_at_entry = _residual_at(res_ts, res_vals, ts)
        open_pos = {
            'direction': direction,
            'entry_price': closes[conf_idx],
            'entry_ts': ts,
            'entry_conf_idx': conf_idx,
            'entry_pivot_type': ptype,
            'entry_residual': residual_at_entry,
        }

    # Day close: if still holding, close at last 1m close
    if open_pos is not None:
        _close(open_pos, closes[-1], int(ts_1m[-1]), 'day_end',
                res_ts, res_vals, trades, day)
        open_pos = None

    return trades, day


def _residual_at(res_ts, res_vals, target_ts):
    if len(res_ts) == 0:
        return float('nan')
    idx = np.searchsorted(res_ts, target_ts, side='right') - 1
    if idx < 0:
        return float('nan')
    return float(res_vals[idx])


def _close(pos, exit_price, ts, reason, res_ts, res_vals, trades, day):
    pnl_points = (exit_price - pos['entry_price']) if pos['direction'] == 'long' \
                 else (pos['entry_price'] - exit_price)
    pnl = pnl_points * DOLLAR_PER_POINT
    held_min = max(0, (ts - pos['entry_ts']) // 60)
    exit_residual = _residual_at(res_ts, res_vals, ts)
    trades.append({
        'day': day,
        'dir': pos['direction'],
        'entry_tier': 'RM_PIVOT',
        'timestamp': pos['entry_ts'],
        'entry_ts': pos['entry_ts'],
        'exit_ts': ts,
        'entry_price': pos['entry_price'],
        'exit_price': exit_price,
        'pnl': pnl,
        'held': int(held_min),
        'exit_reason': reason,
        'entry_pivot_type': pos['entry_pivot_type'],
        'entry_residual': pos['entry_residual'],
        'exit_residual': exit_residual,
    })


# ══════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════

def pnl_ratio_wr(trades):
    """Trade WR per user spec: (∑profit/|∑loss|) − 1. 0=BE, +=win, -=loss."""
    pnls = np.array([t['pnl'] for t in trades])
    profit = float(pnls[pnls > 0].sum())
    loss = float(abs(pnls[pnls < 0].sum()))
    if loss < 1e-9:
        return float('inf') if profit > 0 else 0.0
    return profit / loss - 1


def mode_bucket(values, step=25):
    """Return (bucket_lo, bucket_hi, n_in_bucket, pct) for the modal bucket."""
    if len(values) == 0:
        return None
    arr = np.asarray(values, dtype=np.float64)
    bins = np.floor(arr / step) * step
    u, c = np.unique(bins, return_counts=True)
    idx = int(c.argmax())
    lo = float(u[idx])
    return (lo, lo + step, int(c[idx]), c[idx] / len(arr) * 100)


def summarize_cohort(trades, label):
    """Return a dict of all metrics for one cohort (IS or OOS)."""
    if not trades:
        return None
    n = len(trades)
    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    days = defaultdict(float)
    for t in trades:
        days[t['day']] += t['pnl']
    daily_pnls = np.array(list(days.values()))
    daily_wins = (daily_pnls > 0).sum()
    daily_total = len(daily_pnls)

    # Trade-profitability floor
    net_floor_trades = pnls[pnls >= NET_FLOOR_DOLLARS]
    n_above_floor = int((pnls >= NET_FLOOR_DOLLARS).sum())
    n_below_neg_floor = int((pnls <= -NET_FLOOR_DOLLARS).sum())

    # Mode buckets
    trade_mode = mode_bucket(pnls, step=5)     # $5 bucket for trades
    daily_mode = mode_bucket(daily_pnls, step=25)  # $25 bucket for days

    return {
        'label': label,
        'n_trades': n,
        'n_days': daily_total,
        'trades_per_day': n / max(daily_total, 1),
        'total_pnl': float(pnls.sum()),
        'mean_pnl_per_trade': float(pnls.mean()),
        'median_pnl_per_trade': float(np.median(pnls)),
        'trade_mode': trade_mode,
        'trade_wr_pnl_ratio': pnl_ratio_wr(trades),
        'count_win_ratio': len(wins) / n,
        'n_above_10_net': n_above_floor,
        'n_below_neg10': n_below_neg_floor,
        'pct_above_10': n_above_floor / n * 100,
        'gross_win': float(wins.sum()),
        'gross_loss': float(losses.sum()),
        'largest_win': float(wins.max()) if len(wins) else 0.0,
        'largest_loss': float(losses.min()) if len(losses) else 0.0,
        'mean_hold_min': float(np.mean([t['held'] for t in trades])),
        'median_hold_min': float(np.median([t['held'] for t in trades])),
        'daily_total': float(daily_pnls.sum()),
        'daily_mean': float(daily_pnls.mean()),
        'daily_median': float(np.median(daily_pnls)),
        'daily_std': float(daily_pnls.std()),
        'daily_p25': float(np.percentile(daily_pnls, 25)),
        'daily_p75': float(np.percentile(daily_pnls, 75)),
        'daily_p5': float(np.percentile(daily_pnls, 5)),
        'daily_p95': float(np.percentile(daily_pnls, 95)),
        'daily_min': float(daily_pnls.min()),
        'daily_max': float(daily_pnls.max()),
        'day_win_pct': daily_wins / max(daily_total, 1) * 100,
        'daily_mode': daily_mode,
        'daily_pnls': daily_pnls,
        'pnls': pnls,
    }


def render_report(is_sum, oos_sum, n_is_days, n_oos_days):
    lines = []
    date_tag = datetime.now().strftime('%Y-%m-%d')
    lines.append(f'# RM Pivot Forward Pass — Cycle 02')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f'Ref: `research/rm_pivot/cycle_02.md`')
    lines.append('')
    lines.append('## Rules')
    lines.append('')
    lines.append(f'- Entry: confirmed RM zigzag pivot at R=${R_DOLLARS:.0f} (60-bar rolling OLS on 1m closes)')
    lines.append(f'- Direction: LOW pivot → LONG, HIGH pivot → SHORT')
    lines.append(f'- Exit: next RM pivot confirms')
    lines.append(f'- EOD force-close at 20:55 UTC; no stop-loss, no take-profit, no max-hold')
    lines.append(f'- One position at a time (no chains in v1)')
    lines.append(f'- PnL: 1m close at pivot confirmation × $2/pt (1s slippage deferred to Control)')
    lines.append('')

    def _section(title, s):
        if s is None:
            lines.append(f'## {title}')
            lines.append('No trades.')
            lines.append('')
            return
        lines.append(f'## {title}')
        lines.append('')
        lines.append(f'**Trades**: {s["n_trades"]:,} over {s["n_days"]} days ({s["trades_per_day"]:.1f}/day)')
        lines.append(f'**Total PnL**: ${s["total_pnl"]:+,.0f}  ({s["daily_mean"]:+.2f}/day mean)')
        lines.append('')
        lines.append('### Daily distribution')
        lines.append('')
        lines.append(f'| mean | median | p5 | p25 | p75 | p95 | min | max | std | DayWR |')
        lines.append(f'|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        lines.append(
            f'| ${s["daily_mean"]:+,.0f} | ${s["daily_median"]:+,.0f} '
            f'| ${s["daily_p5"]:+,.0f} | ${s["daily_p25"]:+,.0f} '
            f'| ${s["daily_p75"]:+,.0f} | ${s["daily_p95"]:+,.0f} '
            f'| ${s["daily_min"]:+,.0f} | ${s["daily_max"]:+,.0f} '
            f'| ${s["daily_std"]:,.0f} | **{s["day_win_pct"]:.0f}%** |'
        )
        if s['daily_mode'] is not None:
            lo, hi, n, pct = s['daily_mode']
            lines.append('')
            lines.append(f'**Daily mode bucket**: [${lo:+,.0f}..${hi:+,.0f}) — {n} days ({pct:.0f}%)')
        lines.append('')
        lines.append('### Per-trade')
        lines.append('')
        lines.append(f'| mean | median | count-WR | **Trade-WR (PnL ratio)** | %≥$10 net | %≤-$10 |')
        lines.append(f'|---:|---:|---:|---:|---:|---:|')
        lines.append(
            f'| ${s["mean_pnl_per_trade"]:+,.2f} | ${s["median_pnl_per_trade"]:+,.2f} '
            f'| {s["count_win_ratio"]*100:.0f}% | **{s["trade_wr_pnl_ratio"]:+.2f}** '
            f'| {s["pct_above_10"]:.0f}% | {100 - s["pct_above_10"] - (s["n_below_neg10"]/s["n_trades"]*100):.0f}% '
            f'(noise band) |'
        )
        if s['trade_mode'] is not None:
            lo, hi, n, pct = s['trade_mode']
            lines.append('')
            lines.append(f'**Trade mode bucket (\\$5)**: [${lo:+,.0f}..${hi:+,.0f}) — {n} trades ({pct:.0f}%)')
        lines.append(f'**Gross win**: ${s["gross_win"]:+,.0f}  |  **Gross loss**: ${s["gross_loss"]:+,.0f}')
        lines.append(f'**Largest win/loss**: ${s["largest_win"]:+,.0f} / ${s["largest_loss"]:+,.0f}')
        lines.append(f'**Hold**: mean {s["mean_hold_min"]:.0f} min, median {s["median_hold_min"]:.0f} min')
        lines.append('')

    _section('IS (2025)', is_sum)
    _section('OOS (2026)', oos_sum)

    lines.append('## Comparison vs success criteria (from cycle_02.md PLAN)')
    lines.append('')
    def _check(label, pred, actual, op='>='):
        if actual is None:
            return f'| {label} | {pred} | — | — |'
        try:
            cmp_val = float(actual)
        except (TypeError, ValueError):
            return f'| {label} | {pred} | {actual} | — |'
        if op == '>=':
            hit = cmp_val >= pred
        elif op == '<=':
            hit = cmp_val <= pred
        else:
            hit = False
        return f'| {label} | {op} {pred} | {cmp_val:+.2f} | {"✓" if hit else "✗"} |'

    lines.append('| Metric | Gate | Actual (IS) | Pass |')
    lines.append('|---|---|---:|---:|')
    if is_sum:
        lines.append(_check('Trades/day', 20, is_sum['trades_per_day']))
        lines.append(_check('Trade WR (PnL ratio)', 0.5, is_sum['trade_wr_pnl_ratio']))
        mode_lo = is_sum['trade_mode'][0] if is_sum['trade_mode'] else None
        lines.append(_check('Mode $/trade bucket low', 10, mode_lo if mode_lo is not None else -999))
        lines.append(_check('Mean $/trade', 10, is_sum['mean_pnl_per_trade']))
        lines.append(_check('Daily mean PnL', 200, is_sum['daily_mean']))
        lines.append(_check('Daily median PnL', 100, is_sum['daily_median']))
        lines.append(_check('Day WR (%)', 60, is_sum['day_win_pct']))
    lines.append('')
    if oos_sum and is_sum and is_sum['daily_mean'] > 1:
        ratio = oos_sum['daily_mean'] / is_sum['daily_mean']
        lines.append(f'**OOS/IS daily-mean ratio**: {ratio:.2f} (gate ≥ 0.70)')
        lines.append('')

    lines.append('## Reproduction')
    lines.append('')
    lines.append('```')
    lines.append('python tools/forward_pass_rm_pivot.py')
    lines.append('```')
    lines.append('')
    return '\n'.join(lines)


def plot_distributions(is_sum, oos_sum, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: IS daily PnL distribution
    ax = axes[0][0]
    if is_sum:
        ax.hist(is_sum['daily_pnls'], bins=40, color='tab:blue', alpha=0.7,
                edgecolor='black', linewidth=0.3)
        ax.axvline(0, color='black', linewidth=0.6)
        ax.axvline(is_sum['daily_mean'], color='tab:orange', linestyle='--',
                   label=f'mean ${is_sum["daily_mean"]:+,.0f}')
        ax.axvline(is_sum['daily_median'], color='tab:green', linestyle='--',
                   label=f'median ${is_sum["daily_median"]:+,.0f}')
        ax.axvline(300, color='purple', linestyle=':', label='$300 target')
        ax.set_title(f'IS daily PnL (n={is_sum["n_days"]}, DayWR={is_sum["day_win_pct"]:.0f}%)')
        ax.set_xlabel('PnL ($)')
        ax.set_ylabel('Days')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Panel 2: OOS daily
    ax = axes[0][1]
    if oos_sum:
        ax.hist(oos_sum['daily_pnls'], bins=30, color='tab:orange', alpha=0.7,
                edgecolor='black', linewidth=0.3)
        ax.axvline(0, color='black', linewidth=0.6)
        ax.axvline(oos_sum['daily_mean'], color='tab:blue', linestyle='--',
                   label=f'mean ${oos_sum["daily_mean"]:+,.0f}')
        ax.axvline(oos_sum['daily_median'], color='tab:green', linestyle='--',
                   label=f'median ${oos_sum["daily_median"]:+,.0f}')
        ax.axvline(300, color='purple', linestyle=':', label='$300 target')
        ax.set_title(f'OOS daily PnL (n={oos_sum["n_days"]}, DayWR={oos_sum["day_win_pct"]:.0f}%)')
        ax.set_xlabel('PnL ($)')
        ax.set_ylabel('Days')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Panel 3: IS per-trade PnL
    ax = axes[1][0]
    if is_sum:
        arr = is_sum['pnls']
        ax.hist(arr, bins=np.arange(-200, 200, 5), color='tab:blue', alpha=0.7,
                edgecolor='black', linewidth=0.2)
        ax.axvline(0, color='black', linewidth=0.6)
        ax.axvline(10, color='tab:green', linestyle='--', label='$10 floor')
        ax.axvline(-10, color='tab:red', linestyle='--', label='-$10 cap')
        ax.set_title(f'IS per-trade PnL (n={is_sum["n_trades"]}, mean ${is_sum["mean_pnl_per_trade"]:+.2f})')
        ax.set_xlabel('Trade PnL ($)')
        ax.set_ylabel('Trades')
        ax.set_xlim(-200, 200)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Panel 4: IS equity curve
    ax = axes[1][1]
    if is_sum:
        pnls = is_sum['daily_pnls']
        ax.plot(np.cumsum(pnls), color='tab:blue', linewidth=1.2)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_title(f'IS cumulative PnL (total ${pnls.sum():+,.0f})')
        ax.set_xlabel('Day #')
        ax.set_ylabel('Cumulative $')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    global R_DOLLARS, R_POINTS
    ap = argparse.ArgumentParser()
    ap.add_argument('--r', type=float, default=R_DOLLARS,
                    help='RM zigzag retracement in $ (default 4.0)')
    ap.add_argument('--sweep', action='store_true',
                    help='Sweep R ∈ {2, 4, 6, 10, 15, 20} on IS only, print table')
    ap.add_argument('--no-save', action='store_true',
                    help='Skip writing pickles (for sweep)')
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    print(f'IS: {len(is_paths)} days  |  OOS: {len(oos_paths)} days')

    if args.sweep:
        print()
        print(f'{"R":>4} {"N":>6} {"$/d":>8} {"$/tr mean":>9} {"$/tr med":>9} '
              f'{"tradeWR":>8} {"DayWR":>6} {"pct>=10":>7} {"hold_med":>8}')
        print('-' * 78)
        for r in [2.0, 4.0, 6.0, 10.0, 15.0, 20.0]:
            R_DOLLARS = r
            R_POINTS = r / DOLLAR_PER_POINT
            trades = []
            for p in tqdm(is_paths, desc=f'R=${r:.0f}', unit='day', leave=False):
                t, _ = simulate_day(p)
                trades.extend(t)
            s = summarize_cohort(trades, f'R=${r:.0f}')
            if s is None:
                continue
            print(f'${r:>3.0f} {s["n_trades"]:>6} ${s["daily_mean"]:>+7.0f} '
                  f'${s["mean_pnl_per_trade"]:>+8.2f} '
                  f'${s["median_pnl_per_trade"]:>+8.2f} '
                  f'{s["trade_wr_pnl_ratio"]:>+7.2f} '
                  f'{s["day_win_pct"]:>5.0f}% '
                  f'{s["pct_above_10"]:>5.0f}% '
                  f'{s["median_hold_min"]:>7.0f}m')
        return

    R_DOLLARS = args.r
    R_POINTS = args.r / DOLLAR_PER_POINT

    def _run(paths, label):
        all_trades = []
        day_count = 0
        for p in tqdm(paths, desc=label, unit='day'):
            trades, _ = simulate_day(p)
            if trades:
                all_trades.extend(trades)
            day_count += 1
        return all_trades, day_count

    is_trades, n_is = _run(is_paths, 'IS')
    oos_trades, n_oos = _run(oos_paths, 'OOS')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'rm_is.pkl'), 'wb') as f:
        pickle.dump(is_trades, f)
    with open(os.path.join(OUTPUT_DIR, 'rm_oos.pkl'), 'wb') as f:
        pickle.dump(oos_trades, f)
    print(f'Saved: {OUTPUT_DIR}/rm_is.pkl ({len(is_trades)} trades)')
    print(f'Saved: {OUTPUT_DIR}/rm_oos.pkl ({len(oos_trades)} trades)')

    is_sum = summarize_cohort(is_trades, 'IS')
    oos_sum = summarize_cohort(oos_trades, 'OOS')

    os.makedirs(FINDINGS_DIR, exist_ok=True)
    date_tag = datetime.now().strftime('%Y-%m-%d')
    md_path = os.path.join(FINDINGS_DIR, f'{date_tag}_fwd_pass_rm_pivot.md')
    png_path = os.path.join(FINDINGS_DIR, f'{date_tag}_fwd_pass_rm_pivot.png')

    report = render_report(is_sum, oos_sum, n_is, n_oos)
    # Append chart ref
    report += f'\n## Chart\n\n![distributions]({os.path.basename(png_path)})\n'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    plot_distributions(is_sum, oos_sum, png_path)

    # One-line screen summary
    print()
    if is_sum:
        print(f'IS: ${is_sum["total_pnl"]:+,.0f} / {n_is}d = ${is_sum["daily_mean"]:+.0f}/d '
              f'| DayWR {is_sum["day_win_pct"]:.0f}% | tradeWR {is_sum["trade_wr_pnl_ratio"]:+.2f} '
              f'| $/tr mean ${is_sum["mean_pnl_per_trade"]:+.2f}')
    if oos_sum:
        print(f'OOS: ${oos_sum["total_pnl"]:+,.0f} / {n_oos}d = ${oos_sum["daily_mean"]:+.0f}/d '
              f'| DayWR {oos_sum["day_win_pct"]:.0f}% | tradeWR {oos_sum["trade_wr_pnl_ratio"]:+.2f} '
              f'| $/tr mean ${oos_sum["mean_pnl_per_trade"]:+.2f}')
    print(f'Wrote: {md_path}')
    print(f'Wrote: {png_path}')


if __name__ == '__main__':
    main()
