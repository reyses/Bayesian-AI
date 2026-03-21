"""
DMI Reversal Simulator — Multi-Timeframe
==========================================
Computes DMI (DI+/DI-/ADX) on ALL available timeframes simultaneously,
aligns them to the 15s execution clock, and tests multi-TF reversal
detection strategies.

Key question: does requiring DMI agreement across multiple TFs produce
better reversal detection than single-TF (1m) alone?

Multi-TF reversal strategies tested:
  - SINGLE:   1m DI cross + gap (current exit_engine.py behavior)
  - ANY:      crossover on ANY TF with gap
  - MAJORITY: >50% of TFs show DI orientation against trade
  - WEIGHTED: TF-weighted orientation score (slow TFs = heavier)
  - CASCADE:  1m cross + at least N slower TFs confirm orientation

Outputs:
  Part 1: Per-TF crossover stats (frequency, accuracy at forward horizons)
  Part 2: Multi-TF orientation heatmap (how often N TFs agree)
  Part 3: Trade simulation comparing single vs multi-TF exit strategies
  Part 4: Optimal strategy recommendation

Usage:
  python tools/dmi_reversal_simulator.py                         # ATLAS_OOS
  python tools/dmi_reversal_simulator.py --data DATA/ATLAS       # full IS
  python tools/dmi_reversal_simulator.py --data DATA/ATLAS_1WEEK # fast test
  python tools/dmi_reversal_simulator.py --gaps 5,10,15,20,25 --min-hold 20
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# ── TF definitions ────────────────────────────────────────────────────────────

# TFs to load + their bar duration in seconds + weight for composite scoring
TF_CONFIG = [
    ('15s',  15,   1.0),
    ('30s',  30,   1.5),
    ('1m',   60,   2.0),
    ('3m',   180,  3.0),
    ('5m',   300,  4.0),
    ('15m',  900,  6.0),
    ('30m',  1800, 8.0),
    ('1h',   3600, 10.0),
]

TICK_SIZE = 0.25
TICK_VALUE = 0.50  # MNQ


# ── DMI computation ──────────────────────────────────────────────────────────

def compute_dmi(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> dict:
    """Compute DI+, DI-, ADX from OHLC arrays using Wilder smoothing."""
    n = len(closes)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        h, l, c_prev = highs[i], lows[i], closes[i - 1]
        tr[i] = max(h - l, abs(h - c_prev), abs(l - c_prev))
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    # Wilder smoothing
    smooth_tr = np.zeros(n)
    smooth_plus = np.zeros(n)
    smooth_minus = np.zeros(n)

    if n <= period:
        return {'di_plus': np.zeros(n), 'di_minus': np.zeros(n),
                'adx': np.zeros(n), 'n_bars': n}

    smooth_tr[period] = np.sum(tr[1:period + 1])
    smooth_plus[period] = np.sum(plus_dm[1:period + 1])
    smooth_minus[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        smooth_tr[i] = smooth_tr[i - 1] - smooth_tr[i - 1] / period + tr[i]
        smooth_plus[i] = smooth_plus[i - 1] - smooth_plus[i - 1] / period + plus_dm[i]
        smooth_minus[i] = smooth_minus[i - 1] - smooth_minus[i - 1] / period + minus_dm[i]

    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    dx = np.zeros(n)

    for i in range(period, n):
        if smooth_tr[i] > 0:
            di_plus[i] = 100.0 * smooth_plus[i] / smooth_tr[i]
            di_minus[i] = 100.0 * smooth_minus[i] / smooth_tr[i]
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum

    adx = np.zeros(n)
    if n > 2 * period:
        adx[2 * period - 1] = np.mean(dx[period:2 * period])
        for i in range(2 * period, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return {'di_plus': di_plus, 'di_minus': di_minus, 'adx': adx, 'n_bars': n}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_tf_data(data_root: str, tf_name: str):
    """Load ATLAS parquet for one TF. Returns DataFrame with OHLC + timestamp."""
    tf_dir = os.path.join(data_root, tf_name)
    if not os.path.isdir(tf_dir):
        return None
    files = sorted(Path(tf_dir).glob('*.parquet'))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return df


def load_all_tfs(data_root: str, period: int = 14):
    """Load all TFs and compute DMI on each. Returns dict of TF data."""
    tf_data = {}
    for tf_name, tf_secs, tf_weight in TF_CONFIG:
        df = load_tf_data(data_root, tf_name)
        if df is None:
            print(f"  {tf_name:>4}: not found, skipping")
            continue

        dmi = compute_dmi(df['high'].values, df['low'].values,
                          df['close'].values, period=period)

        # Convert timestamps to int64 for fast lookup
        ts = df['timestamp'].values.astype('int64')

        tf_data[tf_name] = {
            'df': df,
            'di_plus': dmi['di_plus'],
            'di_minus': dmi['di_minus'],
            'adx': dmi['adx'],
            'closes': df['close'].values,
            'timestamps_i64': ts,
            'n_bars': len(df),
            'tf_secs': tf_secs,
            'tf_weight': tf_weight,
        }
        print(f"  {tf_name:>4}: {len(df):>9,} bars | DMI warmup done at bar {min(2*period, len(df))}")

    return tf_data


# ── Multi-TF alignment ──────────────────────────────────────────────────────

def build_alignment_index(base_ts_i64: np.ndarray, tf_ts_i64: np.ndarray) -> np.ndarray:
    """For each base (15s) timestamp, find the index of the last completed bar
    on the target TF. Returns array of indices (same length as base_ts_i64).
    -1 means no TF bar completed yet."""
    # Binary search: for each base ts, find rightmost tf ts <= base ts
    idx = np.searchsorted(tf_ts_i64, base_ts_i64, side='right') - 1
    idx = np.clip(idx, -1, len(tf_ts_i64) - 1)
    return idx


# ── Per-TF crossover analysis ───────────────────────────────────────────────

def analyze_tf_crossovers(tf_data: dict, tf_name: str, gap_thresholds: list,
                          forward_bars_15s: tuple = (20, 40, 80, 160)):
    """Analyze crossover accuracy for one TF, measuring forward returns on the
    15s price series (consistent comparison across TFs)."""
    d = tf_data[tf_name]
    di_plus = d['di_plus']
    di_minus = d['di_minus']
    n = d['n_bars']
    warmup = 28

    # Find crossovers on this TF
    events = []
    for i in range(warmup, n - 1):
        bull = (di_plus[i - 1] <= di_minus[i - 1] and di_plus[i] > di_minus[i])
        bear = (di_minus[i - 1] <= di_plus[i - 1] and di_minus[i] > di_plus[i])
        if not bull and not bear:
            continue
        gap = abs(di_plus[i] - di_minus[i])
        events.append({
            'bar_idx': i,
            'direction': 'BULL' if bull else 'BEAR',
            'gap': gap,
            'di_plus': di_plus[i],
            'di_minus': di_minus[i],
        })

    # Measure forward returns on 15s close prices if available
    base_d = tf_data.get('15s')
    if base_d is not None and tf_name != '15s':
        base_closes = base_d['closes']
        align = build_alignment_index(base_d['timestamps_i64'], d['timestamps_i64'])
        # Reverse: for each TF bar idx, find corresponding 15s bar idx
        # (first 15s bar where align[j] >= tf_bar_idx)
        tf_to_base = {}
        for j in range(len(align)):
            tf_idx = align[j]
            if tf_idx >= 0 and tf_idx not in tf_to_base:
                tf_to_base[tf_idx] = j

        for e in events:
            base_bar = tf_to_base.get(e['bar_idx'])
            if base_bar is None:
                continue
            entry_price = base_closes[base_bar]
            for fb in forward_bars_15s:
                if base_bar + fb < len(base_closes):
                    move = (base_closes[base_bar + fb] - entry_price) / TICK_SIZE
                    if e['direction'] == 'BEAR':
                        move = -move
                    e[f'fwd_{fb}'] = move
    else:
        # Same TF or no 15s — use own closes
        closes = d['closes']
        for e in events:
            i = e['bar_idx']
            entry_price = closes[i]
            for fb in forward_bars_15s:
                if i + fb < n:
                    move = (closes[i + fb] - entry_price) / TICK_SIZE
                    if e['direction'] == 'BEAR':
                        move = -move
                    e[f'fwd_{fb}'] = move

    # Summarize by gap threshold
    rows = []
    for gap_min in gap_thresholds:
        filtered = [e for e in events if e['gap'] >= gap_min]
        cnt = len(filtered)
        if cnt == 0:
            rows.append({'gap_min': gap_min, 'n': 0})
            continue
        row = {'gap_min': gap_min, 'n': cnt}
        for fb in forward_bars_15s:
            key = f'fwd_{fb}'
            vals = [e[key] for e in filtered if key in e and not np.isnan(e.get(key, np.nan))]
            if vals:
                row[f'acc_{fb}'] = sum(1 for v in vals if v > 0) / len(vals) * 100
                row[f'avg_{fb}'] = np.mean(vals)
            else:
                row[f'acc_{fb}'] = 0.0
                row[f'avg_{fb}'] = 0.0
        rows.append(row)

    return events, rows


# ── Multi-TF composite signal ──────────────────────────────────────────────

def compute_multi_tf_orientation(tf_data: dict, align_indices: dict,
                                 bar_15s: int, side: str):
    """At a given 15s bar, compute multi-TF DI orientation against `side`.

    Returns:
      n_against: number of TFs with DI oriented against the trade
      n_active:  number of TFs with valid DMI data
      weighted_score: TF-weight-normalized score (0..1, higher = more reversal pressure)
      crossed_tfs: list of TF names that had a crossover against on THIS bar
      details: per-TF orientation dict
    """
    n_against = 0
    n_active = 0
    weight_against = 0.0
    weight_total = 0.0
    crossed_tfs = []
    details = {}

    for tf_name, tf_secs, tf_weight in TF_CONFIG:
        if tf_name not in tf_data or tf_name not in align_indices:
            continue

        tf_idx = align_indices[tf_name][bar_15s]
        if tf_idx < 28:  # DMI warmup
            continue

        d = tf_data[tf_name]
        dip = d['di_plus'][tf_idx]
        dim = d['di_minus'][tf_idx]

        if dip == 0 and dim == 0:
            continue

        n_active += 1
        weight_total += tf_weight

        # DI orientation against trade
        if side == 'long':
            against = dim > dip  # bearish orientation
        else:
            against = dip > dim  # bullish orientation

        gap = abs(dip - dim)

        if against:
            n_against += 1
            weight_against += tf_weight

        # Check for fresh crossover on this bar
        if tf_idx >= 1:
            dip_prev = d['di_plus'][tf_idx - 1]
            dim_prev = d['di_minus'][tf_idx - 1]
            if side == 'long':
                just_crossed = (dip_prev > dim_prev and dim >= dip)
            else:
                just_crossed = (dim_prev > dip_prev and dip >= dim)
            if just_crossed and gap >= 5:  # minimum gap for cross to count
                crossed_tfs.append(tf_name)

        details[tf_name] = {
            'di_plus': dip, 'di_minus': dim, 'gap': gap,
            'against': against,
        }

    weighted_score = weight_against / weight_total if weight_total > 0 else 0.0

    return {
        'n_against': n_against,
        'n_active': n_active,
        'weighted_score': weighted_score,
        'crossed_tfs': crossed_tfs,
        'details': details,
    }


# ── Trade simulation (multi-TF) ─────────────────────────────────────────────

def simulate_multi_tf(tf_data: dict, align_indices: dict,
                      gap_thresholds: list,
                      min_hold_bars: int = 20,
                      sl_ticks: float = 40.0,
                      max_hold_bars: int = 300):
    """
    Simulate trades on the 15s clock. Compare exit strategies:
      SINGLE_1m:  1m DI cross + gap (baseline — current exit_engine behavior)
      ANY_TF:     crossover on ANY TF with gap
      MAJORITY:   >50% of TFs show orientation against
      WEIGHTED:   weighted orientation score > threshold
      CASCADE:    1m cross + at least 2 slower TFs confirm orientation
    """
    base = tf_data.get('15s')
    if base is None:
        print("ERROR: 15s data required as base clock")
        return {}

    closes = base['closes']
    n = len(closes)
    warmup = 200  # enough for slowest TF DMI warmup

    # Sample entries every 50 bars (15s bars → ~12.5 min apart)
    entry_indices = list(range(warmup, n - max_hold_bars, 50))

    strategies = ['SINGLE_1m', 'ANY_TF', 'MAJORITY', 'WEIGHTED_50', 'WEIGHTED_60', 'CASCADE']
    results = {s: {g: [] for g in gap_thresholds} for s in strategies}

    for entry_idx in tqdm(entry_indices, desc="Simulating trades"):
        entry_price = closes[entry_idx]

        for side in ('long', 'short'):
            # Pre-compute: for each strategy+gap, find exit
            for strategy in strategies:
                for gap_min in gap_thresholds:
                    exit_reason = 'max_hold'
                    exit_bar = entry_idx + max_hold_bars
                    exit_price = closes[min(exit_bar, n - 1)]

                    for bar in range(entry_idx + 1, min(entry_idx + max_hold_bars, n)):
                        bars_held = bar - entry_idx
                        price = closes[bar]

                        # PnL
                        if side == 'long':
                            pnl_ticks = (price - entry_price) / TICK_SIZE
                        else:
                            pnl_ticks = (entry_price - price) / TICK_SIZE

                        # SL always active
                        if pnl_ticks <= -sl_ticks:
                            exit_reason = 'stop_loss'
                            exit_bar = bar
                            exit_price = price
                            break

                        # During hold period: only reversal exits
                        if bars_held < min_hold_bars and bars_held >= 4:
                            mtf = compute_multi_tf_orientation(
                                tf_data, align_indices, bar, side)

                            triggered = False

                            if strategy == 'SINGLE_1m':
                                # Only check 1m for crossover + gap
                                if '1m' in mtf['details']:
                                    d1m = mtf['details']['1m']
                                    # Check 1m crossover
                                    a1m = align_indices['1m'][bar]
                                    if a1m >= 1:
                                        dip_prev = tf_data['1m']['di_plus'][a1m - 1]
                                        dim_prev = tf_data['1m']['di_minus'][a1m - 1]
                                        dip = d1m['di_plus']
                                        dim = d1m['di_minus']
                                        if side == 'long':
                                            crossed = (dip_prev > dim_prev and dim >= dip)
                                        else:
                                            crossed = (dim_prev > dip_prev and dip >= dim)
                                        if crossed and d1m['gap'] >= gap_min:
                                            triggered = True

                            elif strategy == 'ANY_TF':
                                # Crossover on ANY TF with sufficient gap
                                for tf_name in mtf['crossed_tfs']:
                                    tf_gap = mtf['details'].get(tf_name, {}).get('gap', 0)
                                    if tf_gap >= gap_min:
                                        triggered = True
                                        break

                            elif strategy == 'MAJORITY':
                                # >50% of active TFs oriented against
                                if (mtf['n_active'] >= 3 and
                                        mtf['n_against'] > mtf['n_active'] / 2):
                                    # Also require at least one fresh cross
                                    if len(mtf['crossed_tfs']) > 0:
                                        triggered = True

                            elif strategy == 'WEIGHTED_50':
                                # Weighted score > 0.50
                                if mtf['weighted_score'] > 0.50 and len(mtf['crossed_tfs']) > 0:
                                    triggered = True

                            elif strategy == 'WEIGHTED_60':
                                # Weighted score > 0.60
                                if mtf['weighted_score'] > 0.60 and len(mtf['crossed_tfs']) > 0:
                                    triggered = True

                            elif strategy == 'CASCADE':
                                # 1m cross + at least 2 slower TFs oriented against
                                has_1m_cross = '1m' in mtf['crossed_tfs']
                                if has_1m_cross:
                                    slow_against = sum(
                                        1 for tf in ('5m', '15m', '30m', '1h')
                                        if tf in mtf['details'] and mtf['details'][tf]['against']
                                    )
                                    if slow_against >= 2:
                                        triggered = True

                            if triggered:
                                exit_reason = 'dmi_reversal'
                                exit_bar = bar
                                exit_price = price
                                break

                        # After hold: giveback exit
                        if bars_held >= min_hold_bars:
                            if side == 'long':
                                peak = np.max(closes[entry_idx:bar + 1])
                                peak_ticks = (peak - entry_price) / TICK_SIZE
                            else:
                                peak = np.min(closes[entry_idx:bar + 1])
                                peak_ticks = (entry_price - peak) / TICK_SIZE

                            if peak_ticks > 8:
                                gave_back = (peak_ticks - pnl_ticks) / peak_ticks
                                if gave_back >= 0.30:
                                    exit_reason = 'giveback'
                                    exit_bar = bar
                                    exit_price = price
                                    break

                    # Final PnL
                    if side == 'long':
                        final_pnl = (exit_price - entry_price) / TICK_SIZE
                    else:
                        final_pnl = (entry_price - exit_price) / TICK_SIZE

                    results[strategy][gap_min].append({
                        'entry_idx': entry_idx,
                        'side': side,
                        'exit_reason': exit_reason,
                        'hold_bars': exit_bar - entry_idx,
                        'pnl_ticks': final_pnl,
                        'pnl_usd': final_pnl * TICK_SIZE * 2.0,
                    })

    return results


# ── Reporting ────────────────────────────────────────────────────────────────

def print_tf_crossover_table(tf_name, rows, forward_bars):
    """Print crossover accuracy table for one TF."""
    fb_labels = [f'@{fb}' for fb in forward_bars]
    print(f"\n  {tf_name:>4}:  {'Gap≥':>5} {'Count':>7}  "
          + "  ".join(f'Acc{l:>4}' for l in fb_labels)
          + "  " + "  ".join(f'Avg{l:>4}' for l in fb_labels))
    print(f"        {'─' * 5} {'─' * 7}  "
          + "  ".join('─' * 8 for _ in fb_labels)
          + "  " + "  ".join('─' * 8 for _ in fb_labels))

    for r in rows:
        if r['n'] == 0:
            print(f"        {r['gap_min']:>5.0f} {0:>7}  -- no events --")
            continue
        acc_str = "  ".join(f"{r.get(f'acc_{fb}', 0):>7.1f}%" for fb in forward_bars)
        avg_str = "  ".join(f"{r.get(f'avg_{fb}', 0):>+7.1f}t" for fb in forward_bars)
        print(f"        {r['gap_min']:>5.0f} {r['n']:>7,}  {acc_str}  {avg_str}")


def print_simulation_results(results, gap_thresholds, strategies):
    """Print trade simulation comparison table."""
    for gap_min in gap_thresholds:
        if gap_min == 0:
            continue
        print(f"\n  ┌─ Gap ≥ {gap_min:.0f} {'─' * 64}")
        print(f"  │ {'Strategy':<14} {'Trades':>7}  "
              f"{'DMI_Rev':>7} {'SL':>7} {'Givebk':>7} {'MaxHld':>7}  "
              f"{'WR':>6} {'Avg$':>8} {'Total$':>10}")
        print(f"  │ {'─' * 14} {'─' * 7}  "
              f"{'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7}  "
              f"{'─' * 6} {'─' * 8} {'─' * 10}")

        for strategy in strategies:
            trades = results[strategy][gap_min]
            n = len(trades)
            if n == 0:
                continue

            n_dmi = sum(1 for t in trades if t['exit_reason'] == 'dmi_reversal')
            n_sl = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')
            n_gb = sum(1 for t in trades if t['exit_reason'] == 'giveback')
            n_mh = sum(1 for t in trades if t['exit_reason'] == 'max_hold')
            wins = sum(1 for t in trades if t['pnl_usd'] > 0)
            total_pnl = sum(t['pnl_usd'] for t in trades)
            avg_pnl = total_pnl / n
            wr = wins / n * 100

            print(f"  │ {strategy:<14} {n:>7,}  "
                  f"{n_dmi:>7,} {n_sl:>7,} {n_gb:>7,} {n_mh:>7,}  "
                  f"{wr:>5.1f}% ${avg_pnl:>7.2f} ${total_pnl:>9,.2f}")

        print(f"  └{'─' * 75}")


def print_dmi_exit_breakdown(results, gap_thresholds, strategies):
    """Detailed PnL breakdown for DMI reversal exits per strategy."""
    for gap_min in gap_thresholds:
        if gap_min == 0:
            continue
        has_any = False
        for strategy in strategies:
            dmi_exits = [t for t in results[strategy][gap_min]
                         if t['exit_reason'] == 'dmi_reversal']
            if dmi_exits:
                has_any = True
                break
        if not has_any:
            continue

        print(f"\n  Gap ≥ {gap_min:.0f}:")
        for strategy in strategies:
            dmi_exits = [t for t in results[strategy][gap_min]
                         if t['exit_reason'] == 'dmi_reversal']
            if not dmi_exits:
                print(f"    {strategy:<14}:  0 DMI exits")
                continue

            wins = [t for t in dmi_exits if t['pnl_usd'] > 0]
            losses = [t for t in dmi_exits if t['pnl_usd'] <= 0]
            net = sum(t['pnl_usd'] for t in dmi_exits)
            wr = len(wins) / len(dmi_exits) * 100
            avg_hold = np.mean([t['hold_bars'] for t in dmi_exits])

            w_str = (f"W:{len(wins)} avg ${np.mean([t['pnl_usd'] for t in wins]):.2f}"
                     if wins else "W:0")
            l_str = (f"L:{len(losses)} avg ${np.mean([t['pnl_usd'] for t in losses]):.2f}"
                     if losses else "L:0")

            print(f"    {strategy:<14}:  {len(dmi_exits):>4} exits | "
                  f"{w_str} | {l_str} | Net ${net:>8,.2f} | "
                  f"WR {wr:.0f}% | Hold {avg_hold:.0f}b")


# ── Multi-TF orientation distribution ───────────────────────────────────────

def analyze_orientation_distribution(tf_data, align_indices, sample_every=20):
    """How often do N TFs agree on direction? Sampled every `sample_every` 15s bars."""
    base = tf_data.get('15s')
    if base is None:
        return

    n = len(base['closes'])
    warmup = 200

    counts_long = defaultdict(int)   # n_bullish_tfs -> count
    counts_short = defaultdict(int)  # n_bearish_tfs -> count
    total = 0

    for bar in range(warmup, n, sample_every):
        n_bull = 0
        n_bear = 0
        n_active = 0

        for tf_name, tf_secs, tf_weight in TF_CONFIG:
            if tf_name not in tf_data or tf_name not in align_indices:
                continue
            tf_idx = align_indices[tf_name][bar]
            if tf_idx < 28:
                continue
            d = tf_data[tf_name]
            dip = d['di_plus'][tf_idx]
            dim = d['di_minus'][tf_idx]
            if dip == 0 and dim == 0:
                continue
            n_active += 1
            if dip > dim:
                n_bull += 1
            else:
                n_bear += 1

        if n_active >= 3:
            counts_long[n_bull] += 1
            counts_short[n_bear] += 1
            total += 1

    return counts_long, counts_short, total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DMI Reversal Simulator — Multi-TF')
    parser.add_argument('--data', default=os.path.join('DATA', 'ATLAS_OOS'),
                        help='ATLAS data root (default: DATA/ATLAS_OOS)')
    parser.add_argument('--period', type=int, default=14, help='DMI period (default: 14)')
    parser.add_argument('--gaps', type=str, default='0,5,10,15,20',
                        help='Comma-separated DI gap thresholds to test')
    parser.add_argument('--min-hold', type=int, default=20,
                        help='Min hold bars (15s) for trade simulation (default: 20 = 5 min)')
    parser.add_argument('--sl', type=int, default=40,
                        help='Stop loss in ticks for simulation (default: 40)')
    parser.add_argument('--max-hold', type=int, default=300,
                        help='Max hold bars before forced exit (default: 300 = 75 min)')
    args = parser.parse_args()

    gap_thresholds = [float(g) for g in args.gaps.split(',')]
    forward_bars_15s = (20, 40, 80, 160)  # 5m, 10m, 20m, 40m in 15s bars

    print("=" * 80)
    print("DMI REVERSAL SIMULATOR — MULTI-TIMEFRAME")
    print("=" * 80)
    print(f"  Data: {args.data}")
    print(f"  DMI period: {args.period}")
    print(f"  Gap thresholds: {gap_thresholds}")
    print(f"  Min hold: {args.min_hold} bars ({args.min_hold * 15 / 60:.1f} min)")
    print(f"  SL: {args.sl} ticks (${args.sl * TICK_VALUE:.2f})")

    # ── Load all TFs ──────────────────────────────────────────────────────
    print(f"\nLoading timeframes from {args.data}...")
    tf_data = load_all_tfs(args.data, period=args.period)

    if '15s' not in tf_data:
        print("ERROR: 15s data required as base execution clock")
        sys.exit(1)

    loaded_tfs = [tf for tf, _, _ in TF_CONFIG if tf in tf_data]
    print(f"\n  Loaded {len(loaded_tfs)} TFs: {', '.join(loaded_tfs)}")

    # ── Build alignment indices ───────────────────────────────────────────
    print("\nBuilding cross-TF alignment index...")
    base_ts = tf_data['15s']['timestamps_i64']
    align_indices = {}
    for tf_name in loaded_tfs:
        align_indices[tf_name] = build_alignment_index(
            base_ts, tf_data[tf_name]['timestamps_i64'])

    # ── Part 1: Per-TF crossover accuracy ─────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 1: PER-TF CROSSOVER ACCURACY")
    print("=" * 80)
    print(f"  Forward horizons: {forward_bars_15s} 15s-bars "
          f"({', '.join(f'{fb*15/60:.0f}m' for fb in forward_bars_15s)})")

    all_tf_rows = {}
    for tf_name in loaded_tfs:
        events, rows = analyze_tf_crossovers(
            tf_data, tf_name, gap_thresholds, forward_bars_15s)
        all_tf_rows[tf_name] = rows
        print_tf_crossover_table(tf_name, rows, forward_bars_15s)

    # ── Summary: best TF at each gap ──────────────────────────────────────
    print(f"\n  BEST TF per gap threshold (accuracy @ 40-bar / 10min horizon):")
    print(f"  {'Gap≥':>5}  {'Best TF':>7}  {'Acc':>6}  {'Count':>6}  {'AvgMov':>8}")
    print(f"  {'─' * 5}  {'─' * 7}  {'─' * 6}  {'─' * 6}  {'─' * 8}")
    for gap_min in gap_thresholds:
        best_tf = ''
        best_acc = 0
        best_n = 0
        best_avg = 0
        for tf_name in loaded_tfs:
            for r in all_tf_rows[tf_name]:
                if r['gap_min'] == gap_min and r['n'] >= 10:
                    acc = r.get('acc_40', 0)
                    if acc > best_acc:
                        best_acc = acc
                        best_tf = tf_name
                        best_n = r['n']
                        best_avg = r.get('avg_40', 0)
        if best_tf:
            print(f"  {gap_min:>5.0f}  {best_tf:>7}  {best_acc:>5.1f}%  {best_n:>6,}  {best_avg:>+7.1f}t")
        else:
            print(f"  {gap_min:>5.0f}  {'---':>7}")

    # ── Part 2: Multi-TF orientation distribution ─────────────────────────
    print("\n" + "=" * 80)
    print("PART 2: MULTI-TF ORIENTATION AGREEMENT")
    print("=" * 80)
    print("  How often do N timeframes agree on bullish/bearish DI orientation?")

    counts_bull, counts_bear, total_samples = analyze_orientation_distribution(
        tf_data, align_indices)

    if total_samples > 0:
        max_tfs = max(max(counts_bull.keys(), default=0), max(counts_bear.keys(), default=0))
        print(f"\n  {'N_TFs':>6} {'Bull':>8} {'%':>6}   {'Bear':>8} {'%':>6}")
        print(f"  {'─' * 6} {'─' * 8} {'─' * 6}   {'─' * 8} {'─' * 6}")
        for k in range(max_tfs + 1):
            b = counts_bull.get(k, 0)
            s = counts_bear.get(k, 0)
            print(f"  {k:>6} {b:>8,} {b / total_samples * 100:>5.1f}%   "
                  f"{s:>8,} {s / total_samples * 100:>5.1f}%")
        print(f"  Total samples: {total_samples:,}")

    # ── Part 3: Trade simulation ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 3: TRADE EXIT SIMULATION — STRATEGY COMPARISON")
    print("=" * 80)
    print(f"  Min hold: {args.min_hold} bars ({args.min_hold * 15 / 60:.1f} min)")
    print(f"  SL: {args.sl}t | Max hold: {args.max_hold} bars | Giveback: 30% after hold")
    print(f"  Entry every 50 bars (15s), LONG+SHORT at each")
    print(f"\n  Strategies:")
    print(f"    SINGLE_1m:  1m DI crossover + gap (current behavior)")
    print(f"    ANY_TF:     crossover on ANY TF + gap")
    print(f"    MAJORITY:   >50% of TFs oriented against + fresh cross")
    print(f"    WEIGHTED_50: weighted orientation > 0.50 + fresh cross")
    print(f"    WEIGHTED_60: weighted orientation > 0.60 + fresh cross")
    print(f"    CASCADE:    1m cross + 2 slow TFs (5m/15m/30m/1h) oriented against")

    strategies = ['SINGLE_1m', 'ANY_TF', 'MAJORITY', 'WEIGHTED_50', 'WEIGHTED_60', 'CASCADE']

    sim_results = simulate_multi_tf(
        tf_data, align_indices,
        gap_thresholds=[g for g in gap_thresholds if g > 0],
        min_hold_bars=args.min_hold,
        sl_ticks=args.sl,
        max_hold_bars=args.max_hold,
    )

    sim_gaps = [g for g in gap_thresholds if g > 0]
    print_simulation_results(sim_results, sim_gaps, strategies)

    # ── Part 4: DMI exit breakdown ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 4: DMI REVERSAL EXITS — DETAILED BREAKDOWN")
    print("=" * 80)

    print_dmi_exit_breakdown(sim_results, sim_gaps, strategies)

    # ── Part 5: Recommendation ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 5: RECOMMENDATION")
    print("=" * 80)

    # Find best strategy+gap by total PnL
    best_strat = ''
    best_gap = 0
    best_pnl = -999999
    best_wr = 0
    for strategy in strategies:
        for gap_min in sim_gaps:
            trades = sim_results.get(strategy, {}).get(gap_min, [])
            if trades:
                total = sum(t['pnl_usd'] for t in trades)
                if total > best_pnl:
                    best_pnl = total
                    best_strat = strategy
                    best_gap = gap_min
                    wins = sum(1 for t in trades if t['pnl_usd'] > 0)
                    best_wr = wins / len(trades) * 100

    print(f"\n  Best strategy by total PnL:")
    print(f"    {best_strat} @ gap >= {best_gap:.0f}")
    print(f"    Total PnL: ${best_pnl:,.2f} | WR: {best_wr:.1f}%")

    # Find best strategy with fewest SL hits (most protective DMI exits)
    print(f"\n  Strategy comparison — DMI reversal saves vs SL hits:")
    for strategy in strategies:
        for gap_min in sim_gaps:
            trades = sim_results.get(strategy, {}).get(gap_min, [])
            if not trades:
                continue
            n_dmi = sum(1 for t in trades if t['exit_reason'] == 'dmi_reversal')
            n_sl = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')
            dmi_pnl = sum(t['pnl_usd'] for t in trades if t['exit_reason'] == 'dmi_reversal')
            sl_pnl = sum(t['pnl_usd'] for t in trades if t['exit_reason'] == 'stop_loss')
            if n_dmi > 0:
                print(f"    {strategy:<14} gap>={gap_min:>2.0f}: "
                      f"DMI exits={n_dmi:>4} (${dmi_pnl:>+8,.2f}) | "
                      f"SL hits={n_sl:>4} (${sl_pnl:>+8,.2f}) | "
                      f"Ratio: {n_dmi / max(n_sl, 1):.2f}x")

    print(f"\n  Current exit_engine.py: SINGLE_1m, gap >= 15.0")
    print(f"  Recommendation: compare SINGLE_1m vs CASCADE/WEIGHTED for your data")


if __name__ == '__main__':
    main()
