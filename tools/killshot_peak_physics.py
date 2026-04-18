"""Identify the PHYSICS of the peak for KILL_SHOT trades.

For every trade whose peak > $3 (i.e. enough MFE to be a real event),
reconstruct the 5s price path from entry_ts → exit_ts, find the peak bar,
and measure what features flipped AT the peak vs the bar before/after.

Output: reports/findings/2026-04-17_killshot_peak_physics.md

Usage:
    python tools/killshot_peak_physics.py
"""
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRADES_PKL = 'training/output/isolated/KILL_SHOT.pkl'
BARS_DIR = 'DATA/ATLAS/5s'
FEATS_DIR = 'DATA/ATLAS/FEATURES_5s'
OUT_MD = 'reports/findings/2026-04-17_killshot_peak_physics.md'

# Feature columns we want to watch at peak
WATCH = [
    '1m_z_se', '1m_velocity', '1m_acceleration', '1m_variance_ratio',
    '1m_vol_rel', '1m_dmi_diff', '1m_wick_ratio', '1m_p_at_center',
    '1m_reversion_prob', '1m_bar_range',
    '5m_velocity', '5m_acceleration', '5m_wick_ratio',
    '15m_wick_ratio', '1h_z_se', '1h_velocity',
]

TICK_VALUE = 0.50  # MNQ: $0.50 per 0.25-point tick = $2/point


def load_day_cache():
    """Lazy-load: {day_name: (bars_df, feats_df)}."""
    return {}


def get_day(cache, day):
    if day in cache:
        return cache[day]
    bars_p = os.path.join(BARS_DIR, f'{day}.parquet')
    feats_p = os.path.join(FEATS_DIR, f'{day}.parquet')
    if not os.path.exists(bars_p) or not os.path.exists(feats_p):
        cache[day] = (None, None)
        return None, None
    bars = pd.read_parquet(bars_p).set_index('timestamp').sort_index()
    feats = pd.read_parquet(feats_p).set_index('timestamp').sort_index()
    cache[day] = (bars, feats)
    return bars, feats


def trade_path(trade, cache):
    """Return (path_df, peak_idx) where path_df has: ts, close, mfe (points
    of favorable excursion from entry), and all watched features. peak_idx
    is the integer bar index of max MFE. Returns (None, None) if insufficient
    data."""
    bars, feats = get_day(cache, trade['day'])
    if bars is None:
        return None, None
    entry_ts = int(trade['entry_ts'])
    exit_ts = int(trade['exit_ts'])
    # Slice inclusive of entry and exit
    sliced = bars.loc[entry_ts:exit_ts]
    if len(sliced) < 2:
        return None, None
    entry_px = float(trade['entry_price'])
    direction = 1 if trade['dir'] == 'long' else -1
    mfe = (sliced['close'].values - entry_px) * direction  # points favorable
    peak_idx = int(np.argmax(mfe))
    path = pd.DataFrame({
        'ts': sliced.index.values,
        'close': sliced['close'].values,
        'mfe_pts': mfe,
    })
    # Attach features
    f_slice = feats.loc[entry_ts:exit_ts]
    for col in WATCH:
        if col in f_slice.columns:
            # Align to bars by ts
            path[col] = f_slice.reindex(sliced.index)[col].values
    return path, peak_idx


def main():
    if not os.path.exists(TRADES_PKL):
        print(f'Missing {TRADES_PKL} — run run_tier_isolated.py KILL_SHOT first')
        return

    with open(TRADES_PKL, 'rb') as f:
        trades = pickle.load(f)
    print(f'Loaded {len(trades):,} KILL_SHOT trades')

    # Filter: primary trades only, peak > $3 (meaningful MFE)
    # peak field is in $, 1 MNQ point = $2, so $3 ~ 1.5 pts
    sub = [t for t in trades
           if not t.get('is_chain', False) and t.get('peak', 0) > 3.0]
    print(f'{len(sub):,} trades with peak > $3 (MFE exists)')

    cache = load_day_cache()
    rows = []  # one row per analyzed trade
    peak_times = []
    velocity_flips = 0
    accel_flips = 0
    volume_fades = 0
    wick_appears = 0
    p_center_hits = 0
    n_valid = 0

    # For feature deltas at peak
    at_peak_vals = defaultdict(list)
    before_peak_vals = defaultdict(list)
    after_peak_vals = defaultdict(list)

    # Lookback window for "before peak" and lookahead for "after peak" (bars)
    K = 3  # 3 bars = 15s

    for t in sub:
        path, pk = trade_path(t, cache)
        if path is None or pk is None:
            continue
        n = len(path)
        if pk < K or pk > n - K - 1:
            # Skip if peak is at boundary; we can't measure ±K
            continue
        direction = 1 if t['dir'] == 'long' else -1
        n_valid += 1
        peak_pts = path['mfe_pts'].iloc[pk]
        time_to_peak = pk * 5  # seconds
        peak_times.append(time_to_peak)

        # Features at peak vs ±K bars
        for col in WATCH:
            if col not in path.columns:
                continue
            vals = path[col].values
            before_peak_vals[col].append(vals[pk - K])
            at_peak_vals[col].append(vals[pk])
            after_peak_vals[col].append(vals[pk + K])

        # Signed signals — align with trade direction:
        #   1m_velocity sign change against trade direction = momentum death
        v1 = path['1m_velocity'].values
        a1 = path['1m_acceleration'].values
        # Adverse velocity means velocity turned against trade dir
        # long trade: profit when close rises → velocity >0 was "for us"
        # peak happens when velocity crosses to 0 or opposite
        v_before = v1[pk - K] * direction
        v_after = v1[pk + K] * direction
        if v_before > 0 and v_after < 0:
            velocity_flips += 1
        a_before = a1[pk - K] * direction
        a_after = a1[pk + K] * direction
        if a_before > 0 and a_after < 0:
            accel_flips += 1

        # Volume fade: vol_rel drops by >20% from -K to +K
        vr_b = path['1m_vol_rel'].iloc[pk - K]
        vr_a = path['1m_vol_rel'].iloc[pk + K]
        if vr_b > 0 and (vr_a - vr_b) / max(abs(vr_b), 0.01) < -0.20:
            volume_fades += 1

        # Wick on the OTHER side appearing: for long trade (direction=1),
        # "wick against us" means upper wick (sell rejection at the peak)
        # 1m_wick_ratio is signed? Check convention.
        # Our wick features are magnitude — so we look at 1m_wick_ratio spike
        # near peak combined with reversal.
        w_at = path['1m_wick_ratio'].iloc[pk]
        w_before = path['1m_wick_ratio'].iloc[pk - K]
        if w_at > 0.3 and w_at > w_before * 1.3:
            wick_appears += 1

        # p_at_center hit — price back to mean
        pc_at = path['1m_p_at_center'].iloc[pk]
        if pc_at > 0.6:
            p_center_hits += 1

        rows.append({
            'day': t['day'],
            'entry_ts': t['entry_ts'],
            'dir': t['dir'],
            'peak_pts': peak_pts,
            'peak_dollar': peak_pts * 2.0,
            'bars_to_peak': pk,
            'sec_to_peak': time_to_peak,
            'path_len': n,
            'pnl': t['pnl'],
        })

    if n_valid == 0:
        print('No usable trades — aborting')
        return

    df = pd.DataFrame(rows)
    print(f'\nAnalyzed {n_valid:,} trades with usable ±{K} bar window around peak')

    # Save per-trade rows
    os.makedirs('reports/findings', exist_ok=True)
    df.to_csv('reports/findings/killshot_peak_paths.csv', index=False)

    # Headline stats
    lines = []
    lines.append('# KILL_SHOT Peak Physics\n')
    lines.append(f'Date: 2026-04-17\n')
    lines.append(f'Source: `{TRADES_PKL}`  ({len(trades):,} total, '
                 f'{len(sub):,} with peak>$3, {n_valid:,} analyzable)\n')
    lines.append('\n## Time to peak\n')
    lines.append(f'- Median: {np.median(peak_times):.0f}s '
                 f'({np.median(peak_times)/60:.1f} min)\n')
    lines.append(f'- p25: {np.percentile(peak_times, 25):.0f}s, '
                 f'p75: {np.percentile(peak_times, 75):.0f}s, '
                 f'p90: {np.percentile(peak_times, 90):.0f}s\n')
    lines.append(f'- Mean bars-to-peak: {df["bars_to_peak"].mean():.1f} '
                 f'(= {df["bars_to_peak"].mean()*5:.0f}s)\n')

    # Signal fire rates at peak
    lines.append(f'\n## Peak-detection signal fire rates (n={n_valid})\n')
    lines.append(f'| Signal | Fires at peak | Rate |\n')
    lines.append(f'|---|---|---|\n')
    lines.append(f'| 1m velocity flips sign against trade (±{K} bars) | '
                 f'{velocity_flips} | {velocity_flips/n_valid*100:.1f}% |\n')
    lines.append(f'| 1m acceleration flips against trade | '
                 f'{accel_flips} | {accel_flips/n_valid*100:.1f}% |\n')
    lines.append(f'| Volume fades >20% across peak | '
                 f'{volume_fades} | {volume_fades/n_valid*100:.1f}% |\n')
    lines.append(f'| 1m wick_ratio > 0.3 at peak (jump >30%) | '
                 f'{wick_appears} | {wick_appears/n_valid*100:.1f}% |\n')
    lines.append(f'| 1m p_at_center > 0.6 at peak | '
                 f'{p_center_hits} | {p_center_hits/n_valid*100:.1f}% |\n')

    # Feature medians: before / at / after peak
    lines.append(f'\n## Feature values: before peak / AT peak / after peak\n')
    lines.append(f'(±{K} bars = ±{K*5}s window)\n\n')
    lines.append(f'| Feature | Before | AT PEAK | After | Δ(after-before) |\n')
    lines.append(f'|---|---|---|---|---|\n')
    for col in WATCH:
        if col not in at_peak_vals or len(at_peak_vals[col]) < 10:
            continue
        b = np.nanmedian(before_peak_vals[col])
        a_ = np.nanmedian(at_peak_vals[col])
        af = np.nanmedian(after_peak_vals[col])
        delta = af - b
        lines.append(f'| {col} | {b:+.3f} | {a_:+.3f} | {af:+.3f} | '
                     f'{delta:+.3f} |\n')

    # Standardized: which features change most from before→after?
    lines.append(f'\n## Largest feature swings across peak (Cohen d)\n')
    lines.append(f'Effect size = (median_after - median_before) / pooled_std. '
                 f'Ranks which signals move most when peak passes.\n\n')
    lines.append(f'| Feature | Before→After | Cohen d |\n')
    lines.append(f'|---|---|---|\n')
    swings = []
    for col in WATCH:
        if col not in at_peak_vals or len(at_peak_vals[col]) < 10:
            continue
        b_arr = np.array(before_peak_vals[col])
        a_arr = np.array(after_peak_vals[col])
        b_arr = b_arr[~np.isnan(b_arr)]
        a_arr = a_arr[~np.isnan(a_arr)]
        if len(b_arr) < 10 or len(a_arr) < 10:
            continue
        pooled = np.sqrt((np.var(b_arr) + np.var(a_arr)) / 2)
        if pooled < 1e-9:
            continue
        d = (np.median(a_arr) - np.median(b_arr)) / pooled
        swings.append((col, np.median(b_arr), np.median(a_arr), d))
    swings.sort(key=lambda r: -abs(r[3]))
    for col, b, a_, d in swings:
        flag = ' ***' if abs(d) > 0.3 else ' **' if abs(d) > 0.2 else ''
        lines.append(f'| {col} | {b:+.3f} → {a_:+.3f} | {d:+.2f}{flag} |\n')

    # Exit rule candidates — rank by how well they catch peak BEFORE giveback
    lines.append(f'\n## Exit rule candidates (back-test on this cohort)\n')
    # Simulate: for each trade, if signal X fires at bar k (k < path_len),
    # exit at bar k close. Compare captured vs peak.
    lines.append(f'Each rule scans bars AFTER entry; exits at first trigger '
                 f'(5s resolution). Compared to the no-rule baseline '
                 f'(final close, $/trade) and peak.\n\n')

    # Rerun paths to simulate exits
    def sim_exit_rule(trigger_fn, name):
        """trigger_fn(path_row) -> bool. Returns (captured_total_$, trades_fired,
        avg_bars_to_exit)."""
        total = 0.0
        fired = 0
        bars_list = []
        for t in sub:
            path, pk = trade_path(t, cache)
            if path is None:
                continue
            direction = 1 if t['dir'] == 'long' else -1
            # Walk from bar 1 (skip entry bar)
            exit_bar = None
            for k in range(1, len(path)):
                if trigger_fn(path.iloc[k], direction):
                    exit_bar = k
                    break
            if exit_bar is None:
                # Natural exit — use original pnl
                total += t['pnl']
            else:
                px = path['close'].iloc[exit_bar]
                captured = (px - t['entry_price']) * direction * 2.0
                total += captured
                fired += 1
                bars_list.append(exit_bar)
        avg_bars = np.mean(bars_list) if bars_list else 0
        return total, fired, avg_bars

    rules = [
        ('1m velocity flips against trade',
         lambda r, d: (r['1m_velocity'] * d) < 0 and r['mfe_pts'] > 0),
        ('1m acceleration negative (against)',
         lambda r, d: (r['1m_acceleration'] * d) < -1.0 and r['mfe_pts'] > 0),
        ('p_at_center > 0.5 after MFE > 1pt',
         lambda r, d: r['1m_p_at_center'] > 0.5 and r['mfe_pts'] > 1.0),
        ('reversion_prob > 0.6 after MFE > 1pt',
         lambda r, d: r['1m_reversion_prob'] > 0.6 and r['mfe_pts'] > 1.0),
        ('wick against trade (1m_wick_ratio > 0.4)',
         lambda r, d: r['1m_wick_ratio'] > 0.4 and r['mfe_pts'] > 0.5),
        ('MFE gave back 50% from running peak (trail)',
         None),  # handled specially below
        ('Fixed target $5',
         lambda r, d: r['mfe_pts'] * 2.0 >= 5.0),
        ('Fixed target $7',
         lambda r, d: r['mfe_pts'] * 2.0 >= 7.0),
        ('Fixed target $10',
         lambda r, d: r['mfe_pts'] * 2.0 >= 10.0),
    ]

    lines.append(f'| Rule | Fires | Total $ | $/trade | Avg bars to exit |\n')
    lines.append(f'|---|---|---|---|---|\n')

    # Baseline
    baseline_total = sum(t['pnl'] for t in sub)
    lines.append(f'| (baseline — natural exit) | — | ${baseline_total:+,.0f} | '
                 f'${baseline_total/len(sub):+.2f} | — |\n')

    for name, fn in rules:
        if fn is None:
            # Trailing stop: exit when giveback >= 50% of running peak (once peak>1pt)
            total = 0.0
            fired = 0
            bars_list = []
            for t in sub:
                path, _ = trade_path(t, cache)
                if path is None:
                    continue
                direction = 1 if t['dir'] == 'long' else -1
                run_peak = 0.0
                exit_bar = None
                for k in range(1, len(path)):
                    mfe_k = path['mfe_pts'].iloc[k]
                    run_peak = max(run_peak, mfe_k)
                    if run_peak > 1.0 and mfe_k < run_peak * 0.5:
                        exit_bar = k
                        break
                if exit_bar is None:
                    total += t['pnl']
                else:
                    px = path['close'].iloc[exit_bar]
                    captured = (px - t['entry_price']) * direction * 2.0
                    total += captured
                    fired += 1
                    bars_list.append(exit_bar)
            avg_b = np.mean(bars_list) if bars_list else 0
            lines.append(f'| {name} | {fired} | ${total:+,.0f} | '
                         f'${total/len(sub):+.2f} | {avg_b:.1f} |\n')
            continue
        total, fired, avg_b = sim_exit_rule(fn, name)
        lines.append(f'| {name} | {fired} | ${total:+,.0f} | '
                     f'${total/len(sub):+.2f} | {avg_b:.1f} |\n')

    # Write out
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))
    print(f'\nReport: {OUT_MD}')
    print(f'Per-trade CSV: reports/findings/killshot_peak_paths.csv')

    # Also print headline to stdout
    print()
    for ln in lines[:40]:
        print(ln, end='')


if __name__ == '__main__':
    main()
