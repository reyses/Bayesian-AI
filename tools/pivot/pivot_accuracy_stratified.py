"""
Pivot direction-prediction accuracy, stratified by chord ratio AND wick.

Combines three signals at each zigzag pivot:
  1. Residual (1m_z_se) — predicts direction via sign
  2. Chord ratio (reg_chord / price_path) — classifies regime
  3. Wick — detects rejection at ceiling/floor

For each pivot compute:
  - residual-based direction prediction (LONG if res<0 else SHORT)
  - actual next-leg direction
  - correct? = predicted == actual

Stratify accuracy by:
  (a) chord_ratio regime (NOISE/TREND)
  (b) wick rejection strength (STRONG/MILD/NONE)
  (c) joint (both together)

Also include AGREEMENT: does wick direction CONFIRM the residual direction?
  - At HIGH pivot (residual says SHORT): CONFIRM = large upper wick
  - At LOW pivot (residual says LONG): CONFIRM = large lower wick

Hypothesis: CONFIRM + NOISE regime = highest accuracy pivots.

Usage:
    python tools/pivot_accuracy_stratified.py
    python tools/pivot_accuracy_stratified.py --threshold 15

Output: reports/findings/pivot_accuracy_stratified.md
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.regression_line_cohen_d import zigzag_pivots


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
OUT_MD = 'reports/findings/pivot_accuracy_stratified.md'
DOLLAR_PER_POINT = 2.0

CHORD_WINDOW = 10

CHORD_BINS = [
    ('VERY_NOISE (<0.05)',   0.00, 0.05),
    ('NOISE (0.05-0.15)',    0.05, 0.15),
    ('MIXED (0.15-0.30)',    0.15, 0.30),
    ('TREND (0.30-0.50)',    0.30, 0.50),
    ('STRONG_TREND (>0.50)', 0.50, float('inf')),
]

WICK_BINS = [
    ('NONE (<0.15)',    0.00, 0.15),
    ('MILD (0.15-0.30)', 0.15, 0.30),
    ('MED (0.30-0.50)',  0.30, 0.50),
    ('STRONG (>0.50)',   0.50, float('inf')),
]

# Volume ratio = volume[pivot] / mean(volume[pivot-20..pivot-1]) (pivot itself NOT in baseline)
VOL_LOOKBACK = 20
VOL_BINS = [
    ('LOW (<0.7)',       0.00, 0.70),
    ('NORMAL (0.7-1.3)', 0.70, 1.30),
    ('ELEVATED (1.3-2)', 1.30, 2.00),
    ('SPIKE (>2)',       2.00, float('inf')),
]

# Velocity measurement — last 5 bars. SIGNED and scaled relative to predicted direction.
# "pred_vel_alignment" = sign(price_vel) matches sign(prediction)?
#   If prediction is LONG and price has been falling (vel<0): still in downtrend, bad entry
#   If prediction is LONG and price has been rising (vel>0): momentum confirms, good entry
# Similar for reg_vel (regression slope).
VEL_WINDOW = 5
# Stratify by the MAGNITUDE of price_vel in the direction OPPOSITE to prediction
# (= residual attracts us toward reversion; price has been moving WITH residual
#  means trend is FLIPPING — good; moving AGAINST residual means trend accelerating
#  — entering against momentum is risky).
# Let pred_dir_vel_sign = +1 if sign(price_vel) matches prediction direction.
VEL_SIGN_BINS = [
    ('AGAINST prediction (vel opposite)', -float('inf'), -0.01),
    ('NEUTRAL (|vel|<0.01 pts/bar)',      -0.01,          0.01),
    ('WITH prediction (vel small)',        0.01,          0.5),
    ('WITH prediction (vel strong)',       0.5,           float('inf')),
]


def rolling_fit(closes, window):
    n = len(closes)
    fitted = np.full(n, np.nan)
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        x = np.arange(window, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        dx = x - xm
        denom = (dx * dx).sum()
        if denom < 1e-9:
            continue
        slope = (dx * (y - ym)).sum() / denom
        intercept = ym - slope * xm
        fitted[i] = intercept + slope * (window - 1)
    return fitted


def process_day(price_path, features_path, threshold):
    df_price = pd.read_parquet(price_path).sort_values('timestamp').reset_index(drop=True)
    closes = df_price['close'].values.astype(np.float64)
    highs = df_price['high'].values.astype(np.float64)
    lows = df_price['low'].values.astype(np.float64)
    opens = df_price['open'].values.astype(np.float64)
    volumes = df_price['volume'].values.astype(np.float64) if 'volume' in df_price.columns else np.zeros(len(df_price))
    ts_1m = df_price['timestamp'].values.astype(np.int64)

    df_feat = pd.read_parquet(features_path).sort_values('timestamp').reset_index(drop=True)
    ts_feat = df_feat['timestamp'].values.astype(np.int64)
    if '1m_z_se' not in df_feat.columns:
        return []
    res_feat = df_feat['1m_z_se'].values.astype(np.float64)
    idx = np.searchsorted(ts_feat, ts_1m, side='right') - 1
    idx = np.clip(idx, 0, len(ts_feat) - 1)
    residuals = res_feat[idx]

    pivots = zigzag_pivots(closes, threshold)
    if len(pivots) < 3:
        return []
    fit = rolling_fit(closes, CHORD_WINDOW)

    events = []
    for i in range(len(pivots) - 1):
        piv = pivots[i]
        next_piv = pivots[i + 1]
        if piv < CHORD_WINDOW:
            continue

        # Chord ratio at pivot
        segment = closes[piv - CHORD_WINDOW: piv + 1]
        price_path = float(np.sum(np.abs(np.diff(segment))))
        if np.isnan(fit[piv]) or np.isnan(fit[piv - CHORD_WINDOW]):
            continue
        reg_chord = float(abs(fit[piv] - fit[piv - CHORD_WINDOW]))
        reg_to_path = reg_chord / max(price_path, 0.1)

        # Volume ratio: pivot bar volume / mean of prior VOL_LOOKBACK bars
        if piv >= VOL_LOOKBACK:
            mean_vol = volumes[piv - VOL_LOOKBACK: piv].mean()
            vol_ratio = float(volumes[piv] / mean_vol) if mean_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        # Velocities over last VEL_WINDOW 1m bars (points per bar, signed)
        if piv >= VEL_WINDOW:
            price_vel = float((closes[piv] - closes[piv - VEL_WINDOW]) / VEL_WINDOW)
            if not np.isnan(fit[piv]) and not np.isnan(fit[piv - VEL_WINDOW]):
                reg_vel = float((fit[piv] - fit[piv - VEL_WINDOW]) / VEL_WINDOW)
            else:
                reg_vel = 0.0
        else:
            price_vel = 0.0
            reg_vel = 0.0

        # Wick at pivot bar
        bar_range = highs[piv] - lows[piv]
        if bar_range < 1e-6:
            continue
        body_top = max(opens[piv], closes[piv])
        body_bot = min(opens[piv], closes[piv])
        upper_wick = highs[piv] - body_top
        lower_wick = body_bot - lows[piv]
        upper_wick_pct = upper_wick / bar_range
        lower_wick_pct = lower_wick / bar_range

        # Residual at pivot (direction prediction)
        r = float(residuals[piv])
        if np.isnan(r) or abs(r) < 0.5:
            continue   # skip low-conviction pivots
        pred_direction = 'LONG' if r < 0 else 'SHORT'

        # Actual next leg direction
        leg_pts = closes[next_piv] - closes[piv]
        actual = 'LONG' if leg_pts > 0 else 'SHORT'
        correct = (pred_direction == actual)

        # Wick confirming the prediction?
        # LONG prediction: we want lower wick (support rejection)
        # SHORT prediction: we want upper wick (resistance rejection)
        if pred_direction == 'LONG':
            rejection_wick_pct = lower_wick_pct
            opposing_wick_pct = upper_wick_pct
        else:
            rejection_wick_pct = upper_wick_pct
            opposing_wick_pct = lower_wick_pct

        # Signed velocities relative to PREDICTION direction:
        # positive = moving WITH the predicted direction (trend is flipping
        # toward us; good entry). negative = moving AGAINST (trend continuing
        # against; risky entry).
        pred_sign = +1 if pred_direction == 'LONG' else -1
        pred_dir_price_vel = pred_sign * price_vel
        pred_dir_reg_vel = pred_sign * reg_vel
        vel_alignment = (np.sign(price_vel) == np.sign(reg_vel))

        events.append({
            'correct': correct,
            'pred_direction': pred_direction,
            'residual': r,
            'reg_to_path': reg_to_path,
            'upper_wick_pct': upper_wick_pct,
            'lower_wick_pct': lower_wick_pct,
            'rejection_wick_pct': rejection_wick_pct,
            'opposing_wick_pct': opposing_wick_pct,
            'vol_ratio': vol_ratio,
            'price_vel': price_vel,
            'reg_vel': reg_vel,
            'pred_dir_price_vel': pred_dir_price_vel,
            'pred_dir_reg_vel': pred_dir_reg_vel,
            'vel_alignment': vel_alignment,
            'leg_dollars': abs(leg_pts) * DOLLAR_PER_POINT,
        })
    return events


def accuracy(events):
    if not events:
        return None
    correct = sum(1 for e in events if e['correct'])
    return correct / len(events) * 100


def stratify(events, key, bins):
    out = []
    for label, lo, hi in bins:
        subset = [e for e in events if lo <= e[key] < hi]
        if len(subset) < 20:
            out.append({'label': label, 'n': len(subset), 'valid': False})
            continue
        acc = accuracy(subset)
        avg_leg = np.mean([e['leg_dollars'] for e in subset])
        out.append({
            'label': label, 'n': len(subset), 'valid': True,
            'acc': acc, 'avg_leg': float(avg_leg),
        })
    return out


def joint_stratify(events, key1, bins1, key2, bins2):
    """2D stratification — accuracy per (bin1, bin2) cell."""
    rows = []
    for label1, lo1, hi1 in bins1:
        row = {'label': label1, 'cells': []}
        for label2, lo2, hi2 in bins2:
            subset = [e for e in events
                       if lo1 <= e[key1] < hi1 and lo2 <= e[key2] < hi2]
            if len(subset) < 20:
                row['cells'].append({'label': label2, 'n': len(subset), 'valid': False})
                continue
            acc = accuracy(subset)
            row['cells'].append({
                'label': label2, 'n': len(subset), 'valid': True,
                'acc': acc,
            })
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=15.0)
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))

    is_events = []
    for p in tqdm(is_paths, desc='IS', unit='day'):
        day = os.path.basename(p).replace('.parquet', '')
        feat = os.path.join(FEATURES_5S_DIR, f'{day}.parquet')
        if not os.path.exists(feat):
            continue
        is_events.extend(process_day(p, feat, args.threshold))
    print(f'IS events: {len(is_events):,}')

    oos_events = []
    for p in tqdm(oos_paths, desc='OOS', unit='day'):
        day = os.path.basename(p).replace('.parquet', '')
        feat = os.path.join(FEATURES_5S_DIR, f'{day}.parquet')
        if not os.path.exists(feat):
            continue
        oos_events.extend(process_day(p, feat, args.threshold))
    print(f'OOS events: {len(oos_events):,}')

    print(f'\nBaseline accuracy (all pivots with |res|>=0.5):')
    print(f'  IS : {accuracy(is_events):.1f}%')
    print(f'  OOS: {accuracy(oos_events):.1f}%')

    # 1D stratification: chord ratio
    print('\n=== By chord ratio (reg_to_path_10) ===')
    print(f'{"Bucket":<25} {"IS N":>6} {"IS acc":>7} {"OOS N":>6} {"OOS acc":>7}')
    is_chord = stratify(is_events, 'reg_to_path', CHORD_BINS)
    oos_chord = stratify(oos_events, 'reg_to_path', CHORD_BINS)
    for i_s, o_s in zip(is_chord, oos_chord):
        is_acc = f'{i_s["acc"]:.1f}%' if i_s.get('valid') else '—'
        oos_acc = f'{o_s["acc"]:.1f}%' if o_s.get('valid') else '—'
        print(f'{i_s["label"]:<25} {i_s["n"]:>6,} {is_acc:>7} '
              f'{o_s["n"]:>6,} {oos_acc:>7}')

    # 1D stratification: rejection wick
    print('\n=== By rejection wick % (pivot-bar wick in the direction of prediction) ===')
    print(f'{"Bucket":<25} {"IS N":>6} {"IS acc":>7} {"OOS N":>6} {"OOS acc":>7}')
    is_wick = stratify(is_events, 'rejection_wick_pct', WICK_BINS)
    oos_wick = stratify(oos_events, 'rejection_wick_pct', WICK_BINS)
    for i_s, o_s in zip(is_wick, oos_wick):
        is_acc = f'{i_s["acc"]:.1f}%' if i_s.get('valid') else '—'
        oos_acc = f'{o_s["acc"]:.1f}%' if o_s.get('valid') else '—'
        print(f'{i_s["label"]:<25} {i_s["n"]:>6,} {is_acc:>7} '
              f'{o_s["n"]:>6,} {oos_acc:>7}')

    # 1D stratification: volume
    print('\n=== By volume ratio (pivot vol / mean of last 20 bars) ===')
    print(f'{"Bucket":<25} {"IS N":>6} {"IS acc":>7} {"OOS N":>6} {"OOS acc":>7}')
    is_vol = stratify(is_events, 'vol_ratio', VOL_BINS)
    oos_vol = stratify(oos_events, 'vol_ratio', VOL_BINS)
    for i_s, o_s in zip(is_vol, oos_vol):
        is_acc = f'{i_s["acc"]:.1f}%' if i_s.get('valid') else '—'
        oos_acc = f'{o_s["acc"]:.1f}%' if o_s.get('valid') else '—'
        print(f'{i_s["label"]:<25} {i_s["n"]:>6,} {is_acc:>7} '
              f'{o_s["n"]:>6,} {oos_acc:>7}')

    # 1D stratification: price velocity in prediction direction
    print('\n=== By pred-direction price velocity (pts/bar) ===')
    print(f'{"Bucket":<35} {"IS N":>6} {"IS acc":>7} {"OOS N":>6} {"OOS acc":>7}')
    is_vel_p = stratify(is_events, 'pred_dir_price_vel', VEL_SIGN_BINS)
    oos_vel_p = stratify(oos_events, 'pred_dir_price_vel', VEL_SIGN_BINS)
    for i_s, o_s in zip(is_vel_p, oos_vel_p):
        is_acc = f'{i_s["acc"]:.1f}%' if i_s.get('valid') else '—'
        oos_acc = f'{o_s["acc"]:.1f}%' if o_s.get('valid') else '—'
        print(f'{i_s["label"]:<35} {i_s["n"]:>6,} {is_acc:>7} '
              f'{o_s["n"]:>6,} {oos_acc:>7}')

    print('\n=== By pred-direction regression velocity (pts/bar) ===')
    print(f'{"Bucket":<35} {"IS N":>6} {"IS acc":>7} {"OOS N":>6} {"OOS acc":>7}')
    is_vel_r = stratify(is_events, 'pred_dir_reg_vel', VEL_SIGN_BINS)
    oos_vel_r = stratify(oos_events, 'pred_dir_reg_vel', VEL_SIGN_BINS)
    for i_s, o_s in zip(is_vel_r, oos_vel_r):
        is_acc = f'{i_s["acc"]:.1f}%' if i_s.get('valid') else '—'
        oos_acc = f'{o_s["acc"]:.1f}%' if o_s.get('valid') else '—'
        print(f'{i_s["label"]:<35} {i_s["n"]:>6,} {is_acc:>7} '
              f'{o_s["n"]:>6,} {oos_acc:>7}')

    # 2D joint
    print('\n=== Joint: chord_ratio x rejection_wick ===')
    is_joint = joint_stratify(is_events, 'reg_to_path', CHORD_BINS,
                               'rejection_wick_pct', WICK_BINS)
    oos_joint = joint_stratify(oos_events, 'reg_to_path', CHORD_BINS,
                                'rejection_wick_pct', WICK_BINS)

    # Wick × volume joint — the "inflection point" combo
    print('\n=== Joint: wick x volume (the inflection combo) ===')
    is_wv = joint_stratify(is_events, 'rejection_wick_pct', WICK_BINS,
                           'vol_ratio', VOL_BINS)
    oos_wv = joint_stratify(oos_events, 'rejection_wick_pct', WICK_BINS,
                            'vol_ratio', VOL_BINS)
    print('IS acc:')
    header2 = ['Wick \\ Vol'] + [b[0].split(' ')[0] for b in VOL_BINS]
    print('  ' + ' | '.join(f'{h:>14}' for h in header2))
    for row in is_wv:
        cells = [row['label'].split(' ')[0]]
        for c in row['cells']:
            if c['valid']:
                cells.append(f'{c["acc"]:.1f}%(n={c["n"]})')
            else:
                cells.append(f'—(n={c["n"]})')
        print('  ' + ' | '.join(f'{c:>14}' for c in cells))
    print('OOS acc:')
    print('  ' + ' | '.join(f'{h:>14}' for h in header2))
    for row in oos_wv:
        cells = [row['label'].split(' ')[0]]
        for c in row['cells']:
            if c['valid']:
                cells.append(f'{c["acc"]:.1f}%(n={c["n"]})')
            else:
                cells.append(f'—(n={c["n"]})')
        print('  ' + ' | '.join(f'{c:>14}' for c in cells))

    header = ['Chord \\ Wick'] + [b[0].split(' ')[0] for b in WICK_BINS]
    print('IS acc:')
    print('  ' + ' | '.join(f'{h:>10}' for h in header))
    for row in is_joint:
        cells = [row['label'].split(' ')[0]]
        for c in row['cells']:
            if c['valid']:
                cells.append(f'{c["acc"]:.1f}%(n={c["n"]})')
            else:
                cells.append(f'—(n={c["n"]})')
        print('  ' + ' | '.join(f'{c:>15}' for c in cells))
    print('OOS acc:')
    print('  ' + ' | '.join(f'{h:>10}' for h in header))
    for row in oos_joint:
        cells = [row['label'].split(' ')[0]]
        for c in row['cells']:
            if c['valid']:
                cells.append(f'{c["acc"]:.1f}%(n={c["n"]})')
            else:
                cells.append(f'—(n={c["n"]})')
        print('  ' + ' | '.join(f'{c:>15}' for c in cells))

    # MD
    out = [f'# Pivot direction accuracy — stratified by chord + wick', '']
    out.append(f'Zigzag threshold ${args.threshold}. Only pivots with |residual|>=0.5.')
    out.append('')
    out.append(f'IS events: {len(is_events):,} | OOS events: {len(oos_events):,}')
    out.append('')
    out.append(f'**Baseline accuracy (no stratification)**')
    out.append(f'- IS: **{accuracy(is_events):.1f}%**')
    out.append(f'- OOS: **{accuracy(oos_events):.1f}%**')
    out.append('')

    out.append('## 1D: by chord ratio (noise vs trend regime)')
    out.append('')
    out.append('| Regime | IS N | IS acc | OOS N | OOS acc |')
    out.append('|---|---:|---:|---:|---:|')
    for i_s, o_s in zip(is_chord, oos_chord):
        is_acc = f'{i_s["acc"]:.1f}%' if i_s.get('valid') else '—'
        oos_acc = f'{o_s["acc"]:.1f}%' if o_s.get('valid') else '—'
        out.append(f'| {i_s["label"]} | {i_s["n"]:,} | {is_acc} | '
                   f'{o_s["n"]:,} | {oos_acc} |')
    out.append('')

    out.append('## 1D: by rejection-wick % at pivot bar')
    out.append('')
    out.append('"Rejection wick" = the wick in the direction of the predicted bounce. '
               'LONG pred → lower wick. SHORT pred → upper wick.')
    out.append('')
    out.append('| Rejection wick | IS N | IS acc | OOS N | OOS acc |')
    out.append('|---|---:|---:|---:|---:|')
    for i_s, o_s in zip(is_wick, oos_wick):
        is_acc = f'{i_s["acc"]:.1f}%' if i_s.get('valid') else '—'
        oos_acc = f'{o_s["acc"]:.1f}%' if o_s.get('valid') else '—'
        out.append(f'| {i_s["label"]} | {i_s["n"]:,} | {is_acc} | '
                   f'{o_s["n"]:,} | {oos_acc} |')
    out.append('')

    out.append('## 2D: chord × wick (IS)')
    out.append('')
    hdr = ['Chord \\ Wick'] + [b[0] for b in WICK_BINS]
    out.append('| ' + ' | '.join(hdr) + ' |')
    out.append('|' + '|'.join(['---'] + ['---:'] * (len(hdr) - 1)) + '|')
    for row in is_joint:
        cells = [row['label']]
        for c in row['cells']:
            if c['valid']:
                cells.append(f'**{c["acc"]:.1f}%** (n={c["n"]})')
            else:
                cells.append(f'— (n={c["n"]})')
        out.append('| ' + ' | '.join(cells) + ' |')
    out.append('')

    out.append('## 2D: chord × wick (OOS)')
    out.append('')
    out.append('| ' + ' | '.join(hdr) + ' |')
    out.append('|' + '|'.join(['---'] + ['---:'] * (len(hdr) - 1)) + '|')
    for row in oos_joint:
        cells = [row['label']]
        for c in row['cells']:
            if c['valid']:
                cells.append(f'**{c["acc"]:.1f}%** (n={c["n"]})')
            else:
                cells.append(f'— (n={c["n"]})')
        out.append('| ' + ' | '.join(cells) + ' |')
    out.append('')

    # Volume 1D
    out.append('## 1D: by volume ratio (pivot volume / mean 20-bar lookback)')
    out.append('')
    out.append('| Vol regime | IS N | IS acc | OOS N | OOS acc |')
    out.append('|---|---:|---:|---:|---:|')
    for i_s, o_s in zip(is_vol, oos_vol):
        is_acc = f'{i_s["acc"]:.1f}%' if i_s.get('valid') else '—'
        oos_acc = f'{o_s["acc"]:.1f}%' if o_s.get('valid') else '—'
        out.append(f'| {i_s["label"]} | {i_s["n"]:,} | {is_acc} | '
                   f'{o_s["n"]:,} | {oos_acc} |')
    out.append('')

    # Velocity 1D — price
    out.append('## 1D: by predicted-direction price velocity (pts/bar over last 5 bars)')
    out.append('')
    out.append('"WITH prediction" = price velocity sign matches predicted direction. '
               'Positive values mean market already moving in the predicted direction at entry.')
    out.append('')
    out.append('| Bucket | IS N | IS acc | OOS N | OOS acc |')
    out.append('|---|---:|---:|---:|---:|')
    for i_s, o_s in zip(is_vel_p, oos_vel_p):
        is_acc = f'{i_s["acc"]:.1f}%' if i_s.get('valid') else '—'
        oos_acc = f'{o_s["acc"]:.1f}%' if o_s.get('valid') else '—'
        out.append(f'| {i_s["label"]} | {i_s["n"]:,} | {is_acc} | '
                   f'{o_s["n"]:,} | {oos_acc} |')
    out.append('')

    # Velocity 1D — regression
    out.append('## 1D: by predicted-direction regression velocity (pts/bar)')
    out.append('')
    out.append('| Bucket | IS N | IS acc | OOS N | OOS acc |')
    out.append('|---|---:|---:|---:|---:|')
    for i_s, o_s in zip(is_vel_r, oos_vel_r):
        is_acc = f'{i_s["acc"]:.1f}%' if i_s.get('valid') else '—'
        oos_acc = f'{o_s["acc"]:.1f}%' if o_s.get('valid') else '—'
        out.append(f'| {i_s["label"]} | {i_s["n"]:,} | {is_acc} | '
                   f'{o_s["n"]:,} | {oos_acc} |')
    out.append('')

    # 2D wick × volume
    out.append('## 2D: wick × volume (inflection combo — IS)')
    out.append('')
    hdr_wv = ['Wick \\ Vol'] + [b[0] for b in VOL_BINS]
    out.append('| ' + ' | '.join(hdr_wv) + ' |')
    out.append('|' + '|'.join(['---'] + ['---:'] * (len(hdr_wv) - 1)) + '|')
    for row in is_wv:
        cells = [row['label']]
        for c in row['cells']:
            if c['valid']:
                cells.append(f'**{c["acc"]:.1f}%** (n={c["n"]})')
            else:
                cells.append(f'— (n={c["n"]})')
        out.append('| ' + ' | '.join(cells) + ' |')
    out.append('')

    out.append('## 2D: wick × volume (inflection combo — OOS)')
    out.append('')
    out.append('| ' + ' | '.join(hdr_wv) + ' |')
    out.append('|' + '|'.join(['---'] + ['---:'] * (len(hdr_wv) - 1)) + '|')
    for row in oos_wv:
        cells = [row['label']]
        for c in row['cells']:
            if c['valid']:
                cells.append(f'**{c["acc"]:.1f}%** (n={c["n"]})')
            else:
                cells.append(f'— (n={c["n"]})')
        out.append('| ' + ' | '.join(cells) + ' |')
    out.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')


if __name__ == '__main__':
    main()
