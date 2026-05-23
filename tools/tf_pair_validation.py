"""TF Pair Validation Research — Child detects peak, parent validates.

Questions:
1. Can we measure peaks at each TF pair independently?
2. What does a validated peak (parent agrees) look like vs unvalidated?
3. Does validation predict longer/better trades in that direction?
4. Can cascade agreement be used as a REVERSAL signal (fade the cascade)?

Usage: python tools/tf_pair_validation.py [--data DATA/ATLAS_OOS]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.statistical_field_engine import StatisticalFieldEngine

TF_PAIRS = [('15s', '1m'), ('1m', '5m'), ('5m', '15m'), ('15m', '1h'), ('1h', '4h')]
TF_SECS = {'1s':1,'5s':5,'15s':15,'30s':30,'1m':60,'2m':120,'3m':180,
            '5m':300,'15m':900,'30m':1800,'1h':3600,'4h':14400}


def load_tf(data_root, tf_label):
    tf_dir = os.path.join(data_root, tf_label)
    if not os.path.isdir(tf_dir):
        return None, None
    files = sorted(f for f in os.listdir(tf_dir) if f.endswith('.parquet'))
    chunks = []
    for fn in files:
        df = pd.read_parquet(os.path.join(tf_dir, fn))
        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        chunks.append(df)
    df = pd.concat(chunks).sort_values('timestamp').reset_index(drop=True)
    engine = StatisticalFieldEngine()
    states = engine.batch_compute_states(df, use_cuda=True)
    return df['timestamp'].values.astype(np.float64), states


def get_peaks(states, timestamps):
    peaks = []
    prev_pc, prev_fm = 0.0, 0.0
    for i, s in enumerate(states):
        ms = s['state'] if isinstance(s, dict) else s
        pc = getattr(ms, 'P_at_center', 0.0) or 0.0
        fm = abs(getattr(ms, 'F_momentum', 0.0) or 0.0)
        raw_fm = getattr(ms, 'F_momentum', 0.0) or 0.0
        coh = getattr(ms, 'oscillation_entropy_normalized', 0.0) or 0.0
        vol = getattr(ms, 'volume_delta', 0.0) or 0.0
        price = getattr(ms, 'price', 0.0) or 0.0
        dmi_p = getattr(ms, 'dmi_plus', 0.0) or 0.0
        dmi_m = getattr(ms, 'dmi_minus', 0.0) or 0.0

        if i > 0 and prev_pc > 0.01:
            pc_d = (pc - prev_pc) / max(abs(prev_pc), 1e-6)
            fm_d = (fm - prev_fm) / max(abs(prev_fm), 1e-6) if prev_fm > 0.5 else 0.0
            if (pc_d > 0.05 or fm_d < -0.10) and coh > 0.55:
                direction = 'SHORT' if raw_fm > 0 else 'LONG'
                peaks.append({
                    'idx': i, 'ts': timestamps[i], 'direction': direction,
                    'pc_delta': pc_d, 'fm_delta': fm_d,
                    'volume': abs(vol), 'momentum': abs(raw_fm),
                    'coherence': coh, 'dmi_gap': abs(dmi_p - dmi_m),
                    'dmi_dir': 'LONG' if dmi_p > dmi_m else 'SHORT',
                    'price': price,
                })
        prev_pc = pc
        prev_fm = fm
    return peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS_OOS')
    parser.add_argument('--lookahead', type=int, default=20)
    args = parser.parse_args()

    print('=' * 70)
    print('TF PAIR VALIDATION RESEARCH')
    print('=' * 70)

    # Load all needed TFs
    tf_data = {}
    for child, parent in TF_PAIRS:
        for tf in (child, parent):
            if tf not in tf_data:
                print(f'  Loading {tf}...', end=' ', flush=True)
                ts, states = load_tf(args.data, tf)
                if ts is not None:
                    peaks = get_peaks(states, ts)
                    n_l = sum(1 for p in peaks if p['direction'] == 'LONG')
                    n_s = sum(1 for p in peaks if p['direction'] == 'SHORT')
                    tf_data[tf] = {'ts': ts, 'states': states, 'peaks': peaks}
                    print(f'{len(states):,} states, {len(peaks):,} peaks ({n_l}L/{n_s}S)')
                else:
                    print('MISSING')

    # Reference prices from 1m
    ref_ts = tf_data['1m']['ts']
    ref_prices = np.array([
        getattr(s['state'] if isinstance(s, dict) else s, 'price', 0.0)
        for s in tf_data['1m']['states']
    ])

    print()
    print('=' * 70)
    print('Q1: PAIR VALIDATION (child peak + parent agrees vs disagrees)')
    print('=' * 70)

    all_results = []

    for child_tf, parent_tf in TF_PAIRS:
        if child_tf not in tf_data or parent_tf not in tf_data:
            continue

        child_peaks = tf_data[child_tf]['peaks']
        parent_peaks = tf_data[parent_tf]['peaks']
        parent_tf_secs = TF_SECS[parent_tf]
        window = parent_tf_secs * 10

        validated = []
        unvalidated = []

        for cp in child_peaks:
            # Find recent parent peak
            parent_match = None
            for pp in reversed(parent_peaks):
                if 0 <= cp['ts'] - pp['ts'] <= window:
                    parent_match = pp
                    break

            # Future MFE from 1m
            ref_idx = int(np.searchsorted(ref_ts, cp['ts'], side='right')) - 1
            if ref_idx < 0 or ref_idx + args.lookahead >= len(ref_prices):
                continue

            entry = ref_prices[ref_idx]
            future = ref_prices[ref_idx + 1: ref_idx + args.lookahead + 1]
            if len(future) == 0:
                continue

            if cp['direction'] == 'LONG':
                mfe = (max(future) - entry) / 0.25
                mae = (entry - min(future)) / 0.25
                final = (future[-1] - entry) / 0.25
            else:
                mfe = (entry - min(future)) / 0.25
                mae = (max(future) - entry) / 0.25
                final = (entry - future[-1]) / 0.25

            r = {
                'pair': f'{child_tf}->{parent_tf}',
                'direction': cp['direction'],
                'mfe': mfe, 'mae': mae, 'final': final,
                'profitable': final > 0,
                'child_vol': cp['volume'], 'child_mom': cp['momentum'],
                'child_dmi_gap': cp['dmi_gap'],
            }

            if parent_match and parent_match['direction'] == cp['direction']:
                r['validated'] = True
                r['parent_dmi_gap'] = parent_match['dmi_gap']
                r['parent_vol'] = parent_match['volume']
                validated.append(r)
            else:
                r['validated'] = False
                r['parent_dmi_gap'] = parent_match['dmi_gap'] if parent_match else 0
                r['parent_vol'] = parent_match['volume'] if parent_match else 0
                unvalidated.append(r)

            all_results.append(r)

        if not validated and not unvalidated:
            continue

        # Stats
        def stats(lst, label):
            if not lst:
                return f'    {label:30s} n=    0'
            n = len(lst)
            wr = np.mean([r['profitable'] for r in lst]) * 100
            avg_mfe = np.mean([r['mfe'] for r in lst])
            avg_mae = np.mean([r['mae'] for r in lst])
            avg_final = np.mean([r['final'] for r in lst])
            gw = sum(r['final'] for r in lst if r['final'] > 0)
            gl = abs(sum(r['final'] for r in lst if r['final'] < 0))
            pf = gw / gl if gl > 0 else 0
            return (f'    {label:30s} n={n:>5}  WR={wr:>5.1f}%  '
                    f'MFE={avg_mfe:>6.1f}t  MAE={avg_mae:>6.1f}t  '
                    f'Final={avg_final:>+7.1f}t  PF={pf:.2f}')

        print(f'\n  PAIR: {child_tf} -> {parent_tf}')
        print(stats(validated, 'Validated (parent agrees)'))
        print(stats(unvalidated, 'Unvalidated (disagrees)'))

        if validated and unvalidated:
            val_wr = np.mean([r['profitable'] for r in validated]) * 100
            unval_wr = np.mean([r['profitable'] for r in unvalidated]) * 100
            val_mfe = np.mean([r['mfe'] for r in validated])
            unval_mfe = np.mean([r['mfe'] for r in unvalidated])
            print(f'    EDGE: {val_wr - unval_wr:+.1f}% WR, {val_mfe - unval_mfe:+.1f}t MFE')

    # Q2: What does a validated peak look like?
    print()
    print('=' * 70)
    print('Q2: VALIDATED vs UNVALIDATED PEAK CHARACTERISTICS')
    print('=' * 70)

    val_all = [r for r in all_results if r.get('validated', False)]
    unval_all = [r for r in all_results if not r.get('validated', False)]

    if val_all and unval_all:
        print(f'  {"Metric":>20s}  {"Validated":>12}  {"Unvalidated":>12}  {"Delta":>10}')
        for metric in ['child_vol', 'child_mom', 'child_dmi_gap', 'parent_dmi_gap', 'parent_vol']:
            v_mean = np.mean([r.get(metric, 0) for r in val_all])
            u_mean = np.mean([r.get(metric, 0) for r in unval_all])
            print(f'  {metric:>20s}  {v_mean:>12.1f}  {u_mean:>12.1f}  {v_mean - u_mean:>+10.1f}')

    # Q3: Cascade as reversal signal
    print()
    print('=' * 70)
    print('Q3: CASCADE AS REVERSAL SIGNAL (enter OPPOSITE to cascade)')
    print('=' * 70)

    # Count how many pairs agree at each 1m bar
    for i in range(len(ref_ts)):
        ts = ref_ts[i]
        n_long = 0
        n_short = 0
        for tf_label, data in tf_data.items():
            for p in reversed(data['peaks']):
                if 0 <= ts - p['ts'] <= 300:
                    if p['direction'] == 'LONG':
                        n_long += 1
                    else:
                        n_short += 1
                    break

    # Simplified: just check if reversing the cascade direction is profitable
    # Re-use the cascade results we already have
    cascade_file = 'reports/findings/resonance_cascade_accuracy.csv'
    if os.path.exists(cascade_file):
        casc = pd.read_csv(cascade_file)
        print(f'  Loaded {len(casc)} cascade signals from previous research')
        print()
        print(f'  {"Agree":>6} {"N":>6} {"Fade WR":>8} {"Fade Final":>11} {"Fade PF":>8} | {"Trend WR":>9} {"Trend PF":>9}')
        for n in sorted(casc['agreement'].unique()):
            sub = casc[casc['agreement'] == n]
            # Fade = enter OPPOSITE (negate final_ticks)
            fade_final = -sub['final_ticks']
            fade_wr = (fade_final > 0).mean() * 100
            fade_gw = fade_final[fade_final > 0].sum()
            fade_gl = abs(fade_final[fade_final < 0].sum())
            fade_pf = fade_gw / fade_gl if fade_gl > 0 else 0
            # Trend = original direction
            trend_wr = sub['profitable'].mean() * 100
            trend_gw = sub[sub['final_ticks'] > 0]['final_ticks'].sum()
            trend_gl = abs(sub[sub['final_ticks'] < 0]['final_ticks'].sum())
            trend_pf = trend_gw / trend_gl if trend_gl > 0 else 0
            label = ' <-- BEST FADE' if n >= 4 and fade_pf > 1.5 else ''
            print(f'  {n:>6} {len(sub):>6} {fade_wr:>7.1f}% {fade_final.mean():>+10.1f}t {fade_pf:>7.2f} | {trend_wr:>8.1f}% {trend_pf:>8.2f}{label}')

    # Q4: Direction bias from validated pairs
    print()
    print('=' * 70)
    print('Q4: CAN VALIDATED PAIRS IMPROVE DIRECTION BIAS?')
    print('=' * 70)

    for pair_label in [f'{c}->{p}' for c, p in TF_PAIRS]:
        pair_results = [r for r in all_results if r['pair'] == pair_label]
        if not pair_results:
            continue
        val = [r for r in pair_results if r.get('validated')]
        unval = [r for r in pair_results if not r.get('validated')]

        if not val:
            continue

        # For validated peaks: LONG vs SHORT performance
        val_long = [r for r in val if r['direction'] == 'LONG']
        val_short = [r for r in val if r['direction'] == 'SHORT']

        if val_long and val_short:
            l_wr = np.mean([r['profitable'] for r in val_long]) * 100
            s_wr = np.mean([r['profitable'] for r in val_short]) * 100
            l_final = np.mean([r['final'] for r in val_long])
            s_final = np.mean([r['final'] for r in val_short])
            print(f'  {pair_label:>12s}  LONG: WR={l_wr:.1f}% final={l_final:+.1f}t ({len(val_long)})'
                  f'  SHORT: WR={s_wr:.1f}% final={s_final:+.1f}t ({len(val_short)})')

    # Save
    out_dir = 'reports/findings'
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(all_results).to_csv(
        os.path.join(out_dir, 'tf_pair_validation.csv'), index=False)
    print(f'\n  Saved to {out_dir}/tf_pair_validation.csv')
    print('=' * 70)


if __name__ == '__main__':
    main()
