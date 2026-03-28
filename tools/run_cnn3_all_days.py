"""Run CNN3 simulation on every trading day individually.

Outputs CSV to reports/findings/cnn3_daily_results.csv
Usage: python -m tools.run_cnn3_all_days
"""
import gc
import glob
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

TICK = 0.25
LOOKBACK = 10
ET = timezone(timedelta(hours=-4))


def main():
    # Load all 1m data + compute features ONCE
    print("Loading all data and computing features...")
    from core.statistical_field_engine import StatisticalFieldEngine
    from training.train_trade_cnn import (
        extract_features_13d, MTF_TFS, extract_4_features_from_sfe,
        extract_4_features_from_raw, load_tf_data, build_alignment_indices,
        assemble_features_29d, N_FEAT_29D,
    )

    # Check for cached 29D features
    cache_path = 'checkpoints/trade_cnn_10/29d/is_features_29d_raw.npy'
    if os.path.exists(cache_path):
        print(f"Loading cached 29D features from {cache_path}")
        feats_29d = np.load(cache_path)
        files = sorted(glob.glob('DATA/ATLAS/1m/*.parquet'))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        # Trim to match cached features length
        if len(df) > len(feats_29d):
            df = df.iloc[:len(feats_29d)].reset_index(drop=True)
        elif len(feats_29d) > len(df):
            feats_29d = feats_29d[:len(df)]
        print(f"  Bars: {len(df):,}, Features: {feats_29d.shape}")
    else:
        print("No cached features — building from scratch (this takes a minute)...")
        files = sorted(glob.glob('DATA/ATLAS/1m/*.parquet'))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        sfe = StatisticalFieldEngine()
        states = sfe.batch_compute_states(df)
        feats_13d = extract_features_13d(states, df)
        del states; gc.collect()

        timestamps_1m = df['timestamp'].values.astype(np.int64)
        mtf_data = {}
        for tf in MTF_TFS:
            df_tf = load_tf_data('DATA/ATLAS', tf)
            if tf == '1s':
                feats_tf = extract_4_features_from_raw(df_tf)
            else:
                sfe2 = StatisticalFieldEngine()
                states_tf = sfe2.batch_compute_states(df_tf)
                feats_tf = extract_4_features_from_sfe(states_tf, df_tf)
                del states_tf
            mtf_data[tf] = {
                'feats': feats_tf,
                'timestamps': df_tf['timestamp'].values.astype(np.int64),
                'df': df_tf,
            }
        alignment = build_alignment_indices(timestamps_1m, mtf_data)
        feats_29d = assemble_features_29d(feats_13d, mtf_data, alignment)
        del mtf_data, feats_13d; gc.collect()
        print(f"  29D features: {feats_29d.shape}")

    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    timestamps = df['timestamp'].values

    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from core.trade_cnn import StatePredictor
    from core.trade_selector import DurationPredictor
    from core.trade_retreat import RetreatPredictor

    ckpt = torch.load('checkpoints/trade_cnn_10/best_model.pt', map_location=device, weights_only=False)
    l1 = StatePredictor(n_features=29, latent_dim=64, n_labels=7).to(device)
    l1.load_state_dict(ckpt['model_state']); l1.eval()

    l2c = torch.load('checkpoints/trade_cnn_10/29d/l2_model.pt', map_location=device, weights_only=False)
    l2 = DurationPredictor(input_dim=l2c['input_dim']).to(device)
    l2.load_state_dict(l2c['model_state']); l2.eval()

    l3c = torch.load('checkpoints/trade_cnn_10/29d/l3_model.pt', map_location=device, weights_only=False)
    l3 = RetreatPredictor(input_dim=l3c['input_dim']).to(device)
    l3.load_state_dict(l3c['model_state']); l3.eval()

    # Split into days
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    day_groups = []
    for date, group in df.groupby('date'):
        day_groups.append({'date': date, 'start': group.index[0], 'end': group.index[-1]})

    print(f"Trading days: {len(day_groups)}")

    # Config
    HARD_SL = 40
    CONF_THRESHOLD = 3.0

    results = []

    for day in tqdm(day_groups, desc="Days"):
        d_start = day['start']
        d_end = day['end']
        n_bars = d_end - d_start + 1

        if n_bars < LOOKBACK + 10:
            continue

        trades = []
        in_trade = False
        sl_count = retreat_count = dur_count = 0

        for i in range(d_start + LOOKBACK, d_end + 1):
            if i >= len(feats_29d):
                break

            x = feats_29d[i - LOOKBACK:i]
            x_t = torch.FloatTensor(x).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = l1(x_t).cpu().numpy()[0]

            price = prices[i]
            high = highs[i]
            low = lows[i]
            pred_dmi = pred[0]
            confidence = abs(pred_dmi)
            pred_dir = 'long' if pred_dmi > 0 else 'short'

            if in_trade:
                if side == 'long':
                    peak_price = max(peak_price, high)
                    pnl_close = (price - entry_fill) / TICK
                    pnl_worst = (low - entry_fill) / TICK
                else:
                    peak_price = min(peak_price, low)
                    pnl_close = (entry_fill - price) / TICK
                    pnl_worst = (entry_fill - high) / TICK

                peak_pnl = (peak_price - entry_fill) / TICK if side == 'long' \
                    else (entry_fill - peak_price) / TICK
                drawdown = peak_pnl - pnl_close
                bars_held = i - entry_bar

                # Hard SL
                if pnl_worst <= -HARD_SL:
                    trades.append({'pnl': -HARD_SL, 'exit': 'hard_sl', 'bars': bars_held})
                    in_trade = False; sl_count += 1
                    continue

                # L3 retreat
                side_agree = 1.0 if pred_dir == side else -1.0
                l3_feat = np.array([
                    pnl_close, peak_pnl, drawdown, bars_held / max(1, predicted_hold),
                    pred_dmi, confidence, side_agree,
                    float(feats_29d[i, 4]), float(feats_29d[i, 5]), float(feats_29d[i, 2]),
                    float(feats_29d[i, 9]), float(feats_29d[i, 1]),
                ], dtype=np.float32)
                with torch.no_grad():
                    p_retreat = l3(torch.FloatTensor(l3_feat).unsqueeze(0).to(device)).item()

                if p_retreat > 0.5:
                    trades.append({'pnl': pnl_close, 'exit': 'retreat', 'bars': bars_held})
                    in_trade = False; retreat_count += 1
                    continue

                # Duration
                if bars_held >= predicted_hold:
                    trades.append({'pnl': pnl_close, 'exit': 'duration', 'bars': bars_held})
                    in_trade = False; dur_count += 1
                continue

            # Entry
            if confidence <= CONF_THRESHOLD:
                continue

            l2_feat = np.zeros(15, dtype=np.float32)
            l2_feat[0:7] = pred[0:7]
            l2_feat[7:11] = feats_29d[i, 7:11]
            ts_sec = int(timestamps[i]) % 86400
            l2_feat[11] = ts_sec / 86400.0
            l2_feat[12] = max(0, (23 * 3600 - ts_sec)) / 3600.0
            l2_feat[13] = confidence
            l2_feat[14] = 0.5
            with torch.no_grad():
                p_take, hold_bars = l2(torch.FloatTensor(l2_feat).unsqueeze(0).to(device))
            if p_take.item() < 0.5:
                continue

            side = pred_dir
            entry_fill = price
            entry_bar = i
            peak_price = price
            predicted_hold = max(3, int(hold_bars.item()))
            in_trade = True

        # Flush open trade
        if in_trade:
            final_pnl = (prices[d_end] - entry_fill) / TICK if side == 'long' \
                else (entry_fill - prices[d_end]) / TICK
            trades.append({'pnl': final_pnl, 'exit': 'eod', 'bars': d_end - entry_bar})

        # Day summary
        n = len(trades)
        total_pnl = sum(t['pnl'] for t in trades)
        wins = len([t for t in trades if t['pnl'] > 0])
        avg_hold = np.mean([t['bars'] for t in trades]) if trades else 0
        sl_pnl = sum(t['pnl'] for t in trades if t['exit'] == 'hard_sl')
        ret_pnl = sum(t['pnl'] for t in trades if t['exit'] == 'retreat')
        dur_pnl = sum(t['pnl'] for t in trades if t['exit'] == 'duration')

        results.append({
            'date': str(day['date']),
            'bars': n_bars,
            'trades': n,
            'wr': wins / n * 100 if n > 0 else 0,
            'pnl_ticks': total_pnl,
            'pnl_dollars': total_pnl * 0.5,
            'avg_pnl': total_pnl / n if n > 0 else 0,
            'avg_hold': avg_hold,
            'sl_count': sl_count,
            'sl_pnl': sl_pnl * 0.5,
            'retreat_count': retreat_count,
            'retreat_pnl': ret_pnl * 0.5,
            'dur_count': dur_count,
            'dur_pnl': dur_pnl * 0.5,
        })

    # Save CSV
    df_results = pd.DataFrame(results)
    out_path = 'reports/findings/cnn3_daily_results.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df_results)} days)")

    # Summary
    total = df_results['pnl_dollars'].sum()
    n_days = len(df_results)
    pos_days = (df_results['pnl_dollars'] > 0).sum()
    neg_days = (df_results['pnl_dollars'] < 0).sum()
    best = df_results.loc[df_results['pnl_dollars'].idxmax()]
    worst = df_results.loc[df_results['pnl_dollars'].idxmin()]

    print(f"\n{'=' * 60}")
    print(f"CNN3 DAILY SUMMARY ({n_days} days)")
    print(f"{'=' * 60}")
    print(f"  Total PnL: ${total:,.2f}")
    print(f"  $/day avg: ${total / n_days:,.2f}")
    print(f"  Positive days: {pos_days}/{n_days} ({pos_days / n_days * 100:.0f}%)")
    print(f"  Negative days: {neg_days}/{n_days}")
    print(f"  Best day: {best['date']} ${best['pnl_dollars']:+,.2f}")
    print(f"  Worst day: {worst['date']} ${worst['pnl_dollars']:+,.2f}")
    print(f"  Avg trades/day: {df_results['trades'].mean():.0f}")
    print(f"  Avg hold: {df_results['avg_hold'].mean():.1f} bars")

    # Monthly breakdown
    df_results['month'] = df_results['date'].str[:7]
    print(f"\n  MONTHLY:")
    for month, grp in df_results.groupby('month'):
        m_pnl = grp['pnl_dollars'].sum()
        m_days = len(grp)
        m_pos = (grp['pnl_dollars'] > 0).sum()
        print(f"    {month}: ${m_pnl:>10,.2f} ({m_days} days, {m_pos} green, ${m_pnl / m_days:>+.0f}/day)")


if __name__ == '__main__':
    main()
