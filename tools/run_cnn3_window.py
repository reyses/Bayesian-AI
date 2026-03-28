"""Run CNN3 three-layer simulation on a specific date window.

Usage:
  python tools/run_cnn3_window.py --start 2026-03-25 --end 2026-03-26
  python tools/run_cnn3_window.py  (defaults to Mar 25 9PM -> Mar 26 9PM ET)
"""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2026-03-25',
                        help='Start date YYYY-MM-DD (9PM ET session start)')
    parser.add_argument('--end', default='2026-03-26',
                        help='End date YYYY-MM-DD (9PM ET session end)')
    parser.add_argument('--min-hold', type=int, default=5)
    parser.add_argument('--max-hold', type=int, default=10)
    parser.add_argument('--hard-sl', type=int, default=40)
    parser.add_argument('--conf', type=float, default=3.0)
    args = parser.parse_args()

    start_ts = datetime.strptime(args.start, '%Y-%m-%d').replace(
        hour=21, tzinfo=ET).timestamp()
    end_ts = datetime.strptime(args.end, '%Y-%m-%d').replace(
        hour=21, tzinfo=ET).timestamp()
    warmup_ts = start_ts - 300 * 60

    print(f"Window: {args.start} 9PM -> {args.end} 9PM ET")
    print(f"Hold: {args.min_hold}-{args.max_hold} bars | SL: {args.hard_sl}t | Conf: {args.conf}")

    # Load 1m data with warmup
    files = sorted(glob.glob('DATA/ATLAS/1m/*.parquet'))
    df_all = pd.concat([pd.read_parquet(f) for f in files[-3:]], ignore_index=True)
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    df = df_all[(df_all['timestamp'] >= warmup_ts) & (df_all['timestamp'] < end_ts)].reset_index(drop=True)
    window_start_idx = int((df['timestamp'] >= start_ts).argmax())
    print(f"Bars: {len(df):,} (warmup={window_start_idx}, window={len(df)-window_start_idx})")

    # 13D features
    from core.statistical_field_engine import StatisticalFieldEngine
    from training.train_trade_cnn import (
        extract_features_13d, MTF_TFS, extract_4_features_from_sfe,
        extract_4_features_from_raw, load_tf_data, build_alignment_indices,
        assemble_features_29d,
    )

    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)
    feats_13d = extract_features_13d(states, df)
    del states; gc.collect()

    # MTF features
    timestamps_1m = df['timestamp'].values.astype(np.int64)
    mtf_data = {}
    tf_warmup = warmup_ts - 7200
    for tf in MTF_TFS:
        df_tf = load_tf_data('DATA/ATLAS', tf)
        df_tf = df_tf[(df_tf['timestamp'] >= tf_warmup) & (df_tf['timestamp'] < end_ts)].reset_index(drop=True)
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
        print(f"  {tf}: {len(df_tf):,} bars")

    alignment = build_alignment_indices(timestamps_1m, mtf_data)
    feats_29d = assemble_features_29d(feats_13d, mtf_data, alignment)
    print(f"29D features: {feats_29d.shape}")
    del mtf_data; gc.collect()

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

    # Simulate
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    timestamps = df['timestamp'].values

    trades = []
    in_trade = False
    l3_retreats = 0

    for i in tqdm(range(max(LOOKBACK, window_start_idx), len(feats_29d)), desc="Sim"):
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

            if pnl_worst <= -args.hard_sl:
                trades.append({'pnl': -args.hard_sl, 'exit': 'hard_sl', 'bars': bars_held,
                               'side': side, 'entry': entry_fill, 'price': price})
                in_trade = False
                continue

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
                trades.append({'pnl': pnl_close, 'exit': 'retreat', 'bars': bars_held,
                               'side': side, 'entry': entry_fill, 'price': price})
                in_trade = False
                l3_retreats += 1
                continue

            if bars_held >= predicted_hold:
                trades.append({'pnl': pnl_close, 'exit': 'duration', 'bars': bars_held,
                               'side': side, 'entry': entry_fill, 'price': price})
                in_trade = False
            continue

        if confidence <= args.conf:
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

    # Results
    total_pnl = sum(t['pnl'] for t in trades)
    n = len(trades)
    w = len([t for t in trades if t['pnl'] > 0])
    wr = w / n * 100 if n > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"CNN3 — {args.start} 9PM to {args.end} 9PM ET (24 hours)")
    print(f"{'=' * 60}")
    print(f"  Trades: {n}")
    print(f"  WR: {wr:.1f}%")
    print(f"  PnL: {total_pnl:.0f}t (${total_pnl * 0.5:,.2f})")
    print(f"  L3 retreats: {l3_retreats}")
    if n > 0:
        print(f"  Avg hold: {np.mean([t['bars'] for t in trades]):.1f} bars")
        print(f"  Avg PnL/trade: {total_pnl / n:.1f}t (${total_pnl * 0.5 / n:.2f})")

    exits = {}
    for t in trades:
        e = t['exit']
        if e not in exits:
            exits[e] = {'n': 0, 'pnl': 0}
        exits[e]['n'] += 1
        exits[e]['pnl'] += t['pnl']
    print(f"\n  EXIT BREAKDOWN:")
    for e, v in sorted(exits.items(), key=lambda x: x[1]['pnl']):
        print(f"    {e:<12} {v['n']:>4} trades  ${v['pnl'] * 0.5:>10,.2f}")


if __name__ == '__main__':
    main()
