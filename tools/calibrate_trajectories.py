"""
Purpose-specific calibration for each TF's TrajectoryPredictor.

Each TF answers a different question. Calibration measures how well
it answers THAT question, not generic direction accuracy.

1h:  "Is the structural trend reliable for the next 4+ hours?"
15m: "When does 15m disagree with 1h AND be right?" (flip detection)
1m:  "At what confidence do oscillation entries produce positive PnL?"
15s: "When does 15s flip AND price actually moves 10+ ticks?" (real vs noise)
1s:  "Does 1s confirmation improve fill quality?"

Usage:
  python -m tools.calibrate_trajectories
"""
import gc
import glob
import json
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from core_v2.calibration import TrajectoryCalibrator, HorizonCalibrator

TICK = 0.25
LOOKBACK = 10
ATLAS_ROOT = 'DATA/ATLAS'
VAL_START = '2026-01-01'


def load_model(tf):
    """Load TrajectoryPredictor for a TF."""
    from core_v2.direction_cnn import TrajectoryPredictor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = f'checkpoints/trajectory_{tf}/best_model.pt'
    if not os.path.exists(path):
        return None, None, None, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    horizons = ckpt.get('horizons', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    model = TrajectoryPredictor(n_features=13, latent_dim=64, n_state=7, horizons=horizons).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, horizons, device, path


def get_predictions(model, feats, horizons, device):
    """Run model on all bars, return raw P(long) per horizon."""
    from training.train_direction import TrajectoryDataset, build_state_labels, build_multi_horizon_labels, N_FEAT_7D
    from torch.utils.data import DataLoader

    state_labels = build_state_labels(feats[:, :N_FEAT_7D], horizons[0])
    dir_labels = build_multi_horizon_labels(feats[:, :N_FEAT_7D], horizons)

    ds = TrajectoryDataset(feats, state_labels, dir_labels, lookback=LOOKBACK, max_forward=max(horizons))
    dl = DataLoader(ds, batch_size=1024, shuffle=False)

    all_p = []
    all_td = []
    with torch.no_grad():
        for x, ys, yd in dl:
            x = x.to(device)
            _, p_longs = model(x)
            all_p.append(p_longs.cpu().numpy())
            all_td.append(yd.numpy())

    return np.concatenate(all_p), np.concatenate(all_td)


def calibrate_1h(pred_p, true_dir, prices, timestamps):
    """1h purpose: Is the structural trend reliable for 4+ hours?

    Measures: when 1h says LONG with confidence X, does price go up over
    the next 4 hours? Not just next bar — sustained direction.
    """
    print(f"\n{'='*60}")
    print(f"1H CALIBRATION: Structural trend reliability")
    print(f"{'='*60}")

    n = len(pred_p)
    p_n1 = pred_p[:, 0]
    confidence = np.abs(p_n1 - 0.5) * 2

    # Sustained direction: is direction correct at ALL of n+1 through n+4?
    sustained = np.ones(n, dtype=bool)
    for hi in range(min(4, pred_p.shape[1])):
        correct = (pred_p[:, hi] > 0.5) == (true_dir[:, hi] > 0.5)
        sustained &= correct

    # Bin by confidence, measure sustained accuracy
    print(f"\n  Confidence vs Sustained Direction (4-bar):")
    print(f"  {'Conf Range':>15} {'N':>8} {'Sustained%':>10} {'1bar%':>8}")
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for lo, hi in bins:
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() > 10:
            sust_rate = sustained[mask].mean() * 100
            bar1_rate = ((p_n1[mask] > 0.5) == (true_dir[mask, 0] > 0.5)).mean() * 100
            print(f"  {lo:.1f} - {hi:.1f}:      {mask.sum():>8} {sust_rate:>9.1f}% {bar1_rate:>7.1f}%")

    # Key metric: at what confidence is sustained direction > 80%?
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = confidence >= thresh
        if mask.sum() > 10:
            rate = sustained[mask].mean() * 100
            print(f"  Conf >= {thresh}: sustained={rate:.1f}% (n={mask.sum()})")

    return {'type': '1h', 'metric': 'sustained_4bar'}


def calibrate_15m(pred_p_15m, true_dir_15m, pred_p_1h, true_dir_1h, timestamps_15m, timestamps_1h):
    """15m purpose: When does 15m disagree with 1h AND be right?

    This detects structural flips. 15m sees the session turning before
    the 1h bar closes.
    """
    print(f"\n{'='*60}")
    print(f"15M CALIBRATION: Structural flip detection")
    print(f"{'='*60}")

    # Align 15m predictions to nearest 1h prediction
    # For each 15m bar, find which 1h bar was last completed
    align_idx = np.searchsorted(timestamps_1h + 3600, timestamps_15m[:len(pred_p_15m)], side='left') - 1
    align_idx = np.clip(align_idx, 0, len(pred_p_1h) - 1)

    p_15m_n1 = pred_p_15m[:, 0]
    dir_15m = p_15m_n1 > 0.5
    conf_15m = np.abs(p_15m_n1 - 0.5) * 2

    p_1h_aligned = pred_p_1h[align_idx, 0]
    dir_1h = p_1h_aligned > 0.5

    # Disagreement: 15m says different direction than 1h
    disagree = dir_15m != dir_1h

    # When 15m disagrees, is 15m correct? (check actual direction)
    actual_dir_15m = true_dir_15m[:, 0] > 0.5

    disagree_correct = disagree & (dir_15m == actual_dir_15m)
    disagree_wrong = disagree & (dir_15m != actual_dir_15m)

    print(f"  Total 15m bars: {len(pred_p_15m)}")
    print(f"  15m agrees with 1h: {(~disagree).sum()} ({(~disagree).mean()*100:.1f}%)")
    print(f"  15m disagrees: {disagree.sum()} ({disagree.mean()*100:.1f}%)")
    if disagree.sum() > 0:
        print(f"    15m RIGHT when disagrees: {disagree_correct.sum()} ({disagree_correct.sum()/disagree.sum()*100:.1f}%)")
        print(f"    15m WRONG when disagrees: {disagree_wrong.sum()} ({disagree_wrong.sum()/disagree.sum()*100:.1f}%)")

    # By confidence when disagreeing
    print(f"\n  When 15m disagrees, accuracy by confidence:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask = disagree & (conf_15m >= thresh)
        if mask.sum() > 5:
            correct = (dir_15m[mask] == actual_dir_15m[mask]).mean() * 100
            print(f"    conf >= {thresh}: {correct:.1f}% correct (n={mask.sum()})")

    return {'type': '15m', 'metric': 'flip_detection'}


def calibrate_1m(pred_p, true_dir, prices, timestamps):
    """1m purpose: At what confidence do oscillation entries produce positive PnL?

    Simulates: enter when 1m says direction with confidence X,
    exit after 4 bars. What's the PnL?
    """
    print(f"\n{'='*60}")
    print(f"1M CALIBRATION: Oscillation entry profitability")
    print(f"{'='*60}")

    n = len(pred_p)
    p_n1 = pred_p[:, 0]
    confidence = np.abs(p_n1 - 0.5) * 2
    direction = p_n1 > 0.5  # True = long

    # Compute PnL for 4-bar hold from each bar
    pnl_4bar = np.zeros(n)
    for i in range(n):
        bar_idx = i + LOOKBACK  # index into prices
        if bar_idx + 4 >= len(prices):
            continue
        if direction[i]:  # long
            pnl_4bar[i] = (prices[bar_idx + 4] - prices[bar_idx]) / TICK
        else:  # short
            pnl_4bar[i] = (prices[bar_idx] - prices[bar_idx + 4]) / TICK

    print(f"\n  Confidence vs 4-bar PnL:")
    print(f"  {'Conf Range':>15} {'N':>8} {'Avg PnL':>10} {'WR%':>8} {'Total$':>10}")
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (confidence >= lo) & (confidence < hi) & (pnl_4bar != 0)
        if mask.sum() > 10:
            avg = pnl_4bar[mask].mean()
            wr = (pnl_4bar[mask] > 0).mean() * 100
            total = pnl_4bar[mask].sum() * 0.5
            print(f"  {lo:.1f} - {hi:.1f}:      {mask.sum():>8} {avg:>9.1f}t {wr:>7.1f}% ${total:>9,.0f}")

    # Key metric: minimum confidence for positive expected PnL
    print(f"\n  Confidence threshold for positive E[PnL]:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = (confidence >= thresh) & (pnl_4bar != 0)
        if mask.sum() > 10:
            avg = pnl_4bar[mask].mean()
            total = pnl_4bar[mask].sum() * 0.5
            print(f"    conf >= {thresh}: E[PnL]={avg:+.2f}t  total=${total:+,.0f}  n={mask.sum()}")

    return {'type': '1m', 'metric': 'entry_profitability'}


def calibrate_15s(pred_p, true_dir, prices, timestamps):
    """15s purpose: When does 15s flip AND price actually moves 10+ ticks?

    Separates real fast moves from noise. A 15s flip with no follow-through
    is just oscillation noise — don't exit.
    """
    print(f"\n{'='*60}")
    print(f"15S CALIBRATION: Real fast move detection")
    print(f"{'='*60}")

    n = len(pred_p)
    p_n1 = pred_p[:, 0]
    confidence = np.abs(p_n1 - 0.5) * 2
    direction = p_n1 > 0.5

    # Detect flips (direction changed from previous bar)
    flips = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if direction[i] != direction[i-1]:
            flips[i] = True

    # After a flip, did price move 10+ ticks in the flip direction within 60 bars (15 min)?
    move_threshold = 10  # ticks
    real_moves = np.zeros(n, dtype=bool)
    for i in range(n):
        if not flips[i]:
            continue
        bar_idx = i + LOOKBACK
        if bar_idx + 60 >= len(prices):
            continue
        future_prices = prices[bar_idx:bar_idx + 60]
        if direction[i]:  # flipped to long
            max_move = (future_prices.max() - prices[bar_idx]) / TICK
        else:  # flipped to short
            max_move = (prices[bar_idx] - future_prices.min()) / TICK
        if max_move >= move_threshold:
            real_moves[i] = True

    n_flips = flips.sum()
    n_real = real_moves.sum()
    print(f"  Total bars: {n}")
    print(f"  Direction flips: {n_flips} ({n_flips/n*100:.1f}%)")
    if n_flips > 0:
        print(f"  Real moves (10t+): {n_real}/{n_flips} ({n_real/n_flips*100:.1f}%)")

    # By confidence at flip
    print(f"\n  Flip confidence vs real move rate:")
    flip_conf = confidence[flips]
    flip_real = real_moves[flips]
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask = flip_conf >= thresh
        if mask.sum() > 5:
            rate = flip_real[mask].mean() * 100
            print(f"    conf >= {thresh}: {rate:.1f}% are real moves (n={mask.sum()})")

    return {'type': '15s', 'metric': 'real_move_detection'}


def calibrate_1s(pred_p, true_dir, prices, timestamps):
    """1s purpose: Does 1s confirmation improve fill quality?

    Compare: enter at 1m bar close vs enter when 1s also confirms.
    Measures ticks saved by waiting for 1s alignment.
    """
    print(f"\n{'='*60}")
    print(f"1S CALIBRATION: Fill quality improvement")
    print(f"{'='*60}")

    n = len(pred_p)
    p_n1 = pred_p[:, 0]
    confidence = np.abs(p_n1 - 0.5) * 2
    direction = p_n1 > 0.5

    # For each bar, compute 5-bar forward PnL in predicted direction
    pnl_5bar = np.zeros(n)
    for i in range(n):
        bar_idx = i + LOOKBACK
        if bar_idx + 5 >= len(prices):
            continue
        if direction[i]:
            pnl_5bar[i] = (prices[bar_idx + 5] - prices[bar_idx]) / TICK
        else:
            pnl_5bar[i] = (prices[bar_idx] - prices[bar_idx + 5]) / TICK

    # High confidence entries vs low confidence
    high_conf = confidence >= 0.7
    low_conf = confidence < 0.3
    mid_conf = (confidence >= 0.3) & (confidence < 0.7)

    print(f"  Entry quality by 1s confidence:")
    for label, mask in [("High (>0.7)", high_conf), ("Mid (0.3-0.7)", mid_conf), ("Low (<0.3)", low_conf)]:
        valid = mask & (pnl_5bar != 0)
        if valid.sum() > 10:
            avg = pnl_5bar[valid].mean()
            wr = (pnl_5bar[valid] > 0).mean() * 100
            print(f"    {label:<15}: avg={avg:+.2f}t  WR={wr:.1f}%  n={valid.sum()}")

    # Fill improvement: difference in avg PnL between high and low confidence
    if high_conf.sum() > 10 and low_conf.sum() > 10:
        improvement = pnl_5bar[high_conf & (pnl_5bar != 0)].mean() - pnl_5bar[low_conf & (pnl_5bar != 0)].mean()
        print(f"\n  Fill improvement (high - low conf): {improvement:+.2f} ticks")

    return {'type': '1s', 'metric': 'fill_quality'}


def main():
    from core_v2.statistical_field_engine import StatisticalFieldEngine
    from training.train_direction import extract_features_13d, N_FEAT_7D

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_ts = pd.Timestamp(VAL_START).timestamp()
    results = {}

    # --- 1h ---
    model, horizons, device, path = load_model('1h')
    if model:
        files = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1h', '*.parquet')))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
        val_mask = df['timestamp'].values >= val_ts
        feat_path = 'checkpoints/trajectory_1h/features_13d.npy'
        feats = np.load(feat_path) if os.path.exists(feat_path) else None
        if feats is None:
            sfe = StatisticalFieldEngine()
            states = sfe.batch_compute_states(df)
            feats = extract_features_13d(states, df)
            del states; gc.collect()
        feats_val = feats[val_mask]
        pred_p, true_d = get_predictions(model, feats_val, horizons, device)
        results['1h'] = calibrate_1h(pred_p, true_d, df[val_mask]['close'].values,
                                      df[val_mask]['timestamp'].values)

        # Standard calibration + save
        cal = TrajectoryCalibrator(horizons)
        cal.fit(pred_p, true_d)
        cal.save('checkpoints/trajectory_1h/calibration.json')
        print(f"  Saved: checkpoints/trajectory_1h/calibration.json")
        del model, feats; gc.collect()

    # --- 15m ---
    model, horizons, device, path = load_model('15m')
    if model:
        files = sorted(glob.glob(os.path.join(ATLAS_ROOT, '15m', '*.parquet')))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
        val_mask = df['timestamp'].values >= val_ts
        feat_path = 'checkpoints/trajectory_15m/features_13d.npy'
        feats = np.load(feat_path) if os.path.exists(feat_path) else None
        if feats is None:
            sfe = StatisticalFieldEngine()
            states = sfe.batch_compute_states(df)
            feats = extract_features_13d(states, df)
            del states; gc.collect()
        feats_val = feats[val_mask]
        pred_p_15m, true_d_15m = get_predictions(model, feats_val, horizons, device)

        # Need 1h predictions for flip detection
        model_1h, h_1h, _, _ = load_model('1h')
        if model_1h:
            files_1h = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1h', '*.parquet')))
            df_1h = pd.concat([pd.read_parquet(f) for f in files_1h], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
            val_mask_1h = df_1h['timestamp'].values >= val_ts
            feat_1h = np.load('checkpoints/trajectory_1h/features_13d.npy')
            feats_1h_val = feat_1h[val_mask_1h]
            pred_p_1h, true_d_1h = get_predictions(model_1h, feats_1h_val, h_1h, device)
            ts_1h = df_1h[val_mask_1h]['timestamp'].values
            ts_15m = df[val_mask]['timestamp'].values

            results['15m'] = calibrate_15m(pred_p_15m, true_d_15m, pred_p_1h, true_d_1h,
                                           ts_15m[:len(pred_p_15m)], ts_1h[:len(pred_p_1h)])
            del model_1h, feat_1h; gc.collect()

        cal = TrajectoryCalibrator(horizons)
        cal.fit(pred_p_15m, true_d_15m)
        cal.save('checkpoints/trajectory_15m/calibration.json')
        print(f"  Saved: checkpoints/trajectory_15m/calibration.json")
        del model, feats; gc.collect()

    # --- 1m ---
    model, horizons, device, path = load_model('1m')
    if model:
        files = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1m', '*.parquet')))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
        val_mask = df['timestamp'].values >= val_ts
        feat_path = 'checkpoints/trajectory_1m/features_13d.npy'
        feats = np.load(feat_path) if os.path.exists(feat_path) else None
        if feats is None:
            sfe = StatisticalFieldEngine()
            states = sfe.batch_compute_states(df)
            feats = extract_features_13d(states, df)
            del states; gc.collect()
        feats_val = feats[val_mask]
        pred_p, true_d = get_predictions(model, feats_val, horizons, device)
        results['1m'] = calibrate_1m(pred_p, true_d, df[val_mask]['close'].values,
                                      df[val_mask]['timestamp'].values)

        cal = TrajectoryCalibrator(horizons)
        cal.fit(pred_p, true_d)
        cal.save('checkpoints/trajectory_1m/calibration.json')
        print(f"  Saved: checkpoints/trajectory_1m/calibration.json")
        del model, feats; gc.collect()

    # --- 15s (use shards) ---
    model, horizons, device, path = load_model('15s')
    if model:
        print(f"\n  15s: loading val shards...")
        src_dir = 'checkpoints/direction_15s/shards'
        if not os.path.exists(src_dir):
            src_dir = 'checkpoints/trajectory_15s/shards'
        shard_files = sorted(glob.glob(os.path.join(src_dir, '2026_*_feat.npy')))  # val only
        all_p, all_d, all_prices = [], [], []
        for sf in shard_files:
            feats = np.load(sf)
            pred_p, true_d = get_predictions(model, feats, horizons, device)
            all_p.append(pred_p)
            all_d.append(true_d)
            # Load prices
            month = os.path.basename(sf).replace('_feat.npy', '')
            pq = os.path.join(ATLAS_ROOT, '15s', f'{month}.parquet')
            if os.path.exists(pq):
                _df = pd.read_parquet(pq)
                all_prices.append(_df['close'].values)
            del feats; gc.collect()

        if all_p:
            pred_p = np.concatenate(all_p)
            true_d = np.concatenate(all_d)
            prices = np.concatenate(all_prices)
            ts = np.zeros(len(pred_p))  # dummy timestamps
            results['15s'] = calibrate_15s(pred_p, true_d, prices, ts)

            cal = TrajectoryCalibrator(horizons)
            cal.fit(pred_p, true_d)
            cal.save('checkpoints/trajectory_15s/calibration.json')
            print(f"  Saved: checkpoints/trajectory_15s/calibration.json")
        del model; gc.collect()

    # --- 1s (use shards) ---
    model, horizons, device, path = load_model('1s')
    if model:
        print(f"\n  1s: loading val shards...")
        src_dir = 'checkpoints/direction_1s/shards'
        shard_files = sorted(glob.glob(os.path.join(src_dir, '2026_*_feat.npy')))
        all_p, all_d, all_prices = [], [], []
        for sf in shard_files:
            feats = np.load(sf)
            pred_p, true_d = get_predictions(model, feats, horizons, device)
            all_p.append(pred_p)
            all_d.append(true_d)
            month = os.path.basename(sf).replace('_feat.npy', '')
            pq = os.path.join(ATLAS_ROOT, '1s', f'{month}.parquet')
            if os.path.exists(pq):
                _df = pd.read_parquet(pq)
                all_prices.append(_df['close'].values)
            del feats; gc.collect()

        if all_p:
            pred_p = np.concatenate(all_p)
            true_d = np.concatenate(all_d)
            prices = np.concatenate(all_prices)
            ts = np.zeros(len(pred_p))
            results['1s'] = calibrate_1s(pred_p, true_d, prices, ts)

            cal = TrajectoryCalibrator(horizons)
            cal.fit(pred_p, true_d)
            cal.save('checkpoints/trajectory_1s/calibration.json')
            print(f"  Saved: checkpoints/trajectory_1s/calibration.json")
        del model; gc.collect()

    # Summary + report file
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("CALIBRATION COMPLETE")
    report_lines.append("=" * 60)
    for tf in ['1h', '15m', '1m', '15s', '1s']:
        cal_path = f'checkpoints/trajectory_{tf}/calibration.json'
        if os.path.exists(cal_path):
            cal = TrajectoryCalibrator.load(cal_path)
            zones = cal.chop_zones
            line = f"  {tf}: chop_zone n+1=[{zones[0][0]:.3f}, {zones[0][1]:.3f}]"
            report_lines.append(line)
            print(line)
        else:
            line = f"  {tf}: no calibration"
            report_lines.append(line)
            print(line)

    # Write full report (capture stdout replay + structured data)
    report_path = 'reports/findings/calibration_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w') as f:
        f.write(f"Calibration Report - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Validation period: >= {VAL_START}\n\n")

        for tf in ['1h', '15m', '1m', '15s', '1s']:
            cal_path = f'checkpoints/trajectory_{tf}/calibration.json'
            if os.path.exists(cal_path):
                cal = TrajectoryCalibrator.load(cal_path)
                f.write(f"\n{'='*60}\n")
                f.write(f"{tf.upper()} CALIBRATION\n")
                f.write(f"{'='*60}\n")
                f.write(f"  Purpose: {results.get(tf, {}).get('metric', 'unknown')}\n")
                f.write(f"  Horizons: {cal.horizons}\n\n")
                f.write(f"  Per-horizon calibration:\n")
                for hi, h in enumerate(cal.horizons):
                    c = cal.calibrators[hi]
                    f.write(f"    n+{h}: a={c.a:.4f} b={c.b:.4f} "
                            f"chop=[{c.chop_low:.3f}, {c.chop_high:.3f}]\n")
                f.write(f"\n  Chop zone n+1: [{cal.chop_zones[0][0]:.3f}, {cal.chop_zones[0][1]:.3f}]\n")

                # Recommended thresholds
                c0 = cal.calibrators[0]
                # Confidence where calibrated P(D) reaches 60%, 70%, 80%, 90%
                from scipy.special import logit
                f.write(f"\n  Recommended confidence thresholds (raw P(long)):\n")
                for target in [0.6, 0.7, 0.8, 0.9, 0.95]:
                    if c0.a != 0:
                        raw_thresh = (logit(target) - c0.b) / c0.a
                        raw_thresh = max(0, min(1, raw_thresh))
                        f.write(f"    {target*100:.0f}% calibrated -> raw P(long) >= {raw_thresh:.3f} "
                                f"(confidence >= {abs(raw_thresh - 0.5)*2:.3f})\n")

        f.write(f"\n\n{'='*60}\n")
        f.write(f"SUMMARY\n")
        f.write(f"{'='*60}\n")
        for line in report_lines:
            f.write(line + "\n")

    print(f"\n  Report saved: {report_path}")


if __name__ == '__main__':
    main()
