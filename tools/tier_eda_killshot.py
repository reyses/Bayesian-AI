"""
KILL_SHOT per-tier EDA — what separates good from bad, what makes bad REALLY bad.

KILL_SHOT entry rule (from training/nightmare_blended.py):
  wick_5m > 0.83 AND wick_15m > 0.77 AND NOT h1_aligned
  (wick rejection without higher-TF confirmation)

Classifies each iso trade into KILL_SHOT bucket, then surgically dissects:
  1. Segment: winners (pnl>$5), small_losers (-$5 to 0), mid_losers (-$15 to -$5),
     tail_losers (pnl<-$15) — the "what makes bad really bad" bucket.
  2. Entry feature separators: median/IQR per segment for every 91D feature,
     rank by winner-vs-tail separation.
  3. Peak behavior: when does peak arrive, how long does it last, giveback shape.
  4. Regime shift within trade: feature state at entry vs at peak vs at exit;
     find the signature of "thesis dying" in tail losers.
  5. Exit reason distribution per segment.

Writes reports/findings/tier_eda_killshot_<ts>.md with full tables.

Usage:
    python tools/tier_eda_killshot.py
"""
import os
import sys
import pickle
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.features import FEATURE_NAMES  # 91 named columns

# 91D index helpers
N_CORE = 12
HELPER_START = 72
N_HELPER = 3
TF_15S, TF_1M, TF_5M, TF_15M, TF_1H, TF_1D = 0, 1, 2, 3, 4, 5


def core_idx(tf, c):
    return tf * N_CORE + c


def help_idx(tf, h):
    return HELPER_START + tf * N_HELPER + h


# KILL_SHOT thresholds (from nn_v2 nightmare_blended.py)
WICK_5M_MIN = 0.83
WICK_15M_MIN = 0.77
H1_Z_MIN = 1.0


def is_killshot(feat, z):
    """KILL_SHOT: wick rejection + NO 1h alignment."""
    wick_5m = feat[help_idx(TF_5M, 2)]
    wick_15m = feat[help_idx(TF_15M, 2)]
    h1_z = feat[core_idx(TF_1H, 0)]
    fade_dir = 'short' if z > 0 else 'long'
    has_wick = wick_5m > WICK_5M_MIN and wick_15m > WICK_15M_MIN
    h1_aligned = ((fade_dir == 'long' and h1_z < -H1_Z_MIN)
                  or (fade_dir == 'short' and h1_z > H1_Z_MIN))
    return has_wick and not h1_aligned


def segment_trade(pnl):
    if pnl >= 5:
        return 'winner'
    if pnl < -15:
        return 'tail_loser'
    if pnl < -5:
        return 'mid_loser'
    return 'small_loser'  # -5 <= pnl < 5 (covers break-even + tiny losses)


def summarize_segment(trades):
    pnls = np.array([t['pnl'] for t in trades])
    peaks = np.array([t['peak'] for t in trades])
    held = np.array([t['held'] for t in trades])
    return {
        'n': len(trades),
        'mean_pnl': pnls.mean() if len(pnls) else 0,
        'total_pnl': pnls.sum(),
        'mean_peak': peaks.mean() if len(peaks) else 0,
        'median_peak': np.median(peaks) if len(peaks) else 0,
        'mean_held': held.mean() if len(held) else 0,
        'median_held': np.median(held) if len(held) else 0,
        'mean_giveback': (peaks - pnls).mean() if len(pnls) else 0,
    }


def feature_separation(segmented, feature_names):
    """For each feature, compute median per segment and a winner-vs-tail Cohen d.
    Returns a list of (d_abs, row_dict) sorted by magnitude.
    """
    results = []
    winners = segmented.get('winner', [])
    tails = segmented.get('tail_loser', [])
    if not winners or not tails:
        return results

    win_X = np.stack([t['entry_79d'][:91] for t in winners])
    tail_X = np.stack([t['entry_79d'][:91] for t in tails])

    for i, name in enumerate(feature_names):
        w = win_X[:, i]
        t = tail_X[:, i]
        mu_w, mu_t = w.mean(), t.mean()
        sd_w, sd_t = w.std(), t.std()
        pooled = np.sqrt((sd_w ** 2 + sd_t ** 2) / 2) + 1e-9
        d = (mu_w - mu_t) / pooled
        row = {
            'feature': name,
            'winner_median': np.median(w),
            'tail_median': np.median(t),
            'winner_mean': mu_w,
            'tail_mean': mu_t,
            'cohen_d': d,
        }
        results.append((abs(d), row))
    results.sort(key=lambda x: -x[0])
    return results


def path_analysis(segmented):
    """For each segment, summarize peak timing + regime shift at exit."""
    out = {}
    for seg_name, trades in segmented.items():
        peaks_bar, peak_to_close_bars = [], []
        feat_at_peak = []
        feat_at_exit = []
        feat_at_entry = []
        for t in trades:
            path = t.get('path', [])
            if not path:
                continue
            pnls = np.array([p.get('pnl', 0.0) for p in path])
            if len(pnls) == 0:
                continue
            peak_idx = int(np.argmax(pnls))
            peaks_bar.append(peak_idx)
            peak_to_close_bars.append(len(path) - 1 - peak_idx)
            # Use features as stored in path; peak is per 5s bar
            if 'features' in path[peak_idx] and len(path[peak_idx]['features']) >= 91:
                feat_at_peak.append(path[peak_idx]['features'])
            if 'features' in path[-1] and len(path[-1]['features']) >= 91:
                feat_at_exit.append(path[-1]['features'])
            feat_at_entry.append(t['entry_79d'][:91])
        out[seg_name] = {
            'peak_bar_mean': np.mean(peaks_bar) if peaks_bar else 0,
            'peak_bar_median': np.median(peaks_bar) if peaks_bar else 0,
            'peak_to_close_mean': np.mean(peak_to_close_bars) if peak_to_close_bars else 0,
            'feat_entry': np.stack(feat_at_entry) if feat_at_entry else None,
            'feat_peak': np.stack(feat_at_peak) if feat_at_peak else None,
            'feat_exit': np.stack(feat_at_exit) if feat_at_exit else None,
        }
    return out


def regime_shift_table(path_stats, feature_names, feature_subset_idx):
    """For each segment and each subset feature, show mean at entry/peak/exit."""
    rows = []
    for seg_name, stats in path_stats.items():
        for idx in feature_subset_idx:
            name = feature_names[idx]
            row = {'segment': seg_name, 'feature': name}
            for key, arr in [('entry', stats.get('feat_entry')),
                             ('peak', stats.get('feat_peak')),
                             ('exit', stats.get('feat_exit'))]:
                row[key] = np.mean(arr[:, idx]) if arr is not None else np.nan
            row['entry_to_exit'] = row['exit'] - row['entry']
            row['peak_to_exit'] = row['exit'] - row['peak']
            rows.append(row)
    return rows


def main():
    with open('training_iso/output/trades/iso_is.pkl', 'rb') as f:
        trades = pickle.load(f)
    print(f'Loaded {len(trades):,} iso trades')

    # Classify KILL_SHOT subset
    killshot = []
    for t in trades:
        entry = t.get('entry_79d', [])
        if len(entry) < 91:
            continue
        feat = np.asarray(entry, dtype=np.float32)
        z = float(feat[core_idx(TF_1M, 0)])
        if is_killshot(feat, z):
            killshot.append(t)
    print(f'KILL_SHOT trades: {len(killshot):,} '
          f'({len(killshot)/len(trades)*100:.1f}% of iso)')
    if not killshot:
        print('No KILL_SHOT trades.')
        return

    # Segment
    segmented = defaultdict(list)
    for t in killshot:
        segmented[segment_trade(t['pnl'])].append(t)

    print('\nSegment summary:')
    print(f'  {"Segment":<13} {"N":>5} {"MeanPnL":>9} {"Total":>10} '
          f'{"MeanPeak":>10} {"MedianPeak":>11} {"MeanGive":>10} {"MedHeld":>9}')
    for seg in ['winner', 'small_loser', 'mid_loser', 'tail_loser']:
        sub = segmented.get(seg, [])
        if not sub:
            print(f'  {seg:<13} 0  (no trades)')
            continue
        s = summarize_segment(sub)
        print(f'  {seg:<13} {s["n"]:>5,} ${s["mean_pnl"]:>+7.2f} '
              f'${s["total_pnl"]:>+9,.0f} ${s["mean_peak"]:>+8.2f} '
              f'${s["median_peak"]:>+9.2f} ${s["mean_giveback"]:>+8.2f} '
              f'{s["median_held"]:>8.0f}m')

    # Exit reason distribution by segment
    print('\nExit reason by segment:')
    for seg in ['winner', 'small_loser', 'mid_loser', 'tail_loser']:
        sub = segmented.get(seg, [])
        if not sub:
            continue
        reasons = Counter(t.get('exit_reason', '?') for t in sub)
        top = ', '.join(f'{r}:{c}' for r, c in reasons.most_common(4))
        print(f'  {seg:<13}: {top}')

    # Feature separation (winner vs tail_loser)
    print('\nTop 20 feature separators (|Cohen d| winner-vs-tail):')
    print(f'  {"feature":<28} {"winner_med":>12} {"tail_med":>12} {"d":>8}')
    seps = feature_separation(segmented, FEATURE_NAMES)
    for _, r in seps[:20]:
        print(f'  {r["feature"]:<28} {r["winner_median"]:>12.4f} '
              f'{r["tail_median"]:>12.4f} {r["cohen_d"]:>+8.3f}')

    # Peak / path analysis
    print('\nPath analysis (peak timing + regime shift):')
    path_stats = path_analysis(segmented)
    for seg in ['winner', 'small_loser', 'mid_loser', 'tail_loser']:
        stats = path_stats.get(seg)
        if not stats:
            continue
        print(f'  {seg:<13}: peak_bar_median={stats["peak_bar_median"]:.0f} '
              f'peak_to_close_mean={stats["peak_to_close_mean"]:.1f} bars')

    # Regime shift on key features (wick, vr, p_center, z)
    key_features = [
        help_idx(TF_1M, 2), help_idx(TF_5M, 2), help_idx(TF_15M, 2), help_idx(TF_1H, 2),  # wicks
        core_idx(TF_1M, 2), core_idx(TF_5M, 2),  # vr
        core_idx(TF_1M, 9), core_idx(TF_5M, 9),  # p_center
        core_idx(TF_1M, 0), core_idx(TF_5M, 0), core_idx(TF_15M, 0), core_idx(TF_1H, 0),  # z_se
        core_idx(TF_1M, 3), core_idx(TF_5M, 3),  # velocity
    ]
    shift_rows = regime_shift_table(path_stats, FEATURE_NAMES, key_features)
    print('\nRegime shift: feature mean at entry -> peak -> exit')
    print(f'  {"segment":<13} {"feature":<24} {"entry":>8} {"peak":>8} {"exit":>8} '
          f'{"delta_ent_exit":>15}')
    for r in shift_rows:
        print(f'  {r["segment"]:<13} {r["feature"]:<24} {r["entry"]:>8.4f} '
              f'{r["peak"]:>8.4f} {r["exit"]:>8.4f} {r["entry_to_exit"]:>+14.4f}')

    # Write report
    out_dir = 'reports/findings'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md_path = os.path.join(out_dir, f'tier_eda_killshot_{ts}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# KILL_SHOT tier EDA — {ts}\n\n')
        f.write(f'KILL_SHOT classifier: `wick_5m > 0.83 AND wick_15m > 0.77 AND NOT h1_aligned`\n\n')
        f.write(f'Trades matched: **{len(killshot):,}** of {len(trades):,} iso trades '
                f'({len(killshot)/len(trades)*100:.1f}%)\n\n')
        f.write('## Segment summary\n\n')
        f.write('| Segment | N | MeanPnL | Total | MeanPeak | MedianPeak | MeanGiveback | MedHeld |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for seg in ['winner', 'small_loser', 'mid_loser', 'tail_loser']:
            sub = segmented.get(seg, [])
            if not sub:
                continue
            s = summarize_segment(sub)
            f.write(f'| {seg} | {s["n"]:,} | ${s["mean_pnl"]:+.2f} | ${s["total_pnl"]:+,.0f} | '
                    f'${s["mean_peak"]:+.2f} | ${s["median_peak"]:+.2f} | '
                    f'${s["mean_giveback"]:+.2f} | {s["median_held"]:.0f}m |\n')
        f.write('\n## Exit reason by segment\n\n')
        for seg in ['winner', 'small_loser', 'mid_loser', 'tail_loser']:
            sub = segmented.get(seg, [])
            if not sub:
                continue
            reasons = Counter(t.get('exit_reason', '?') for t in sub)
            top = ', '.join(f'`{r}`:{c}' for r, c in reasons.most_common(5))
            f.write(f'- **{seg}**: {top}\n')
        f.write('\n## Top 30 feature separators (winner vs tail_loser, |Cohen d|)\n\n')
        f.write('| Feature | Winner median | Tail median | Cohen d |\n')
        f.write('|---|---:|---:|---:|\n')
        for _, r in seps[:30]:
            f.write(f'| `{r["feature"]}` | {r["winner_median"]:.4f} | '
                    f'{r["tail_median"]:.4f} | {r["cohen_d"]:+.3f} |\n')
        f.write('\n## Peak timing per segment\n\n')
        f.write('| Segment | Peak bar (median) | Peak→close bars (mean) |\n')
        f.write('|---|---:|---:|\n')
        for seg in ['winner', 'small_loser', 'mid_loser', 'tail_loser']:
            stats = path_stats.get(seg)
            if not stats:
                continue
            f.write(f'| {seg} | {stats["peak_bar_median"]:.0f} | '
                    f'{stats["peak_to_close_mean"]:.1f} |\n')
        f.write('\n## Regime shift: feature mean at entry → peak → exit\n\n')
        f.write('| Segment | Feature | Entry | Peak | Exit | Δ entry→exit |\n')
        f.write('|---|---|---:|---:|---:|---:|\n')
        for r in shift_rows:
            f.write(f'| {r["segment"]} | `{r["feature"]}` | {r["entry"]:.4f} | '
                    f'{r["peak"]:.4f} | {r["exit"]:.4f} | {r["entry_to_exit"]:+.4f} |\n')

    print(f'\nWrote report: {md_path}')


if __name__ == '__main__':
    main()
