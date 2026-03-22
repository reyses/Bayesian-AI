"""
Build Peak Seeds — Convert pivot scanner output to auto-swing seed JSON format.

Pipeline stages (with audit checkpoints):
  Stage 1: Load & validate pivot CSV
  Stage 2: Filter REAL pivots (drop fakeouts/marginals)
  Stage 3: Compute conviction features (15s + 1m cross-TF agreement)
  Stage 4: Cluster into seed templates (K-Means on entry features)
  Stage 5: Conviction analysis — compare peak seeds vs auto-swing seeds
  Stage 6: Export as seed JSON compatible with --seeds pipeline

Each stage saves a checkpoint to reports/findings/peak_seed_audit/
so if the pipeline hangs, you can retry from the last checkpoint.

Usage:
    python tools/build_peak_seeds.py --pivot-csv reports/findings/pivot_seeds_mtf.csv
    python tools/build_peak_seeds.py --pivot-csv reports/findings/pivot_seeds_mtf.csv --auto-seeds DATA/regime_seeds/auto_swing/auto_seeds_all_20260313_200730.json
    python tools/build_peak_seeds.py --resume 3  # resume from stage 3
"""

import argparse
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────
AUDIT_DIR = os.path.join('reports', 'findings', 'peak_seed_audit')
OUTPUT_JSON = os.path.join('DATA', 'regime_seeds', 'peak_seeds.json')
OUTPUT_PKL = os.path.join('checkpoints', 'peak_seed_library.pkl')

# Features used for clustering peak seeds
CLUSTER_FEATURES_15S = [
    's15_F_momentum', 's15_z_score', 's15_sigma', 's15_velocity',
    's15_hurst', 's15_adx', 's15_dmi_diff', 's15_P_center',
    's15_entropy', 's15_coherence', 's15_term_pid', 's15_volume_delta',
]
CLUSTER_FEATURES_1M = [
    's1m_F_momentum', 's1m_z_score', 's1m_sigma', 's1m_velocity',
    's1m_hurst', 's1m_adx', 's1m_dmi_diff', 's1m_P_center',
    's1m_entropy', 's1m_coherence', 's1m_term_pid', 's1m_volume_delta',
]
ALL_CLUSTER_FEATURES = CLUSTER_FEATURES_15S + CLUSTER_FEATURES_1M

# Conviction scoring weights (from MTF scanner signal analysis)
CONVICTION_WEIGHTS = {
    's1m_volume_delta': 0.25,    # strongest discriminator
    's1m_F_momentum': 0.20,      # strong
    's15_F_momentum': 0.15,
    's15_P_center': 0.10,
    's15_coherence': 0.10,
    's15_dmi_diff': 0.10,
    's1m_dmi_diff': 0.10,
}


def _ensure_audit_dir():
    os.makedirs(AUDIT_DIR, exist_ok=True)


def _save_checkpoint(stage: int, data: dict, label: str):
    """Save stage checkpoint for resume capability."""
    _ensure_audit_dir()
    path = os.path.join(AUDIT_DIR, f'stage{stage}_{label}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'  [CHECKPOINT] Stage {stage}: {label} -> {path}')


def _load_checkpoint(stage: int, label: str):
    """Load stage checkpoint if it exists."""
    path = os.path.join(AUDIT_DIR, f'stage{stage}_{label}.pkl')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


# ── Stage 1: Load & Validate ──────────────────────────────────────────

def stage1_load(pivot_csv: str) -> pd.DataFrame:
    print('=' * 70)
    print('STAGE 1: Load & Validate Pivot CSV')
    print('=' * 70)

    df = pd.read_csv(pivot_csv)
    print(f'  Loaded {len(df)} pivots from {pivot_csv}')
    print(f'  Columns: {len(df.columns)}')
    print(f'  Labels: {df.label.value_counts().to_dict()}')
    print(f'  Directions: {df.direction.value_counts().to_dict()}')
    print(f'  PnL range: {df.pnl_ticks.min():.0f} to {df.pnl_ticks.max():.0f} ticks')

    # Validate required columns exist
    missing = [c for c in ALL_CLUSTER_FEATURES if c not in df.columns]
    if missing:
        print(f'  WARNING: missing features: {missing}')

    _save_checkpoint(1, {'df': df}, 'loaded')
    return df


# ── Stage 2: Filter REAL pivots ────────────────────────────────────────

def stage2_filter(df: pd.DataFrame, min_pnl_ticks: float = 4.0) -> tuple:
    print()
    print('=' * 70)
    print('STAGE 2: Filter REAL Pivots')
    print('=' * 70)

    n_before = len(df)
    df_real = df[df.label == 'REAL'].copy()
    print(f'  REAL pivots: {len(df_real)} / {n_before} ({len(df_real)/n_before*100:.0f}%)')

    df_fake = df[df.label == 'FAKEOUT'].copy()
    print(f'  FAKEOUT pivots: {len(df_fake)} (kept for conviction analysis)')

    # Filter minimum PnL
    df_profitable = df_real[df_real.pnl_ticks >= min_pnl_ticks].copy()
    print(f'  After min_pnl >= {min_pnl_ticks}t: {len(df_profitable)} ({len(df_profitable)/len(df_real)*100:.0f}%)')

    print(f'  Avg PnL: {df_profitable.pnl_ticks.mean():.1f}t')
    print(f'  Avg hold: {df_profitable.hold_seconds.mean():.0f}s')
    print(f'  Direction: {df_profitable.direction.value_counts().to_dict()}')

    _save_checkpoint(2, {'real': df_profitable, 'fake': df_fake}, 'filtered')
    return df_profitable, df_fake


# ── Stage 3: Conviction Features ───────────────────────────────────────

def stage3_conviction(df_real: pd.DataFrame, df_fake: pd.DataFrame) -> pd.DataFrame:
    print()
    print('=' * 70)
    print('STAGE 3: Compute Conviction Scores')
    print('=' * 70)

    for df_label, df_tmp in [('REAL', df_real), ('FAKE', df_fake)]:
        if len(df_tmp) == 0:
            continue
        scores = np.zeros(len(df_tmp))
        for feat, weight in CONVICTION_WEIGHTS.items():
            if feat in df_tmp.columns:
                vals = df_tmp[feat].fillna(0).values
                vmin, vmax = np.percentile(vals, [5, 95])
                if vmax > vmin:
                    normed = np.clip((vals - vmin) / (vmax - vmin), 0, 1)
                else:
                    normed = np.full_like(vals, 0.5)
                scores += weight * normed
        df_tmp['conviction'] = scores
        print(f'  {df_label}: conviction mean={scores.mean():.3f} std={scores.std():.3f}')

    real_conv = df_real['conviction'].mean()
    fake_conv = df_fake['conviction'].mean() if len(df_fake) > 0 else 0
    print(f'  Separation: REAL={real_conv:.3f} vs FAKE={fake_conv:.3f} (delta={real_conv-fake_conv:+.3f})')

    q25, q50, q75 = df_real['conviction'].quantile([0.25, 0.50, 0.75])
    print(f'  REAL quartiles: Q25={q25:.3f} Q50={q50:.3f} Q75={q75:.3f}')

    _save_checkpoint(3, {'real': df_real, 'fake': df_fake}, 'conviction')
    return df_real


# ── Stage 4: Cluster into Seeds ────────────────────────────────────────

def stage4_cluster(df: pd.DataFrame, n_clusters: int = 15) -> dict:
    print()
    print('=' * 70)
    print(f'STAGE 4: Cluster into {n_clusters} Seed Templates')
    print('=' * 70)

    available = [c for c in ALL_CLUSTER_FEATURES if c in df.columns]
    X = df[available].fillna(0).values
    print(f'  Feature matrix: {X.shape} ({len(available)} features)')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    seeds = {}
    for direction in ['LONG', 'SHORT']:
        mask = df.direction == direction
        X_dir = X_scaled[mask]
        df_dir = df[mask]

        n_k = min(n_clusters, len(df_dir))
        if n_k < 2:
            print(f'  {direction}: too few samples ({len(df_dir)}), skipping')
            continue

        km = KMeans(n_clusters=n_k, n_init=10, random_state=42)
        labels = km.fit_predict(X_dir)

        for k in range(n_k):
            cluster_mask = labels == k
            cluster_df = df_dir[cluster_mask]

            seed_id = f'peak_{direction.lower()}_{k}'
            seeds[seed_id] = {
                'seed_id': seed_id,
                'direction': direction,
                'centroid': km.cluster_centers_[k].tolist(),
                'centroid_raw': scaler.inverse_transform(km.cluster_centers_[k:k+1])[0].tolist(),
                'n_members': int(cluster_mask.sum()),
                'avg_pnl_ticks': float(cluster_df.pnl_ticks.mean()),
                'total_pnl_ticks': float(cluster_df.pnl_ticks.sum()),
                'win_rate': float((cluster_df.pnl_ticks > 0).mean()),
                'avg_hold_seconds': float(cluster_df.hold_seconds.mean()),
                'avg_conviction': float(cluster_df.conviction.mean()),
                'feature_names': available,
                'feature_means': {
                    feat: float(cluster_df[feat].mean())
                    for feat in available
                },
            }

        print(f'  {direction}: {n_k} clusters from {mask.sum()} pivots')

    ranked = sorted(seeds.values(), key=lambda s: s['total_pnl_ticks'], reverse=True)
    print()
    print(f'  {"Seed":<25} {"N":>5} {"PnL/t":>8} {"WR":>6} {"Hold":>6} {"Conv":>6}')
    print(f'  {"-"*25} {"---":>5} {"---":>8} {"---":>6} {"---":>6} {"---":>6}')
    for s in ranked:
        print(f'  {s["seed_id"]:<25} {s["n_members"]:>5} '
              f'{s["avg_pnl_ticks"]:>8.1f} {s["win_rate"]:>5.0%} '
              f'{s["avg_hold_seconds"]:>5.0f}s {s["avg_conviction"]:>6.3f}')

    _save_checkpoint(4, {'seeds': seeds, 'scaler': scaler, 'features': available}, 'clustered')
    return seeds


# ── Stage 5: Conviction Analysis ──────────────────────────────────────

def stage5_analysis(seeds: dict, auto_seeds_path: str = None):
    print()
    print('=' * 70)
    print('STAGE 5: Conviction Analysis')
    print('=' * 70)

    profitable = [s for s in seeds.values() if s['avg_pnl_ticks'] > 0]
    losers = [s for s in seeds.values() if s['avg_pnl_ticks'] <= 0]
    print(f'  Profitable seeds: {len(profitable)} / {len(seeds)}')
    print(f'  Losing seeds: {len(losers)} / {len(seeds)}')

    if profitable:
        avg_conv_win = np.mean([s['avg_conviction'] for s in profitable])
        avg_conv_lose = np.mean([s['avg_conviction'] for s in losers]) if losers else 0
        print(f'  Avg conviction -- winners: {avg_conv_win:.3f}, losers: {avg_conv_lose:.3f}')
        print(f'  Conviction separates? {"YES" if avg_conv_win > avg_conv_lose else "NO"}')

    for direction in ['LONG', 'SHORT']:
        dir_seeds = [s for s in seeds.values() if s['direction'] == direction]
        if dir_seeds:
            total_pnl = sum(s['total_pnl_ticks'] for s in dir_seeds)
            total_n = sum(s['n_members'] for s in dir_seeds)
            print(f'  {direction}: {len(dir_seeds)} seeds, {total_n} members, {total_pnl:.0f}t total PnL')

    # Compare with auto-swing seeds if provided
    if auto_seeds_path and os.path.isfile(auto_seeds_path):
        print()
        print('  -- Auto-Swing Seed Comparison --')
        with open(auto_seeds_path) as f:
            auto = json.load(f)

        auto_seeds_list = []
        for day_data in auto['days'].values():
            auto_seeds_list.extend(day_data.get('seeds', []))

        n_auto = len(auto_seeds_list)
        n_peak = sum(s['n_members'] for s in seeds.values())
        print(f'  Auto-swing seeds: {n_auto}')
        print(f'  Peak seeds (members): {n_peak}')

        auto_pnl = [s['change_ticks'] for s in auto_seeds_list]
        auto_wr = sum(1 for p in auto_pnl if p > 0) / len(auto_pnl) * 100
        print(f'  Auto-swing: avg={np.mean(auto_pnl):.1f}t WR={auto_wr:.0f}% total={sum(auto_pnl):.0f}t')

        peak_total = sum(s['total_pnl_ticks'] for s in seeds.values())
        peak_wr = np.mean([s['win_rate'] for s in seeds.values()]) * 100
        peak_avg = np.mean([s['avg_pnl_ticks'] for s in seeds.values()])
        print(f'  Peak seeds:  avg={peak_avg:.1f}t WR={peak_wr:.0f}% total={peak_total:.0f}t')

    # Save analysis report
    _ensure_audit_dir()
    report_path = os.path.join(AUDIT_DIR, 'conviction_analysis.txt')
    with open(report_path, 'w') as f:
        f.write('PEAK SEED CONVICTION ANALYSIS\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Total seeds: {len(seeds)}\n')
        f.write(f'Profitable: {len(profitable)} / {len(seeds)}\n\n')
        for s in sorted(seeds.values(), key=lambda x: x['total_pnl_ticks'], reverse=True):
            f.write(f'{s["seed_id"]}: N={s["n_members"]} PnL={s["avg_pnl_ticks"]:.1f}t '
                    f'WR={s["win_rate"]:.0%} conv={s["avg_conviction"]:.3f}\n')
            top_feats = sorted(s['feature_means'].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            for feat, val in top_feats:
                f.write(f'    {feat}: {val:.4f}\n')
            f.write('\n')
    print(f'  Report: {report_path}')

    _save_checkpoint(5, {'analysis': 'done'}, 'analyzed')


# ── Stage 6: Export as Seed JSON ───────────────────────────────────────

def stage6_export(seeds: dict, pivot_csv: str):
    print()
    print('=' * 70)
    print('STAGE 6: Export Seed JSON + PKL')
    print('=' * 70)

    df = pd.read_csv(pivot_csv)
    df_real = df[df.label == 'REAL'].copy()
    df_real['date'] = pd.to_datetime(df_real['timestamp'], unit='s', utc=True).dt.strftime('%Y-%m-%d')

    days = {}
    for date, grp in df_real.groupby('date'):
        day_seeds = []
        for _, row in grp.iterrows():
            seed = {
                'trade_id': int(row.name),
                'direction': row.direction,
                'ts_start': float(row.timestamp),
                'ts_end': float(row.timestamp + row.hold_seconds),
                'entry_price': float(row.entry_price),
                'exit_price': float(row.exit_price),
                'change_ticks': float(row.pnl_ticks),
                'change_dollars': float(row.pnl_dollars),
                'mfe_ticks': float(row.pnl_ticks) if row.pnl_ticks > 0 else 4.0,
                'mae_ticks': abs(float(row.pnl_ticks)) if row.pnl_ticks < 0 else 1.0,
                'mfe_dollars': float(row.pnl_dollars) if row.pnl_dollars > 0 else 2.0,
                'mae_dollars': abs(float(row.pnl_dollars)) if row.pnl_dollars < 0 else 0.5,
                'duration_mins': float(row.hold_seconds / 60),
                'time_to_mfe_mins': float(row.hold_seconds / 120),
                'n_bars': max(1, int(row.hold_seconds / 15)),
                'lookback_bars': 10,
                'lookback_start_idx': 0,
                'lookback_timestamps': [],
                'regime_start_idx': 0,
                'source': 'pivot_peak',
            }
            day_seeds.append(seed)

        days[date] = {
            'date': date,
            'timeframe': '1m',
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_seeds': len(day_seeds),
            'source': 'pivot_peak',
            'params': {'lookback_1s': 60, 'min_swing': 8, 'confirm_1m': True},
            'seeds': day_seeds,
        }

    output = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'pivot_seed_scanner_mtf',
        'params': {'detect_tf': '1s', 'confirm_tf': '1m', 'context_tf': '15s'},
        'n_days': len(days),
        'n_seeds_total': sum(d['n_seeds'] for d in days.values()),
        'days': days,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    size_mb = os.path.getsize(OUTPUT_JSON) / 1024 / 1024
    print(f'  Seed JSON: {OUTPUT_JSON} ({size_mb:.1f} MB, {output["n_seeds_total"]} seeds)')

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(seeds, f)
    print(f'  Seed PKL: {OUTPUT_PKL} ({len(seeds)} cluster templates)')

    _save_checkpoint(6, {'json': OUTPUT_JSON, 'pkl': OUTPUT_PKL}, 'exported')
    print()
    print(f'  Run with: python training/trainer.py --fresh --seeds {OUTPUT_JSON}')


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build peak seeds from pivot scanner output')
    parser.add_argument('--pivot-csv', default='reports/findings/pivot_seeds_mtf.csv',
                        help='Path to pivot scanner CSV')
    parser.add_argument('--auto-seeds', default=None,
                        help='Path to auto-swing seed JSON for comparison')
    parser.add_argument('--resume', type=int, default=0,
                        help='Resume from stage N (loads checkpoints)')
    parser.add_argument('--n-clusters', type=int, default=15,
                        help='Number of K-Means clusters per direction')
    parser.add_argument('--min-pnl', type=float, default=4.0,
                        help='Minimum PnL in ticks to include')
    args = parser.parse_args()

    resume = args.resume

    # Stage 1
    if resume <= 1:
        df = stage1_load(args.pivot_csv)
    else:
        data = _load_checkpoint(1, 'loaded')
        df = data['df']
        print(f'[RESUME] Loaded stage 1 checkpoint ({len(df)} pivots)')

    # Stage 2
    if resume <= 2:
        df_real, df_fake = stage2_filter(df, min_pnl_ticks=args.min_pnl)
    else:
        data = _load_checkpoint(2, 'filtered')
        df_real, df_fake = data['real'], data['fake']
        print(f'[RESUME] Loaded stage 2 checkpoint ({len(df_real)} real, {len(df_fake)} fake)')

    # Stage 3
    if resume <= 3:
        df_real = stage3_conviction(df_real, df_fake)
    else:
        data = _load_checkpoint(3, 'conviction')
        df_real = data['real']
        print(f'[RESUME] Loaded stage 3 checkpoint')

    # Stage 4
    if resume <= 4:
        seeds = stage4_cluster(df_real, n_clusters=args.n_clusters)
    else:
        data = _load_checkpoint(4, 'clustered')
        seeds = data['seeds']
        print(f'[RESUME] Loaded stage 4 checkpoint ({len(seeds)} seeds)')

    # Stage 5
    if resume <= 5:
        stage5_analysis(seeds, auto_seeds_path=args.auto_seeds)

    # Stage 6
    stage6_export(seeds, args.pivot_csv)


if __name__ == '__main__':
    main()
