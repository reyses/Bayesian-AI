"""
TF Mixing Analysis — quantify timeframe mixing in K-Means clustering.

Loads pattern_library.pkl and analyzes how patterns from different timeframes
get mixed within the same template.  The centroid[4] = log2(tf_seconds) encodes
the AVERAGE TF of all patterns in a template.  Distance from the nearest clean
TF value is a proxy for mixing.

Usage:
    python tools/research_tf_mixing.py                            # default
    python tools/research_tf_mixing.py --checkpoint checkpoints   # explicit

Outputs:
    - Console report: TF distribution, mixing indicator, oracle stats per TF bucket
    - Plot: tools/plots/tf_mixing/tf_mixing_overview.png
"""

import sys, os, math, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Known TF mappings ──
TF_SECONDS = {
    '1s': 1, '5s': 5, '15s': 15, '30s': 30,
    '1m': 60, '2m': 120, '3m': 180, '5m': 300,
    '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400,
    '1D': 86400, '1W': 604800,
}
TF_LOG2 = {tf: math.log2(s) for tf, s in TF_SECONDS.items()}
TF_ORDER = sorted(TF_SECONDS.keys(), key=lambda t: TF_SECONDS[t])
PURE_THRESHOLD = 0.15  # max log2 distance to count as "pure TF"


def nearest_tf(log2_val: float):
    """Find nearest TF given log2(seconds) value.  Returns (tf_label, distance)."""
    best_tf, best_dist = '?', float('inf')
    for tf, l2 in TF_LOG2.items():
        d = abs(log2_val - l2)
        if d < best_dist:
            best_dist = d
            best_tf = tf
    return best_tf, best_dist


def analyze_checkpoint(ckpt_dir: str = 'checkpoints'):
    lib_path = os.path.join(ckpt_dir, 'pattern_library.pkl')
    if not os.path.exists(lib_path):
        print(f"ERROR: {lib_path} not found.  Run training first.")
        return None

    with open(lib_path, 'rb') as f:
        lib = pickle.load(f)

    print(f"\n{'='*70}")
    print(f"  TF MIXING ANALYSIS — {len(lib)} templates")
    print(f"  Source: {lib_path}")
    print(f"{'='*70}")

    # ── Decode centroid[4] for each template ──
    rows = []
    for tid, entry in lib.items():
        c = entry.get('centroid')
        if c is None or len(c) < 5:
            continue
        tf, dist = nearest_tf(c[4])
        rows.append({
            'tid': tid, 'log2_tf': c[4], 'nearest_tf': tf,
            'dist_to_clean': dist,
            'members': entry.get('member_count', 0),
            'name': entry.get('semantic_name', '?'),
            'mfe': entry.get('mean_mfe_ticks', 0),
            'p75_mfe': entry.get('p75_mfe_ticks', 0),
            'mfe_bar': entry.get('avg_mfe_bar', 0),
            'wr': entry.get('stats_win_rate', 0),
        })

    df = pd.DataFrame(rows)
    total_members = df['members'].sum()

    # ── 1. TF distribution ──
    print(f"\n  1. TEMPLATE DISTRIBUTION BY TIMEFRAME")
    print(f"  {'TF':<6} {'Tmpls':>6} {'Members':>8} {'%':>6} {'AvgMFE':>8} {'WR':>7}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*7}")
    for tf in TF_ORDER:
        g = df[df['nearest_tf'] == tf]
        if len(g) == 0:
            continue
        mem = g['members'].sum()
        w_mfe = (g['mfe'] * g['members']).sum() / max(mem, 1)
        w_wr = (g['wr'] * g['members']).sum() / max(mem, 1)
        print(f"  {tf:<6} {len(g):>6} {mem:>8} {mem/total_members*100:>5.1f}% "
              f"{w_mfe:>8.1f} {w_wr:>6.1%}")

    # ── 2. TF purity indicator ──
    pure = df[df['dist_to_clean'] < PURE_THRESHOLD]
    mixed = df[df['dist_to_clean'] >= PURE_THRESHOLD]
    print(f"\n  2. TF PURITY (threshold = {PURE_THRESHOLD} log2-distance)")
    print(f"  Pure:  {len(pure):>5} templates ({len(pure)/len(df)*100:.1f}%), "
          f"{pure['members'].sum():>6} members ({pure['members'].sum()/total_members*100:.1f}%)")
    print(f"  Mixed: {len(mixed):>5} templates ({len(mixed)/len(df)*100:.1f}%), "
          f"{mixed['members'].sum():>6} members ({mixed['members'].sum()/total_members*100:.1f}%)")

    if len(mixed) > 0:
        print(f"\n  Top 10 most mixed templates:")
        print(f"  {'TID':>5} {'log2':>7} {'Near':>5} {'Dist':>6} {'Mem':>5} {'Name'}")
        for _, r in mixed.sort_values('dist_to_clean', ascending=False).head(10).iterrows():
            print(f"  {int(r.tid):>5} {r.log2_tf:>7.2f} {r.nearest_tf:>5} "
                  f"{r.dist_to_clean:>6.3f} {int(r.members):>5} {r['name']}")

    # ── 3. Scale mismatch: oracle stats by TF ──
    print(f"\n  3. SCALE MISMATCH — Oracle MFE by TF bucket (member-weighted)")
    print(f"  {'TF':<6} {'MFE':>8} {'P75_MFE':>8} {'MFE_bar':>8} {'WR':>7} {'Members':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")
    for tf in TF_ORDER:
        g = df[df['nearest_tf'] == tf]
        if len(g) == 0:
            continue
        m = g['members'].sum()
        w = lambda col: (g[col] * g['members']).sum() / max(m, 1)
        print(f"  {tf:<6} {w('mfe'):>8.1f} {w('p75_mfe'):>8.1f} "
              f"{w('mfe_bar'):>8.1f} {w('wr'):>6.1%} {m:>8}")

    # ── 4. Within-TF template spread ──
    print(f"\n  4. WITHIN-TF TEMPLATE SPREAD (how much MFE varies within a TF bucket)")
    print(f"  {'TF':<6} {'MFE_std':>8} {'Bar_std':>8} {'N':>5} {'CV':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*5} {'-'*8}")
    for tf in TF_ORDER:
        g = df[df['nearest_tf'] == tf]
        if len(g) < 3:
            continue
        mfe_mean = g['mfe'].mean()
        cv = g['mfe'].std() / max(mfe_mean, 0.01)
        print(f"  {tf:<6} {g['mfe'].std():>8.1f} {g['mfe_bar'].std():>8.1f} "
              f"{len(g):>5} {cv:>8.2f}")

    # ── 5. Key question: do faster TFs dominate? ──
    fast_tfs = {'1s', '5s', '15s', '30s', '1m'}
    slow_tfs = {'5m', '15m', '30m', '1h', '4h', '1D', '1W'}
    fast = df[df['nearest_tf'].isin(fast_tfs)]
    slow = df[df['nearest_tf'].isin(slow_tfs)]
    print(f"\n  5. FAST vs SLOW TF COMPARISON")
    print(f"  Fast (<=1m): {len(fast)} templates, {fast['members'].sum()} members, "
          f"MFE={fast['mfe'].mean():.1f}, MFE_bar={fast['mfe_bar'].mean():.1f}")
    print(f"  Slow (>=5m): {len(slow)} templates, {slow['members'].sum()} members, "
          f"MFE={slow['mfe'].mean():.1f}, MFE_bar={slow['mfe_bar'].mean():.1f}")

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plots_dir = 'tools/plots/tf_mixing'
        os.makedirs(plots_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('TF Mixing in K-Means Clustering', fontsize=14, fontweight='bold')

        # (a) Templates per TF
        ax = axes[0, 0]
        counts = [len(df[df['nearest_tf'] == tf]) for tf in TF_ORDER]
        ax.bar(range(len(TF_ORDER)), counts, color='steelblue')
        ax.set_xticks(range(len(TF_ORDER)))
        ax.set_xticklabels(TF_ORDER, rotation=45, fontsize=8)
        ax.set_title('Templates per TF')
        ax.set_ylabel('Count')

        # (b) Mixing distance histogram
        ax = axes[0, 1]
        ax.hist(df['dist_to_clean'], bins=30, color='coral', edgecolor='black')
        ax.axvline(PURE_THRESHOLD, color='red', ls='--', lw=1.5, label=f'threshold={PURE_THRESHOLD}')
        ax.set_title('TF Mixing Distance (centroid[4] from nearest clean TF)')
        ax.set_xlabel('log2 distance')
        ax.legend()

        # (c) MFE scatter by TF scale
        ax = axes[1, 0]
        sc = ax.scatter(df['log2_tf'], df['mfe'], c=df['wr'], cmap='RdYlGn',
                        s=np.clip(df['members'] / 3, 5, 200), alpha=0.6,
                        edgecolors='k', linewidths=0.3)
        for tf in TF_ORDER:
            ax.axvline(TF_LOG2[tf], color='grey', alpha=0.2, lw=0.5)
        ax.set_xlabel('log2(TF seconds)')
        ax.set_ylabel('Mean MFE (ticks)')
        ax.set_title('MFE vs TF Scale (color=WR, size=members)')
        fig.colorbar(sc, ax=ax, label='Win Rate')

        # (d) Member-weighted WR by TF
        ax = axes[1, 1]
        tf_wrs = []
        for tf in TF_ORDER:
            g = df[df['nearest_tf'] == tf]
            if len(g) > 0 and g['members'].sum() > 0:
                tf_wrs.append((g['wr'] * g['members']).sum() / g['members'].sum())
            else:
                tf_wrs.append(0)
        ax.bar(range(len(TF_ORDER)), tf_wrs, color='seagreen')
        ax.set_xticks(range(len(TF_ORDER)))
        ax.set_xticklabels(TF_ORDER, rotation=45, fontsize=8)
        ax.set_title('Win Rate by TF (member-weighted)')
        ax.set_ylabel('Win Rate')

        plt.tight_layout()
        out = os.path.join(plots_dir, 'tf_mixing_overview.png')
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"\n  [saved] {out}")
    except Exception as e:
        print(f"  [plot error] {e}")

    # ── Summary ──
    print(f"\n  SUMMARY")
    print(f"  -------")
    if len(mixed) == 0:
        print(f"  All templates are TF-pure — no mixing detected.")
        print(f"  TF bucketing would have no effect on current library.")
    elif len(mixed) / len(df) < 0.1:
        print(f"  Very low mixing ({len(mixed)/len(df)*100:.1f}%) — feature[4] provides good separation.")
        print(f"  TF bucketing is unlikely to improve results significantly.")
    else:
        pct = len(mixed) / len(df) * 100
        print(f"  {pct:.1f}% of templates show TF mixing.")
        mixed_wr = (mixed['wr'] * mixed['members']).sum() / max(mixed['members'].sum(), 1)
        pure_wr = (pure['wr'] * pure['members']).sum() / max(pure['members'].sum(), 1)
        print(f"  Pure WR: {pure_wr:.1%}  vs  Mixed WR: {mixed_wr:.1%}")
        if mixed_wr < pure_wr - 0.02:
            print(f"  Mixed templates underperform by {(pure_wr - mixed_wr)*100:.1f}pp — TF bucketing may help.")
        else:
            print(f"  No significant WR gap — mixing doesn't appear to hurt quality.")

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TF Mixing Analysis')
    parser.add_argument('--checkpoint', default='checkpoints', help='Checkpoint directory')
    args = parser.parse_args()
    analyze_checkpoint(args.checkpoint)
