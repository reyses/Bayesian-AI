#!/usr/bin/env python
"""
Gate Interaction Matrix — C&E Matrix empirical validation tool.

For each X parameter, computes its relationship to every Y response variable
using the appropriate statistical method (correlation for continuous,
group comparison for categorical). Reports sample sizes, flags underpowered
cells, and generates interaction heatmaps for top X pairs.

Usage:
    python tools/gate_interaction_matrix.py                        # IS data
    python tools/gate_interaction_matrix.py --dir reports/oos      # OOS data
    python tools/gate_interaction_matrix.py --plot                 # generate plots
    python tools/gate_interaction_matrix.py --top 5                # top 5 X per Y
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore', category=RuntimeWarning)

# -- Column mappings: oracle_trade_log columns → X identifiers --------------
# Each entry: (X_id, display_name, column_or_callable, type)
# type: 'cont' = continuous, 'cat' = categorical

def _safe_col(df, col):
    """Return series if column exists, else NaN series."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


X_MAP = [
    ('X1',  'z_score',       'entry_z_score',            'cont'),
    ('X3',  'F_momentum',    'F_momentum',               'cont'),
    ('X5',  'mom_rev_ratio', 'mom_rev_ratio',            'cont'),
    ('X7',  'hurst',         'entry_hurst',              'cont'),
    ('X8',  'tunnel_prob',   'tunnel_prob',              'cont'),
    ('X12', 'ADX',           'entry_adx',                'cont'),
    ('X13', 'DMI_diff',      'dmi_diff',                 'cont'),
    ('X14', 'sigma',         'sigma',                    'cont'),
    ('X16', 'osc_coherence', 'entry_oscillation_coherence', 'cont'),
    ('X17', 'conviction',    'belief_conviction',        'cont'),
    ('X19', 'wave_maturity', 'wave_maturity',            'cont'),
    ('X20', 'dec_wave_mat',  'decision_wave_maturity',   'cont'),
    ('X22', 'depth',         'entry_depth',              'cat'),
    ('X31', 'band_speed',    'band_speed',               'cont'),
]

# -- Y response variables ---------------------------------------------------

def _compute_ys(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all Y response variables from oracle trade log."""
    ys = pd.DataFrame(index=df.index)

    # Y1: Win Rate (binary per trade)
    ys['Y1_win'] = (df['result'] == 'WIN').astype(float)

    # Y2: PnL per trade (in ticks — actual_pnl is in ticks already)
    ys['Y2_pnl'] = df['actual_pnl'].astype(float)

    # Y3: Capture % (actual / oracle MFE)
    _mfe = df['oracle_mfe'].astype(float).replace(0, np.nan)
    ys['Y3_capture'] = df['actual_pnl'].astype(float) / _mfe

    # Y4: Reversal indicator (correct direction but lost money → reversed)
    # Proxy: WIN on direction but actual_pnl < 0
    ys['Y4_reversed'] = ((df['oracle_mfe'].astype(float) > 10) &
                         (df['actual_pnl'].astype(float) < 0)).astype(float)

    # Y5: Direction accuracy (oracle_mfe > some threshold = real move existed)
    # If MFE > 5 ticks, the direction had a real move
    ys['Y5_dir_correct'] = (df['oracle_mfe'].astype(float) > 5).astype(float)

    # Y6: Hold efficiency (PnL per bar held)
    _bars = df['hold_bars'].astype(float).replace(0, np.nan)
    ys['Y6_hold_eff'] = df['actual_pnl'].astype(float) / _bars

    # Y7: Hold time (bars)
    ys['Y7_hold_time'] = df['hold_bars'].astype(float)

    # Y8: Trade decay proxy (capture rate when held > median bars)
    # Per-trade: if held longer than median, did capture % drop?
    _med_bars = df['hold_bars'].median()
    ys['Y8_long_hold'] = (df['hold_bars'] > _med_bars).astype(float)

    # Y12: Risk-adjusted path ratio (PnL / MAE)
    _mae = df['oracle_mae'].astype(float).replace(0, np.nan)
    ys['Y12_risk_adj'] = df['actual_pnl'].astype(float) / _mae

    return ys


Y_NAMES = {
    'Y1_win':        'Win Rate',
    'Y2_pnl':        'PnL/trade',
    'Y3_capture':    'Capture%',
    'Y4_reversed':   'Reversal',
    'Y5_dir_correct':'Direction',
    'Y6_hold_eff':   'Hold Eff',
    'Y7_hold_time':  'Hold Time',
    'Y12_risk_adj':  'Risk-Adj',
}

# Domain assignment for display
Y_ENTRY = {'Y1_win', 'Y5_dir_correct'}
Y_EXIT  = {'Y2_pnl', 'Y3_capture', 'Y4_reversed', 'Y6_hold_eff',
           'Y7_hold_time', 'Y12_risk_adj'}


# -- Analysis functions -----------------------------------------------------

def analyze_continuous(x: pd.Series, y: pd.Series):
    """Correlation + regression for continuous X vs Y."""
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_c, y_c = x[mask], y[mask]
    n = len(x_c)
    if n < 10:
        return {'n': n, 'r': np.nan, 'p': np.nan, 'slope': np.nan,
                'effect': 'insufficient', 'power': 'low'}

    r, p = sp_stats.spearmanr(x_c, y_c)
    # Linear regression slope
    slope, intercept, r_val, p_val, se = sp_stats.linregress(x_c, y_c)

    # Effect size interpretation
    abs_r = abs(r)
    if abs_r >= 0.5:
        effect = 'LARGE'
    elif abs_r >= 0.3:
        effect = 'MEDIUM'
    elif abs_r >= 0.1:
        effect = 'small'
    else:
        effect = 'none'

    # Power estimate (simplified)
    # For correlation test: power ≈ Φ(|r|√(n-2) - z_α/2)
    if n > 2:
        z_obs = abs(r) * math.sqrt(n - 2)
        power = 'high' if z_obs > 2.8 else ('medium' if z_obs > 1.96 else 'low')
    else:
        power = 'low'

    return {'n': n, 'r': round(r, 3), 'p': round(p, 4), 'slope': round(slope, 4),
            'effect': effect, 'power': power}


def analyze_categorical(x: pd.Series, y: pd.Series):
    """Group comparison for categorical X vs continuous Y."""
    mask = x.notna() & y.notna() & np.isfinite(y)
    x_c, y_c = x[mask], y[mask]

    groups = {}
    for val in sorted(x_c.unique()):
        g = y_c[x_c == val]
        if len(g) >= 5:
            groups[val] = g

    if len(groups) < 2:
        return {'n_groups': len(groups), 'n_total': len(x_c),
                'h_stat': np.nan, 'p': np.nan, 'effect': 'insufficient',
                'group_means': {}, 'group_ns': {}}

    # Kruskal-Wallis test
    group_arrays = list(groups.values())
    h_stat, p = sp_stats.kruskal(*group_arrays)

    # Effect size: eta-squared approximation
    n_total = sum(len(g) for g in group_arrays)
    k = len(group_arrays)
    eta_sq = (h_stat - k + 1) / (n_total - k) if n_total > k else 0
    eta_sq = max(0, eta_sq)

    if eta_sq >= 0.14:
        effect = 'LARGE'
    elif eta_sq >= 0.06:
        effect = 'MEDIUM'
    elif eta_sq >= 0.01:
        effect = 'small'
    else:
        effect = 'none'

    group_means = {str(k): round(float(v.mean()), 4) for k, v in groups.items()}
    group_ns = {str(k): len(v) for k, v in groups.items()}

    return {'n_groups': len(groups), 'n_total': n_total,
            'h_stat': round(h_stat, 2), 'p': round(p, 4),
            'effect': effect, 'eta_sq': round(eta_sq, 4),
            'group_means': group_means, 'group_ns': group_ns}


def interaction_heatmap(x1: pd.Series, x2: pd.Series, y: pd.Series,
                        n_bins: int = 3, min_cell: int = 10):
    """2D interaction: bin x1 and x2 into n_bins, compute Y mean per cell."""
    mask = x1.notna() & x2.notna() & y.notna() & np.isfinite(y)
    x1c, x2c, yc = x1[mask], x2[mask], y[mask]
    if len(x1c) < n_bins * n_bins * min_cell:
        return None

    # Bin into quantile-based groups
    try:
        x1_bins = pd.qcut(x1c, n_bins, labels=['low', 'mid', 'high'][:n_bins],
                          duplicates='drop')
        x2_bins = pd.qcut(x2c, n_bins, labels=['low', 'mid', 'high'][:n_bins],
                          duplicates='drop')
    except ValueError:
        return None

    grid = pd.DataFrame({'x1': x1_bins, 'x2': x2_bins, 'y': yc})
    pivot_mean = grid.groupby(['x1', 'x2'])['y'].mean().unstack(fill_value=np.nan)
    pivot_n = grid.groupby(['x1', 'x2'])['y'].count().unstack(fill_value=0)

    # Flag cells with insufficient N
    underpowered = (pivot_n < min_cell).sum().sum()

    return {'means': pivot_mean, 'counts': pivot_n, 'underpowered_cells': underpowered}


# -- Main pipeline ----------------------------------------------------------

def load_data(data_dir: str) -> pd.DataFrame:
    """Load oracle trade log from reports dir or checkpoint dir."""
    candidates = [
        os.path.join(data_dir, 'oracle_trade_log.csv'),
        os.path.join(data_dir, 'is_trade_log.csv'),
        os.path.join(data_dir, 'oos_trade_log.csv'),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded {path}: {len(df)} trades")
            return df

    # Try sharded
    import glob
    shards = sorted(glob.glob(os.path.join(data_dir, 'shards', '*trade_log*.csv')))
    if shards:
        dfs = [pd.read_csv(s) for s in shards]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(shards)} shards: {len(df)} trades")
        return df

    print(f"ERROR: No trade log found in {data_dir}")
    sys.exit(1)


def run_matrix(df: pd.DataFrame, top_n: int = 5, do_plot: bool = False):
    """Run the full C&E interaction matrix analysis."""
    ys = _compute_ys(df)
    y_cols = [c for c in ys.columns if c in Y_NAMES]

    # Power analysis header
    n_trades = len(df)
    print(f"\n{'='*80}")
    print(f"CAUSE & EFFECT INTERACTION MATRIX")
    print(f"{'='*80}")
    print(f"Total trades: {n_trades}")
    print(f"Max feasible bins (1D, >=30/bin): {n_trades // 30}")
    print(f"Max feasible 2D grid (>=15/cell): {int(math.sqrt(n_trades / 15))}"
          f"x{int(math.sqrt(n_trades / 15))}")

    # Detectable WR effect (binary, 80% power, 95% CI)
    if n_trades > 0:
        # delta = z * sqrt(p*(1-p)/n) * 2 for two-group comparison
        delta_wr = 2.8 * math.sqrt(0.25 / n_trades) * 100
        print(f"Min detectable WR effect: ~{delta_wr:.1f}% (80% power)")

    # -- Build results matrix --
    results = {}  # results[x_id][y_col] = analysis_dict

    available_xs = []
    for x_id, x_name, x_col, x_type in X_MAP:
        if x_col in df.columns:
            available_xs.append((x_id, x_name, x_col, x_type))

    if not available_xs:
        print("\nERROR: No X columns found in data. Run a forward pass first.")
        return

    print(f"\nAvailable X parameters: {len(available_xs)} / {len(X_MAP)}")
    missing = [f"{xid}({xn})" for xid, xn, xc, _ in X_MAP if xc not in df.columns]
    if missing:
        print(f"Missing (need new IS run): {', '.join(missing)}")

    for x_id, x_name, x_col, x_type in available_xs:
        results[x_id] = {}
        x_series = df[x_col]

        for y_col in y_cols:
            y_series = ys[y_col]
            if x_type == 'cont':
                results[x_id][y_col] = analyze_continuous(x_series, y_series)
            else:
                results[x_id][y_col] = analyze_categorical(x_series, y_series)

    # -- Print summary matrix --
    print(f"\n{'-'*80}")
    print("CORRELATION MATRIX (Spearman r / Kruskal eta²)")
    print(f"{'-'*80}")

    # Header
    header = f"{'X':>18}"
    for y_col in y_cols:
        header += f" {Y_NAMES[y_col]:>10}"
    header += f" {'Domain':>8}"
    print(header)
    print("-" * len(header))

    # Compute domain scores for sorting
    x_domain_scores = {}
    for x_id, x_name, x_col, x_type in available_xs:
        entry_sum = 0.0
        exit_sum = 0.0
        for y_col in y_cols:
            res = results[x_id][y_col]
            if x_type == 'cont':
                val = abs(res.get('r', 0) or 0)
            else:
                val = res.get('eta_sq', 0) or 0
            if y_col in Y_ENTRY:
                entry_sum += val
            elif y_col in Y_EXIT:
                exit_sum += val
        x_domain_scores[x_id] = (entry_sum, exit_sum)

    for x_id, x_name, x_col, x_type in available_xs:
        row = f"{x_id:>4} {x_name:>13}"
        for y_col in y_cols:
            res = results[x_id][y_col]
            if x_type == 'cont':
                r = res.get('r', np.nan)
                p = res.get('p', 1.0)
                if np.isnan(r):
                    cell = "     --"
                else:
                    sig = '**' if p < 0.01 else ('* ' if p < 0.05 else '  ')
                    cell = f" {r:>7.3f}{sig}"
            else:
                eta = res.get('eta_sq', np.nan)
                p = res.get('p', 1.0)
                if np.isnan(eta):
                    cell = "     --"
                else:
                    sig = '**' if p < 0.01 else ('* ' if p < 0.05 else '  ')
                    cell = f" {eta:>7.4f}{sig}"
            row += cell

        # Domain label
        e, x_ = x_domain_scores[x_id]
        if e > x_ * 1.5:
            domain = 'ENTRY'
        elif x_ > e * 1.5:
            domain = 'EXIT'
        else:
            domain = 'BOTH'
        row += f" {domain:>8}"
        print(row)

    print(f"\n  ** p<0.01  * p<0.05  -- insufficient data")

    # -- Top X per Y --
    print(f"\n{'-'*80}")
    print(f"TOP {top_n} PARAMETERS PER RESPONSE VARIABLE")
    print(f"{'-'*80}")

    for y_col in y_cols:
        y_domain = 'ENTRY' if y_col in Y_ENTRY else 'EXIT'
        print(f"\n  {Y_NAMES[y_col]} ({y_col}) [{y_domain}]:")

        scores = []
        for x_id, x_name, x_col, x_type in available_xs:
            res = results[x_id][y_col]
            if x_type == 'cont':
                score = abs(res.get('r', 0) or 0)
                detail = f"r={res.get('r', 0):+.3f} p={res.get('p', 1):.4f}"
            else:
                score = res.get('eta_sq', 0) or 0
                detail = f"eta²={score:.4f} p={res.get('p', 1):.4f}"
                # Add group means
                gm = res.get('group_means', {})
                if gm:
                    detail += f" means={gm}"
            scores.append((score, x_id, x_name, detail, res.get('effect', ''),
                           res.get('power', '')))

        scores.sort(reverse=True)
        for rank, (score, x_id, x_name, detail, effect, power) in enumerate(scores[:top_n], 1):
            pw = f"[{power}]" if power else ''
            print(f"    {rank}. {x_id} {x_name:>15}: {detail}  "
                  f"effect={effect} {pw}")

    # -- Sample size warnings --
    print(f"\n{'-'*80}")
    print("SAMPLE SIZE WARNINGS")
    print(f"{'-'*80}")

    for x_id, x_name, x_col, x_type in available_xs:
        for y_col in y_cols:
            res = results[x_id][y_col]
            n = res.get('n', res.get('n_total', 0))
            if n < 30:
                print(f"  WARNING: {x_id} {x_name} x {y_col}: N={n} (<30)")
            if x_type == 'cat':
                gn = res.get('group_ns', {})
                for grp, cnt in gn.items():
                    if cnt < 15:
                        print(f"  WARNING: {x_id} {x_name} group={grp}: "
                              f"N={cnt} (<15 per cell)")

    # -- Interaction plots for top pair per Y --
    if do_plot:
        _generate_plots(df, ys, y_cols, available_xs, results, top_n)

    return results


def _generate_plots(df, ys, y_cols, available_xs, results, top_n):
    """Generate interaction heatmaps for top X pairs per Y."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not available — skipping plots")
        return

    out_dir = Path('reports/interaction_matrix')
    out_dir.mkdir(parents=True, exist_ok=True)

    for y_col in y_cols:
        # Find top 2 continuous X's for this Y
        scores = []
        for x_id, x_name, x_col, x_type in available_xs:
            if x_type != 'cont':
                continue
            res = results[x_id][y_col]
            r = abs(res.get('r', 0) or 0)
            scores.append((r, x_id, x_name, x_col))
        scores.sort(reverse=True)

        if len(scores) < 2:
            continue

        _, x1_id, x1_name, x1_col = scores[0]
        _, x2_id, x2_name, x2_col = scores[1]

        hm = interaction_heatmap(df[x1_col], df[x2_col], ys[y_col])
        if hm is None:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(hm['means'].values, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(hm['means'].columns)))
        ax.set_xticklabels(hm['means'].columns)
        ax.set_yticks(range(len(hm['means'].index)))
        ax.set_yticklabels(hm['means'].index)
        ax.set_xlabel(f"{x2_id} {x2_name}")
        ax.set_ylabel(f"{x1_id} {x1_name}")
        ax.set_title(f"{Y_NAMES[y_col]} ({y_col}) — Interaction: "
                     f"{x1_id} x {x2_id}")

        # Annotate cells with mean + N
        for i in range(len(hm['means'].index)):
            for j in range(len(hm['means'].columns)):
                val = hm['means'].iloc[i, j]
                n = hm['counts'].iloc[i, j]
                color = 'white' if abs(val) > 0.5 * hm['means'].values.max() else 'black'
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}\nn={n}",
                            ha='center', va='center', color=color, fontsize=9)

        plt.colorbar(im, ax=ax, label=Y_NAMES[y_col])
        if hm['underpowered_cells'] > 0:
            ax.set_title(ax.get_title() +
                         f"\n({hm['underpowered_cells']} underpowered cells)",
                         fontsize=10)
        plt.tight_layout()
        path = out_dir / f"interaction_{y_col}.png"
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='C&E Interaction Matrix')
    parser.add_argument('--dir', default='reports/is',
                        help='Directory with oracle_trade_log.csv')
    parser.add_argument('--top', type=int, default=5,
                        help='Show top N parameters per Y')
    parser.add_argument('--plot', action='store_true',
                        help='Generate interaction heatmap PNGs')
    args = parser.parse_args()

    df = load_data(args.dir)
    run_matrix(df, top_n=args.top, do_plot=args.plot)


if __name__ == '__main__':
    main()
