"""Graphical report for full-stack forward pass (B7 + B9 + B10).

Re-runs the per-leg sizing computation from `tools/forward_pass_full_stack.py`
and produces a multi-panel chart suite for OOS (sealed) results.

Outputs (under reports/findings/regret_oracle/charts/):
  - 2026-05-18_forward_pass_OOS_summary.png    (6-panel summary)
  - 2026-05-18_forward_pass_OOS_equity.png     (equity curves zoomed)
  - 2026-05-18_forward_pass_IS_vs_OOS.png      (IS vs OOS deltas)
  - 2026-05-18_per_day_OOS.csv                 (per-day per-scheme $)
"""
from __future__ import annotations
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


B7_PKL = 'reports/findings/regret_oracle/b7_leg_sizer.pkl'
B9_PKL = 'reports/findings/regret_oracle/b9_remaining_amplitude_K5.pkl'
B10_HIGH_PKL = 'reports/findings/regret_oracle/b10_vol_regime_high.pkl'
B10_LOW_PKL  = 'reports/findings/regret_oracle/b10_vol_regime_low.pkl'
CROSS_DAY_FEATURES = 'DATA/CROSS_DAY/cross_day_features.parquet'

B10_THR_HIGH = 0.5
B10_THR_LOW  = 0.7
B10_BOOST = 1.3
B10_CAP   = 0.7

B10_FEATS = [
    'overnight_gap_pct', 'overnight_range_pct',
    'prior_day_range_pct', 'prior_day_c2c_pct',
    'vix_close_prior', 'vix_chg_prior',
    'dxy_close_prior', 'dxy_chg_prior',
    'is_fomc', 'is_cpi', 'is_nfp', 'is_opex',
    'days_since_fomc', 'days_to_next_fomc', 'dow',
]

SCHEME_ORDER = [
    'FLAT (baseline)',
    'B7 only',
    'B9 only',
    'B10 only',
    'B7 + B9',
    'B7 + B10',
    'B7 + B9 + B10 (full stack)',
]
SCHEME_COLORS = {
    'FLAT (baseline)':              '#888888',
    'B7 only':                      '#377eb8',
    'B9 only':                      '#4daf4a',
    'B10 only':                     '#984ea3',
    'B7 + B9':                      '#a65628',
    'B7 + B10':                     '#f781bf',
    'B7 + B9 + B10 (full stack)':   '#e41a1c',
}

OUT_DIR = Path('reports/findings/regret_oracle/charts')


def b7_size(p):  return float(np.clip(max(p - 1.0, 0.0), 0.0, 3.0))
def b9_size(p):
    if p > 50:   return 1.5
    if p > 10:   return 1.0
    if p > -10:  return 1.0
    if p > -50:  return 0.5
    return 0.0
def b10_mult(ph, pl):
    if ph >= B10_THR_HIGH: return B10_BOOST
    if pl >= B10_THR_LOW:  return B10_CAP
    return 1.0


def bootstrap_ci(values, n_boot=4000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = values[rng.integers(0, n, n)].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_stack(label, legs_path, traj_path, truth_path, source_filter):
    print(f'\n=== {label} ===')
    with open(B7_PKL, 'rb') as f:   b7 = pickle.load(f)
    with open(B9_PKL, 'rb') as f:   b9 = pickle.load(f)
    with open(B10_HIGH_PKL, 'rb') as f: b10h = pickle.load(f)
    with open(B10_LOW_PKL,  'rb') as f: b10l = pickle.load(f)

    legs = pd.read_csv(legs_path).sort_values(['day', 'entry_ts']).reset_index(drop=True)
    legs['leg_idx'] = legs.index
    print(f'  legs={len(legs):,}  days={legs["day"].nunique()}')

    truth = pd.read_parquet(truth_path)
    traj_full = pd.read_parquet(traj_path)
    traj_k5 = traj_full[traj_full['K'] == 5].copy()
    feats = pd.read_parquet(CROSS_DAY_FEATURES)
    feats = feats[feats['source'] == source_filter].copy()

    # B7 at R-trigger
    truth_sorted = truth.sort_values(['day', 'timestamp']).reset_index(drop=True)
    day_truth = {d: g.reset_index(drop=True) for d, g in truth_sorted.groupby('day')}
    b7_preds = []
    for _, leg in legs.iterrows():
        day = leg['day']
        if day not in day_truth:
            b7_preds.append(np.nan); continue
        td = day_truth[day]
        i = int(np.searchsorted(td['timestamp'].values, leg['entry_ts'], side='right') - 1)
        if i < 0 or i >= len(td):
            b7_preds.append(np.nan); continue
        r = td.iloc[i]
        X = np.array([float(r[c]) if not pd.isna(r[c]) else 0.0
                       for c in b7['v2_cols']], dtype=np.float32).reshape(1, -1)
        b7_preds.append(float(b7['model'].predict(X)[0]))
    legs['b7_pred'] = b7_preds
    legs['b7_size'] = legs['b7_pred'].apply(lambda p: b7_size(p) if not pd.isna(p) else 1.0)

    # B9 at K=5
    X_t = traj_k5[b9['feat_cols']].fillna(0.0).values
    traj_k5['b9_pred'] = b9['model'].predict(X_t)
    traj_k5['b9_size'] = traj_k5['b9_pred'].apply(b9_size)
    legs_b = legs.merge(traj_k5[['leg_id', 'b9_size', 'pnl_usd_so_far']],
                         left_on='leg_idx', right_on='leg_id', how='left')
    legs_b['b9_size'] = legs_b['b9_size'].fillna(1.0)
    legs_b['pnl_usd_so_far'] = legs_b['pnl_usd_so_far'].fillna(0.0)

    # B10 per day
    X_f = feats[B10_FEATS].fillna(0.0).values.astype(np.float32)
    feats['p_high'] = b10h['model'].predict_proba(X_f)[:, 1]
    feats['p_low']  = b10l['model'].predict_proba(X_f)[:, 1]
    feats['b10_mult'] = feats.apply(lambda r: b10_mult(r['p_high'], r['p_low']), axis=1)
    day_to_mult = dict(zip(feats['date_label'], feats['b10_mult']))
    legs_b['b10_mult'] = legs_b['day'].map(day_to_mult).fillna(1.0)

    pnl_total = legs_b['pnl_usd'].values
    pnl_k5    = legs_b['pnl_usd_so_far'].values
    pnl_post  = pnl_total - pnl_k5
    b7sz = legs_b['b7_size'].fillna(1.0).values
    b9sz = legs_b['b9_size'].values
    b10m = legs_b['b10_mult'].values

    schemes = {
        'FLAT (baseline)':            pnl_total,
        'B7 only':                    pnl_total * b7sz,
        'B9 only':                    pnl_k5 + pnl_post * b9sz,
        'B10 only':                   pnl_total * b10m,
        'B7 + B9':                    pnl_k5 * b7sz + pnl_post * b7sz * b9sz,
        'B7 + B10':                   pnl_total * b7sz * b10m,
        'B7 + B9 + B10 (full stack)': pnl_k5 * b7sz * b10m + pnl_post * b7sz * b10m * b9sz,
    }
    return legs_b, schemes, feats


def per_day_table(legs_b, schemes):
    rows = []
    days = sorted(legs_b['day'].unique())
    for d in days:
        m = legs_b['day'] == d
        row = {'day': d, 'n_legs': int(m.sum()),
               'b10_mult': float(legs_b.loc[m, 'b10_mult'].iloc[0])}
        for name, vals in schemes.items():
            row[name] = float(vals[m].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def panel_scheme_bars(ax, per_day_df, title):
    means, lo, hi, sig = [], [], [], []
    flat = per_day_df['FLAT (baseline)'].values
    for name in SCHEME_ORDER:
        v = per_day_df[name].values
        means.append(v.mean())
        cl, ch = bootstrap_ci(v)
        lo.append(cl); hi.append(ch)
        if name == 'FLAT (baseline)':
            sig.append(False)
        else:
            dl, _ = bootstrap_ci(v - flat)
            sig.append(dl > 0)
    xs = np.arange(len(SCHEME_ORDER))
    colors = [SCHEME_COLORS[n] for n in SCHEME_ORDER]
    bars = ax.bar(xs, means, color=colors, edgecolor='black', linewidth=0.6)
    yerr_lo = [m - l for m, l in zip(means, lo)]
    yerr_hi = [h - m for h, m in zip(hi, means)]
    ax.errorbar(xs, means, yerr=[yerr_lo, yerr_hi], fmt='none',
                  ecolor='black', capsize=4, linewidth=1)
    for x, m, s in zip(xs, means, sig):
        if s:
            ax.text(x, m + max(yerr_hi) * 0.05, '*', ha='center',
                      va='bottom', fontsize=14, fontweight='bold', color='darkred')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticks(xs)
    short_labels = ['FLAT', 'B7', 'B9', 'B10', 'B7+B9', 'B7+B10', 'B7+B9+B10']
    ax.set_xticklabels(short_labels, rotation=0, fontsize=9)
    ax.set_ylabel('$/day  (95% CI)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    # annotate values
    for x, m in zip(xs, means):
        ax.text(x, m / 2, f'${m:.0f}', ha='center', va='center',
                  fontsize=8, color='white' if m > 200 else 'black', fontweight='bold')


def _to_dt(day_series):
    return pd.to_datetime(day_series.astype(str).str.replace('_', '-'),
                            format='%Y-%m-%d')


def panel_equity(ax, per_day_df, title):
    days = _to_dt(per_day_df['day'])
    for name in SCHEME_ORDER:
        cum = per_day_df[name].cumsum()
        ax.plot(days, cum, label=name.replace(' (baseline)', '').replace(' (full stack)', ''),
                  color=SCHEME_COLORS[name],
                  linewidth=2.0 if name == 'B7 + B9 + B10 (full stack)' else 1.2,
                  linestyle='-' if name in ('FLAT (baseline)', 'B7 + B9 + B10 (full stack)') else '--',
                  alpha=0.85)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Cumulative $', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)


def panel_per_day_delta(ax, per_day_df, title):
    days = _to_dt(per_day_df['day'])
    delta = (per_day_df['B7 + B9 + B10 (full stack)']
             - per_day_df['FLAT (baseline)']).values
    colors = ['#2ca02c' if d >= 0 else '#d62728' for d in delta]
    ax.bar(days, delta, color=colors, width=0.7, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axhline(delta.mean(), color='blue', linestyle='--', linewidth=1.2,
                 label=f'mean ${delta.mean():.0f}/day')
    ax.set_ylabel('Full-stack − FLAT  ($)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)


def panel_day_distribution(ax, per_day_df, title):
    flat = per_day_df['FLAT (baseline)'].values
    full = per_day_df['B7 + B9 + B10 (full stack)'].values
    bins = np.linspace(min(flat.min(), full.min()) - 100,
                        max(flat.max(), full.max()) + 100, 25)
    ax.hist(flat, bins=bins, color=SCHEME_COLORS['FLAT (baseline)'],
              alpha=0.55, label=f'FLAT  μ=${flat.mean():.0f}', edgecolor='black')
    ax.hist(full, bins=bins, color=SCHEME_COLORS['B7 + B9 + B10 (full stack)'],
              alpha=0.55, label=f'Full stack  μ=${full.mean():.0f}', edgecolor='black')
    ax.axvline(0, color='k', linewidth=0.5)
    ax.axvline(flat.mean(), color=SCHEME_COLORS['FLAT (baseline)'],
                 linestyle='--', linewidth=1.5)
    ax.axvline(full.mean(), color=SCHEME_COLORS['B7 + B9 + B10 (full stack)'],
                 linestyle='--', linewidth=1.5)
    ax.set_xlabel('$/day', fontsize=10)
    ax.set_ylabel('Count of days', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)


def panel_b10_timeline(ax, per_day_df, title):
    days = _to_dt(per_day_df['day'])
    mults = per_day_df['b10_mult'].values
    colors = []
    for m in mults:
        if m == B10_BOOST: colors.append('#2ca02c')      # boost = green
        elif m == B10_CAP: colors.append('#d62728')      # cap = red
        else:              colors.append('#888888')      # hold = grey
    ax.bar(days, mults, color=colors, edgecolor='black', linewidth=0.3, width=0.7)
    ax.axhline(1.0, color='k', linewidth=0.5)
    ax.set_ylim(0.5, 1.5)
    ax.set_ylabel('B10 day-multiplier', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    n_boost = int((mults == B10_BOOST).sum())
    n_cap   = int((mults == B10_CAP).sum())
    n_hold  = len(mults) - n_boost - n_cap
    ax.text(0.02, 0.95,
              f'BOOST 1.3x: {n_boost}d  ({n_boost/len(mults)*100:.0f}%)\n'
              f'HOLD  1.0x: {n_hold}d  ({n_hold/len(mults)*100:.0f}%)\n'
              f'CAP   0.7x: {n_cap}d  ({n_cap/len(mults)*100:.0f}%)',
              transform=ax.transAxes, fontsize=9, verticalalignment='top',
              bbox=dict(facecolor='white', alpha=0.85, edgecolor='black'))
    ax.grid(alpha=0.3, axis='y')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)


def panel_b9_sizing(ax, legs_b, title):
    sizes = legs_b['b9_size'].values
    bins = [-0.05, 0.05, 0.55, 1.05, 1.55]
    labels = ['CUT (0)', 'HALF (0.5)', 'FULL (1.0)', 'PYRAMID (1.5)']
    counts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        counts.append(int(((sizes > lo) & (sizes <= hi)).sum()))
    colors = ['#d62728', '#ff7f0e', '#888888', '#2ca02c']
    bars = ax.bar(labels, counts, color=colors, edgecolor='black')
    for b, c in zip(bars, counts):
        pct = c / len(sizes) * 100
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                  f'{c}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Legs at K=5', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    # add b7 sizing as text overlay
    b7sz = legs_b['b7_size'].fillna(1.0).values
    ax.text(0.98, 0.95,
              f'B7 entry size:\n'
              f'  mean={b7sz.mean():.2f}\n'
              f'  median={np.median(b7sz):.2f}\n'
              f'  zero%={(b7sz == 0).mean()*100:.1f}%',
              transform=ax.transAxes, fontsize=9,
              verticalalignment='top', horizontalalignment='right',
              bbox=dict(facecolor='white', alpha=0.85, edgecolor='black'))


def render_oos_summary(per_day_df, legs_b):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    panel_scheme_bars(ax1, per_day_df, 'A. $/day per scheme (51 OOS days, * = sig vs FLAT)')

    ax2 = fig.add_subplot(gs[0, 1])
    panel_equity(ax2, per_day_df, 'B. Cumulative equity by scheme')

    ax3 = fig.add_subplot(gs[1, 0])
    panel_per_day_delta(ax3, per_day_df,
                          'C. Per-day delta: full stack − FLAT')

    ax4 = fig.add_subplot(gs[1, 1])
    panel_day_distribution(ax4, per_day_df,
                              'D. Distribution of $/day (FLAT vs full stack)')

    ax5 = fig.add_subplot(gs[2, 0])
    panel_b10_timeline(ax5, per_day_df,
                         'E. B10 day-multiplier timeline')

    ax6 = fig.add_subplot(gs[2, 1])
    panel_b9_sizing(ax6, legs_b,
                       'F. B9 action distribution at K=5 (per-leg)')

    flat = per_day_df['FLAT (baseline)'].values
    full = per_day_df['B7 + B9 + B10 (full stack)'].values
    delta = full - flat
    dl, dh = bootstrap_ci(delta)
    fig.suptitle(
        f'FULL-STACK FORWARD PASS — OOS SEALED  (51 days, 2026-03-19 to 2026-05-18)\n'
        f'FLAT ${flat.mean():.0f}/day  →  Full stack ${full.mean():.0f}/day  '
        f'(Δ ${delta.mean():+.0f}/day, 95% CI [${dl:+.0f}, ${dh:+.0f}],  '
        f'SIG {dl > 0})',
        fontsize=13, fontweight='bold', y=0.995)
    out = OUT_DIR / '2026-05-18_forward_pass_OOS_summary.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote: {out}')


def render_equity_zoom(per_day_df):
    fig, ax = plt.subplots(figsize=(14, 8))
    panel_equity(ax, per_day_df,
                   'OOS SEALED — Cumulative equity by sizing scheme  (51 days)')

    # annotate end values
    days = _to_dt(per_day_df['day'])
    for name in SCHEME_ORDER:
        cum = per_day_df[name].cumsum()
        ax.annotate(f'${cum.iloc[-1]:,.0f}',
                      xy=(days.iloc[-1], cum.iloc[-1]),
                      xytext=(8, 0), textcoords='offset points',
                      fontsize=9, va='center',
                      color=SCHEME_COLORS[name], fontweight='bold')

    out = OUT_DIR / '2026-05-18_forward_pass_OOS_equity.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote: {out}')


def render_is_vs_oos(is_per_day, oos_per_day):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, df, label in [(axes[0], is_per_day,  'IS (275d, in-sample CAVEAT)'),
                            (axes[1], oos_per_day, 'OOS SEALED (51d, honest)')]:
        means, lo, hi, sig = [], [], [], []
        flat = df['FLAT (baseline)'].values
        for name in SCHEME_ORDER:
            v = df[name].values
            means.append(v.mean())
            cl, ch = bootstrap_ci(v); lo.append(cl); hi.append(ch)
            sig.append(name != 'FLAT (baseline)' and bootstrap_ci(v - flat)[0] > 0)
        xs = np.arange(len(SCHEME_ORDER))
        colors = [SCHEME_COLORS[n] for n in SCHEME_ORDER]
        ax.bar(xs, means, color=colors, edgecolor='black', linewidth=0.6)
        ax.errorbar(xs, means,
                      yerr=[[m - l for m, l in zip(means, lo)],
                              [h - m for h, m in zip(hi, means)]],
                      fmt='none', ecolor='black', capsize=4)
        for x, m, s in zip(xs, means, sig):
            if s:
                ax.text(x, m, '*', ha='center', va='bottom',
                          fontsize=14, fontweight='bold', color='darkred')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(['FLAT', 'B7', 'B9', 'B10',
                              'B7+B9', 'B7+B10', 'B7+B9+B10'],
                             fontsize=9)
        ax.set_ylabel('$/day  (95% CI)', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for x, m in zip(xs, means):
            ax.text(x, m / 2, f'${m:.0f}', ha='center', va='center',
                      color='white' if m > 400 else 'black',
                      fontsize=9, fontweight='bold')

    fig.suptitle('IS vs OOS — full-stack forward pass  (* = significant vs FLAT, 95% bootstrap)',
                   fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = OUT_DIR / '2026-05-18_forward_pass_IS_vs_OOS.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote: {out}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Computing OOS sealed...')
    oos_legs, oos_schemes, _ = compute_stack(
        'OOS SEALED (51 days)',
        'reports/findings/regret_oracle/oos_hardened_legs_full.csv',
        'reports/findings/regret_oracle/trade_trajectory_OOS_full.parquet',
        'reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet',
        'NT8',
    )
    oos_per_day = per_day_table(oos_legs, oos_schemes)
    csv_out = OUT_DIR / '2026-05-18_per_day_OOS.csv'
    oos_per_day.to_csv(csv_out, index=False)
    print(f'  wrote: {csv_out}')

    print('\nComputing IS (caveat)...')
    is_legs, is_schemes, _ = compute_stack(
        'IS (275 days, CAVEAT)',
        'reports/findings/regret_oracle/is_hardened_legs.csv',
        'reports/findings/regret_oracle/trade_trajectory_IS.parquet',
        'reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet',
        'ATLAS',
    )
    is_per_day = per_day_table(is_legs, is_schemes)
    is_per_day.to_csv(OUT_DIR / '2026-05-18_per_day_IS.csv', index=False)

    print('\nRendering charts...')
    render_oos_summary(oos_per_day, oos_legs)
    render_equity_zoom(oos_per_day)
    render_is_vs_oos(is_per_day, oos_per_day)

    flat = oos_per_day['FLAT (baseline)'].values
    full = oos_per_day['B7 + B9 + B10 (full stack)'].values
    dl, dh = bootstrap_ci(full - flat)
    print(f'\n=== OOS HEADLINE ===')
    print(f'FLAT       ${flat.mean():+.0f}/day  total ${flat.sum():+,.0f}')
    print(f'Full stack ${full.mean():+.0f}/day  total ${full.sum():+,.0f}')
    print(f'Delta      ${(full - flat).mean():+.0f}/day  CI [${dl:+.0f}, ${dh:+.0f}]  SIG {dl > 0}')


if __name__ == '__main__':
    main()
