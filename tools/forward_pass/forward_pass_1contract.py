"""Phase-1 1-contract OOS forward pass with HONEST IS-calibrated thresholds.

The natural sizing-derived thresholds (B7 skip if pred_R < 1.0, B9 cut if
pred < -50) admit/keep everything because B7 was trained to predict R
multiples typically in [1.5, 4.0] and B9 predicts remaining amplitude
typically in [-15, +75]. At 1-contract cap, the sizing surface collapses
and these natural thresholds become no-ops.

The honest fix:
  1. On IS legs (275d, 17,767 legs), sweep B7 skip and B9 cut thresholds
  2. Pick the IS-optimal (max IS $/day delta over FLAT)
  3. Apply LOCKED thresholds to OOS sealed (51d, 2,936 legs)
  4. Report honest OOS number with bootstrap CI

Same protocol as the B9 K=5 single-shot OOS sealed test that gave +$67/day
in the multi-contract surface -- only this time at 1-contract cap.

Outputs:
    reports/findings/regret_oracle/forward_pass_1contract_OOS.txt
    reports/findings/regret_oracle/charts/2026-05-19_1contract_OOS.png
    reports/findings/regret_oracle/charts/2026-05-19_1contract_per_day.csv
"""
from __future__ import annotations
import pickle
import sys, os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

B7_PKL = 'reports/findings/regret_oracle/b7_leg_sizer.pkl'
B9_PKL = 'reports/findings/regret_oracle/b9_remaining_amplitude_K5.pkl'
B10_HIGH_PKL = 'reports/findings/regret_oracle/b10_vol_regime_high.pkl'
B10_LOW_PKL  = 'reports/findings/regret_oracle/b10_vol_regime_low.pkl'
CROSS_DAY = 'DATA/CROSS_DAY/cross_day_features.parquet'
OUT_DIR = Path('reports/findings/regret_oracle/charts')

B10_P_LOW_THR = 0.7
B10_FEATS = [
    'overnight_gap_pct', 'overnight_range_pct',
    'prior_day_range_pct', 'prior_day_c2c_pct',
    'vix_close_prior', 'vix_chg_prior',
    'dxy_close_prior', 'dxy_chg_prior',
    'is_fomc', 'is_cpi', 'is_nfp', 'is_opex',
    'days_since_fomc', 'days_to_next_fomc', 'dow',
]


def bootstrap_ci(values, n_boot=4000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = values[rng.integers(0, n, n)].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_models():
    with open(B7_PKL, 'rb') as f: b7 = pickle.load(f)
    with open(B9_PKL, 'rb') as f: b9 = pickle.load(f)
    with open(B10_HIGH_PKL, 'rb') as f: b10h = pickle.load(f)
    with open(B10_LOW_PKL,  'rb') as f: b10l = pickle.load(f)
    return b7, b9, b10h, b10l


def compute_b7_preds_on_legs(b7, legs_path, truth_path):
    """Return (legs DataFrame with leg_idx + b7_pred col)."""
    legs = pd.read_csv(legs_path)
    legs = legs.sort_values(['day', 'entry_ts']).reset_index(drop=True)
    legs['leg_idx'] = legs.index
    truth = pd.read_parquet(truth_path)
    day_truth = {d: g.reset_index(drop=True)
                  for d, g in truth.sort_values(['day', 'timestamp']).groupby('day')}
    preds = []
    for _, leg in legs.iterrows():
        day = leg['day']
        if day not in day_truth:
            preds.append(np.nan); continue
        td = day_truth[day]
        i = int(np.searchsorted(td['timestamp'].values, leg['entry_ts'],
                                  side='right') - 1)
        if i < 0 or i >= len(td):
            preds.append(np.nan); continue
        r = td.iloc[i]
        X = np.array([float(r[c]) if not pd.isna(r[c]) else 0.0
                       for c in b7['v2_cols']], dtype=np.float32).reshape(1, -1)
        preds.append(float(b7['model'].predict(X)[0]))
    legs['b7_pred'] = preds
    return legs


def compute_b9_preds_on_legs(b9, legs, traj_path):
    traj = pd.read_parquet(traj_path)
    traj_k5 = traj[traj['K'] == 5].copy()
    X = traj_k5[b9['feat_cols']].fillna(0.0).values
    traj_k5['b9_pred'] = b9['model'].predict(X)
    legs_b = legs.merge(traj_k5[['leg_id', 'b9_pred', 'pnl_usd_so_far']],
                          left_on='leg_idx', right_on='leg_id', how='left')
    return legs_b


def calibrate_thresholds_on_IS(legs_is):
    """Sweep B7 skip + B9 cut thresholds on IS. Pick IS-best (in terms of
    paired delta vs FLAT mean) and return (b7_thr, b9_thr, sweep_table)."""
    pnl = legs_is['pnl_usd'].values
    pnl_k5 = legs_is['pnl_usd_so_far'].fillna(0.0).values
    b7p = legs_is['b7_pred'].fillna(np.inf).values
    b9p = legs_is['b9_pred'].fillna(np.inf).values
    n_days = len(set(legs_is['day']))

    # B7 sweep (skip-filter): pred_R must be >= thr to take
    b7_sweep = []
    for thr in np.arange(1.0, 2.51, 0.1):
        take = (b7p >= thr) | (b7p == np.inf)
        realized = np.where(take, pnl, 0.0)
        b7_sweep.append((thr, realized.sum() / n_days,
                          realized.sum() - pnl.sum()))
    b7_sweep_df = pd.DataFrame(b7_sweep, columns=['thr', 'per_day', 'total_delta'])

    # B9 sweep (cut-filter): cut if pred < thr (sweep -50..+30)
    b9_sweep = []
    for thr in np.arange(-50.0, 30.01, 5.0):
        cut = (b9p < thr) & (b9p != np.inf)
        realized = np.where(cut, pnl_k5, pnl)
        b9_sweep.append((thr, realized.sum() / n_days,
                          realized.sum() - pnl.sum()))
    b9_sweep_df = pd.DataFrame(b9_sweep, columns=['thr', 'per_day', 'total_delta'])

    # Pick IS-optimal independently for B7 and B9 (additive assumption)
    best_b7 = b7_sweep_df.loc[b7_sweep_df['total_delta'].idxmax()]
    best_b9 = b9_sweep_df.loc[b9_sweep_df['total_delta'].idxmax()]

    return (float(best_b7['thr']), float(best_b9['thr']),
            b7_sweep_df, b9_sweep_df)


def apply_phase1_logic(legs, b7_thr, b9_thr):
    """Apply B7 skip + B9 cut to legs DataFrame. Returns realized array."""
    pnl = legs['pnl_usd'].values
    pnl_k5 = legs['pnl_usd_so_far'].fillna(0.0).values
    b7p = legs['b7_pred'].fillna(np.inf).values
    b9p = legs['b9_pred'].fillna(np.inf).values
    take = (b7p >= b7_thr) | (b7p == np.inf)
    cut = (b9p < b9_thr) & (b9p != np.inf)
    after_b9 = np.where(cut, pnl_k5, pnl)
    realized = np.where(take, after_b9, 0.0)
    return realized


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b7, b9, b10h, b10l = load_models()

    # 1. IS: load + predict + sweep
    print('Loading IS legs + predictions...')
    legs_is = compute_b7_preds_on_legs(
        b7,
        'reports/findings/regret_oracle/is_hardened_legs.csv',
        'reports/findings/regret_oracle/zigzag_pivot_dataset_IS_atr4.parquet')
    legs_is = compute_b9_preds_on_legs(
        b9, legs_is,
        'reports/findings/regret_oracle/trade_trajectory_IS.parquet')
    print(f'IS: {len(legs_is)} legs, {legs_is["day"].nunique()} days')

    # 2. OOS: load + predict
    print('Loading OOS legs + predictions...')
    legs_oos = compute_b7_preds_on_legs(
        b7,
        'reports/findings/regret_oracle/oos_hardened_legs_full.csv',
        'reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    legs_oos = compute_b9_preds_on_legs(
        b9, legs_oos,
        'reports/findings/regret_oracle/trade_trajectory_OOS_full.parquet')
    print(f'OOS: {len(legs_oos)} legs, {legs_oos["day"].nunique()} days')

    # 3. B10 mode lookup per day
    feats = pd.read_parquet(CROSS_DAY)
    nt8_feats = feats[feats['source'] == 'NT8'].copy()
    X = nt8_feats[B10_FEATS].fillna(0.0).values.astype(np.float32)
    nt8_feats['p_low'] = b10l['model'].predict_proba(X)[:, 1]
    nt8_feats['b10_mode'] = np.where(nt8_feats['p_low'] >= B10_P_LOW_THR,
                                       'cautious', 'normal')
    day_to_mode_oos = dict(zip(nt8_feats['date_label'], nt8_feats['b10_mode']))
    legs_oos['b10_mode'] = legs_oos['day'].map(day_to_mode_oos).fillna('normal')

    # For IS calibration, B10 modes from ATLAS source
    atlas_feats = feats[feats['source'] == 'ATLAS'].copy()
    if not atlas_feats.empty:
        X2 = atlas_feats[B10_FEATS].fillna(0.0).values.astype(np.float32)
        atlas_feats['p_low'] = b10l['model'].predict_proba(X2)[:, 1]
        atlas_feats['b10_mode'] = np.where(
            atlas_feats['p_low'] >= B10_P_LOW_THR, 'cautious', 'normal')
        day_to_mode_is = dict(zip(atlas_feats['date_label'], atlas_feats['b10_mode']))
        legs_is['b10_mode'] = legs_is['day'].map(day_to_mode_is).fillna('normal')
    else:
        legs_is['b10_mode'] = 'normal'

    # 4. IS-calibrate (normal-mode legs only -- cautious is a rare special case)
    legs_is_normal = legs_is[legs_is['b10_mode'] == 'normal'].reset_index(drop=True)
    print(f'\nIS calibration on {len(legs_is_normal)} normal-mode legs...')
    b7_thr_normal, b9_thr_normal, b7_sw, b9_sw = calibrate_thresholds_on_IS(legs_is_normal)
    print(f'IS-OPTIMAL  B7 skip threshold: {b7_thr_normal:.2f}  '
          f'B9 cut threshold: {b9_thr_normal:.2f}')

    # For cautious mode, use TIGHTER thresholds (heuristic: +0.2 to B7, +10 to B9)
    b7_thr_cautious = b7_thr_normal + 0.2
    b9_thr_cautious = b9_thr_normal + 10.0
    print(f'CAUTIOUS    B7 skip threshold: {b7_thr_cautious:.2f}  '
          f'B9 cut threshold: {b9_thr_cautious:.2f}')

    # 5. Apply LOCKED thresholds to OOS
    legs_oos_norm = legs_oos[legs_oos['b10_mode'] == 'normal'].reset_index(drop=True)
    legs_oos_caut = legs_oos[legs_oos['b10_mode'] == 'cautious'].reset_index(drop=True)
    realized_norm = apply_phase1_logic(legs_oos_norm, b7_thr_normal, b9_thr_normal)
    realized_caut = apply_phase1_logic(legs_oos_caut, b7_thr_cautious, b9_thr_cautious)

    # Combine back into single DataFrame, per-day aggregation
    legs_oos['_phase1'] = np.nan
    legs_oos.loc[legs_oos['b10_mode'] == 'normal',  '_phase1'] = realized_norm
    legs_oos.loc[legs_oos['b10_mode'] == 'cautious','_phase1'] = realized_caut

    pnl_total = legs_oos['pnl_usd'].values
    pnl_k5    = legs_oos['pnl_usd_so_far'].fillna(0.0).values
    b7p       = legs_oos['b7_pred'].fillna(np.inf).values
    b9p       = legs_oos['b9_pred'].fillna(np.inf).values
    mode      = legs_oos['b10_mode'].values
    b7_thr_arr = np.where(mode == 'cautious', b7_thr_cautious, b7_thr_normal)
    b9_thr_arr = np.where(mode == 'cautious', b9_thr_cautious, b9_thr_normal)

    schemes = {
        'FLAT 1c':            pnl_total.copy(),
        'B7 skip only':       pnl_total * ((b7p >= b7_thr_arr) | (b7p == np.inf)).astype(float),
        'B9 cut only':        np.where(((b9p < b9_thr_arr) & (b9p != np.inf)),
                                          pnl_k5, pnl_total),
        'B7 + B9':            ((b7p >= b7_thr_arr) | (b7p == np.inf)).astype(float) *
                              np.where(((b9p < b9_thr_arr) & (b9p != np.inf)),
                                          pnl_k5, pnl_total),
        'B7 + B9 + B10 (Phase-1 1c)': legs_oos['_phase1'].values,
    }

    days = sorted(legs_oos['day'].unique())
    per_day = pd.DataFrame({'day': days})
    per_day['n_legs'] = [int((legs_oos['day'] == d).sum()) for d in days]
    per_day['b10_mode'] = [day_to_mode_oos.get(d, 'normal') for d in days]
    for name, vals in schemes.items():
        per_day[name] = [float(vals[legs_oos['day'] == d].sum()) for d in days]

    # ─── Report ────────────────────────────────────────────────────────
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 100)
    out('PHASE-1 1-CONTRACT FORWARD PASS: OOS SEALED (IS-CALIBRATED THRESHOLDS)')
    out('=' * 100)
    out(f'IS-locked thresholds (normal mode):  '
        f'B7_skip={b7_thr_normal:.2f}  B9_cut={b9_thr_normal:.2f}')
    out(f'IS-locked thresholds (cautious mode): '
        f'B7_skip={b7_thr_cautious:.2f}  B9_cut={b9_thr_cautious:.2f}')
    n_caut = sum(1 for d in days if day_to_mode_oos.get(d) == 'cautious')
    out(f'OOS days: {len(per_day)}  Legs: {len(legs_oos):,}  '
        f'Cautious days: {n_caut}')
    out('')
    out(f'{"scheme":<35}  {"total_$":>11}  {"$/day":>9}  '
        f'{"95% CI":>22}  {"sig vs FLAT":>13}')

    flat = per_day['FLAT 1c'].values
    flat_lo, flat_hi = bootstrap_ci(flat)
    out(f'{"FLAT 1c":<35}  ${flat.sum():>+9,.0f}  ${flat.mean():>+7.0f}  '
        f'[${flat_lo:>+5.0f}, ${flat_hi:>+5.0f}]      {"baseline":>13}')
    for name in ['B7 skip only', 'B9 cut only', 'B7 + B9',
                  'B7 + B9 + B10 (Phase-1 1c)']:
        v = per_day[name].values
        cl, ch = bootstrap_ci(v)
        delta = v - flat
        dl, dh = bootstrap_ci(delta)
        sig = dl > 0
        out(f'{name:<35}  ${v.sum():>+9,.0f}  ${v.mean():>+7.0f}  '
            f'[${cl:>+5.0f}, ${ch:>+5.0f}]      {str(sig):>13}')

    out('')
    out('--- Delta vs FLAT (95% bootstrap CI) ---')
    for name in ['B7 skip only', 'B9 cut only', 'B7 + B9',
                  'B7 + B9 + B10 (Phase-1 1c)']:
        delta = per_day[name].values - flat
        cl, ch = bootstrap_ci(delta)
        sig = cl > 0
        out(f'  {name:<35}  delta ${delta.mean():>+5.0f}/day  '
            f'CI [${cl:>+5.0f}, ${ch:>+5.0f}]    sig {sig}')

    out('')
    out('--- IS calibration summary (top 3 B7 thresholds tested) ---')
    for _, r in b7_sw.nlargest(3, 'total_delta').iterrows():
        out(f'  B7 skip thr={r["thr"]:.2f}: IS $/day delta=${r["total_delta"]/legs_is_normal["day"].nunique():+.0f}')
    out('--- IS calibration summary (top 3 B9 thresholds tested) ---')
    for _, r in b9_sw.nlargest(3, 'total_delta').iterrows():
        out(f'  B9 cut thr={r["thr"]:+.0f}: IS $/day delta=${r["total_delta"]/legs_is_normal["day"].nunique():+.0f}')

    txt_path = 'reports/findings/regret_oracle/forward_pass_1contract_OOS.txt'
    Path(txt_path).write_text('\n'.join(lines), encoding='utf-8')
    csv_path = OUT_DIR / '2026-05-19_1contract_per_day.csv'
    per_day.to_csv(csv_path, index=False)
    print(f'\nWrote: {txt_path}')
    print(f'       {csv_path}')

    render_chart(per_day, b7_thr_normal, b9_thr_normal)

    full = per_day['B7 + B9 + B10 (Phase-1 1c)'].values
    delta = full - flat
    dl, dh = bootstrap_ci(delta)
    out('')
    out('=' * 100)
    out(f'HEADLINE  Phase-1 1c OOS:  ${full.mean():+.0f}/day  '
        f'(FLAT 1c ${flat.mean():+.0f}/day)')
    out(f'   delta vs FLAT 1c: ${delta.mean():+.0f}/day  '
        f'CI [${dl:+.0f}, ${dh:+.0f}]   SIG {dl > 0}')
    out('=' * 100)


def render_chart(per_day, b7_thr, b9_thr):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22)

    schemes = ['FLAT 1c', 'B7 skip only', 'B9 cut only', 'B7 + B9',
                 'B7 + B9 + B10 (Phase-1 1c)']
    colors = ['#888888', '#377eb8', '#4daf4a', '#a65628', '#e41a1c']
    flat = per_day['FLAT 1c'].values

    ax = fig.add_subplot(gs[0, 0])
    means, lo, hi, sig = [], [], [], []
    for s in schemes:
        v = per_day[s].values
        means.append(v.mean())
        cl, ch = bootstrap_ci(v); lo.append(cl); hi.append(ch)
        if s == 'FLAT 1c':
            sig.append(False)
        else:
            dl, _ = bootstrap_ci(v - flat); sig.append(dl > 0)
    xs = np.arange(len(schemes))
    ax.bar(xs, means, color=colors, edgecolor='black')
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
    ax.set_xticklabels(['FLAT', 'B7', 'B9', 'B7+B9', 'Phase-1'], fontsize=9)
    ax.set_ylabel('$/day (95% CI)', fontsize=10)
    ax.set_title(f'A. $/day per scheme (51 OOS days)\n'
                   f'(IS-locked thresholds: B7>={b7_thr:.2f}, B9<{b9_thr:.0f})',
                   fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for x, m in zip(xs, means):
        ax.text(x, m / 2, f'${m:.0f}', ha='center', va='center',
                  color='white' if abs(m) > 100 else 'black',
                  fontsize=9, fontweight='bold')

    ax = fig.add_subplot(gs[0, 1])
    days = pd.to_datetime(per_day['day'].astype(str).str.replace('_', '-'))
    for s, c in zip(schemes, colors):
        cum = per_day[s].cumsum()
        ax.plot(days, cum, label=s, color=c,
                  linewidth=2.0 if 'Phase-1' in s else 1.2)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Cumulative $', fontsize=10)
    ax.set_title('B. Cumulative equity', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    ax = fig.add_subplot(gs[1, 0])
    delta = (per_day['B7 + B9 + B10 (Phase-1 1c)'] - per_day['FLAT 1c']).values
    cdeltas = ['#2ca02c' if d >= 0 else '#d62728' for d in delta]
    ax.bar(days, delta, color=cdeltas, width=0.7,
             edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axhline(delta.mean(), color='blue', linestyle='--',
                 label=f'mean ${delta.mean():.0f}/day')
    ax.set_ylabel('Phase-1 - FLAT ($)', fontsize=10)
    ax.set_title('C. Per-day delta', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    ax = fig.add_subplot(gs[1, 1])
    modes_num = (per_day['b10_mode'].values == 'cautious').astype(int)
    cmodes = ['#d62728' if m == 1 else '#888888' for m in modes_num]
    ax.bar(days, np.ones(len(days)), color=cmodes,
             width=0.8, edgecolor='black', linewidth=0.3)
    ax.set_yticks([])
    ax.set_title('D. B10 day-mode  (red=cautious, grey=normal)',
                   fontsize=11, fontweight='bold')
    n_caut = int(modes_num.sum())
    ax.text(0.02, 0.95,
              f'Cautious: {n_caut}d / {len(modes_num)}d',
              transform=ax.transAxes, fontsize=10,
              verticalalignment='top',
              bbox=dict(facecolor='white', edgecolor='black', alpha=0.9))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    flat_mean = per_day['FLAT 1c'].mean()
    full_mean = per_day['B7 + B9 + B10 (Phase-1 1c)'].mean()
    delta_mean = full_mean - flat_mean
    dl, dh = bootstrap_ci(delta)
    fig.suptitle(
        f'PHASE-1 1-CONTRACT  (OOS sealed 51d, IS-calibrated thresholds, '
        f'SIM deploy candidate)\n'
        f'FLAT 1c ${flat_mean:.0f}/day  ->  Phase-1 ${full_mean:.0f}/day  '
        f'(delta ${delta_mean:+.0f}/day  95% CI [${dl:+.0f}, ${dh:+.0f}]  '
        f'SIG {dl > 0})',
        fontsize=11, fontweight='bold', y=0.998)
    out_png = OUT_DIR / '2026-05-19_1contract_OOS.png'
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'       {out_png}')


if __name__ == '__main__':
    main()
