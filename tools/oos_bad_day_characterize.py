"""MEASURE phase: characterize the OOS bad days for the L5 stack.

DMAIC MEASURE step. Goal of the parent project: lift the OOS bad days.
This script does NOT modify production code or train anything -- it only
characterizes the 51-day sealed OOS window.

Inputs (all pre-existing, validated forward-pass artifacts):
  - reports/findings/regret_oracle/charts/2026-05-19_1contract_per_day.csv
        Per-day OOS P&L for FLAT 1c and the full Phase-1 1c stack
        (B7 skip + B9 cut + B10 mode). This is the validated 1-contract
        forward pass with IS-locked thresholds.
  - reports/findings/regret_oracle/oos_hardened_legs_full.csv
        Per-leg hardened zigzag legs (entry/exit ts, pnl, amplitude).
  - DATA/CROSS_DAY/cross_day_features.parquet  (source == 'NT8')
        Cross-day context: overnight gap, prior-day range, VIX/DXY, events.
  - DATA/ATLAS_NT8/1m/{day}.parquet
        1-minute bars -> intraday realized vol (std of 1m log returns).

Outputs:
  - reports/findings/oos_bad_days/oos_day_features.csv   (per-day table)
  - reports/findings/oos_bad_days/2026-05-21_measure.md  (report)

Metric conventions (CLAUDE.md):
  - $/day deltas reported with 95% bootstrap CI (4,000 resamples, percentile).
  - "Bad day" = negative-P&L OOS day (defined on the FLAT baseline; the
    stacked-defined set is also reported).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PER_DAY_CSV = REPO / 'reports/findings/regret_oracle/charts/2026-05-19_1contract_per_day.csv'
LEGS_CSV = REPO / 'reports/findings/regret_oracle/oos_hardened_legs_full.csv'
CROSS_DAY = REPO / 'DATA/CROSS_DAY/cross_day_features.parquet'
ATLAS_1M = REPO / 'DATA/ATLAS_NT8/1m'

OUT_DIR = REPO / 'reports/findings/oos_bad_days'
OUT_CSV = OUT_DIR / 'oos_day_features.csv'
OUT_MD = OUT_DIR / '2026-05-21_measure.md'

STACK_COL = 'B7 + B9 + B10 (Phase-1 1c)'   # validated full stack column
FLAT_COL = 'FLAT 1c'

N_BOOT = 4000
BOOT_SEED = 42


def bootstrap_ci(values, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile bootstrap CI of the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = values[rng.integers(0, n, n)].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def intraday_vol(day_label: str) -> dict:
    """Realized-vol + structure features from the 1m parquet for one day.

    Returns std of 1m log returns (realized vol), 1m bar count, and the
    intraday close-to-close range as a pct of the open.
    """
    fp = ATLAS_1M / f'{day_label}.parquet'
    if not fp.exists():
        return dict(rv_1m=np.nan, n_1m_bars=np.nan, intraday_range_pct=np.nan)
    m = pd.read_parquet(fp)
    if len(m) < 3:
        return dict(rv_1m=np.nan, n_1m_bars=len(m), intraday_range_pct=np.nan)
    c = m['close'].astype(float).values
    logret = np.diff(np.log(c))
    rv = float(np.std(logret, ddof=1))
    rng_pct = float((m['high'].max() - m['low'].min()) / m['open'].iloc[0])
    return dict(rv_1m=rv, n_1m_bars=int(len(m)), intraday_range_pct=rng_pct)


def leg_structure(g: pd.DataFrame) -> dict:
    """Per-day intraday leg-structure features from the hardened leg CSV."""
    amp = g['pnl_pts'].abs().values            # leg amplitude proxy (pts)
    dur = (g['exit_ts'] - g['entry_ts']).values  # seconds
    pnl = g['pnl_usd'].values
    return dict(
        n_legs=int(len(g)),
        leg_amp_mean=float(np.mean(amp)),
        leg_amp_median=float(np.median(amp)),
        leg_amp_std=float(np.std(amp, ddof=1)) if len(amp) > 1 else 0.0,
        leg_dur_mean_s=float(np.mean(dur)),
        leg_dur_total_s=float(np.sum(dur)),
        leg_pnl_mean=float(np.mean(pnl)),
        leg_pnl_std=float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0,
        frac_winner_legs=float(np.mean(pnl > 0)),
        atr_pts=float(g['atr_pts'].iloc[0]) if 'atr_pts' in g else np.nan,
        worst_leg_usd=float(np.min(pnl)),
        best_leg_usd=float(np.max(pnl)),
    )


def build_feature_table() -> pd.DataFrame:
    per_day = pd.read_csv(PER_DAY_CSV)
    per_day = per_day.rename(columns={FLAT_COL: 'pnl_flat',
                                      STACK_COL: 'pnl_stack'})
    per_day = per_day[['day', 'n_legs', 'b10_mode', 'pnl_flat', 'pnl_stack',
                       'B7 skip only', 'B9 cut only']].copy()
    per_day = per_day.rename(columns={'B7 skip only': 'pnl_b7_only',
                                      'B9 cut only': 'pnl_b9_only',
                                      'n_legs': 'n_legs_perday'})

    # --- leg structure per day ---
    legs = pd.read_csv(LEGS_CSV)
    struct_rows = []
    for day, g in legs.groupby('day'):
        row = dict(day=day)
        row.update(leg_structure(g))
        struct_rows.append(row)
    struct = pd.DataFrame(struct_rows)

    # --- intraday realized vol per day ---
    rv_rows = []
    for day in per_day['day']:
        row = dict(day=day)
        row.update(intraday_vol(day))
        rv_rows.append(row)
    rv = pd.DataFrame(rv_rows)

    # --- cross-day features (NT8 source) ---
    cd = pd.read_parquet(CROSS_DAY)
    cd = cd[cd['source'] == 'NT8'].copy()
    cd_cols = ['date_label', 'dow', 'overnight_gap_pct', 'overnight_range_pct',
               'prior_day_range_pct', 'prior_day_c2c_pct',
               'vix_close_prior', 'vix_chg_prior',
               'dxy_close_prior', 'dxy_chg_prior',
               'is_fomc', 'is_cpi', 'is_nfp', 'is_opex',
               'days_since_fomc', 'days_to_next_fomc']
    cd = cd[cd_cols].rename(columns={'date_label': 'day'})

    tbl = (per_day
           .merge(struct, on='day', how='left')
           .merge(rv, on='day', how='left')
           .merge(cd, on='day', how='left'))

    # weekday name (works even where cross-day dow is missing)
    tbl['weekday'] = tbl['day'].apply(
        lambda d: pd.Timestamp(str(d).replace('_', '-')).day_name())
    tbl['has_crossday'] = tbl['overnight_gap_pct'].notna()

    # bad-day flags
    tbl['bad_flat'] = tbl['pnl_flat'] < 0
    tbl['bad_stack'] = tbl['pnl_stack'] < 0
    tbl['stack_minus_flat'] = tbl['pnl_stack'] - tbl['pnl_flat']

    return tbl.sort_values('day').reset_index(drop=True)


def fmt_money(x):
    return f'${x:+,.0f}'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl = build_feature_table()
    tbl.to_csv(OUT_CSV, index=False)

    L = []
    def out(s=''):
        L.append(s)

    n = len(tbl)
    flat = tbl['pnl_flat'].values
    stack = tbl['pnl_stack'].values

    bad = tbl[tbl['bad_flat']].sort_values('pnl_flat')
    good = tbl[~tbl['bad_flat']]
    n_bad, n_good = len(bad), len(good)

    total_oos_flat = flat.sum()
    total_lost = bad['pnl_flat'].sum()             # negative
    total_won_good = good['pnl_flat'].sum()        # positive
    gross_loss = -total_lost
    gross_profit = total_won_good

    out('# OOS Bad-Day Characterization (MEASURE)')
    out('')
    out('DMAIC MEASURE phase. Project goal: lift the OOS bad days -- reduce '
        'how often and how badly the L5 stack loses on its worst sealed-OOS '
        'days. This document characterizes only; no code or model changed.')
    out('')
    out(f'- **Date:** 2026-05-21')
    out(f'- **OOS window:** 51 days, 2026-03-19 to 2026-05-18 (sealed)')
    out(f'- **Strategy:** L5 stack = zigzag FLAT baseline + B7 entry-skip + '
        f'B9 during-trade sizer + B10 day-regime mode, 1-contract cap')
    out(f'- **P&L source:** validated forward pass '
        f'`reports/findings/regret_oracle/charts/2026-05-19_1contract_per_day.csv`')
    out(f'- **Per-day feature table:** `{OUT_CSV.relative_to(REPO)}`')
    out('')

    # ---- 1. headline P&L ----
    flat_lo, flat_hi = bootstrap_ci(flat)
    stack_lo, stack_hi = bootstrap_ci(stack)
    out('## 1. Headline OOS P&L')
    out('')
    out('| scheme | total $ | $/day | 95% CI on $/day |')
    out('|---|--:|--:|--:|')
    out(f'| FLAT 1c | {fmt_money(flat.sum())} | {fmt_money(flat.mean())} '
        f'| [{fmt_money(flat_lo)}, {fmt_money(flat_hi)}] |')
    out(f'| L5 full stack 1c | {fmt_money(stack.sum())} | {fmt_money(stack.mean())} '
        f'| [{fmt_money(stack_lo)}, {fmt_money(stack_hi)}] |')
    d = stack - flat
    d_lo, d_hi = bootstrap_ci(d)
    out('')
    out(f'Stack vs FLAT paired delta: **{fmt_money(d.mean())}/day** '
        f'(95% CI [{fmt_money(d_lo)}, {fmt_money(d_hi)}]) -- '
        f'{"SIGNIFICANT" if d_lo > 0 else "NOT significant (CI includes 0)"}.')
    out('')
    out('Note: the 1-contract Phase-1 stack delta is small and CI-crosses-zero '
        '(consistent with `forward_pass_1contract_OOS.txt`). The multi-contract '
        'sizing surface (`forward_pass_stack_OOS.txt`) shows a larger lift; '
        'this MEASURE uses the 1-contract per-day series because it is the '
        'SIM-deploy candidate with a full validated per-day breakdown.')
    out('')

    # ---- 2. bad-day count / severity / concentration ----
    out('## 2. Bad days: count, severity, concentration')
    out('')
    out(f'- **Bad OOS days (FLAT P&L < 0): {n_bad} / {n} '
        f'({100*n_bad/n:.0f}%)**. Good days: {n_good} ({100*n_good/n:.0f}%).')
    out(f'- Day win rate (count-based): {n_good}/{n} = {100*n_good/n:.1f}%.')
    out(f'- **Total $ lost on bad days: {fmt_money(total_lost)}** '
        f'(gross loss {fmt_money(-gross_loss)}).')
    out(f'- Total $ won on good days: {fmt_money(total_won_good)}.')
    out(f'- Net OOS (FLAT): {fmt_money(total_oos_flat)}.')
    bad_mean = bad['pnl_flat'].mean()
    good_mean = good['pnl_flat'].mean()
    out(f'- Mean bad day {fmt_money(bad_mean)}; mean good day '
        f'{fmt_money(good_mean)}. Asymmetry ratio (good/|bad|): '
        f'{good_mean/abs(bad_mean):.2f}x.')
    out('')
    # concentration
    worst8 = bad.head(8)
    worst8_sum = worst8['pnl_flat'].sum()
    out(f'- **Loss concentration:** the 8 worst days lose '
        f'{fmt_money(worst8_sum)} = {100*worst8_sum/total_lost:.0f}% of all '
        f'bad-day losses, and {100*abs(worst8_sum)/gross_profit:.0f}% of total '
        f'good-day gains.')
    best8 = good.sort_values('pnl_flat', ascending=False).head(8)
    best8_sum = best8['pnl_flat'].sum()
    out(f'- **Gain concentration:** the 8 best days make '
        f'{fmt_money(best8_sum)} = {100*best8_sum/gross_profit:.0f}% of all '
        f'good-day gains. Net OOS is carried by a thin tail on both sides.')
    out('')
    out('### Worst 8 OOS days (by FLAT P&L)')
    out('')
    out('| day | weekday | FLAT $ | stack $ | stack-FLAT | n_legs | '
        'leg_amp_med (pts) | rv_1m | VIX prior | event |')
    out('|---|---|--:|--:|--:|--:|--:|--:|--:|---|')
    for _, r in worst8.iterrows():
        evs = [e for e, f in [('FOMC', r['is_fomc']), ('CPI', r['is_cpi']),
                              ('NFP', r['is_nfp']), ('OPEX', r['is_opex'])]
               if f == 1]
        ev = ','.join(evs) if evs else '-'
        vix = f'{r["vix_close_prior"]:.1f}' if pd.notna(r['vix_close_prior']) else 'n/a'
        rvv = f'{r["rv_1m"]*1e4:.1f}' if pd.notna(r['rv_1m']) else 'n/a'
        out(f'| {r["day"]} | {r["weekday"][:3]} | {fmt_money(r["pnl_flat"])} '
            f'| {fmt_money(r["pnl_stack"])} | {fmt_money(r["stack_minus_flat"])} '
            f'| {int(r["n_legs_perday"])} | {r["leg_amp_median"]:.1f} '
            f'| {rvv} | {vix} | {ev} |')
    out('')
    out('(`rv_1m` shown x1e4 = std of 1-minute log returns, in basis points.)')
    out('')

    # ---- 3. does the stack help or hurt on bad days? ----
    out('## 3. Does the B-stack help or hurt on bad days?')
    out('')
    bad_flat_v = bad['pnl_flat'].values
    bad_stack_v = bad['pnl_stack'].values
    bad_delta = bad_stack_v - bad_flat_v
    bd_lo, bd_hi = bootstrap_ci(bad_delta)
    out(f'On the {n_bad} FLAT-negative days:')
    out('')
    out(f'- FLAT mean {fmt_money(bad_flat_v.mean())}/day -> '
        f'stack mean {fmt_money(bad_stack_v.mean())}/day.')
    out(f'- **Paired delta on bad days: {fmt_money(bad_delta.mean())}/day** '
        f'(95% CI [{fmt_money(bd_lo)}, {fmt_money(bd_hi)}]) -- '
        f'{"SIGNIFICANT" if (bd_lo > 0 or bd_hi < 0) else "NOT significant (CI includes 0)"}.')
    n_improved = int((bad_delta > 0).sum())
    n_worse = int((bad_delta < 0).sum())
    n_flipped_pos = int((bad_stack_v > 0).sum())
    out(f'- Stack improved {n_improved}/{n_bad} bad days, worsened '
        f'{n_worse}/{n_bad}, unchanged {n_bad - n_improved - n_worse}.')
    out(f'- Bad days the stack flipped to POSITIVE: {n_flipped_pos}/{n_bad}.')
    out(f'- Total bad-day $ : FLAT {fmt_money(bad_flat_v.sum())} -> '
        f'stack {fmt_money(bad_stack_v.sum())} '
        f'(recovered {fmt_money(bad_stack_v.sum() - bad_flat_v.sum())}).')
    # worst-day tail behaviour
    out('')
    out(f'- Worst single FLAT day {fmt_money(bad_flat_v.min())}; under the '
        f'stack the worst day is {fmt_money(bad_stack_v.min())}.')
    # good-day side effect
    good_delta = good['pnl_stack'].values - good['pnl_flat'].values
    gd_lo, gd_hi = bootstrap_ci(good_delta)
    out(f'- For context, on the {n_good} good days the stack delta is '
        f'{fmt_money(good_delta.mean())}/day (CI [{fmt_money(gd_lo)}, '
        f'{fmt_money(gd_hi)}]).')
    out('')
    if bad_delta.mean() > 0:
        out('**Read:** the B-stack is net-positive on bad days in point '
            'estimate (it cuts/skips some losing legs), but the per-day delta '
            'CI behaviour determines significance -- see number above. The '
            'stack does NOT reliably turn bad days good; it shaves the loss.')
    else:
        out('**Read:** the B-stack does NOT help bad days on average -- it is '
            'flat-to-negative there. Its OOS edge comes from good days.')
    out('')

    # ---- 4. what distinguishes bad days from good days ----
    out('## 4. First read: what distinguishes bad days from good days')
    out('')
    out('Group means (bad vs good), and point-biserial correlation of each '
        'feature with the bad-day flag. No models -- eyeball only. Features '
        'with the largest standardized gap are listed first. Cross-day '
        f'features available on {int(tbl["has_crossday"].sum())}/{n} days '
        '(11 Sunday-evening sessions have no prior-RTH day -> NaN; '
        'correlations for those features use the available subset).')
    out('')
    feat_cols = [
        'n_legs_perday', 'leg_amp_mean', 'leg_amp_median', 'leg_amp_std',
        'leg_dur_mean_s', 'leg_dur_total_s', 'leg_pnl_std', 'frac_winner_legs',
        'atr_pts', 'worst_leg_usd', 'best_leg_usd',
        'rv_1m', 'n_1m_bars', 'intraday_range_pct',
        'overnight_gap_pct', 'overnight_range_pct', 'prior_day_range_pct',
        'prior_day_c2c_pct', 'vix_close_prior', 'vix_chg_prior',
        'dxy_close_prior', 'dxy_chg_prior', 'days_since_fomc',
        'days_to_next_fomc',
    ]
    rows = []
    bad_mask = tbl['bad_flat'].values.astype(float)
    for c in feat_cols:
        v = tbl[c].astype(float)
        sub = tbl[v.notna()]
        if len(sub) < 5 or sub['bad_flat'].nunique() < 2:
            continue
        bm = sub.loc[sub['bad_flat'], c].mean()
        gm = sub.loc[~sub['bad_flat'], c].mean()
        pooled_std = sub[c].std(ddof=1)
        std_gap = (bm - gm) / pooled_std if pooled_std > 0 else 0.0
        corr = np.corrcoef(sub['bad_flat'].astype(float), sub[c])[0, 1]
        rows.append(dict(feature=c, bad_mean=bm, good_mean=gm,
                         std_gap=std_gap, corr_bad=corr, n=len(sub)))
    fd = pd.DataFrame(rows).reindex(
        pd.DataFrame(rows)['std_gap'].abs().sort_values(ascending=False).index)
    out('| feature | bad-day mean | good-day mean | std gap (bad-good) | '
        'corr w/ bad flag | n |')
    out('|---|--:|--:|--:|--:|--:|')
    for _, r in fd.iterrows():
        out(f'| {r["feature"]} | {r["bad_mean"]:.4g} | {r["good_mean"]:.4g} '
            f'| {r["std_gap"]:+.2f} | {r["corr_bad"]:+.2f} | {int(r["n"])} |')
    out('')

    # event-day rate
    out('### Event-day rate (bad vs good)')
    out('')
    for ev in ['is_fomc', 'is_cpi', 'is_nfp', 'is_opex']:
        sub = tbl[tbl[ev].notna()]
        br = sub.loc[sub['bad_flat'], ev].mean()
        gr = sub.loc[~sub['bad_flat'], ev].mean()
        out(f'- {ev}: bad-day rate {br:.2f}, good-day rate {gr:.2f}')
    out('')
    # weekday rate
    out('### Bad-day rate by weekday')
    out('')
    wk = tbl.groupby('weekday')['bad_flat'].agg(['mean', 'count'])
    order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    for w in order:
        if w in wk.index:
            out(f'- {w}: bad rate {wk.loc[w, "mean"]:.2f} '
                f'(n={int(wk.loc[w, "count"])})')
    out('')

    # top correlations summary
    top = fd.head(5)
    out('### Eyeball summary')
    out('')
    out('Strongest separators (by |standardized gap|):')
    out('')
    for _, r in top.iterrows():
        direction = 'higher' if r['std_gap'] > 0 else 'lower'
        out(f'- **{r["feature"]}** is {direction} on bad days '
            f'(bad {r["bad_mean"]:.4g} vs good {r["good_mean"]:.4g}, '
            f'corr {r["corr_bad"]:+.2f}).')
    out('')
    out('Caveats for the ANALYZE phase:')
    out('')
    out('- n is small (51 days, ~14 bad). Single-feature correlations here '
        'are noisy -- treat as hypotheses, not findings.')
    out('- Several leg-structure features (leg_pnl_std, worst_leg_usd, '
        'frac_winner_legs) are partly mechanical consequences of a bad day, '
        'not causes -- do NOT use them as a same-day selector signal. '
        'Cross-day and pre-open features (overnight gap, prior-day range, '
        'VIX, day-of-week, event flags) are the legitimate ex-ante levers.')
    out('- The 11 Sunday sessions confound weekday analysis: they are short '
        '(low n_legs) and lack cross-day features. Check whether bad days '
        'concentrate in short low-leg-count sessions vs full sessions.')
    out('')

    OUT_MD.write_text('\n'.join(L), encoding='utf-8')
    print(f'Wrote {OUT_CSV}')
    print(f'Wrote {OUT_MD}')
    print()
    print(f'Bad days: {n_bad}/{n}  total lost {fmt_money(total_lost)}')
    print(f'Stack delta on bad days: {fmt_money(bad_delta.mean())}/day '
          f'CI [{fmt_money(bd_lo)}, {fmt_money(bd_hi)}]')


if __name__ == '__main__':
    main()
