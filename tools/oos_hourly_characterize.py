"""MEASURE (hour-by-hour granularity) — decompose the FLAT zigzag $/day into
hour-of-day "shifts".

DMAIC MEASURE, hourly cut. User 2026-05-21: "add the granularity to hour by
hour (like a manufacturing site)." Each ET hour-of-day is treated as a
production shift; the product is $/day. This script decomposes the FLAT
hardened-leg $/day into per-hour-of-day contributions, finds which shifts
make money and which bleed, and previews a skip-the-defective-shift rule with
the mandatory IS-discover / OOS-confirm protocol.

FLAT substrate only: per-leg P&L is read directly from the hardened-leg CSVs.
Each leg is attributed to the ET hour-of-day of its ENTRY timestamp (a leg
that spans an hour boundary is counted wholly in its entry hour). A per-leg
STACK (B7/B9/B10) output is not available leg-level — stack-hourly is deferred.

Decomposition identity:  $/day  =  sum over h of contribution(h),
where contribution(h) = (total FLAT pnl from legs entered in ET hour h) / n_days.

Output: reports/findings/oos_bad_days/2026-05-21_hourly.md
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
IS_LEGS = REPO / 'reports/findings/regret_oracle/is_hardened_legs.csv'
OOS_LEGS = REPO / 'reports/findings/regret_oracle/oos_hardened_legs_full.csv'
OUT_MD = REPO / 'reports/findings/oos_bad_days/2026-05-21_hourly.md'
TZ = 'America/New_York'
N_BOOT = 4000
BOOT_SEED = 42


def bootstrap_ci(values, n_boot=N_BOOT, seed=BOOT_SEED):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_legs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = pd.to_datetime(df['entry_ts'], unit='s', utc=True).dt.tz_convert(TZ)
    df['et_hour'] = dt.dt.hour.values
    return df


def hour_decomp(df: pd.DataFrame):
    """Per ET hour-of-day: $/day contribution (per-day array WITH zeros for
    days that did not trade the hour), CI, totals, defect rate."""
    days = sorted(df['day'].unique())
    n_days = len(days)
    piv_sum = df.pivot_table(index='day', columns='et_hour', values='pnl_usd',
                             aggfunc='sum')
    piv_cnt = df.pivot_table(index='day', columns='et_hour', values='pnl_usd',
                             aggfunc='count')
    out = {}
    for h in range(24):
        if h in piv_sum.columns:
            col = piv_sum[h].reindex(days)
            contrib = col.fillna(0.0).values            # 0 = day didn't trade h
            active = int(col.notna().sum())
            neg_active = int((col < 0).sum())            # NaN < 0 -> False
            n_legs = int(piv_cnt[h].sum())
        else:
            contrib = np.zeros(n_days)
            active = neg_active = n_legs = 0
        lo, hi = bootstrap_ci(contrib)
        out[h] = dict(
            contribution=float(contrib.mean()),
            ci=(lo, hi),
            total=float(contrib.sum()),
            n_legs=n_legs,
            dollar_per_leg=(float(contrib.sum() / n_legs) if n_legs else 0.0),
            active_days=active,
            bad_bucket_rate=((neg_active / active) if active else float('nan')),
            per_day=contrib,
        )
    return out, n_days, days


def fmt(x):
    if not np.isfinite(x):
        return '   n/a'
    return f'${x:+,.0f}'


def hour_table(decomp, n_days, label, lines):
    lines.append(f'### {label} — {n_days} days')
    lines.append('')
    lines.append('| ET hr | $/day contrib | 95% CI | total $ | legs | $/leg '
                 '| act.days | defect% |')
    lines.append('|--:|--:|--:|--:|--:|--:|--:|--:|')
    tot = 0.0
    for h in range(24):
        d = decomp[h]
        tot += d['contribution']
        br = d['bad_bucket_rate']
        brs = f'{br*100:.0f}%' if np.isfinite(br) else '--'
        dpl = fmt(d['dollar_per_leg']) if d['n_legs'] else '--'
        lines.append(
            f'| {h:02d} | {fmt(d["contribution"])} | '
            f'[{fmt(d["ci"][0])}, {fmt(d["ci"][1])}] | {fmt(d["total"])} | '
            f'{d["n_legs"]} | {dpl} | {d["active_days"]} | {brs} |')
    lines.append(f'| **sum** | **{fmt(tot)}** | | | | | | |')
    lines.append('')


def main():
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    is_df = load_legs(IS_LEGS)
    oos_df = load_legs(OOS_LEGS)
    is_dec, is_nd, _ = hour_decomp(is_df)
    oos_dec, oos_nd, oos_days = hour_decomp(oos_df)

    L = []
    def out(s=''):
        L.append(s)

    out('# Hour-of-Day Decomposition — FLAT zigzag $/day (MEASURE, hourly)')
    out('')
    out('DMAIC MEASURE, hour-by-hour granularity. User 2026-05-21: "add the '
        'granularity to hour by hour (like a manufacturing site)." Each ET '
        'hour-of-day = a production shift; the product is $/day. FLAT '
        'substrate (per-leg P&L from the hardened-leg CSVs); each leg '
        'attributed to the ET hour of its entry. `$/day = sum of the '
        'contribution column`. `defect%` = share of (day,hour) buckets that '
        'were net-negative, among days that traded that hour.')
    out('')
    out('## 1. Hour-of-day P&L decomposition')
    out('')
    hour_table(is_dec, is_nd, 'IS (2025)', L)
    hour_table(oos_dec, oos_nd, 'OOS (2026-03-19..05-18, sealed)', L)

    # ---- 2. defective shifts ----
    is_neg = [h for h in range(24) if is_dec[h]['contribution'] < 0
              and is_dec[h]['n_legs'] > 0]
    oos_neg = [h for h in range(24) if oos_dec[h]['contribution'] < 0
               and oos_dec[h]['n_legs'] > 0]
    robust = sorted(set(is_neg) & set(oos_neg))
    is_strict = [h for h in is_neg if is_dec[h]['ci'][1] < 0]   # CI fully < 0
    out('## 2. Defective shifts (negative-contribution hours)')
    out('')
    out(f'- IS-negative hours: {is_neg if is_neg else "none"}')
    out(f'- IS-negative with 95% CI fully below 0 (confidently bad): '
        f'{is_strict if is_strict else "none"}')
    out(f'- OOS-negative hours: {oos_neg if oos_neg else "none"}')
    out(f'- **ROBUST defective shifts (negative in BOTH IS and OOS): '
        f'{robust if robust else "none"}** — these are the morning lever '
        f'candidates.')
    out('')

    # ---- 3. skip-the-defective-shifts preview (IS-select / OOS-confirm) ----
    out('## 3. Skip-defective-shifts preview (IS-discover / OOS-confirm)')
    out('')
    out('Honest protocol: pick the hours to flatten on IS, then evaluate the '
        'untouched OOS. "Skip hour h" = take no entries in ET hour h (legs '
        'already open are unaffected — entry-time filter).')
    out('')
    for tag, S in [('IS contribution < 0', is_neg),
                   ('IS contribution CI fully < 0', is_strict)]:
        if not S:
            out(f'- Selector "{tag}": empty set — nothing to skip.')
            continue
        # per-day OOS delta = -(pnl entered in S hours that day)
        delta_per_day = -np.sum([oos_dec[h]['per_day'] for h in S], axis=0)
        full = sum(oos_dec[h]['contribution'] for h in range(24))
        skipped = full + delta_per_day.mean()
        lo, hi = bootstrap_ci(delta_per_day)
        sig = 'SIGNIFICANT' if (lo > 0 or hi < 0) else 'not significant'
        out(f'- Selector "{tag}" -> skip hours {S}:')
        out(f'    OOS $/day {fmt(full)} -> {fmt(skipped)}  '
            f'(delta {fmt(delta_per_day.mean())}/day, 95% CI '
            f'[{fmt(lo)}, {fmt(hi)}], {sig}).')
    out('')

    # ---- 4. bad-day loss localization ----
    out('## 4. Bad-day loss localization')
    out('')
    day_pnl = oos_df.groupby('day')['pnl_usd'].sum()
    bad_days = day_pnl[day_pnl < 0].index.tolist()
    out(f'{len(bad_days)} negative OOS days. For each, how concentrated is the '
        'loss across hours (is a bad day one bad hour, or bad all day)?')
    out('')
    worst1_frac, worst2_frac = [], []
    rows = []
    for d in bad_days:
        sub = oos_df[oos_df['day'] == d]
        hp = sub.groupby('et_hour')['pnl_usd'].sum().sort_values()
        tot = hp.sum()
        w1 = hp.iloc[0]
        w2 = hp.iloc[:2].sum()
        worst1_frac.append(w1 / tot)
        worst2_frac.append(w2 / tot)
        rows.append((d, tot, int(hp.index[0]), w1, int(hp.index[1])
                     if len(hp) > 1 else -1))
    out(f'- Median fraction of a bad day\'s loss in its single worst hour: '
        f'{np.median(worst1_frac)*100:.0f}%  '
        f'(worst 2 hours: {np.median(worst2_frac)*100:.0f}%).')
    out('')
    out('| bad day | day $ | worst hr | worst-hr $ | 2nd-worst hr |')
    out('|---|--:|--:|--:|--:|')
    for d, tot, wh, w1, wh2 in sorted(rows, key=lambda r: r[1]):
        wh2s = f'{wh2:02d}' if wh2 >= 0 else '--'
        out(f'| {d} | {fmt(tot)} | {wh:02d} | {fmt(w1)} | {wh2s} |')
    out('')
    # which hours host the bad-day damage
    bad_hour_loss = {}
    for d in bad_days:
        sub = oos_df[oos_df['day'] == d]
        for h, s in sub.groupby('et_hour')['pnl_usd'].sum().items():
            bad_hour_loss[h] = bad_hour_loss.get(h, 0.0) + s
    out('Total bad-day P&L by ET hour (where the 14 bad days actually bled):')
    out('')
    out('| ET hr | bad-day total $ |')
    out('|--:|--:|')
    for h in sorted(bad_hour_loss, key=lambda k: bad_hour_loss[k]):
        out(f'| {h:02d} | {fmt(bad_hour_loss[h])} |')
    out('')

    out('## 5. Read / next (ANALYZE)')
    out('')
    if robust:
        out(f'- Robust defective shifts exist ({robust}). ANALYZE: confirm the '
            'IS->OOS stability and whether the skip-preview delta above holds '
            'up; if so, an hour-of-day entry filter is the IMPROVE candidate.')
    else:
        out('- No hour-of-day is negative in BOTH IS and OOS — the bleed is '
            'not cleanly localized to a fixed clock hour. ANALYZE: look at '
            'hour-WITHIN-session or session-type rather than absolute ET hour.')
    out('- Stack-hourly is deferred (needs a per-leg B7/B9/B10 output); the '
        'day-level MEASURE already showed the stack shaves bad days +$175/day.')

    OUT_MD.write_text('\n'.join(L), encoding='utf-8')
    print(f'Wrote {OUT_MD}')
    print()
    print(f'IS  $/day sum-of-hours = '
          f'${sum(is_dec[h]["contribution"] for h in range(24)):,.2f}')
    print(f'OOS $/day sum-of-hours = '
          f'${sum(oos_dec[h]["contribution"] for h in range(24)):,.2f}')
    print(f'Robust defective shifts (neg in IS and OOS): {robust}')


if __name__ == '__main__':
    main()
