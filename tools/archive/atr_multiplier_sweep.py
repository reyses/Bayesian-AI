"""ATR-multiplier sweep — FLAT 1-contract zigzag forward pass across ATR
multipliers, IS + OOS, stratified by volatility regime.

!!! WARNING (2026-05-21): the FLAT pass output is VALID ONLY at a fixed ATR
multiplier. It is NOT valid for cross-ATR comparison — the offline zigzag is a
hindsight-clean partition, so FLAT $/day is monotonic in pivot SUBDIVISION, not
tradeability (the sweep is monotonic with no interior peak; X=1 prints an
impossible $3,480/day OOS at 100% winning days). To compare multipliers, use a
CAUSAL streaming forward pass. See memory feedback_flat_pipeline_cross_param.md.

QUESTION (user, 2026-05-21): the zigzag's reversal threshold is ATR(14) x a
fixed multiplier (production = 4.0). Watching the live SIM, the user thinks 4
is too coarse and a lower multiplier (~2-3) may track price better. This sweep
produces the pick-and-choose table: per multiplier, the FLAT forward pass
$/day (IS and OOS) with leg-count / duration / amplitude / Trade-WR, and the
same broken out by volatility regime. The vol-stratified table answers whether
the optimal multiplier MOVES with regime (-> a per-day dynamic ATR selector is
worth building) or is flat (-> just pick one static number).

FLAT = enter every zigzag leg at its R-trigger, exit at the next pivot's
R-trigger, 1 contract, $6/leg friction, NO ML filter.

METHOD — reuses the production pipeline exactly, so the X=4 column reproduces
the published baseline ($690.10/day IS, $454.02/day OOS):
  Stage A  tools.build_zigzag_pivot_dataset.detect_day_pivots(day, X) -> pivots
           marked onto the production V2-grid (the `timestamp` column of the
           existing zigzag_pivot_dataset_*_atr4.parquet truth files).
  Stage B  tools.build_is_hardened_legs.process_day, with the module ATR_MULT
           patched to X -> R-trigger entries/exits, legs, realized P&L.
A SELF-TEST at X=4 checks the harness reproduces the baseline before the sweep
table is trusted; a failure is flagged loudly in the report.

Run:  python -m tools.atr_multiplier_sweep                 (full grid)
      python -m tools.atr_multiplier_sweep --self-test-only (X=4 check, ~1 min)

Output: reports/findings/regret_oracle/2026-05-21_atr_multiplier_sweep.txt
        reports/findings/regret_oracle/atr_sweep/  (per-ATR leg CSVs + summary)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import tools.build_zigzag_pivot_dataset as Z
import tools.build_is_hardened_legs as H
from tools.build_is_hardened_legs import DOLLAR_PER_POINT

# ATR multiplier grid — fine (0.25) resolution through 1-4 where the user's
# prior sits, integer steps 5-10 above. 4.0 is included as the baseline check.
MULTIPLIERS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5,
               3.75, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
BASELINE_MULT = 4.0
SELF_TEST_TOL = 1.0          # $/day; X=4 must reproduce baseline within this
N_BOOTSTRAP = 4000
BOOTSTRAP_SEED = 42
DAY_PNL_BIN = 25.0           # $ — mode histogram bin for $/day (CLAUDE.md)
VOL_ANNUALIZE = np.sqrt(390.0)   # ~minutes in an RTH session; vol proxy scale

OUT_DIR = Path('reports/findings/regret_oracle')
SWEEP_DIR = OUT_DIR / 'atr_sweep'
REPORT = OUT_DIR / '2026-05-21_atr_multiplier_sweep.txt'

# (target, truth parquet = the production V2-grid, raw-bars root, X=4 baseline)
TARGETS = [
    ('IS',  OUT_DIR / 'zigzag_pivot_dataset_IS_atr4.parquet',     'DATA/ATLAS',     690.10),
    ('OOS', OUT_DIR / 'zigzag_pivot_dataset_NT8_OOS_atr4.parquet', 'DATA/ATLAS_NT8', 454.02),
]


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_delta_ci(a, b, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Paired CI of mean(b) - mean(a); a and b indexed by the same days."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(a) != len(b):
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boots = (b[idx].mean(axis=1) - a[idx].mean(axis=1))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def mode_hist(values, bin_width):
    """Histogram-mode: center of the most populated bin."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float('nan')
    lo = np.floor(values.min() / bin_width) * bin_width
    hi = np.ceil(values.max() / bin_width) * bin_width + bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    if len(edges) < 2:
        return float(values.mean())
    counts, edges = np.histogram(values, bins=edges)
    k = int(np.argmax(counts))
    return float((edges[k] + edges[k + 1]) / 2.0)


def trade_wr_pf(pnl):
    """PF-based Trade WR = sum(winners) / |sum(losers)| - 1   (CLAUDE.md)."""
    pnl = np.asarray(pnl, dtype=np.float64)
    win = pnl[pnl > 0].sum()
    loss = pnl[pnl < 0].sum()
    if loss == 0:
        return float('nan')
    return float(win / abs(loss) - 1.0)


def intraday_vol(bars1m: pd.DataFrame) -> float:
    """Realized-vol proxy: std of 1m log-returns x sqrt(390). Matches the
    vol-regime feature used by tools/train_b10_vol_regime_sizer.py."""
    c = bars1m['close'].values.astype(np.float64)
    if len(c) < 3:
        return float('nan')
    lr = np.diff(np.log(c))
    lr = lr[np.isfinite(lr)]
    if len(lr) < 2:
        return float('nan')
    return float(np.std(lr) * VOL_ANNUALIZE)


def build_truth_for_X(grid_ts: np.ndarray, day: str, X: float):
    """Stage A: zigzag pivots for multiplier X, marked onto the production
    V2-grid timestamps for `day`. Replicates build_zigzag_pivot_dataset's
    is_pivot marking exactly. Returns a truth-day DataFrame, or None if the
    raw bars for the day are missing."""
    info = Z.detect_day_pivots(day, X)
    if info is None:
        return None
    g = pd.DataFrame({'day': day, 'timestamp': grid_ts.astype(np.int64)})
    g['is_pivot'] = 0
    g['pivot_dir'] = ''
    g['pivot_price'] = 0.0
    for piv_ts, piv_px, piv_dir in sorted(info['pivots'], key=lambda p: p[0]):
        m = (g['timestamp'] >= piv_ts - 60) & (g['timestamp'] < piv_ts + 60)
        if m.any():
            g.loc[m, 'is_pivot'] = 1
            g.loc[m, 'pivot_dir'] = piv_dir
            g.loc[m, 'pivot_price'] = piv_px
    return g


def run_target(target, truth_path, root, multipliers):
    """Returns (leg_tables: {X -> legs DataFrame}, vol: {day -> float})."""
    Z.RAW_1M_DIR = Path(root) / '1m'
    Z.RAW_5S_DIR = Path(root) / '5s'
    bars1m_dir = Path(root) / '1m'
    bars5s_dir = Path(root) / '5s'

    grid_df = pd.read_parquet(truth_path, columns=['day', 'timestamp'])
    grid_by_day = {d: g['timestamp'].values.astype(np.int64)
                   for d, g in grid_df.groupby('day')}
    days = sorted(grid_by_day)

    vol = {}
    for d in days:
        p = bars1m_dir / f'{d}.parquet'
        vol[d] = (intraday_vol(pd.read_parquet(p, columns=['close']))
                  if p.exists() else float('nan'))

    leg_tables = {}
    for X in tqdm(multipliers, desc=f'{target} ATR sweep'):
        H.ATR_MULT = X
        truth_parts = []
        for d in days:
            g = build_truth_for_X(grid_by_day[d], d, X)
            if g is not None:
                truth_parts.append(g)
        truth_X = pd.concat(truth_parts, ignore_index=True)
        rows = []
        for d in days:
            rows.extend(H.process_day(d, truth_X, bars1m_dir, bars5s_dir))
        leg_tables[X] = pd.DataFrame(rows)
    return leg_tables, vol


def summarize(legs: pd.DataFrame) -> dict:
    """Per-X aggregate stats over a leg set."""
    o = {'n_legs': len(legs)}
    if len(legs) == 0:
        o.update(n_days=0, legs_per_day=float('nan'), day_pnl=np.array([]),
                 mean=float('nan'), median=float('nan'), mode=float('nan'),
                 ci=(float('nan'), float('nan')), day_wr=float('nan'),
                 trade_wr=float('nan'), leg_min=float('nan'),
                 leg_amp=float('nan'))
        return o
    pd_ = legs.groupby('day')['pnl_usd'].sum()
    o['n_days'] = len(pd_)
    o['legs_per_day'] = len(legs) / len(pd_)
    o['day_pnl'] = pd_.values
    o['mean'] = float(pd_.mean())
    o['median'] = float(pd_.median())
    o['mode'] = mode_hist(pd_.values, DAY_PNL_BIN)
    o['ci'] = bootstrap_ci(pd_.values)
    o['day_wr'] = float((pd_.values > 0).mean())
    o['trade_wr'] = trade_wr_pf(legs['pnl_usd'].values)
    dur = (legs['exit_ts'] - legs['entry_ts']).values / 60.0
    amp = (legs['exit_price'] - legs['entry_price']).abs().values * DOLLAR_PER_POINT
    o['leg_min'] = float(np.median(dur))
    o['leg_amp'] = float(np.median(amp))
    o['day_pnl_by_day'] = pd_          # Series indexed by day
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test-only', action='store_true',
                    help='Run only X=4 and check the baseline; skip the sweep.')
    ap.add_argument('--multipliers', default=None,
                    help='Comma-separated override of the multiplier grid.')
    args = ap.parse_args()

    if args.self_test_only:
        grid = [BASELINE_MULT]
    elif args.multipliers:
        grid = [float(x) for x in args.multipliers.split(',')]
    else:
        grid = MULTIPLIERS

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    store = {}     # target -> (leg_tables, vol)
    for target, truth_path, root, _base in TARGETS:
        leg_tables, vol = run_target(target, truth_path, root, grid)
        store[target] = (leg_tables, vol)
        for X, legs in leg_tables.items():
            legs.to_csv(SWEEP_DIR / f'legs_{target.lower()}_atr{X:g}.csv',
                        index=False)

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    # ---- self-test ---------------------------------------------------------
    self_test_ok = True
    st = []
    for target, _tp, _r, base in TARGETS:
        if BASELINE_MULT not in store[target][0]:
            continue
        got = summarize(store[target][0][BASELINE_MULT])['mean']
        ok = abs(got - base) <= SELF_TEST_TOL
        self_test_ok &= ok
        st.append((target, got, base, ok))

    out('=' * 80)
    out('ATR-MULTIPLIER SWEEP  --  FLAT 1-contract zigzag forward pass')
    out('=' * 80)
    out(f'Grid: {", ".join(f"{x:g}" for x in grid)}')
    out('FLAT = enter every leg at its R-trigger, exit at next pivot, 1 '
        'contract,')
    out('       $6/leg friction, no ML filter.  $/day net of friction.')
    out('')
    out('SELF-TEST  (X=4 must reproduce the published baseline)')
    for target, got, base, ok in st:
        out(f'  {target:<4} X=4: computed ${got:8.2f}/day   baseline '
            f'${base:8.2f}   [{"PASS" if ok else "FAIL"}]')
    if not st:
        out('  (X=4 not in grid — self-test skipped)')
    elif not self_test_ok:
        out('  *** SELF-TEST FAILED — the harness does not reproduce the '
            'baseline.')
        out('  *** The sweep numbers below are NOT trustworthy. Investigate.')
    out('')

    if args.self_test_only:
        REPORT.write_text('\n'.join(lines), encoding='utf-8')
        print(f'\nWrote: {REPORT}')
        return

    # ---- vol terciles from IS ---------------------------------------------
    is_vol = np.array([v for v in store['IS'][1].values() if np.isfinite(v)])
    q33, q67 = np.percentile(is_vol, [100 / 3.0, 200 / 3.0])

    def vol_bucket(v):
        if not np.isfinite(v):
            return None
        return 'low' if v < q33 else ('high' if v >= q67 else 'mid')

    # ---- per-X summaries ---------------------------------------------------
    summ = {t: {X: summarize(store[t][0][X]) for X in grid} for t in store}

    out('-' * 80)
    out('MAIN TABLE  ($/day net of friction)')
    out('-' * 80)
    out(f'{"ATR":>5}  {"IS $/day":>9} {"[95% CI]":>20}   '
        f'{"OOS $/day":>9} {"[95% CI]":>20}   {"leg/day":>7}  '
        f'{"legmin":>6}  {"legamp$":>7}  {"OOS DayWR":>9}  {"OOS TWR":>8}')
    for X in grid:
        si, so = summ['IS'][X], summ['OOS'][X]
        tag = '  <- baseline' if X == BASELINE_MULT else ''
        out(f'{X:>5g}  {si["mean"]:>9.0f} '
            f'[{si["ci"][0]:>8.0f},{si["ci"][1]:>8.0f}]   '
            f'{so["mean"]:>9.0f} '
            f'[{so["ci"][0]:>8.0f},{so["ci"][1]:>8.0f}]   '
            f'{so["legs_per_day"]:>7.1f}  {so["leg_min"]:>6.1f}  '
            f'{so["leg_amp"]:>7.0f}  {so["day_wr"]*100:>8.1f}%  '
            f'{so["trade_wr"]:>+8.2f}{tag}')
    out('')
    out('  legmin = median leg duration (min);  legamp$ = median leg '
        'amplitude (gross $);')
    out('  OOS TWR = profit-factor Trade WR  (0 = break-even, +1 = PF 2).')
    out('')

    # ---- vol-regime stratification ----------------------------------------
    buckets = ['low', 'mid', 'high']
    day_bucket = {t: {d: vol_bucket(v) for d, v in store[t][1].items()}
                  for t in store}

    def strat(target, X):
        s = summ[target][X]
        if s['n_legs'] == 0:
            return {b: (float('nan'), 0) for b in buckets}
        dpb = s['day_pnl_by_day']
        res = {}
        for b in buckets:
            days_b = [d for d in dpb.index if day_bucket[target].get(d) == b]
            vals = dpb.loc[days_b].values if days_b else np.array([])
            res[b] = (float(vals.mean()) if len(vals) else float('nan'),
                      len(vals))
        return res

    for target in ('OOS', 'IS'):
        nb = {b: sum(1 for d in store[target][1]
                     if day_bucket[target].get(d) == b) for b in buckets}
        out('-' * 80)
        out(f'{target} VOL-REGIME STRATIFICATION  ($/day by ATR x vol tercile)')
        out(f'  vol proxy = std(1m log-returns) x sqrt(390); IS terciles '
            f'q33={q33:.4f} q67={q67:.4f}')
        out(f'  bucket day counts: low={nb["low"]}  mid={nb["mid"]}  '
            f'high={nb["high"]}')
        out('-' * 80)
        out(f'{"ATR":>5}  {"low-vol":>10}  {"mid-vol":>10}  {"high-vol":>10}  '
            f'{"argmax":>8}')
        col_best = {b: (None, -1e18) for b in buckets}
        rows_s = {}
        for X in grid:
            s = strat(target, X)
            rows_s[X] = s
            for b in buckets:
                if np.isfinite(s[b][0]) and s[b][0] > col_best[b][1]:
                    col_best[b] = (X, s[b][0])
        for X in grid:
            s = rows_s[X]
            vals = [s[b][0] for b in buckets]
            amax = buckets[int(np.nanargmax(vals))] if any(
                np.isfinite(v) for v in vals) else '-'
            out(f'{X:>5g}  {s["low"][0]:>10.0f}  {s["mid"][0]:>10.0f}  '
                f'{s["high"][0]:>10.0f}  {amax:>8}')
        out(f'  best ATR per bucket:  low=x{col_best["low"][0]:g}  '
            f'mid=x{col_best["mid"][0]:g}  high=x{col_best["high"][0]:g}')
        out('')

    # ---- verdict -----------------------------------------------------------
    # Pick the best static ATR on IS; confirm on OOS. Then the regime-optimal
    # policy: best ATR per vol bucket chosen on IS, evaluated on OOS.
    is_means = {X: summ['IS'][X]['mean'] for X in grid}
    best_static = max(is_means, key=is_means.get)

    is_strat = {X: strat('IS', X) for X in grid}
    best_X_bucket = {}
    for b in buckets:
        cand = {X: is_strat[X][b][0] for X in grid
                if np.isfinite(is_strat[X][b][0])}
        best_X_bucket[b] = max(cand, key=cand.get) if cand else best_static

    # OOS daily P&L under each policy, aligned by day
    oos_days = sorted(store['OOS'][1])
    static_daily, dynamic_daily = [], []
    for d in oos_days:
        b = day_bucket['OOS'].get(d)
        s_series = summ['OOS'][best_static]['day_pnl_by_day'] \
            if summ['OOS'][best_static]['n_legs'] else pd.Series(dtype=float)
        sv = float(s_series.get(d, 0.0))
        xb = best_X_bucket.get(b, best_static)
        d_series = summ['OOS'][xb]['day_pnl_by_day'] \
            if summ['OOS'][xb]['n_legs'] else pd.Series(dtype=float)
        dv = float(d_series.get(d, 0.0))
        static_daily.append(sv)
        dynamic_daily.append(dv)
    static_daily = np.array(static_daily)
    dynamic_daily = np.array(dynamic_daily)
    dyn_lo, dyn_hi = bootstrap_delta_ci(static_daily, dynamic_daily)
    delta = dynamic_daily.mean() - static_daily.mean()

    out('=' * 80)
    out('VERDICT')
    out('=' * 80)
    out('*** WARNING: FLAT $/day is NOT valid for cross-ATR comparison. The')
    out('*** offline zigzag is a hindsight-clean partition; FLAT $/day scales')
    out('*** with pivot SUBDIVISION, not tradeability (monotonic, no peak).')
    out('*** Use a forward pass streaming forward pass to compare multipliers.')
    out('')
    bs_oos = summ['OOS'][best_static]
    out(f'Best STATIC ATR (chosen on IS $/day): x{best_static:g}')
    out(f'  IS  ${summ["IS"][best_static]["mean"]:.0f}/day   '
        f'OOS ${bs_oos["mean"]:.0f}/day '
        f'[{bs_oos["ci"][0]:.0f}, {bs_oos["ci"][1]:.0f}]   '
        f'OOS mode ${bs_oos["mode"]:.0f}  median ${bs_oos["median"]:.0f}')
    out(f'  Baseline x4:  IS ${summ["IS"][BASELINE_MULT]["mean"]:.0f}   '
        f'OOS ${summ["OOS"][BASELINE_MULT]["mean"]:.0f}/day'
        if BASELINE_MULT in grid else '  (x4 not in grid)')
    out('')
    out(f'Regime-optimal policy (best ATR per IS vol bucket): '
        f'low=x{best_X_bucket["low"]:g}  mid=x{best_X_bucket["mid"]:g}  '
        f'high=x{best_X_bucket["high"]:g}')
    out(f'  OOS regime-optimal ${dynamic_daily.mean():.0f}/day  vs  '
        f'best-static ${static_daily.mean():.0f}/day')
    out(f'  delta ${delta:+.0f}/day   95% CI [{dyn_lo:+.0f}, {dyn_hi:+.0f}]'
        f'   {"SIGNIFICANT" if (dyn_lo > 0 or dyn_hi < 0) else "NOT significant"}')
    out('')
    same_bucket_choice = len(set(best_X_bucket.values())) == 1
    if same_bucket_choice or not (dyn_lo > 0):
        out('-> The optimal multiplier does NOT move meaningfully with regime')
        out('   (or the dynamic gain is not significant). A single STATIC ATR')
        out(f'   multiplier of x{best_static:g} is the recommendation. Building')
        out('   a per-day dynamic ATR selector is NOT justified by this data.')
    else:
        out('-> The optimal multiplier MOVES with vol regime and the OOS gain')
        out('   over the best static choice is significant. A per-day ATR')
        out('   selector (keyed on a session-start vol-regime call, e.g. B10)')
        out('   is justified — proceed to Stage 2 (per-ATR B-stack bundles).')

    REPORT.write_text('\n'.join(lines), encoding='utf-8')

    summary_rows = []
    for X in grid:
        si, so = summ['IS'][X], summ['OOS'][X]
        summary_rows.append({
            'atr_mult': X,
            'is_dollar_day': si['mean'], 'is_ci_lo': si['ci'][0],
            'is_ci_hi': si['ci'][1],
            'oos_dollar_day': so['mean'], 'oos_ci_lo': so['ci'][0],
            'oos_ci_hi': so['ci'][1], 'oos_median': so['median'],
            'oos_mode': so['mode'], 'oos_legs_per_day': so['legs_per_day'],
            'oos_leg_min': so['leg_min'], 'oos_leg_amp_usd': so['leg_amp'],
            'oos_day_wr': so['day_wr'], 'oos_trade_wr': so['trade_wr'],
        })
    pd.DataFrame(summary_rows).to_csv(SWEEP_DIR / 'sweep_summary.csv',
                                      index=False)
    print(f'\nWrote: {REPORT}')
    print(f'Wrote: {SWEEP_DIR / "sweep_summary.csv"}  + per-ATR leg CSVs')


if __name__ == '__main__':
    main()
