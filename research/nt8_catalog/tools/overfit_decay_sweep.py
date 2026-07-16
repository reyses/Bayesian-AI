"""
OVERFIT-DECAY SHELF-LIFE SWEEP — doc 075 standard.

Standing test: fit the pooled combiner (same design as combiner_preview.py) on a short
training window, then measure how many weeks pass before its out-of-window edge decays
below 70% of its initial edge. Sets the retune cadence before anything feeds the RL
engine. Mechanical sweep only — no hyperparameter tuning, no feature-list changes.

Method:
  - Load the pooled fires exactly like combiner_preview.load_pool() (BASE + consensus +
    per-stream one-hots, det list computed GLOBALLY over the full pool so train/eval
    slices share identical columns, mirroring combiner_preview.fit_report).
  - Training windows: 8 calendar weeks (and, for comparison, 16 calendar weeks), window
    START stepping every 4 weeks from 2024-01-01 through 2025-12-31 (both are Mondays;
    all steps land on Mondays => training/eval windows are exact ISO-week multiples).
    Skip windows with < 5,000 training fires.
  - Fit LogisticRegression(max_iter=1000) on the window, standardized by the window's own
    train mean/std (no leakage from eval weeks).
  - Evaluate on each subsequent ISO week (Mon-Sun; grouped by the Monday of the week —
    numerically identical to grouping by (iso_year, iso_week)). Skip eval weeks with
    < 500 fires (counted separately from weeks skipped for being single-class, where AUC
    is undefined).
  - initial_edge = mean edge (AUC-0.5) of the first 2 QUALIFYING eval weeks (temporal
    order, skipped weeks excluded from the count/index).
  - shelf_life_weeks = 1-based index of the first qualifying eval week where a 3-week
    rolling mean of edge_w first drops below 0.70 * initial_edge. Requires initial_edge
    > 0 (otherwise decay is undefined against a non-positive baseline -> flagged
    'no_positive_edge'). If the rolling condition never fires within the available
    qualifying weeks -> right-censored at the horizon (= count of qualifying weeks).
    If horizon < 2 -> 'insufficient_eval' (can't even form initial_edge).

Runtime guard: if a window's LogisticRegression fit exceeds ~2 min, redo it on a fixed
100k-fire subsample (seed=42) and log/document it (expected to never trigger given
dataset size, but implemented per spec).

Reads:  reports/signal_rows_<det>.parquet (via combiner_preview.load_pool)
Writes: reports/overfit_decay.md         (report: two per-window tables + distribution
                                           summary + subsampling notes)
        reports/overfit_decay_run.log    (raw log of every window/week decision)
        reports/overfit_decay_rows.parquet (window_start, train_weeks, eval_week, N, auc)
"""
import os, sys, time, logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from combiner_preview import load_pool, BASE  # noqa: E402  (reuse combiner's exact pool/feature design)

REP = os.path.abspath(os.path.join(HERE, '..', 'reports'))
LOG_PATH = os.path.join(REP, 'overfit_decay_run.log')
ROWS_PATH = os.path.join(REP, 'overfit_decay_rows.parquet')
MD_PATH = os.path.join(REP, 'overfit_decay.md')

MIN_TRAIN_FIRES = 5000        # skip training windows below this many fires
MIN_WEEK_FIRES = 500          # skip eval weeks below this many fires
STEP_WEEKS = 4                # window START step (calendar weeks)
SWEEP_START = pd.Timestamp('2024-01-01')   # first window START (a Monday)
SWEEP_END = pd.Timestamp('2025-12-31')     # last allowed window START (inclusive)
DECAY_FRAC = 0.70             # shelf-life threshold: 70% of initial edge
ROLL_WEEKS = 3                # rolling window for the decay trigger
INIT_WEEKS = 2                # qualifying eval weeks used for initial_edge
FIT_TIME_BUDGET_S = 120.0     # runtime guard (~2 min) per window fit
SUBSAMPLE_N = 100_000
SUBSAMPLE_SEED = 42           # fixed seed for the runtime-guard subsample

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(message)s',
    handlers=[logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8'), logging.StreamHandler()])
log = logging.getLogger('overfit_decay')


def build_features(P):
    """Mirror combiner_preview.fit_report's column construction exactly, but keep every
    fire (no year-based train/test split — the sweep does its own windowing)."""
    P = P.dropna(subset=['y']).copy()
    P['date'] = pd.to_datetime(P['day'], format='%Y_%m_%d')
    # Monday of the ISO week each fire falls in -- grouping by this date is numerically
    # identical to grouping by (iso_year, iso_week), and side-steps iso-year boundary
    # display quirks (e.g. late-Dec dates reporting into iso_year+1).
    P['week_start'] = P['date'] - pd.to_timedelta(P['date'].dt.weekday, unit='D')
    dets = sorted(P['det'].unique())          # GLOBAL det list -> train/eval share columns
    for d in dets:
        P[f'is_{d}'] = (P['det'] == d).astype(int)
    cols = BASE + ['consensus'] + [f'is_{d}' for d in dets]
    return P, cols, dets


def make_starts():
    starts = []
    s = SWEEP_START
    while s <= SWEEP_END:
        starts.append(s)
        s = s + pd.Timedelta(weeks=STEP_WEEKS)
    return starts


def fit_window(Xtr, ytr, label, w_start):
    """Fit with the runtime guard: if the fit exceeds FIT_TIME_BUDGET_S, refit on a fixed
    100k-fire subsample and document it. Returns (clf, mu, sd, subsampled: bool, n_used)."""
    t0 = time.time()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=1000).fit((Xtr - mu) / sd, ytr)
    fit_time = time.time() - t0
    if fit_time <= FIT_TIME_BUDGET_S:
        return clf, mu, sd, False, len(ytr), fit_time
    log.info(f'[{label}] window {w_start.date()} fit took {fit_time:.1f}s > '
             f'{FIT_TIME_BUDGET_S:.0f}s budget -> SUBSAMPLING to {SUBSAMPLE_N} fires (seed={SUBSAMPLE_SEED})')
    rng = np.random.RandomState(SUBSAMPLE_SEED)
    idx = rng.choice(len(ytr), size=min(SUBSAMPLE_N, len(ytr)), replace=False)
    Xs, ys = Xtr[idx], ytr[idx]
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    t1 = time.time()
    clf = LogisticRegression(max_iter=1000).fit((Xs - mu) / sd, ys)
    log.info(f'[{label}] window {w_start.date()} SUBSAMPLED refit took {time.time() - t1:.1f}s')
    return clf, mu, sd, True, len(ys), fit_time


def shelf_life_from_edges(edge_list):
    """Returns (initial_edge_or_None, status, shelf_life_or_censor_horizon).
    status in {'ok', 'censored', 'insufficient_eval', 'no_positive_edge'}."""
    horizon = len(edge_list)
    if horizon < INIT_WEEKS:
        return None, 'insufficient_eval', horizon
    initial_edge = float(np.mean(edge_list[:INIT_WEEKS]))
    if initial_edge <= 0:
        return initial_edge, 'no_positive_edge', horizon
    threshold = DECAY_FRAC * initial_edge
    roll = pd.Series(edge_list).rolling(ROLL_WEEKS, min_periods=ROLL_WEEKS).mean()
    hit = roll[roll < threshold]
    if len(hit) > 0:
        shelf_life = int(hit.index[0]) + 1   # 1-based index
        return initial_edge, 'ok', shelf_life
    return initial_edge, 'censored', horizon


def sweep(P, cols, train_weeks, label, rows_out):
    starts = make_starts()
    week_starts_all = np.array(sorted(P['week_start'].unique()))
    results = []
    n_skipped_train = 0
    for w_start in tqdm(starts, desc=f'{label} ({train_weeks}wk) windows'):
        train_end = w_start + pd.Timedelta(weeks=train_weeks)
        train_mask = (P['date'] >= w_start) & (P['date'] < train_end)
        n_train = int(train_mask.sum())
        if n_train < MIN_TRAIN_FIRES:
            log.info(f'[{label}] window {w_start.date()} SKIPPED (train N={n_train} < {MIN_TRAIN_FIRES})')
            n_skipped_train += 1
            continue

        Xtr = P.loc[train_mask, cols].values.astype(float)
        ytr = P.loc[train_mask, 'y'].astype(int).values
        clf, mu, sd, subsampled, n_used, fit_time = fit_window(Xtr, ytr, label, w_start)

        eval_weeks = week_starts_all[week_starts_all >= np.datetime64(train_end)]
        edge_list = []
        skipped_low_n = 0
        skipped_one_class = 0
        for ew in eval_weeks:
            ew_ts = pd.Timestamp(ew)
            m = P['week_start'] == ew_ts
            n = int(m.sum())
            if n < MIN_WEEK_FIRES:
                skipped_low_n += 1
                continue
            y = P.loc[m, 'y'].astype(int).values
            if len(np.unique(y)) < 2:
                skipped_one_class += 1
                continue
            X = P.loc[m, cols].values.astype(float)
            p = clf.predict_proba((X - mu) / sd)[:, 1]
            auc = roc_auc_score(y, p)
            edge_list.append(auc - 0.5)
            rows_out.append({'window_start': w_start.date().isoformat(), 'train_weeks': train_weeks,
                              'eval_week': ew_ts.date().isoformat(), 'N': n, 'auc': auc})

        initial_edge, status, sl = shelf_life_from_edges(edge_list)
        log.info(f'[{label}] window {w_start.date()} N_train={n_train} (used={n_used}, '
                 f'subsampled={subsampled}, fit={fit_time:.2f}s) eval_qual={len(edge_list)} '
                 f'skip_lowN={skipped_low_n} skip_oneclass={skipped_one_class} '
                 f'initial_edge={initial_edge} status={status} shelf_life/horizon={sl}')
        results.append({
            'window_start': w_start.date().isoformat(), 'n_train': n_train,
            'subsampled': subsampled, 'n_qual_eval_weeks': len(edge_list),
            'skipped_low_n_weeks': skipped_low_n, 'skipped_one_class_weeks': skipped_one_class,
            'initial_edge': initial_edge, 'status': status, 'shelf_life_or_horizon': sl,
        })
    return results, n_skipped_train


def render_table(results):
    lines = ['| window start | N_train | initial_edge | shelf_life_weeks |',
             '|---|---|---|---|']
    for r in results:
        if r['status'] == 'ok':
            sl = f"{r['shelf_life_or_horizon']}"
        elif r['status'] == 'censored':
            sl = f"censored @ {r['shelf_life_or_horizon']}"
        elif r['status'] == 'insufficient_eval':
            sl = f"insufficient eval data (only {r['shelf_life_or_horizon']} qualifying wk)"
        else:  # no_positive_edge
            sl = f"no positive edge (censored @ {r['shelf_life_or_horizon']})"
        ie = f"{r['initial_edge']:.4f}" if r['initial_edge'] is not None else 'n/a'
        flag = ' *subsampled*' if r['subsampled'] else ''
        lines.append(f"| {r['window_start']} | {r['n_train']} | {ie} | {sl}{flag} |")
    return lines


def distribution_summary(results):
    ok = [r for r in results if r['status'] == 'ok']
    censored = [r for r in results if r['status'] in ('censored', 'no_positive_edge')]
    insuff = [r for r in results if r['status'] == 'insufficient_eval']
    total = len(results)
    lines = []
    if ok:
        vals = np.array([r['shelf_life_or_horizon'] for r in ok], dtype=float)
        mode_val = pd.Series(vals).mode().iloc[0]
        median_val = float(np.median(vals))
        lines.append(f"- Observed (uncensored) shelf-life: N={len(ok)} windows. "
                      f"MODE = {mode_val:.0f} weeks, MEDIAN = {median_val:.1f} weeks.")
    else:
        lines.append("- Observed (uncensored) shelf-life: N=0 windows (no window crossed the "
                      "70%-of-initial-edge threshold within its horizon) -> MODE/MEDIAN undefined.")
    lines.append(f"- Censoring: {len(censored)} of {total} windows never crossed the threshold "
                 f"within their available eval horizon (right-censored: true shelf-life >= the "
                 f"reported horizon for those windows) "
                 f"[{sum(1 for r in results if r['status']=='censored')} censored-with-positive-edge, "
                 f"{sum(1 for r in results if r['status']=='no_positive_edge')} had non-positive "
                 f"initial_edge (decay undefined)].")
    lines.append(f"- {len(insuff)} of {total} windows had insufficient eval data "
                 f"(< {INIT_WEEKS} qualifying eval weeks available, usually windows starting near "
                 f"the end of the sweep range where the data horizon runs out) — excluded from "
                 f"MODE/MEDIAN.")
    lines.append("- CAVEAT: MODE/MEDIAN above are computed over UNCENSORED windows only; since "
                 f"{len(censored)}/{total} windows are censored, the true population shelf-life is "
                 "likely LONGER than these numbers suggest (naive underestimate, no survival-curve "
                 "correction applied here per spec).")
    return lines


def main():
    log.info('Loading pool via combiner_preview.load_pool() ...')
    P = load_pool()
    P, cols, dets = build_features(P)
    log.info(f'Pool: {len(P)} fires, {len(dets)} streams, {P["date"].min().date()} .. '
             f'{P["date"].max().date()}, {len(cols)} feature columns')

    rows_out = []
    results_8, skipped_8 = sweep(P, cols, 8, '8wk', rows_out)
    results_16, skipped_16 = sweep(P, cols, 16, '16wk', rows_out)

    rows_df = pd.DataFrame(rows_out)
    rows_df.to_parquet(ROWS_PATH, index=False)
    log.info(f'wrote {ROWS_PATH} ({len(rows_df)} rows)')

    any_subsampled = any(r['subsampled'] for r in results_8 + results_16)

    lines = [
        '# Overfit-decay shelf-life sweep (doc 075 standard)', '',
        'Pooled combiner (same design as combiner_preview.py: BASE + consensus + per-stream '
        'one-hots, GLOBAL det list) fit on rolling training windows, evaluated weekly OOS. '
        'shelf_life_weeks = first qualifying eval week where a 3-week rolling mean of edge '
        '(AUC-0.5) drops below 70% of initial_edge (mean of first 2 qualifying eval weeks). '
        'Right-censored at the available horizon if it never crosses.', '',
        f'- Pool: {len(P)} fires, {len(dets)} streams, {P["date"].min().date()} .. {P["date"].max().date()}',
        f'- Feature columns ({len(cols)}): BASE={BASE} + consensus + is_<det> x {len(dets)} streams',
        f'- Window starts: every {STEP_WEEKS} weeks from {SWEEP_START.date()} through {SWEEP_END.date()}, '
        f'{len(make_starts())} candidate starts per training-length pass',
        f'- Skip rule: training windows with < {MIN_TRAIN_FIRES} fires skipped '
        f'(8wk pass: {skipped_8} skipped; 16wk pass: {skipped_16} skipped); eval weeks with '
        f'< {MIN_WEEK_FIRES} fires skipped (see per-window skip counts in the run log)',
        f'- Runtime guard: fit-time budget {FIT_TIME_BUDGET_S:.0f}s/window; '
        + ('TRIGGERED for at least one window -- see run log for which window(s) and their '
           f'subsampled N={SUBSAMPLE_N} (seed={SUBSAMPLE_SEED})' if any_subsampled
           else f'never triggered (all fits stayed under budget; no window was subsampled)'),
        '',
        '## Pass 1 — 8-week training windows', '',
    ]
    lines += render_table(results_8)
    lines += ['']
    lines += distribution_summary(results_8)
    lines += ['', '## Pass 2 — 16-week training windows (comparison)', '']
    lines += render_table(results_16)
    lines += ['']
    lines += distribution_summary(results_16)
    lines += ['', '## Files', '',
              f'- Raw log: `{os.path.relpath(LOG_PATH, os.path.join(HERE, "..", ".."))}`',
              f'- Per-window/eval-week rows: `{os.path.relpath(ROWS_PATH, os.path.join(HERE, "..", ".."))}`']

    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log.info(f'wrote {MD_PATH}')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
