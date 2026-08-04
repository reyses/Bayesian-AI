"""LEVEL-FRAME STABILITY GATE — is the coordinate frame stationary enough to
model on? (2026-07-28, follows level_coordinate_system.py)

For each test day D: fit the TF telescope on the PRIOR days only (causal), then
measure how day D's own pivots relate to that pre-existing frame:
  - capture: % of day-D pivot events within ±tau of a prior-frame line
  - control: same stat for RANDOM line sets (same count, same price range,
    resampled NULL_DRAWS times) -> null distribution + percentile
A frame that beats its null consistently = stationary enough to be a feature.
A frame that matches its null = the levels decay overnight; coordinate features
would be built on sand. Either answer is a result.

Run:
  python research/dojo_forge/tools/level_frame_stability.py \
      --fit-days 3 --test-days 2026_07_10 2026_07_13 2026_07_14 2026_07_15 2026_07_16 2026_07_17
Output -> research/dojo_forge/reports/human_dojo/frame_stability.{md,json}
"""
import argparse
import json
import os

import numpy as np

from level_coordinate_system import (DATA, OUT, TELESCOPE, load_days,
                                     telescope, zigzag_pivots)

NULL_DRAWS = 2000       # random-line resamples for the null distribution
RNG_SEED = 7            # deterministic null


def day_list():
    return sorted(f[:-8] for f in os.listdir(DATA) if f.endswith('.parquet'))


def capture_stat(pivot_prices, pivot_times, lines_with_tau):
    """% of distinct pivot EVENTS with >=1 candidate price within a band."""
    if not lines_with_tau or len(pivot_prices) == 0:
        return np.nan
    ev = {}
    for p, t in zip(pivot_prices, pivot_times):
        hit = any(abs(p - lp) <= lt for lp, lt in lines_with_tau)
        ev[t] = ev.get(t, False) or hit
    return float(np.mean(list(ev.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit-days', type=int, default=3, help='prior days per fit')
    ap.add_argument('--test-days', nargs='+', default=None,
                    help='days to test (default: all with enough history)')
    args = ap.parse_args()

    all_days = day_list()
    tests = args.test_days or all_days[args.fit_days:]
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for td in tests:
        if td not in all_days:
            print(f'skip {td}: no data'); continue
        i = all_days.index(td)
        fit = all_days[max(0, i - args.fit_days):i]
        if len(fit) < args.fit_days:
            print(f'skip {td}: only {len(fit)} prior days'); continue

        # frame from PRIOR days only (causal)
        scales = telescope(load_days(fit))
        lines = [(L['price'], sc['tau']) for sc in scales for L in sc['lines']]
        if not lines:
            print(f'skip {td}: empty frame'); continue

        # day-D pivots (finest scale radius — the touches the frame should catch)
        df = load_days([td])
        piv = zigzag_pivots(df['high'].to_numpy(), df['low'].to_numpy(),
                            df['close'].to_numpy(), TELESCOPE[-1]['R'],
                            df['open'].to_numpy())
        pp = np.array([p for _, p, _ in piv]); pt = np.array([t for t, _, _ in piv])
        if len(pp) < 5:
            print(f'skip {td}: {len(pp)} pivots'); continue

        cap = capture_stat(pp, pt, lines)

        # null: random lines, same count & taus, uniform over day-D price range
        lo, hi = df['low'].min(), df['high'].max()
        taus = [t for _, t in lines]
        null = np.empty(NULL_DRAWS)
        for k in range(NULL_DRAWS):
            rand_lines = list(zip(rng.uniform(lo, hi, len(lines)), taus))
            null[k] = capture_stat(pp, pt, rand_lines)
        pct = float((cap > null).mean())
        rows.append({'day': td, 'fit_days': fit, 'n_lines': len(lines),
                     'n_pivot_events': int(len(np.unique(pt))),
                     'capture': round(cap, 3),
                     'null_mean': round(float(null.mean()), 3),
                     'null_p95': round(float(np.quantile(null, 0.95)), 3),
                     'pctile_vs_null': round(pct, 3)})
        print(f"{td}: capture={cap:.2f} null={null.mean():.2f} "
              f"(p95={np.quantile(null,0.95):.2f}) pctile={pct:.2f}")

    if not rows:
        raise SystemExit('no test days ran')
    caps = np.array([r['capture'] for r in rows])
    nulls = np.array([r['null_mean'] for r in rows])
    beats = sum(r['pctile_vs_null'] >= 0.95 for r in rows)
    verdict = ('STABLE — frame carries overnight'
               if np.median(caps - nulls) > 0 and beats >= len(rows) * 0.5
               else 'UNSTABLE — levels decay; frame must be intraday-refit')
    summary = {'n_test_days': len(rows), 'mean_capture': round(float(caps.mean()), 3),
               'mean_null': round(float(nulls.mean()), 3),
               'days_beating_null_p95': beats, 'verdict': verdict, 'rows': rows}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'frame_stability.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, 'frame_stability.md'), 'w') as f:
        f.write(f"# Level-frame stability gate\n\nfit_days={args.fit_days} · "
                f"{len(rows)} test days · NULL_DRAWS={NULL_DRAWS}\n\n"
                f"**{verdict}**\n\nmean capture {caps.mean():.2f} vs null "
                f"{nulls.mean():.2f}; {beats}/{len(rows)} days beat null p95\n\n"
                "|day|capture|null mean|null p95|pctile|\n|--|--|--|--|--|\n")
        for r in rows:
            f.write(f"|{r['day']}|{r['capture']}|{r['null_mean']}|"
                    f"{r['null_p95']}|{r['pctile_vs_null']}|\n")
    print('\nVERDICT:', verdict)
    print('->', os.path.join(OUT, 'frame_stability.md'))


if __name__ == '__main__':
    main()
