"""Failure autopsy: WHERE does the deployed signal lose? (Moises, 2026-07-08)

Instead of blind lever sweeps, decompose every 2025 dev-set trade:
  - immediate-adverse (direction wrong at entry: MAE hit before any MFE)
  - gave-back (right direction, trail returned the open profit)
  - by hour (CT), by score strength, by hold time, by stretch side
Each trade logs: pnl, MFE, MAE (from the 5s stream), hold_min, hour, score,
direction, zsum. Output: loss decomposition table -> which lever matters.
"""
import argparse
import glob
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nt8_deploy_test import (day_stream, score, train_2024,  # noqa: E402
                             COLS, ZH, ZL, TICK, TICK_VALUE, COST_TICKS, ATLAS)

REPORT_DIR = os.path.join(_REPO, 'research', 'leg_clock', 'reports')
lines = []


def log(s):
    print(s, flush=True); lines.append(s)


def run(days, model, th, trail):
    import pytz, datetime as dtm
    central = pytz.timezone('US/Central')
    recs = []
    for day in days:
        try:
            ts_m, F, px_m, ts5, px5 = day_stream(day, ATLAS)
        except Exception:
            continue
        if len(ts_m) < 100:
            continue
        Fn = np.nan_to_num(F)
        s = score(Fn, model)
        zsum = Fn[:, ZH] + Fn[:, ZL]
        d0 = dtm.datetime.fromtimestamp(ts5[-1], tz=dtm.timezone.utc).astimezone(central)
        cut = central.localize(dtm.datetime(d0.year, d0.month, d0.day, 15, 55)).timestamp()
        pos, entry, ext, e_ts, e_i, mfe, mae = 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0
        k5 = 0
        for i in range(30, len(ts_m)):
            t = ts_m[i]
            while k5 < len(ts5) and ts5[k5] <= t:
                p = px5[k5]
                if pos != 0:
                    fav = (p - entry) * pos / TICK
                    mfe = max(mfe, fav); mae = min(mae, fav)
                    ext = max(ext, p) if pos > 0 else min(ext, p)
                    hit = (ext - p >= trail * TICK) if pos > 0 else (p - ext >= trail * TICK)
                    if hit or ts5[k5] >= cut:
                        pnl = ((p - entry) * pos / TICK - COST_TICKS) * TICK_VALUE
                        hr = dtm.datetime.fromtimestamp(e_ts, tz=dtm.timezone.utc)\
                            .astimezone(central).hour
                        recs.append(dict(pnl=pnl, mfe=mfe, mae=mae,
                                         hold=(ts5[k5] - e_ts) / 60.0, hour=hr,
                                         score=float(s[e_i]), zsum=float(zsum[e_i])))
                        pos = 0
                k5 += 1
            if pos == 0 and t < cut and s[i] >= th:
                pos = -1 if zsum[i] > 0 else 1
                entry = px_m[i]; ext = px_m[i]; e_ts = t; e_i = i
                mfe = 0.0; mae = 0.0
        # skip open-at-EOD for autopsy cleanliness
    return recs


def bucket_table(recs, key, edges, labels):
    log(f"\n-- by {key} --")
    log(f"{'bucket':<12}{'n':>6}{'$/tr':>8}{'win%':>7}{'medMFE':>8}{'medMAE':>8}")
    v = np.array([r[key] for r in recs])
    pnl = np.array([r['pnl'] for r in recs])
    mfe = np.array([r['mfe'] for r in recs])
    mae = np.array([r['mae'] for r in recs])
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (v >= lo) & (v < hi)
        if m.sum() < 10:
            continue
        log(f"{lab:<12}{m.sum():>6}{pnl[m].mean():>8.2f}{100*(pnl[m]>0).mean():>7.1f}"
            f"{np.median(mfe[m]):>8.1f}{np.median(mae[m]):>8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trail', type=float, default=20)
    ap.add_argument('--tier', type=float, default=0.995)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    model, _ = train_2024(rng)
    days24 = sorted(os.path.basename(f).replace('.parquet', '')
                    for f in glob.glob(os.path.join(ATLAS, '1m', '2024_*.parquet')))
    samp = []
    for day in days24[::13][:20]:
        try:
            _, F, _, _, _ = day_stream(day, ATLAS)
            samp.append(score(np.nan_to_num(F), model))
        except Exception:
            pass
    th = float(np.quantile(np.concatenate(samp), args.tier))
    days25 = sorted(os.path.basename(f).replace('.parquet', '')
                    for f in glob.glob(os.path.join(ATLAS, '1m', '2025_*.parquet')))
    recs = run(days25, model, th, args.trail)
    pnl = np.array([r['pnl'] for r in recs])
    mfe = np.array([r['mfe'] for r in recs])
    mae = np.array([r['mae'] for r in recs])
    log(f"autopsy: tier q{args.tier} trail {args.trail}t | {len(recs)} closed trades | "
        f"$/tr {pnl.mean():+.2f} | win% {100*(pnl>0).mean():.1f}")

    # Failure taxonomy
    never_worked = mfe < 8            # <2pt favorable ever = direction/timing wrong
    gave_back = (mfe >= 20) & (pnl < 0)  # had 5pt+ open profit, still lost
    log(f"\nFailure taxonomy:")
    log(f"  never-worked (<8t MFE):     {100*never_worked.mean():.1f}% of trades, "
        f"$/tr {pnl[never_worked].mean():+.2f}")
    log(f"  gave-back (>=20t MFE, lost): {100*gave_back.mean():.1f}% of trades, "
        f"$/tr {pnl[gave_back].mean():+.2f}, med MFE {np.median(mfe[gave_back]):.0f}t")
    winners = pnl > 0
    log(f"  winners:                     {100*winners.mean():.1f}%, "
        f"$/tr {pnl[winners].mean():+.2f}, med MFE {np.median(mfe[winners]):.0f}t")

    bucket_table(recs, 'hour', [0, 7, 9, 11, 13, 15, 24],
                 ['pre-7', '7-9', '9-11', '11-13', '13-15', '15+'])
    sc = np.array([r['score'] for r in recs])
    qs = list(np.quantile(sc, [0, .25, .5, .75])) + [sc.max() + 1e-9]
    bucket_table(recs, 'score', list(qs), ['scoreQ1', 'scoreQ2', 'scoreQ3', 'scoreQ4'])
    bucket_table(recs, 'zsum', [-99, -2, 0, 2, 99],
                 ['deep-low', 'low', 'high', 'deep-high'])
    bucket_table(recs, 'hold', [0, 2, 5, 15, 60, 1e9],
                 ['<2m', '2-5m', '5-15m', '15-60m', '>60m'])

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, 'failure_autopsy.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
