"""Dev-loop harness: iterate the entry signal on 2025 Databento; NT8 stays sealed.

Protocol (agreed 2026-07-08):
- Train logistic on 2024 label-entries vs matched nulls (FPS-extracted, causal).
- Deploy-style evaluation on 2025 Databento: causal entries + causal exits,
  labels NEVER touched at deploy time. Big N — this is where we tune.
- NT8 (138 pristine days) is the FINAL GATE: one look per milestone via
  nt8_deploy_test.py, never iterated against.

Levers exposed as flags so variants are one CLI call each:
  --direction {fade, follow, vel}   entry direction rule
  --trail N                         trail width in ticks
  --tiers q1,q2                     null-score quantile thresholds
Reports $/day + day-block bootstrap CI + PF per (tier), per variant.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nt8_deploy_test import (day_stream, fit_logistic, score, train_2024,  # noqa: E402
                             COLS, ZH, ZL, TICK, TICK_VALUE, COST_TICKS, ATLAS)

VEL = COLS.index('L2_1m_price_velocity_30')
REPORT_DIR = os.path.join(_REPO, 'research', 'leg_clock', 'reports')
lines = []


def log(s):
    print(s, flush=True); lines.append(s)


def deploy_days(days, root, model, thr, direction, trail, cut_hhmm=(15, 55)):
    import pytz, datetime as dtm
    central = pytz.timezone('US/Central')
    results = {k: [] for k in thr}
    for day in days:
        try:
            ts_m, F, px_m, ts5, px5 = day_stream(day, root)
        except Exception:
            continue
        if len(ts_m) < 100:
            continue
        Fn = np.nan_to_num(F)
        s = score(Fn, model)
        zsum = Fn[:, ZH] + Fn[:, ZL]
        vel = Fn[:, VEL]
        d0 = dtm.datetime.fromtimestamp(ts5[-1], tz=dtm.timezone.utc).astimezone(central)
        cut = central.localize(dtm.datetime(d0.year, d0.month, d0.day, *cut_hhmm)).timestamp()
        for tier, th in thr.items():
            trades = []
            pos, entry, ext = 0, 0.0, 0.0
            k5 = 0
            for i in range(30, len(ts_m)):
                t = ts_m[i]
                while k5 < len(ts5) and ts5[k5] <= t:
                    p = px5[k5]
                    if pos != 0:
                        ext = max(ext, p) if pos > 0 else min(ext, p)
                        hit = (ext - p >= trail * TICK) if pos > 0 else (p - ext >= trail * TICK)
                        if hit or ts5[k5] >= cut:
                            trades.append(((p - entry) / TICK * pos - COST_TICKS) * TICK_VALUE)
                            pos = 0
                    k5 += 1
                if pos == 0 and t < cut and s[i] >= th:
                    if direction == 'fade':
                        pos = -1 if zsum[i] > 0 else 1
                    elif direction == 'follow':
                        pos = 1 if zsum[i] > 0 else -1
                    else:  # vel: fade the recent velocity (pullback-reversion)
                        pos = -1 if vel[i] > 0 else 1
                    entry = px_m[i]; ext = px_m[i]
            if pos != 0:
                trades.append(((px5[-1] - entry) / TICK * pos - COST_TICKS) * TICK_VALUE)
            results[tier].append(trades)
    return results


def report(tag, results):
    for tier, per_day in results.items():
        days_n = len(per_day)
        daily = np.array([sum(t) for t in per_day])
        allt = (np.concatenate([np.array(t) for t in per_day if t])
                if any(per_day) else np.array([]))
        if len(allt) == 0:
            log(f"[{tag}:{tier}] no trades"); continue
        wins = allt[allt > 0].sum(); losses = -allt[allt < 0].sum()
        pf = wins / losses if losses > 0 else float('inf')
        rng = np.random.default_rng(1)
        boots = np.array([daily[rng.integers(0, days_n, days_n)].mean()
                          for _ in range(4000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = 'NOT sig' if lo <= 0 <= hi else 'SIG'
        log(f"[{tag}:{tier}] {len(allt)/days_n:.1f} tr/d | $/tr {allt.mean():+.2f} | "
            f"$/day {daily.mean():+.1f} [{lo:+.1f},{hi:+.1f}] {sig} | PF {pf:.2f} | "
            f"{days_n}d/{len(allt)}tr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--direction', choices=['fade', 'follow', 'vel'], default='fade')
    ap.add_argument('--trail', type=float, default=20)
    ap.add_argument('--tiers', type=str, default='0.98,0.995')
    ap.add_argument('--max-days', type=int, default=0, help='cap 2025 days (speed)')
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    model, _ = train_2024(rng)
    # thresholds re-derived from train nulls at requested quantiles
    days24 = sorted(os.path.basename(f).replace('.parquet', '')
                    for f in glob.glob(os.path.join(ATLAS, '1m', '2024_*.parquet')))
    # reuse a small sample of ordinary-bar scores for thresholds
    samp = []
    for day in days24[::13][:20]:
        try:
            _, F, _, _, _ = day_stream(day, ATLAS)
            samp.append(score(np.nan_to_num(F), model))
        except Exception:
            pass
    pool = np.concatenate(samp)
    thr = {f"q{q}": float(np.quantile(pool, float(q))) for q in args.tiers.split(',')}
    log(f"variant: direction={args.direction} trail={args.trail}t tiers={thr}")

    days25 = sorted(os.path.basename(f).replace('.parquet', '')
                    for f in glob.glob(os.path.join(ATLAS, '1m', '2025_*.parquet')))
    if args.max_days:
        days25 = days25[:args.max_days]
    res = deploy_days(days25, ATLAS, model, thr, args.direction, args.trail)
    report(f"2025dev:{args.direction}:T{int(args.trail)}", res)

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, 'dev_loop_2025.txt')
    with open(out, 'a') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nAppended to {out}')


if __name__ == '__main__':
    main()
