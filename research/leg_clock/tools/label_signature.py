"""What do the labeled opportunities have in common? (Moises, 2026-07-08)

Use the truth we already have (DATA/ai_cusp_picks golden trades) instead of
inventing regimes. Pass 1 (this): descriptive signature of the labels
themselves — duration, extent, MAE (heat), direction balance, time-of-day,
velocity. Pass 2 (next): pre-entry causal features vs a non-entry null, to see
if the signature is DISTINCT (common-to-labels AND rare-elsewhere), not just
common.
"""
import glob
import json
import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
REPORT_DIR = os.path.join(_REPO, 'research', 'leg_clock', 'reports')
TICK = 0.25
lines = []


def log(s):
    print(s); lines.append(s)


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    files = sorted(glob.glob(os.path.join(_REPO, 'DATA', 'ai_cusp_picks', '*_multi.json')))
    dur, ext, pnl, mae, marg, ct_hour = [], [], [], [], [], []
    long_n = short_n = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for t in d.get('trades', []):
            dt = (t['exit_ts'] - t['entry_ts']) / 60.0
            if dt <= 0:
                continue
            dur.append(dt)
            ext.append(abs(t['exit_price'] - t['entry_price']) / TICK)
            pnl.append(t.get('pnl_dollars', 0.0))
            mae.append(t.get('mae_dollars', 0.0))
            marg.append(1 if t.get('is_marginal') else 0)
            if str(t.get('direction', '')).upper().startswith('L'):
                long_n += 1
            else:
                short_n += 1
            # central time hour of entry
            h = datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc) - timedelta(hours=6)
            ct_hour.append(h.hour)
    dur, ext, pnl, mae = map(np.array, (dur, ext, pnl, mae))
    n = len(dur)
    log(f"labels: {n} trades across {len(files)} days ({n/len(files):.1f}/day)")
    log(f"direction: LONG {long_n} ({100*long_n/n:.0f}%) | SHORT {short_n} ({100*short_n/n:.0f}%)")
    log(f"marginal: {100*np.mean(marg):.0f}%")

    def dist(name, a, unit):
        hist, edges = np.histogram(a, bins=40)
        mode = edges[hist.argmax()] + (edges[1] - edges[0]) / 2
        log(f"{name:<10} mode~{mode:>7.1f} median {np.median(a):>7.1f} mean {a.mean():>7.1f}"
            f"  p10 {pct(a,10):>6.1f} p90 {pct(a,90):>7.1f} ({unit})")

    log("\n-- label distributions --")
    dist('duration', dur, 'min')
    dist('extent', ext, 'ticks')
    dist('pnl', pnl, '$')
    dist('mae', mae, '$ heat')
    # velocity = extent / duration (ticks per min)
    vel = ext / np.maximum(dur, 1e-9)
    dist('velocity', vel, 't/min')
    # MAE/extent ratio = heat taken relative to reward
    heat_ratio = mae / np.maximum(ext * TICK * 2.0, 1e-9)  # mae$ vs extent$ (2$/tick MNQ... approx)
    log(f"\nMAE=0 (no heat) trades: {100*np.mean(mae==0):.0f}%  "
        f"(entered right at the turn — clean)")
    # time-of-day
    hh = np.array(ct_hour)
    log("\n-- entry hour (CT) histogram --")
    for h in range(24):
        cnt = (hh == h).sum()
        if cnt:
            bar = '#' * int(40 * cnt / max(1, (hh == np.bincount(hh).argmax()).sum()))
            log(f"  {h:02d}:00  {cnt:>5}  {bar}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, 'label_signature.txt')
    with open(out, 'w') as fo:
        fo.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
