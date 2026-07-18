"""Which labeled trades does the 9-13CT config MISS? (Moises, 2026-07-08)

For every 2025 labeled trade with entry inside the window: score at the last
1m close strictly before entry_ts (FPS-causal). Caught = score >= tier.
Characterize missed vs caught: score percentile (just-below vs deep-below),
stretch (zsum), and the label's size (extent ticks / duration min) — are we
missing the big ones or the small ones? Labels are hindsight; this is a
diagnostic for HOW to widen entries, not an edge claim.
"""
import glob
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nt8_deploy_test import day_stream, score, train_2024, ZH, ZL, TICK, ATLAS  # noqa: E402

REPORT_DIR = os.path.join(_REPO, 'research', 'leg_clock', 'reports')
lines = []


def log(s):
    print(s, flush=True); lines.append(s)


def main():
    import pytz, datetime as dtm
    central = pytz.timezone('US/Central')
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
    pool = np.sort(np.concatenate(samp))
    th = float(np.quantile(pool, 0.995))

    days25 = sorted(os.path.basename(f).replace('.parquet', '')
                    for f in glob.glob(os.path.join(ATLAS, '1m', '2025_*.parquet')))
    recs = []
    for day in days25:
        pick = os.path.join(_REPO, 'DATA', 'ai_cusp_picks',
                            f"ai_picks_{day.replace('_', '-')}_multi.json")
        if not os.path.exists(pick):
            continue
        try:
            ts_m, F, _, _, _ = day_stream(day, ATLAS)
        except Exception:
            continue
        if len(ts_m) < 100:
            continue
        s = score(np.nan_to_num(F), model)
        Fn = np.nan_to_num(F)
        zsum = Fn[:, ZH] + Fn[:, ZL]
        for t in json.load(open(pick)).get('trades', []):
            hr = dtm.datetime.fromtimestamp(t['entry_ts'], tz=dtm.timezone.utc)\
                .astimezone(central).hour
            if not (9 <= hr < 13):
                continue
            i = np.searchsorted(ts_m, t['entry_ts']) - 1
            if i < 30:
                continue
            pctile = np.searchsorted(pool, s[i]) / len(pool)
            recs.append(dict(
                score=s[i], pct=pctile, caught=s[i] >= th, zsum=zsum[i],
                ext=abs(t['exit_price'] - t['entry_price']) / TICK,
                dur=(t['exit_ts'] - t['entry_ts']) / 60.0))
    n = len(recs)
    caught = np.array([r['caught'] for r in recs])
    pct = np.array([r['pct'] for r in recs])
    ext = np.array([r['ext'] for r in recs])
    dur = np.array([r['dur'] for r in recs])
    zs = np.array([r['zsum'] for r in recs])
    log(f"2025 labels in 9-13 CT: {n} | caught by q0.995: {caught.mean()*100:.1f}% "
        f"({caught.sum()})")
    m = ~caught
    log(f"\nmissed ({m.sum()}):")
    log(f"  score percentile: median {np.median(pct[m])*100:.1f} | "
        f"25-75%: {np.percentile(pct[m],25)*100:.0f}-{np.percentile(pct[m],75)*100:.0f}")
    log(f"  within-reach (pct>=0.98): {(pct[m]>=0.98).mean()*100:.1f}% of missed")
    log(f"  label size:  ext med {np.median(ext[m]):.0f}t vs caught {np.median(ext[caught]):.0f}t"
        f" | dur med {np.median(dur[m]):.0f}m vs caught {np.median(dur[caught]):.0f}m")
    log(f"  |zsum| med:  missed {np.median(np.abs(zs[m])):.2f} vs caught "
        f"{np.median(np.abs(zs[caught])):.2f}")
    # if we lowered the tier to q0.98, recall becomes:
    for q in (0.99, 0.98, 0.95):
        thq = float(np.quantile(pool, q))
        r = np.array([r_['score'] >= thq for r_ in recs]).mean()
        log(f"  recall at q{q}: {r*100:.1f}%")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, 'label_recall.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
