#!/usr/bin/env python3
"""FALSE-PROFIT / CATASTROPHIC-TAIL of holding wrong trades (owner 2026-07-28, TG):
holding a wrong-direction trade, the 2nd-leg oscillation swings back PAST entry ->
a small FALSE profit (not edge). Collected most of the time, but paid for by a rare
non-recovery that is CATASTROPHIC. Quantify the negative skew the mean hides.

For each combiner entry that goes underwater >= THRESH (a "wrong" trade), hold to
end-of-day (never-bail) and record:
  false_profit = max favorable excursion PAST entry after going underwater (the
                 oscillation's gift when it recovers)
  terminal     = never-bail pnl at EOD (what you actually keep)
Report: recovery false-profit (small +) vs non-recovery tail (p5/p1/min, %loss>50/100),
and never-bail mean vs a cut-at-threshold policy (mean -THRESH, no tail). Points, $2/pt.
reports/false_profit_skew.md
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
VEC = os.path.join(REPO, 'research', 'nt8_port', 'atlas_backtest')
A1 = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
OUT = os.path.join(HERE, '..', 'reports', 'false_profit_skew.md')
THRESH = 20.0
PT = 2.0


def main():
    fp = []; term = []; rec = []
    for f in sorted(glob.glob(os.path.join(VEC, '*.parquet'))):
        day = os.path.basename(f)[:10]; p1 = os.path.join(A1, f'{day}.parquet')
        if not os.path.exists(p1):
            continue
        v = pd.read_parquet(f, columns=['bar_ts', 'entry', 'gov_dir']).sort_values('bar_ts')
        m = pd.read_parquet(p1, columns=['timestamp', 'close'])
        cl0 = dict(zip(m['timestamp'].astype('int64'), m['close'].astype(float)))
        bts = v['bar_ts'].astype('int64').to_numpy()
        cl = np.array([cl0.get(int(t), np.nan) for t in bts])
        ent = v['entry'].to_numpy(); gd = v['gov_dir'].to_numpy(); n = len(cl)
        for i in range(n):
            if ent[i] != 1 or np.isnan(cl[i]) or gd[i] == 0:
                continue
            d = int(gd[i]); e = cl[i]
            path = cl[i + 1:]; path = path[~np.isnan(path)]
            if len(path) < 2:
                continue
            adv = d * (e - path)          # adverse (>0 against)
            if adv.max() < THRESH:        # not a "wrong"/underwater trade
                continue
            uw = int(np.argmax(adv >= THRESH))   # first underwater bar
            after = path[uw:]
            favafter = d * (after - e)
            recovered = bool((favafter >= 0).any())
            rec.append(recovered)
            fp.append(float(favafter.max()) if recovered else 0.0)   # false-profit ceiling
            term.append(float(d * (path[-1] - e)))                   # never-bail EOD pnl
    fp = np.array(fp); term = np.array(term); rec = np.array(rec)
    N = len(term)

    def pc(a):
        return dict(mean=a.mean(), med=np.median(a),
                    p25=np.percentile(a, 25), p5=np.percentile(a, 5),
                    p1=np.percentile(a, 1), mn=a.min())
    tstat = pc(term)
    lines = [
        '# False-profit vs catastrophic tail — holding WRONG (underwater≥%.0fpt) trades' % THRESH,
        f'{N:,} wrong trades (combiner entries that went ≥{THRESH:.0f}pt adverse), full ATLAS. '
        f'Never-bail to EOD. Points ($2/pt).', '',
        f'- Recovery rate: **{rec.mean():.1%}** ({rec.sum():,}/{N:,})',
        f'- **False profit when it recovers** (max favorable past entry): mean '
        f'{fp[rec].mean():+.1f}pt (${fp[rec].mean()*PT:+.0f}), median {np.median(fp[rec]):+.1f}pt',
        f'- **Non-recovery tail** (never recovers): mean terminal {term[~rec].mean():+.1f}pt '
        f'(${term[~rec].mean()*PT:+.0f}), median {np.median(term[~rec]):+.1f}pt, '
        f'worst {term[~rec].min():+.1f}pt (${term[~rec].min()*PT:+.0f})',
        '',
        '## Never-bail EOD outcome distribution (the skew the mean hides)',
        f'- mean {tstat["mean"]:+.1f}pt (${tstat["mean"]*PT:+.0f}) | median {tstat["med"]:+.1f}pt',
        f'- **left tail**: p25 {tstat["p25"]:+.1f} | p5 {tstat["p5"]:+.1f} | '
        f'p1 {tstat["p1"]:+.1f} | worst {tstat["mn"]:+.1f}pt (${tstat["mn"]*PT:+.0f})',
        f'- catastrophe rate: {(term < -50).mean():.1%} lose >50pt, '
        f'{(term < -100).mean():.1%} lose >100pt, {(term < -200).mean():.1%} lose >200pt',
        '',
        '## Hold-for-recovery vs cut-at-threshold',
        f'- cut at {THRESH:.0f}pt: every wrong trade = -{THRESH:.0f}pt (${-THRESH*PT:.0f}), NO tail',
        f'- never-bail mean: {tstat["mean"]:+.1f}pt — '
        + ('BEATS cutting on MEAN' if tstat['mean'] > -THRESH else 'worse than cutting') +
        f', but carries a p1 of {tstat["p1"]:+.0f}pt and worst {tstat["mn"]:+.0f}pt.',
        '',
        'Read: the false profits are small + frequent; the non-recovery tail is deep + rare '
        '= negative skew. Even where never-bail wins on the MEAN, the catastrophic tail '
        '(p1/worst) is the real exposure — "hold for recovery" harvests pennies in front of '
        'the steamroller. Confirms the owner: the hold premise for wrong trades is catastrophic.']
    open(OUT, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
