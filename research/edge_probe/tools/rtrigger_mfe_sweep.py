#!/usr/bin/env python3
"""R-TRIGGER MFE + R-SWEEP (owner 2026-07-27): does the causal R-trigger exit
leave money on the table, or does the leg just not pay after the dossier enters?

For every dossier entry in the full-ATLAS ledger, walk the forward 1m close path
(entry -> session end, causal) and measure:
  MFE  = max favorable excursion (pts) — the CEILING any exit could capture
  bar@MFE = how many bars after entry the peak lands
Then simulate a TRAILING-REVERSAL exit (the operative R-trigger behaviour: exit
when price retraces R pts from the favorable running extreme) over a grid of R,
and report captured pts + capture efficiency (captured / MFE). This isolates the
EXIT: if a smaller R captures much more, the ATR*4 R-trigger is too late; if even
the MFE ceiling is ~0, the leg doesn't pay and the problem is the entry/ride, not R.

Uses research/nt8_port/atlas_backtest/<day>.parquet (entries) + DATA/ATLAS/1m
(close path). Causal, per-day, no lookahead. CPU-only.
reports/rtrigger_mfe_sweep.md + assets/rtrigger_mfe_sweep.png
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
VEC = os.path.join(REPO, 'research', 'nt8_port', 'atlas_backtest')
ATLAS1M = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
OUT_MD = os.path.join(PROJ, 'reports', 'rtrigger_mfe_sweep.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'rtrigger_mfe_sweep.png')

R_GRID = [4, 6, 8, 12, 16, 20, 30, 40, 60]   # retrace-from-peak exit (pts)
PT_USD = 2.0


def trailing_exit(path, d, R):
    """Causal trailing-reversal: enter at path[0], exit when favorable extreme
    retraces R pts. Returns captured pts (dir-signed) at exit close."""
    e = path[0]
    peak = 0.0                       # best favorable excursion so far (pts)
    for j in range(1, len(path)):
        fav = d * (path[j] - e)
        if fav > peak:
            peak = fav
        elif peak - fav >= R:        # retraced R from the peak -> exit
            return fav
    return d * (path[-1] - e)        # session close


def main():
    # per-day close path aligned to the RTH vector bar_ts
    rows = []
    for f in sorted(glob.glob(os.path.join(VEC, '*.parquet'))):
        day = os.path.basename(f)[:10]
        v = pd.read_parquet(f, columns=['bar_ts', 'entry', 'entry_dir']).sort_values('bar_ts')
        p1 = os.path.join(ATLAS1M, f'{day}.parquet')
        if not os.path.exists(p1):
            continue
        m = pd.read_parquet(p1, columns=['timestamp', 'close'])
        cl = dict(zip(m['timestamp'].astype('int64'), m['close'].astype(float)))
        bts = v['bar_ts'].astype('int64').to_numpy()
        closes = np.array([cl.get(int(t), np.nan) for t in bts])
        ent = v['entry'].to_numpy()
        edir = v['entry_dir'].to_numpy()
        pos = 0
        for i in range(len(bts)):
            if np.isnan(closes[i]):
                continue
            if pos == 0 and ent[i] == 1:              # one position at a time
                pos = int(edir[i])
                fwd = closes[i:]
                fwd = fwd[~np.isnan(fwd)]
                if len(fwd) < 2:
                    pos = 0
                    continue
                d = pos
                mfe = float(np.max(d * (fwd - fwd[0])))
                bar_mfe = int(np.argmax(d * (fwd - fwd[0])))
                rec = dict(day=day, dir=d, mfe=mfe, bar_mfe=bar_mfe,
                           mae=float(np.min(d * (fwd - fwd[0]))))
                for R in R_GRID:
                    rec[f'cap_{R}'] = trailing_exit(fwd, d, R)
                rows.append(rec)
            elif pos != 0:
                # crude re-flat when an opposite entry-dir would flip (keeps 1-pos)
                if ent[i] == 1 and int(edir[i]) == -pos:
                    pos = 0
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROJ, 'reports', 'rtrigger_mfe_sweep.csv'), index=False)

    n = len(df)
    mfe_mean = df['mfe'].mean()
    lines = [
        '# R-trigger MFE + R-sweep (full ATLAS, dossier entries)',
        f'{n:,} entries, {df["day"].nunique()} days. Forward 1m close path per '
        'entry; trailing-reversal exit at each R vs the MFE ceiling.',
        '',
        f'- **Mean MFE (ceiling any exit could capture): {mfe_mean:+.1f} pts** '
        f'(${mfe_mean*PT_USD:+.2f}); median bar@MFE = {df["bar_mfe"].median():.0f} '
        f'bars after entry',
        f'- Mean MAE (worst adverse before exit): {df["mae"].mean():+.1f} pts',
        '',
        '| exit R (pt) | mean captured (pt) | capture eff (cap/MFE) | mean $ (1 ct) | win% |',
        '|---|---|---|---|---|',
    ]
    curve = []
    for R in R_GRID:
        c = df[f'cap_{R}']
        eff = c.mean() / mfe_mean if mfe_mean else float('nan')
        win = (c > 0).mean()
        curve.append((R, c.mean(), eff, win))
        lines.append(f'| {R} | {c.mean():+.2f} | {eff:.0%} | '
                     f'{c.mean()*PT_USD:+.2f} | {win:.0%} |')
    best = max(curve, key=lambda t: t[1])
    lines += ['',
              f'Best R by mean-captured: **R={best[0]}** ({best[1]:+.2f} pts, '
              f'{best[2]:.0%} of MFE). ATR*4 R-trigger sits near the large-R end.',
              '',
              'Read: if capture peaks at a SMALL R well below ATR*4, the R-trigger '
              'exits too late and a tighter reversal captures more. If mean captured '
              'stays ~0 across all R while MFE is large, the leg peaks early then '
              'fully round-trips — the ride structure (not R) is the problem. If MFE '
              'itself is small, the dossier entry has little favorable room and the '
              'issue is entry timing, not exit.']
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)
    Rs = [c[0] for c in curve]
    ax1.plot(Rs, [c[1] for c in curve], 'o-', color='tab:blue', label='mean captured (pt)')
    ax1.axhline(mfe_mean, color='tab:green', ls='--', label=f'MFE ceiling {mfe_mean:.1f}')
    ax1.axhline(0, color='black', lw=1)
    ax1.set_xlabel('exit R = retrace-from-peak threshold (pts)')
    ax1.set_ylabel('mean captured pts', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(Rs, [c[3] * 100 for c in curve], 's--', color='tab:red', label='win %')
    ax2.set_ylabel('win %', color='tab:red')
    ax1.set_title('Does the R-trigger leave money on the table? (capture vs R)')
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
