#!/usr/bin/env python3
"""LEG-TERMINATED drift (owner 2026-07-27): the prior MFE/MAE walked to session
close -> spanned MULTIPLE ebb/flow legs -> symmetric by construction (that's just
the day's range). Correct window = entry -> NEXT R-trigger reversal = ONE leg.

For each dossier entry, terminate the forward walk at the next zz_confirm == -dir
(the actual R-trigger exit) or session end, and measure MFE/MAE/captured WITHIN
that single leg. Plus a short-horizon curve (MFE vs |MAE| at N = 3..30 bars) to
locate where — if anywhere — the entry has directional drift before symmetry
sets in. If within-leg MFE > |MAE| and short-horizon MFE dominates, the entry
DOES have a ride and the exit is the problem. If still symmetric leg-by-leg, the
entry has no drift. Causal, per-day, 1m close. CPU-only.
reports/rtrigger_leg_drift.md + assets/rtrigger_leg_drift.png
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
OUT_MD = os.path.join(PROJ, 'reports', 'rtrigger_leg_drift.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'rtrigger_leg_drift.png')

HORIZONS = [3, 5, 8, 10, 15, 20, 30]
PT_USD = 2.0


def main():
    legs = []
    hz = {N: {'mfe': [], 'mae': []} for N in HORIZONS}
    for f in sorted(glob.glob(os.path.join(VEC, '*.parquet'))):
        day = os.path.basename(f)[:10]
        v = pd.read_parquet(f, columns=['bar_ts', 'entry', 'entry_dir', 'zz_confirm']).sort_values('bar_ts')
        p1 = os.path.join(ATLAS1M, f'{day}.parquet')
        if not os.path.exists(p1):
            continue
        m = pd.read_parquet(p1, columns=['timestamp', 'close'])
        cl = dict(zip(m['timestamp'].astype('int64'), m['close'].astype(float)))
        bts = v['bar_ts'].astype('int64').to_numpy()
        closes = np.array([cl.get(int(t), np.nan) for t in bts])
        ent = v['entry'].to_numpy(); edir = v['entry_dir'].to_numpy()
        zc = v['zz_confirm'].to_numpy()
        pos = 0
        for i in range(len(bts)):
            if np.isnan(closes[i]):
                continue
            if pos == 0 and ent[i] == 1:
                d = int(edir[i]); e = closes[i]
                # leg end = next R-trigger reversal against pos, else session end
                end = len(bts) - 1
                for j in range(i + 1, len(bts)):
                    if not np.isnan(closes[j]) and int(zc[j]) == -d:
                        end = j
                        break
                path = closes[i:end + 1]
                path = path[~np.isnan(path)]
                if len(path) < 2:
                    continue
                fav = d * (path - path[0])
                legs.append(dict(day=day, dir=d, leg_bars=len(path) - 1,
                                 mfe=float(fav.max()), mae=float(fav.min()),
                                 bar_mfe=int(np.argmax(fav)),
                                 captured=float(fav[-1])))
                for N in HORIZONS:
                    w = fav[1:N + 1]
                    if len(w):
                        hz[N]['mfe'].append(float(w.max()))
                        hz[N]['mae'].append(float(w.min()))
                pos = 0                       # flat again at leg end (approx)
    df = pd.DataFrame(legs)
    df.to_csv(os.path.join(PROJ, 'reports', 'rtrigger_leg_drift.csv'), index=False)

    mfe, mae = df['mfe'].mean(), df['mae'].mean()
    cap = df['captured'].mean()
    lines = [
        '# Leg-terminated drift (entry -> next R-trigger reversal, ATLAS 1m)',
        f'{len(df):,} legs, {df["day"].nunique()} days. Window = ONE ebb/flow leg '
        '(entry to next reversal), not session close.',
        '',
        f'- median leg length: {df["leg_bars"].median():.0f} bars '
        f'(mean {df["leg_bars"].mean():.1f})',
        f'- **within-leg MFE {mfe:+.1f} pts vs MAE {mae:+.1f} pts** '
        f'-> asymmetry MFE/|MAE| = {mfe/abs(mae):.2f}',
        f'- captured at R-trigger exit: {cap:+.2f} pts (${cap*PT_USD:+.2f}); '
        f'that is {cap/mfe:.0%} of within-leg MFE',
        f'- median bar@MFE within leg: {df["bar_mfe"].median():.0f}',
        '',
        '## Short-horizon drift: MFE vs |MAE| at N bars after entry',
        '| horizon N (bars) | mean MFE | mean |MAE| | MFE/|MAE| | drift? |',
        '|---|---|---|---|---|',
    ]
    curve = []
    for N in HORIZONS:
        a = np.mean(hz[N]['mfe']); b = abs(np.mean(hz[N]['mae']))
        ratio = a / b if b else float('nan')
        curve.append((N, a, b, ratio))
        lines.append(f'| {N} | {a:+.1f} | {b:.1f} | {ratio:.2f} | '
                     f'{"UP-drift" if ratio > 1.15 else "symmetric" if ratio > 0.87 else "DOWN-drift"} |')
    lines += ['',
              'Read: MFE/|MAE| > 1 within the leg (and at short horizons) => the '
              'entered leg drifts favorably and a ride exists (exit is the lever). '
              '~1 => symmetric even leg-by-leg => the entry has no directional '
              'drift and no exit can extract $ from it.']
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    Ns = [c[0] for c in curve]
    ax.plot(Ns, [c[1] for c in curve], 'o-', color='tab:green', label='mean MFE')
    ax.plot(Ns, [c[2] for c in curve], 's-', color='tab:red', label='mean |MAE|')
    ax.set_xlabel('bars after entry (N)'); ax.set_ylabel('pts')
    ax.set_title(f'Within-leg drift: MFE vs |MAE| (leg MFE/|MAE|={mfe/abs(mae):.2f})')
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
