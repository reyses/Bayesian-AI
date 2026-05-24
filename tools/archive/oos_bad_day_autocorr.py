"""ANALYZE — do OOS bad days CLUSTER day-to-day?

The "lift the OOS bad days" research has shown every reactive lever fails
(per-trade drawdown stop; intraday cumulative-P&L session-stop) and
session-start macro prediction failed earlier (DRS). This checks the last
open ex-ante signal: is a bad session followed by another bad one? If daily
P&L — or the chop regime behind it — is autocorrelated, then "the prior
session was bad -> de-risk today" is an actionable session-start signal. If
not, that door is closed too.

FLAT substrate: daily P&L = sum of per-leg pnl_usd from the hardened-leg CSVs.
Also lag-1 autocorrelation of two chop proxies (legs/day, mean leg amplitude),
since the chop regime can persist even when noisy P&L does not.
IS 275 days + OOS 51 sealed days.

Output: reports/findings/oos_bad_days/2026-05-21_autocorr.md
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LEGS = {'IS':  REPO / 'reports/findings/regret_oracle/is_hardened_legs.csv',
        'OOS': REPO / 'reports/findings/regret_oracle/oos_hardened_legs_full.csv'}
OUT_MD = REPO / 'reports/findings/oos_bad_days/2026-05-21_autocorr.md'
N_PERM = 4000
SEED = 42


def per_day(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    g = df.groupby('day')
    return pd.DataFrame({
        'pnl': g['pnl_usd'].sum(),
        'n_legs': g.size(),
        'leg_amp': g['pnl_pts'].apply(lambda s: s.abs().mean()),
    }).sort_index()


def lag1(x) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.std(x) == 0:
        return float('nan')
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def perm_p(x, obs, n=N_PERM, seed=SEED) -> float:
    """Two-sided permutation p-value for the lag-1 autocorrelation."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    hits = sum(abs(lag1(rng.permutation(x))) >= abs(obs) for _ in range(n))
    return hits / n


def longest_run(flags) -> int:
    best = cur = 0
    for v in flags:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main():
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    L = []
    def out(s=''):
        L.append(s)

    out('# Do OOS bad days cluster? — lag-1 persistence (ANALYZE)')
    out('')
    out('Last open ex-ante signal for the "lift the OOS bad days" goal: is a '
        'bad session followed by another? FLAT daily P&L from the hardened-leg '
        f'CSVs; lag-1 autocorrelation + bad-day persistence; {N_PERM}-shuffle '
        'permutation test. Also lag-1 of two chop proxies (legs/day, mean leg '
        'amplitude).')
    out('')

    for label, path in LEGS.items():
        d = per_day(path)
        x = d['pnl'].values
        bad = (x < 0).astype(int)
        ac = lag1(x)
        p = perm_p(x, ac)
        ac_legs = lag1(d['n_legs'].values)
        ac_amp = lag1(d['leg_amp'].values)
        base = bad.mean()
        prior, today = bad[:-1], bad[1:]
        pbb = today[prior == 1].mean() if (prior == 1).any() else float('nan')
        pbg = today[prior == 0].mean() if (prior == 0).any() else float('nan')
        obs_run = longest_run(bad)
        rng = np.random.default_rng(SEED)
        run_p = float(np.mean([longest_run(rng.permutation(bad)) >= obs_run
                               for _ in range(N_PERM)]))

        verdict = 'persistent' if p < 0.05 else 'NOT distinguishable from iid'
        out(f'## {label} — {len(d)} days')
        out('')
        out(f'- Daily P&L lag-1 autocorrelation: **{ac:+.3f}** '
            f'(permutation p = {p:.3f} -> {verdict}).')
        out(f'- Chop-proxy lag-1 autocorr: legs/day {ac_legs:+.3f}, '
            f'mean leg amplitude {ac_amp:+.3f}.')
        out(f'- Bad-day base rate {base*100:.0f}%.  '
            f'P(bad | prior day bad) = {pbb*100:.0f}%;  '
            f'P(bad | prior day good) = {pbg*100:.0f}%.')
        out(f'- Longest observed bad streak: {obs_run} days '
            f'(iid permutation p = {run_p:.3f}).')
        out('')

    OUT_MD.write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))
    print(f'Wrote {OUT_MD}')


if __name__ == '__main__':
    main()
