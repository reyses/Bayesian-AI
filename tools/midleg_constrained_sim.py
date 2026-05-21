"""Mid-leg entry  --  E3: position-constrained 1-contract late-join simulation.

E1 found B9-gated late join is +EV only at K<=10 (within ~50s of the missed
trigger) and decays to noise by K>=30. E1/E2 are UNCONSTRAINED -- they treat
every leg as a candidate. This script is the operational truth: a 1-contract
engine can only late-join while FLAT, and late-joining occupies the engine
(it may displace the next natural R-trigger).

SIM (per day, legs sorted by entry_ts):
  baseline  : greedy 1-contract -- whenever flat, take the next leg with
              entry_ts >= free_at, ride to its hardened exit, repeat.
  late-join : same greedy, but whenever flat the engine may instead late-join
              a currently-running missed leg if B9 pred (at snapped elapsed
              K) > friction. Late-joining occupies the engine until that
              leg's exit -> displacement of natural legs is captured.

Incremental $/day = late-join total - baseline total, paired bootstrap CI.
The KEY diagnostic is the K-distribution of realised late-joins: if the
engine mostly catches stale (K>=30) legs, the +EV from E1 never materialises.

Snapping: a late-join's elapsed time is snapped to the nearest available K
horizon {5,10,30,60,120}; B9 pred + remaining_pnl from that K-row are used.
This is an approximation (exact would need 5s-bar recompute) but the verdict
is robust to +/-1 K of snap error.

Output: reports/findings/regret_oracle/2026-05-20_midleg_constrained_sim.txt
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


K_HORIZONS = [5, 10, 30, 60, 120]
FRICTION_USD_PER_LEG = 6.0                 # round-trip commission+slippage, MNQ project convention
N_BOOTSTRAP = 4000
BOOTSTRAP_SEED = 42
RD = 'reports/findings/regret_oracle'
NEG_INF = -1e18


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_b9(K):
    with open(Path(RD) / f'b9_remaining_amplitude_K{K}.pkl', 'rb') as f:
        return pickle.load(f)


def simulate(day_legs, lookup, late_join, max_k, friction):
    """Greedy 1-contract sim for one day.

    day_legs : list of (leg_id, entry_ts, exit_ts, pnl_usd) -- any order.
    lookup   : leg_id -> list of (K, K_seconds, remaining_pnl, b9_pred).
    Returns  : (total_pnl, n_natural, n_latejoin, latejoin_records).
    """
    free_at = NEG_INF
    total = 0.0
    taken = set()
    n_nat = n_lj = 0
    lj_recs = []
    while True:
        future = [L for L in day_legs
                  if L[0] not in taken and L[1] >= free_at]
        best_lj = None
        if late_join:
            for L in day_legs:
                lid, ent, ext, _ = L
                if lid in taken or not (ent < free_at < ext):
                    continue
                cands = [c for c in lookup.get(lid, []) if c[0] <= max_k]
                if not cands:
                    continue
                elapsed = free_at - ent
                snp = min(cands, key=lambda c: abs(c[1] - elapsed))
                if snp[3] > friction and (best_lj is None or snp[3] > best_lj[1][3]):
                    best_lj = (L, snp)
        if best_lj is not None:
            L, snp = best_lj
            total += snp[2] - friction
            taken.add(L[0])
            free_at = L[2]
            n_lj += 1
            lj_recs.append((snp[0], snp[2]))          # (K, remaining_pnl)
        elif future:
            L = min(future, key=lambda x: x[1])
            total += L[3] - friction
            taken.add(L[0])
            free_at = L[2]
            n_nat += 1
        else:
            break
    return total, n_nat, n_lj, lj_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oos', default=f'{RD}/trade_trajectory_OOS_full.parquet')
    ap.add_argument('--out', default=f'{RD}/2026-05-20_midleg_constrained_sim.txt')
    args = ap.parse_args()

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    traj = pd.read_parquet(args.oos)
    traj['remaining_pnl_usd'] = traj['exit_pnl_usd'] - traj['pnl_usd_so_far']

    # B9 predictions per (leg_id, K)
    lookup = {}
    for K in K_HORIZONS:
        tK = traj[traj['K'] == K].reset_index(drop=True)
        b9 = load_b9(K)
        pred = b9['model'].predict(tK[b9['feat_cols']].fillna(0.0).values)
        for lid, rem, p in zip(tK['leg_id'].values,
                               tK['remaining_pnl_usd'].values, pred):
            lookup.setdefault(int(lid), []).append((K, K * 5, float(rem), float(p)))

    # Leg universe = one row per leg (K=5 rows carry entry/exit/pnl/day)
    legs5 = traj[traj['K'] == 5]
    legs_by_day = {}
    for r in legs5.itertuples(index=False):
        legs_by_day.setdefault(r.day, []).append(
            (int(r.leg_id), int(r.entry_ts), int(r.exit_ts), float(r.exit_pnl_usd)))
    days = sorted(legs_by_day)
    n_days = len(days)

    out('=' * 84)
    out('MID-LEG ENTRY  --  E3: position-constrained 1-contract late-join sim')
    out('=' * 84)
    out(f'OOS {len(legs5):,} legs over {n_days} sealed days  '
        f'(~{len(legs5)/n_days:.0f} legs/day, overlapping)')
    out(f'Friction ${FRICTION_USD_PER_LEG:.0f}/leg charged on EVERY entry '
        f'(natural + late-join), both sims.')
    out('B9 gate: late-join iff B9 predicted remaining > friction.')
    out('')

    # Baseline
    base_pd = np.array([simulate(legs_by_day[d], lookup, False, 120,
                                 FRICTION_USD_PER_LEG)[0] for d in days])
    base_nat = sum(simulate(legs_by_day[d], lookup, False, 120,
                            FRICTION_USD_PER_LEG)[1] for d in days)
    blo, bhi = bootstrap_ci(base_pd)
    out('-' * 84)
    out('BASELINE  (greedy 1-contract, no late-join)')
    out(f'  ${base_pd.mean():+.0f}/day  CI [${blo:+.0f}, ${bhi:+.0f}]   '
        f'{base_nat:,} natural legs taken ({base_nat/n_days:.1f}/day)')
    out('  (net of friction; memory FLAT-1c reference ~$454/day)')
    out('')

    # Late-join variants
    out('-' * 84)
    out('LATE-JOIN  (B9-gated; max_k = oldest missed leg the engine will join)')
    out('-' * 84)
    verdict = []
    for max_k in (120, 30, 10):
        lj_pd = np.zeros(n_days)
        tot_nat = tot_lj = 0
        all_recs = []
        for i, d in enumerate(days):
            tot, n_nat, n_lj, recs = simulate(
                legs_by_day[d], lookup, True, max_k, FRICTION_USD_PER_LEG)
            lj_pd[i] = tot
            tot_nat += n_nat
            tot_lj += n_lj
            all_recs.extend(recs)
        incr = lj_pd - base_pd
        ilo, ihi = bootstrap_ci(incr)
        itag = 'SIG +' if ilo > 0 else ('SIG -' if ihi < 0 else 'not sig')
        out(f'max_k={max_k:>3}:  total ${lj_pd.mean():+.0f}/day   '
            f'incremental ${incr.mean():+.0f}/day  CI [${ilo:+.0f}, ${ihi:+.0f}]  {itag}')
        out(f'           {tot_nat:,} natural + {tot_lj:,} late-joins  '
            f'({tot_lj/n_days:.1f} late-joins/day; '
            f'{base_nat - tot_nat:+,} natural displaced)')
        if all_recs:
            kc = pd.Series([r[0] for r in all_recs]).value_counts().sort_index()
            kdist = '  '.join(f'K{k}:{int(v)}' for k, v in kc.items())
            rem = np.array([r[1] for r in all_recs])
            out(f'           late-join K-distribution: {kdist}')
            out(f'           realised remaining_pnl: mean ${rem.mean():+.0f}  '
                f'median ${np.median(rem):+.0f}  frac>0 {(rem>0).mean()*100:.0f}%')
        out('')
        verdict.append((max_k, incr.mean(), ilo, ihi))

    out('=' * 84)
    out('VERDICT  --  E3 position-constrained late-join')
    out('=' * 84)
    any_sig = False
    for max_k, mean, lo, hi in verdict:
        if lo > 0:
            any_sig = True
            t = f'SIGNIFICANT  +${mean:.0f}/day'
        elif hi < 0:
            t = f'SIGNIFICANTLY NEGATIVE  ${mean:.0f}/day'
        else:
            t = f'not significant  (${mean:+.0f}/day)'
        out(f'  max_k={max_k:>3}  CI [${lo:+.0f}, ${hi:+.0f}]  {t}')
    out('')
    if any_sig:
        out('>> Position-constrained late-join adds a significant amount. This is')
        out('   a deployable result -- wire B9-gated late-join into the engine.')
    else:
        out('>> Under the 1-contract constraint, late-join does NOT add a')
        out('   significant amount. The unconstrained E1 ceiling does not')
        out('   survive contact with the position constraint: the engine goes')
        out('   flat too late to catch fresh (K<=10) missed legs, and stale')
        out('   late-joins are noise. Mid-leg entry is not worth deploying.')

    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
