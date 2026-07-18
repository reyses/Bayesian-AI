"""
NMP9 quantile-match SOLVER (doc 102, Opus drone). Reads the 2024 probe table and solves
the RETUNE thresholds by MARGINAL PASS-RATE ONLY -- no AI-label, no PnL contact (the
quantile-cell-overfit trap, MEMORY 3). The validated method (Z_ENTRY 2.0->1.8481, 2026-06-11):
the rolling windows changed => z / wick / h1z distributions changed => the verbatim 2026-04
thresholds sit on shifted quantiles. We find, per threshold, the value on the CURRENT
estimator whose 2024 marginal pass-rate reproduces the ERA occupancy target, solving in
WATERFALL order (base gate -> wick pair -> 1h gates). HOLD: vr<1.0, velocity 50/100 ticks.

Era anchors (journals 2026-04-06 / 04-08, documented per-threshold in the JSON):
  base ROCHE          -> total entry-universe rate  ~ 9,277 / 277d ~ 33 boundaries/day  (04-08)
  wick pair           -> has_wick (KILL_SHOT+CASCADE) entry rate ~ 2.5 /day             (04-06)
  H1_Z_MIN (cascade)  -> cascade / has_wick occupancy ratio ~ 70 / 486 = 0.1440         (04-06)
  H1_AGAINST_Z_MIN    -> |h1z|>1.5 aligned-tail ratio         ~ 29 / 486 = 0.0597        (04-06)
                         (same h1z estimator as H1_Z_MIN; the derived threshold is also
                          applied to the h1_vel AGAINST gate, verbatim to the era's single
                          constant -- negligible effect there, |h1vel| median ~11 ticks.)

Run:  python3.11 research/nt8_catalog/tools/nmp9_quantile_match.py
Out:  reports/nmp9_retuned_constants.json  (+ console diagnostics)
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.abspath(os.path.join(HERE, '..', 'reports'))
PROBE = os.path.join(REP, 'nmp9_probe_2024.parquet')

# verbatim 2026-04-08 constants (the BEFORE / reproducible baseline)
VERBATIM = dict(ROCHE=2.0, VR_ENTRY=1.0, WICK_5M_MIN=0.83, WICK_15M_MIN=0.77,
                VELOCITY_THRESHOLD=50.0, FREIGHT_TRAIN_THRESHOLD=100.0,
                H1_Z_MIN=1.0, H1_AGAINST_Z_MIN=1.5)
# era occupancy anchors
T_BASE_PER_DAY = 9277 / 277.0        # ~33.5 boundaries/day (entry universe)
T_WICK_PER_DAY = 2.5                 # has_wick entry rate/day (04-06 kill-shot 2.5-2.8)
R_CASCADE = 70 / 486.0              # cascade / has_wick occupancy ratio (04-06 ladder)
R_H1_15 = 29 / 486.0               # |h1z|>1.5 aligned-tail ratio (04-06 ladder)
WICK_GAP = 0.83 - 0.77             # preserve the era 5m-15m stringency gap (additive shift)


def load():
    F = pd.read_parquet(PROBE).sort_values(['day', 'ts']).reset_index(drop=True)
    F['dir_long'] = F['z'] < 0                 # NMP default: fade z (short if z>0)
    return F, F['day'].nunique()


def rate_per_day(mask, ND):
    return float(np.asarray(mask).sum()) / ND


def bisect(f, lo, hi, target, tol=1e-3, iters=60):
    """Solve f(x)=target for MONOTONE f on [lo,hi] by bisection (no derivative)."""
    flo, fhi = f(lo), f(hi)
    inc = fhi > flo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm - target) < tol:
            return mid
        if (fm < target) == inc:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    F, ND = load()
    print(f'2024 probe: {len(F)} boundaries, {ND} days, {len(F)/ND:.1f} bnd/day\n')
    z = F['z'].values
    vr = F['vr'].values
    w5, w15 = F['wick5'].values, F['wick15'].values
    h1z = F['h1z'].values
    dl = F['dir_long'].values                  # direction == long
    fin_h1 = np.isfinite(h1z)

    # ---- 1. BASE GATE: solve ROCHE so |z|>ROCHE & vr<1.0 -> ~33 boundaries/day ---------
    vr_ok = np.isfinite(vr) & (vr < VERBATIM['VR_ENTRY'])   # vr<1.0 HELD (regime boundary)

    def base_rate(roche):
        return rate_per_day((np.abs(z) > roche) & vr_ok, ND)

    r0 = base_rate(VERBATIM['ROCHE'])
    ROCHE = bisect(base_rate, 1.0, 3.0, T_BASE_PER_DAY, tol=0.05)
    uni = (np.abs(z) > ROCHE) & vr_ok
    print(f'[1] BASE  ROCHE {VERBATIM["ROCHE"]:.4f} -> {ROCHE:.4f}  '
          f'(2024 universe {r0:.2f} -> {base_rate(ROCHE):.2f}/day; target {T_BASE_PER_DAY:.2f})')

    # ---- 2. WICK PAIR: additive shift d so has_wick within universe -> ~2.5/day --------
    def wick_rate(d):
        hw = uni & (w5 > VERBATIM['WICK_5M_MIN'] + d) & (w15 > VERBATIM['WICK_15M_MIN'] + d)
        return rate_per_day(hw, ND)

    w0 = wick_rate(0.0)
    dW = bisect(wick_rate, -0.60, 0.30, T_WICK_PER_DAY, tol=0.01)
    W5, W15 = VERBATIM['WICK_5M_MIN'] + dW, VERBATIM['WICK_15M_MIN'] + dW
    hw = uni & (w5 > W5) & (w15 > W15)
    print(f'[2] WICK  5m {VERBATIM["WICK_5M_MIN"]:.4f}->{W5:.4f}  15m {VERBATIM["WICK_15M_MIN"]:.4f}'
          f'->{W15:.4f}  (shift {dW:+.4f}; has_wick {w0:.2f}->{wick_rate(dW):.2f}/day; target {T_WICK_PER_DAY})')

    # ---- 3. H1_Z_MIN: signed-aligned |h1z|>t = fraction R_CASCADE of has_wick ----------
    # aligned: sign(h1z)==sign(z) and |h1z|>t. dir_long => z<0 => aligned if h1z<-t.
    def aligned_frac(t):
        al = fin_h1 & (((dl) & (h1z < -t)) | ((~dl) & (h1z > t)))
        return float((hw & al).sum()) / max(hw.sum(), 1)

    f0 = aligned_frac(VERBATIM['H1_Z_MIN'])
    H1_Z_MIN = bisect(aligned_frac, 0.2, 5.0, R_CASCADE, tol=1e-3)
    print(f'[3] H1_Z  {VERBATIM["H1_Z_MIN"]:.4f} -> {H1_Z_MIN:.4f}  '
          f'(cascade/has_wick {f0:.4f} -> {aligned_frac(H1_Z_MIN):.4f}; target {R_CASCADE:.4f})')

    # ---- 4. H1_AGAINST_Z_MIN: same h1z estimator, |h1z|>t aligned-tail = R_H1_15 -------
    # calibrated on the aligned-tail (symmetric); applied to h1z (FADEAGAINST) & h1vel (RIDEAGAINST).
    g0 = aligned_frac(VERBATIM['H1_AGAINST_Z_MIN'])
    H1_AGAINST = bisect(aligned_frac, H1_Z_MIN, 6.0, R_H1_15, tol=1e-3)
    print(f'[4] H1_AG {VERBATIM["H1_AGAINST_Z_MIN"]:.4f} -> {H1_AGAINST:.4f}  '
          f'(|h1z|-tail {g0:.4f} -> {aligned_frac(H1_AGAINST):.4f}; target {R_H1_15:.4f})')

    RETUNED = dict(ROCHE=round(ROCHE, 4), VR_ENTRY=1.0,
                   WICK_5M_MIN=round(W5, 4), WICK_15M_MIN=round(W15, 4),
                   VELOCITY_THRESHOLD=50.0, FREIGHT_TRAIN_THRESHOLD=100.0,
                   H1_Z_MIN=round(H1_Z_MIN, 4), H1_AGAINST_Z_MIN=round(H1_AGAINST, 4))

    out = {
        'method': 'quantile-match on 2024 marginal pass-rate; no AI-label / no PnL in loop',
        'probe': dict(rows=len(F), days=int(ND), bnd_per_day=round(len(F)/ND, 2)),
        'anchors': {
            'ROCHE': dict(target='entry-universe boundaries/day', value=round(T_BASE_PER_DAY, 3),
                          source='journal 2026-04-08 (9,277 phase-1 trades / 277 IS days)'),
            'WICK_pair': dict(target='has_wick entry/day', value=T_WICK_PER_DAY,
                              source='journal 2026-04-06 (kill-shot 2.5-2.8 tr/day)',
                              note=f'additive shift d preserves era 5m-15m gap {WICK_GAP:.2f}'),
            'H1_Z_MIN': dict(target='cascade/has_wick occupancy ratio', value=round(R_CASCADE, 4),
                             source='journal 2026-04-06 (ladder 486->70 at |h1z|>1.0 aligned)'),
            'H1_AGAINST_Z_MIN': dict(target='|h1z|>1.5 aligned-tail ratio', value=round(R_H1_15, 4),
                                     source='journal 2026-04-06 (ladder 486->29 at |h1z|>1.5)',
                                     note='same h1z estimator; also applied to h1_vel gate verbatim'),
        },
        'held_absolute': dict(VR_ENTRY=1.0, VELOCITY_THRESHOLD=50.0, FREIGHT_TRAIN_THRESHOLD=100.0,
                              reason='vr<1.0 = regime boundary; velocity in ticks = verbatim formula'),
        'verbatim': VERBATIM,
        'retuned': RETUNED,
        'diagnostics_2024': {
            'base_rate_before': round(r0, 3), 'base_rate_after': round(base_rate(ROCHE), 3),
            'wick_rate_before': round(w0, 3), 'wick_rate_after': round(wick_rate(dW), 3),
            'cascade_frac_before': round(f0, 4), 'cascade_frac_after': round(aligned_frac(H1_Z_MIN), 4),
            'h1against_tail_before': round(g0, 4), 'h1against_tail_after': round(aligned_frac(H1_AGAINST), 4),
        },
    }
    dest = os.path.join(REP, 'nmp9_retuned_constants.json')
    with open(dest, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {dest}')
    print('VERBATIM', VERBATIM)
    print('RETUNED ', RETUNED)


if __name__ == '__main__':
    main()
