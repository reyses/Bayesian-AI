#!/usr/bin/env python
"""Score the two-stage exit on ITS OWN objective: not losing money.

THE CORRECTION (owner, 2026-08-02): "the strategy is designed not to lose money
(the exit part of it) since you're waiting for a MFE retrace of 20%, and then
exit if it continues for 10% more, ratcheting up if new highs are encountered."

Every evaluation today scored this exit on EV. That is the mistake recorded in
the standing memory `feedback-score-the-stated-objective` — and it is the same
arc as the BE+2 stop, which was "dead" on EV and then measured −46% volatility
and −24% drawdown at zero EV cost when scored as the risk control it was.

THE DESIGN GUARANTEE UNDER TEST: once a trade is ARMED (peak profit > ARM_PT),
the frozen-70% exit bounds the giveback — an armed trade should essentially
never become a loss, let alone a −20pt stop-out. Gap-through of the floor is
the only leak; it is measured, not assumed away.

Three exits on IDENTICAL entries (paired), all with the same 20pt hard stop and
honest fills (ratchet exits at the close AFTER the breach — booking the floor
itself fabricated +1.44pt/trade in an earlier build):
  BAND      exit when z reaches the opposite band (the standing benchmark)
  RATCH80   continuous ratchet: exit when profit <= 80% of the running peak
  TWO8070   the owner's: warn at 80% (freeze the peak), exit at 70% of the
            FROZEN peak, a new extreme releases the freeze

Metrics per exit: mean [95% CI], std, %losers, p05, p01, CVaR5, worst trade,
max drawdown of the chronological equity curve — plus armed%, P(loss|armed),
and worst armed trade. Paired bootstrap CIs on the deltas that matter.

PRE-COMMITTED VERDICT v2 (amended BEFORE any result was read — the v1 run was
killed and its report deleted unread; adversarial verification caught (a) a
SIGN INVERSION in v1's verdict arithmetic that would have printed the opposite
of the truth, and (b) that v1's "mean CI includes 0" rewarded low power and
accepted the benefit on a point estimate while excusing the cost — a double
standard):
  - VALIDATED FAIL-SAFE: CVaR5 delta CI (day-block bootstrap) excludes 0 in
    the improving direction AND point improvement >= 20% AND the EV cost is
    BOUNDED, not merely undetected: mean-delta lower CI > -0.25pt (equivalence
    bound), all on day-block resamples.
  - PROTECTION AT A PRICE: significant >=20% tail cut, EV cost breaches the
    equivalence bound -> report the price with its CI.
  - SUGGESTIVE ONLY: point tail cut >=20% but its CI includes 0.
  - FAILS ON ITS OWN AXIS: point tail cut < 20%.
Also per verification: day-block bootstrap (i.i.d. trade resampling understates
CI width under day clustering and overlapping holds), P(loss|armed) reported by
CUSHION SIZE (the floor at peak=2pt is paper against a 20pt/5s bar; at peak=30pt
it is armor — pooling them answers the wrong question), counts alongside every
conditional probability, and the full-stop threshold friction-corrected.

Writes to research/dojo_forge/reports/.
Usage: python research/dojo_forge/tools/exit_risk_profile.py --days 700
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'research', 'dojo_forge', 'tools'))
import cubic_regression as _cub                                    # noqa: E402

D5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports',
                   'exit_risk_profile.md')

CUBIC_5S_WINDOW = 90
SIGMA_MIN = 20
BANDS = (1.5, 3.0)          # the standing benchmark scale + the owner's scale
HARD_STOP = 20.0
MAX_HOLD_S = 1800
ARM_PT = 2.0
WARN, EXIT_F = 0.80, 0.70
ARM_REGION_FRAC = 0.85   # PROTECT PROTOCOL: "nearing the expected region" =
                         # favourable excursion has covered >=85% of the
                         # entry-to-opposite-band distance (mirrors the 15%
                         # region proximity used by the live `region` tool)
RTH_FROM, RTH_TO = 570, 960
FRICTION_PT = 0.89
PT_USD = 2.0
BOOT = 4000
SEED = 11


def scan_day(day, BAND):
    d = pd.read_parquet(os.path.join(D5, f'{day}.parquet'))[
        ['timestamp', 'high', 'low', 'close']]
    if len(d) < 2000:
        return []
    ts = d['timestamp'].to_numpy(); c = d['close'].to_numpy()
    hi = d['high'].to_numpy(); lo = d['low'].to_numpy()
    cub, _, _ = _cub.rolling(c, CUBIC_5S_WINDOW, 5)
    res = c - cub
    sig = pd.Series(res).rolling(SIGMA_MIN * 12, min_periods=5 * 12).std().to_numpy()
    z = np.where(sig > 0, res / sig, np.nan)
    e = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
    m = (e.hour * 60 + e.minute).to_numpy()
    side = np.where(z >= BAND, 1, np.where(z <= -BAND, -1, 0))
    ff = pd.Series(np.where(side == 0, np.nan, side)).ffill().to_numpy()
    flip = np.flatnonzero((~np.isnan(ff[1:])) & (~np.isnan(ff[:-1]))
                          & (ff[1:] != ff[:-1])) + 1
    flip = flip[(m[flip] >= RTH_FROM) & (m[flip] < RTH_TO)]

    rows = []
    for i in flip:
        sgn = 1 if int(ff[i]) < 0 else -1
        p0 = float(c[i])
        w = (ts > ts[i]) & (ts <= ts[i] + MAX_HOLD_S)
        if w.sum() < 24:
            continue
        hh, ll, cc, zz = hi[w], lo[w], c[w], z[w]
        nb = len(cc)
        fav = (hh - p0) if sgn > 0 else (p0 - ll)
        adv = (p0 - ll) if sgn > 0 else (hh - p0)

        # BAND exit — opposite band or hard stop, else last close
        tgt = np.flatnonzero(zz >= BAND) if sgn > 0 else np.flatnonzero(zz <= -BAND)
        stp = np.flatnonzero(adv >= HARD_STOP)
        jt = tgt[0] if len(tgt) else None
        js = stp[0] if len(stp) else None
        if jt is not None and (js is None or jt < js):
            band_out = float((cc[jt] - p0) * sgn)
        elif js is not None:
            band_out = -HARD_STOP
        else:
            band_out = float((cc[-1] - p0) * sgn)

        # RATCH80 (continuous) and TWO8070 (frozen), both honest fills.
        # Stop is checked FIRST in each bar (intrabar order unknowable at 5s;
        # conservative), consistent with every prior test today.
        # arm_dist: PROTECT-PROTOCOL mode (owner, canonical 2026-08-02) — the
        # protection arms only once the favourable excursion has carried price
        # NEAR the expected region (>= arm_dist). Before that only the hard
        # stop exists. Entry-armed ratchets (arm_dist=None) mis-state his
        # design: they spend most of their life at paper-thin cushions.
        def run_ratchet(two_stage, arm_dist=None):
            pk, frozen, armed, out = 0.0, None, False, None
            for j in range(nb):
                if adv[j] >= HARD_STOP:
                    out = -HARD_STOP
                    break
                cur = (cc[j] - p0) * sgn
                if fav[j] > pk:
                    pk = fav[j]
                    frozen = None                     # new extreme releases
                if arm_dist is not None and not armed and pk >= arm_dist:
                    armed = True                      # reached the arm zone
                active = (pk > ARM_PT) if arm_dist is None else armed
                if active:
                    if arm_dist is None:
                        armed = True
                    if two_stage:
                        if frozen is None:
                            if cur <= pk * WARN:
                                frozen = pk           # register the MFE
                        if frozen is not None and cur <= frozen * EXIT_F:
                            out = cur                 # honest: the close
                            break
                    else:
                        if cur <= pk * WARN:
                            out = cur
                            break
            if out is None:
                out = float((cc[-1] - p0) * sgn)
            return out, armed, pk

        r80, armed_r, _ = run_ratchet(False)
        t87, armed_t, pk_t = run_ratchet(True)
        # expected-region distance at entry: entry price to the OPPOSITE band
        dist_opp = ((cub[i] + BAND * sig[i]) - p0) if sgn > 0 \
            else (p0 - (cub[i] - BAND * sig[i]))
        dist_opp = max(float(dist_opp), 1.0)
        tR, armed_R, pk_R = run_ratchet(True, arm_dist=ARM_REGION_FRAC * dist_opp)
        rows.append(dict(day=day, ts=int(ts[i]),
                         band=band_out - FRICTION_PT,
                         r80=r80 - FRICTION_PT,
                         two=t87 - FRICTION_PT,
                         twoR=tR - FRICTION_PT,
                         armed=bool(armed_t), peak=float(pk_t),
                         armedR=bool(armed_R), peakR=float(pk_R)))
    return rows


def maxdd(x):
    eq = np.cumsum(x)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(x) else 0.0


def risk_row(x):
    x = np.asarray(x, float)
    xs = np.sort(x)
    k = max(1, int(0.05 * len(xs)))
    return dict(mean=x.mean(), std=x.std(ddof=1), losers=(x < 0).mean(),
                p05=np.percentile(x, 5), p01=np.percentile(x, 1),
                cvar5=xs[:k].mean(), worst=xs[0], dd=maxdd(x))


def boot_mean_ci(x):
    rng = np.random.default_rng(SEED)
    s = [rng.choice(x, len(x), replace=True).mean() for _ in range(BOOT)]
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def block_deltas(day_ids, a, b):
    """Paired deltas fn(TWO)−fn(BAND) under a DAY-BLOCK bootstrap.

    Verifier finding: i.i.d. per-trade resampling understates CI width — trades
    cluster within days and holding windows overlap, so days are the honest
    exchangeable unit. Days are resampled with replacement; every delta is
    computed on the same resample, preserving the pairing."""
    uniq = np.unique(day_ids)
    idx_by_day = [np.flatnonzero(day_ids == d) for d in uniq]
    rng = np.random.default_rng(SEED)
    acc = {k: [] for k in ('mean', 'std', 'cvar5', 'losers')}
    nd = len(idx_by_day)
    for _ in range(BOOT):
        pick = rng.integers(0, nd, nd)
        idx = np.concatenate([idx_by_day[p] for p in pick])
        va, vb = a[idx], b[idx]
        k5 = max(1, int(0.05 * len(idx)))
        acc['mean'].append(vb.mean() - va.mean())
        acc['std'].append(vb.std(ddof=1) - va.std(ddof=1))
        acc['cvar5'].append(np.sort(vb)[:k5].mean() - np.sort(va)[:k5].mean())
        acc['losers'].append((vb < 0).mean() - (va < 0).mean())
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5))) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=700)
    ap.add_argument('--exclude', nargs='*', default=['2024_09_16'])
    a = ap.parse_args()
    days = sorted(f[:-8] for f in os.listdir(D5) if f.endswith('.parquet')
                  and f[:-8] not in a.exclude)
    rng = np.random.default_rng(SEED)
    if len(days) > a.days:
        days = sorted(rng.choice(days, a.days, replace=False).tolist())

    L = ['# The two-stage exit scored on its OWN objective: risk', '',
         'Owner: the exit is **designed not to lose money** — warn at a 20% '
         'retrace of MFE, exit 10% further, ratchet on new highs. Every prior '
         'evaluation scored it on EV; this scores the left tail. Precedent: '
         'the BE+2 stop was "dead" on EV and then measured **−46% vol / −24% '
         'DD at zero EV cost** when finally scored as the risk control it was.',
         '',
         f'Identical entries, paired. Honest fills, {HARD_STOP:g}pt hard stop, '
         f'friction {FRICTION_PT}pt, arm above {ARM_PT:g}pt. '
         f'Sessions: **{len(days)}**. Excluded: {", ".join(a.exclude)}.', '']

    for B in BANDS:
        rows = []
        skipped = []                        # verifier: silent day-drops can
        for d in tqdm(days, desc=f'band {B}'):   # hide exactly the tail days
            try:
                rows += scan_day(d, B)
            except Exception:
                skipped.append(d)
        df = pd.DataFrame(rows).sort_values(['day', 'ts'])
        if df.empty:
            continue
        band = df['band'].to_numpy(); r80 = df['r80'].to_numpy()
        two = df['two'].to_numpy(); armed = df['armed'].to_numpy()
        nses = df['day'].nunique()

        L += [f'## Entries at ±{B:g}σ — {len(df):,} trades, {nses} sessions, '
              f'{len(df) / nses:.1f}/session'
              + (f' · **{len(skipped)} days skipped on error: '
                 f'{", ".join(skipped[:5])}{"…" if len(skipped) > 5 else ""}**'
                 if skipped else ''), '',
              '| exit | mean | 95% CI | std | %losers | p05 | p01 | CVaR5 | '
              'worst | maxDD (pt) |',
              '|---|---|---|---|---|---|---|---|---|---|']
        twoR = df['twoR'].to_numpy()
        armedR = df['armedR'].to_numpy()
        for name, x in (('BAND', band), ('RATCH-80', r80),
                        ('**TWO-STAGE 80/70 (entry-armed)**', two),
                        ('**PROTECT PROTOCOL (region-armed)**', twoR)):
            r = risk_row(x)
            lo, hi = boot_mean_ci(x)
            L.append(f'| {name} | `{r["mean"]:+.2f}` | `[{lo:+.2f}, {hi:+.2f}]` | '
                     f'{r["std"]:.2f} | {r["losers"]:.1%} | `{r["p05"]:+.2f}` | '
                     f'`{r["p01"]:+.2f}` | `{r["cvar5"]:+.2f}` | '
                     f'`{r["worst"]:+.2f}` | {r["dd"]:,.0f} |')

        # paired deltas TWO vs BAND, day-block bootstrap
        D = block_deltas(df['day'].to_numpy(), band, two)
        dm, dmlo, dmhi = D['mean']
        ds, dslo, dshi = D['std']
        dc, dclo, dchi = D['cvar5']
        dl, dllo, dlhi = D['losers']
        rB, rT = risk_row(band), risk_row(two)
        # SIGN: dc = CVaR5(TWO) − CVaR5(BAND); CVaR is negative, so dc > 0 is
        # an IMPROVEMENT. v1 negated this and would have printed the opposite
        # verdict — caught by the methodology auditor before any result was
        # read.
        cvar_impr = dc / abs(rB['cvar5'])
        full_stop = -(HARD_STOP + FRICTION_PT) + 1e-6   # nets are post-friction
        L += ['', 'Paired deltas, TWO-STAGE − BAND (day-block bootstrap, '
                  '4,000 resamples):',
              f'- mean: `{dm:+.3f}` CI `[{dmlo:+.3f}, {dmhi:+.3f}]`',
              f'- std: `{ds:+.2f}` CI `[{dslo:+.2f}, {dshi:+.2f}]` '
              f'({ds / rB["std"] * 100:+.0f}% of BAND std)',
              f'- CVaR5: `{dc:+.2f}` CI `[{dclo:+.2f}, {dchi:+.2f}]` — '
              f'improvement `{cvar_impr:+.0%}` of BAND CVaR5 `{rB["cvar5"]:+.2f}`',
              f'- %losers: `{dl:+.3f}` CI `[{dllo:+.3f}, {dlhi:+.3f}]`',
              f'- maxDD: BAND {rB["dd"]:,.0f}pt → TWO {rT["dd"]:,.0f}pt '
              f'(`{(rT["dd"] / rB["dd"] - 1) * 100:+.0f}%`)',
              f'- full −{HARD_STOP:g}pt stop-outs: BAND '
              f'{(band <= full_stop).mean():.1%} → TWO '
              f'{(two <= full_stop).mean():.1%}', '',
              '### The design guarantee, by cushion size', '',
              'The floor at peak=2pt is paper against a 20pt/5s bar; at '
              'peak=30pt it is armor. Pooling them answers the wrong question '
              '(blind-reimplementation finding: pooled P(loss|armed) ≈ 27% is '
              'dominated by tiny cushions).', '',
              '| peak at exit | N | P(loss) | mean net | worst |',
              '|---|---|---|---|---|']
        pk_arr = df['peak'].to_numpy()
        for lo_b, hi_b, lab in ((ARM_PT, 5.0, f'{ARM_PT:g}–5pt'),
                                (5.0, 10.0, '5–10pt'),
                                (10.0, 20.0, '10–20pt'),
                                (20.0, 1e18, '≥20pt')):
            msk = (pk_arr > lo_b) & (pk_arr <= hi_b)
            if not msk.any():
                continue
            xx = two[msk]
            L.append(f'| {lab} | {int(msk.sum()):,} | '
                     f'{(xx < 0).mean():.1%} ({int((xx < 0).sum())}/'
                     f'{int(msk.sum())}) | `{xx.mean():+.2f}` | '
                     f'`{xx.min():+.2f}` |')
        L.append('')
        if armed.any():
            aw = two[armed]
            un = two[~armed]
            L += [f'- armed overall (peak > {ARM_PT:g}pt): **{armed.mean():.1%}** · '
                  f'P(loss|armed) **{(aw < 0).mean():.2%}** '
                  f'({int((aw < 0).sum())}/{int(armed.sum())})',
                  f'- worst armed `{aw.min():+.2f}pt` · worst unarmed '
                  + (f'`{un.min():+.2f}pt`' if len(un) else '(none unarmed)'),
                  '']

        # PROTECT PROTOCOL (region-armed) — the owner's canonical design.
        # Pre-registered expectation from the cushion curve: P(loss |
        # region-armed) in the low single digits, since arming near the
        # region implies a near-full-traverse cushion by construction.
        if armedR.any():
            aR = twoR[armedR]
            bR = band[armedR]
            DR = block_deltas(df['day'].to_numpy()[armedR], bR, aR)
            drm, drmlo, drmhi = DR['mean']
            L += ['### PROTECT PROTOCOL — the guarantee where it actually arms',
                  '',
                  f'- reached the arm zone (≥{ARM_REGION_FRAC:.0%} of the '
                  f'entry→opposite-band distance): **{armedR.mean():.1%}** of '
                  f'trades ({int(armedR.sum()):,})',
                  f'- cushion at exit among armed: mean '
                  f'`{df["peakR"].to_numpy()[armedR].mean():.1f}pt`',
                  f'- **P(loss | region-armed): '
                  f'{(aR < 0).mean():.2%}** ({int((aR < 0).sum())}/'
                  f'{int(armedR.sum())}) · mean `{aR.mean():+.2f}pt` · worst '
                  f'`{aR.min():+.2f}pt`',
                  f'- paired vs BAND on the same armed subset: mean delta '
                  f'`{drm:+.2f}` CI `[{drmlo:+.2f}, {drmhi:+.2f}]`',
                  f'- never-armed complement (fail-safe stop territory): '
                  f'{(~armedR).mean():.1%} of trades, mean '
                  f'`{twoR[~armedR].mean():+.2f}pt`' if (~armedR).any() else '',
                  '']

        # PRE-COMMITTED VERDICT v2 (see docstring — amended before any result
        # was read; benefit must be significant, cost must be BOUNDED)
        EQUIV = 0.25
        if dclo > 0 and cvar_impr >= 0.20 and dmlo > -EQUIV:
            L.append('**VERDICT: validated fail-safe.** Tail cut ≥20% and '
                     'significant (day-block CI excludes 0); EV cost bounded '
                     f'within {EQUIV}pt — the BE+2 pattern, reproduced on the '
                     'exit side.')
        elif dclo > 0 and cvar_impr >= 0.20:
            L.append(f'**VERDICT: protection at a price.** Tail cut '
                     f'{cvar_impr:.0%} (significant); EV cost `{dm:+.2f}pt` '
                     f'CI `[{dmlo:+.2f}, {dmhi:+.2f}]` breaches the '
                     f'±{EQUIV}pt equivalence bound.')
        elif cvar_impr >= 0.20:
            L.append(f'**VERDICT: suggestive, not established.** Point tail '
                     f'cut {cvar_impr:.0%} but its CI `[{dclo:+.2f}, '
                     f'{dchi:+.2f}]` includes 0.')
        else:
            L.append(f'**VERDICT: the design claim fails on its own axis** — '
                     f'tail improvement `{cvar_impr:+.0%}` < 20%.')
        L.append('')

    L += ['## Scope note', '',
          'A validated fail-safe does not create expectancy: on a losing entry '
          'stream it makes you lose *less badly*. Its value is as the safety '
          'layer under the owner\'s SELECTIVE entries — which remain the open '
          'question (corpus, not compute).', '']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
