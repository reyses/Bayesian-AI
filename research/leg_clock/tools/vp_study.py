"""Volume-at-price study (Moises' pick, 2026-07-08, autonomous overnight).

Everything killed so far was volume over TIME. This measures volume AT PRICE:
the causal session profile (POC, 70% Value Area, high/low-volume nodes),
accumulated 5s-bar by 5s-bar (bar volume spread uniformly over its range —
standard approximation without tick data; profile spans the full session file
including Globex, like a session-anchored chart VP).

Test A (wall thesis): do labeled turn-entries sit at HVNs / VA edges more than
same-hour matched nulls? Per-feature separation + house signal bar.
Test B (niche gate): record VP context at every niche trade entry
(fade/T20/9-13CT/q0.995); split PnL by VP buckets. Both years; a gate is
accepted only if it helps BOTH.
"""
import glob
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core_v2.FPS.forward_pass_system import ForwardPassSystem  # noqa: E402
from nt8_deploy_test import (fit_logistic, score, train_2024, IDX, ZH, ZL,  # noqa: E402
                             TICK, TICK_VALUE, COST_TICKS, ATLAS)

BUCKET = 1.0          # profile bucket, points (4 ticks)
REPORT_DIR = os.path.join(_REPO, 'research', 'leg_clock', 'reports')
lines = []


def log(s):
    print(s, flush=True); lines.append(s)


def day_pass(day, model, th, rng):
    """One FPS pass: 1m scores + causal VP features + niche trades + label/null rows."""
    fps = ForwardPassSystem(day=day, atlas_root=ATLAS,
                            features_root=os.path.join(ATLAS, 'FEATURES_5s_v2'),
                            labels_csv=os.path.join(ATLAS, 'regime_labels_2d.csv'),
                            build_v2_dict=False)
    ts_m, F, px_m, vp_rows = [], [], [], []
    ts5, px5 = [], []
    hist = {}
    tot_vol = 0.0

    def vp_snapshot(price):
        nonlocal hist, tot_vol
        if not hist or tot_vol <= 0:
            return None
        keys = np.array(sorted(hist))
        vols = np.array([hist[k] for k in keys])
        poc = keys[vols.argmax()] * BUCKET
        order = vols.argsort()[::-1]
        cum = vols[order].cumsum()
        va_keys = keys[order[:int(np.searchsorted(cum, 0.7 * tot_vol)) + 1]]
        va_lo, va_hi = va_keys.min() * BUCKET, va_keys.max() * BUCKET
        b = int(price // BUCKET)
        node = hist.get(b, 0.0)
        med = np.median(vols[vols > 0])
        return dict(dist_poc=abs(price - poc) / TICK,
                    in_va=va_lo <= price <= va_hi,
                    above_va=price > va_hi,
                    node_ratio=node / med if med > 0 else 0.0,
                    va_width=(va_hi - va_lo) / TICK)

    for bar in fps:
        if bar.v2_vector is None:
            continue
        o5 = bar.ohlcv_5s
        lo_, hi_, v = o5['low'], o5['high'], o5['volume']
        b0, b1 = int(lo_ // BUCKET), int(hi_ // BUCKET)
        nb = max(1, b1 - b0 + 1)
        for b in range(b0, b1 + 1):
            hist[b] = hist.get(b, 0.0) + v / nb
        tot_vol += v
        ts5.append(bar.timestamp)
        px5.append(bar.price)
        if bar.is_1m_close:
            ts_m.append(bar.timestamp)
            F.append(bar.v2_vector[IDX])
            px_m.append(bar.price)
            vp_rows.append(vp_snapshot(bar.price))
    return (np.array(ts_m), np.array(F, dtype=np.float64), np.array(px_m),
            np.array(ts5), np.array(px5), vp_rows)


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
            _, F, _, _, _, _ = day_pass(day, model, None, rng)
            samp.append(score(np.nan_to_num(F), model))
        except Exception:
            pass
    th = float(np.quantile(np.concatenate(samp), 0.995))

    for year in ('2025', '2024'):
        days = sorted(os.path.basename(f).replace('.parquet', '')
                      for f in glob.glob(os.path.join(ATLAS, '1m', f'{year}_*.parquet')))
        lab_rows, null_rows, trades = [], [], []
        for day in days:
            try:
                ts_m, F, px_m, ts5, px5, vp = day_pass(day, model, th, rng)
            except Exception:
                continue
            if len(ts_m) < 100:
                continue
            Fn = np.nan_to_num(F)
            s = score(Fn, model)
            zsum = Fn[:, ZH] + Fn[:, ZL]
            d0 = dtm.datetime.fromtimestamp(ts5[-1], tz=dtm.timezone.utc).astimezone(central)
            cut = central.localize(dtm.datetime(d0.year, d0.month, d0.day, 15, 55)).timestamp()
            h9 = central.localize(dtm.datetime(d0.year, d0.month, d0.day, 9, 0)).timestamp()
            h13 = central.localize(dtm.datetime(d0.year, d0.month, d0.day, 13, 0)).timestamp()

            # -- Test A rows: labels vs same-hour nulls (all day, VP known) --
            pick = os.path.join(_REPO, 'DATA', 'ai_cusp_picks',
                                f"ai_picks_{day.replace('_', '-')}_multi.json")
            if os.path.exists(pick):
                hours_of = np.array([dtm.datetime.fromtimestamp(t, tz=dtm.timezone.utc)
                                    .astimezone(central).hour for t in ts_m])
                for t in json.load(open(pick)).get('trades', []):
                    i = np.searchsorted(ts_m, t['entry_ts']) - 1
                    if i < 60 or vp[i] is None:
                        continue
                    lab_rows.append(vp[i])
                    hr = hours_of[i]
                    cand = np.nonzero((hours_of == hr)
                                      & (np.arange(len(ts_m)) >= 60))[0]
                    cand = [c for c in cand if abs(c - i) > 5 and vp[c] is not None]
                    if cand:
                        null_rows.append(vp[int(rng.choice(cand))])

            # -- Test B: niche trades with VP context --
            pos, entry, ext, e_i = 0, 0.0, 0.0, 0
            k5 = 0
            for i in range(60, len(ts_m)):
                t = ts_m[i]
                while k5 < len(ts5) and ts5[k5] <= t:
                    p = px5[k5]
                    if pos != 0:
                        ext = max(ext, p) if pos > 0 else min(ext, p)
                        hit = ((ext - p >= 20 * TICK) if pos > 0
                               else (p - ext >= 20 * TICK))
                        if hit or ts5[k5] >= cut:
                            pnl = ((p - entry) / TICK * pos - COST_TICKS) * TICK_VALUE
                            if vp[e_i] is not None:
                                trades.append(dict(pnl=pnl, **vp[e_i]))
                            pos = 0
                    k5 += 1
                if pos == 0 and h9 <= t < min(cut, h13) and s[i] >= th:
                    pos = -1 if zsum[i] > 0 else 1
                    entry = px_m[i]; ext = px_m[i]; e_i = i

        # ---- report Test A ----
        log(f"\n== {year} Test A: labels ({len(lab_rows)}) vs same-hour nulls "
            f"({len(null_rows)}) ==")
        for k in ('dist_poc', 'node_ratio', 'va_width'):
            a = np.array([r[k] for r in lab_rows], dtype=float)
            b = np.array([r[k] for r in null_rows], dtype=float)
            # rank AUC
            allv = np.r_[a, b]; y = np.r_[np.ones(len(a)), np.zeros(len(b))]
            o = allv.argsort(); rk = np.empty(len(allv)); rk[o] = np.arange(1, len(allv) + 1)
            auc = (rk[y == 1].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
            log(f"  {k:<11} label med {np.median(a):8.1f} | null med {np.median(b):8.1f} "
                f"| AUC {auc:.3f} (gap {auc-0.5:+.3f})")
        for k in ('in_va', 'above_va'):
            a = np.mean([r[k] for r in lab_rows])
            b = np.mean([r[k] for r in null_rows])
            log(f"  {k:<11} label {a:.3f} | null {b:.3f} | diff {a-b:+.3f}")

        # ---- report Test B ----
        pnl = np.array([r['pnl'] for r in trades])
        log(f"== {year} Test B: niche trades with VP context (n={len(pnl)}, "
            f"$/tr {pnl.mean():+.2f}) ==")
        inva = np.array([r['in_va'] for r in trades])
        for name, m in (('inside VA', inva), ('outside VA', ~inva)):
            if m.sum() >= 20:
                log(f"  {name:<11} n={m.sum():4d} $/tr {pnl[m].mean():+7.2f} "
                    f"win% {100*(pnl[m]>0).mean():.1f}")
        nr = np.array([r['node_ratio'] for r in trades])
        for name, m in (('HVN>=1.5', nr >= 1.5), ('mid', (nr > 0.5) & (nr < 1.5)),
                        ('LVN<=0.5', nr <= 0.5)):
            if m.sum() >= 20:
                log(f"  {name:<11} n={m.sum():4d} $/tr {pnl[m].mean():+7.2f} "
                    f"win% {100*(pnl[m]>0).mean():.1f}")
        dp = np.array([r['dist_poc'] for r in trades])
        med = np.median(dp)
        for name, m in (('near POC', dp <= med), ('far POC', dp > med)):
            if m.sum() >= 20:
                log(f"  {name:<11} n={m.sum():4d} $/tr {pnl[m].mean():+7.2f} "
                    f"win% {100*(pnl[m]>0).mean():.1f}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, 'vp_study.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
