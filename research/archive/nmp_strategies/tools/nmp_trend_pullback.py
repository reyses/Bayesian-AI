"""Trend-with-Pullback simulation (HONEST units).
Direction: 5m mean slope (L2_5m_price_velocity_9)  ->  >0 LONG, <0 SHORT
Location : 1m z-score (L3_1m_z_se_15)               ->  enter on a pullback against the dip
Exit     : 5m trend FLIP (slope crosses 0), then z-timed back to the mean (Z_EXIT)

Rules (LONG; SHORT mirrors):
  enter  if slope_5m > 0 AND z <= -Z_ENTRY     (buy the dip in the uptrend)
  exit   when slope_5m <= 0 (flip latched) AND z >= Z_EXIT  (trend turned; leave near mean)

MNQ economics: $2/point (NOT $20 = full-size NQ). Costs modeled explicitly; GROSS and
NET reported separately so we can tell a no-edge signal from an over-traded one.
Z_ENTRY is SWEPT to test the claim "z entry doesn't work" with data.

Run: python research/nmp_trend_pullback.py
"""
import os, sys, glob
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem

# ── MNQ economics ──────────────────────────────────────────────────────────
USD_PER_PT = 2.0                 # MNQ micro = $2/point (tick 0.25 = $0.50)
COMMISSION_PER_SIDE = 0.50       # ~NT8 discount all-in per side
SLIPPAGE_PTS_PER_SIDE = 0.25     # 1 tick per side
COST_PER_TRADE = (COMMISSION_PER_SIDE * 2) + (SLIPPAGE_PTS_PER_SIDE * 2 * USD_PER_PT)  # = $2.00 RT

Z_EXIT = 0.0                     # after the trend flips, leave when z returns to the mean
Z_ENTRY_SWEEP = [1.0, 1.5, 1.8481, 2.5]

ATLAS = 'DATA/ATLAS'
FEAT = f'{ATLAS}/FEATURES_5s_v2'
LABELS = f'{ATLAS}/regime_labels_2d.csv'
RNG = np.random.RandomState(42)


def dayblock_ci(per_day):
    nd = len(per_day)
    boots = [per_day[RNG.randint(0, nd, nd)].mean() for _ in range(4000)]
    return float(per_day.mean()), tuple(np.percentile(boots, [2.5, 97.5]))


def simulate(rows_by_day, z_entry):
    """Run the state machine over the pre-collected (ts, px, z, slope) stream.
    pos resets every day (no overnight holds); EOD force-close."""
    trades, holds = [], []
    for day, rows in rows_by_day.items():
        pos = 0; eprice = 0.0; ets = 0; want_exit = False
        ts = px = None
        for (ts, px, z, slope) in rows:
            if pos == 0:
                if slope > 0 and z <= -z_entry:
                    pos, eprice, ets, want_exit = 1, px, ts, False
                elif slope < 0 and z >= z_entry:
                    pos, eprice, ets, want_exit = -1, px, ts, False
            else:
                if pos == 1:
                    if slope <= 0:
                        want_exit = True
                    hit = want_exit and z >= Z_EXIT
                else:
                    if slope >= 0:
                        want_exit = True
                    hit = want_exit and z <= -Z_EXIT
                if hit:
                    pts = (px - eprice) if pos == 1 else (eprice - px)
                    trades.append((day, pts, pts * USD_PER_PT))
                    holds.append((ts - ets) / 60.0)
                    pos = 0
        if pos != 0:                              # EOD close
            pts = (px - eprice) if pos == 1 else (eprice - px)
            trades.append((day, pts, pts * USD_PER_PT))
            holds.append((ts - ets) / 60.0)
    return trades, holds


def metrics(trades, holds, days):
    n = len(trades)
    if n == 0:
        return None
    gross = np.array([t[2] for t in trades])
    net = gross - COST_PER_TRADE
    g_pf = gross[gross > 0].sum() / abs(gross[gross < 0].sum()) if (gross < 0).any() else np.inf
    n_pf = net[net > 0].sum() / abs(net[net < 0].sum()) if (net < 0).any() else np.inf
    per_day = pd.Series([t[2] for t in trades], index=[t[0] for t in trades]).groupby(level=0).sum()
    per_day_net = per_day - COST_PER_TRADE * pd.Series([t[0] for t in trades]).value_counts()
    pd_net = per_day_net.reindex(days).fillna(0).to_numpy()
    m, ci = dayblock_ci(pd_net)
    return dict(n=n, tpd=n/len(days), gross=gross.mean(), net=net.mean(),
                g_pf=g_pf, n_pf=n_pf, mday=m, ci=ci,
                wdays=int((pd_net > 0).sum()), hold=np.mean(holds))


def main():
    files = sorted(glob.glob(os.path.join(FEAT, 'L0', '2024_*.parquet')))
    days = [os.path.basename(f).replace('.parquet', '') for f in files]
    if not days:
        print("No 2024 data found!"); return

    fps = MultiDayForwardPassSystem(atlas_root=ATLAS, features_root=FEAT,
                                    labels_csv=LABELS, days=days)

    # ── single pass through the forward pass; collect the decision stream ──
    rows_by_day = OrderedDict()
    kept = 0
    for state in fps:
        if not state.is_1m_close:
            continue
        z = state.v2.get('L3_1m_z_se_15', np.nan)
        slope = state.v2.get('L2_5m_price_velocity_9', np.nan)
        if np.isnan(z) or np.isnan(slope):
            continue
        rows_by_day.setdefault(state.day, []).append(
            (state.timestamp, state.price, float(z), float(slope)))
        kept += 1
    print(f"collected {kept} 1m-close decision rows over {len(rows_by_day)} days")

    L = ["# NMP Trend-with-Pullback — HONEST units (MNQ $2/pt), 2024\n",
         f"Direction: 5m slope sign | Entry: z pullback (swept) | Exit: trend-flip + z->mean ({Z_EXIT})",
         f"Cost: ${COST_PER_TRADE:.2f}/trade RT ({COMMISSION_PER_SIDE}/side comm + "
         f"{SLIPPAGE_PTS_PER_SIDE}pt/side slip @ ${USD_PER_PT}/pt)\n",
         "| Z_ENTRY | trades | t/day | gross $/tr | **net $/tr** | gross PF | net PF | "
         "net $/day [95% CI] | win days | hold(min) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    best = None
    for ze in Z_ENTRY_SWEEP:
        tr, ho = simulate(rows_by_day, ze)
        mx = metrics(tr, ho, days)
        if mx is None:
            L.append(f"| {ze} | 0 | - | - | - | - | - | - | - | - |"); continue
        L.append(f"| {ze} | {mx['n']} | {mx['tpd']:.1f} | {mx['gross']:+.2f} | "
                 f"**{mx['net']:+.2f}** | {mx['g_pf']:.2f} | {mx['n_pf']:.2f} | "
                 f"{mx['mday']:+.0f} [{mx['ci'][0]:+.0f},{mx['ci'][1]:+.0f}] | "
                 f"{mx['wdays']}/{len(days)} | {mx['hold']:.0f} |")
        if best is None or mx['gross'] > best[1]['gross']:
            best = (ze, mx)

    # ── kill-or-continue gate on GROSS edge vs cost ──
    ze, mx = best
    gate = ("DEAD: best gross/trade can't cover cost -> trend-with-pullback confirmed unviable"
            if mx['gross'] < COST_PER_TRADE else
            f"SIGNAL: best gross/trade (Z_ENTRY={ze}) clears cost -> worth drilling (frequency/regime)")
    L += ["", f"Best gross = Z_ENTRY {ze}: gross ${mx['gross']:+.2f}/tr vs cost ${COST_PER_TRADE:.2f}/tr.",
          f"VERDICT: {gate}",
          "CAVEAT: thresholds calibrated on a 2024 subset; full-2024 mixes IS/OOS. Pre-commit OOS split next."]
    rep = "\n".join(L)
    os.makedirs('reports/findings', exist_ok=True)
    open('reports/findings/nmp_trend_pullback_2024.md', 'w').write(rep)
    print(rep)


if __name__ == '__main__':
    main()
