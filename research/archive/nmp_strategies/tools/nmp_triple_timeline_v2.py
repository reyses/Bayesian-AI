"""Triple-Timeline trend alignment — CORRECT math, honest MNQ units.

Three timescales (the user's design, from plot_option2_trade_gif.py):
  BLUE   = slope of a 24-min QUADRATIC OLS fit   (macro anchor)   window 1440 (1s)
  ORANGE = slope of a 7.5-min CUBIC OLS fit       (micro anchor)  window  450 (1s)
  PINK   = 15-min KINEMATIC projection displacement  disp = v*T + 0.5*a*T*T
           (v,a = 15-min EMA of velocity/accel; T = 15 min horizon)   the trigger

Entry: all three agree in sign (LONG if all > 0, SHORT if all < 0).
Exit : ORANGE flips OR PINK flips OR stop (in POINTS, unit-safe).

Slopes are computed as a CONVOLUTION of price with fixed regression-derivative
weights (Vandermonde pinv) -> whole day vectorized, no per-bar 1440-mult loop.

MNQ = $2/point (NOT $20). Costs explicit. IS=2024, OOS=2025-26. Day-block CI.
Run: python research/nmp_triple_timeline_v2.py
"""
import os, sys, glob
import numpy as np
import pandas as pd

ATLAS = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS'
ONE_S = f'{ATLAS}/1s'
OUT_CSV = 'reports/findings/nmp_triple_timeline_v2_trades.csv'
OUT_MD = 'reports/findings/nmp_triple_timeline_v2.md'

# ── windows (1s samples) ──
BLUE_W, BLUE_DEG = 1440, 2     # 24 min quadratic
ORANGE_W, ORANGE_DEG = 450, 3  # 7.5 min cubic
EMA_SPAN = 900                 # 15 min kinematics
PROJ_T = 900.0                 # 15 min projection horizon (seconds)

# ── MNQ economics ──
USD_PER_PT = 2.0
COST_PER_TRADE = 2.0           # ~$1 comm RT + 1 tick slip RT (market entry on momentum)
STOP_PTS = 5.0                 # hard stop in POINTS (orig was -$100/$20 = 5pt)

RNG = np.random.RandomState(42)


def poly_slope_weights(window, degree):
    """Fixed weights w s.t. dot(w, price_window) = slope of a degree-`degree`
    OLS fit, evaluated at the window's right endpoint."""
    x = np.arange(window)
    P = np.linalg.pinv(np.vander(x, degree + 1))   # coeffs c = P @ y, c[0]=highest deg
    xe = x[-1]
    if degree == 2:
        return 2 * xe * P[0] + P[1]
    if degree == 3:
        return 3 * xe**2 * P[0] + 2 * xe * P[1] + P[2]
    raise ValueError(degree)


BLUE_WT = poly_slope_weights(BLUE_W, BLUE_DEG)
ORANGE_WT = poly_slope_weights(ORANGE_W, ORANGE_DEG)


def rolling_slope(prices, wt, window):
    """slope at every bar i (using prices[i-window:i]); NaN until warmed up.
    dot(wt, prices[i-window:i]) == convolution of prices with reversed wt."""
    out = np.full(len(prices), np.nan)
    if len(prices) <= window:           # short/holiday session: can't fill the window
        return out
    conv = np.convolve(prices, wt[::-1], mode='valid')   # len = N-window+1, aligned to window-1
    out[window:] = conv[:-1]            # slope available at bar i uses window ending at i-1
    return out


def dayblock_ci(per_day):
    if len(per_day) < 3:
        return float(np.mean(per_day)) if len(per_day) else 0.0, (np.nan, np.nan)
    b = [per_day[RNG.randint(0, len(per_day), len(per_day))].mean() for _ in range(4000)]
    return float(per_day.mean()), tuple(np.percentile(b, [2.5, 97.5]))


def sim_day(day, prices, ts):
    blue = rolling_slope(prices, BLUE_WT, BLUE_W)
    orange = rolling_slope(prices, ORANGE_WT, ORANGE_W)
    s = pd.Series(prices)
    v = s.diff().ewm(span=EMA_SPAN, adjust=False).mean()
    a = v.diff().ewm(span=EMA_SPAN, adjust=False).mean()
    pink = (v * PROJ_T + 0.5 * a * PROJ_T * PROJ_T).to_numpy()   # projected displacement

    trades = []
    pos = 0; eprice = ets = 0; mfe = mae = 0.0
    start = max(BLUE_W, ORANGE_W, EMA_SPAN) + 1
    for i in range(start, len(prices)):
        b, o, pk, px = blue[i], orange[i], pink[i], prices[i]
        if np.isnan(b) or np.isnan(o) or np.isnan(pk):
            continue
        if pos == 0:
            if b > 0 and o > 0 and pk > 0:
                pos, eprice, ets, mfe, mae = 1, px, ts[i], px, px
            elif b < 0 and o < 0 and pk < 0:
                pos, eprice, ets, mfe, mae = -1, px, ts[i], px, px
        else:
            if pos == 1:
                mfe, mae = max(mfe, px), min(mae, px)
                pnl_pts = px - eprice
                exit_now = (o < 0) or (pk < 0) or (pnl_pts <= -STOP_PTS)
            else:
                mfe, mae = min(mfe, px), max(mae, px)
                pnl_pts = eprice - px
                exit_now = (o > 0) or (pk > 0) or (pnl_pts <= -STOP_PTS)
            if exit_now:
                mfe_pts = (mfe - eprice) if pos == 1 else (eprice - mfe)
                mae_pts = (mae - eprice) if pos == 1 else (eprice - mae)
                g = pnl_pts * USD_PER_PT
                trades.append((day, int(ets), 'LONG' if pos == 1 else 'SHORT',
                               float(eprice), int(ts[i]), float(px), float(pnl_pts),
                               g, g - COST_PER_TRADE, mfe_pts * USD_PER_PT, mae_pts * USD_PER_PT))
                pos = 0
    return trades


def block(name, df, days):
    if len(df) == 0:
        return f"| {name} | 0 | - | - | - | - | - |"
    net = df['net_usd'].to_numpy()
    g = df['gross_usd'].to_numpy()
    gpf = g[g > 0].sum() / abs(g[g < 0].sum()) if (g < 0).any() else np.inf
    npf = net[net > 0].sum() / abs(net[net < 0].sum()) if (net < 0).any() else np.inf
    pday = df.groupby('day')['net_usd'].sum().reindex(days).fillna(0).to_numpy()
    m, ci = dayblock_ci(pday)
    return (f"| {name} | {len(df)} | {len(df)/max(len(days),1):.1f} | {g.mean():+.2f} | "
            f"**{net.mean():+.2f}** | g{gpf:.2f}/n{npf:.2f} | "
            f"{m:+.0f} [{ci[0]:+.0f},{ci[1]:+.0f}] | {(pday>0).sum()}/{len(days)} |")


def main():
    files = sorted(glob.glob(f'{ONE_S}/*.parquet'))
    days = [os.path.basename(f)[:-8] for f in files]
    print(f"{len(days)} day files; computing...")
    all_tr = []
    for k, day in enumerate(days):
        df = pd.read_parquet(files[k], columns=['timestamp', 'close']).sort_values('timestamp')
        all_tr += sim_day(day, df['close'].to_numpy(np.float64), df['timestamp'].to_numpy(np.int64))
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(days)} days, {len(all_tr)} trades")

    cols = ['day', 'entry_ts', 'leg_dir', 'entry_price', 'exit_ts', 'exit_price',
            'pnl_pts', 'gross_usd', 'net_usd', 'mfe_usd', 'mae_usd']
    tr = pd.DataFrame(all_tr, columns=cols)
    os.makedirs('reports/findings', exist_ok=True)
    tr.to_csv(OUT_CSV, index=False)

    is_days = [d for d in days if d.startswith('2024_')]
    oos_days = [d for d in days if d.startswith(('2025_', '2026_'))]
    is_tr = tr[tr['day'].str.startswith('2024_')]
    oos_tr = tr[tr['day'].str.startswith(('2025_', '2026_'))]

    L = ["# Triple-Timeline v2 — correct math, MNQ $2/pt\n",
         f"BLUE 24m quad slope | ORANGE 7.5m cubic slope | PINK 15m kinematic projection (v*T+0.5a*T^2)",
         f"Entry: 3-sign agree | Exit: orange/pink flip or {STOP_PTS}pt stop | cost ${COST_PER_TRADE}/tr\n",
         "| split | trades | t/day | gross $/tr | net $/tr | PF g/n | net $/day [CI] | win days |",
         "|---|---|---|---|---|---|---|---|",
         block("IS 2024", is_tr, is_days),
         block("OOS 2025-26", oos_tr, oos_days),
         block("ALL", tr, days)]
    # MFE diagnostic (let-winners-run): is the captured move clipped vs the peak?
    if len(tr):
        win = tr[tr['gross_usd'] > 0]
        L += ["", f"MFE diag (all): avg captured ${tr['gross_usd'].mean():+.2f}  avg MFE ${tr['mfe_usd'].mean():+.2f}  "
              f"avg MAE ${tr['mae_usd'].mean():+.2f}",
              f"winners n={len(win)}: captured ${win['gross_usd'].mean():+.2f} vs MFE ${win['mfe_usd'].mean():+.2f} "
              f"(kept {100*win['gross_usd'].mean()/max(win['mfe_usd'].mean(),1e-9):.0f}% of peak)"]
    rep = "\n".join(L)
    open(OUT_MD, 'w').write(rep)
    print("\n" + rep)
    print(f"\n[trades -> {OUT_CSV}]")


if __name__ == '__main__':
    main()
