"""
ADX window FREQUENCY sweep (design tool, NOT an edge search).

Purpose (Moises 2026-07-14): find rolling-window / threshold settings that make ADX-08
fire at an ACTIONABLE rate. This measures FREQUENCY ONLY — it never looks at returns, so
it cannot overfit an edge. The edge is measured LATER, out-of-sample, on the frozen
setting, with the no-stops horizon method. Frequency = design knob; edge = read-out.

Indicators computed CONTINUOUSLY across the whole concatenated stream (no cold start,
per doc 073), then triggers counted only on RTH bars (08:30-15:15 CT).
Trigger = ADX > threshold AND close crosses the N_sma MA (up=bull / down=bear),
one per direction per day (matches the detector).

Run: python adx_window_sweep.py START_YYYY_MM END_YYYY_MM   (default 2024_01 2024_03)
"""
import os, sys, glob
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
D5 = os.path.join(ROOT, 'DATA', 'ATLAS', '5s')
RTH0, RTH1 = pd.Timestamp('08:30').time(), pd.Timestamp('15:15').time()

N_ADX = [84, 168, 336, 720]          # 7, 14, 28, 60 minutes on 5s bars
N_SMA = 240                          # 20-min cross MA (legacy)
THRESH = [15.0, 20.0, 25.0]
SMOOTH = ['SMA', 'WILDER']


def load_stream(a, b):
    days = sorted(glob.glob(os.path.join(D5, '*.parquet')))
    days = [d for d in days if a <= os.path.basename(d)[:7] <= b]
    frames = []
    for p in days:
        df = pd.read_parquet(p, columns=['timestamp', 'high', 'low', 'close'])
        frames.append(df)
    s = pd.concat(frames).sort_values('timestamp').reset_index(drop=True)
    dt = pd.to_datetime(s['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    s['rth'] = (dt.dt.time >= RTH0) & (dt.dt.time <= RTH1)
    s['day'] = dt.dt.strftime('%Y-%m-%d')
    return s


def smooth(x, n, how):
    if how == 'SMA':
        return x.rolling(n, min_periods=n).mean()
    return x.ewm(alpha=1.0 / n, adjust=False).mean()      # Wilder RMA


def adx_series(h, l, c, n, how):
    up = h.diff(); dn = -l.diff()
    dm_p = np.where((up > dn) & (up > 0), up, 0.0)
    dm_m = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    tr_s = smooth(tr, n, how).replace(0, np.nan)
    di_p = 100 * smooth(pd.Series(dm_p, index=c.index), n, how) / tr_s
    di_m = 100 * smooth(pd.Series(dm_m, index=c.index), n, how) / tr_s
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    return smooth(dx, n, how)


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else '2024_01'
    b = sys.argv[2] if len(sys.argv) > 2 else '2024_03'
    s = load_stream(a, b)
    h, l, c = s['high'], s['low'], s['close']
    ndays = s.loc[s['rth'], 'day'].nunique()
    print(f'stream {a}..{b}: {len(s)} bars, {ndays} RTH days, continuous window\n')

    sma_cross = c.rolling(N_SMA, min_periods=N_SMA).mean()
    prev_c, prev_m = c.shift(1), sma_cross.shift(1)
    cross_up = (prev_c <= prev_m) & (c > sma_cross)
    cross_dn = (prev_c >= prev_m) & (c < sma_cross)
    rth = s['rth'].values

    print(f"{'smooth':7} {'N_adx':>6} {'thr':>4} {'trig/day':>9} {'days_w_trig%':>12} {'total':>6}")
    rows = []
    for how in SMOOTH:
        for n in N_ADX:
            adx = adx_series(h, l, c, n, how).values
            for thr in THRESH:
                fire = ((adx > thr) & (cross_up.values | cross_dn.values) & rth)
                # one-per-direction-per-day would need a groupby; count raw fires and
                # trig-days (a day with >=1 fire) — both are frequency proxies.
                total = int(fire.sum())
                dser = pd.Series(s['day'].values[fire])
                days_fired = dser.nunique()
                rows.append((how, n, thr, total / ndays, 100 * days_fired / ndays, total))
                print(f"{how:7} {n:>6} {thr:>4.0f} {total/ndays:>9.2f} "
                      f"{100*days_fired/ndays:>11.0f}% {total:>6}")
    # actionable band hint
    print("\nActionable ~= 0.3-2.0 trig/day (roughly 1 setup per session, both dirs).")
    good = [r for r in rows if 0.3 <= r[3] <= 2.0]
    print(f"Combos in that band: {len(good)}")
    for r in sorted(good, key=lambda x: -x[4])[:8]:
        print(f"  {r[0]} N_adx={r[1]} thr={r[2]:.0f}: {r[3]:.2f}/day, {r[4]:.0f}% days")


if __name__ == '__main__':
    main()
