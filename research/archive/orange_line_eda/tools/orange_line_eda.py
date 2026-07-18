"""EDA on the ORANGE line (7.5-min cubic regression, fit to 1s) — find CAUSAL exit signals.

Plots, at 1-second resolution over a day:
  panel 1: price + orange cubic curve VALUE (the smooth anchor), swing-turns marked
  panel 2: orange SLOPE (1st deriv)      -> sign = direction; zero-cross = the turn (LATE)
  panel 3: orange CURVATURE (2nd deriv)  -> goes against the move while it decelerates;
           curvature zero-cross = inflection = a candidate EARLY, causal exit warning

All three (value/slope/curvature) are computed from the cubic OLS fit endpoint -> CAUSAL
(uses only the trailing 7.5 min). EDA also measures the LEAD TIME between a curvature
flip and the following slope flip (how early does curvature warn?), and the swing-size
distribution, so we pick the "big move" cutoff from data — never from hindsight.

Run: python research/orange_line_eda.py            (default day + week)
     python research/orange_line_eda.py 2024_03_18
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ATLAS = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS'
ONE_S = f'{ATLAS}/1s'
ORANGE_W, ORANGE_DEG = 450, 3       # 7.5 min cubic (1s samples)
X = np.arange(ORANGE_W) / 60.0      # x in MINUTES (well-conditioned vs sample index)
XE = X[-1]


def _cubic_weights():
    """Fixed conv weights giving the cubic-fit VALUE, SLOPE, CURVATURE at the endpoint."""
    P = np.linalg.pinv(np.vander(X, ORANGE_DEG + 1))     # c = P@y; c[0]=a..c[3]=d
    val = XE**3 * P[0] + XE**2 * P[1] + XE * P[2] + P[3]
    slope = 3 * XE**2 * P[0] + 2 * XE * P[1] + P[2]
    curv = 6 * XE * P[0] + 2 * P[1]
    return val, slope, curv


VAL_WT, SLOPE_WT, CURV_WT = _cubic_weights()


def _roll(prices, wt):
    out = np.full(len(prices), np.nan)
    if len(prices) <= ORANGE_W:
        return out
    out[ORANGE_W:] = np.convolve(prices, wt[::-1], 'valid')[:-1]
    return out


def orange_lines(day):
    f = f'{ONE_S}/{day}.parquet'
    if not os.path.exists(f):
        return None
    df = pd.read_parquet(f, columns=['timestamp', 'close']).sort_values('timestamp')
    p = df['close'].to_numpy(np.float64)
    ts = df['timestamp'].to_numpy(np.int64)
    return ts, p, _roll(p, VAL_WT), _roll(p, SLOPE_WT), _roll(p, CURV_WT)


def zero_crossings(x):
    """indices where x crosses 0 (sign change), ignoring NaN."""
    s = np.sign(np.nan_to_num(x))
    idx = np.where((s[:-1] != 0) & (s[1:] != 0) & (s[:-1] != s[1:]))[0] + 1
    return idx


def swing_eda(ts, val, slope, curv):
    """Between consecutive slope zero-crossings = one swing of the orange curve.
    Amplitude = |val change| over the swing; duration in minutes.
    Lead = seconds the curvature flip PRECEDES the slope flip that ends the swing."""
    sx = zero_crossings(slope)
    cx = zero_crossings(curv)
    swings, leads = [], []
    for a, b in zip(sx[:-1], sx[1:]):
        amp = abs(val[b] - val[a])
        dur = (ts[b] - ts[a]) / 60.0
        swings.append((amp, dur))
        prior_curv = cx[(cx > a) & (cx <= b)]      # curvature flip inside this swing
        if len(prior_curv):
            leads.append((ts[b] - ts[prior_curv[-1]]))   # secs from last curv-flip to the slope-flip
    return np.array(swings), np.array(leads)


def plot_day(day, out):
    r = orange_lines(day)
    if r is None:
        print("no data", day); return
    ts, p, val, slope, curv = r
    t = (ts - ts[0]) / 3600.0    # hours from session start
    sx = zero_crossings(slope)

    fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1.2, 1.2]})
    ax[0].plot(t, p, color='#9e9e9e', lw=0.4, alpha=0.7, label='price (1s)')
    ax[0].plot(t, val, color='#e8730c', lw=1.8, label='orange 7.5m cubic (value)')
    for i in sx:
        ax[0].axvline(t[i], color='#1565c0', lw=0.5, alpha=0.35)   # swing turns (slope=0)
    ax[0].set_ylabel('price'); ax[0].legend(loc='upper left', fontsize=8)
    ax[0].set_title(f"Orange line EDA {day} — blue verticals = orange turns (slope=0). "
                    f"Curvature (bottom) flips BEFORE the turn?", fontsize=11, loc='left')

    ax[1].plot(t, slope, color='#e8730c', lw=0.9); ax[1].axhline(0, color='k', lw=0.6)
    ax[1].fill_between(t, slope, 0, where=slope > 0, color='#2e7d32', alpha=0.18)
    ax[1].fill_between(t, slope, 0, where=slope < 0, color='#c62828', alpha=0.18)
    ax[1].set_ylabel('slope (price/min)')

    ax[2].plot(t, curv, color='#6a1b9a', lw=0.9); ax[2].axhline(0, color='k', lw=0.6)
    for i in zero_crossings(curv):
        ax[2].axvline(t[i], color='#6a1b9a', lw=0.4, alpha=0.25)
    ax[2].set_ylabel('curvature'); ax[2].set_xlabel('hours from session start')

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out}")


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else '2024_03_18'
    # aggregate EDA over a small week around the plotted day
    days = sorted(os.path.basename(f)[:-8] for f in glob.glob(f'{ONE_S}/2024_03_*.parquet'))[:10]
    all_sw, all_ld = [], []
    for d in days:
        r = orange_lines(d)
        if r is None:
            continue
        sw, ld = swing_eda(r[0], r[2], r[3], r[4])
        if len(sw):
            all_sw.append(sw)
        if len(ld):
            all_ld.append(ld)
    sw = np.vstack(all_sw); ld = np.concatenate(all_ld)
    amp, dur = sw[:, 0], sw[:, 1]
    big = amp >= np.percentile(amp, 75)

    L = [f"# Orange-line EDA (7.5m cubic, 1s) — {len(days)} days ({days[0]}..{days[-1]})\n",
         f"swings (slope zero-cross to zero-cross): n={len(amp)}",
         f"  amplitude pts: median {np.median(amp):.2f}, p75 {np.percentile(amp,75):.2f}, p90 {np.percentile(amp,90):.2f}",
         f"  duration min : median {np.median(dur):.1f}, p75 {np.percentile(dur,75):.1f}",
         f"BIG swings (amp>=p75, n={big.sum()}): median amp {np.median(amp[big]):.2f} pts, dur {np.median(dur[big]):.1f} min",
         "",
         "CAUSAL early-exit test — curvature flip leads the slope flip (the turn):",
         f"  swings with a curvature flip inside: {len(ld)}/{len(amp)}",
         f"  LEAD time (curv-flip -> slope-flip): median {np.median(ld):.0f}s, p25 {np.percentile(ld,25):.0f}s, p75 {np.percentile(ld,75):.0f}s",
         f"  -> curvature warns ~{np.median(ld):.0f}s before the orange peak, causally (no hindsight).",
         "",
         "Read: exit on curvature-flip (leading) vs slope-flip (lagging) is the lever to test next."]
    rep = "\n".join(L)
    os.makedirs('reports/findings', exist_ok=True)
    open('reports/findings/orange_line_eda.md', 'w').write(rep)
    print(rep)
    plot_day(day, f'reports/findings/orange_line_eda_{day}.png')


if __name__ == '__main__':
    main()
