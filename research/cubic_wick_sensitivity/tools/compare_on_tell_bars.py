#!/usr/bin/env python3
"""Close-only cubic vs wick-sensitive variants on the two captured tell bars.

Run (from repo root, CPU-light):
  python research/cubic_wick_sensitivity/tools/compare_on_tell_bars.py            # both presets + stability scan
  python research/cubic_wick_sensitivity/tools/compare_on_tell_bars.py --preset top

Presets (bar indices are plain row indices of DATA/ATLAS/1m/<day>.parquet,
exactly how pocket_dojo._bars() indexes — provenance guards assert it):
  top  2025_08_24 bar 107 — "buyers struggling" exhaustion top (OWNER_PROCESS.md:
       rally bar94 vol 1190, struggle bars 99-106 w/ growing upper wicks +
       thinning volume, crash bar 107 body -10.25 vol 715).
  dip  2025_06_05 bar 880 — fake dip / stop-run (docs/daily/2026-07-30.md: low
       ran the 21770.75 stop at 14:40:20, 16.5pt lower wick, closed 21772.25,
       next bar +57.75).

PRE-REGISTERED rules (fixed before results were seen; identical for every
variant so no rule shopping):
  DOWN-flag = first bar i in the scan window with slope[i] < 0 AND
              slope[i+1] < 0 (2-bar sustain; causally CONFIRMED at i+1).
  UP-flag   = same with > 0.
  Stability = sign flips of the slope per 100 warm bars over whole days.
An earlier flag from a line that flips sign more often is NOT an improvement —
both numbers are reported together.

Outputs: reports/comparison_results.json + PNGs in reports/assets/.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.patches import Rectangle           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wick_series as ws                           # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
REPORTS = os.path.join(REPO, 'research', 'cubic_wick_sensitivity', 'reports')
ASSETS = os.path.join(REPORTS, 'assets')

SUSTAIN_BARS = 2       # consecutive same-sign slope bars required for a flag
FLIP_RATE_PER = 100    # whipsaw metric denominator (flips per 100 warm bars)
TICK_EVERY = 5         # x-axis label cadence (bars)
BIAS_SPIKE_LVL = 0.30  # |wick_bias| above this = a "loud" single-bar shape tell
                       # (0.30 ~= wick asymmetry covering 30% of the bar range;
                       # descriptive threshold for reporting, not a trade rule)

# Whole-day whipsaw scan set: the pocket-dojo replay days that have 1m parquet
# coverage (2025_06_05 / 2025_08_24 / 2025_12_19 — the owner's own tape; the
# 2026_07_* pocket days have no ATLAS 1m files, coverage ends 2026_03_20) plus
# three mid-month days spread across recent months for regime variety, chosen
# by calendar position before any results were seen.
STABILITY_DAYS = ('2025_06_05', '2025_08_24', '2025_12_19',
                  '2026_01_15', '2026_02_16', '2026_03_16')

PRESETS = {
    'top': dict(
        day='2025_08_24', tell_bar=107,
        plot_lo=88, plot_hi=118,
        scan_lo=99, scan_hi=112,      # owner's struggle window start .. crash+5
        up_scan=(92, 98),             # secondary: rally-onset UP-flag timing
        guards=((94, 'volume', 1190), (107, 'volume', 715)),
        title='2025_08_24 exhaustion top — "buyers struggling" bars 99-106, crash at 107'),
    'dip': dict(
        day='2025_06_05', tell_bar=880,
        plot_lo=864, plot_hi=894,
        scan_lo=872, scan_hi=885,     # flush-recover prelude .. post-rip
        up_scan=None,
        guards=((879, 'high', 21810.0), (880, 'low', 21755.75)),
        title='2025_06_05 fake dip — bar 880 stop-run (16.5pt lower wick), bar 881 +57.75'),
}

# ---------- palette (dataviz skill defaults, validated: 3-series line palette
# #2a78d6/#eb6834/#4a3aa7 passes all six checks on the light surface) ----------
SURFACE = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#898781'
GRID = '#e1e0d9'; BASELINE = '#c3c2b7'
C_CLOSE = '#2a78d6'   # baseline close-only cubic       (slot 1 blue)
C_REJ = '#eb6834'     # rejection family k=1.0 / k=0.5  (slot 2 orange)
C_EXC = '#4a3aa7'     # excursion HL2 blend             (slot 7 violet)
C_UP = '#1baf7a'; C_DN = '#e34948'          # candle polarity (slots 3 / 8)
C_BIAS_POS = '#86b6ef'                      # panel-3 bars, light sequential step
CANDLE_ALPHA = 0.55   # candles are context; series lines must dominate


def load_day(day):
    """Row-indexed exactly like pocket_dojo._bars()."""
    df = pd.read_parquet(os.path.join(DATA, f'{day}.parquet'))[
        ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df.reset_index(drop=True)


def build_variants(df):
    """All input series + their cubic (value, slope) under the live 8-bar/1m grid."""
    o, h, l, c = (df[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    inputs = {
        'close': c,
        'rej05': ws.rejection_price(o, h, l, c, k=ws.K_REJECTION_GRID[0]),
        'rej10': ws.rejection_price(o, h, l, c, k=ws.K_REJECTION_GRID[1]),
        'exc': ws.excursion_price(h, l, c),
    }
    cub = {name: ws.rolling_cubic(s) for name, s in inputs.items()}   # (val, slp, cur)
    bias = ws.wick_bias(o, h, l, c)
    bfrac = ws.body_frac(o, h, l, c)
    return dict(inputs=inputs, cub=cub, bias=bias,
                bias_mean=ws.rolling_mean(bias), bias_cub=ws.rolling_cubic(bias),
                bfrac=bfrac, bfrac_mean=ws.rolling_mean(bfrac))


def first_sustained(slope, lo, hi, sign):
    """First bar i in [lo,hi] whose slope sign matches for SUSTAIN_BARS bars."""
    for i in range(lo, hi + 1):
        seg = slope[i:i + SUSTAIN_BARS]
        if len(seg) == SUSTAIN_BARS and np.isfinite(seg).all() and (np.sign(seg) == sign).all():
            return i
    return None


def flip_rate(slope):
    """Sign flips of the slope per FLIP_RATE_PER warm bars (whole array)."""
    s = slope[np.isfinite(slope)]
    if len(s) < 2:
        return np.nan, 0
    sg = np.sign(s)
    flips = int(np.sum(sg[1:] != sg[:-1]))
    return flips / len(s) * FLIP_RATE_PER, len(s)


def _j(x):
    """JSON sanitizer."""
    if isinstance(x, dict):
        return {k: _j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ---------- analysis per preset ----------
def analyze(preset, V, df):
    p = PRESETS[preset]
    lo, hi, tell = p['scan_lo'], p['scan_hi'], p['tell_bar']
    out = {'preset': preset, 'day': p['day'], 'tell_bar': tell,
           'scan_window': [lo, hi], 'variants': {}}
    base_flag = None
    for name in ('close', 'rej05', 'rej10', 'exc'):
        slp = V['cub'][name][1]
        down = first_sustained(slp, lo, hi, -1)
        up = (first_sustained(slp, *p['up_scan'], 1) if p['up_scan'] else None)
        rec = {'down_flag': down, 'confirmed_at': None if down is None else down + 1,
               'slope_at': {str(b): slp[b] for b in range(tell - 3, tell + 3)},
               'slope_min_in_window': float(np.nanmin(slp[lo:hi + 1])),
               'up_flag': up}
        if name == 'close':
            base_flag = down
        rec['lead_vs_close'] = (None if down is None or base_flag is None
                                else base_flag - down)
        out['variants'][name] = rec
    # direction (b): companion metrics around the tell
    bm, slp_c = V['bias_mean'], V['cub']['close'][1]
    neg = first_sustained(np.where(np.isfinite(bm), bm, np.nan), lo, hi, -1)
    div = [int(i) for i in range(lo, hi + 1)
           if np.isfinite(bm[i]) and np.isfinite(slp_c[i]) and bm[i] < 0 and slp_c[i] > 0]
    spikes = [int(i) for i in range(lo, hi + 1) if abs(V['bias'][i]) >= BIAS_SPIKE_LVL]
    out['variant_b'] = {
        'bias_at_tell': float(V['bias'][tell]),
        'bias_mean_at_tell': float(bm[tell]),
        'bias_mean_neg_flag': neg,
        'divergence_bars_priceUp_biasNeg': div,
        'loud_bias_bars_absGE_%.2f' % BIAS_SPIKE_LVL: spikes,
        'bias_by_bar': {str(b): float(V['bias'][b]) for b in range(lo, hi + 1)},
        'body_frac_mean_at_tell': float(V['bfrac_mean'][tell]),
    }
    if preset == 'top':   # conviction fade: struggle window vs rally window
        out['variant_b']['body_frac_rally_94_98'] = float(np.mean(V['bfrac'][94:99]))
        out['variant_b']['body_frac_struggle_99_106'] = float(np.mean(V['bfrac'][99:107]))
    if preset == 'dip':   # recovery: bars until slope back >0 after the tell
        for name in ('close', 'rej05', 'rej10', 'exc'):
            slp = V['cub'][name][1]
            back = next((i for i in range(tell, hi + 1) if slp[i] > 0), None)
            out['variants'][name]['first_pos_slope_from_tell'] = back
    return out


def stability_scan():
    rows = {}
    for day in STABILITY_DAYS:
        df = load_day(day)
        V = build_variants(df)
        rows[day] = {'n_bars': len(df)}
        for name in ('close', 'rej05', 'rej10', 'exc'):
            val, slp, _ = V['cub'][name]
            rate, n = flip_rate(slp)
            rows[day][name] = {
                'flips_per_100': rate,
                'mean_abs_input_shift_pts': float(np.nanmean(
                    np.abs(V['inputs'][name] - V['inputs']['close']))),
                'mean_abs_value_dev_from_close_cubic_pts': float(np.nanmean(
                    np.abs(val - V['cub']['close'][0]))),
            }
        rows[day]['bias_mean_smoother'] = {'flips_per_100': flip_rate(
            np.where(np.isfinite(V['bias_mean']), V['bias_mean'], np.nan))[0]}
        rows[day]['bias_cubic_slope'] = {'flips_per_100': flip_rate(V['bias_cub'][1])[0]}
    agg = {}
    for name in ('close', 'rej05', 'rej10', 'exc'):
        agg[name] = {k: float(np.nanmean([rows[d][name][k] for d in STABILITY_DAYS]))
                     for k in ('flips_per_100', 'mean_abs_input_shift_pts',
                               'mean_abs_value_dev_from_close_cubic_pts')}
    agg['bias_mean_smoother'] = {'flips_per_100': float(np.nanmean(
        [rows[d]['bias_mean_smoother']['flips_per_100'] for d in STABILITY_DAYS]))}
    agg['bias_cubic_slope'] = {'flips_per_100': float(np.nanmean(
        [rows[d]['bias_cubic_slope']['flips_per_100'] for d in STABILITY_DAYS]))}
    return {'days': rows, 'mean_across_days': agg}


# ---------- plotting ----------
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)


def _candles(ax, df, lo, hi):
    for i in range(lo, hi + 1):
        o, h, l, c = df.loc[i, ['open', 'high', 'low', 'close']]
        col = C_UP if c >= o else C_DN
        ax.plot([i, i], [l, h], color=col, lw=1.0, alpha=CANDLE_ALPHA, zorder=2)
        ax.add_patch(Rectangle((i - 0.32, min(o, c)), 0.64, max(abs(c - o), 1e-9),
                               facecolor=col, edgecolor='none',
                               alpha=CANDLE_ALPHA, zorder=2))


def _xticks(ax, df, lo, hi):
    ticks = list(range(lo + (-lo) % TICK_EVERY, hi + 1, TICK_EVERY))
    tt = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{i}\n{tt[i]:%H:%M}' for i in ticks])


def _tell_line(ax, x, label=None):
    ax.axvline(x, color=MUTED, lw=1.0, ls='--', zorder=1)
    if label:
        ax.annotate(label, xy=(x, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(4, -2), textcoords='offset points',
                    fontsize=8, color=INK2, va='top')


def plot_preset(preset, V, df):
    p = PRESETS[preset]
    lo, hi, tell = p['plot_lo'], p['plot_hi'], p['tell_bar']
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, dpi=150,
                             gridspec_kw={'height_ratios': [3, 1.6, 1.4]})
    fig.patch.set_facecolor(SURFACE)

    # panel 1: candles + cubic value lines
    ax = axes[0]
    _style(ax); _candles(ax, df, lo, hi)
    x = np.arange(lo, hi + 1)
    series = [('close', C_CLOSE, '-', 2.0, 'close-only cubic (live)'),
              ('rej10', C_REJ, '-', 2.0, 'rejection k=1.0'),
              ('rej05', C_REJ, '--', 1.4, 'rejection k=0.5'),
              ('exc', C_EXC, '-', 2.0, 'excursion HL2 blend')]
    for name, col, ls, lw, lab in series:
        ax.plot(x, V['cub'][name][0][lo:hi + 1], color=col, ls=ls, lw=lw,
                label=lab, zorder=3)
    _tell_line(ax, tell, f'bar {tell}')
    ax.legend(loc='upper left', frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title(p['title'], color=INK, fontsize=11, loc='left', pad=10)
    ax.set_ylabel('price', color=INK2, fontsize=9)

    # panel 2: cubic slopes + flag markers
    ax = axes[1]
    _style(ax)
    ax.axhline(0, color=BASELINE, lw=1.2, zorder=1)
    for name, col, ls, lw, lab in series:
        slp = V['cub'][name][1]
        ax.plot(x, slp[lo:hi + 1], color=col, ls=ls, lw=lw, zorder=3)
        flag = first_sustained(slp, p['scan_lo'], p['scan_hi'], -1)
        if flag is not None and lo <= flag <= hi:
            ax.plot([flag], [slp[flag]], marker='v', ms=8, color=col,
                    mec=SURFACE, mew=1.0, zorder=4)
    _tell_line(ax, tell)
    ax.set_ylabel('slope (pts/min)', color=INK2, fontsize=9)
    ax.set_title(f'cubic slope — triangle = first {SUSTAIN_BARS}-bar-sustained '
                 f'negative in scan [{p["scan_lo"]}..{p["scan_hi"]}]',
                 color=INK2, fontsize=9, loc='left')

    # panel 3: variant (b) companion — per-bar wick bias + trailing mean
    ax = axes[2]
    _style(ax)
    ax.axhline(0, color=BASELINE, lw=1.2, zorder=1)
    b = V['bias'][lo:hi + 1]
    ax.bar(x, b, width=0.64, color=[C_BIAS_POS if v >= 0 else C_DN for v in b],
           alpha=0.9, zorder=2,
           label='wick bias per bar  (lw-uw)/range, bullish +')
    ax.plot(x, V['bias_mean'][lo:hi + 1], color=INK, lw=2.0, zorder=3,
            label=f'{ws.CUBIC_W}-bar mean')
    _tell_line(ax, tell)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel('wick bias', color=INK2, fontsize=9)
    ax.legend(loc='upper left', frameon=False, fontsize=8, labelcolor=INK2)
    _xticks(ax, df, lo, hi)
    ax.set_xlabel(f'bar index ({p["day"]}, UTC)', color=INK2, fontsize=9)

    fig.align_ylabels(axes)
    fig.tight_layout()
    out = os.path.join(ASSETS, f'tell_{p["day"]}_bar{tell}.png')
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=['top', 'dip', 'both'], default='both')
    ap.add_argument('--no-stability', action='store_true')
    args = ap.parse_args()

    os.makedirs(ASSETS, exist_ok=True)
    results = {}
    for preset in (['top', 'dip'] if args.preset == 'both' else [args.preset]):
        p = PRESETS[preset]
        df = load_day(p['day'])
        for b, colname, expect in p['guards']:   # provenance: indexing must match
            got = float(df.loc[b, colname])      # the captured session narratives
            assert got == expect, f'{p["day"]} bar{b} {colname}={got} != {expect}'
        V = build_variants(df)
        results[preset] = analyze(preset, V, df)
        png = plot_preset(preset, V, df)
        results[preset]['plot'] = os.path.relpath(png, REPO)
        print(f'[{preset}] {p["day"]} plotted -> {png}')

    if not args.no_stability:
        results['stability'] = stability_scan()

    out = os.path.join(REPORTS, 'comparison_results.json')
    with open(out, 'w') as f:
        json.dump(_j(results), f, indent=1)
    print(f'results -> {out}')

    # console digest
    for preset in ('top', 'dip'):
        if preset not in results:
            continue
        r = results[preset]
        print(f'\n== {preset} ({r["day"]} bar {r["tell_bar"]}) '
              f'scan {r["scan_window"]} ==')
        for name, v in r['variants'].items():
            print(f'  {name:6s} down_flag={str(v["down_flag"]):>5s} '
                  f'lead_vs_close={str(v["lead_vs_close"]):>5s} '
                  f'up_flag={str(v["up_flag"]):>5s} '
                  f'slope_min={v["slope_min_in_window"]:+.2f}')
        vb = r['variant_b']
        print(f'  B: bias@tell={vb["bias_at_tell"]:+.2f} '
              f'mean@tell={vb["bias_mean_at_tell"]:+.2f} '
              f'neg_flag={vb["bias_mean_neg_flag"]} '
              f'divergence={vb["divergence_bars_priceUp_biasNeg"]}')
    if 'stability' in results:
        print('\n== stability (mean across days) ==')
        for name, v in results['stability']['mean_across_days'].items():
            extra = ('' if 'mean_abs_input_shift_pts' not in v else
                     f' input_shift={v["mean_abs_input_shift_pts"]:.2f}pts '
                     f'value_dev={v["mean_abs_value_dev_from_close_cubic_pts"]:.2f}pts')
            print(f'  {name:18s} flips/100={v["flips_per_100"]:.1f}{extra}')


if __name__ == '__main__':
    main()
