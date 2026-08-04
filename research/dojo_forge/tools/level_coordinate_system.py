"""LEVEL COORDINATE SYSTEM — codify the owner's hand-drawn reference levels
(owner 2026-07-28, "touch as many highs/lows as possible, or be as near as
possible ... it is a region expressed as a line").

The owner's placement rule is a 1D clustering objective: place K horizontal
lines to MINIMIZE the distance from every swing high/low to its nearest line.
Lines sit at the DENSITY PEAKS of the pivot-price distribution. A line is a
REGION (band ±TAU), not a hairline. A STRONG line is revisited at temporally
SEPARATED times (Fig 5: far-right + middle + far-left touches) — so line
strength = count of DISTINCT-TIME touches within the band, not raw pivot count.

Pipeline:
  1. load N days of bars (concat for higher-TF context)
  2. causal zigzag pivots at radius R -> snap to bar high (peak) / low (trough)
  3. k-medians on pivot PRICES (L1 = "be as near as possible"); K grows until
     coverage (% pivots within TAU) hits target, capped at KMAX
  4. score each line by distinct-time touches + temporal spread
  5. metrics (mean/median pivot->line dist, % within TAU) = the acceptance test
  6. render overlay + emit per-bar coordinate features
     (norm position between bracketing lines, signed dist to nearest up/down)

Run:
  python research/dojo_forge/tools/level_coordinate_system.py \
      --days 2026_07_14 2026_07_15 2026_07_16 --R 25 --tau 12 --kmax 8
Outputs -> research/dojo_forge/reports/human_dojo/levels_<lastday>.{png,json}
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
if __name__ == '__main__':          # headless only when run as a script — as an
    import matplotlib               # import (recorder plugin) keep the caller's
    matplotlib.use('Agg')           # interactive backend intact
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA = os.path.join(REPO, 'DATA', 'ATLAS_NT8', '1m')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'human_dojo')


def load_days(days):
    """Concatenate 1m OHLC for the given YYYY_MM_DD day list (chronological)."""
    frames = []
    for d in days:
        p = os.path.join(DATA, f'{d}.parquet')
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        df = pd.read_parquet(p)[['timestamp', 'open', 'high', 'low', 'close']].copy()
        df['day'] = d
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def zigzag_pivots(high, low, close, R, opn=None):
    """Causal zigzag on close. Each confirmed pivot contributes BOTH its wick
    extreme (high/low) and its body edge (max/min of open,close) as candidate
    touch prices — the owner catches "either wicks or bodies", so the density
    fit sees both and a line settles wherever more touches pile up. Returns
    list of (idx, price, kind) kind in {'H','L'}; wick and body rows share idx
    (same pivot EVENT — dedup by idx when counting distinct-time touches)."""
    n = len(close)
    hi = lo = close[0]; hii = loi = 0; d = 0
    out = []
    def emit(i, kind):
        if kind == 'H':
            wick = float(high[i]); body = float(max(opn[i], close[i])) if opn is not None else wick
        else:
            wick = float(low[i]); body = float(min(opn[i], close[i])) if opn is not None else wick
        out.append((i, wick, kind))
        if abs(body - wick) > 1e-9:
            out.append((i, body, kind))
    for i in range(1, n):
        p = close[i]
        if not np.isfinite(p):
            continue
        if d >= 0 and p > hi: hi, hii = p, i
        if d <= 0 and p < lo: lo, loi = p, i
        if d >= 0 and hi - p >= R:
            emit(hii, 'H'); d = -1; lo, loi = p, i
        elif d <= 0 and p - lo >= R:
            emit(loi, 'L'); d = 1; hi, hii = p, i
    return out


def kmedians_1d(prices, k, iters=50):
    """1D k-medians (L1). Deterministic quantile init. Returns sorted centers."""
    prices = np.sort(np.asarray(prices, float))
    if k >= len(prices):
        return prices.copy()
    q = np.linspace(0, 1, k + 2)[1:-1]
    c = np.quantile(prices, q)
    for _ in range(iters):
        # assign to nearest center
        idx = np.abs(prices[:, None] - c[None, :]).argmin(1)
        newc = c.copy()
        for j in range(k):
            m = prices[idx == j]
            if len(m):
                newc[j] = np.median(m)          # median = L1 optimum
        if np.allclose(newc, c):
            break
        c = np.sort(newc)
    return np.sort(c)


def fit_levels(pivots, tau, kmax, min_touches=3, sep=None):
    """GREEDY DENSITY-PEAK picking — the owner's rule "touch as many highs/lows
    as possible per line". Repeatedly place a line at the price with the most
    distinct-time pivots within +/-TAU, then non-max-suppress (remove pivots it
    claimed + forbid another line within SEP). Stop at KMAX or when the best
    remaining peak has < MIN_TOUCHES. Refine each center to the L1-median of its
    claimed pivots (a region -> a line). Returns (lines, dist_to_nearest, None)."""
    prices = np.array([p for _, p, _ in pivots], float)
    times = np.array([i for i, _, _ in pivots], float)
    sep = sep if sep is not None else 2 * tau        # min gap between lines (a
    # region is ±tau wide; adjacent regions should not overlap or the "lines"
    # fragment into clutter the owner would never draw)
    claimed = np.zeros(len(prices), bool)
    lines = []
    # candidate centers: every pivot price (dense enough for 1D)
    for _ in range(kmax):
        avail = ~claimed
        if avail.sum() < min_touches:
            break
        cand = prices[avail]
        best_c, best_n, best_mask = None, -1, None
        for c in np.unique(np.round(cand / (tau / 2)) * (tau / 2)):
            m = avail & (np.abs(prices - c) <= tau)
            # distinct-time touches: wick+body rows from the same pivot share an
            # idx — count unique EVENTS, not rows
            n = int(len(np.unique(times[m])))
            if n > best_n and (not lines or min(abs(c - L['price']) for L in lines) >= sep):
                best_c, best_n, best_mask = c, n, m
        if best_c is None or best_n < min_touches:
            break
        mprice = float(np.median(prices[best_mask]))   # L1 region -> line
        tt = np.unique(times[best_mask])               # distinct pivot events
        lines.append({
            'price': mprice, 'touches': int(best_n),
            'temporal_spread_bars': float(tt.max() - tt.min()) if len(tt) > 1 else 0.0,
            'first_bar': float(tt.min()), 'last_bar': float(tt.max()),
        })
        claimed |= best_mask
    lines.sort(key=lambda L: L['price'])
    centers = np.array([L['price'] for L in lines], float) if lines else np.array([np.median(prices)])
    dist = np.abs(prices[:, None] - centers[None, :]).min(1)
    trace = [{'captured_pct': round(float((dist <= tau).mean()), 3),
              'lines': len(lines)}]
    return lines, dist, trace


# TF TELESCOPE: coarse R / long window -> few big lines; fine R / recent window
# -> micro lines. Anchored at the RIGHT edge (current price), like the owner's
# day -> 4h -> 1h -> now descent. lookback in 1m bars (None = all).
TELESCOPE = [
    {'name': 'day', 'lookback': None, 'R': 60, 'tau': 22, 'kmax': 5, 'min_touches': 4,
     'color': '#0D47A1', 'lw': 3.2},
    {'name': '4h',  'lookback': 300,  'R': 28, 'tau': 12, 'kmax': 5, 'min_touches': 3,
     'color': '#1976D2', 'lw': 2.2},
    {'name': '1h',  'lookback': 120,  'R': 10, 'tau': 6,  'kmax': 6, 'min_touches': 2,
     'color': '#42A5F5', 'lw': 1.3},
]


def telescope(df, scales=TELESCOPE):
    """Fit levels at each scale over its own recent window. Returns list of
    per-scale results, coarsest first. Dedup: a finer line within its tau of an
    already-placed coarser line is dropped (the coarse line owns that region)."""
    high = df['high'].to_numpy(); low = df['low'].to_numpy(); close = df['close'].to_numpy()
    opn = df['open'].to_numpy()
    n = len(close)
    placed = []            # (price, tau) already owned by a coarser scale
    out = []
    for sc in scales:
        lb = sc['lookback']
        s = 0 if lb is None else max(0, n - lb)
        piv = zigzag_pivots(high[s:], low[s:], close[s:], sc['R'], opn[s:])
        piv = [(i + s, p, k) for i, p, k in piv]          # shift idx back to global
        if len(piv) < sc['min_touches']:
            out.append({**sc, 'lines': [], 'pivots': piv, 'start': s}); continue
        lines, dist, _ = fit_levels(piv, sc['tau'], sc['kmax'], sc['min_touches'])
        keep = []
        for L in lines:
            if any(abs(L['price'] - pp) <= max(pt, sc['tau']) for pp, pt in placed):
                continue                                    # owned by coarser scale
            keep.append(L); placed.append((L['price'], sc['tau']))
        out.append({**sc, 'lines': keep, 'pivots': piv, 'start': s})
    return out


def coordinate_features(close, lines):
    """Per-bar local-coordinate features against the fitted lines."""
    prices = np.array([L['price'] for L in lines], float)
    n = len(close)
    up = np.full(n, np.nan); dn = np.full(n, np.nan); pos = np.full(n, np.nan)
    for i in range(n):
        p = close[i]
        above = prices[prices > p]; below = prices[prices < p]
        u = above.min() if len(above) else np.nan
        d = below.max() if len(below) else np.nan
        up[i] = u - p if np.isfinite(u) else np.nan          # room to next line up
        dn[i] = p - d if np.isfinite(d) else np.nan          # room to next line down
        if np.isfinite(u) and np.isfinite(d) and u > d:
            pos[i] = (p - d) / (u - d)                        # 0=on lower line, 1=upper
    return {'dist_up': up, 'dist_dn': dn, 'norm_pos': pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', nargs='+', required=True)
    ap.add_argument('--view', type=int, default=480, help='bars to render at the right edge')
    args = ap.parse_args()

    df = load_days(args.days)
    close = df['close'].to_numpy()
    n = len(close)
    scales = telescope(df)

    # union of all lines -> coordinate features against the full nested grid
    all_lines = [L for sc in scales for L in sc['lines']]
    feats = coordinate_features(close, all_lines)

    report = {'days': args.days, 'n_bars': n, 'scales': []}
    for sc in scales:
        report['scales'].append({
            'name': sc['name'], 'R': sc['R'], 'tau': sc['tau'],
            'lookback': sc['lookback'], 'n_pivots': len(sc['pivots']),
            'n_lines': len(sc['lines']),
            'lines': [{'price': L['price'], 'touches': L['touches'],
                       'spread_bars': int(L['temporal_spread_bars'])} for L in sc['lines']],
        })
    os.makedirs(OUT, exist_ok=True)
    last = args.days[-1]
    with open(os.path.join(OUT, f'levels_{last}.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # ---- render: price (right-edge view) + nested telescope lines ----
    v0 = max(0, n - args.view)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(v0, n)
    ax.plot(x, close[v0:], color='#37474F', lw=0.9, zorder=3)
    vlo, vhi = close[v0:].min(), close[v0:].max(); pad = (vhi - vlo) * 0.08
    for sc in scales:
        for L in sc['lines']:
            if not (vlo - pad <= L['price'] <= vhi + pad):
                continue
            ax.axhline(L['price'], color=sc['color'], lw=sc['lw'], alpha=0.7, zorder=2)
            ax.axhspan(L['price'] - sc['tau'], L['price'] + sc['tau'],
                       color=sc['color'], alpha=0.05, zorder=1)
            ax.text(n + args.view * 0.005, L['price'],
                    f"{sc['name']} {L['price']:.0f} ({L['touches']}t)",
                    va='center', fontsize=7.5, color=sc['color'])
    for d in args.days[1:]:
        b = df.index[df['day'] == d][0]
        if b >= v0:
            ax.axvline(b, color='#B0BEC5', lw=0.6, ls=':', zorder=1)
    ax.set_xlim(v0, n + args.view * 0.10)
    ax.set_ylim(vlo - pad, vhi + pad)
    counts = ' · '.join(f"{sc['name']}:{len(sc['lines'])}" for sc in scales)
    ax.set_title(f"TF TELESCOPE — LEVEL COORDINATE SYSTEM {'/'.join(args.days)}  "
                 f"[{counts}] (thick=coarse day, thin=fine 1h)", fontsize=10)
    ax.set_xlabel('bar'); ax.set_ylabel('price')
    fig.tight_layout()
    png = os.path.join(OUT, f'levels_{last}.png')
    fig.savefig(png, dpi=110)

    for sc in scales:
        print(f"{sc['name']:>4} R={sc['R']:>3} τ={sc['tau']:>2}  {len(sc['lines'])} lines: "
              + ', '.join(f"{L['price']:.0f}({L['touches']}t)" for L in sc['lines']))
    print('->', png)


if __name__ == '__main__':
    main()
