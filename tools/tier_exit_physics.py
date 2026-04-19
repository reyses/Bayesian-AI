"""
Tier Exit Physics Report — comprehensive dump of every physics quantity
useful for designing exits on a single tier.

For the given tier, loads trade pickle + 5s ATLAS closes and emits a full
report covering everything we'd want to answer with iterative Q2/Q3 EDA
in a single pass:

  1. Cohort summary (N, WR, avg/median pnl+peak, hold time, asymmetry)
  2. Peak TIMING — bar distribution where winners and losers hit max pnl
  3. Peak MAGNITUDE — pnl/peak distributions by cohort
  4. Bar-N trajectory — median PnL, MFE (running max), MAE (running min)
     at fixed bar checkpoints
  5. Fork analysis — bar where winner/loser median trajectories diverge
  6. Give-back from peak — winners retrace %, loser round-trip behavior
  7. Regression-mean slope β — velocity at entry, peak, exit; where does
     |β| decay to the "diminishing returns" threshold?  (1m horizon)
  8. Cut-rule scan — (bar_N × peak_threshold) grid, loser% − winner%
     delta at each cell; identifies every viable no-progress cut
  9. Peak-signature features — Cohen d for entry→peak delta (only if path
     features retained in the pickle)
  10. Entry-time discrimination — Cohen d winners vs losers at entry

Usage:
    python tools/tier_exit_physics.py --tier KILL_SHOT
    python tools/tier_exit_physics.py --tier KILL_SHOT_INVERSE
    python tools/tier_exit_physics.py --tier RIDE_AGAINST --trades training_iso/output/trades/iso_is_RIDE_AGAINST.pkl

Auto-resolves the pickle: if `--trades` not given, tries
`iso_is_<TIER>.pkl` first, falls back to `iso_is.pkl`.

Output:
    reports/findings/exit_physics_{TIER}.md
"""
import os
import sys
import pickle
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import FEATURE_NAMES


TRADES_DIR = 'training_iso/output/trades'
ATLAS_5S = 'DATA/ATLAS/5s'
OUT_DIR = 'reports/findings'

# Bar checkpoints
BAR_NS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 60]

# Cut rule peak thresholds
PEAK_THRESHOLDS = [1, 3, 5, 10, 15, 20]

# Slope computation window (12 × 5s = 60s)
SLOPE_WINDOW_BARS = 12

# "Diminishing returns" slope threshold (|β| below this = decay signal)
# Units: price per 5s bar. Calibrated off TREND_FOLLOWER EDA (2026-04-18).
SLOPE_DECAY_THRESHOLD = 0.05


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def resolve_pickle(tier: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    per_tier = os.path.join(TRADES_DIR, f'iso_is_{tier}.pkl')
    if os.path.exists(per_tier):
        return per_tier
    return os.path.join(TRADES_DIR, 'iso_is.pkl')


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def pnl_at_bar(trade: dict, bar_n: int) -> float | None:
    for p in trade.get('path', []):
        if p.get('bar', 0) == bar_n:
            return p.get('pnl', 0.0)
    # Trade didn't reach bar_n
    return None


def running_max_at_bar(trade: dict, bar_n: int) -> float | None:
    vals = [p.get('pnl', 0.0) for p in trade.get('path', [])
            if p.get('bar', 0) <= bar_n]
    return max(vals) if vals and trade.get('held', 0) >= bar_n else None


def running_min_at_bar(trade: dict, bar_n: int) -> float | None:
    vals = [p.get('pnl', 0.0) for p in trade.get('path', [])
            if p.get('bar', 0) <= bar_n]
    return min(vals) if vals and trade.get('held', 0) >= bar_n else None


def peak_bar(trade: dict) -> int | None:
    """Bar where trade hit its max pnl."""
    path = trade.get('path', [])
    if not path:
        return None
    best_bar, best_pnl = 0, path[0].get('pnl', 0.0)
    for p in path:
        if p.get('pnl', 0.0) > best_pnl:
            best_pnl = p.get('pnl', 0.0)
            best_bar = p.get('bar', 0)
    return best_bar


def ols_slope(closes: np.ndarray) -> float:
    """OLS β on a fixed-interval series. Units: price per step."""
    n = len(closes)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = closes.astype(np.float64)
    xm = x.mean()
    ym = y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - xm) * (y - ym)) / denom)


# ═══════════════════════════════════════════════════════════════════════
# Slope β — offline computation from 5s closes per day
# ═══════════════════════════════════════════════════════════════════════

class SlopeComputer:
    """Compute 1m-horizon regression-mean slope at arbitrary timestamps.

    Caches the 5s close series per-day. Call slope_at(day, ts) to get β
    over the most recent 12 5s bars ending at or before ts.
    """

    def __init__(self, sec_dir: str = ATLAS_5S):
        self.sec_dir = sec_dir
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _load(self, day: str) -> tuple[np.ndarray, np.ndarray] | None:
        if day in self._cache:
            return self._cache[day]
        path = os.path.join(self.sec_dir, f'{day}.parquet')
        if not os.path.exists(path):
            self._cache[day] = None
            return None
        df = pd.read_parquet(path).sort_values('timestamp')
        arr = (df['timestamp'].values.astype(np.int64),
               df['close'].values.astype(np.float64))
        self._cache[day] = arr
        return arr

    def slope_at(self, day: str, ts: int) -> float:
        data = self._load(day)
        if data is None:
            return 0.0
        ts_arr, close_arr = data
        # Find last bar <= ts
        idx = np.searchsorted(ts_arr, ts, side='right') - 1
        if idx < SLOPE_WINDOW_BARS - 1:
            return 0.0
        window = close_arr[idx - SLOPE_WINDOW_BARS + 1: idx + 1]
        return ols_slope(window)


# ═══════════════════════════════════════════════════════════════════════
# Analysis sections
# ═══════════════════════════════════════════════════════════════════════

def section_summary(winners, losers):
    w_pnl = np.array([t['pnl'] for t in winners])
    l_pnl = np.array([t['pnl'] for t in losers])
    w_peak = np.array([t.get('peak', 0.0) for t in winners])
    l_peak = np.array([t.get('peak', 0.0) for t in losers])
    w_held = np.array([t.get('held', 0) for t in winners])
    l_held = np.array([t.get('held', 0) for t in losers])
    return {
        'n_winners': len(winners),
        'n_losers': len(losers),
        'wr': len(winners) / max(len(winners) + len(losers), 1) * 100,
        'avg_w': float(w_pnl.mean()) if len(w_pnl) else 0.0,
        'avg_l': float(l_pnl.mean()) if len(l_pnl) else 0.0,
        'med_w': float(np.median(w_pnl)) if len(w_pnl) else 0.0,
        'med_l': float(np.median(l_pnl)) if len(l_pnl) else 0.0,
        'avg_peak_w': float(w_peak.mean()) if len(w_peak) else 0.0,
        'avg_peak_l': float(l_peak.mean()) if len(l_peak) else 0.0,
        'med_held_w': int(np.median(w_held)) if len(w_held) else 0,
        'med_held_l': int(np.median(l_held)) if len(l_held) else 0,
        'asymmetry': (abs(l_pnl.mean()) / max(w_pnl.mean(), 1e-9)
                      if len(w_pnl) and w_pnl.mean() > 0 else 0.0),
        'sum_w': float(w_pnl.sum()) if len(w_pnl) else 0.0,
        'sum_l': float(l_pnl.sum()) if len(l_pnl) else 0.0,
    }


def section_peak_timing(winners, losers):
    """Bar where max pnl occurred — for winners and losers."""
    def stats(trades):
        bars = [peak_bar(t) for t in trades]
        bars = [b for b in bars if b is not None]
        if not bars:
            return {}
        arr = np.array(bars)
        return {
            'n': len(arr),
            'mean': float(arr.mean()),
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
            'p95': float(np.percentile(arr, 95)),
        }
    return {'winners': stats(winners), 'losers': stats(losers)}


def section_bar_trajectory(winners, losers):
    """Median PnL, MFE, MAE at each bar checkpoint."""
    rows = []
    for N in BAR_NS:
        def median_over(fn, grp):
            vals = [fn(t, N) for t in grp]
            vals = [v for v in vals if v is not None]
            return float(np.median(vals)) if vals else None
        row = {
            'bar_N': N,
            'n_w': sum(1 for t in winners if t.get('held', 0) >= N),
            'n_l': sum(1 for t in losers if t.get('held', 0) >= N),
            'pnl_w': median_over(pnl_at_bar, winners),
            'pnl_l': median_over(pnl_at_bar, losers),
            'mfe_w': median_over(running_max_at_bar, winners),
            'mfe_l': median_over(running_max_at_bar, losers),
            'mae_w': median_over(running_min_at_bar, winners),
            'mae_l': median_over(running_min_at_bar, losers),
        }
        rows.append(row)
    return rows


def section_fork_bar(traj_rows, threshold: float = 5.0):
    """Smallest bar where median(winner pnl) - median(loser pnl) >= threshold."""
    for row in traj_rows:
        w = row.get('pnl_w')
        l = row.get('pnl_l')
        if w is not None and l is not None and (w - l) >= threshold:
            return {'bar_N': row['bar_N'], 'diff': w - l, 'w': w, 'l': l}
    return None


def section_giveback(winners, losers):
    """For each trade: (peak - final) / peak = retrace fraction. Also
    compute how many losers had a peak (round-tripped through positive)."""
    def stats(trades):
        retraces = []
        round_trip = 0
        for t in trades:
            peak = t.get('peak', 0.0)
            final = t.get('pnl', 0.0)
            if peak > 0:
                retrace = (peak - final) / peak
                retraces.append(retrace)
            if t.get('pnl', 0.0) < 0 and peak >= 5.0:
                round_trip += 1
        arr = np.array(retraces) if retraces else np.array([])
        return {
            'n_with_peak': len(retraces),
            'mean_retrace_pct': float(arr.mean() * 100) if len(arr) else 0.0,
            'median_retrace_pct': float(np.median(arr) * 100) if len(arr) else 0.0,
            'round_trip_losers': round_trip,
        }
    return {'winners': stats(winners), 'losers': stats(losers)}


def section_slope(trades, winners, losers, slope_comp: SlopeComputer):
    """Slope β at entry, at peak, at exit for each cohort."""
    def stats(group, label):
        entries, peaks, exits = [], [], []
        for t in group:
            day = t.get('day')
            if not day:
                continue
            path = t.get('path', [])
            if not path:
                continue
            entry_ts = path[0].get('timestamp')
            exit_ts = path[-1].get('timestamp')
            pb = peak_bar(t)
            peak_ts = None
            for p in path:
                if p.get('bar') == pb:
                    peak_ts = p.get('timestamp')
                    break
            if entry_ts:
                entries.append(slope_comp.slope_at(day, int(entry_ts)))
            if peak_ts:
                peaks.append(slope_comp.slope_at(day, int(peak_ts)))
            if exit_ts:
                exits.append(slope_comp.slope_at(day, int(exit_ts)))
        def summarize(arr):
            if not arr:
                return {'n': 0, 'mean_abs': 0.0, 'pct_decayed': 0.0}
            a = np.array(arr)
            return {
                'n': len(a),
                'mean_abs': float(np.abs(a).mean()),
                'median_abs': float(np.median(np.abs(a))),
                'pct_decayed': float((np.abs(a) < SLOPE_DECAY_THRESHOLD).mean() * 100),
            }
        return {
            'entry': summarize(entries),
            'peak': summarize(peaks),
            'exit': summarize(exits),
        }
    return {'winners': stats(winners, 'win'), 'losers': stats(losers, 'los')}


def section_cut_scan(winners, losers):
    """For each (bar, thr), what fraction of each cohort has peak_pnl < thr?"""
    def frac_below(group, bar_n, thr):
        vals = [running_max_at_bar(t, bar_n) for t in group]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return sum(1 for v in vals if v < thr) / len(vals) * 100
    rows = []
    for N in BAR_NS:
        row = {'bar_N': N}
        for thr in PEAK_THRESHOLDS:
            w = frac_below(winners, N, thr)
            l = frac_below(losers, N, thr)
            row[f'w_{thr}'] = w
            row[f'l_{thr}'] = l
            row[f'd_{thr}'] = (l - w) if (w is not None and l is not None) else None
        rows.append(row)
    return rows


def section_entry_disc(winners, losers, top_k=15):
    W, L = [], []
    for t in winners:
        ef = t.get('entry_79d')
        if ef is not None and len(ef) >= 91:
            W.append(ef[:91])
    for t in losers:
        ef = t.get('entry_79d')
        if ef is not None and len(ef) >= 91:
            L.append(ef[:91])
    if not W or not L:
        return []
    W, L = np.array(W, dtype=float), np.array(L, dtype=float)
    out = []
    for i, name in enumerate(FEATURE_NAMES[:91]):
        d = cohen_d(W[:, i], L[:, i])
        out.append((name, d, float(W[:, i].mean()), float(L[:, i].mean())))
    out.sort(key=lambda x: abs(x[1]), reverse=True)
    return out[:top_k]


def section_peak_signature(trades, top_k=15):
    """Cohen d for entry→peak feature delta (requires path features)."""
    deltas = defaultdict(list)
    have_path_features = False
    for t in trades:
        entry = t.get('entry_79d')
        path = t.get('path', [])
        pb = peak_bar(t)
        if entry is None or not path or pb is None:
            continue
        peak_feat = None
        for p in path:
            if p.get('bar') == pb:
                peak_feat = p.get('features')
                break
        if peak_feat is None:
            continue
        have_path_features = True
        entry_arr = np.array(entry[:91], dtype=float)
        peak_arr = np.array(peak_feat[:91], dtype=float)
        delta = peak_arr - entry_arr
        for i in range(91):
            deltas[FEATURE_NAMES[i]].append(delta[i])
    if not have_path_features:
        return None
    ranked = []
    for name, arr in deltas.items():
        a = np.array(arr)
        if len(a) < 5:
            continue
        # "Effect size" = mean / std of delta (one-sample d against 0)
        mean = float(a.mean())
        std = float(a.std(ddof=1))
        if std == 0:
            continue
        d_norm = mean / std
        ranked.append((name, d_norm, mean, std, len(a)))
    ranked.sort(key=lambda x: abs(x[1]), reverse=True)
    return ranked[:top_k]


# ═══════════════════════════════════════════════════════════════════════
# Report writer
# ═══════════════════════════════════════════════════════════════════════

def write_report(tier, summary, timing, traj, fork, giveback, slope, cut_rows,
                 entry_disc, peak_sig, out_path):
    L = []
    L.append(f'# Tier Exit Physics — {tier}')
    L.append('')
    L.append('## 1. Cohort summary')
    L.append('')
    L.append(f'- **N:** {summary["n_winners"] + summary["n_losers"]:,}  '
             f'WR **{summary["wr"]:.1f}%**  '
             f'(winners {summary["n_winners"]:,} / losers {summary["n_losers"]:,})')
    L.append(f'- **Avg PnL:** winner ${summary["avg_w"]:+.2f}, '
             f'loser ${summary["avg_l"]:+.2f}  '
             f'(asymmetry {summary["asymmetry"]:.2f}×)')
    L.append(f'- **Median PnL:** winner ${summary["med_w"]:+.2f}, '
             f'loser ${summary["med_l"]:+.2f}')
    L.append(f'- **Avg peak:** winner ${summary["avg_peak_w"]:+.2f}, '
             f'loser ${summary["avg_peak_l"]:+.2f}')
    L.append(f'- **Median hold:** winner {summary["med_held_w"]}m, '
             f'loser {summary["med_held_l"]}m')
    L.append(f'- **Total PnL:** winners ${summary["sum_w"]:+,.0f}, '
             f'losers ${summary["sum_l"]:+,.0f}, '
             f'net ${summary["sum_w"] + summary["sum_l"]:+,.0f}')
    L.append('')

    L.append('## 2. Peak timing — when does max pnl occur?')
    L.append('')
    L.append('| Cohort | n | mean | p25 | p50 | p75 | p90 | p95 |')
    L.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for label, stats in [('winners', timing['winners']), ('losers', timing['losers'])]:
        if stats:
            L.append(f'| {label} | {stats["n"]} | {stats["mean"]:.1f} | '
                     f'{stats["p25"]:.0f} | {stats["p50"]:.0f} | '
                     f'{stats["p75"]:.0f} | {stats["p90"]:.0f} | {stats["p95"]:.0f} |')
    L.append('')
    L.append('_Bar at which the trade reached its maximum PnL. Low p50 = '
             'winners peak fast (capture early). High p50 = winners develop '
             'slowly (hold longer)._')
    L.append('')

    L.append('## 3. Bar-N trajectory (median PnL / MFE / MAE by cohort)')
    L.append('')
    L.append('| bar | n_w | n_l | pnl_w | pnl_l | mfe_w | mfe_l | mae_w | mae_l |')
    L.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    def fmt(x):
        return f'${x:+.1f}' if x is not None else '—'
    for r in traj:
        L.append(f'| {r["bar_N"]} | {r["n_w"]} | {r["n_l"]} | '
                 f'{fmt(r["pnl_w"])} | {fmt(r["pnl_l"])} | '
                 f'{fmt(r["mfe_w"])} | {fmt(r["mfe_l"])} | '
                 f'{fmt(r["mae_w"])} | {fmt(r["mae_l"])} |')
    L.append('')
    if fork:
        L.append(f'**Fork bar (winner − loser ≥ $5):** bar {fork["bar_N"]}  '
                 f'(winner ${fork["w"]:+.1f} vs loser ${fork["l"]:+.1f}, '
                 f'spread ${fork["diff"]:.1f})')
    else:
        L.append('_No clean fork at the $5 separator. Trajectories overlap._')
    L.append('')

    L.append('## 4. Give-back from peak')
    L.append('')
    L.append('| Cohort | n_with_peak | median retrace % | mean retrace % | round-trip losers |')
    L.append('|---|---:|---:|---:|---:|')
    for label, s in [('winners', giveback['winners']), ('losers', giveback['losers'])]:
        L.append(f'| {label} | {s["n_with_peak"]} | {s["median_retrace_pct"]:.0f}% | '
                 f'{s["mean_retrace_pct"]:.0f}% | {s["round_trip_losers"]} |')
    L.append('')
    L.append('_Retrace % = (peak − final) / peak. High retrace = winners give '
             'back gains before exit → trail stop helps. Round-trip losers = '
             'trades that had a peak ≥ $5 but exited negative → those were '
             'catchable with a tighter trail or trailing peak-rule gate._')
    L.append('')

    L.append('## 5. Regression-mean slope β (1m horizon, 12 × 5s)')
    L.append('')
    L.append('|β| is the magnitude of price drift over the last 60s. '
             '"% decayed" = fraction of trades where |β| < '
             f'{SLOPE_DECAY_THRESHOLD:.2f} (diminishing-returns threshold).')
    L.append('')
    L.append('| Cohort | checkpoint | n | mean |β| | median |β| | % decayed |')
    L.append('|---|---|---:|---:|---:|---:|')
    for label, st in [('winners', slope['winners']), ('losers', slope['losers'])]:
        for cp in ('entry', 'peak', 'exit'):
            s = st[cp]
            if s['n'] > 0:
                L.append(f'| {label} | {cp} | {s["n"]} | '
                         f'{s["mean_abs"]:.3f} | {s["median_abs"]:.3f} | '
                         f'{s["pct_decayed"]:.0f}% |')
    L.append('')
    L.append('_Winners' " entry→peak → |β| growth = riding acceleration. "
             'Winners peak→exit → |β| decay = natural slow-down (use as '
             'trailing exit signal). Losers entry β tells us whether the '
             'setup fires in a stagnant or trending regime._')
    L.append('')

    L.append('## 6. Cut-rule scan — (bar × peak threshold) loser%−winner% delta')
    L.append('')
    L.append('Each cell = (winner% < thr) / (loser% < thr) / Δ. `**` = Δ ≥ 20pp '
             '(strong cut candidate). `*` = Δ ≥ 15pp (moderate).')
    L.append('')
    hdr = ['bar'] + [f'<${t}' for t in PEAK_THRESHOLDS]
    L.append('| ' + ' | '.join(hdr) + ' |')
    L.append('|' + '|'.join(['---:'] * len(hdr)) + '|')
    for r in cut_rows:
        cells = [str(r['bar_N'])]
        for thr in PEAK_THRESHOLDS:
            w = r.get(f'w_{thr}')
            l = r.get(f'l_{thr}')
            d = r.get(f'd_{thr}')
            if w is None or l is None:
                cells.append('—')
            else:
                mark = ' **' if d >= 20 else ('  *' if d >= 15 else '')
                cells.append(f'{w:.0f}/{l:.0f}/{d:+.0f}{mark}')
        L.append('| ' + ' | '.join(cells) + ' |')
    L.append('')

    L.append('## 7. Entry-time discrimination (Cohen d: winners vs losers at entry)')
    L.append('')
    if entry_disc:
        L.append('| feature | d | W mean | L mean |')
        L.append('|---|---:|---:|---:|')
        for name, d, w_mu, l_mu in entry_disc:
            mark = ' **' if abs(d) >= 0.5 else ('  *' if abs(d) >= 0.3 else '')
            L.append(f'| {name} | {d:+.3f}{mark} | {w_mu:.3f} | {l_mu:.3f} |')
        top_d = abs(entry_disc[0][1])
        L.append('')
        if top_d >= 0.5:
            L.append('**Strong entry separator found — consider entry gate.**')
        elif top_d >= 0.3:
            L.append('_Moderate separator. Entry gate marginal; exit rules preferred._')
        else:
            L.append('_No entry discrimination. Winners and losers identical at '
                     'entry → the fix is an exit rule, not an entry filter._')
    else:
        L.append('_No entry features available in pickle._')
    L.append('')

    L.append('## 8. Peak-signature features (entry→peak delta, Cohen d/σ)')
    L.append('')
    if peak_sig is None:
        L.append('_Peak signature unavailable — path features were stripped '
                 'from pickle (run single-tier for features-intact pickle).  '
                 'Re-run with `--tier ' + tier + '` to populate this section._')
    elif not peak_sig:
        L.append('_Not enough feature samples to compute peak signature._')
    else:
        L.append('Top features where value shifts most from entry to peak.  '
                 '`d/σ > 2` is a strong signal; `> 5` is dominant.')
        L.append('')
        L.append('| feature | d/σ | mean Δ | std Δ | n |')
        L.append('|---|---:|---:|---:|---:|')
        for name, dnorm, mean, std, n in peak_sig:
            mark = ' **' if abs(dnorm) >= 5 else ('  *' if abs(dnorm) >= 2 else '')
            L.append(f'| {name} | {dnorm:+.2f}{mark} | {mean:+.3f} | '
                     f'{std:.3f} | {n} |')
    L.append('')

    L.append('## 9. Rule candidates (synthesized)')
    L.append('')
    # Synthesize best candidates automatically
    # Best cut rule = highest-delta cell
    best_cut = None
    for r in cut_rows:
        for thr in PEAK_THRESHOLDS:
            d = r.get(f'd_{thr}')
            if d is None:
                continue
            if best_cut is None or d > best_cut[2]:
                best_cut = (r['bar_N'], thr, d, r.get(f'w_{thr}'), r.get(f'l_{thr}'))
    if best_cut:
        bar_n, thr, d, w, l = best_cut
        L.append(f'- **Cut candidate:** `bars_held >= {bar_n} AND peak_pnl < ${thr}`  '
                 f'(Δ={d:+.0f}pp, cuts {l:.0f}% of losers / {w:.0f}% of winners)')
    if timing['winners']:
        wp50 = int(timing['winners']['p50'])
        wp90 = int(timing['winners']['p90'])
        L.append(f'- **Winner peak p50/p90:** {wp50}m / {wp90}m — timeout '
                 f'candidate at p90 + buffer ≈ {wp90 + 5}m')
    if giveback['winners']['median_retrace_pct'] >= 30:
        L.append(f'- **Trail candidate:** winners give back '
                 f'{giveback["winners"]["median_retrace_pct"]:.0f}% median → '
                 f'consider trailing-peak exit at ~50% retrace')
    slope_decay_w = slope['winners']['peak']['pct_decayed']
    if slope_decay_w >= 30:
        L.append(f'- **Slope-decay exit:** {slope_decay_w:.0f}% of winner peaks '
                 f'occur when |β| < {SLOPE_DECAY_THRESHOLD} → diminishing-returns '
                 f'exit rule viable')
    L.append('')
    L.append('---')
    L.append(f'_Generated by `tools/tier_exit_physics.py --tier {tier}`_')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True, help='tier name to analyze')
    ap.add_argument('--trades', default=None, help='pickle path override')
    ap.add_argument('--out', default=None, help='output md path')
    ap.add_argument('--no-slope', action='store_true',
                    help='skip slope β analysis (faster)')
    args = ap.parse_args()

    pkl_path = resolve_pickle(args.tier, args.trades)
    print(f'Loading {pkl_path}...')
    with open(pkl_path, 'rb') as f:
        trades = pickle.load(f)
    print(f'  {len(trades):,} total trades')
    counts = Counter(t.get('entry_tier', '?') for t in trades)
    print(f'  Tiers in file: {dict(counts)}')

    sub = [t for t in trades if t.get('entry_tier') == args.tier]
    if not sub:
        print(f'ERROR: no trades for tier {args.tier!r} in {pkl_path}')
        return
    winners = [t for t in sub if t.get('pnl', 0) > 0]
    losers = [t for t in sub if t.get('pnl', 0) < 0]
    print(f'  {args.tier}: {len(sub):,} trades  '
          f'({len(winners):,} W / {len(losers):,} L)')

    print()
    print('[1/8] Cohort summary...')
    summary = section_summary(winners, losers)
    print('[2/8] Peak timing...')
    timing = section_peak_timing(winners, losers)
    print('[3/8] Bar-N trajectory...')
    traj = section_bar_trajectory(winners, losers)
    fork = section_fork_bar(traj, threshold=5.0)
    print('[4/8] Give-back from peak...')
    giveback = section_giveback(winners, losers)
    print('[5/8] Slope beta (entry/peak/exit)...')
    if args.no_slope:
        slope = {'winners': {'entry': {'n': 0, 'mean_abs': 0, 'median_abs': 0, 'pct_decayed': 0},
                             'peak': {'n': 0, 'mean_abs': 0, 'median_abs': 0, 'pct_decayed': 0},
                             'exit': {'n': 0, 'mean_abs': 0, 'median_abs': 0, 'pct_decayed': 0}},
                 'losers':  {'entry': {'n': 0, 'mean_abs': 0, 'median_abs': 0, 'pct_decayed': 0},
                             'peak': {'n': 0, 'mean_abs': 0, 'median_abs': 0, 'pct_decayed': 0},
                             'exit': {'n': 0, 'mean_abs': 0, 'median_abs': 0, 'pct_decayed': 0}}}
    else:
        slope_comp = SlopeComputer()
        slope = section_slope(sub, winners, losers, slope_comp)
    print('[6/8] Cut-rule scan...')
    cut_rows = section_cut_scan(winners, losers)
    print('[7/8] Entry-time discrimination...')
    entry_disc = section_entry_disc(winners, losers)
    print('[8/8] Peak-signature features...')
    peak_sig = section_peak_signature(sub)

    out_path = args.out or os.path.join(OUT_DIR, f'exit_physics_{args.tier}.md')
    write_report(args.tier, summary, timing, traj, fork, giveback, slope,
                 cut_rows, entry_disc, peak_sig, out_path)
    print()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
