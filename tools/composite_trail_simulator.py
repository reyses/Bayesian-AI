"""Composite trail simulator — tests trade-management value of the
pivot probability cloud + leg-phase composite.

Hypothesis (user 2026-05-17): "if i know a flip is coming then we can
tighten to prepare for the flip even if we catch it and act on it in the
10 seconds after is still a win".

Method:
  For each zigzag leg in NT8 OOS, simulate entry at start of leg
  (extreme price of last pivot) and run TWO exit policies in parallel:

    BASELINE: trail-stop at R = 4xATR ticks (the indicator's R-trigger).
              Exit when 5s close crosses extreme +/- R.

    COMPOSITE: trail-stop distance ADAPTS to the composite zone at each
               bar. When zone says pivot-near, trail tightens dramatically.

       Zone -> trail_distance multiplier:
         CLEAR         : 1.00 * R   (full wide)
         WATCH         : 0.75 * R
         WIDE_ZONE     : 0.50 * R
         NEAR_5m       : 0.40 * R
         NEAR_3m       : 0.30 * R
         IMMINENT      : 0.20 * R
         NEAR_PIVOT    : 0.20 * R
         AT_PIVOT      : 0.10 * R   (very tight — flip-ready)

Output per leg:
  - leg_dir (LONG/SHORT)
  - entry_price (last pivot's extreme)
  - actual_peak_price (most favorable price during leg)
  - baseline_exit_price + baseline_exit_ts
  - composite_exit_price + composite_exit_ts
  - edge_pts = composite captured MORE of the move (signed)
  - edge_usd = edge_pts * 2.0 (MNQ multiplier)

Aggregate stats per day, per regime, per zone seen during exit.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_zigzag_baseline import compute_atr, TICK_SIZE


DOLLAR_PER_POINT = 2.0   # MNQ
TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')
REGIME_CSV = Path('DATA/ATLAS/regime_labels_2d.csv')
SEGMENT_MIN = 90


# Zone -> trail-distance multiplier on R-price
ZONE_TRAIL_MULT = {
    'CLEAR':       1.00,
    'WATCH':       0.75,
    'WIDE_ZONE':   0.50,
    'NEAR_5m':     0.40,
    'NEAR_3m':     0.30,
    'IMMINENT':    0.20,
    'NEAR_PIVOT':  0.20,
    'AT_PIVOT':    0.10,
}


def load_regime_label(day_str: str):
    if not REGIME_CSV.exists():
        return None
    df = pd.read_csv(REGIME_CSV, usecols=['date', 'regime_2d'])
    iso_day = day_str.replace('_', '-')
    row = df[df['date'] == iso_day]
    if len(row) == 0:
        return None
    return str(row['regime_2d'].iloc[0])


def derive_pivot_events(truth_day: pd.DataFrame):
    """Return list of (centroid_ts, pivot_dir, pivot_price) per pivot."""
    piv = truth_day[truth_day['is_pivot'] == 1].sort_values('timestamp')
    if len(piv) == 0:
        return []
    ts = piv['timestamp'].values.astype(np.int64)
    pd_ = piv['pivot_dir'].values
    pp_ = piv['pivot_price'].values
    groups = [[0]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] > 90:
            groups.append([i])
        else:
            groups[-1].append(i)
    out = []
    for grp in groups:
        ts_c = int(np.median(ts[grp]))
        vals, counts = np.unique(pd_[grp], return_counts=True)
        d = str(vals[np.argmax(counts)])
        p = float(np.mean(pp_[grp]))
        out.append((ts_c, d, p))
    return out


def simulate_leg(closes5s: np.ndarray, ts5s: np.ndarray,
                  bars1m_ts: np.ndarray, zone_per_1m: np.ndarray,
                  entry_ts: int, entry_price: float,
                  leg_dir: str, end_ts: int, r_price: float):
    """Simulate baseline + composite trail exits for one leg.

    leg_dir: 'LONG' means we ENTER LONG (long-leg starting at LOW pivot).
             'SHORT' means we ENTER SHORT.

    Returns dict with exit info for both policies.
    """
    # Filter 5s bars to leg window [entry_ts, end_ts]
    mask = (ts5s >= entry_ts) & (ts5s <= end_ts)
    closes_leg = closes5s[mask]
    ts_leg = ts5s[mask]
    if len(closes_leg) == 0:
        return None

    # Running extreme during leg
    if leg_dir == 'LONG':
        run_ext = np.maximum.accumulate(closes_leg)
    else:
        run_ext = np.minimum.accumulate(closes_leg)

    # BASELINE: exit when price crosses extreme +/- R
    baseline_trail = run_ext - r_price if leg_dir == 'LONG' else run_ext + r_price
    if leg_dir == 'LONG':
        baseline_hit = np.where(closes_leg <= baseline_trail)[0]
    else:
        baseline_hit = np.where(closes_leg >= baseline_trail)[0]
    if len(baseline_hit) == 0:
        baseline_exit_idx = len(closes_leg) - 1   # never hit, exit at leg end
    else:
        baseline_exit_idx = int(baseline_hit[0])
    baseline_exit_price = float(closes_leg[baseline_exit_idx])
    baseline_exit_ts    = int(ts_leg[baseline_exit_idx])

    # COMPOSITE: per-bar trail distance adapts to zone at that time
    # For each 5s bar, look up the zone at the most recent 1m close
    idx_1m = np.searchsorted(bars1m_ts, ts_leg, side='right') - 1
    idx_1m = np.clip(idx_1m, 0, len(zone_per_1m) - 1)
    zone_at_bar = zone_per_1m[idx_1m]
    mult = np.array([ZONE_TRAIL_MULT.get(str(z), 1.0) for z in zone_at_bar])
    composite_trail_dist = r_price * mult
    composite_trail = run_ext - composite_trail_dist if leg_dir == 'LONG' else \
                       run_ext + composite_trail_dist
    if leg_dir == 'LONG':
        comp_hit = np.where(closes_leg <= composite_trail)[0]
    else:
        comp_hit = np.where(closes_leg >= composite_trail)[0]
    if len(comp_hit) == 0:
        comp_exit_idx = len(closes_leg) - 1
    else:
        comp_exit_idx = int(comp_hit[0])
    comp_exit_price = float(closes_leg[comp_exit_idx])
    comp_exit_ts    = int(ts_leg[comp_exit_idx])

    # Edge: how much MORE of the favorable move did composite capture?
    # For LONG: better exit = HIGHER price (closer to peak)
    # For SHORT: better exit = LOWER price (closer to trough)
    if leg_dir == 'LONG':
        edge_pts = comp_exit_price - baseline_exit_price
    else:
        edge_pts = baseline_exit_price - comp_exit_price

    # Track which zone caused the composite exit
    comp_exit_zone = str(zone_at_bar[comp_exit_idx])

    actual_peak = float(run_ext[-1])   # leg's final extreme

    return {
        'entry_ts': int(entry_ts),
        'entry_price': float(entry_price),
        'leg_dir': leg_dir,
        'r_price': float(r_price),
        'actual_peak_price': actual_peak,
        'leg_end_ts': int(end_ts),
        'baseline_exit_ts': baseline_exit_ts,
        'baseline_exit_price': baseline_exit_price,
        'baseline_exit_idx': baseline_exit_idx,
        'composite_exit_ts': comp_exit_ts,
        'composite_exit_price': comp_exit_price,
        'composite_exit_idx': comp_exit_idx,
        'composite_exit_zone': comp_exit_zone,
        'edge_pts': float(edge_pts),
        'edge_usd': float(edge_pts * DOLLAR_PER_POINT),
    }


def render_day_segments(day, bars1m, truth_day, cloud_day, leg_results,
                          out_dir, regime, segment_min=90):
    """Render per-day chart segments showing baseline vs composite exits."""
    bars1m = bars1m.copy()
    bars1m['ts_dt'] = pd.to_datetime(bars1m['timestamp'], unit='s')
    cloud_day = cloud_day.copy()
    cloud_day['ts_dt'] = pd.to_datetime(cloud_day['timestamp'], unit='s')

    ts_min = bars1m['timestamp'].min()
    ts_max = bars1m['timestamp'].max()
    seg_starts = list(range(int(ts_min), int(ts_max), segment_min * 60))

    zone_color = {
        'CLEAR': '#a0e0a0', 'WATCH': '#ffe0a0', 'WIDE_ZONE': '#ffc080',
        'NEAR_5m': '#ffa080', 'NEAR_3m': '#ff8080', 'IMMINENT': '#ff6060',
        'NEAR_PIVOT': '#ff8060', 'AT_PIVOT': '#ff4040',
    }

    for si, seg_lo in enumerate(seg_starts):
        seg_hi = seg_lo + segment_min * 60
        seg_bars = bars1m[(bars1m['timestamp'] >= seg_lo) &
                           (bars1m['timestamp'] < seg_hi)]
        if len(seg_bars) < 5:
            continue
        seg_cloud = cloud_day[(cloud_day['timestamp'] >= seg_lo) &
                                (cloud_day['timestamp'] < seg_hi)]
        seg_legs = [L for L in leg_results
                    if L['entry_ts'] < seg_hi and L['leg_end_ts'] > seg_lo]

        fig, (ax_top, ax) = plt.subplots(2, 1, figsize=(13, 6.5),
                                          gridspec_kw={'height_ratios': [0.6, 4]},
                                          sharex=True)
        plt.subplots_adjust(hspace=0.05, top=0.91)

        # Top: zone color bar
        for _, r in seg_cloud.iterrows():
            z = str(r['zone'])
            c = zone_color.get(z, '#cccccc')
            t = r['ts_dt']
            ax_top.bar(t, 1.0, width=pd.Timedelta(seconds=55), color=c,
                        edgecolor='none', align='edge')
        ax_top.set_ylim(0, 1)
        ax_top.set_yticks([])
        ax_top.set_ylabel('zone', fontsize=8)

        # Bottom: candles + leg simulations
        import matplotlib.dates as mdates
        w_days = 50 / 86400.0
        for _, b in seg_bars.iterrows():
            color = '#d0d0d0' if b['close'] >= b['open'] else '#606060'
            bottom = min(b['open'], b['close'])
            height = max(abs(b['close'] - b['open']), 0.01)
            x = mdates.date2num(b['ts_dt']) - w_days / 2
            rect = Rectangle((x, bottom), w_days, height,
                              facecolor=color, edgecolor='black',
                              linewidth=0.3, alpha=0.65, zorder=2)
            ax.add_patch(rect)
            ax.vlines([b['ts_dt']], [b['low']], [b['high']],
                      color='black', linewidth=0.4, alpha=0.6)

        # Mark entries (X) and exits (baseline=red dot, composite=green dot)
        # Plus connect entry->exits with thin lines so reader can pair them.
        for L in seg_legs:
            color = 'green' if L['leg_dir'] == 'LONG' else 'red'
            t_entry = pd.to_datetime(L['entry_ts'], unit='s')
            ax.scatter([t_entry], [L['entry_price']], marker='x', s=70,
                        color=color, linewidth=2.0, zorder=8)
            # Baseline exit
            t_b = pd.to_datetime(L['baseline_exit_ts'], unit='s')
            ax.scatter([t_b], [L['baseline_exit_price']], marker='o', s=55,
                        facecolor='gray', edgecolor='black',
                        linewidth=0.8, zorder=9)
            # Composite exit (color by edge sign)
            t_c = pd.to_datetime(L['composite_exit_ts'], unit='s')
            comp_color = '#00d050' if L['edge_pts'] > 0 else '#cc4040'
            ax.scatter([t_c], [L['composite_exit_price']], marker='D', s=55,
                        facecolor=comp_color, edgecolor='black',
                        linewidth=0.8, zorder=10)
            # Edge label near composite exit
            ax.annotate(f"${L['edge_usd']:+.0f}", xy=(t_c, L['composite_exit_price']),
                         xytext=(5, 8), textcoords='offset points', fontsize=7,
                         color='white',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor=comp_color,
                                   edgecolor='none', alpha=0.85), zorder=11)

        ylo = seg_bars['low'].min(); yhi = seg_bars['high'].max()
        ypad = (yhi - ylo) * 0.06
        ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.set_xlim(seg_bars['ts_dt'].iloc[0], seg_bars['ts_dt'].iloc[-1])
        ax.set_ylabel('price')
        ax.set_xlabel('time')
        ax.grid(True, alpha=0.3)

        # Stats
        edges_in_seg = [L['edge_usd'] for L in seg_legs]
        seg_total_edge = sum(edges_in_seg)
        n_pos = sum(1 for e in edges_in_seg if e > 0)
        n_neg = sum(1 for e in edges_in_seg if e < 0)
        title = (f'{day}  seg {si+1}/{len(seg_starts)} ({segment_min}min)  '
                 f'regime={regime}  '
                 f'legs={len(seg_legs)} (+{n_pos}/-{n_neg})  '
                 f'edge ${seg_total_edge:+.0f}')
        fig.suptitle(title, fontsize=10)

        # Legend
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker='x', color='green', linewidth=0, markersize=8,
                   markeredgewidth=2, label='LONG entry'),
            Line2D([0], [0], marker='x', color='red', linewidth=0, markersize=8,
                   markeredgewidth=2, label='SHORT entry'),
            Line2D([0], [0], marker='o', color='gray', linewidth=0, markersize=7,
                   label='baseline exit (R-trigger)'),
            Line2D([0], [0], marker='D', color='#00d050', linewidth=0, markersize=7,
                   label='composite exit (positive edge)'),
            Line2D([0], [0], marker='D', color='#cc4040', linewidth=0, markersize=7,
                   label='composite exit (negative edge)'),
        ]
        ax.legend(handles=handles, loc='upper left', fontsize=6, ncol=2)

        seg_path = out_dir / f'{day}_seg{si+1:02d}.png'
        fig.savefig(seg_path, dpi=105, bbox_inches='tight')
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--cloud',
                    default='reports/findings/regret_oracle/pivot_probability_cloud.parquet')
    ap.add_argument('--days', nargs='*', default=None)
    ap.add_argument('--n-days', type=int, default=6,
                    help='Number of days to render charts for (results computed for all)')
    ap.add_argument('--out-csv',
                    default='reports/findings/regret_oracle/composite_trail_sim.csv')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/composite_trail_sim.txt')
    ap.add_argument('--chart-dir',
                    default='reports/findings/regret_oracle/composite_trail_charts/')
    args = ap.parse_args()

    print('Loading truth + cloud...')
    truth = pd.read_parquet(args.truth)
    cloud = pd.read_parquet(args.cloud)
    print(f'  truth: {len(truth):,}   cloud: {len(cloud):,}')

    chart_dir = Path(args.chart_dir); chart_dir.mkdir(parents=True, exist_ok=True)

    all_legs = []
    days_all = sorted(truth['day'].unique())
    chart_days = args.days if args.days else None

    for day in tqdm(days_all, desc='days'):
        bars1m_path = NT8_1M_DIR / f'{day}.parquet'
        bars5s_path = NT8_5S_DIR / f'{day}.parquet'
        if not bars1m_path.exists() or not bars5s_path.exists():
            continue
        bars1m = pd.read_parquet(bars1m_path).sort_values('timestamp').reset_index(drop=True)
        bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
        atr_pts = compute_atr(bars1m, 14)
        min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * TRAIN_ATR_MULT)))
        r_price = min_rev_ticks * TICK_SIZE

        truth_day = truth[truth['day'] == day]
        cloud_day = cloud[cloud['day'] == day].sort_values('timestamp')
        bars1m_ts = bars1m['timestamp'].values.astype(np.int64)

        # Align cloud zone to 1m bar grid (cloud already at 1m close ts)
        zone_per_1m_lookup = dict(zip(cloud_day['timestamp'].values.astype(np.int64),
                                       cloud_day['zone'].values))
        zone_per_1m = np.array([zone_per_1m_lookup.get(int(ts), 'CLEAR')
                                  for ts in bars1m_ts], dtype=object)

        events = derive_pivot_events(truth_day)
        if len(events) < 2:
            continue

        closes5s = bars5s['close'].values.astype(np.float64)
        ts5s = bars5s['timestamp'].values.astype(np.int64)

        # Simulate each leg (entry at pivot[k], end at pivot[k+1])
        day_legs = []
        for k in range(len(events) - 1):
            entry_ts, leg_dir, entry_price = events[k]
            next_ts = events[k + 1][0]
            r = simulate_leg(closes5s, ts5s, bars1m_ts, zone_per_1m,
                              entry_ts, entry_price, leg_dir,
                              next_ts, r_price)
            if r is None:
                continue
            r['day'] = day
            r['atr_pts'] = atr_pts
            day_legs.append(r)
        all_legs.extend(day_legs)

        # Chart selected days
        if chart_days is None or day in chart_days:
            regime = load_regime_label(day) or 'unknown'
            day_out = chart_dir / day
            day_out.mkdir(parents=True, exist_ok=True)
            render_day_segments(day, bars1m, truth_day, cloud_day,
                                  day_legs, day_out, regime,
                                  segment_min=SEGMENT_MIN)

    df = pd.DataFrame(all_legs)
    df.to_csv(args.out_csv, index=False)

    # === Aggregate report ===
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('COMPOSITE TRAIL SIMULATOR — NT8 OOS')
    out('  Baseline: trail at R = 4xATR ticks (zigzag default)')
    out('  Composite: trail tightens by zone (CLEAR=1.0R ... AT_PIVOT=0.1R)')
    out('=' * 78)
    out(f'Legs simulated: {len(df):,}   Days: {df["day"].nunique()}')
    out('')

    edge = df['edge_usd'].dropna().values
    rng = np.random.default_rng(42)
    boots = np.array([edge[rng.integers(0, len(edge), len(edge))].mean()
                       for _ in range(4000)])
    mean = edge.mean(); ci_lo = np.percentile(boots, 2.5); ci_hi = np.percentile(boots, 97.5)
    out(f'Edge per leg: mean ${mean:+.2f}   95% CI [${ci_lo:+.2f}, ${ci_hi:+.2f}]')
    out(f'              median ${np.median(edge):+.2f}   '
        f'p25 ${np.percentile(edge, 25):+.2f}   p75 ${np.percentile(edge, 75):+.2f}')
    out(f'  positive (composite > baseline): {(edge > 0).mean()*100:.1f}%')
    out(f'  negative (composite < baseline): {(edge < 0).mean()*100:.1f}%')
    out(f'  zero (no difference):            {(edge == 0).mean()*100:.1f}%')
    out('')

    # Per-day aggregate
    pd_per_day = df.groupby('day').agg(
        n_legs=('edge_usd', 'count'),
        sum_edge=('edge_usd', 'sum'),
        mean_edge=('edge_usd', 'mean'),
    ).reset_index()
    rng = np.random.default_rng(42)
    sum_d = pd_per_day['sum_edge'].values
    boots = np.array([sum_d[rng.integers(0, len(sum_d), len(sum_d))].mean()
                       for _ in range(4000)])
    out('Per-day edge:')
    out(f'  mean      ${pd_per_day["sum_edge"].mean():+.2f}/day   '
        f'CI [${np.percentile(boots,2.5):+.2f}, ${np.percentile(boots,97.5):+.2f}]')
    out(f'  median    ${pd_per_day["sum_edge"].median():+.2f}/day')
    out(f'  positive days: {(pd_per_day["sum_edge"] > 0).sum()} / {len(pd_per_day)}')
    out('')

    # Per-regime aggregate (where regime label exists)
    pd_per_day['regime'] = pd_per_day['day'].apply(load_regime_label)
    pd_per_day['regime'] = pd_per_day['regime'].fillna('UNKNOWN')
    out('Per-regime edge (using DATA/ATLAS/regime_labels_2d.csv):')
    out(f'  {"regime":<14}  {"days":>5}  {"sum_edge":>11}  {"mean_edge_/day":>15}')
    for regime, sub in pd_per_day.groupby('regime'):
        out(f'  {regime:<14}  {len(sub):>5}  ${sub["sum_edge"].sum():>10,.2f}  '
            f'${sub["sum_edge"].mean():>14,.2f}')

    # Per-zone breakdown (which zone caused the composite exit?)
    out('')
    out('Composite exits BY ZONE (which zone closed the trade?):')
    out(f'  {"zone":<14}  {"n":>6}  {"% of all":>9}  {"mean_edge":>11}  {"median_edge":>12}')
    for zone, sub in df.groupby('composite_exit_zone'):
        out(f'  {zone:<14}  {len(sub):>6}  {len(sub)/len(df)*100:>8.1f}%  '
            f'${sub["edge_usd"].mean():>10,.2f}  '
            f'${sub["edge_usd"].median():>11,.2f}')

    # Save
    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    pd_per_day.to_csv(Path(args.out_csv).with_suffix('.per_day.csv'), index=False)
    print(f'\nWrote: {args.out_csv}')
    print(f'Wrote: {args.out_report}')
    print(f'Charts: {chart_dir}')


if __name__ == '__main__':
    main()
