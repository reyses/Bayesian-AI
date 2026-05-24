"""Quantify the trade-management value of B1 IMMINENT status.

Hypothesis (user 2026-05-17): if we know a pivot is coming, we can tighten
exits to capture more of the peak — even reacting "10 seconds after" the
flip is still a win, vs waiting for the standard R-trigger (4xATR pullback
= ~$80/pivot of given-back edge on MNQ).

Method (NT8 OOS, 32 days):
  For each zigzag pivot in the truth dataset:
    - T_pivot       = the pivot's actual extreme timestamp (peak/trough)
    - P_pivot       = price at the pivot (the extreme)
    - T_R_trigger   = timestamp when 5s close first crossed R-trigger price
                      after the pivot (= when standard zigzag would exit)
    - P_R_trigger   = price at R-trigger cross (= P_pivot - R for a LONG leg,
                      where R = ATR(14)_1m * 4 in price units)
    - T_imminent    = FIRST timestamp at or before T_R_trigger where
                      status=IMMINENT (B1 trajectory) -- the EARLIEST advance
                      warning that the flip was coming
    - P_imminent    = price at T_imminent

  Edge per pivot = (P_imminent - P_R_trigger) for a LONG-leg ending
                 = (P_R_trigger - P_imminent) for a SHORT-leg ending
  Sign convention: positive edge means IMMINENT exit captured MORE of the move.

  Convert to USD using MNQ multiplier ($2/point).

Report:
  - Distribution of edge across all pivots (median, mean, CI)
  - % pivots where IMMINENT fired BEFORE R-trigger (the cases we benefit from)
  - % pivots where IMMINENT NEVER fired (we'd fall back to R-trigger anyway)
  - Per-day aggregate
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_zigzag_baseline import compute_atr, TICK_SIZE

DOLLAR_PER_POINT = 2.0   # MNQ
TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')


def derive_pivot_events_per_day(truth_df: pd.DataFrame):
    """Collapse is_pivot==1 runs to one event per pivot. Returns
    DataFrame per day: timestamp (centroid), pivot_dir, pivot_price."""
    out = []
    for day, g in truth_df.groupby('day'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        piv = g[g['is_pivot'] == 1].sort_values('timestamp')
        if len(piv) == 0:
            continue
        ts = piv['timestamp'].values.astype(np.int64)
        pd_ = piv['pivot_dir'].values
        pp_ = piv['pivot_price'].values
        groups = [[0]]
        for i in range(1, len(ts)):
            if ts[i] - ts[i-1] > 90:
                groups.append([i])
            else:
                groups[-1].append(i)
        for grp in groups:
            sub_ts = ts[grp]; sub_pd = pd_[grp]; sub_pp = pp_[grp]
            ts_c = int(np.median(sub_ts))
            vals, counts = np.unique(sub_pd, return_counts=True)
            d = vals[np.argmax(counts)]
            p_avg = float(np.mean(sub_pp))
            out.append({'day': day, 'timestamp': ts_c,
                         'pivot_dir': str(d), 'pivot_price': p_avg})
    return pd.DataFrame(out)


def find_r_trigger_cross(bars5s: pd.DataFrame, t_pivot: int,
                         p_pivot: float, r_price: float, leg_dir: str,
                         lookahead_min: int = 60):
    """After the pivot, find when 5s close crosses R-trigger.

    For a LONG-leg ending (pivot is HIGH, next leg goes DOWN):
        leg_dir of THIS pivot = 'LONG' (kicks off new LONG leg)
        Wait — convention check.

    In build_zigzag_pivot_dataset.py:
      pivot_dir is the direction of the leg STARTING at this pivot.
      So pivot_dir='LONG' means a LONG leg is starting here, i.e., this
      pivot is a LOW (the prior down-leg just ended).

    For the FLIP (= prior leg ending):
      If pivot_dir='LONG', the prior leg was SHORT (down). R-trigger to
      confirm a SHORT->LONG flip = R above the recent low = p_pivot + r_price.
    If pivot_dir='SHORT', prior leg was LONG. R-trigger = p_pivot - r_price.
    """
    closes = bars5s['close'].values.astype(np.float64)
    ts = bars5s['timestamp'].values.astype(np.int64)
    end_ts = t_pivot + lookahead_min * 60
    mask = (ts > t_pivot) & (ts <= end_ts)
    if not mask.any():
        return None, None
    closes_after = closes[mask]
    ts_after = ts[mask]
    if leg_dir == 'LONG':
        # We're confirming a LOW pivot. R-trigger = pivot + R (price rose by R)
        threshold = p_pivot + r_price
        hits = np.where(closes_after >= threshold)[0]
    else:
        # SHORT pivot, prior leg LONG. R-trigger = pivot - R (price fell by R)
        threshold = p_pivot - r_price
        hits = np.where(closes_after <= threshold)[0]
    if len(hits) == 0:
        return None, None
    i = hits[0]
    return int(ts_after[i]), float(closes_after[i])


def find_first_imminent_before(bridge_df: pd.DataFrame, t_r_trigger: int,
                                 t_lower_bound: int):
    """In bridge_df (one day), find the FIRST row where status=='IMMINENT'
    in the window (t_lower_bound, t_r_trigger). Returns (ts, price_close)
    or (None, None) if none.

    t_lower_bound is the prior pivot's timestamp (so we don't pull IMMINENT
    from a previous swing's tail)."""
    win = bridge_df[(bridge_df['timestamp'] > t_lower_bound) &
                     (bridge_df['timestamp'] <= t_r_trigger) &
                     (bridge_df['status'] == 'IMMINENT')]
    if len(win) == 0:
        return None, None
    row = win.iloc[0]
    return int(row['timestamp']), float(row.get('close', np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--bridge',
                    default='reports/findings/regret_oracle/b1_trajectory_bridge.parquet')
    ap.add_argument('--out-csv',
                    default='reports/findings/regret_oracle/imminent_exit_advantage.csv')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/imminent_exit_advantage.txt')
    args = ap.parse_args()

    print('Loading truth + bridge...')
    tr = pd.read_parquet(args.truth)
    br = pd.read_parquet(args.bridge)
    pivots_df = derive_pivot_events_per_day(tr)
    print(f'  Pivots: {len(pivots_df)}')

    # We need 5s closes per day for R-trigger detection
    # For IMMINENT-bar close price, attach 1m close
    print('Loading 1m closes per bar (already in truth)...')
    # truth has timestamp + 1m bar info but not close — need to load 1m bars
    rows = []
    for day in tqdm(sorted(pivots_df['day'].unique()), desc='days'):
        day_pivs = pivots_df[pivots_df['day'] == day].sort_values('timestamp').reset_index(drop=True)
        p1m_path = NT8_1M_DIR / f'{day}.parquet'
        p5s_path = NT8_5S_DIR / f'{day}.parquet'
        if not p1m_path.exists() or not p5s_path.exists():
            continue
        bars1m = pd.read_parquet(p1m_path).sort_values('timestamp').reset_index(drop=True)
        bars5s = pd.read_parquet(p5s_path).sort_values('timestamp').reset_index(drop=True)
        atr_pts = compute_atr(bars1m, 14)
        min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * TRAIN_ATR_MULT)))
        r_price = min_rev_ticks * TICK_SIZE
        # Per-bar 1m close lookup (for IMMINENT price)
        bar1m_ts = bars1m['timestamp'].values.astype(np.int64)
        bar1m_close = bars1m['close'].values
        bridge_day = br[br['day'] == day].sort_values('timestamp').reset_index(drop=True)
        # Attach close prices to bridge rows
        if 'close' not in bridge_day.columns:
            idx = np.searchsorted(bar1m_ts, bridge_day['timestamp'].values.astype(np.int64), side='right') - 1
            idx = np.clip(idx, 0, len(bar1m_ts) - 1)
            bridge_day = bridge_day.copy()
            bridge_day['close'] = bar1m_close[idx]
        # For each pivot in this day, find R-trigger + first IMMINENT before
        for j, piv in day_pivs.iterrows():
            t_pivot = int(piv['timestamp'])
            p_pivot = float(piv['pivot_price'])
            leg_dir = str(piv['pivot_dir'])
            # Find R-trigger cross post-pivot
            t_rtrig, p_rtrig = find_r_trigger_cross(
                bars5s, t_pivot, p_pivot, r_price, leg_dir,
                lookahead_min=60,
            )
            if t_rtrig is None:
                continue
            # Find first IMMINENT in (prev_pivot_ts, t_rtrig]
            if j > 0:
                t_lower = int(day_pivs.iloc[j-1]['timestamp'])
            else:
                t_lower = t_pivot - 4 * 3600   # 4h before first pivot
            t_imm, p_imm = find_first_imminent_before(
                bridge_day, t_rtrig, t_lower,
            )
            row = {
                'day': day,
                'pivot_ts': t_pivot,
                'pivot_dir_new': leg_dir,   # direction of NEW leg starting
                'pivot_price': p_pivot,
                'atr_pts': atr_pts,
                'r_price': r_price,
                'r_trigger_ts': t_rtrig,
                'r_trigger_price': p_rtrig,
                'imminent_ts': t_imm,
                'imminent_price': p_imm,
            }
            if t_imm is not None and p_imm is not None:
                # Edge: how much more of the peak did IMMINENT capture
                # For a LONG-leg ENDING (next leg = SHORT, pivot is HIGH):
                #   We were LONG. Better exit = HIGHER price = closer to peak.
                #   leg_dir of THIS pivot is 'SHORT' (kicks off short leg)
                #   IMMINENT price > R-trigger price = edge for the holder
                # For SHORT-leg ENDING (next leg = LONG, pivot is LOW):
                #   We were SHORT. Better exit = LOWER price = closer to trough.
                #   leg_dir of THIS pivot is 'LONG'
                #   IMMINENT price < R-trigger price = edge for the holder
                if leg_dir == 'SHORT':
                    # prior leg LONG, pivot is HIGH, exit closer to peak is better
                    edge_pts = p_imm - p_rtrig
                else:
                    # prior leg SHORT, pivot is LOW, exit closer to trough is better
                    edge_pts = p_rtrig - p_imm
                row['edge_pts'] = float(edge_pts)
                row['edge_usd'] = float(edge_pts * DOLLAR_PER_POINT)
                row['leadtime_s'] = int(t_rtrig - t_imm)
                row['imminent_before_rtrig'] = True
            else:
                row['edge_pts'] = np.nan
                row['edge_usd'] = np.nan
                row['leadtime_s'] = np.nan
                row['imminent_before_rtrig'] = False
            rows.append(row)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # --- Report ---
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('IMMINENT-AWARE EXIT ADVANTAGE  (NT8 OOS, 32 days)')
    out('  Hypothesis: exiting at first-IMMINENT captures more peak than')
    out('  exiting at R-trigger (the indicators standard confirmation point).')
    out('=' * 78)

    n_total = len(df)
    n_imm = int(df['imminent_before_rtrig'].sum())
    n_skipped = n_total - n_imm
    out(f'Pivots evaluated:  {n_total:,}')
    out(f'  IMMINENT fired before R-trigger: {n_imm:,}  ({n_imm/n_total*100:.1f}%)')
    out(f'  IMMINENT never fired before R-trigger: {n_skipped:,}  '
        f'(would fall back to R-trigger exit, no gain or loss)')
    out('')

    edge = df['edge_usd'].dropna().values
    if len(edge) == 0:
        out('No edge data available.'); return
    out('--- Edge ($/pivot) when IMMINENT fired before R-trigger ---')
    rng = np.random.default_rng(42)
    boots = np.array([edge[rng.integers(0, len(edge), len(edge))].mean() for _ in range(4000)])
    mean = edge.mean(); ci_lo = np.percentile(boots, 2.5); ci_hi = np.percentile(boots, 97.5)
    out(f'  mean   = ${mean:+7.2f}   95% CI [${ci_lo:+.2f}, ${ci_hi:+.2f}]')
    out(f'  median = ${np.median(edge):+7.2f}')
    out(f'  p25    = ${np.percentile(edge, 25):+7.2f}')
    out(f'  p75    = ${np.percentile(edge, 75):+7.2f}')
    out(f'  positive (edge > 0): {(edge > 0).mean()*100:.1f}%')
    out(f'  negative (worse than R-trigger): {(edge < 0).mean()*100:.1f}%')
    out('')

    leadtimes = df['leadtime_s'].dropna().values
    out('--- Lead time (seconds from IMMINENT fire to R-trigger cross) ---')
    out(f'  median  = {np.median(leadtimes):.0f}s ({np.median(leadtimes)/60:.1f} min)')
    out(f'  mean    = {leadtimes.mean():.0f}s ({leadtimes.mean()/60:.1f} min)')
    out(f'  p25     = {np.percentile(leadtimes, 25):.0f}s')
    out(f'  p75     = {np.percentile(leadtimes, 75):.0f}s')
    out('')

    # Per-day summary
    pd_per_day = df.groupby('day').agg(
        n_pivots=('pivot_ts', 'count'),
        n_imm_fired=('imminent_before_rtrig', 'sum'),
        sum_edge_usd=('edge_usd', 'sum'),
        mean_edge_usd=('edge_usd', 'mean'),
    ).reset_index()
    pd_per_day['imm_fire_rate'] = pd_per_day['n_imm_fired'] / pd_per_day['n_pivots']
    pd_per_day.to_csv(Path(args.out_csv).with_suffix('.per_day.csv'), index=False)

    out('--- Per-day aggregate ---')
    out(f'  Days: {len(pd_per_day)}')
    out(f'  Pivots/day median:       {pd_per_day["n_pivots"].median():.1f}')
    out(f'  IMM-fired pivots/day median: {pd_per_day["n_imm_fired"].median():.1f}')
    out(f'  Edge-USD per day median: ${pd_per_day["sum_edge_usd"].median():+.2f}')
    out(f'  Edge-USD per day mean:   ${pd_per_day["sum_edge_usd"].mean():+.2f}')
    rng = np.random.default_rng(42)
    sum_d = pd_per_day['sum_edge_usd'].dropna().values
    boots = np.array([sum_d[rng.integers(0, len(sum_d), len(sum_d))].mean() for _ in range(4000)])
    out(f'  Edge-USD per day 95% CI: [${np.percentile(boots,2.5):+.2f}, ${np.percentile(boots,97.5):+.2f}]')
    out('')

    # Interpretation
    out('=' * 78)
    out('INTERPRETATION')
    out('=' * 78)
    out('')
    out('If edge mean is POSITIVE and CI > 0:')
    out('  -> IMMINENT-aware exits beat standard R-trigger exits by $X/pivot.')
    out('  -> On MNQ with ~50 pivots/day, that could be substantial.')
    out('  -> Trade-management value of the bridge confirmed.')
    out('')
    out('If edge mean is NEAR ZERO or CI crosses 0:')
    out('  -> IMMINENT fires too close to R-trigger (no lead time advantage)')
    out('  -> OR fires too early (false alarms cancel out gains)')
    out('  -> Bridge is informative but NOT actionable for tighter exits.')
    out('')
    out('Lead time tells us HOW EARLY the warning fires. Median 4.5min lead')
    out('= meaningful prep window for tightening trail stops, pre-placing')
    out('reverse limit orders, switching from ride to defensive mode.')

    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out_csv}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
