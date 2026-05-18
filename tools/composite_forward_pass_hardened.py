"""HARDENED OOS Forward Pass — no pivot peek.

Earlier `composite_forward_pass.py` claimed +$1,613/day OOS with B7 sizing.
But B7 was trained on features at the PIVOT bar (the actual extreme),
which is unavailable in live trading. This forward pass fixes that:

  - Detect R-trigger fire from streaming 5s closes (live-equivalent detection)
  - At R-trigger fire moment, look up V2 features at THAT 1m close
    (not at the pivot bar, which would be a peek)
  - Apply existing B7 model to honest features -> sizing decision
  - Exit at NEXT R-trigger fire
  - Realistic P&L = (exit_5s_close - entry_5s_close) * leg_dir
                    - 2*tick*tick_value (1-tick slippage each side)
                    - commission per round-trip

Comparison schemes (all with R-trigger entries):
  - flat (no sizing)
  - hand_aggressive (composite zone + B6 at R-trigger bar)
  - gbm_ev (B7 trained on pivot features, queried at R-trigger bar — the
    sizing decision uses ONLY features available at decision time)

If gbm_ev still beats flat by >$500/day with CI strictly positive,
the architecture survives the hardening. If it collapses to flat,
the previous claim was largely from the pivot-feature peek.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_zigzag_baseline import compute_atr, TICK_SIZE


DOLLAR_PER_POINT = 2.0
COMMISSION_PER_LEG = 4.00
SLIPPAGE_PER_LEG   = 2.00
FRICTION_PER_LEG   = COMMISSION_PER_LEG + SLIPPAGE_PER_LEG

TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')
REGIME_CSV = Path('DATA/ATLAS/regime_labels_2d.csv')


def derive_pivot_events(truth_day):
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


def detect_r_trigger_fire(closes5s, ts5s, pivot_ts, pivot_price, leg_dir,
                            r_price, max_lookahead_min=120):
    """Walk forward from pivot to find first 5s close that crosses R-trigger
    threshold. Returns (fire_ts, fire_price) or (None, None) if never.

    For LONG leg (new leg starts up from low pivot):
      Wait for 5s close >= pivot_price + r_price (price reversed R from low)
    For SHORT leg (new leg starts down from high pivot):
      Wait for 5s close <= pivot_price - r_price
    """
    end_ts = pivot_ts + max_lookahead_min * 60
    mask = (ts5s > pivot_ts) & (ts5s <= end_ts)
    sub_closes = closes5s[mask]
    sub_ts = ts5s[mask]
    if len(sub_closes) == 0:
        return None, None
    if leg_dir == 'LONG':
        hits = np.where(sub_closes >= pivot_price + r_price)[0]
    else:
        hits = np.where(sub_closes <= pivot_price - r_price)[0]
    if len(hits) == 0:
        return None, None
    return int(sub_ts[hits[0]]), float(sub_closes[hits[0]])


def gbm_ev(pred_R):
    return float(np.clip(max(pred_R - 1.0, 0.0), 0.0, 3.0))


def gbm_quantile(pred_R, percentile_rank):
    if percentile_rank >= 0.80: return 2.0
    if percentile_rank >= 0.50: return 1.5
    if percentile_rank >= 0.20: return 0.8
    return 0.5


def hand_aggressive(zone, b6_match):
    if zone == 'AT_PIVOT' or b6_match >= 0.70: return 2.0
    elif zone in ('IMMINENT', 'NEAR_PIVOT', 'NEAR_3m', 'NEAR_5m'): return 1.2
    elif zone == 'CLEAR' and b6_match < 0.50: return 0.0
    else: return 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--b7-pkl',
                    default='reports/findings/regret_oracle/b7_leg_sizer.pkl')
    ap.add_argument('--cloud',
                    default='reports/findings/regret_oracle/pivot_probability_cloud.parquet')
    ap.add_argument('--b6',
                    default='reports/findings/regret_oracle/b6_proba_OOS_NT8.parquet')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened.csv')
    ap.add_argument('--report',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened.txt')
    args = ap.parse_args()

    print('Loading inputs...')
    truth = pd.read_parquet(args.truth)
    with open(args.b7_pkl, 'rb') as f:
        b7 = pickle.load(f)
    b7_model = b7['model']
    v2_cols = b7['v2_cols']
    cloud = pd.read_parquet(args.cloud)
    b6 = pd.read_parquet(args.b6)
    print(f'  truth {len(truth):,}  cloud {len(cloud):,}  b6 {len(b6):,}')

    # Build per-day mappings for fast lookup
    rows = []
    skipped_no_rtrig = 0
    skipped_no_next = 0
    skipped_no_features = 0

    for day in tqdm(sorted(truth['day'].unique()), desc='days'):
        bars5s_path = NT8_5S_DIR / f'{day}.parquet'
        bars1m_path = NT8_1M_DIR / f'{day}.parquet'
        if not bars5s_path.exists() or not bars1m_path.exists():
            continue
        bars1m = pd.read_parquet(bars1m_path).sort_values('timestamp').reset_index(drop=True)
        bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
        atr_pts = compute_atr(bars1m, 14)
        min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * TRAIN_ATR_MULT)))
        r_price = min_rev_ticks * TICK_SIZE

        truth_day = truth[truth['day'] == day].sort_values('timestamp').reset_index(drop=True)
        truth_ts = truth_day['timestamp'].values.astype(np.int64)
        cloud_day = cloud[cloud['day'] == day].sort_values('timestamp').reset_index(drop=True)
        b6_day = b6[b6['day'] == day].sort_values('timestamp').reset_index(drop=True)
        cloud_ts = cloud_day['timestamp'].values.astype(np.int64)
        b6_ts = b6_day['timestamp'].values.astype(np.int64)

        events = derive_pivot_events(truth_day)
        if len(events) < 2:
            continue
        closes5s = bars5s['close'].values.astype(np.float64)
        ts5s = bars5s['timestamp'].values.astype(np.int64)

        # For each pivot, detect R-trigger fire = ENTRY for THE NEW leg
        rtrig_fires = []
        for (piv_ts, piv_dir, piv_price) in events:
            fire_ts, fire_price = detect_r_trigger_fire(
                closes5s, ts5s, piv_ts, piv_price, piv_dir, r_price,
                max_lookahead_min=120,
            )
            if fire_ts is None:
                continue
            rtrig_fires.append({
                'pivot_ts': piv_ts, 'pivot_dir': piv_dir, 'pivot_price': piv_price,
                'fire_ts': fire_ts, 'fire_price': fire_price,
            })

        # Build leg pairs: each leg from rtrig[k].fire to rtrig[k+1].fire
        for k in range(len(rtrig_fires) - 1):
            entry = rtrig_fires[k]
            exit_  = rtrig_fires[k + 1]
            leg_dir = entry['pivot_dir']    # direction of THIS new leg
            entry_ts = entry['fire_ts']
            entry_price = entry['fire_price']
            exit_ts = exit_['fire_ts']
            exit_price = exit_['fire_price']

            # Lookup V2 features at the 1m close AT or just before entry_ts
            i_feat = int(np.searchsorted(truth_ts, entry_ts, side='right') - 1)
            if i_feat < 0 or i_feat >= len(truth_day):
                skipped_no_features += 1
                continue
            feat_row = truth_day.iloc[i_feat]
            X = np.array([float(feat_row[c]) if not pd.isna(feat_row[c]) else 0.0
                            for c in v2_cols], dtype=np.float32).reshape(1, -1)
            pred_amp_R = float(b7_model.predict(X)[0])

            # Composite zone at entry bar (for hand_aggressive)
            ci = int(np.searchsorted(cloud_ts, entry_ts, side='right') - 1)
            ci = min(max(ci, 0), len(cloud_day) - 1)
            entry_zone = str(cloud_day['zone'].iloc[ci])

            # B6 directional match at entry bar
            bi = int(np.searchsorted(b6_ts, entry_ts, side='right') - 1)
            bi = min(max(bi, 0), len(b6_day) - 1)
            p_long  = float(b6_day['p_PIVOT_TO_LONG_10m'].iloc[bi])
            p_short = float(b6_day['p_PIVOT_TO_SHORT_10m'].iloc[bi])
            entry_p_b6_match = p_long if leg_dir == 'LONG' else p_short

            # Realistic per-leg P&L: (exit - entry) signed - friction
            if leg_dir == 'LONG':
                pnl_pts = exit_price - entry_price
            else:
                pnl_pts = entry_price - exit_price
            pnl_usd = pnl_pts * DOLLAR_PER_POINT - FRICTION_PER_LEG

            rows.append({
                'day': day,
                'entry_ts': entry_ts,
                'leg_dir': leg_dir,
                'entry_price': entry_price,
                'exit_ts': exit_ts,
                'exit_price': exit_price,
                'pnl_pts': pnl_pts,
                'pnl_usd': pnl_usd,
                'r_price': r_price,
                'atr_pts': atr_pts,
                'pred_amp_R_hardened': pred_amp_R,
                'entry_zone': entry_zone,
                'entry_p_b6_match': entry_p_b6_match,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f'\nWrote: {args.out}   ({len(df):,} legs)')

    n_legs = len(df)
    pnl = df['pnl_usd'].values
    pos = int((pnl > 0).sum())

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('HARDENED OOS FORWARD PASS — features at R-trigger bar (no pivot peek)')
    out('=' * 78)
    out(f'Days: {df["day"].nunique()}   Legs: {n_legs:,}')
    out(f'Friction per leg: ${FRICTION_PER_LEG:.2f}')
    out('')
    out(f'  Mean per-leg P&L (NET):   ${pnl.mean():+.2f}')
    out(f'  Median per-leg P&L:       ${np.median(pnl):+.2f}')
    out(f'  Positive legs:            {pos}/{n_legs} ({pos/n_legs*100:.1f}%)')
    out(f'  B7 hardened pred amp_R:   median {df["pred_amp_R_hardened"].median():.2f}   '
        f'mean {df["pred_amp_R_hardened"].mean():.2f}')

    # Sizing schemes
    pred = df['pred_amp_R_hardened'].values
    pred_rank = pd.Series(pred).rank(pct=True).values
    schemes = {
        'flat':            np.ones(n_legs),
        'gbm_ev':          np.array([gbm_ev(p) for p in pred]),
        'gbm_quantile':    np.array([gbm_quantile(p, r) for p, r in zip(pred, pred_rank)]),
        'hand_aggressive': np.array([hand_aggressive(z, b)
                                       for z, b in zip(df['entry_zone'].values,
                                                        df['entry_p_b6_match'].values)]),
    }

    out('')
    out(f'{"scheme":<18}  {"n_taken":>7}  {"n_skip":>7}  '
        f'{"total_$":>10}  {"per_leg":>9}  {"per_unit":>9}  '
        f'{"mean_size":>9}  {"$/day":>9}')

    rng = np.random.default_rng(42)
    per_day_results = {}
    for name, sizes in schemes.items():
        weighted = pnl * sizes
        total_pnl = float(weighted.sum())
        total_size = float(sizes.sum())
        n_taken = int((sizes > 0).sum())
        n_skipped = int((sizes == 0).sum())
        per_day_v = total_pnl / df['day'].nunique()
        per_leg = total_pnl / max(n_legs, 1)
        per_unit = total_pnl / max(total_size, 1e-9)
        out(f'{name:<18}  {n_taken:>7}  {n_skipped:>7}  '
            f'${total_pnl:>9,.0f}  ${per_leg:>+7.2f}  '
            f'${per_unit:>+7.2f}  {sizes.mean():>9.3f}  '
            f'${per_day_v:>+7.0f}')

        df_copy = df.copy()
        df_copy['wpnl'] = weighted
        per_day_results[name] = df_copy.groupby('day')['wpnl'].sum().values

    out('')
    out('--- Per-day P&L (bootstrap CI 4000 resamples) ---')
    for name, per_day_v in per_day_results.items():
        boots = np.array([per_day_v[rng.integers(0, len(per_day_v), len(per_day_v))].mean()
                           for _ in range(4000)])
        out(f'  {name:<18}  mean ${per_day_v.mean():+.2f}/day  '
            f'95% CI [${np.percentile(boots, 2.5):+.2f}, ${np.percentile(boots, 97.5):+.2f}]  '
            f'pos {(per_day_v > 0).sum()}/{len(per_day_v)}  '
            f'median ${np.median(per_day_v):+.2f}')

    out('')
    out('--- Paired delta vs flat ---')
    base = per_day_results['flat']
    for name, per_day_v in per_day_results.items():
        if name == 'flat': continue
        delta = per_day_v - base
        boots = np.array([delta[rng.integers(0, len(delta), len(delta))].mean()
                           for _ in range(4000)])
        out(f'  {name:<18}  delta ${delta.mean():+.2f}/day  '
            f'CI [${np.percentile(boots, 2.5):+.2f}, ${np.percentile(boots, 97.5):+.2f}]  '
            f'wins {(per_day_v > base).sum()}/{len(per_day_v)}')

    # Compare to non-hardened previous result
    out('')
    out('=' * 78)
    out('COMPARISON: peeky vs hardened forward pass')
    out('=' * 78)
    out('  Peeky (B7 features at pivot bar):')
    out('    flat:            $899/day  CI [$599, $1,233]')
    out('    gbm_ev:        $1,613/day  CI [$1,107, $2,191]   delta vs flat +$713')
    out('    hand_aggressive: $1,312/day')
    out('')
    out('  Hardened (B7 features at R-trigger bar):')
    flat_pd = per_day_results['flat']
    ev_pd = per_day_results['gbm_ev']
    hand_pd = per_day_results['hand_aggressive']
    out(f'    flat:            ${flat_pd.mean():+.0f}/day')
    out(f'    gbm_ev:          ${ev_pd.mean():+.0f}/day   delta vs flat {ev_pd.mean()-flat_pd.mean():+.0f}')
    out(f'    hand_aggressive: ${hand_pd.mean():+.0f}/day')

    Path(args.report).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.report}')


if __name__ == '__main__':
    main()
