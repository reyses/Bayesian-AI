"""DRS Path A - Step 3: hardened forward pass on IS days.

Copy of the deliverable's `composite_forward_pass_hardened.py` with paths
re-pointed to DATA/ATLAS (IS) instead of DATA/ATLAS_NT8 (OOS). Does NOT
modify the deliverable's script — runs the same algorithm against IS data.

Reads:
  - IS zigzag truth dataset (deliverables/.../caches/zigzag_pivot_dataset_IS_atr4.parquet)
  - B7 model pkl (deliverables/.../models/b7_leg_sizer.pkl)
  - IS pivot cloud (DATA/CROSS_DAY/predictions_IS/pivot_probability_cloud_IS.parquet)
  - IS B6 predictions (DATA/CROSS_DAY/predictions_IS/b6_proba_IS.parquet)
  - DATA/ATLAS/5s/*.parquet, DATA/ATLAS/1m/*.parquet for raw bars

Run:
  python tools/sourcing/drs_a_step3_forward_pass_is.py

Expected runtime: ~10-20 min (277 IS days vs 31 OOS in the original).

Output:
  DATA/CROSS_DAY/predictions_IS/composite_forward_pass_hardened_IS.csv
  DATA/CROSS_DAY/predictions_IS/composite_forward_pass_hardened_IS.txt
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Re-use helpers from the deliverable so logic stays identical
DELIVER = Path('deliverables/composite_zigzag_pipeline').resolve()
sys.path.insert(0, str(DELIVER / 'tools'))
from live_zigzag_baseline import compute_atr, TICK_SIZE
from composite_forward_pass_hardened import (
    derive_pivot_events, detect_r_trigger_fire,
    gbm_ev, gbm_quantile, hand_aggressive,
    DOLLAR_PER_POINT, FRICTION_PER_LEG, TRAIN_ATR_MULT,
)

# Re-point to IS paths
IS_5S_DIR = Path('DATA/ATLAS/5s')
IS_1M_DIR = Path('DATA/ATLAS/1m')

TRUTH    = DELIVER / 'caches' / 'zigzag_pivot_dataset_IS_atr4.parquet'
B7_PKL   = DELIVER / 'models' / 'b7_leg_sizer.pkl'
CLOUD    = Path('DATA/CROSS_DAY/predictions_IS/pivot_probability_cloud_IS.parquet')
B6_PROBA = Path('DATA/CROSS_DAY/predictions_IS/b6_proba_IS.parquet')

OUT_DIR = Path('DATA/CROSS_DAY/predictions_IS')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'composite_forward_pass_hardened_IS.csv'
OUT_TXT = OUT_DIR / 'composite_forward_pass_hardened_IS.txt'


def main():
    print('Loading inputs...')
    truth = pd.read_parquet(TRUTH)
    with open(B7_PKL, 'rb') as f:
        b7 = pickle.load(f)
    b7_model = b7['model']
    v2_cols = b7['v2_cols']
    cloud = pd.read_parquet(CLOUD)
    b6 = pd.read_parquet(B6_PROBA)
    print(f'  truth {len(truth):,}  cloud {len(cloud):,}  b6 {len(b6):,}')

    rows = []
    skipped_no_features = 0

    for day in tqdm(sorted(truth['day'].unique()), desc='IS days'):
        bars5s_path = IS_5S_DIR / f'{day}.parquet'
        bars1m_path = IS_1M_DIR / f'{day}.parquet'
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

        for k in range(len(rtrig_fires) - 1):
            entry = rtrig_fires[k]
            exit_ = rtrig_fires[k + 1]
            leg_dir = entry['pivot_dir']
            entry_ts = entry['fire_ts']
            entry_price = entry['fire_price']
            exit_ts = exit_['fire_ts']
            exit_price = exit_['fire_price']

            i_feat = int(np.searchsorted(truth_ts, entry_ts, side='right') - 1)
            if i_feat < 0 or i_feat >= len(truth_day):
                skipped_no_features += 1
                continue
            feat_row = truth_day.iloc[i_feat]
            X = np.array([float(feat_row[c]) if not pd.isna(feat_row[c]) else 0.0
                            for c in v2_cols], dtype=np.float32).reshape(1, -1)
            pred_amp_R = float(b7_model.predict(X)[0])

            ci = int(np.searchsorted(cloud_ts, entry_ts, side='right') - 1)
            ci = min(max(ci, 0), len(cloud_day) - 1)
            entry_zone = str(cloud_day['zone'].iloc[ci])

            bi = int(np.searchsorted(b6_ts, entry_ts, side='right') - 1)
            bi = min(max(bi, 0), len(b6_day) - 1)
            p_long  = float(b6_day['p_PIVOT_TO_LONG_10m'].iloc[bi])
            p_short = float(b6_day['p_PIVOT_TO_SHORT_10m'].iloc[bi])
            entry_p_b6_match = p_long if leg_dir == 'LONG' else p_short

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
    df.to_csv(OUT_CSV, index=False)
    print(f'\nWrote: {OUT_CSV}   ({len(df):,} legs)')

    # Sizing schemes (gbm_ev is the target for DRS)
    pnl = df['pnl_usd'].values
    pred = df['pred_amp_R_hardened'].values
    pred_rank = pd.Series(pred).rank(pct=True).values
    schemes = {
        'flat':            np.ones(len(df)),
        'gbm_ev':          np.array([gbm_ev(p) for p in pred]),
        'gbm_quantile':    np.array([gbm_quantile(p, r) for p, r in zip(pred, pred_rank)]),
        'hand_aggressive': np.array([hand_aggressive(z, b)
                                       for z, b in zip(df['entry_zone'].values,
                                                        df['entry_p_b6_match'].values)]),
    }

    lines = []
    def out(s=''):
        print(s); lines.append(s)
    out('=' * 78)
    out('HARDENED IS FORWARD PASS — features at R-trigger bar (no pivot peek)')
    out('=' * 78)
    out(f'Days: {df["day"].nunique()}   Legs: {len(df):,}')
    out(f'Friction per leg: ${FRICTION_PER_LEG:.2f}')
    out('')

    for name, sizes in schemes.items():
        weighted = pnl * sizes
        per_day = pd.DataFrame({'day': df['day'].values, 'w': weighted}) \
                    .groupby('day')['w'].sum()
        out(f'  {name:<18}  total ${weighted.sum():>10,.0f}  '
            f'$/day {per_day.mean():>+8.0f}  median ${per_day.median():>+7.0f}  '
            f'pos {(per_day > 0).sum()}/{len(per_day)}')

    OUT_TXT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote: {OUT_TXT}')


if __name__ == '__main__':
    main()
