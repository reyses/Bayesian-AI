"""Blended signal forward pass on NT8 OOS — no trade management.

Composite of four signals we've built this session:
  1. trend3 raw           : per-1m-bar (p_long, p_short, p_neutral) — direction
  2. Live zigzag indicator : per-5s-bar streaming direction (lagged but causal)
  3. B1 pivot-imminent     : per-1m-bar P(pivot in next K min)
  4. B2 fakeout            : per-pivot P(this leg dies within K min)

Per user 2026-05-17: directional signal validation, NOT trading. No exits,
no sizing, no P&L. Just measure where the blended signal agrees with
hindsight `leg_direction` truth.

Composite rules (per 1m bar t):
  ride_long  =  trend3_dir == LONG  AND  live_zz_dir == LONG
                AND trend3_conf >= TRENDS_CONF_MIN
                AND p_pivot_10m  <  PIVOT_IMMINENT_MAX
                AND last_p_fakeout_10m < FAKEOUT_MAX  (or NaN if no recent pivot)

  ride_short =  same with SHORT

  fade_long  =  trend3_dir == SHORT  AND  p_pivot_10m  >  PIVOT_IMMINENT_MIN
                (anticipate flip from a confident short into a long)
  fade_short =  trend3_dir == LONG   AND  p_pivot_10m  >  PIVOT_IMMINENT_MIN

Output:
  - per-bar parquet with all signals + fire flags
  - text report with composite accuracy, fire counts, per-day stats
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_zigzag_baseline import live_zigzag_direction, compute_atr, TICK_SIZE


TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')

# Composite thresholds — chosen at "high-confidence ride" operating points
TRENDS_CONF_MIN     = 0.40   # trend3 p_dir - p_neutral
PIVOT_IMMINENT_MAX  = 0.30   # B1 threshold below = "safe to ride"
PIVOT_IMMINENT_MIN  = 0.70   # B1 above = "fade entry" zone
FAKEOUT_MAX         = 0.50   # B2 threshold below = "last pivot real"


def compute_live_zz_at_1m(bars1m: pd.DataFrame, bars5s: pd.DataFrame,
                           atr_mult: float = TRAIN_ATR_MULT,
                           min_bars: int = 36) -> np.ndarray:
    """Compute live zigzag direction at each 1m close timestamp.
    Returns int8 array per 1m bar: +1=LONG, -1=SHORT, 0=NEUTRAL."""
    atr_pts = compute_atr(bars1m, 14)
    min_rev = max(4, int(round(atr_pts / TICK_SIZE * atr_mult)))
    closes5s = bars5s['close'].values.astype(np.float64)
    ts5s = bars5s['timestamp'].values.astype(np.int64)
    live_dir = live_zigzag_direction(closes5s, min_rev, min_bars)
    ts1m = bars1m['timestamp'].values.astype(np.int64)
    idx5s = np.searchsorted(ts5s, ts1m, side='right') - 1
    idx5s = np.clip(idx5s, 0, len(ts5s) - 1)
    return live_dir[idx5s]


def attach_last_pivot_b2(bar_ts: np.ndarray, b2_ts: np.ndarray,
                          b2_p: np.ndarray, max_age_min: int = 30) -> np.ndarray:
    """For each 1m bar timestamp, return the P(fakeout) of the most recent
    pivot at or before that bar — but only if the pivot is within
    `max_age_min` minutes. Otherwise NaN.

    Rationale: a fakeout score from 2h ago has no bearing on the current
    leg's quality. Only the most recent confirmed pivot matters.
    """
    out = np.full(len(bar_ts), np.nan, dtype=np.float64)
    if len(b2_ts) == 0:
        return out
    max_age_s = max_age_min * 60
    idx = np.searchsorted(b2_ts, bar_ts, side='right') - 1
    for i, k in enumerate(idx):
        if 0 <= k < len(b2_ts):
            age = bar_ts[i] - b2_ts[k]
            if 0 <= age <= max_age_s:
                out[i] = b2_p[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trend3-cache',
                    default='reports/findings/regret_oracle/trend3_cache_OOS_NT8.parquet')
    ap.add_argument('--b1-cache',
                    default='reports/findings/regret_oracle/b1_proba_OOS_NT8.parquet')
    ap.add_argument('--b2-cache',
                    default='reports/findings/regret_oracle/b2_proba_OOS_NT8.parquet')
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--out-parquet',
                    default='reports/findings/regret_oracle/blended_signal_OOS_NT8.parquet')
    ap.add_argument('--out-report',
                    default='reports/findings/regret_oracle/blended_signal_OOS_NT8.txt')
    args = ap.parse_args()

    print('Loading caches...')
    t3  = pd.read_parquet(args.trend3_cache)
    b1  = pd.read_parquet(args.b1_cache)
    b2  = pd.read_parquet(args.b2_cache)
    tr  = pd.read_parquet(args.truth)
    print(f'  trend3: {len(t3)}   B1: {len(b1)}   B2: {len(b2)}   truth: {len(tr)}')

    # Truth labels
    if 'leg_direction' not in tr.columns:
        from tools.direction_signal_accuracy import derive_leg_direction
        tr = derive_leg_direction(tr)

    # Merge per-bar signals onto truth (1m closes)
    merged = tr[['timestamp', 'day', 'leg_direction']].merge(
        t3[['timestamp', 'day', 'p_long', 'p_short', 'p_neutral']],
        on=['timestamp', 'day'], how='inner',
    ).merge(
        b1[['timestamp', 'day', 'p_pivot_1m', 'p_pivot_3m',
            'p_pivot_5m', 'p_pivot_10m']],
        on=['timestamp', 'day'], how='inner',
    )

    # trend3 direction + confidence
    pl = merged['p_long'].values
    ps = merged['p_short'].values
    pn = merged['p_neutral'].values
    pred_dir = np.where(pl >= np.maximum(ps, pn), 'LONG',
                np.where(ps >= np.maximum(pl, pn), 'SHORT', 'NEUTRAL'))
    pred_conf = np.where(pred_dir == 'LONG',  pl - pn,
                 np.where(pred_dir == 'SHORT', ps - pn, 0.0))
    merged['trend3_dir'] = pred_dir
    merged['trend3_conf'] = pred_conf

    # Live zigzag per 1m bar — compute per day
    print('Computing live zigzag direction per day...')
    live_dir_arr = np.zeros(len(merged), dtype=np.int8)
    for day, g in tqdm(merged.groupby('day'), desc='live zz'):
        p1m = NT8_1M_DIR / f'{day}.parquet'
        p5s = NT8_5S_DIR / f'{day}.parquet'
        if not p1m.exists() or not p5s.exists():
            continue
        b1m = pd.read_parquet(p1m).sort_values('timestamp').reset_index(drop=True)
        b5s = pd.read_parquet(p5s).sort_values('timestamp').reset_index(drop=True)
        # Compute live ZZ direction at each 1m close in this day
        # We need to align to merged's timestamps — same 1m closes
        atr_pts = compute_atr(b1m, 14)
        min_rev = max(4, int(round(atr_pts / TICK_SIZE * TRAIN_ATR_MULT)))
        closes5s = b5s['close'].values.astype(np.float64)
        ts5s_arr = b5s['timestamp'].values.astype(np.int64)
        live_per_5s = live_zigzag_direction(closes5s, min_rev, 36)
        # For each row in merged for this day, look up the live direction
        # at that 1m close
        day_mask = (merged['day'] == day).values
        merge_ts = merged.loc[day_mask, 'timestamp'].values.astype(np.int64)
        idx5s = np.searchsorted(ts5s_arr, merge_ts, side='right') - 1
        idx5s = np.clip(idx5s, 0, len(ts5s_arr) - 1)
        live_dir_arr[day_mask] = live_per_5s[idx5s]
    merged['live_zz_dir'] = np.where(live_dir_arr == 1, 'LONG',
                              np.where(live_dir_arr == -1, 'SHORT', 'NEUTRAL'))

    # B2 fakeout — attach most recent pivot's score per 1m bar
    print('Attaching B2 fakeout to most-recent pivot per bar...')
    p_fakeout_attached = np.full(len(merged), np.nan)
    for day, g in merged.groupby('day'):
        b2d = b2[b2['day'] == day].sort_values('timestamp')
        if len(b2d) == 0:
            continue
        bar_ts = g['timestamp'].values.astype(np.int64)
        b2_ts  = b2d['timestamp'].values.astype(np.int64)
        for K in [10]:   # use K=10 for the blend
            col = f'p_fakeout_{K}m'
            p2 = b2d[col].values
            attached = attach_last_pivot_b2(bar_ts, b2_ts, p2, max_age_min=30)
            p_fakeout_attached[g.index.values] = attached
    merged['p_fakeout_recent_10m'] = p_fakeout_attached

    # --- Blended fire rules ---
    pivot_im_10 = merged['p_pivot_10m'].values
    p_fakeout   = merged['p_fakeout_recent_10m'].values
    conf        = merged['trend3_conf'].values
    t3_dir      = merged['trend3_dir'].values
    zz_dir      = merged['live_zz_dir'].values

    # Safe ride: trend3 and zz agree, conf high, pivot not imminent, last pivot real
    fakeout_ok = np.isnan(p_fakeout) | (p_fakeout < FAKEOUT_MAX)
    ride_base = (conf >= TRENDS_CONF_MIN) & (pivot_im_10 < PIVOT_IMMINENT_MAX) & fakeout_ok
    ride_long_dir   = (t3_dir == 'LONG')  & (zz_dir == 'LONG')
    ride_short_dir  = (t3_dir == 'SHORT') & (zz_dir == 'SHORT')
    ride_long  = ride_base & ride_long_dir
    ride_short = ride_base & ride_short_dir

    # Fade entry: trend3 strong, pivot imminent → enter opposite
    fade_base = (conf >= TRENDS_CONF_MIN) & (pivot_im_10 > PIVOT_IMMINENT_MIN)
    fade_long  = fade_base & (t3_dir == 'SHORT')   # short about to flip up
    fade_short = fade_base & (t3_dir == 'LONG')    # long about to flip down

    merged['blend_ride_long']   = ride_long
    merged['blend_ride_short']  = ride_short
    merged['blend_fade_long']   = fade_long
    merged['blend_fade_short']  = fade_short
    merged['blend_fire']        = (ride_long | ride_short |
                                    fade_long | fade_short)
    merged['blend_dir'] = np.where(ride_long | fade_long, 'LONG',
                          np.where(ride_short | fade_short, 'SHORT', 'NONE'))
    merged['blend_mode'] = np.where(ride_long | ride_short, 'RIDE',
                            np.where(fade_long | fade_short, 'FADE', 'NONE'))

    # --- Accuracy vs leg_direction (truth) ---
    truth = merged['leg_direction'].values
    in_leg = (truth == 'LONG') | (truth == 'SHORT')
    fired = merged['blend_fire'].values
    correct = (merged['blend_dir'].values == truth) & fired & in_leg
    wrong   = (merged['blend_dir'].values != truth) & fired & in_leg & (merged['blend_dir'].values != 'NONE')

    n_total = len(merged)
    n_in_leg = int(in_leg.sum())
    n_fire = int(fired.sum())
    n_fire_in_leg = int((fired & in_leg).sum())
    n_correct = int(correct.sum())
    n_wrong = int(wrong.sum())
    acc = n_correct / max(n_correct + n_wrong, 1)

    # Per-mode breakdown
    def mode_stats(mask, label):
        n = int((mask & in_leg).sum())
        c = int(((merged['blend_dir'].values == truth) & mask & in_leg).sum())
        w = n - c
        return label, n, c, w, c / max(n, 1)

    rides = mode_stats(ride_long | ride_short, 'RIDE')
    fades = mode_stats(fade_long | fade_short, 'FADE')

    # Per-day
    days = sorted(merged['day'].unique())
    per_day = []
    for day in days:
        g = merged[merged['day'] == day]
        in_l = g['leg_direction'].isin(['LONG', 'SHORT']).values
        f = g['blend_fire'].values
        c = (g['blend_dir'].values == g['leg_direction'].values) & f & in_l
        per_day.append({
            'day': day, 'n_bars': len(g),
            'n_fires': int(f.sum()),
            'n_fires_in_leg': int((f & in_l).sum()),
            'n_correct': int(c.sum()),
            'fire_rate': f.sum() / max(len(g), 1),
            'acc': c.sum() / max((f & in_l).sum(), 1),
        })
    per_day_df = pd.DataFrame(per_day)

    # Bootstrap CI on per-day acc
    accs = per_day_df['acc'].dropna().values
    rng = np.random.default_rng(42)
    boots = np.array([accs[rng.integers(0, len(accs), len(accs))].mean() for _ in range(4000)])
    acc_mean = float(accs.mean()); acc_lo = float(np.percentile(boots, 2.5)); acc_hi = float(np.percentile(boots, 97.5))

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('BLENDED SIGNAL FORWARD PASS -- NT8 OOS (no trade management)')
    out('  Composite: trend3 raw + live zigzag + B1 pivot-imminent + B2 fakeout')
    out('=' * 78)
    out(f'Days: {len(per_day_df)}   bars total: {n_total:,}   '
        f'bars with truth leg_dir: {n_in_leg:,}')
    out('')
    out(f'Composite thresholds:')
    out(f'  trend3 conf min      >= {TRENDS_CONF_MIN}')
    out(f'  B1 P(pivot<10m) max  <  {PIVOT_IMMINENT_MAX}     (ride mode)')
    out(f'  B1 P(pivot<10m) min  >  {PIVOT_IMMINENT_MIN}     (fade mode)')
    out(f'  B2 P(fakeout<10m) max <  {FAKEOUT_MAX}       (last pivot real)')
    out('')
    out(f'Total fires: {n_fire:,}   fires/day: {n_fire/max(len(per_day_df),1):.1f}')
    out(f'Of fires with truth label: {n_fire_in_leg:,}   '
        f'correct {n_correct:,}   wrong {n_wrong:,}   '
        f'acc {acc*100:.2f}%')
    out(f'Per-day acc mean: {acc_mean*100:.2f}%   '
        f'95% CI [{acc_lo*100:.2f}%, {acc_hi*100:.2f}%]')
    out('')
    out('--- Mode breakdown ---')
    for label, n, c, w, a in [rides, fades]:
        out(f'  {label}:  fires={n:,}   correct={c:,}   wrong={w:,}   '
            f'acc={a*100:.2f}%')
    out('')

    out('--- Vs individual signals (per-day acc, 32 days) ---')
    out(f'  Live ZZ alone  : 64.85% [63.22, 66.66]   coverage 97.6%')
    out(f'  trend3 raw     : 66.19% [63.41, 68.79]   coverage 66.0%')
    out(f'  BLENDED        : {acc_mean*100:.2f}% [{acc_lo*100:.2f}, {acc_hi*100:.2f}]   '
        f'coverage {n_fire/max(n_total,1)*100:.2f}%')

    # Save
    out_parq = Path(args.out_parquet); out_parq.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_parq, index=False)
    Path(args.out_report).write_text('\n'.join(lines), encoding='utf-8')
    per_day_df.to_csv(Path(args.out_report).with_suffix('.per_day.csv'), index=False)
    print(f'\nWrote: {args.out_parquet}')
    print(f'Wrote: {args.out_report}')


if __name__ == '__main__':
    main()
