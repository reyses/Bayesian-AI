"""
cascade_pivot_quality.py -- 1m zigzag pivot quality conditioned on cascade alignment
====================================================================================

The 1m zigzag is the WORKING engine (v1.0 / v1.3). Day 1 NT8 result was +$455 with
17% capture rate (avg MFE $97, avg ETD $80). Hypothesis: pivots that fire when
multiple higher TFs are aligned in the trade direction give back less ETD because
the higher-TF structure is actually behind the move.

For every 1m zigzag pivot at threshold R, we:
  1. Find its CONFIRMATION timestamp (= when the engine actually places the order).
  2. Look up `<TF>_z_se` from FEATURES_5s/<day>.parquet at that timestamp.
  3. Count cascade alignment IN THE PIVOT'S INTENDED DIRECTION:
        - low  pivot -> wants LONG  -> count TFs with z_se < -threshold
        - high pivot -> wants SHORT -> count TFs with z_se > +threshold
  4. Walk forward 1m closes until the OPPOSITE pivot confirms (= the leg the
     v1.0/v1.3 strategy holds). Record:
        - leg_pts:      end_price - entry_price (signed by direction)
        - mfe_pts:      max favorable excursion within the leg
        - mae_pts:      max adverse excursion
        - etd_pts:      mfe_pts - leg_pts  (the ETD that Day 1 surfaced)
        - capture_pct:  100 * leg_pts / mfe_pts
        - leg_minutes:  duration
  5. Bucket pivots by alignment level. If WR / capture rises monotonically with
     alignment, the cascade-as-risk-gate thesis is confirmed.

Usage:
    python tools/cascade_pivot_quality.py                    # all days, R=30, threshold 2.0
    python tools/cascade_pivot_quality.py --r 30 --threshold 1.5
    python tools/cascade_pivot_quality.py --tfs 5m 15m 1h 1D  # macro-only context
    python tools/cascade_pivot_quality.py --day 2026-02-09    # single-day diagnostic
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.zigzag_backtest import zigzag_pivots_with_confirmation

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_TFS = ['15s', '1m', '5m', '15m', '1h', '1D']
DEFAULT_R = 30.0
DEFAULT_THRESHOLD = 2.0
DOLLAR_PER_POINT = 2.0   # MNQ

ATLAS_ROOT = 'DATA/ATLAS'
FEATURES_DIR = os.path.join(ATLAS_ROOT, 'FEATURES_5s')


# ── Loaders ───────────────────────────────────────────────────────────────

def load_1m_day(atlas_root: str, day_label: str) -> pd.DataFrame | None:
    p = os.path.join(atlas_root, '1m', f'{day_label}.parquet')
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)


def load_features_day(features_dir: str, day_label: str) -> pd.DataFrame | None:
    p = os.path.join(features_dir, f'{day_label}.parquet')
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p).sort_values('timestamp').reset_index(drop=True)


# ── Core ──────────────────────────────────────────────────────────────────

def alignment_at_timestamp(df_features: pd.DataFrame,
                           ts_query: int,
                           tfs: list[str],
                           threshold: float,
                           pivot_kind: str) -> tuple[int, float, float, list[float]]:
    """At time ts_query, compute multi-TF stretch metrics IN THE PIVOT'S DIRECTION.

    pivot_kind == 'low'  => intended trade is LONG  => "in direction" means z_se < 0
    pivot_kind == 'high' => intended trade is SHORT => "in direction" means z_se > 0

    Returns (alignment_count, dir_energy, sum_abs_z, raw_z_values):
      alignment_count: #TFs with |z_se| > threshold AND signed in pivot direction
      dir_energy:      sum of (|z_se| if signed in pivot direction else 0)
                       continuous "energy in the right direction" metric
      sum_abs_z:       sum of |z_se| across all TFs (raw stretch magnitude, undirected)
    """
    ts_arr = df_features['timestamp'].values.astype(np.int64)
    idx = int(np.searchsorted(ts_arr, ts_query, side='right')) - 1
    if idx < 0:
        return 0, 0.0, 0.0, [0.0] * len(tfs)

    z_vals: list[float] = []
    aligned = 0
    dir_energy = 0.0
    sum_abs = 0.0
    for tf in tfs:
        col = f'{tf}_z_se'
        if col not in df_features.columns:
            z_vals.append(0.0)
            continue
        z = float(df_features[col].iloc[idx])
        if not np.isfinite(z):
            z = 0.0
        z_vals.append(z)
        sum_abs += abs(z)

        if pivot_kind == 'low':
            # LONG trade. Stretch DOWN (z<0) is "in direction" (= energy that snaps back up).
            if z < 0:
                dir_energy += abs(z)
            if z < -threshold:
                aligned += 1
        else:  # 'high'
            if z > 0:
                dir_energy += abs(z)
            if z > threshold:
                aligned += 1
    return aligned, dir_energy, sum_abs, z_vals


def measure_leg(closes_1m: np.ndarray,
                ts_1m: np.ndarray,
                lows_1m: np.ndarray,
                highs_1m: np.ndarray,
                entry_idx: int,       # 1m bar where entry fills (= pivot confirm bar + 1)
                exit_idx: int,        # 1m bar where exit fills (= next pivot confirm bar + 1)
                direction: int) -> dict | None:
    """Compute leg-level outcome stats for the v1.0/v1.3 trade between two pivots.

    direction: +1 = LONG, -1 = SHORT.
    """
    if entry_idx >= len(closes_1m) or exit_idx >= len(closes_1m):
        return None
    if exit_idx <= entry_idx:
        return None

    entry_p = float(closes_1m[entry_idx])
    exit_p = float(closes_1m[exit_idx])

    # Excursions across the held bars
    held_high = float(highs_1m[entry_idx:exit_idx + 1].max())
    held_low  = float(lows_1m[entry_idx:exit_idx + 1].min())

    if direction > 0:    # LONG
        mfe = held_high - entry_p
        mae = entry_p - held_low
        leg = exit_p - entry_p
    else:                # SHORT
        mfe = entry_p - held_low
        mae = held_high - entry_p
        leg = entry_p - exit_p

    etd = mfe - leg
    capture_pct = (100.0 * leg / mfe) if mfe > 0 else 0.0
    leg_minutes = float(ts_1m[exit_idx] - ts_1m[entry_idx]) / 60.0

    return dict(
        entry_p=entry_p, exit_p=exit_p,
        mfe_pts=float(mfe), mae_pts=float(mae),
        leg_pts=float(leg), etd_pts=float(etd),
        capture_pct=float(capture_pct),
        leg_minutes=float(leg_minutes),
    )


# ── Per-day pipeline ──────────────────────────────────────────────────────

def process_day(day_label: str,
                atlas_root: str,
                features_dir: str,
                tfs: list[str],
                r: float,
                threshold: float) -> list[dict]:
    df_1m = load_1m_day(atlas_root, day_label)
    df_feat = load_features_day(features_dir, day_label)
    if df_1m is None or df_feat is None or len(df_1m) < 2:
        return []

    closes = df_1m['close'].values.astype(np.float64)
    ts_1m  = df_1m['timestamp'].values.astype(np.int64)
    lows_  = df_1m['low'].values.astype(np.float64)
    highs_ = df_1m['high'].values.astype(np.float64)

    pivots = zigzag_pivots_with_confirmation(closes, r)
    if len(pivots) < 2:
        return []

    out_rows: list[dict] = []
    for i in range(len(pivots) - 1):
        ext_idx, ext_price, kind, confirm_idx = pivots[i]
        _, _, _, next_confirm_idx = pivots[i + 1]

        # v1.0/v1.3 fills on the bar AFTER confirmation
        entry_idx = confirm_idx + 1
        exit_idx = next_confirm_idx + 1
        if exit_idx >= len(closes):
            continue

        # Cascade alignment lookup at confirmation moment (= bar close)
        confirm_ts = int(ts_1m[confirm_idx]) + 60
        align_count, dir_energy, sum_abs_z, z_vals = alignment_at_timestamp(
            df_feat, confirm_ts, tfs, threshold, kind)

        # Direction from kind: low pivot -> next leg LONG (+1); high pivot -> SHORT (-1)
        direction = +1 if kind == 'low' else -1

        leg = measure_leg(closes, ts_1m, lows_, highs_, entry_idx, exit_idx, direction)
        if leg is None:
            continue

        row = dict(
            day=day_label,
            confirm_ts=confirm_ts,
            kind=kind,
            direction=direction,
            ext_price=float(ext_price),
            alignment=int(align_count),
            dir_energy=float(dir_energy),
            sum_abs_z=float(sum_abs_z),
            **leg,
        )
        for tf, z in zip(tfs, z_vals):
            row[f'{tf}_z_se'] = float(z)
        out_rows.append(row)

    return out_rows


# ── Reporting ─────────────────────────────────────────────────────────────

def _is_2025(day: str) -> bool: return day.startswith('2025_')
def _is_2026(day: str) -> bool: return day.startswith('2026_')


def print_table(label: str, df: pd.DataFrame, n_tfs: int):
    print(f'\n=== {label} (N={len(df):,} pivots) — by THRESHOLD-COUNT alignment ===')
    print(f'{"Align":>5} {"N":>7} {"WR%":>6} '
          f'{"MFE$":>9} {"ETD$":>9} {"Capture%":>10} '
          f'{"Final$":>10} {"LegMin":>8} '
          f'{"%Long":>7} {"%Short":>7}')
    for k in range(0, n_tfs + 1):
        sub = df[df['alignment'] == k]
        if len(sub) == 0:
            continue
        wr = (sub['leg_pts'] > 0).mean() * 100.0
        mfe_usd = sub['mfe_pts'].mean() * DOLLAR_PER_POINT
        etd_usd = sub['etd_pts'].mean() * DOLLAR_PER_POINT
        cap = sub['capture_pct'].mean()
        final_usd = sub['leg_pts'].mean() * DOLLAR_PER_POINT
        leg_min = sub['leg_minutes'].mean()
        n_long = int((sub['direction'] == 1).sum())
        n_short = int((sub['direction'] == -1).sum())
        print(f'{k:>5} {len(sub):>7,} {wr:>5.1f}% '
              f'${mfe_usd:>+7.2f} ${etd_usd:>+7.2f} {cap:>9.1f}% '
              f'${final_usd:>+8.2f} {leg_min:>7.1f}m '
              f'{100.0*n_long/len(sub):>6.1f}% {100.0*n_short/len(sub):>6.1f}%')


def print_continuous_table(label: str, df: pd.DataFrame, col: str, n_quantiles: int = 5):
    """Quantile-bucket on a continuous score column (e.g. dir_energy)."""
    if len(df) == 0:
        return
    print(f'\n=== {label} — by {col} quantile (N={len(df):,}) ===')
    qs = pd.qcut(df[col], q=n_quantiles, labels=False, duplicates='drop')
    print(f'{"Bucket":>6} {col + " range":>20} {"N":>7} {"WR%":>6} '
          f'{"MFE$":>9} {"ETD$":>9} {"Final$":>10} {"%Long":>7}')
    for q in sorted(qs.dropna().unique()):
        sub = df[qs == q]
        if len(sub) == 0:
            continue
        lo = sub[col].min()
        hi = sub[col].max()
        wr = (sub['leg_pts'] > 0).mean() * 100.0
        mfe_usd = sub['mfe_pts'].mean() * DOLLAR_PER_POINT
        etd_usd = sub['etd_pts'].mean() * DOLLAR_PER_POINT
        final_usd = sub['leg_pts'].mean() * DOLLAR_PER_POINT
        n_long = int((sub['direction'] == 1).sum())
        rng = f'[{lo:.2f},{hi:.2f}]'
        print(f'{int(q):>6} {rng:>20} {len(sub):>7,} {wr:>5.1f}% '
              f'${mfe_usd:>+7.2f} ${etd_usd:>+7.2f} ${final_usd:>+8.2f} '
              f'{100.0*n_long/len(sub):>6.1f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', default=FEATURES_DIR)
    ap.add_argument('--atlas', default=ATLAS_ROOT)
    ap.add_argument('--tfs', nargs='+', default=DEFAULT_TFS,
                    help='TFs whose z_se columns to use (default: 15s 1m 5m 15m 1h 1D)')
    ap.add_argument('--r', type=float, default=DEFAULT_R, help='Zigzag R in points')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                    help='|z_se| threshold for "stretched" (default 2.0)')
    ap.add_argument('--day', default=None,
                    help='Single-day diagnostic (YYYY-MM-DD); else processes all')
    ap.add_argument('--out', default=None,
                    help='Output CSV path; default reports/findings/cascade_pivot_quality_R{r}_z{threshold}.csv')
    args = ap.parse_args()

    if args.day:
        days = [args.day.replace('-', '_')]
    else:
        files = sorted(glob.glob(os.path.join(args.features_dir, '*.parquet')))
        days = [os.path.splitext(os.path.basename(p))[0] for p in files]

    print('=' * 90)
    print(f'CASCADE-CONDITIONED 1m PIVOT QUALITY')
    print(f'Days: {len(days)}  |  R={args.r:g}  |  threshold |z|>{args.threshold:g}  |  TFs: {args.tfs}')
    print('=' * 90)

    all_rows: list[dict] = []
    for d in tqdm(days, desc='days') if len(days) > 1 else days:
        all_rows.extend(process_day(d, args.atlas, args.features_dir,
                                     args.tfs, args.r, args.threshold))

    if not all_rows:
        print('No pivots produced.')
        return

    df = pd.DataFrame(all_rows)
    n_tfs = len(args.tfs)

    # Reports
    is_df  = df[df['day'].apply(_is_2025)]
    oos_df = df[df['day'].apply(_is_2026)]
    print_table(f'IS  2025 — R={args.r:g}, |z|>{args.threshold:g}', is_df, n_tfs)
    print_table(f'OOS 2026 — R={args.r:g}, |z|>{args.threshold:g}', oos_df, n_tfs)

    # Continuous score: dir_energy = sum of |z_se| signed in pivot direction.
    # This bucket is INDEPENDENT of the threshold (uses all magnitudes, not just past-threshold).
    print_continuous_table(f'IS  2025 (continuous gradient)',  is_df,  'dir_energy', n_quantiles=10)
    print_continuous_table(f'OOS 2026 (continuous gradient)', oos_df, 'dir_energy', n_quantiles=10)

    # Save
    out_csv = args.out or f'reports/findings/cascade_pivot_quality_R{args.r:g}_z{args.threshold:g}.csv'
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote per-pivot CSV: {out_csv} ({len(df):,} rows)')


if __name__ == '__main__':
    main()
