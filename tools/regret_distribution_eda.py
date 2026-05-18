"""Phase 2a — Distribution EDA on raw oracle entries.

Per user 2026-05-14: "the metric is $/trade." For each conditioning cell
(tod-bucket, regime, direction, duration-bucket, liq-bucket, and key joints),
compute the CLAUDE.md mandated stats:
    - n_entries
    - mode_$  (histogram bin $2)
    - median_$
    - mean_$ with 95% bootstrap CI
    - noise_floor_$  (random-walk benchmark from bar_range)
    - pass_mode : mode_$ > noise_floor_$    (real trades, not measurement artifact)
    - pass_ci   : 95% CI excludes 0          (significant edge)
    - tradeable : both pass

Goal: identify which cells contain real-trade edge and which are noise zones
to skip. If per-cell stratification cleanly separates real from noise, the
selector just gates on cells — fusion may be unnecessary. That is an
empirical question this tool answers.

LOOKAHEAD GUARD — only entry-time-knowable axes are used for stratification:
    OK   : tod_bucket, direction, liq_bucket, d_stack, d_z_15m_bin,
           d_rail_position, d_fan_bin, d_slope_15m_sign
    OUT  : duration_bucket (derived from time_to_mfe — future),
           regime_2d       (derived from end-of-day daily stats — lookahead)
regime_2d is still joined on session_date for diagnostic visibility, but is
NOT used as a stratification axis until a real-time regime detector exists.

Noise floor derivation
----------------------
Random walk with per-bar std σ has expected one-sided MFE over N bars of
σ·sqrt(2N/π).  For a Brownian-like bar, expected range ≈ σ·sqrt(8/π) ≈ 1.6σ,
so σ ≈ bar_range / 1.6.  Plugging in:
    E[MFE_$] = (bar_range / 1.6) · sqrt(2 N_fwd / π) · $/point
            ≈ bar_range · sqrt(N_fwd) · ($/point / 2)
For MNQ: $/point = 2 → E[MFE_$] ≈ bar_range_points · sqrt(N_fwd_bars).
That clean form is what we use.

Usage
-----
    python tools/regret_distribution_eda.py \\
        --input  reports/findings/regret_oracle/oracle_entries_IS_full.csv \\
        --tf-min 0.0833    # 5s = 1/12 min; use 1.0 for legacy 1m oracle CSVs
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR_DEFAULT    = Path('reports/findings/regret_oracle')
REGIME_LABELS_PATH = Path('DATA/ATLAS/regime_labels_2d.csv')
DOLLAR_PER_POINT   = 2.0   # MNQ: $0.50/tick × 4 ticks/point
BIN_W              = 2.0   # CLAUDE.md histogram bin width for $/trade
N_BOOT             = 4000
MIN_N_PER_CELL     = 10    # below this, stats are too noisy to report

# Session-phase buckets (minutes since session open at the post-halt minute).
# A Globex session is ~23h (1380 min). 4-hour blocks cover most of it.
TOD_BUCKETS = [
    ('phase_1_post_halt',  0,    240),    # First 4h of session (Asian open)
    ('phase_2_overnight',  240,  480),    # Europe overnight
    ('phase_3_pre_rth',    480,  720),    # Pre-NY-open
    ('phase_4_rth_am',     720,  960),    # NY morning
    ('phase_5_rth_pm',     960,  1200),   # NY afternoon
    ('phase_6_pre_halt',   1200, 9999),   # Final hours before halt
]
DURATION_BUCKETS = [
    ('FLASH',  0,    3),
    ('FAST',   3,    10),
    ('MEDIUM', 10,   25),
    ('SLOW',   25,   9999),
]


# ── stats helpers (CLAUDE.md spec) ──────────────────────────────────────────

def histogram_mode(vals: np.ndarray, bin_w: float = BIN_W) -> float:
    if len(vals) == 0:
        return float('nan')
    lo = np.floor(vals.min() / bin_w) * bin_w
    hi = np.ceil(vals.max() / bin_w) * bin_w
    if hi <= lo:
        hi = lo + bin_w
    bins = np.arange(lo, hi + bin_w, bin_w)
    counts, edges = np.histogram(vals, bins=bins)
    k = int(np.argmax(counts))
    return float((edges[k] + edges[k + 1]) / 2)


def bootstrap_mean_ci(vals: np.ndarray, n_boot: int = N_BOOT):
    if len(vals) == 0:
        return float('nan'), float('nan'), float('nan')
    if len(vals) < 2:
        m = float(vals[0])
        return m, m, m
    rng = np.random.default_rng(42)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = rng.choice(vals, size=len(vals), replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def bucket_assign(value, buckets):
    if pd.isna(value):
        return 'NA'
    for name, lo, hi in buckets:
        if lo <= value < hi:
            return name
    return 'NA'


# ── derived columns ────────────────────────────────────────────────────────

def join_regime_labels(df: pd.DataFrame, regime_path: Path) -> pd.DataFrame:
    if 'session_date' not in df.columns or not regime_path.exists():
        df = df.copy()
        df['regime_2d'] = 'NA'
        return df
    reg = pd.read_csv(regime_path)
    if 'regime_2d' not in reg.columns or 'date' not in reg.columns:
        df = df.copy()
        df['regime_2d'] = 'NA'
        return df
    df = df.merge(reg[['date', 'regime_2d']], how='left',
                  left_on='session_date', right_on='date')
    df.drop(columns=['date'], inplace=True, errors='ignore')
    df['regime_2d'] = df['regime_2d'].fillna('NA')
    return df


def add_derived_columns(df: pd.DataFrame, tf_min: float) -> pd.DataFrame:
    df = df.copy()
    # Time-of-day session-phase bucket
    if 'tod_minutes' in df.columns:
        df['tod_bucket'] = df['tod_minutes'].apply(
            lambda v: bucket_assign(v, TOD_BUCKETS))
    # Duration bucket
    if 'time_to_mfe_min' in df.columns:
        df['duration_bucket'] = df['time_to_mfe_min'].apply(
            lambda v: bucket_assign(v, DURATION_BUCKETS))
    # Liquidity quartile from volume (fallback bar_range, then skip)
    liq_col = None
    for c in ('volume', 'bar_range'):
        if c in df.columns and df[c].notna().any():
            liq_col = c
            break
    if liq_col is not None:
        try:
            q = pd.qcut(df[liq_col].astype(float).rank(method='first'),
                        4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'])
            df['liq_bucket'] = q.astype(str)
        except Exception:
            df['liq_bucket'] = 'NA'
    # Noise floor (random-walk benchmark on bar_range)
    if 'bar_range' in df.columns and 'available_fwd_min' in df.columns:
        n_fwd_bars = df['available_fwd_min'].astype(float) / max(tf_min, 1e-9)
        df['noise_floor_$'] = (df['bar_range'].astype(float)
                               * np.sqrt(np.maximum(n_fwd_bars, 0)))
    else:
        df['noise_floor_$'] = np.nan
    return df


# ── per-cell stats ─────────────────────────────────────────────────────────

def per_cell_stats(df: pd.DataFrame, axes: list, label: str) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(axes, dropna=False):
        cell_id = ' x '.join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        vals = sub['mfe_dollars'].astype(float).values
        if len(vals) < MIN_N_PER_CELL:
            continue
        mode_d  = histogram_mode(vals)
        mean_d, ci_lo, ci_hi = bootstrap_mean_ci(vals)
        median_d = float(np.median(vals))
        if 'noise_floor_$' in sub.columns and sub['noise_floor_$'].notna().any():
            noise = float(sub['noise_floor_$'].median())
            pass_mode = bool(mode_d > noise)
        else:
            noise, pass_mode = float('nan'), None
        pass_ci = bool(ci_lo > 0)
        rows.append({
            'axes': label, 'cell': cell_id, 'n': len(vals),
            'mode_$':   round(mode_d, 2),
            'median_$': round(median_d, 2),
            'mean_$':   round(mean_d, 2),
            'ci_lo_$':  round(ci_lo, 2),
            'ci_hi_$':  round(ci_hi, 2),
            'noise_floor_$': round(noise, 2) if not np.isnan(noise) else None,
            'pass_mode': pass_mode,
            'pass_ci':   pass_ci,
            'tradeable': bool(pass_mode and pass_ci) if pass_mode is not None else None,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values('mean_$', ascending=False).reset_index(drop=True)
    return out


# ── orchestrator ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='oracle_entries_*.csv')
    ap.add_argument('--out-dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--regime-labels', default=str(REGIME_LABELS_PATH))
    ap.add_argument('--tf-min', type=float, default=5 / 60,
                    help='Base TF in minutes (5s = 0.0833 — the default; '
                         '1m oracle CSVs need --tf-min 1.0)')
    ap.add_argument('--name', default='IS_full',
                    help='Tag for output filenames')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f'Loaded {len(df)} entries from {args.input}')

    df = join_regime_labels(df, Path(args.regime_labels))
    df = add_derived_columns(df, tf_min=args.tf_min)

    if 'full_window' in df.columns:
        n_before = len(df)
        df = df[df['full_window'] == 1].reset_index(drop=True)
        print(f'  full_window=1 filter: {len(df)} kept '
              f'(dropped {n_before - len(df)} truncated)')

    print(f'\n  regime_2d join rate : {(df["regime_2d"] != "NA").mean() * 100:.0f}%')
    if 'noise_floor_$' in df.columns and df['noise_floor_$'].notna().any():
        print(f'  noise_floor median  : ${df["noise_floor_$"].median():.2f}  '
              f'(p25 ${df["noise_floor_$"].quantile(.25):.2f} / '
              f'p75 ${df["noise_floor_$"].quantile(.75):.2f})')
    print(f'  $/trade overall     : mode=${histogram_mode(df["mfe_dollars"].values):.0f}  '
          f'median=${df["mfe_dollars"].median():.0f}  '
          f'mean=${df["mfe_dollars"].mean():.0f}')

    # ── ENTRY-TIME-CLEAN AXES (no lookahead) ──
    # duration_bucket is derived from time_to_mfe (future) — EXCLUDED.
    # regime_2d uses end-of-day stats (lookahead) — EXCLUDED until a real-time
    # detector exists; the join is kept above for diagnostics only.
    print('\nNOTE: duration_bucket and regime_2d are excluded as stratification')
    print('      axes (both contain lookahead). Selector-usable axes only.')

    single_axes = [c for c in (
        'tod_bucket', 'direction', 'liq_bucket',
        'd_stack', 'd_z_15m_bin', 'd_rail_position',
        'd_fan_bin', 'd_slope_15m_sign',
    ) if c in df.columns]
    all_stats = []
    for ax in single_axes:
        st = per_cell_stats(df, [ax], label=ax)
        if st.empty:
            continue
        all_stats.append(st)
        st.to_csv(out_dir / f'cell_stats_{args.name}_{ax}.csv', index=False)
        print(f'\n=== {ax} ===')
        cols = ['cell', 'n', 'mode_$', 'median_$', 'mean_$',
                'ci_lo_$', 'ci_hi_$', 'noise_floor_$', 'tradeable']
        print(st[cols].to_string(index=False))

    # ── Key joint distributions (entry-time-clean axes only) ──
    key_pairs = [
        ('tod_bucket',       'direction'),
        ('tod_bucket',       'liq_bucket'),
        ('direction',        'liq_bucket'),
        ('tod_bucket',       'd_z_15m_bin'),
        ('tod_bucket',       'd_rail_position'),
        ('direction',        'd_z_15m_bin'),
        ('direction',        'd_rail_position'),
        ('d_z_15m_bin',      'd_rail_position'),
        ('d_stack',          'd_z_15m_bin'),
    ]
    for a, b in key_pairs:
        if a in df.columns and b in df.columns:
            st = per_cell_stats(df, [a, b], label=f'{a}_x_{b}')
            if st.empty:
                continue
            all_stats.append(st)
            st.to_csv(out_dir / f'cell_stats_{args.name}_{a}_x_{b}.csv', index=False)

    # ── Combined + tradeable view ──
    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        combined.to_csv(out_dir / f'per_cell_per_trade_stats_{args.name}.csv', index=False)
        tradeable = combined[combined['tradeable'] == True].copy()
        tradeable = tradeable.sort_values('mean_$', ascending=False).reset_index(drop=True)
        tradeable.to_csv(out_dir / f'tradeable_cells_{args.name}.csv', index=False)

        print(f'\n{"=" * 70}')
        print(f'TRADEABLE CELLS (mode > noise floor AND 95%CI excludes 0)')
        print(f'  {len(tradeable)} of {len(combined)} cells '
              f'({100 * len(tradeable) / max(len(combined), 1):.0f}%)')
        if len(tradeable):
            cols = ['axes', 'cell', 'n', 'mode_$', 'mean_$',
                    'ci_lo_$', 'noise_floor_$']
            print(tradeable.head(20)[cols].to_string(index=False))

        # Also list NOISE-FLOOR-FAILED cells (mode at/below noise) — diagnostic
        noise_zones = combined[combined['pass_mode'] == False].copy()
        noise_zones = noise_zones.sort_values('n', ascending=False).reset_index(drop=True)
        if len(noise_zones):
            print(f'\nNOISE-FLOOR ZONES (mode <= noise floor) — top 10 by n:')
            cols = ['axes', 'cell', 'n', 'mode_$', 'noise_floor_$',
                    'mean_$', 'ci_lo_$']
            print(noise_zones.head(10)[cols].to_string(index=False))

    print(f'\nWrote outputs under {out_dir}/  (prefix: cell_stats_{args.name}_*, '
          f'per_cell_per_trade_stats_{args.name}.csv, tradeable_cells_{args.name}.csv)')


if __name__ == '__main__':
    main()
