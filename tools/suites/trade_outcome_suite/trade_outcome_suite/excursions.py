"""Shared data layer for the trade-outcome question suite.

Builds (or loads from cache) the per-leg excursion dataset: every hardened
zigzag leg with the full entry->exit MAE / MFE reconstructed from the 5s
intra-trade price path. entry-to-close is the CSV pnl (authoritative, net of
$6/leg friction). Every question module consumes this one dataset -- it is
built once and reused.

Cache: reports/findings/trade_outcome_table/per_leg_excursions_{IS,OOS}.parquet
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / 'reports/findings/trade_outcome_table'

# ── Leg-list source registry ─────────────────────────────────────────────
# Each source defines (legs_csv, 5s_bars_dir) per sample. Same schema
# (day, entry_ts, leg_dir, entry_price, exit_ts, exit_price, pnl_pts,
# pnl_usd, r_price, atr_pts), different leg populations.
#
#   'hardened'    — OFFLINE zigzag pivots (oracle-clean, zero whipsaw).
#                   Built by tools/build_is_hardened_legs.py from
#                   is_pivot==1 labels in the whole-day truth dataset.
#                   This is the LOOKAHEAD population — every leg is a
#                   genuine swing confirmed in hindsight. +$454/day OOS.
#   'causal_flat' — CAUSAL streaming zigzag pivots, NO model filters
#                   (no B7/B9/B10). Built by training_zigzag/forward_zigzag.py.
#                   Honest forward pass trade list — includes whipsaws. ~-$330/day.
SOURCES = {
    'hardened': {
        'IS':  (REPO / 'reports/findings/regret_oracle/is_hardened_legs.csv',
                REPO / 'DATA/ATLAS/5s'),
        'OOS': (REPO / 'reports/findings/regret_oracle/oos_hardened_legs_full.csv',
                REPO / 'DATA/ATLAS_NT8/5s'),
    },
    'causal_flat': {
        # IS  = Databento  (DATA/ATLAS)     — historical training corpus.
        # OOS = NT8 dump   (DATA/ATLAS_NT8) — fresh data the system never saw.
        # NO GBM filters — pure structural zigzag legs.
        'IS':  (REPO / 'reports/findings/trade_outcome_table/causal_flat_zigzag_legs_IS.csv',
                REPO / 'DATA/ATLAS/5s'),
        'OOS': (REPO / 'reports/findings/trade_outcome_table/causal_flat_zigzag_legs_OOS.csv',
                REPO / 'DATA/ATLAS_NT8/5s'),
    },
    'forward pass': {
        # Same IS/OOS sources, but WITH the L5 stack active (B7 skip + B9 cut +
        # B10 day mode). Lift over causal_flat: +$118/day IS (Databento), but
        # only -$10/day on OOS NT8 — filters trained on Databento don't
        # transfer to the cleaner NT8 tape.
        'IS':  (REPO / 'reports/findings/trade_outcome_table/causal_zigzag_legs_IS.csv',
                REPO / 'DATA/ATLAS/5s'),
        'OOS': (REPO / 'reports/findings/trade_outcome_table/causal_zigzag_legs_OOS.csv',
                REPO / 'DATA/ATLAS_NT8/5s'),
    },
    'strategy_run': {
        # Output of training/run_strategy.py — V2 ForwardPass + registered
        # Strategy class (zigzag today). Canonical paths from the runner's
        # default --out (ATR×4). Use this source from --analyze to consume
        # the fresh trades the runner just produced.
        'IS':  (REPO / 'reports/findings/strategy_runs/zigzag_is_atr4.csv',
                REPO / 'DATA/ATLAS/5s'),
        'OOS': (REPO / 'reports/findings/strategy_runs/zigzag_oos_atr4.csv',
                REPO / 'DATA/ATLAS_NT8/5s'),
    },
    'zigzag_lstm': {
        'IS':  (REPO / 'reports/findings/strategy_runs/zigzag_lstm_is_atr4.csv',
                REPO / 'DATA/ATLAS/5s'),
        'OOS': (REPO / 'reports/findings/strategy_runs/zigzag_lstm_oos_atr4.csv',
                REPO / 'DATA/ATLAS_NT8/5s'),
    },
    'entry_ml_filtered': {
        'IS':  (REPO / 'reports/findings/trade_outcome_table/entry_ml_filtered_IS.csv',
                REPO / 'DATA/ATLAS/5s'),
        'OOS': (REPO / 'reports/findings/trade_outcome_table/entry_ml_filtered_OOS.csv',
                REPO / 'DATA/ATLAS_NT8/5s'),
    },
}
DEFAULT_SOURCE = 'causal_flat'   # honest forward pass — supersedes lookahead-tainted hardened

TICK = 0.25                 # MNQ tick size (price units)
DOLLAR_PER_POINT = 2.0      # MNQ: $0.50/tick * 4 ticks/pt
FRICTION_USD = 6.0          # $4 commission + $2 slippage per leg (already in pnl_usd)
N_BOOT = 4000               # bootstrap resamples (project metric standard)
SEED = 42
BAR_SEC = 5.0               # 5s bars
MIN_CELL_N = 30             # cells thinner than this are flagged

# (legacy SRC dict removed -- replaced by SOURCES above)


# --- per-leg excursion build ----------------------------------------------
def build_excursions(legs_csv: Path, bars_dir: Path, label: str) -> pd.DataFrame:
    """One row per hardened leg, full entry->exit MAE/MFE from the 5s path."""
    legs = pd.read_csv(legs_csv)
    legs = legs.reset_index().rename(columns={'index': 'leg_id'})
    rows, skipped = [], 0
    for day, day_legs in tqdm(legs.groupby('day'), desc=f'{label} days',
                              total=legs['day'].nunique()):
        bp = bars_dir / f'{day}.parquet'
        if not bp.exists():
            skipped += len(day_legs)
            continue
        b = pd.read_parquet(bp).sort_values('timestamp').reset_index(drop=True)
        ts = b['timestamp'].values.astype(np.int64)
        hi = b['high'].values.astype(np.float64)
        lo = b['low'].values.astype(np.float64)
        for _, leg in day_legs.iterrows():
            entry_ts, exit_ts = int(leg['entry_ts']), int(leg['exit_ts'])
            ep = float(leg['entry_price'])
            d = str(leg['leg_dir'])
            ei = int(np.searchsorted(ts, entry_ts, side='left'))
            if ei >= len(ts) or ts[ei] != entry_ts:
                ei = int(np.searchsorted(ts, entry_ts, side='right') - 1)
            if ei < 0:
                skipped += 1
                continue
            xi = int(np.searchsorted(ts, exit_ts, side='right') - 1)
            xi = max(xi, ei)
            sh, sl = hi[ei:xi + 1], lo[ei:xi + 1]
            if len(sh) == 0:
                skipped += 1
                continue
            if d == 'LONG':
                fav, adv = sh - ep, ep - sl
            else:  # SHORT
                fav, adv = ep - sl, sh - ep
            mfe = max(0.0, float(fav.max()))
            mae = max(0.0, float(adv.max()))
            mfe_i = int(np.argmax(fav))
            mae_i = int(np.argmax(adv))
            pnl_pts = float(leg['pnl_pts'])
            mfe = max(mfe, pnl_pts)      # realised close is itself a path point
            mae = max(mae, -pnl_pts)
            r = float(leg['r_price'])
            rows.append({
                'leg_id': int(leg['leg_id']), 'day': day, 'leg_dir': d,
                'entry_ts': entry_ts, 'exit_ts': exit_ts,
                'entry_price': ep, 'exit_price': float(leg['exit_price']),
                'r_price': r, 'atr_pts': float(leg['atr_pts']),
                'pnl_pts': pnl_pts, 'pnl_usd': float(leg['pnl_usd']),
                'mae_pts': mae, 'mfe_pts': mfe,
                'mae_R': mae / r, 'mfe_R': mfe / r, 'close_R': pnl_pts / r,
                'duration_bars': xi - ei,
                'mae_bar': mae_i, 'mfe_bar': mfe_i,
            })
    df = pd.DataFrame(rows)
    if skipped:
        print(f'  {label}: {skipped} legs skipped (missing bars / bad index)')
    return df


def _augment(d: pd.DataFrame) -> pd.DataFrame:
    """Add the derived dollar / timing columns every question module uses."""
    d = d.copy()
    d['mfe_usd'] = d['mfe_pts'] * DOLLAR_PER_POINT
    d['mae_usd'] = d['mae_pts'] * DOLLAR_PER_POINT
    d['close_usd'] = d['pnl_usd']
    dur = d['duration_bars'].replace(0, np.nan)
    d['frac_to_bottom'] = (d['mae_bar'] / dur).fillna(0.0)
    d['frac_to_peak'] = (d['mfe_bar'] / dur).fillna(0.0)
    d['dur_min'] = d['duration_bars'] * BAR_SEC / 60.0
    d['bottom_to_exit_min'] = (d['duration_bars'] - d['mae_bar']) * BAR_SEC / 60.0
    d['t_to_bottom_min'] = d['mae_bar'] * BAR_SEC / 60.0
    return d


def load(sample: str, source: str = DEFAULT_SOURCE,
         rebuild: bool = False) -> pd.DataFrame:
    """Load the per-leg excursion dataset for 'IS' or 'OOS' under the given
    leg-list source ('causal_flat' default, or 'hardened' = the legacy
    lookahead population). Cache-first; tagged by source so populations
    coexist on disk."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {list(SOURCES)}, got {source!r}")
    if sample not in SOURCES[source]:
        raise ValueError(f"sample must be 'IS' or 'OOS', got {sample!r}")
    # Legacy hardened cache uses untagged filenames (backward compat).
    if source == 'hardened':
        cache = OUT_DIR / f'per_leg_excursions_{sample}.parquet'
    else:
        cache = OUT_DIR / f'per_leg_excursions_{source}_{sample}.parquet'
    if cache.exists() and not rebuild:
        d = pd.read_parquet(cache)
    else:
        legs_csv, bars_dir = SOURCES[source][sample]
        d = build_excursions(legs_csv, bars_dir, f'{source}/{sample}')
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        d.to_parquet(cache, index=False)
        print(f'{source}/{sample}: built {len(d):,} legs -> {cache.name}')
    return _augment(d)


# --- stats / formatting helpers -------------------------------------------
def bootstrap_ci(v, n=N_BOOT, seed=SEED):
    """Percentile 95% CI of the mean of v (binary array -> CI of a proportion)."""
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def pct(x):
    return 'n/a' if x != x else f'{x * 100:.0f}%'


def pct1(x):
    return 'n/a' if x != x else f'{x * 100:.1f}%'


def usd(x):
    return 'n/a' if x != x else f'${x:+,.0f}'


def md_table(headers, rows):
    out = ['| ' + ' | '.join(str(h) for h in headers) + ' |',
           '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)
