"""Forward pass zigzag forward pass — drives the LIVE L5 engine bar-by-bar over
historical days, producing a leg list with NO lookahead.

This is the lean batch equivalent of `engine_v2._step7_trade()` minus the
NT8/async/bridge layer (zero-slip immediate fills, the mock_bridge
convention). It does NOT re-implement the engine: it imports and drives the
real `L5Decider` + `core.ledger.Ledger`, with `pivot_source='stream'` (the
forward pass streaming zigzag detector).

Contrast with the OFFLINE hardened legs (build_is_hardened_legs.py): those
take their pivot sequence from offline `is_pivot` labels — hindsight-clean,
zero whipsaw. This pass detects pivots causally, so it includes the whipsaw
trades a live engine actually takes.

Output (hardened-legs schema — consumed unchanged by
tools/trade_outcome_suite/excursions.py):
    reports/findings/trade_outcome_table/causal_zigzag_legs_{IS,OOS}.csv
    reports/findings/trade_outcome_table/causal_zigzag_forward_pass.txt

Usage:
    python training_zigzag/forward_zigzag.py --oos          # OOS (2026)
    python training_zigzag/forward_zigzag.py --is           # IS  (2025)
    python training_zigzag/forward_zigzag.py --with-oos     # both
    python training_zigzag/forward_zigzag.py --oos --limit 3   # smoke test

See docs/JULES_TRAINING_ZIGZAG.md for the design.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ledger import Ledger                                  # noqa: E402
from live.l5_decider import L5Decider, L5Context, ATR_MULTIPLIER  # noqa: E402
from training.live_feature_engine_v2 import LiveFeatureEngineV2  # noqa: E402
from v2_ticker import V2Ticker                                  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / 'reports' / 'findings' / 'trade_outcome_table'
DOLLAR_PER_POINT = 2.0
FRICTION_USD = 6.0                       # $4 commission + $2 slip (hardened-legs convention)
_ZEROS = np.zeros(91, dtype=np.float32)  # L5 ignores V1 features; ledger only needs len>12

# IS / OOS are defined by DATA SOURCE, not calendar year:
#   IS  = Databento (DATA/ATLAS)       — the historical training corpus
#   OOS = NT8 dump  (DATA/ATLAS_NT8)   — fresh data the system has never seen
# A calendar day may exist in both with different OHLCV (Databento vs NT8) —
# they are distinct populations.
ATLAS_ROOTS = {
    'is':  str(REPO / 'DATA' / 'ATLAS'),
    'oos': str(REPO / 'DATA' / 'ATLAS_NT8'),
}


def _resolve_days(target: str) -> tuple[str, list[str]]:
    """Returns (atlas_root, sorted_per_day_5s_parquet_paths) for the target.
    No calendar-year filter — IS uses ALL days in DATA/ATLAS, OOS uses ALL
    days in DATA/ATLAS_NT8."""
    root = ATLAS_ROOTS[target]
    files = sorted(glob.glob(os.path.join(root, '5s', '*.parquet')))
    return root, files


def run_day(day: str, day_bars: pd.DataFrame, prior_1m: pd.DataFrame,
            lfe, ctx, flat: bool = False) -> list[dict]:
    """Drive the L5 engine over one day's 5s bars via V2Ticker.

    Mirrors `training/forward_blended.py`'s FeatureTicker / on_state pattern:
    the ticker yields per-bar state dicts (V2 via the lfe's get_v2_vector — the
    precomputed FEATURES_5s_v2 L0/L1_* layers, same path the live engine
    uses), the consumer injects positions per bar and feeds to engine.evaluate.

    `flat=True` bypasses B7 (always enter on R-trigger) and B9 (never cut) —
    the "without GBM" mode that matches build_is_hardened_legs's filter-less
    pivot-to-pivot legs.
    """
    engine = L5Decider(ctx)
    if prior_1m is not None and len(prior_1m) > 0:
        engine.prime_atr_from_history(prior_1m)

    if flat:
        from core.engine_signals import EntrySignal
        # Bypass B7: always enter on R-trigger fire
        engine._b7_query = lambda ts, v2: EntrySignal(
            tier='L5_ZIGZAG_RTRIG',
            direction=engine._this_bar_rtrig_dir,
            cnn_flipped=False,
        ) if engine._this_bar_rtrig_dir else None
        # Bypass B9: never cut
        engine._b9_query_full = lambda pos, ts, current_close, v2: None

    ledger = Ledger()
    ticker = V2Ticker(day_bars, lfe)

    for state in ticker:
        ts, close = state['timestamp'], state['price']

        # 1. advance per-bar ledger state (bars_held / peak_pnl)
        ledger.update_bar(_ZEROS, close, ts, current_volume=state['volume'])

        # 2. inject fresh positions snapshot, then stateless engine eval
        state['positions'] = ledger.snapshot()
        batch = engine.evaluate(state)

        # 3. persist per-position counter updates
        for pdcn in batch.position_decisions:
            ledger.apply_position_decision(pdcn)

        # 4. exits — zero-slip fill at bar close (process before entries so a
        #    same-bar flip leaves the ledger flat for the new entry)
        for ex in batch.exits:
            if ledger.get(ex.contract_id) is not None:
                ledger.remove_position(ex.contract_id, close, ts, ex.reason)

        # 5. negative exit — close the primary
        if batch.negative_exit is not None:
            cid = batch.negative_exit.contract_id
            if ledger.get(cid) is not None:
                ledger.remove_position(cid, close, ts, batch.negative_exit.reason)

        # 6. fresh entry — only if flat after exits (L5 Phase-1: no chains)
        if batch.entry is not None and ledger.is_flat:
            ledger.add_position(batch.entry.direction, close, ts,
                                batch.entry.tier, _ZEROS)

    # day end — force-close anything still open at the last bar
    if not ledger.is_flat:
        last = ticker.last_bar
        for pos in [ledger.primary] + ledger.chains:
            if pos is not None:
                ledger.remove_position(pos.contract_id, float(last['close']),
                                       float(last['timestamp']), 'eod_close')

    r_price = float(getattr(engine, '_r_price', 0.0) or 0.0)
    atr_pts = r_price / ATR_MULTIPLIER if r_price > 0 else 0.0
    rows = []
    for t in ledger.closed_trades:
        pnl_pts = t['pnl'] / DOLLAR_PER_POINT
        rows.append({
            'day': day, 'entry_ts': int(t['entry_ts']),
            'leg_dir': str(t['dir']).upper(), 'entry_price': t['entry_price'],
            'exit_ts': int(t['exit_ts']), 'exit_price': t['exit_price'],
            'pnl_pts': pnl_pts, 'pnl_usd': t['pnl'] - FRICTION_USD,
            'r_price': r_price, 'atr_pts': atr_pts,
            'exit_reason': t['exit_reason'],
        })
    return rows


# Offline hardened-leg CSVs — used as the R-trigger source in replay mode
# (the positive-control test: same harness, offline pivots instead of streaming).
HARDENED_LEGS = {
    'is': 'reports/findings/regret_oracle/is_hardened_legs.csv',
    'oos': 'reports/findings/regret_oracle/oos_hardened_legs_full.csv',
}


def run_forward(target: str, limit: int = 0,
                pivot_source: str = 'stream', flat: bool = False) -> pd.DataFrame:
    """L5 forward pass over the IS (DATA/ATLAS = Databento) or
    OOS (DATA/ATLAS_NT8 = NT8 dump) corpus. Source is determined by target
    via ATLAS_ROOTS — no calendar-year filter, no atlas_root override.

    pivot_source='stream'  -> forward pass streaming zigzag (the real forward pass pass).
    pivot_source='replay'  -> offline hardened-leg R-triggers fed in
                              (positive-control: should reproduce the offline
                              forward-pass number).
    """
    atlas_root, day_files = _resolve_days(target)
    if limit:
        day_files = day_files[:limit]
    if not day_files:
        print(f'  no 5s day files for target={target!r} under {atlas_root}/5s')
        return pd.DataFrame()

    # L5Context is read-only — build it ONCE and share across days.
    replay_csv = (str(REPO / HARDENED_LEGS[target])
                  if pivot_source == 'replay' else None)
    ctx = L5Context.load(pivot_source=pivot_source,
                         replay_pivot_parquet=replay_csv)

    print(f'  warming feature engine ({atlas_root}) ...')
    lfe = LiveFeatureEngineV2(atlas_root, v2_only=True)
    lfe.load_history()                       # get_v2_vector is forward pass-by-anchor
    bars_1m = lfe._bars.get('1m')

    all_rows = []
    for fpath in tqdm(day_files, desc=f'{target.upper()} {pivot_source}', unit='day'):
        day = os.path.basename(fpath).replace('.parquet', '')
        day_bars = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
        if len(day_bars) == 0:
            continue
        day_start = float(day_bars['timestamp'].iloc[0])
        prior_1m = (bars_1m[bars_1m['timestamp'] < day_start]
                    if bars_1m is not None else None)
        all_rows.extend(run_day(day, day_bars, prior_1m, lfe, ctx, flat=flat))
    return pd.DataFrame(all_rows)


def _summary(df: pd.DataFrame, target: str, lines: list):
    def o(s=''):
        print(s); lines.append(s)
    o('=' * 70)
    o(f'CAUSAL ZIGZAG FORWARD PASS — {target.upper()}')
    o('=' * 70)
    if df.empty:
        o('(no legs produced)')
        return
    n_days = df['day'].nunique()
    pnl = df['pnl_usd'].values
    per_day = df.groupby('day')['pnl_usd'].sum()
    o(f'Legs: {len(df):,}   Days: {n_days}   '
      f'Leg-dir: {(df["leg_dir"]=="LONG").sum()}L / {(df["leg_dir"]=="SHORT").sum()}S')
    o(f'$/day:        ${pnl.sum()/max(n_days,1):+.2f}')
    o(f'Total $:      ${pnl.sum():+,.2f}')
    o(f'P(close>0):   {(pnl>0).mean()*100:.1f}%   mean ${pnl.mean():+.1f}/leg')
    o(f'Day WR:       {(per_day>0).sum()}/{n_days}   '
      f'worst ${per_day.min():+.0f}  best ${per_day.max():+.0f}')
    o(f'eod_close legs: {(df["exit_reason"]=="eod_close").sum()} '
      f'(open at session end, force-closed)')
    o('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is', dest='do_is', action='store_true',
                    help='IS = DATA/ATLAS (Databento corpus)')
    ap.add_argument('--oos', dest='do_oos', action='store_true',
                    help='OOS = DATA/ATLAS_NT8 (fresh NT8 dump)')
    ap.add_argument('--with-oos', action='store_true', help='IS + OOS')
    ap.add_argument('--limit', type=int, default=0,
                    help='cap days per window (smoke test)')
    ap.add_argument('--pivot-source', choices=['stream', 'replay'],
                    default='stream',
                    help="'stream' = forward pass zigzag (real forward pass pass); "
                         "'replay' = offline hardened-leg R-triggers "
                         "(positive-control parity check)")
    ap.add_argument('--flat', action='store_true',
                    help='Bypass B7 and B9 to run the pure structural zigzag strategy')
    args = ap.parse_args()

    targets = []
    if args.do_is or args.with_oos:
        targets.append('is')
    if args.do_oos or args.with_oos or (not args.do_is and not args.with_oos):
        targets.append('oos')

    tag = 'causal_flat' if args.flat else ('forward pass' if args.pivot_source == 'stream' else 'replay')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for tgt in targets:
        print(f'\n=== {tag} zigzag forward pass: {tgt.upper()} '
              f'(pivot_source={args.pivot_source}) ===')
        df = run_forward(tgt, limit=args.limit,
                         pivot_source=args.pivot_source, flat=args.flat)
        if not df.empty:
            out_csv = OUT_DIR / f'{tag}_zigzag_legs_{tgt.upper()}.csv'
            df.drop(columns=['exit_reason']).to_csv(out_csv, index=False)
            print(f'  wrote {out_csv}  ({len(df):,} legs)')
        _summary(df, tgt, lines)

    report = OUT_DIR / f'{tag}_zigzag_forward_pass.txt'
    report.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote {report}')


if __name__ == '__main__':
    main()
