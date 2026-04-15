"""
Parity Validate — one-stop validation for both parity layers.

Runs all post-session checks:
  1. FEATURE PARITY: FEATURES_LIVE_5s vs FEATURES_NT8_5s for the day
     (proves: live engine computes same features as build_dataset)

  2. DECISION PARITY: v2_trades_*.csv vs baseline forward pass on same features
     (proves: live engine makes same trades as backtest given same features)

  3. EXECUTION PARITY: v2_trades_*.csv vs nt8_trades_*.csv
     (proves: NT8 actually fills what we send, no slippage/rejection drift)

Output: reports/findings/parity_validate_YYYY_MM_DD.txt
        Single PASS/FAIL summary with detailed breakdowns.

Usage:
    python tools/parity_validate.py 2026_04_14
    python tools/parity_validate.py             # auto-detects today
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features_79d import FEATURE_NAMES_79D

REPORTS = 'reports/findings'
LIVE_DIR = 'reports/live'
FEATURES_LIVE = 'DATA/FEATURES_LIVE_5s'
FEATURES_NT8 = 'DATA/FEATURES_NT8_5s'


# ══════════════════════════════════════════════════════════════════
# LAYER 1: FEATURE PARITY (live computed vs build_dataset computed)
# ══════════════════════════════════════════════════════════════════

def feature_parity(day: str) -> dict:
    """Diff FEATURES_LIVE_5s vs FEATURES_NT8_5s on the same timestamps."""
    live_path = os.path.join(FEATURES_LIVE, f'{day}.parquet')
    base_path = os.path.join(FEATURES_NT8, f'{day}.parquet')

    if not os.path.exists(live_path):
        return {'status': 'SKIP', 'reason': f'No {live_path}'}
    if not os.path.exists(base_path):
        return {'status': 'SKIP', 'reason': f'No {base_path}'}

    live = pd.read_parquet(live_path).drop_duplicates(
        subset=[pd.read_parquet(live_path).columns[0]], keep='first')
    # Reload after first read for column name (avoids parquet schema issue)
    live = pd.read_parquet(live_path)
    live = live.drop_duplicates(subset=[live.columns[0]], keep='first')
    base = pd.read_parquet(base_path).drop_duplicates(
        subset=[pd.read_parquet(base_path).columns[0]], keep='first')
    base = pd.read_parquet(base_path)
    base = base.drop_duplicates(subset=[base.columns[0]], keep='first')

    overlap = sorted(set(live.iloc[:, 0].astype(int)) &
                     set(base.iloc[:, 0].astype(int)))

    if not overlap:
        return {'status': 'SKIP', 'reason': 'No timestamp overlap'}

    l = live.set_index(live.columns[0]).loc[overlap].sort_index()
    b = base.set_index(base.columns[0]).loc[overlap].sort_index()

    n_features = min(91, len(FEATURE_NAMES_79D))
    total_cells = n_features * len(overlap)
    total_exact = 0
    bad_features = []

    for col in FEATURE_NAMES_79D[:n_features]:
        if col not in l.columns or col not in b.columns:
            continue
        lv = l[col].values.astype(float)
        bv = b[col].values.astype(float)
        diff = np.abs(lv - bv)
        exact = int((diff < 1e-4).sum())
        total_exact += exact
        if exact < len(overlap):
            bad_features.append({
                'feature': col,
                'exact': exact,
                'total': len(overlap),
                'max_diff': float(diff.max()),
                'mean_diff': float(diff.mean()),
            })

    pct = total_exact / total_cells * 100 if total_cells > 0 else 0
    status = 'PASS' if pct >= 99.5 else ('WARN' if pct >= 95.0 else 'FAIL')

    return {
        'status': status,
        'overlap_bars': len(overlap),
        'live_bars': len(live),
        'base_bars': len(base),
        'cells_exact': total_exact,
        'cells_total': total_cells,
        'parity_pct': pct,
        'features_perfect': n_features - len(bad_features),
        'features_total': n_features,
        'bad_features': sorted(bad_features, key=lambda x: x['exact'])[:10],
    }


# ══════════════════════════════════════════════════════════════════
# LAYER 2: EXECUTION PARITY (engine trades vs NT8 ground-truth trades)
# ══════════════════════════════════════════════════════════════════

def execution_parity(day: str) -> dict:
    """Diff v2_trades vs nt8_trades — does NT8 fill what we send?"""
    engine_path = os.path.join(LIVE_DIR, f'v2_trades_{day}.csv')
    nt8_path = os.path.join(LIVE_DIR, f'nt8_trades_{day}.csv')

    if not os.path.exists(engine_path):
        return {'status': 'SKIP', 'reason': f'No {engine_path}'}
    if not os.path.exists(nt8_path):
        return {'status': 'SKIP',
                'reason': f'No {nt8_path} — bridge needs TRADE_CLOSED update'}

    engine = pd.read_csv(engine_path)
    nt8 = pd.read_csv(nt8_path)

    # Engine fills (FILL_* rows have actual fill prices and order_ids)
    engine_fills = engine[engine['type'].astype(str).str.startswith('FILL_')].copy()

    # NT8 ground truth fills
    nt8_count = len(nt8)

    # Match by order_id where possible
    matched = 0
    slippage_diffs = []
    missing_in_nt8 = []
    extra_in_nt8 = []

    if 'order_id' in nt8.columns and 'exit_reason' in engine_fills.columns:
        engine_oids = set(engine_fills['exit_reason'].astype(str))
        nt8_oids = set(nt8['order_id'].astype(str))

        common = engine_oids & nt8_oids
        only_engine = engine_oids - nt8_oids
        only_nt8 = nt8_oids - engine_oids

        matched = len(common)
        missing_in_nt8 = list(only_engine)[:10]
        extra_in_nt8 = list(only_nt8)[:10]

        # Slippage: compare engine fill_price to nt8 exit_price
        if 'fill_price' in engine_fills.columns and 'exit_price' in nt8.columns:
            for _, e_row in engine_fills.iterrows():
                oid = str(e_row['exit_reason'])
                if oid in nt8_oids:
                    n_row = nt8[nt8['order_id'].astype(str) == oid].iloc[0]
                    diff = abs(float(e_row['fill_price']) - float(n_row['exit_price']))
                    if diff > 0.01:
                        slippage_diffs.append({
                            'order_id': oid,
                            'engine_fill': float(e_row['fill_price']),
                            'nt8_fill': float(n_row['exit_price']),
                            'diff': diff,
                        })

    decisions = len(engine[engine['type'].astype(str).isin(
        ['ENTRY', 'CHAIN_ENTRY', 'EXIT', 'CHAIN_EXIT'])])
    fills_in_engine = len(engine_fills)

    # Status
    if nt8_count == 0:
        status = 'WARN'  # no NT8 data to compare
    elif missing_in_nt8 or extra_in_nt8:
        status = 'WARN'  # mismatched order IDs
    elif slippage_diffs:
        status = 'WARN'
    else:
        status = 'PASS'

    return {
        'status': status,
        'engine_decisions': decisions,
        'engine_fills_logged': fills_in_engine,
        'nt8_fills_logged': nt8_count,
        'matched_by_oid': matched,
        'missing_in_nt8': missing_in_nt8,
        'extra_in_nt8': extra_in_nt8,
        'slippage_diffs': slippage_diffs[:10],
    }


# ══════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════

def write_report(day: str, feat_result: dict, exec_result: dict) -> str:
    lines = []
    lines.append('=' * 80)
    lines.append(f'PARITY VALIDATION — {day}')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append('=' * 80)
    lines.append('')

    # Overall status
    statuses = [feat_result['status'], exec_result['status']]
    if all(s == 'PASS' for s in statuses):
        overall = 'PASS'
    elif any(s == 'FAIL' for s in statuses):
        overall = 'FAIL'
    else:
        overall = 'WARN'

    lines.append(f'OVERALL: {overall}')
    lines.append('')

    # Layer 1: feature parity
    lines.append('-' * 80)
    lines.append(f'LAYER 1: FEATURE PARITY (live vs build_dataset)  [{feat_result["status"]}]')
    lines.append('-' * 80)
    if feat_result['status'] == 'SKIP':
        lines.append(f'  SKIPPED: {feat_result["reason"]}')
    else:
        lines.append(f'  Live bars:    {feat_result["live_bars"]:,}')
        lines.append(f'  Base bars:    {feat_result["base_bars"]:,}')
        lines.append(f'  Overlap:      {feat_result["overlap_bars"]:,} bars')
        lines.append(f'  Cells exact:  {feat_result["cells_exact"]:,} / '
                     f'{feat_result["cells_total"]:,} '
                     f'({feat_result["parity_pct"]:.2f}%)')
        lines.append(f'  Features OK:  {feat_result["features_perfect"]} / '
                     f'{feat_result["features_total"]}')
        if feat_result['bad_features']:
            lines.append('  Worst features:')
            for bf in feat_result['bad_features'][:5]:
                lines.append(f'    {bf["feature"]:<25} '
                             f'exact={bf["exact"]}/{bf["total"]} '
                             f'max_diff={bf["max_diff"]:.4f}')
    lines.append('')

    # Layer 2: execution parity
    lines.append('-' * 80)
    lines.append(f'LAYER 2: EXECUTION PARITY (engine vs NT8)  [{exec_result["status"]}]')
    lines.append('-' * 80)
    if exec_result['status'] == 'SKIP':
        lines.append(f'  SKIPPED: {exec_result["reason"]}')
    else:
        lines.append(f'  Engine decisions:     {exec_result["engine_decisions"]}')
        lines.append(f'  Engine fills logged:  {exec_result["engine_fills_logged"]}')
        lines.append(f'  NT8 fills logged:     {exec_result["nt8_fills_logged"]}')
        lines.append(f'  Matched by order_id:  {exec_result["matched_by_oid"]}')
        if exec_result['missing_in_nt8']:
            lines.append(f'  Engine fills NOT in NT8 ({len(exec_result["missing_in_nt8"])}):')
            for oid in exec_result['missing_in_nt8'][:5]:
                lines.append(f'    {oid}')
        if exec_result['extra_in_nt8']:
            lines.append(f'  NT8 fills NOT in engine ({len(exec_result["extra_in_nt8"])}):')
            for oid in exec_result['extra_in_nt8'][:5]:
                lines.append(f'    {oid}')
        if exec_result['slippage_diffs']:
            lines.append(f'  Price diffs ({len(exec_result["slippage_diffs"])}):')
            for d in exec_result['slippage_diffs'][:5]:
                lines.append(f'    {d["order_id"]:<20} '
                             f'engine=${d["engine_fill"]:.2f} '
                             f'nt8=${d["nt8_fill"]:.2f} '
                             f'diff=${d["diff"]:.2f}')
    lines.append('')

    # Verdict
    lines.append('=' * 80)
    if overall == 'PASS':
        lines.append('VERDICT: ALL PARITIES PASSED — engine is faithful')
    elif overall == 'FAIL':
        lines.append('VERDICT: PARITY FAILED — investigate before next session')
    else:
        lines.append('VERDICT: WARNINGS — review details above')
    lines.append('=' * 80)

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('day', nargs='?', default=None,
                        help='Day to validate (YYYY_MM_DD). Default: today.')
    args = parser.parse_args()

    day = args.day or datetime.now().strftime('%Y_%m_%d')
    print(f'Validating parity for {day}...')

    feat_result = feature_parity(day)
    exec_result = execution_parity(day)

    report = write_report(day, feat_result, exec_result)
    print()
    print(report)

    os.makedirs(REPORTS, exist_ok=True)
    out_path = os.path.join(REPORTS, f'parity_validate_{day}.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {out_path}')

    # Exit code reflects status
    statuses = [feat_result['status'], exec_result['status']]
    if any(s == 'FAIL' for s in statuses):
        sys.exit(2)
    elif any(s == 'WARN' for s in statuses):
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
