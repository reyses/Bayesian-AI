"""
Parity Check — compare live trade log vs baseline forward pass on same day.

Runs baseline engine on feature files, captures every entry/exit decision,
then diffs against the live trade log timestamp by timestamp.

Usage:
    python tools/parity_check.py 2026_04_12

Output: reports/findings/parity_YYYY_MM_DD.txt
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from training.nightmare_blended import BlendedEngine

FEATURES_DIR = 'DATA/FEATURES_NT8_5s'
ATLAS_1M = 'DATA/ATLAS_NT8/1m'
LIVE_TRADES = 'reports/live'


def run_baseline_day(day_name):
    """Run forward pass on one day, return list of trade events with timestamps."""
    fpath = os.path.join(FEATURES_DIR, f'{day_name}.parquet')
    if not os.path.exists(fpath):
        print(f'No features for {day_name}')
        return []

    price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
    if not os.path.exists(price_file):
        price_file = None

    engine = BlendedEngine(use_cnn=False)
    events = []

    prev_in_pos = False
    prev_trades = 0
    prev_chains = 0

    ft = FeatureTicker(fpath, price_file=price_file)
    for state in ft:
        ts = state['timestamp']
        price = state['price']
        feat = state['features_79d']

        engine.on_state(state)

        # Detect events (same logic as engine_v2)
        entered = engine.in_pos and not prev_in_pos
        exited = not engine.in_pos and prev_in_pos
        new_trades = len(engine.trades) - prev_trades
        curr_chains = len(engine._chain_contracts)
        chain_opened = curr_chains > prev_chains

        if entered:
            events.append({
                'timestamp': ts, 'type': 'ENTRY',
                'tier': engine.entry_tier, 'direction': engine.direction,
                'price': price, 'z': feat[12], 'vr': feat[14],
            })

        if chain_opened and engine.in_pos:
            cc = engine._chain_contracts[-1]
            events.append({
                'timestamp': ts, 'type': 'CHAIN_ENTRY',
                'tier': cc['entry_tier'], 'direction': cc['direction'],
                'price': price, 'z': feat[12], 'vr': feat[14],
            })

        # Chain exits
        chain_exit_count = new_trades - (1 if exited else 0)
        if chain_exit_count > 0:
            for ci in range(chain_exit_count):
                ct = engine.trades[-(new_trades - ci - (1 if exited else 0))]
                events.append({
                    'timestamp': ts, 'type': 'CHAIN_EXIT',
                    'tier': ct.get('entry_tier', '?'),
                    'direction': ct.get('direction', '?'),
                    'price': price, 'pnl': ct['pnl'],
                    'exit_reason': ct.get('exit_reason', ''),
                })

        if exited and new_trades > 0:
            t = engine.trades[-1] if chain_exit_count == 0 else \
                engine.trades[-(new_trades)]
            events.append({
                'timestamp': ts, 'type': 'EXIT',
                'tier': t.get('entry_tier', '?'),
                'direction': t.get('direction', '?'),
                'price': price, 'pnl': t['pnl'],
                'exit_reason': t.get('exit_reason', ''),
            })

        prev_in_pos = engine.in_pos
        prev_trades = len(engine.trades)
        prev_chains = curr_chains

    engine.force_close()
    # Capture force_close trades
    if len(engine.trades) > prev_trades:
        for t in engine.trades[prev_trades:]:
            events.append({
                'timestamp': ts, 'type': 'FORCE_CLOSE',
                'tier': t.get('entry_tier', '?'),
                'direction': t.get('direction', '?'),
                'price': price, 'pnl': t['pnl'],
                'exit_reason': t.get('exit_reason', 'end_of_day'),
            })

    return events


def load_live_trades(day_name):
    """Load live trade log for a day."""
    path = os.path.join(LIVE_TRADES, f'v2_trades_{day_name}.csv')
    if not os.path.exists(path):
        print(f'No live trades for {day_name}')
        return []
    df = pd.read_csv(path)
    events = []
    for _, r in df.iterrows():
        events.append({
            'timestamp': r['timestamp'],
            'type': r['type'],
            'tier': r.get('tier', '?'),
            'direction': r.get('direction', '?'),
            'price': r.get('price', 0) if 'price' in r else r.get('requested_price', 0),
            'pnl': r.get('pnl', 0),
        })
    return events


def compare(baseline, live, day_name):
    """Compare baseline vs live events. Returns report lines."""
    lines = []
    lines.append('=' * 75)
    lines.append(f'PARITY CHECK: {day_name}')
    lines.append(f'  Baseline events: {len(baseline)}')
    lines.append(f'  Live events: {len(live)}')
    lines.append('=' * 75)
    lines.append('')

    # Filter to entries + exits only (skip FILL_ rows from live)
    b_entries = [e for e in baseline if 'ENTRY' in e['type']]
    b_exits = [e for e in baseline if 'EXIT' in e['type'] or e['type'] == 'FORCE_CLOSE']
    l_entries = [e for e in live if 'ENTRY' in e['type'] and 'FILL' not in e['type']]
    l_exits = [e for e in live if 'EXIT' in e['type'] and 'FILL' not in e['type']]

    lines.append(f'ENTRIES: baseline={len(b_entries)} live={len(l_entries)}')
    lines.append(f'EXITS:   baseline={len(b_exits)} live={len(l_exits)}')
    lines.append('')

    # Summary by tier
    from collections import Counter
    b_tier_count = Counter(e['tier'] for e in b_entries)
    l_tier_count = Counter(e['tier'] for e in l_entries)
    all_tiers = sorted(set(list(b_tier_count.keys()) + list(l_tier_count.keys())))

    lines.append(f'{"Tier":<20} {"Baseline":>10} {"Live":>10} {"Match":>8}')
    lines.append('-' * 52)
    for tier in all_tiers:
        b = b_tier_count.get(tier, 0)
        l = l_tier_count.get(tier, 0)
        match = 'OK' if abs(b - l) <= max(1, b * 0.1) else f'DIFF {l-b:+d}'
        lines.append(f'{tier:<20} {b:>10} {l:>10} {match:>8}')
    lines.append('')

    # PnL comparison
    b_pnl = sum(e.get('pnl', 0) for e in baseline)
    l_pnl = sum(e.get('pnl', 0) for e in live)
    lines.append(f'PnL: baseline=${b_pnl:+.0f} live=${l_pnl:+.0f} diff=${l_pnl-b_pnl:+.0f}')
    lines.append('')

    # Timestamp matching — find entries within 60s of each other
    matched = 0
    unmatched_baseline = []
    unmatched_live = list(l_entries)

    lines.append('ENTRY-BY-ENTRY MATCHING (within 60s):')
    lines.append(f'{"Time":>8} {"B_tier":<18} {"L_tier":<18} {"Match":>6} {"B_price":>10} {"L_price":>10}')
    lines.append('-' * 75)

    for be in b_entries:
        best_match = None
        best_dt = 999
        for le in unmatched_live:
            dt = abs(be['timestamp'] - le['timestamp'])
            if dt < best_dt and dt < 60:
                best_dt = dt
                best_match = le

        time_str = datetime.utcfromtimestamp(be['timestamp']).strftime('%H:%M:%S')
        if best_match:
            unmatched_live.remove(best_match)
            tier_ok = be['tier'] == best_match['tier']
            dir_ok = be.get('direction', '') == best_match.get('direction', '')
            status = 'OK' if (tier_ok and dir_ok) else 'DIFF'
            if not tier_ok:
                status = f'TIER'
            if not dir_ok:
                status = f'DIR'
            matched += 1
            lines.append(f'{time_str:>8} {be["tier"]:<18} {best_match["tier"]:<18} '
                         f'{status:>6} {be["price"]:>10.2f} {best_match["price"]:>10.2f}')
        else:
            unmatched_baseline.append(be)
            lines.append(f'{time_str:>8} {be["tier"]:<18} {"---MISSING---":<18} '
                         f'{"MISS":>6} {be["price"]:>10.2f} {"":>10}')

    lines.append('')
    lines.append(f'Matched: {matched}/{len(b_entries)} baseline entries')
    if unmatched_live:
        lines.append(f'Extra live entries (not in baseline): {len(unmatched_live)}')
        for le in unmatched_live[:10]:
            t = datetime.utcfromtimestamp(le['timestamp']).strftime('%H:%M:%S')
            lines.append(f'  {t} {le["type"]} {le["tier"]} {le.get("direction","")}')
    if unmatched_baseline:
        lines.append(f'Missing live entries (in baseline not live): {len(unmatched_baseline)}')
        for be in unmatched_baseline[:10]:
            t = datetime.utcfromtimestamp(be['timestamp']).strftime('%H:%M:%S')
            lines.append(f'  {t} {be["type"]} {be["tier"]} {be.get("direction","")}')

    lines.append('')
    parity_pct = matched / max(len(b_entries), 1) * 100
    lines.append(f'PARITY SCORE: {parity_pct:.0f}% ({matched}/{len(b_entries)} entries matched)')
    lines.append('=' * 75)

    return lines


def main():
    if len(sys.argv) < 2:
        print('Usage: python tools/parity_check.py YYYY_MM_DD')
        return

    day_name = sys.argv[1]
    print(f'Running baseline on {day_name}...')
    baseline = run_baseline_day(day_name)
    print(f'  Baseline: {len(baseline)} events')

    print(f'Loading live trades for {day_name}...')
    live = load_live_trades(day_name)
    print(f'  Live: {len(live)} events')

    report_lines = compare(baseline, live, day_name)
    report = '\n'.join(report_lines)
    print(report)

    os.makedirs('reports/findings', exist_ok=True)
    out_path = f'reports/findings/parity_{day_name}.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
