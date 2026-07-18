"""
DOJO FORGE - Gen-0 Set Selection
Builds a 150-episode set (100 training, 50 held-out baseline) of winner/midflip 
episodes from 2025-26 on FRESH days (never seen in pilot, full_run, or wrongdir).
"""
import os
import sys
import json
import argparse
from typing import Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
EXIT_DOJO_ROOT = os.path.join(ROOT, 'research', 'exit_dojo')
sys.path.insert(0, os.path.join(EXIT_DOJO_ROOT, 'builders'))

import telescope_packet_builder as tb

DOJO_FORGE_ROOT = os.path.abspath(os.path.join(HERE, '..'))
REPORTS_DIR = os.path.join(DOJO_FORGE_ROOT, 'reports', 'gen0')
SELECTION_JSON = os.path.join(REPORTS_DIR, 'selection.json')

def load_excluded_days() -> set:
    excluded = set()
    paths = [
        os.path.join(EXIT_DOJO_ROOT, 'reports', 'pilot_10_eps', 'selection.json'),
        os.path.join(EXIT_DOJO_ROOT, 'reports', 'full_run', 'selection.json'),
        os.path.join(EXIT_DOJO_ROOT, 'reports', 'wrongdir', 'selection.json')
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r') as f:
                data = json.load(f)
                if 'episodes' in data:
                    for ep in data['episodes']:
                        if 'day' in ep:
                            excluded.add(ep['day'])
                elif isinstance(data, list):
                    for ep in data:
                        if 'day' in ep:
                            excluded.add(ep['day'])
    print(f"Loaded {len(excluded)} excluded days from pilot, full_run, wrongdir.")
    return excluded

def engagements_with_exclusions(excluded_days: set):
    """Overrides the default telescope_packet_builder engagement filtering to exclude additional days."""
    eng = tb.engagements()
    sub = eng[~eng['day'].isin(excluded_days)].copy()
    sub.attrs['p90_thr'] = eng.attrs.get('p90_thr', 0)
    return sub

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20260718)
    ap.add_argument('--n-winner', type=int, default=75)
    ap.add_argument('--n-midflip', type=int, default=75)
    args = ap.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    excluded_days = load_excluded_days()

    # Instead of tb.select_full_run (which enforces 1 per day), we do a flat selection
    eng = engagements_with_exclusions(excluded_days)
    thr = eng.attrs['p90_thr']
    days = sorted(eng['day'].unique())

    # Scan
    candidates = {'winner': [], 'midflip': []}
    day_groups = {d: g for d, g in eng.groupby('day', sort=False)}
    import tqdm
    for day in tqdm.tqdm(days, desc='scan days'):
        dd = tb.eb.load_day_data(day)
        if dd is None:
            continue
        for r in day_groups[day].itertuples(index=False):
            ets, isl = int(r.ts), bool(r.is_long)
            lem = tb.eb.label_flip_minute(dd.oracle_ivals, ets, isl,
                                          int(min(tb.WINDOW_CAP, (dd.session_end - ets) // 60)))
            wm = tb._window_minutes(dd.session_end, ets, lem)
            if wm < tb.MIN_WINDOW_MIN:
                continue
            dp, entry_price = tb.eb.signed_drift_path(dd.ts5, dd.c5, ets, isl, wm)
            buckets = tb.eb.natural_buckets(lem, chop=False)
            for b in buckets:
                if b in candidates:
                    om = lem if lem is not None else wm
                    candidates[b].append(dict(
                        day=day, ts=ets, is_long=isl, P=float(r.P), det=r.det, type=b,
                        window_minutes=wm, label_end_minute=lem, oracle_minute=om,
                        oracle_capture=float(dp[om]), per_minute_forward_drift=dp,
                        entry_price=entry_price, chop_tol=None))

    rng = np.random.default_rng(args.seed)
    rng.shuffle(candidates['winner'])
    rng.shuffle(candidates['midflip'])
    
    selected = candidates['winner'][:args.n_winner] + candidates['midflip'][:args.n_midflip]
    selected.sort(key=lambda s: (s['type'], s['day'], s['ts']))
    
    meta = dict(seed=args.seed, p90_thr=thr, n_days_scanned=len(days),
                targets={'winner': args.n_winner, 'midflip': args.n_midflip},
                n_selected=len(selected))
    
    print(f"Chose {len(selected)} episodes (winner={len(candidates['winner'][:args.n_winner])}, midflip={len(candidates['midflip'][:args.n_midflip])}) on {len({s['day'] for s in selected})} days.")
    
    with open(SELECTION_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(meta=meta, episodes=[tb._manifest_row(s) for s in selected]), f, indent=2)
    
    print(f"Wrote {SELECTION_JSON}")

if __name__ == '__main__':
    main()
