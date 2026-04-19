"""
Isolated pipeline V2 — 9-tier physics, no CNN.

Defaults (common tier-fixing workflow):
  * Phase 1 forward pass only, 9 tiers isolated, chains OFF (one position
    per tier at a time), regret skipped.
  * Each tier runs in its own IsoEngine with only_tier=T. No cross-tier
    interference. Per-tier trades tagged with entry_tier in the output.

Flags (all opt-in):
  --regret          Run Phase 2 regret analysis (10/30-min bounded)
  --chains N        Enable chain positions up to N per tier (N>=1, default 1)
  --tier NAME ...   Restrict to specific tiers (e.g. --tier NMP_FADE RIDE_AGAINST)

Pickle output naming:
  Full-engine runs write `iso_is.pkl` (canonical baseline).
  Single-tier or restricted runs write `iso_is_<TIERS>.pkl` with a `+`-joined
  suffix so they don't clobber the baseline. Example:
    --tier KILL_SHOT                       →  iso_is_KILL_SHOT.pkl
    --tier KILL_SHOT KILL_SHOT_INVERSE     →  iso_is_KILL_SHOT+KILL_SHOT_INVERSE.pkl

Back-compat: --no-regret is accepted as a no-op (matches current default).

Usage:
    python training_iso/run_iso.py                         # default Phase-1 all-tier
    python training_iso/run_iso.py --chains 4              # with chain multiplier
    python training_iso/run_iso.py --tier RIDE_AGAINST     # single tier
    python training_iso/run_iso.py --regret                # also run regret
"""
import os
import sys
import glob
import time as _time
import subprocess
import pickle
import shutil
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Features live inside the atlas folder after the refactor
FEATURES_DIR_SEQ = 'DATA/ATLAS/FEATURES_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
ATLAS_5S = 'DATA/ATLAS/5s'   # source for slope (β) computation — available to any tier
OUTPUT_DIR = 'training_iso/output'


def _resolve_days(target, source_dir):
    all_files = sorted(glob.glob(os.path.join(source_dir, '*.parquet')))
    if target == 'is':
        return [f for f in all_files if '2025_' in os.path.basename(f)]
    elif target == 'oos':
        return [f for f in all_files if '2026_' in os.path.basename(f)]
    return all_files


def _print_summary(results):
    if not results:
        print('No results.')
        return
    n_days = len(results)
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    winning_days = sum(1 for r in results if r['pnl'] > 0)
    print(f'\n{"="*60}')
    print(f'RESULTS: {n_days} days | {total_trades} trades | ${total_pnl:.2f}')
    print(f'  $/day: ${total_pnl / max(n_days, 1):.2f}')
    print(f'  Winning days: {winning_days}/{n_days}')
    print(f'{"="*60}')


def run_iso_forward(target='is', only_tiers=None, max_chains=4):
    """Run isolated forward pass — each tier gets its OWN engine, no interference.

    By default all 9 tiers run in parallel on the same bar stream. Pass
    only_tiers=['TREND_FOLLOWER'] (etc) to restrict to a subset — useful
    for fast single-tier iteration.

    `max_chains` controls how many concurrent positions each tier engine can
    hold. Default 4 = primary + up to 3 chains. Pass 1 to disable chaining
    (single position per tier, legacy behavior).

    Output trades carry `entry_tier` and `chain_idx` so per-tier metrics
    roll up cleanly (chain_idx=0 is primary, 1+ are chain positions).
    """
    from training_iso.nightmare_iso import IsoEngine, TIER_PRIORITY
    from training.sfe_ticker import FeatureTicker
    from tqdm import tqdm
    from collections import Counter

    feat_files = _resolve_days(target, FEATURES_DIR_SEQ)
    if not feat_files:
        print(f'No feature files for "{target}"')
        return [], []

    active_tiers = only_tiers if only_tiers else TIER_PRIORITY
    print(f'ISO FORWARD — {len(feat_files)} day(s), tiers: {active_tiers}, '
          f'max_chains={max_chains}')
    all_results = []
    all_trades = []

    for fpath in tqdm(feat_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None
        # 5s closes for slope (β) computation — loaded once, shared across engines
        sec_file = os.path.join(ATLAS_5S, f'{day_name}.parquet')
        sec_df = pd.read_parquet(sec_file) if os.path.exists(sec_file) else None

        engines = {t: IsoEngine(only_tier=t, max_chains=max_chains)
                   for t in active_tiers}
        for eng in engines.values():
            eng.set_sec_closes(sec_df)
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            for eng in engines.values():
                eng.on_state(state)
        for eng in engines.values():
            eng.force_close()

        day_trades = []
        for tier, eng in engines.items():
            for t in eng.trades:
                t['day'] = day_name
            day_trades.extend(eng.get_full_trades())
        all_trades.extend(day_trades)

        day_pnl = sum(eng.daily_pnl for eng in engines.values())
        n_trades = len(day_trades)
        all_results.append({
            'day': day_name,
            'trades': n_trades,
            'pnl': day_pnl,
            'wr': sum(1 for t in day_trades if t['pnl'] > 0) / max(n_trades, 1) * 100,
        })

    _print_summary(all_results)

    if all_trades:
        print()
        print(f'{"Tier":<17} {"N":>6} {"WR":>5} {"Total":>10} {"$/trade":>9}')
        print('-' * 55)
        tiers = Counter(t.get('entry_tier', '?') for t in all_trades)
        for tier in active_tiers:
            count = tiers.get(tier, 0)
            if count == 0:
                print(f'{tier:<17} {0:>6}  (no trades)')
                continue
            sub = [t for t in all_trades if t.get('entry_tier') == tier]
            wr = sum(1 for t in sub if t['pnl'] > 0) / len(sub) * 100
            total = sum(t['pnl'] for t in sub)
            per = total / len(sub)
            print(f'{tier:<17} {count:>6,} {wr:>4.0f}% ${total:>+9,.0f} ${per:>+8.2f}')

        # Per-tier PnL MODE bucket breakdown — surfaces where each tier's
        # trades land on the $/trade spectrum. Mode in REAL_WIN/STRONG_WIN
        # = healthy; mode in MED_LOSS/BIG_LOSS = tier needs a fix.
        _print_tier_mode_buckets(all_trades, active_tiers)

    return all_results, all_trades


# PnL bucket boundaries (mirror tick-buckets on dollars, signed).
# 10 buckets: 5 loss tiers + 5 win tiers.
_PNL_BUCKETS = [
    ('BIG_LOSS',      float('-inf'),  -50.0),
    ('MED_LOSS',      -50.0,          -25.0),
    ('REAL_LOSS',     -25.0,          -10.0),
    ('MARG_LOSS',     -10.0,           -5.0),
    ('NOISE_LOSS',     -5.0,            0.0),
    ('NOISE_WIN',       0.0,            5.0),
    ('MARG_WIN',        5.0,           10.0),
    ('REAL_WIN',       10.0,           25.0),
    ('STRONG_WIN',     25.0,           50.0),
    ('BIG_WIN',        50.0, float('inf')),
]


def _bucket_of(pnl):
    for name, lo, hi in _PNL_BUCKETS:
        if lo <= pnl < hi:
            return name
    return 'BIG_WIN' if pnl >= 50 else 'BIG_LOSS'


def _print_tier_mode_buckets(trades, active_tiers):
    """Per-tier PnL bucket distribution + mode. Highlights where each tier
    bleeds (mode in loss buckets) vs thrives (mode in win buckets)."""
    from collections import Counter
    print()
    print(f'{"="*100}')
    print(f'PnL MODE BUCKETS per tier (which bucket is most populated)')
    print(f'{"="*100}')
    print(f'  Buckets: BIG_LOSS<-$50 · MED_LOSS<-$25 · REAL_LOSS<-$10 · '
          f'MARG_LOSS<-$5 · NOISE_LOSS<$0')
    print(f'           NOISE_WIN<$5 · MARG_WIN<$10 · REAL_WIN<$25 · '
          f'STRONG_WIN<$50 · BIG_WIN>=$50')
    print()
    # Compact header (bucket abbreviations)
    abbrevs = ['BL', 'ML', 'RL', 'mL', 'nL', 'nW', 'mW', 'RW', 'SW', 'BW']
    header = (f'  {"Tier":<18} {"MODE":<12} '
              + ' '.join(f'{a:>4}' for a in abbrevs))
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for tier in active_tiers:
        sub = [t for t in trades if t.get('entry_tier') == tier]
        if not sub:
            continue
        counts = Counter(_bucket_of(t.get('pnl', 0.0)) for t in sub)
        mode_name = counts.most_common(1)[0][0]
        mode_n = counts[mode_name]
        mode_pct = mode_n / len(sub) * 100
        cells = []
        for name, _, _ in _PNL_BUCKETS:
            n = counts.get(name, 0)
            pct = n / len(sub) * 100 if n else 0
            cells.append(f'{pct:>3.0f}%' if n else '   -')
        mode_str = f'{mode_name}({mode_pct:.0f}%)'
        print(f'  {tier:<18} {mode_str:<12} ' + ' '.join(cells))

    # Overall
    counts = Counter(_bucket_of(t.get('pnl', 0.0)) for t in trades)
    mode_name = counts.most_common(1)[0][0]
    mode_n = counts[mode_name]
    mode_pct = mode_n / len(trades) * 100
    cells = []
    for name, _, _ in _PNL_BUCKETS:
        n = counts.get(name, 0)
        pct = n / len(trades) * 100 if n else 0
        cells.append(f'{pct:>3.0f}%' if n else '   -')
    print('  ' + '-' * (len(header) - 2))
    mode_str = f'{mode_name}({mode_pct:.0f}%)'
    print(f'  {"ALL":<18} {mode_str:<12} ' + ' '.join(cells))
    print()


def run_regret(pkl_name: str = 'iso_is.pkl'):
    """Run bounded regret + produce corrected trades.

    Uses training_iso/regret.py which caps the counterfactual window to
    LOOKBACK_MIN=10 / LOOKAHEAD_MIN=30, gates EXTENDED options on the
    peak-validity check, and produces corrected trades that exit at the
    SPECIFIC peak bar each gated best_action points at (not the overall
    argmax).

    `pkl_name` defaults to the canonical full-engine baseline. Single-tier
    runs pass their tier-suffixed pickle name so Phase 2 analyzes the same
    trades Phase 1 just produced.
    """
    from training_iso.regret import compute_all_regrets, correct_trades

    trade_path = os.path.join(OUTPUT_DIR, 'trades', pkl_name)
    with open(trade_path, 'rb') as f:
        trades = pickle.load(f)
    print(f'Regret on {len(trades)} ISO trades...')

    regret_df = compute_all_regrets(trades)
    actual = regret_df['actual_pnl'].sum()
    optimal = regret_df['best_pnl'].sum()
    print(f'  Actual:  ${actual:,.0f}')
    print(f'  Optimal: ${optimal:,.0f}')
    print(f'  Capture: {actual / max(optimal, 1) * 100:.1f}%')

    # Per-tier
    if 'entry_tier' in regret_df.columns:
        for tier in regret_df['entry_tier'].unique():
            sub = regret_df[regret_df['entry_tier'] == tier]
            print(f'    {tier}: {len(sub)} trades, '
                  f'actual=${sub["actual_pnl"].sum():,.0f}, '
                  f'optimal=${sub["best_pnl"].sum():,.0f}')

    # Best action breakdown + extended-validity rates
    n_counter = regret_df['best_action'].str.contains('counter').sum()
    print(f'  Counter: {n_counter} ({n_counter / len(regret_df) * 100:.0f}%)')
    if 'same_extended_valid' in regret_df.columns:
        se_valid = regret_df['same_extended_valid'].sum()
        ce_valid = regret_df['counter_extended_valid'].sum()
        print(f'  Peak-valid extended: same={se_valid} '
              f'({se_valid/len(regret_df)*100:.0f}%)  '
              f'counter={ce_valid} ({ce_valid/len(regret_df)*100:.0f}%)')

    os.makedirs(os.path.join(OUTPUT_DIR, 'tree'), exist_ok=True)
    regret_df.to_csv(os.path.join(OUTPUT_DIR, 'tree', 'regret_analysis.csv'), index=False)

    # ── Corrected trades (oracle ground truth: exit at the gated peak) ─
    print()
    print('Generating corrected trades (peak-aware oracle)...')
    corrected = correct_trades(trades)
    with open(os.path.join(OUTPUT_DIR, 'trades', 'corrected_is.pkl'), 'wb') as f:
        pickle.dump(corrected, f)
    flat = [{k: v for k, v in t.items() if not isinstance(v, (list, dict, np.ndarray))}
            for t in corrected]
    pd.DataFrame(flat).to_csv(os.path.join(OUTPUT_DIR, 'trades', 'corrected_is.csv'),
                              index=False)

    # Compare actual vs corrected PnL (sanity + delta)
    actual_total = sum(t['original_pnl'] for t in corrected)
    corrected_total = sum(t['pnl'] for t in corrected)
    flips = sum(1 for t in corrected if t['dir'] != t['original_dir'])
    avg_corrected_held = np.mean([t['held'] for t in corrected]) if corrected else 0
    avg_original_held = np.mean([t['original_held'] for t in corrected]) if corrected else 0
    print(f'  Corrected trades: {len(corrected):,}')
    print(f'  Direction flips:  {flips:,} ({flips/max(len(corrected),1)*100:.0f}%)')
    print(f'  Actual    $: ${actual_total:+,.0f}  avg held={avg_original_held:.1f} bars')
    print(f'  Corrected $: ${corrected_total:+,.0f}  avg held={avg_corrected_held:.1f} bars')
    print(f'  Delta:      ${corrected_total - actual_total:+,.0f}  '
          f'({(corrected_total/max(actual_total,1)-1)*100:+.0f}%)')


def main():
    """Iso pipeline — Phase 1 (forward pass) only by default.

    Defaults (for the common tier-fixing workflow):
        no regret phase, chains=1 (isolated single position per tier)

    Opt-in flags:
        --regret         : also run Phase 2 regret analysis
        --chains N       : enable chain positions up to N per tier (1 = off)
        --tier NAME ...  : restrict to specific tier(s)

    Back-compat: --no-regret is accepted but a no-op (that is now the default).
    """
    from training_iso.nightmare_iso import TIER_PRIORITY
    args = sys.argv[1:]

    # Default: skip regret. --regret opts IN. --no-regret is accepted as
    # a no-op for back-compat (that's the default now).
    run_regret_phase = '--regret' in args

    # --tier TIER_NAME (one or more) restricts run to those tiers only
    only_tiers = None
    if '--tier' in args:
        idx = args.index('--tier')
        tier_args = []
        for a in args[idx + 1:]:
            if a.startswith('--'):
                break
            tier_args.append(a)
        if not tier_args:
            raise SystemExit('--tier requires at least one tier name')
        for t in tier_args:
            if t not in TIER_PRIORITY:
                raise SystemExit(f'unknown tier {t!r}; valid: {TIER_PRIORITY}')
        only_tiers = tier_args

    # --chains N sets max concurrent positions per tier (default 1 = off)
    max_chains = 1
    if '--chains' in args:
        idx = args.index('--chains')
        if idx + 1 >= len(args) or args[idx + 1].startswith('--'):
            raise SystemExit('--chains requires an integer')
        try:
            max_chains = int(args[idx + 1])
        except ValueError:
            raise SystemExit(f'--chains value must be an integer, got {args[idx+1]!r}')
        if max_chains < 1:
            raise SystemExit('--chains must be >= 1')

    n_tiers = len(only_tiers) if only_tiers else len(TIER_PRIORITY)
    print(f'{"="*60}')
    print(f'ISOLATED PIPELINE V2 — {n_tiers} per-tier engines, no CNN')
    print(f'  Isolation: ON (1 engine per tier, no cross-tier interference)')
    if only_tiers:
        print(f'  Restricted to tiers: {only_tiers}')
    else:
        print(f'  Tiers active: all {n_tiers}')
    print(f'  Max chains per tier: {max_chains}'
          + ('' if max_chains > 1 else '  [chaining OFF]'))
    print(f'  Regret phase: '
          + ('ENABLED (10/30-min window)' if run_regret_phase else 'SKIPPED [default]'))
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*60}')

    pipeline_start = _time.perf_counter()

    # Phase 1: Collect trades
    print(f'\n{"="*40}')
    print(f'PHASE 1: NMP two-mode forward pass')
    print(f'{"="*40}')
    t0 = _time.perf_counter()
    results, trades = run_iso_forward('is', only_tiers=only_tiers,
                                      max_chains=max_chains)
    # Compute the pickle name even if trades is empty — regret phase needs
    # the path to know which file to read (it was written by a prior run
    # with same tier scope, most likely).
    if only_tiers:
        pkl_name = f'iso_is_{"+".join(only_tiers)}.pkl'
        csv_name = f'iso_is_{"+".join(only_tiers)}.csv'
    else:
        pkl_name = 'iso_is.pkl'
        csv_name = 'iso_is.csv'
    if trades:
        os.makedirs(os.path.join(OUTPUT_DIR, 'trades'), exist_ok=True)
        # With chains enabled we can hit 30k+ trades; keeping per-bar 91D
        # feature vectors in the path list blows out RAM during pickle dump.
        # For large runs, strip `features` from path/approach bars — keep
        # bar/ts/price/pnl/peak_pnl (small scalars). Top-level `entry_79d`
        # and `exit_79d` remain for Q-analysis at segment level.
        #
        # Small runs (e.g. --tier X --chains 1) keep features so Q3
        # peak-signature re-analysis continues to work.
        STRIP_THRESHOLD = 12000  # ~180MB at 8k trades with features; bumped so
                                 # single-tier runs keep features for Q3 EDA.
                                 # Full chains=4 runs (33k trades) still strip.
        needs_strip = len(trades) > STRIP_THRESHOLD
        if needs_strip:
            def _strip_feat(bar):
                return {k: v for k, v in bar.items() if k != 'features'}
            slim = []
            for t in trades:
                t2 = dict(t)
                if 'path' in t2 and t2['path']:
                    t2['path'] = [_strip_feat(b) for b in t2['path']]
                if 'approach' in t2 and t2['approach']:
                    t2['approach'] = [_strip_feat(b) for b in t2['approach']]
                slim.append(t2)
            payload = slim
            print(f'  Large run ({len(trades):,} trades) — stripped path features to fit RAM')
        else:
            payload = trades

        # Pickle name: full-engine runs go to `iso_is.pkl` (canonical baseline
        # consumed by downstream analysis tools). Single-tier or restricted
        # runs get a suffix so they don't clobber the full-engine pickle.
        # Suffix = slugged tier list (e.g. iso_is_KILL_SHOT.pkl,
        # iso_is_KILL_SHOT+KILL_SHOT_INVERSE.pkl).
        pkl_path = os.path.join(OUTPUT_DIR, 'trades', pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        flat = [{k: v for k, v in t.items() if not isinstance(v, (list, dict, np.ndarray))}
                for t in trades]
        pd.DataFrame(flat).to_csv(os.path.join(OUTPUT_DIR, 'trades', csv_name), index=False)
        print(f'  Wrote: training_iso/output/trades/{pkl_name}')
    print(f'  Done in {_time.perf_counter()-t0:.0f}s')

    # Phase 2: Regret (opt-in via --regret flag; skipped by default)
    if run_regret_phase:
        print(f'\n{"="*40}')
        print(f'PHASE 2: Bounded regret (10/30 min, peak-validity gated)')
        print(f'{"="*40}')
        t0 = _time.perf_counter()
        run_regret(pkl_name=pkl_name)
        print(f'  Done in {_time.perf_counter()-t0:.0f}s')
    else:
        print(f'\n(Phase 2 regret skipped — default; pass --regret to enable)')

    # Summary
    print(f'\n{"="*40}')
    print(f'SUMMARY')
    print(f'{"="*40}')
    elapsed = _time.perf_counter() - pipeline_start
    print(f'ISO V2 PIPELINE COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)')
    print(f'  Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


# Need numpy for flat export
import numpy as np

if __name__ == '__main__':
    main()
