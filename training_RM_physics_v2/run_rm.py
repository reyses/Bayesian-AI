"""
RM Physics pipeline — v2 (185D features).

Same engine logic as v1 but consumes v2 feature parquets from
DATA/ATLAS/FEATURES_5s_v2/ via core_v2.features.load_features(),
and reads the 1m band z-score from named column L3_1m_z_se_15
(not v1's hardcoded feature-vector index 12).

Flags:
  --regret          Run Phase 2 regret analysis
  --chains N        Enable chain positions up to N (default 1)
  --oos             Run against 2026 days only. Writes rm_oos_v2.pkl.
  --with-oos        Run IS first, then OOS. Writes BOTH pickles.
  --log [PATH]      Tee stdout to a file.

Pickle output:
  IS forward pass  →  training_RM_physics_v2/output/trades/rm_is_v2.pkl
  OOS forward pass →  training_RM_physics_v2/output/trades/rm_oos_v2.pkl

Usage:
    python training_RM_physics_v2/run_rm.py              # IS only
    python training_RM_physics_v2/run_rm.py --with-oos   # IS + OOS
    python training_RM_physics_v2/run_rm.py --oos        # OOS only
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

# v2 features live under FEATURES_5s_v2 alongside (not replacing) v1 FEATURES_5s.
# FEATURES_DIR_SEQ is used here as a day-discovery crutch only — we glob the
# L0 subdir to enumerate available days. Actual per-day feature loading lives
# in training_RM_physics_v2.ticker_1s.OneSecondTicker via load_features().
FEATURES_DIR_SEQ = 'DATA/ATLAS/FEATURES_5s_v2/L0'
ATLAS_1M = 'DATA/ATLAS/1m'
ATLAS_5S = 'DATA/ATLAS/5s'
OUTPUT_DIR = 'training_RM_physics_v2/output'


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


def run_iso_forward(target='is', only_tiers=None, max_chains=4, cascade=False):
    """Forward pass. Two modes:

    - **isolated** (default): each tier gets its OWN engine, all fire in
      parallel on the same bar stream. No priority-based arbitration —
      every tier sees every bar, enters whenever its conditions match.
      Useful for per-tier EDA; NOT what live does.

    - **cascade** (`cascade=True`): ONE engine with TIER_PRIORITY first-
      match-wins. Simulates live/blended forward pass: when multiple
      tiers fire on the same bar, the highest-priority tier takes it.
      Chain cap per position uses MAX_CHAINS_PER_TIER based on the
      current open tier; `max_chains` is the CLI global ceiling.

    `only_tiers` restricts to a subset (isolated: those tiers run;
    cascade: classifier only considers those tiers). `max_chains`
    controls how many concurrent positions each engine can hold.

    Output trades carry `entry_tier` and `chain_idx` so per-tier metrics
    roll up cleanly.
    """
    from training_RM_physics_v2.rm_physics_engine import IsoEngine, TIER_PRIORITY
    from training_RM_physics_v2.ticker_1s import OneSecondTicker
    from tqdm import tqdm
    from collections import Counter

    feat_files = _resolve_days(target, FEATURES_DIR_SEQ)
    if not feat_files:
        print(f'No feature files for "{target}"')
        return [], []

    active_tiers = only_tiers if only_tiers else TIER_PRIORITY
    mode_str = 'CASCADE (forward-pass parity)' if cascade else 'ISOLATED (parallel per-tier)'
    print(f'ISO FORWARD [{mode_str}] — {len(feat_files)} day(s), '
          f'tiers: {active_tiers}, max_chains={max_chains}')
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

        if cascade:
            # Single engine, priority-ordered classifier. Restrict TIER_PRIORITY
            # to only_tiers at runtime (monkey-patch approach: the engine's
            # _classify reads module-level TIER_PRIORITY, so we keep it
            # untouched and the engine will try every tier; only_tiers just
            # affects downstream filtering/summary in this branch).
            engines = {'CASCADE_ALL': IsoEngine(only_tier=None,
                                                max_chains=max_chains,
                                                honor_per_tier_caps=False)}
        else:
            engines = {t: IsoEngine(only_tier=t, max_chains=max_chains)
                       for t in active_tiers}
        for eng in engines.values():
            eng.set_sec_closes(sec_df)
        # 1s ticker: per-1s state with 1s high/low + 5s features
        # (nearest-past) + most-recently-completed 1m bar (no lookahead).
        try:
            ft = OneSecondTicker(day_name)
        except FileNotFoundError as e:
            print(f'  [SKIP {day_name}] missing 1s data: {e}')
            continue
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
            # Per-day WR: (profit/|loss| - 1) * 100. 0 = break-even, >0 net profit.
            'wr': (((sum(t['pnl'] for t in day_trades if t['pnl'] > 0)
                    / max(abs(sum(t['pnl'] for t in day_trades if t['pnl'] < 0)), 1e-9))
                   - 1) * 100),
        })

    _print_summary(all_results)

    if all_trades:
        from collections import Counter as _Counter
        # Main per-tier KPI table.
        # WR = (profit / |loss| - 1) * 100.
        #   0%    = break-even (profit == loss)
        #   +100% = profit is 2x loss (profit-factor 2.0)
        #  -100%  = pure loss, no profit
        # This is profit-factor-minus-1 expressed as a percent — trader-native,
        # unambiguous on whether a tier makes money.
        print()
        print(f'{"Tier":<17} {"N":>6} {"WR":>7} {"Total":>10} {"$/tr":>7} '
              f'{"Mode":>20}')
        print('-' * 75)
        for tier in active_tiers:
            sub = [t for t in all_trades if t.get('entry_tier') == tier]
            if not sub:
                print(f'{tier:<17} {0:>6}  (no trades)')
                continue
            count = len(sub)
            wins_pnl = sum(t['pnl'] for t in sub if t['pnl'] > 0)
            loss_pnl_abs = abs(sum(t['pnl'] for t in sub if t['pnl'] < 0))
            if loss_pnl_abs > 0:
                wr = (wins_pnl / loss_pnl_abs - 1) * 100
                wr_str = f'{wr:>+5.0f}%'
            else:
                wr_str = '  inf%' if wins_pnl > 0 else '    0%'
            total = wins_pnl - loss_pnl_abs
            per = total / count
            b_counts = _Counter(_bucket_of(t.get('pnl', 0.0)) for t in sub)
            mode_name, mode_n = b_counts.most_common(1)[0]
            mode_pct = mode_n / count * 100
            mode_str = f'{mode_name}({mode_pct:.0f}%)'
            print(f'{tier:<17} {count:>6,} {wr_str:>7} ${total:>+9,.0f} '
                  f'${per:>+6.2f} {mode_str:>20}')

        # Per-tier 10-bucket distribution (details)
        _print_tier_mode_buckets(all_trades, active_tiers)

        # MAE-cut lift summary (separate section, lift-only — no full table).
        _print_mae_cut_lift(all_trades, active_tiers)

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


# MAE-stop threshold sweep (ticks / dollars). Negative = drawdown threshold.
_MAE_SWEEP = [-20, -25, -30, -35, -40, -50, -60, -75, -100]


def _running_min(trade):
    """Worst PnL ever seen during the trade (MAE). Returns 0 if no path."""
    path = trade.get('path', [])
    if not path:
        return 0.0
    return min(p.get('pnl', 0.0) for p in path)


def _mae_cut_delta(trades, threshold):
    """Estimate PnL delta from applying an MAE stop at `threshold`.

    For each trade whose MAE touches threshold:
      - We would exit at threshold (approximation: actual cross price ≈ T).
      - Delta = threshold − current_pnl (positive = save, negative = cost).
    Returns (net_delta, n_cut, n_winners_cut, n_losers_saved).
    """
    net = 0.0
    n_cut = 0
    n_winner_cut = 0
    n_loser_saved = 0
    for t in trades:
        mae = _running_min(t)
        if mae > threshold:
            continue   # never touched threshold — no change
        current = t.get('pnl', 0.0)
        delta = threshold - current
        net += delta
        n_cut += 1
        if current > 0:
            n_winner_cut += 1
        elif current < threshold:
            n_loser_saved += 1
    return net, n_cut, n_winner_cut, n_loser_saved


def _print_mae_cut_lift(trades, active_tiers):
    """MAE-stop potential lift (post-physics risk-management estimate).

    Adaptive-mix lift only — per-tier threshold is auto-picked to maximize
    net savings. No full table: show just the total $/day lift and the
    global benchmark alongside.
    """
    per_tier_best = {}
    for tier in active_tiers:
        sub = [t for t in trades if t.get('entry_tier') == tier]
        if not sub:
            continue
        best = None
        for T in _MAE_SWEEP:
            net, _, _, _ = _mae_cut_delta(sub, T)
            if best is None or net > best['net']:
                best = {'T': T, 'net': net}
        if best:
            per_tier_best[tier] = best

    global_best = None
    for T in _MAE_SWEEP:
        net, _, _, _ = _mae_cut_delta(trades, T)
        if global_best is None or net > global_best['net']:
            global_best = {'T': T, 'net': net}

    n_days = max(1, len(set(t.get('day', '') for t in trades)))
    mix_save = sum(b['net'] for b in per_tier_best.values())

    print(f'{"="*100}')
    print(f'MAE-cut lift (risk-management layer, post-physics; NOT applied '
          f'to engine total)')
    print(f'{"="*100}')
    if global_best:
        print(f'  Global best T=${global_best["T"]}   save ${global_best["net"]:>+12,.0f}   '
              f'->  ${global_best["net"]/n_days:+.2f}/day')
    print(f'  Adaptive mix (per-tier T)    save ${mix_save:>+12,.0f}   '
          f'->  ${mix_save/n_days:+.2f}/day')
    print()


def run_regret(pkl_name: str = 'rm_is.pkl'):
    """Run bounded regret + produce corrected trades.

    Uses training_RM_physics/regret.py which caps the counterfactual window to
    LOOKBACK_MIN=10 / LOOKAHEAD_MIN=30, gates EXTENDED options on the
    peak-validity check, and produces corrected trades that exit at the
    SPECIFIC peak bar each gated best_action points at (not the overall
    argmax).

    `pkl_name` defaults to the canonical full-engine baseline. Single-tier
    runs pass their tier-suffixed pickle name so Phase 2 analyzes the same
    trades Phase 1 just produced.
    """
    from training_RM_physics_v2.regret import compute_all_regrets, correct_trades

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
    from training_RM_physics_v2.rm_physics_engine import TIER_PRIORITY
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

    # --cascade: switch to priority-ordered first-match-wins cascade.
    # Simulates live/blended forward pass. Without this flag, runs in
    # isolated per-tier mode (current default).
    cascade = '--cascade' in args

    # --oos: run against 2026 days instead of 2025 (IS default).
    # --with-oos: run IS first, then OOS right after (both phases saved to
    # separate pickles: iso_is.pkl + iso_oos.pkl).
    with_oos = '--with-oos' in args
    target = 'oos' if ('--oos' in args and not with_oos) else 'is'

    # --log [PATH]: tee stdout to a file. Without PATH, auto-generates
    # reports/findings/rm_run_<target>_<YYYYMMDD_HHMMSS>.txt.
    log_path = None
    if '--log' in args:
        idx = args.index('--log')
        if idx + 1 < len(args) and not args[idx + 1].startswith('--'):
            log_path = args[idx + 1]
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            t_tag = 'is_oos' if with_oos else target
            log_path = os.path.join('reports', 'findings',
                                    f'rm_run_{t_tag}_{ts}.txt')
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, s):
                for st in self.streams:
                    try: st.write(s); st.flush()
                    except Exception: pass
            def flush(self):
                for st in self.streams:
                    try: st.flush()
                    except Exception: pass
        _log_fh = open(log_path, 'w', encoding='utf-8', errors='replace')
        sys.stdout = _Tee(sys.__stdout__, _log_fh)
        print(f'[LOG] Writing output to {log_path}')

    n_tiers = len(only_tiers) if only_tiers else len(TIER_PRIORITY)
    print(f'{"="*60}')
    if cascade:
        print(f'ISO PIPELINE V2 — CASCADE forward-pass (live parity)')
        print(f'  Mode: CASCADE (1 engine, TIER_PRIORITY first-match-wins)')
    else:
        print(f'ISOLATED PIPELINE V2 — {n_tiers} per-tier engines, no CNN')
        print(f'  Isolation: ON (1 engine per tier, no cross-tier interference)')
    if only_tiers:
        print(f'  Restricted to tiers: {only_tiers}')
    else:
        print(f'  Tiers active: all {n_tiers}')
    print(f'  Max chains: {max_chains}'
          + ('' if max_chains > 1 else '  [chaining OFF]'))
    print(f'  Regret phase: '
          + ('ENABLED (10/30-min window)' if run_regret_phase else 'SKIPPED [default]'))
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*60}')

    pipeline_start = _time.perf_counter()

    # Phase 1: Collect trades — single target or both (IS + OOS).
    def _phase1_one_target(tgt):
        print(f'\n{"="*40}')
        print(f'PHASE 1: forward pass [{tgt.upper()}]')
        print(f'{"="*40}')
        t0_local = _time.perf_counter()
        results_, trades_ = run_iso_forward(tgt, only_tiers=only_tiers,
                                            max_chains=max_chains, cascade=cascade)
        prefix = 'rm_oos' if tgt == 'oos' else 'rm_is'
        if only_tiers:
            pkl_name_ = f'{prefix}_{"+".join(only_tiers)}.pkl'
            csv_name_ = f'{prefix}_{"+".join(only_tiers)}.csv'
        else:
            pkl_name_ = f'{prefix}.pkl'
            csv_name_ = f'{prefix}.csv'
        if trades_:
            os.makedirs(os.path.join(OUTPUT_DIR, 'trades'), exist_ok=True)
            STRIP_THRESHOLD = 12000
            if len(trades_) > STRIP_THRESHOLD:
                def _strip_feat(bar):
                    return {k: v for k, v in bar.items() if k != 'features'}
                slim = []
                for t in trades_:
                    t2 = dict(t)
                    if 'path' in t2 and t2['path']:
                        t2['path'] = [_strip_feat(b) for b in t2['path']]
                    if 'approach' in t2 and t2['approach']:
                        t2['approach'] = [_strip_feat(b) for b in t2['approach']]
                    slim.append(t2)
                payload = slim
                print(f'  Large run ({len(trades_):,} trades) — stripped path features')
            else:
                payload = trades_
            pkl_path = os.path.join(OUTPUT_DIR, 'trades', pkl_name_)
            with open(pkl_path, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            flat = [{k: v for k, v in t.items() if not isinstance(v, (list, dict, np.ndarray))}
                    for t in trades_]
            pd.DataFrame(flat).to_csv(os.path.join(OUTPUT_DIR, 'trades', csv_name_), index=False)
            print(f'  Wrote: {pkl_path}')
        print(f'  Done in {_time.perf_counter()-t0_local:.0f}s')
        return pkl_name_, trades_

    pkl_name, trades = _phase1_one_target(target)
    if with_oos and target != 'oos':
        pkl_name_oos, _ = _phase1_one_target('oos')

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
