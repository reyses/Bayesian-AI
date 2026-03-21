#!/usr/bin/env python
"""
Pattern Selection Research -- score competition analysis.

Answers: "When multiple candidates pass gates on the same bar, does the
score competition pick the candidate with the best oracle outcome?"

Data source:
  - signal_log shards (gate='traded' and gate='score_loser' on same ts)

Usage:
  python tools/research_pattern_selection.py               # IS (default)
  python tools/research_pattern_selection.py --oos          # OOS
  python tools/research_pattern_selection.py --save         # save report
"""
import argparse
import os
import sys
import glob

import numpy as np
import pandas as pd


def _load_signal_log(mode='is'):
    """Load and concatenate all signal_log shards."""
    base = os.path.join('reports', mode, 'shards')
    pattern = os.path.join(base, 'signal_log_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        # Try single file
        single = os.path.join('reports', mode, 'signal_log.csv')
        if os.path.exists(single):
            files = [single]
    if not files:
        print(f"[ERROR] No signal_log found in {base}")
        sys.exit(1)
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
    sl = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(sl):,} signal_log records from {len(files)} shards ({mode})")
    return sl


def _load_trade_log(mode='is'):
    """Load oracle_trade_log for actual trade outcomes (from checkpoints/)."""
    name = 'oos_trade_log.csv' if mode in ('oos', 'oos2') else 'oracle_trade_log.csv'
    path = os.path.join('checkpoints', name)
    if not os.path.exists(path):
        return None
    tl = pd.read_csv(path)
    print(f"Loaded {len(tl):,} trade records from {path}")
    return tl


def section_competition_overview(sl):
    """How often does score competition happen?"""
    traded = sl[sl['gate'] == 'traded']
    losers = sl[sl['gate'] == 'score_loser']

    n_traded = len(traded)
    n_losers = len(losers)

    if n_losers == 0:
        print("\n[!] No score_loser records in signal_log.")
        print("    Run `python training/trainer.py --forward-pass` to generate them.")
        print("    (Score_loser logging was just added.)")
        return None

    # Group by timestamp to find bars with competition
    loser_ts = set(losers['ts'].unique())
    traded_with_comp = traded[traded['ts'].isin(loser_ts)]
    n_competed = len(traded_with_comp)
    n_solo = n_traded - n_competed

    print(f"\n{'='*70}")
    print(f"  SECTION 1: Score Competition Overview")
    print(f"{'='*70}")
    print(f"  Total traded entries:      {n_traded:>6,}")
    print(f"  Trades with competition:   {n_competed:>6,}  ({n_competed/n_traded*100:.1f}%)")
    print(f"  Trades solo (no contest):  {n_solo:>6,}  ({n_solo/n_traded*100:.1f}%)")
    print(f"  Total score_loser records: {n_losers:>6,}")

    # Avg competitors per contested bar
    comp_counts = losers.groupby('ts').size()
    print(f"  Avg losers per contest:    {comp_counts.mean():.1f}")
    print(f"  Max losers on one bar:     {comp_counts.max()}")

    return traded_with_comp


def section_oracle_comparison(sl, tl):
    """Compare actual trade outcome when we picked the "wrong" oracle candidate.

    The question isn't "did we pick the best oracle?" — it's "did it matter?"
    A wrong oracle pick that still exits profitably is fine.
    A wrong oracle pick that results in a loss is the real problem.
    """
    traded = sl[sl['gate'] == 'traded'].copy()
    losers = sl[sl['gate'] == 'score_loser'].copy()

    if losers.empty:
        return

    # Join traded + losers by timestamp
    loser_ts = set(losers['ts'].unique())
    competed = traded[traded['ts'].isin(loser_ts)].copy()

    if competed.empty:
        print("\n  No contested bars to analyze.")
        return

    print(f"\n{'='*70}")
    print(f"  SECTION 2: Score Competition Outcome Analysis")
    print(f"{'='*70}")

    # Build per-contest comparison: winner oracle vs best loser oracle
    results = []
    for ts in competed['ts'].unique():
        winner_rows = traded[traded['ts'] == ts]
        loser_rows = losers[losers['ts'] == ts]
        if winner_rows.empty:
            continue
        w = winner_rows.iloc[0]
        w_oracle = float(w.get('oracle_pnl', 0))
        w_label = str(w.get('oracle_label', 'NOISE'))
        w_depth = int(w.get('depth', 6))
        w_actual = float(w.get('trade_pnl', 0))
        w_result = str(w.get('trade_result', ''))
        w_exit = str(w.get('exit_reason', ''))

        best_loser_idx = loser_rows['oracle_pnl'].idxmax()
        bl = loser_rows.loc[best_loser_idx]
        bl_oracle = float(bl['oracle_pnl'])
        bl_label = str(bl['oracle_label'])
        bl_depth = int(bl['depth'])

        picked_best = w_oracle >= bl_oracle
        results.append({
            'ts': ts,
            'w_oracle': w_oracle, 'w_label': w_label, 'w_depth': w_depth,
            'w_actual': w_actual, 'w_result': w_result, 'w_exit': w_exit,
            'bl_oracle': bl_oracle, 'bl_label': bl_label, 'bl_depth': bl_depth,
            'picked_best': picked_best,
            'oracle_gap': w_oracle - bl_oracle,
        })

    rdf = pd.DataFrame(results)
    n = len(rdf)
    n_best = int(rdf['picked_best'].sum())
    n_worse = n - n_best

    print(f"\n  Contested bars: {n:,}")
    print(f"  Picked best oracle candidate:  {n_best:>5,}  ({n_best/n*100:.1f}%)")
    print(f"  Picked worse oracle candidate: {n_worse:>5,}  ({n_worse/n*100:.1f}%)")

    # The key question: when we picked the worse oracle, what happened?
    if n_worse > 0:
        wrong = rdf[~rdf['picked_best']].copy()

        # Cross with actual trade outcome
        wrong_won = wrong[wrong['w_actual'] > 0]
        wrong_be = wrong[wrong['w_actual'] == 0]
        wrong_lost = wrong[wrong['w_actual'] < 0]

        print(f"\n  -- When we picked the worse oracle candidate ({n_worse} trades) --")
        print(f"  Exit engine saved it (actual PnL > 0): {len(wrong_won):>5,}  ({len(wrong_won)/n_worse*100:.1f}%)")
        print(f"  Breakeven exit (actual PnL = 0):       {len(wrong_be):>5,}  ({len(wrong_be)/n_worse*100:.1f}%)")
        print(f"  Actual loss:                           {len(wrong_lost):>5,}  ({len(wrong_lost)/n_worse*100:.1f}%)")

        if not wrong_won.empty:
            print(f"    Saved trades avg PnL:   ${wrong_won['w_actual'].mean():>7.2f}")
        if not wrong_lost.empty:
            print(f"    Lost trades avg PnL:    ${wrong_lost['w_actual'].mean():>7.2f}")
            print(f"    Lost trades total:      ${wrong_lost['w_actual'].sum():>10,.2f}")
            print(f"    Avg oracle gap on losses: ${abs(wrong_lost['oracle_gap'].mean()):>7.2f}")

        # The actual cost: only losses matter
        actual_cost = wrong_lost['w_actual'].sum() if not wrong_lost.empty else 0
        oracle_upside = abs(wrong['oracle_gap'].sum())
        print(f"\n  Summary:")
        print(f"    Oracle PnL left on table (hypothetical): ${oracle_upside:>10,.2f}")
        print(f"    Actual cost (realized losses only):      ${abs(actual_cost):>10,.2f}")
        if actual_cost == 0:
            print(f"    --> Score competition is safe: exit engine covers all wrong picks")
        else:
            print(f"    --> {len(wrong_lost)} trades where wrong pick + bad exit = real damage")

        # Depth pattern on actual losses
        if not wrong_lost.empty:
            print(f"\n  Depth pattern on actual losses:")
            wrong_lost = wrong_lost.copy()
            wrong_lost['swap'] = wrong_lost.apply(
                lambda r: f"d{r['w_depth']}->d{r['bl_depth']}", axis=1)
            for swap, grp in wrong_lost.groupby('swap'):
                print(f"    {swap}: {len(grp)} trades, total ${grp['w_actual'].sum():>8,.2f}")

    # When we picked the best oracle — confirm it's working
    if n_best > 0:
        correct = rdf[rdf['picked_best']]
        print(f"\n  -- When we picked the best oracle candidate ({n_best} trades) --")
        correct_won = correct[correct['w_actual'] > 0]
        correct_lost = correct[correct['w_actual'] < 0]
        print(f"  Won:  {len(correct_won):>5}  avg ${correct_won['w_actual'].mean():>7.2f}" if not correct_won.empty else "")
        print(f"  Lost: {len(correct_lost):>5}  avg ${correct_lost['w_actual'].mean():>7.2f}" if not correct_lost.empty else "")

    return rdf


def section_score_formula(sl):
    """Analyze if score components (depth, dist, tier) correlate with oracle PnL."""
    traded = sl[sl['gate'] == 'traded'].copy()
    losers = sl[sl['gate'] == 'score_loser'].copy()

    if losers.empty or 'competition_score' not in losers.columns:
        return

    print(f"\n{'='*70}")
    print(f"  SECTION 3: Score Formula Analysis")
    print(f"{'='*70}")

    # All gate passers (traded + losers with scores)
    all_passers = pd.concat([traded, losers], ignore_index=True)
    passers_with_score = all_passers[all_passers['competition_score'] != 0].copy()

    if passers_with_score.empty:
        print("  No score data available yet.")
        return

    # Correlation: score vs oracle_pnl
    corr = passers_with_score[['competition_score', 'oracle_pnl']].corr()
    r = corr.loc['competition_score', 'oracle_pnl']
    print(f"\n  Score vs Oracle PnL correlation: r = {r:.3f}")
    if abs(r) < 0.1:
        print(f"  --> Weak: score formula barely predicts oracle outcome")
    elif r < -0.1:
        print(f"  --> Good: lower score (better rank) correlates with higher oracle PnL")
    else:
        print(f"  --> Inverted! Lower score correlates with LOWER oracle PnL")

    # By depth: avg oracle PnL
    print(f"\n  Oracle PnL by depth (all gate passers):")
    for depth, grp in passers_with_score.groupby('depth'):
        print(f"    Depth {depth}: n={len(grp):>4}  "
              f"avg oracle=${grp['oracle_pnl'].mean():>7.2f}  "
              f"avg score={grp['competition_score'].mean():.2f}")

    # By template tier
    if 'tier' in passers_with_score.columns:
        print(f"\n  Oracle PnL by tier:")
        for tier, grp in passers_with_score.groupby('tier'):
            if len(grp) >= 5:
                print(f"    Tier {tier}: n={len(grp):>4}  "
                      f"avg oracle=${grp['oracle_pnl'].mean():>7.2f}  "
                      f"avg score={grp['competition_score'].mean():.2f}")

    # By distance bucket
    if 'gate1_dist' in passers_with_score.columns:
        dist_col = passers_with_score['gate1_dist']
        passers_with_score['dist_bucket'] = pd.cut(dist_col, bins=[0, 1, 2, 3, 5, 10],
                                                    labels=['0-1', '1-2', '2-3', '3-5', '5-10'])
        print(f"\n  Oracle PnL by cluster distance:")
        for bucket, grp in passers_with_score.groupby('dist_bucket', observed=True):
            if len(grp) >= 3:
                print(f"    dist {bucket}: n={len(grp):>4}  "
                      f"avg oracle=${grp['oracle_pnl'].mean():>7.2f}")


def section_actual_trade_pnl(sl, tl):
    """Compare actual trade PnL for contested vs solo entries."""
    if tl is None:
        return

    traded = sl[sl['gate'] == 'traded'].copy()
    losers = sl[sl['gate'] == 'score_loser']

    if losers.empty:
        return

    print(f"\n{'='*70}")
    print(f"  SECTION 4: Actual Trade P&L -- Contested vs Solo")
    print(f"{'='*70}")

    loser_ts = set(losers['ts'].unique())
    if 'n_competitors' in traded.columns:
        contested = traded[traded['n_competitors'] > 0]
        solo = traded[traded['n_competitors'] == 0]
    else:
        contested = traded[traded['ts'].isin(loser_ts)]
        solo = traded[~traded['ts'].isin(loser_ts)]

    # Match with trade_log by entry_time
    if 'entry_time' in tl.columns:
        contested_ts = set(contested['ts'])
        solo_ts = set(solo['ts'])
        tl_contested = tl[tl['entry_time'].isin(contested_ts)]
        tl_solo = tl[tl['entry_time'].isin(solo_ts)]

        if not tl_contested.empty:
            print(f"\n  Contested trades: {len(tl_contested):,}")
            print(f"    Avg PnL:  ${tl_contested['actual_pnl'].mean():>7.2f}")
            print(f"    Total:    ${tl_contested['actual_pnl'].sum():>10,.2f}")
            print(f"    Win rate: {(tl_contested['actual_pnl'] > 0).mean()*100:.1f}%")

        if not tl_solo.empty:
            print(f"\n  Solo trades (no competition): {len(tl_solo):,}")
            print(f"    Avg PnL:  ${tl_solo['actual_pnl'].mean():>7.2f}")
            print(f"    Total:    ${tl_solo['actual_pnl'].sum():>10,.2f}")
            print(f"    Win rate: {(tl_solo['actual_pnl'] > 0).mean()*100:.1f}%")


def section_worst_picks(sl):
    """Show the worst score competition picks (largest negative delta)."""
    traded = sl[sl['gate'] == 'traded']
    losers = sl[sl['gate'] == 'score_loser']

    if losers.empty:
        return

    print(f"\n{'='*70}")
    print(f"  SECTION 5: Worst Score Competition Picks (top 15)")
    print(f"{'='*70}")

    results = []
    for ts in losers['ts'].unique():
        winner_rows = traded[traded['ts'] == ts]
        loser_rows = losers[losers['ts'] == ts]
        if winner_rows.empty:
            continue
        w = winner_rows.iloc[0]
        w_pnl = float(w.get('oracle_pnl', 0))

        best_loser_idx = loser_rows['oracle_pnl'].idxmax()
        best_loser = loser_rows.loc[best_loser_idx]
        bl_pnl = float(best_loser['oracle_pnl'])

        if bl_pnl > w_pnl:
            results.append({
                'ts': ts,
                'w_depth': int(w['depth']),
                'w_label': w['oracle_label'],
                'w_pnl': w_pnl,
                'l_depth': int(best_loser['depth']),
                'l_label': best_loser['oracle_label'],
                'l_pnl': bl_pnl,
                'gap': bl_pnl - w_pnl,
            })

    if not results:
        print("  No wrong picks found!")
        return

    worst = sorted(results, key=lambda x: -x['gap'])[:15]
    print(f"\n  {'TS':>12}  {'Win':>5} {'W_Label':>8} {'W_PnL':>8}  "
          f"{'Lose':>5} {'L_Label':>8} {'L_PnL':>8}  {'Gap':>8}")
    print(f"  {'-'*12}  {'-'*5} {'-'*8} {'-'*8}  {'-'*5} {'-'*8} {'-'*8}  {'-'*8}")
    for r in worst:
        print(f"  {r['ts']:>12}  d{r['w_depth']:>4} {r['w_label']:>8} ${r['w_pnl']:>7.2f}  "
              f"d{r['l_depth']:>4} {r['l_label']:>8} ${r['l_pnl']:>7.2f}  ${r['gap']:>7.2f}")


def main():
    parser = argparse.ArgumentParser(description='Pattern Selection Research')
    parser.add_argument('--oos', action='store_true', help='Use OOS data (shorthand for --mode oos)')
    parser.add_argument('--mode', choices=['is', 'oos', 'oos2'], default=None,
                        help='Data mode: is, oos, or oos2')
    parser.add_argument('--save', action='store_true', help='Save report to file')
    args = parser.parse_args()

    mode = args.mode or ('oos' if args.oos else 'is')
    print(f"\n  Pattern Selection Research ({mode.upper()})")
    print(f"  {'='*50}")

    sl = _load_signal_log(mode)
    tl = _load_trade_log(mode)

    section_competition_overview(sl)
    section_oracle_comparison(sl, tl)
    section_score_formula(sl)
    section_actual_trade_pnl(sl, tl)
    section_worst_picks(sl)

    if args.save:
        out_dir = os.path.join('reports', 'research')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'pattern_selection_{mode}.txt')
        print(f"\n  Report would be saved to: {out_path}")
        # Re-run with stdout redirect for save
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        sl2 = _load_signal_log(mode)
        tl2 = _load_trade_log(mode)
        section_competition_overview(sl2)
        section_oracle_comparison(sl2, tl2)
        section_score_formula(sl2)
        section_actual_trade_pnl(sl2, tl2)
        section_worst_picks(sl2)
        sys.stdout = old_stdout
        with open(out_path, 'w') as f:
            f.write(buf.getvalue())
        print(f"  Saved to {out_path}")


if __name__ == '__main__':
    main()
