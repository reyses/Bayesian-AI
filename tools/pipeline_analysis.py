"""
Pipeline Analysis Module
========================
Systematic analysis of forward pass results (IS and OOS).
Answers: Are the components working? Where is the signal? Where is the leak?

Usage:
    python tools/pipeline_analysis.py
    python tools/pipeline_analysis.py --is reports/is --oos reports/oos
    python tools/pipeline_analysis.py --analysis A       # run specific analysis
    python tools/pipeline_analysis.py --analysis A,A2,B  # run multiple
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'pipeline')
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _load(report_dir):
    """Load trade log + fn_oracle from a report directory."""
    trades = pd.read_csv(os.path.join(report_dir, 'oracle_trade_log.csv'))
    fn_path = os.path.join(report_dir, 'fn_oracle_log.csv')
    fn = pd.read_csv(fn_path) if os.path.exists(fn_path) else pd.DataFrame()
    return trades, fn


def _wr(df):
    return (df.actual_pnl > 0).mean() if len(df) > 0 else 0.0


def _print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
#  Analysis A: High-level effectiveness
# ---------------------------------------------------------------------------

def analysis_a(is_df, is_fn, oos_df, oos_fn):
    """High-level: direction, WR, PnL, MFE/MAE, filter test."""
    _print_header("ANALYSIS A: HIGH-LEVEL EFFECTIVENESS")

    for label, df, fn in [('IS', is_df, is_fn), ('OOS', oos_df, oos_fn)]:
        n = len(df)
        if n == 0:
            print(f"\n  {label}: no trades")
            continue

        wins = df[df.actual_pnl > 0]
        losses = df[df.actual_pnl <= 0]
        right = df[df.oracle_mfe > df.oracle_mae]
        wrong = df[df.oracle_mae > df.oracle_mfe]

        print(f"\n  --- {label} ---")
        print(f"  Trades: {n}, WR: {_wr(df):.1%}, PnL: ${df.actual_pnl.sum():,.2f}, "
              f"Avg: ${df.actual_pnl.mean():.2f}/trade")
        print(f"  Direction: {df.direction.value_counts().to_dict()}")
        print(f"  Oracle labels: {df.oracle_label_name.value_counts().to_dict()}")
        print(f"  MFE > MAE (right dir): {len(right)} ({len(right)/n:.1%})")
        print(f"  MAE > MFE (wrong dir): {len(wrong)} ({len(wrong)/n:.1%})")

        # MFE/MAE
        print(f"  MFE mean={df.oracle_mfe.mean():.1f}, MAE mean={df.oracle_mae.mean():.1f}, "
              f"gap={( df.oracle_mfe - df.oracle_mae).mean():+.1f}")

        # Oracle potential
        pot = df.oracle_potential_pnl.sum()
        cap = df.actual_pnl.sum() / pot * 100 if pot > 0 else 0
        print(f"  Oracle potential: ${pot:,.2f}, Capture: {cap:.1f}%")

        # Missed signals
        if len(fn) > 0:
            print(f"  Missed signals: {len(fn)}, reasons: {fn.reason.value_counts().to_dict()}")
            if 'fn_potential_pnl' in fn.columns:
                print(f"  Missed PnL: ${fn.fn_potential_pnl.sum():,.2f}")

    # IS vs OOS comparison table
    _print_header("IS vs OOS COMPARISON")
    fmt = "  {:<30} {:>12} {:>12}"
    print(fmt.format('Metric', 'IS', 'OOS'))
    print(fmt.format('-'*30, '-'*12, '-'*12))

    for lbl, metric_fn in [
        ('Trades', lambda d: str(len(d))),
        ('Win Rate', lambda d: f"{_wr(d):.1%}"),
        ('Total PnL', lambda d: f"${d.actual_pnl.sum():,.0f}"),
        ('Avg PnL/trade', lambda d: f"${d.actual_pnl.mean():.2f}"),
        ('MFE > MAE (right dir)', lambda d: f"{(d.oracle_mfe > d.oracle_mae).mean():.1%}"),
    ]:
        is_val = metric_fn(is_df) if len(is_df) > 0 else "n/a"
        oos_val = metric_fn(oos_df) if len(oos_df) > 0 else "n/a"
        print(fmt.format(lbl, is_val, oos_val))


# ---------------------------------------------------------------------------
#  Analysis A2: Component scorecard
# ---------------------------------------------------------------------------

def analysis_a2(is_df, is_fn, oos_df, oos_fn):
    """Component-by-component: physics, template matching, direction."""
    _print_header("ANALYSIS A2: COMPONENT SCORECARD")

    for label, df in [('IS', is_df), ('OOS', oos_df)]:
        if len(df) == 0:
            continue

        wins = df[df.actual_pnl > 0]
        losses = df[df.actual_pnl <= 0]

        print(f"\n  --- {label} ---")

        # Component 1: Live physics (feature separation)
        print(f"\n  COMPONENT 1: LIVE PHYSICS")
        for col in ['entry_z_score', 'dmi_diff', 'entry_hurst', 'entry_adx',
                     'entry_coherence', 'entry_oscillation_coherence',
                     'entry_escape_prob', 'belief_conviction']:
            if col not in df.columns:
                continue
            w_m = wins[col].mean()
            l_m = losses[col].mean()
            diff = w_m - l_m
            # Simple t-test significance indicator
            sig = '***' if abs(diff) > 2 * df[col].std() / np.sqrt(len(df)) else ''
            print(f"    {col:>35}: win={w_m:>8.3f}  loss={l_m:>8.3f}  "
                  f"diff={diff:>+8.3f} {sig}")

        # Component 2: Template matching
        print(f"\n  COMPONENT 2: TEMPLATE MATCHING")
        bypass = (df.oracle_label_name == 'WORKER_BYPASS').sum()
        print(f"    WORKER_BYPASS: {bypass}/{len(df)} ({bypass/len(df):.1%})")
        unique_templates = df.template_id.nunique()
        print(f"    Unique templates matched: {unique_templates}")
        print(f"    template_id distribution: {df.template_id.value_counts().head(5).to_dict()}")

        # Component 3: Direction
        print(f"\n  COMPONENT 3: DIRECTION")
        print(f"    Direction split: {df.direction.value_counts().to_dict()}")
        right = df[df.oracle_mfe > df.oracle_mae]
        wrong = df[df.oracle_mae > df.oracle_mfe]
        print(f"    Right dir PnL: ${right.actual_pnl.sum():,.2f} "
              f"(avg ${right.actual_pnl.mean():.2f}, WR={_wr(right):.1%})")
        print(f"    Wrong dir PnL: ${wrong.actual_pnl.sum():,.2f} "
              f"(avg ${wrong.actual_pnl.mean():.2f}, WR={_wr(wrong):.1%})")

        # Conviction
        print(f"\n  CONVICTION CALIBRATION")
        print(f"    Winners: {wins.belief_conviction.mean():.3f}")
        print(f"    Losers:  {losses.belief_conviction.mean():.3f}")
        print(f"    Non-predictive" if abs(wins.belief_conviction.mean() -
              losses.belief_conviction.mean()) < 0.01 else "    Has edge")


# ---------------------------------------------------------------------------
#  Analysis B: Filter tests (z_score, dmi_diff, depth)
# ---------------------------------------------------------------------------

def analysis_b(is_df, is_fn, oos_df, oos_fn):
    """Test entry filters: z_score, dmi_diff, depth, combinations."""
    _print_header("ANALYSIS B: ENTRY FILTER TESTS")

    filters = [
        ("Baseline (all)",           lambda d: d),
        ("z < 0",                    lambda d: d[d.entry_z_score < 0]),
        ("z < -1",                   lambda d: d[d.entry_z_score < -1]),
        ("dmi_diff < -5",            lambda d: d[d.dmi_diff < -5]),
        ("dmi_diff < -10",           lambda d: d[d.dmi_diff < -10]),
        ("z < 0 AND dmi < -5",      lambda d: d[(d.entry_z_score < 0) & (d.dmi_diff < -5)]),
        ("z < -1 AND dmi < -5",     lambda d: d[(d.entry_z_score < -1) & (d.dmi_diff < -5)]),
        ("depth <= 3",              lambda d: d[d.entry_depth <= 3]),
        ("depth <= 2",              lambda d: d[d.entry_depth <= 2]),
        ("depth <= 3 AND z < 0",   lambda d: d[(d.entry_depth <= 3) & (d.entry_z_score < 0)]),
        ("hurst > 0.55",           lambda d: d[d.entry_hurst > 0.55] if 'entry_hurst' in d.columns else d.head(0)),
        ("hurst > 0.6",            lambda d: d[d.entry_hurst > 0.6] if 'entry_hurst' in d.columns else d.head(0)),
    ]

    fmt = "  {:<30} {:>6} {:>7} {:>12} {:>10}  {:>6} {:>7} {:>12} {:>10}"
    print(fmt.format('Filter', 'IS_n', 'IS_WR', 'IS_PnL', 'IS_avg',
                      'OOS_n', 'OOS_WR', 'OOS_PnL', 'OOS_avg'))
    print(fmt.format('-'*30, '-'*6, '-'*7, '-'*12, '-'*10,
                      '-'*6, '-'*7, '-'*12, '-'*10))

    for name, filt_fn in filters:
        is_f = filt_fn(is_df)
        oos_f = filt_fn(oos_df)

        is_n = len(is_f)
        is_wr = f"{_wr(is_f):.1%}" if is_n > 0 else "n/a"
        is_pnl = f"${is_f.actual_pnl.sum():,.0f}" if is_n > 0 else "n/a"
        is_avg = f"${is_f.actual_pnl.mean():.2f}" if is_n > 0 else "n/a"

        oos_n = len(oos_f)
        oos_wr = f"{_wr(oos_f):.1%}" if oos_n > 0 else "n/a"
        oos_pnl = f"${oos_f.actual_pnl.sum():,.0f}" if oos_n > 0 else "n/a"
        oos_avg = f"${oos_f.actual_pnl.mean():.2f}" if oos_n > 0 else "n/a"

        print(fmt.format(name, is_n, is_wr, is_pnl, is_avg,
                          oos_n, oos_wr, oos_pnl, oos_avg))


# ---------------------------------------------------------------------------
#  Analysis C: Exit analysis
# ---------------------------------------------------------------------------

def analysis_c(is_df, is_fn, oos_df, oos_fn):
    """Exit reason breakdown, stop loss impact, trail stop performance."""
    _print_header("ANALYSIS C: EXIT ANALYSIS")

    for label, df in [('IS', is_df), ('OOS', oos_df)]:
        if len(df) == 0:
            continue

        print(f"\n  --- {label} ---")
        print(f"  {'Exit Reason':>20}  {'n':>5}  {'WR':>6}  {'Avg PnL':>10}  {'Total PnL':>12}  {'% of PnL':>8}")
        print(f"  {'-'*20}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*8}")

        total_pnl = df.actual_pnl.sum()
        for reason in df.exit_reason.value_counts().index:
            sub = df[df.exit_reason == reason]
            n = len(sub)
            wr = _wr(sub)
            avg = sub.actual_pnl.mean()
            tot = sub.actual_pnl.sum()
            pct = tot / total_pnl * 100 if total_pnl != 0 else 0
            print(f"  {reason:>20}  {n:>5}  {wr:>5.1%}  ${avg:>9.2f}  ${tot:>11,.2f}  {pct:>7.1f}%")

        # Stop loss deep dive
        sl = df[df.exit_reason == 'stop_loss']
        if len(sl) > 0:
            print(f"\n  STOP LOSS DEEP DIVE ({len(sl)} trades):")
            print(f"    Total drain: ${sl.actual_pnl.sum():,.2f}")
            print(f"    Avg hold: {sl.hold_bars.mean():.0f} bars")
            print(f"    Oracle MFE of SL trades: {sl.oracle_mfe.mean():.1f} "
                  f"(these trades HAD {sl.oracle_mfe.mean():.0f} pts of favorable move available)")
            pct_had_move = (sl.oracle_mfe > 20).mean()
            print(f"    {pct_had_move:.1%} of SL trades had MFE > 20 pts (real move existed)")

        # Direction paradox
        right = df[df.oracle_mfe > df.oracle_mae]
        wrong = df[df.oracle_mae > df.oracle_mfe]
        if len(right) > 0 and len(wrong) > 0:
            print(f"\n  DIRECTION PARADOX:")
            print(f"    Right dir: avg=${right.actual_pnl.mean():.2f}, WR={_wr(right):.1%}")
            print(f"    Wrong dir: avg=${wrong.actual_pnl.mean():.2f}, WR={_wr(wrong):.1%}")
            if wrong.actual_pnl.mean() > right.actual_pnl.mean():
                print(f"    -> Wrong direction makes MORE money (exit system = mean reversion catcher)")


# ---------------------------------------------------------------------------
#  Analysis D: Depth analysis
# ---------------------------------------------------------------------------

def analysis_d(is_df, is_fn, oos_df, oos_fn):
    """PnL by entry depth (TF trigger level)."""
    _print_header("ANALYSIS D: DEPTH ANALYSIS")

    for label, df in [('IS', is_df), ('OOS', oos_df)]:
        if len(df) == 0:
            continue
        print(f"\n  --- {label} ---")
        print(f"  {'Depth':>7}  {'n':>5}  {'WR':>6}  {'Avg PnL':>10}  {'Total PnL':>12}")
        print(f"  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*12}")

        for depth in sorted(df.entry_depth.unique()):
            sub = df[df.entry_depth == depth]
            print(f"  {depth:>7.0f}  {len(sub):>5}  {_wr(sub):>5.1%}  "
                  f"${sub.actual_pnl.mean():>9.2f}  ${sub.actual_pnl.sum():>11,.2f}")

        # Cumulative from top
        print(f"\n  Cumulative (depth <= N):")
        for max_d in [1, 2, 3, 4, 5]:
            sub = df[df.entry_depth <= max_d]
            if len(sub) > 0:
                print(f"    depth <= {max_d}: {len(sub)} trades, WR={_wr(sub):.1%}, "
                      f"avg=${sub.actual_pnl.mean():.2f}, total=${sub.actual_pnl.sum():,.2f}")


# ---------------------------------------------------------------------------
#  Analysis E: Equity curve + time clustering
# ---------------------------------------------------------------------------

def analysis_e(is_df, is_fn, oos_df, oos_fn):
    """Equity curve and time-based clustering of wins/losses."""
    _print_header("ANALYSIS E: EQUITY CURVE & TIME PATTERNS")
    import matplotlib.pyplot as plt

    for label, df in [('IS', is_df), ('OOS', oos_df)]:
        if len(df) == 0:
            continue

        # Sort by entry time
        df_sorted = df.sort_values('entry_time').reset_index(drop=True)
        equity = df_sorted.actual_pnl.cumsum()

        print(f"\n  --- {label} ---")
        print(f"  Max equity: ${equity.max():,.2f}")
        print(f"  Min equity: ${equity.min():,.2f}")
        print(f"  Max drawdown: ${(equity - equity.cummax()).min():,.2f}")
        print(f"  Final equity: ${equity.iloc[-1]:,.2f}")

        # Streak analysis
        results = (df_sorted.actual_pnl > 0).astype(int)
        streaks = results.groupby((results != results.shift()).cumsum())
        win_streaks = [len(g) for _, g in streaks if g.iloc[0] == 1]
        loss_streaks = [len(g) for _, g in streaks if g.iloc[0] == 0]
        print(f"  Max win streak: {max(win_streaks) if win_streaks else 0}")
        print(f"  Max loss streak: {max(loss_streaks) if loss_streaks else 0}")

        # Plot equity curve
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(equity.values, linewidth=1.5, color='#2ca02c' if equity.iloc[-1] > 0 else '#d62728')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.fill_between(range(len(equity)), 0, equity.values,
                        where=equity.values >= 0, alpha=0.15, color='green')
        ax.fill_between(range(len(equity)), 0, equity.values,
                        where=equity.values < 0, alpha=0.15, color='red')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.set_title(f'{label} Equity Curve ({len(df)} trades)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f'equity_{label.lower()}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

ANALYSES = {
    'A':  ('High-level effectiveness',    analysis_a),
    'A2': ('Component scorecard',          analysis_a2),
    'B':  ('Entry filter tests',           analysis_b),
    'C':  ('Exit analysis',                analysis_c),
    'D':  ('Depth analysis',               analysis_d),
    'E':  ('Equity curve & time patterns', analysis_e),
}


def run_analysis(is_dir='reports/is', oos_dir='reports/oos', analyses=None):
    """Programmatic entry point (called from orchestrator auto-chain)."""
    is_df, is_fn = _load(is_dir)
    print(f"  Pipeline analysis: {len(is_df)} IS trades loaded")

    oos_df, oos_fn = pd.DataFrame(), pd.DataFrame()
    if os.path.exists(oos_dir):
        oos_df, oos_fn = _load(oos_dir)
        print(f"  Pipeline analysis: {len(oos_df)} OOS trades loaded")

    to_run = analyses or list(ANALYSES.keys())
    for aid in to_run:
        if aid in ANALYSES:
            name, fn = ANALYSES[aid]
            fn(is_df, is_fn, oos_df, oos_fn)

    print(f"  Pipeline analysis complete. Plots: {PLOTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description='Pipeline analysis module')
    parser.add_argument('--is', dest='is_dir', default='reports/is',
                        help='IS report directory')
    parser.add_argument('--oos', dest='oos_dir', default='reports/oos',
                        help='OOS report directory')
    parser.add_argument('--analysis', default=None,
                        help='Comma-separated analysis IDs (A,A2,B,C,D,E) or "all"')
    args = parser.parse_args()

    # Load data
    print(f"Loading IS from {args.is_dir}...")
    is_df, is_fn = _load(args.is_dir)
    print(f"  {len(is_df)} trades, {len(is_fn)} missed signals")

    oos_df, oos_fn = pd.DataFrame(), pd.DataFrame()
    if os.path.exists(args.oos_dir):
        print(f"Loading OOS from {args.oos_dir}...")
        oos_df, oos_fn = _load(args.oos_dir)
        print(f"  {len(oos_df)} trades, {len(oos_fn)} missed signals")

    # Run analyses
    if args.analysis is None or args.analysis.lower() == 'all':
        to_run = list(ANALYSES.keys())
    else:
        to_run = [a.strip().upper() for a in args.analysis.split(',')]

    for aid in to_run:
        if aid in ANALYSES:
            name, fn = ANALYSES[aid]
            fn(is_df, is_fn, oos_df, oos_fn)
        else:
            print(f"  Unknown analysis: {aid}")
            print(f"  Available: {', '.join(ANALYSES.keys())}")

    print(f"\n{'='*70}")
    print(f"  DONE. Plots: {PLOTS_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
