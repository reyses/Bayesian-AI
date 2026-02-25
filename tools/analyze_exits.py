"""
Exit Analysis Tool — Deep dive into a single run's exit quality

Usage:
    python tools/analyze_exits.py
    python tools/analyze_exits.py --file reports/is/oracle_trade_log.csv
    python tools/analyze_exits.py --file reports/is/oracle_trade_log_prev.csv

Analyzes exit reasons, direction accuracy, trail stop behavior,
CST structural breaks, hold times, and MFE capture.
"""
import argparse
import pandas as pd
import numpy as np
import os


def load(path):
    df = pd.read_csv(path)
    if 'actual_pnl' in df.columns and 'pnl' not in df.columns:
        df['pnl'] = df['actual_pnl']
    if 'entry_time' in df.columns and 'exit_time' in df.columns:
        df['_bars'] = ((df['exit_time'] - df['entry_time']) / 15).clip(lower=1)
    elif 'duration' in df.columns:
        df['_bars'] = (df['duration'] / 15).clip(lower=1)
    else:
        df['_bars'] = np.nan
    if 'oracle_mfe' in df.columns:
        df['_mfe_dollar'] = df['oracle_mfe'] * 50
    return df


def section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Deep exit analysis for a single OOS run")
    parser.add_argument('--file', default='reports/is/oracle_trade_log.csv')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = load(args.file)
    print(f"Loaded {len(df)} trades from {args.file}")

    # ── Summary ──────────────────────────────────────────────────────────────
    section("OVERALL SUMMARY")
    print(f"  Trades: {len(df)}")
    print(f"  Win Rate: {(df.pnl > 0).mean():.1%}")
    print(f"  Total PnL: ${df.pnl.sum():,.0f}")
    print(f"  Avg PnL/trade: ${df.pnl.mean():.2f}")
    if not df._bars.isna().all():
        print(f"  Avg hold: {df._bars.mean():.1f} bars ({df._bars.mean() * 15 / 60:.1f} min)")

    # ── Exit reasons ─────────────────────────────────────────────────────────
    section("EXIT REASON BREAKDOWN")
    for reason, g in df.groupby('exit_reason'):
        bars = f", bars={g._bars.mean():.1f}" if not g._bars.isna().all() else ""
        print(f"  {reason:20s}: n={len(g):>5} ({len(g)/len(df):5.1%}), "
              f"WR={(g.pnl > 0).mean():5.1%}, avg=${g.pnl.mean():>7.1f}, total=${g.pnl.sum():>10,.0f}{bars}")

    # ── Exit reason x Win/Loss ───────────────────────────────────────────────
    section("EXIT REASON x OUTCOME")
    print(f"  {'Reason':<20s} | {'Winners':>30s} | {'Losers':>30s}")
    print(f"  {'─' * 85}")
    for reason, g in df.groupby('exit_reason'):
        w = g[g.pnl > 0]
        l = g[g.pnl <= 0]
        w_str = f"n={len(w):>4}, avg=${w.pnl.mean():>6.1f}" if len(w) > 0 else "n=   0"
        l_str = f"n={len(l):>4}, avg=${l.pnl.mean():>6.1f}" if len(l) > 0 else "n=   0"
        print(f"  {reason:<20s} | {w_str:>30s} | {l_str:>30s}")

    # ── Direction accuracy ───────────────────────────────────────────────────
    if 'oracle_label' in df.columns:
        section("DIRECTION ACCURACY")
        correct = df[df.oracle_label > 0]
        wrong = df[df.oracle_label < 0]
        noise = df[df.oracle_label == 0]
        print(f"  Correct: {len(correct):>5} ({len(correct)/len(df):5.1%}), PnL: ${correct.pnl.sum():>10,.0f} (${correct.pnl.mean():.1f}/trade)")
        print(f"  Wrong:   {len(wrong):>5} ({len(wrong)/len(df):5.1%}), PnL: ${wrong.pnl.sum():>10,.0f} (${wrong.pnl.mean():.1f}/trade)")
        print(f"  Noise:   {len(noise):>5} ({len(noise)/len(df):5.1%}), PnL: ${noise.pnl.sum():>10,.0f}")

        # Wrong direction by exit reason
        if len(wrong) > 0:
            print(f"\n  Wrong-direction by exit:")
            for reason, g in wrong.groupby('exit_reason'):
                print(f"    {reason:20s}: {len(g):>4}, avg ${g.pnl.mean():>7.1f}")

    # ── Trail stop analysis ──────────────────────────────────────────────────
    ts = df[df.exit_reason == 'trail_stop']
    if len(ts) > 0:
        section(f"TRAIL STOP ANALYSIS ({len(ts)} trades)")
        print(f"  WR: {(ts.pnl > 0).mean():.1%}, avg: ${ts.pnl.mean():.1f}, total: ${ts.pnl.sum():,.0f}")
        for col, label in [('entry_trail_ticks', 'Trail ticks'), ('entry_trail_act', 'Trail activation'),
                           ('entry_sl_ticks', 'SL ticks'), ('entry_tp_ticks', 'TP ticks')]:
            if col in ts.columns and not ts[col].isna().all():
                v = ts[col].dropna()
                print(f"  {label}: min={v.min():.0f}, median={v.median():.0f}, max={v.max():.0f}")
        if 'entry_depth' in ts.columns:
            print(f"  By depth:")
            for d, g in ts.groupby('entry_depth'):
                print(f"    depth {d}: n={len(g):>4}, WR={(g.pnl > 0).mean():5.1%}, avg=${g.pnl.mean():>7.1f}")

    # ── Structural break analysis ────────────────────────────────────────────
    sb = df[df.exit_reason == 'structural_break']
    if len(sb) > 0:
        section(f"STRUCTURAL BREAK ANALYSIS ({len(sb)} trades)")
        w = sb[sb.pnl > 0]
        l = sb[sb.pnl <= 0]
        print(f"  Winners: n={len(w)}, avg ${w.pnl.mean():.1f}, avg bars {w._bars.mean():.1f}")
        print(f"  Losers:  n={len(l)}, avg ${l.pnl.mean():.1f}, avg bars {l._bars.mean():.1f}")
        if 'entry_depth' in sb.columns:
            print(f"  By depth:")
            for d, g in sb.groupby('entry_depth'):
                if len(g) < 2: continue
                print(f"    depth {d}: n={len(g):>4}, WR={(g.pnl > 0).mean():5.1%}, avg=${g.pnl.mean():>7.1f}, bars={g._bars.mean():.1f}")

    # ── Per-depth summary ────────────────────────────────────────────────────
    if 'entry_depth' in df.columns:
        section("PER-DEPTH SUMMARY")
        for d, g in df.groupby('entry_depth'):
            if len(g) < 3: continue
            bars = f", bars={g._bars.mean():.1f}" if not g._bars.isna().all() else ""
            print(f"  depth {d}: n={len(g):>4}, WR={(g.pnl > 0).mean():5.1%}, "
                  f"avg=${g.pnl.mean():>6.1f}, total=${g.pnl.sum():>8,.0f}{bars}")

    # ── Exit parameters ──────────────────────────────────────────────────────
    section("EXIT PARAMETER DISTRIBUTION BY DEPTH")
    for col, label in [('entry_tp_ticks', 'TP'), ('entry_sl_ticks', 'SL'),
                       ('entry_trail_ticks', 'Trail'), ('entry_trail_act', 'TrailAct')]:
        if col not in df.columns or df[col].isna().all():
            continue
        print(f"\n  {label} ticks:")
        if 'entry_depth' in df.columns:
            for d, g in df.groupby('entry_depth'):
                if len(g) < 3: continue
                v = g[col].dropna()
                if len(v) == 0: continue
                print(f"    depth {d}: min={v.min():.0f}, median={v.median():.0f}, p75={v.quantile(0.75):.0f}, max={v.max():.0f}")

    # ── MFE capture ──────────────────────────────────────────────────────────
    if '_mfe_dollar' in df.columns:
        section("ORACLE MFE CAPTURE")
        winners = df[df.pnl > 0]
        if len(winners) > 0:
            cap = (winners.pnl / winners._mfe_dollar.replace(0, np.nan)).dropna()
            print(f"  Winners (n={len(winners)}): avg MFE=${winners._mfe_dollar.mean():.0f}, "
                  f"actual=${winners.pnl.mean():.1f}, capture={cap.mean():.1%}")

    # ── Entry quality ────────────────────────────────────────────────────────
    section("ENTRY QUALITY")
    if 'entry_z_score' in df.columns:
        z = df.entry_z_score.abs()
        print(f"  |z| at entry: mean={z.mean():.2f}, median={z.median():.2f}")
    if 'entry_coherence' in df.columns:
        print(f"  Coherence: mean={df.entry_coherence.mean():.3f}")
    if 'entry_hurst' in df.columns:
        print(f"  Hurst: mean={df.entry_hurst.mean():.3f}")
    if 'entry_lagrange_zone' in df.columns:
        print(f"  Lagrange zones: {df.entry_lagrange_zone.value_counts().to_dict()}")


if __name__ == '__main__':
    main()
