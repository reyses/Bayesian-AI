"""
Compare OOS Runs — Reusable analysis tool

Usage:
    python tools/compare_oos_runs.py
    python tools/compare_oos_runs.py --prev reports/is/oracle_trade_log_prev.csv --curr reports/is/oracle_trade_log.csv

Compares exit quality, direction accuracy, trail behavior, hold times,
and MFE capture between two forward-pass runs.
"""
import argparse
import pandas as pd
import numpy as np
import os

def load(path):
    df = pd.read_csv(path)
    if 'actual_pnl' in df.columns and 'pnl' not in df.columns:
        df['pnl'] = df['actual_pnl']
    # Compute bars held
    if 'entry_time' in df.columns and 'exit_time' in df.columns:
        df['_bars'] = ((df['exit_time'] - df['entry_time']) / 15).clip(lower=1)
    elif 'duration' in df.columns:
        df['_bars'] = (df['duration'] / 15).clip(lower=1)
    else:
        df['_bars'] = np.nan
    # MFE in dollars (MNQ point_value=2.0)
    if 'oracle_mfe' in df.columns:
        df['_mfe_dollar'] = df['oracle_mfe'] * 50  # tick_value * ticks
    return df


def section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def analyze_single(label, df):
    print(f"\n{'_' * 40}")
    print(f"  {label}: {len(df)} trades, WR: {(df.pnl > 0).mean():.1%}, PnL: ${df.pnl.sum():,.0f}")
    print(f"{'_' * 40}")


def exit_breakdown(label, df):
    section(f"{label} — EXIT REASON BREAKDOWN")
    for reason, g in df.groupby('exit_reason'):
        bars = f", bars={g._bars.mean():.1f}" if '_bars' in g.columns and not g._bars.isna().all() else ""
        print(f"  {reason:20s}: n={len(g):>5}, WR={(g.pnl > 0).mean():5.1%}, "
              f"avg=${g.pnl.mean():>7.1f}, total=${g.pnl.sum():>10,.0f}{bars}")


def direction_analysis(label, df):
    if 'oracle_label' not in df.columns:
        return
    section(f"{label} — DIRECTION ACCURACY")
    correct = df[df.oracle_label > 0]
    wrong = df[df.oracle_label < 0]
    noise = df[df.oracle_label == 0]
    print(f"  Correct direction: {len(correct)/len(df):.1%} ({len(correct)}), PnL: ${correct.pnl.sum():,.0f} (${correct.pnl.mean():.1f}/trade)")
    print(f"  Wrong direction:   {len(wrong)/len(df):.1%} ({len(wrong)}), PnL: ${wrong.pnl.sum():,.0f} (${wrong.pnl.mean():.1f}/trade)")
    print(f"  Noise:             {len(noise)/len(df):.1%} ({len(noise)}), PnL: ${noise.pnl.sum():,.0f}")

    # Trail stops: direction split
    ts = df[df.exit_reason == 'trail_stop']
    if len(ts) > 0:
        ts_c = ts[ts.oracle_label > 0]
        ts_w = ts[ts.oracle_label < 0]
        print(f"\n  Trail stops ({len(ts)} total):")
        if len(ts_c) > 0:
            print(f"    Correct dir: {len(ts_c)} ({len(ts_c)/len(ts):.0%}), avg ${ts_c.pnl.mean():.1f}")
        if len(ts_w) > 0:
            print(f"    Wrong dir:   {len(ts_w)} ({len(ts_w)/len(ts):.0%}), avg ${ts_w.pnl.mean():.1f}")


def trail_stop_analysis(label, df):
    ts = df[df.exit_reason == 'trail_stop']
    if len(ts) == 0:
        return
    section(f"{label} — TRAIL STOP DEEP DIVE ({len(ts)} trades)")
    print(f"  WR: {(ts.pnl > 0).mean():.1%}, avg PnL: ${ts.pnl.mean():.1f}, total: ${ts.pnl.sum():,.0f}")
    if 'entry_trail_ticks' in ts.columns:
        t = ts.entry_trail_ticks
        print(f"  Trail ticks: min={t.min():.0f}, p25={t.quantile(0.25):.0f}, "
              f"median={t.median():.0f}, p75={t.quantile(0.75):.0f}, max={t.max():.0f}")
    if 'entry_trail_act' in ts.columns:
        a = ts.entry_trail_act
        print(f"  Trail activation: min={a.min():.0f}, median={a.median():.0f}, max={a.max():.0f}")
    if 'entry_depth' in ts.columns:
        print(f"  By depth:")
        for d, g in ts.groupby('entry_depth'):
            trail_info = ""
            if 'entry_trail_ticks' in g.columns:
                trail_info = f", trail={g.entry_trail_ticks.median():.0f}"
            print(f"    depth {d}: n={len(g):>4}, WR={(g.pnl > 0).mean():5.1%}, avg=${g.pnl.mean():>7.1f}{trail_info}")


def depth_breakdown(label, df):
    if 'entry_depth' not in df.columns:
        return
    section(f"{label} — PER-DEPTH SUMMARY")
    for d, g in df.groupby('entry_depth'):
        if len(g) < 3:
            continue
        bars_info = f", bars={g._bars.mean():.1f}" if not g._bars.isna().all() else ""
        print(f"  depth {d}: n={len(g):>4}, WR={(g.pnl > 0).mean():5.1%}, "
              f"avg=${g.pnl.mean():>6.1f}, total=${g.pnl.sum():>8,.0f}{bars_info}")


def mfe_capture(label, df):
    if '_mfe_dollar' not in df.columns:
        return
    section(f"{label} — ORACLE MFE CAPTURE")
    winners = df[df.pnl > 0]
    if len(winners) == 0:
        print("  No winners")
        return
    cap = (winners.pnl / winners._mfe_dollar.replace(0, np.nan)).dropna()
    print(f"  Winners (n={len(winners)}): avg MFE=${winners._mfe_dollar.mean():.0f}, "
          f"avg PnL=${winners.pnl.mean():.1f}, capture={cap.mean():.1%}")
    if 'entry_depth' in winners.columns:
        print(f"  By depth:")
        for d, g in winners.groupby('entry_depth'):
            if len(g) < 3:
                continue
            gc = (g.pnl / g._mfe_dollar.replace(0, np.nan)).dropna()
            print(f"    depth {d}: n={len(g):>4}, MFE=${g._mfe_dollar.mean():>7.0f}, "
                  f"actual=${g.pnl.mean():>7.1f}, capture={gc.mean():>5.1%}")


def entry_quality(label, df):
    section(f"{label} — ENTRY QUALITY")
    if 'entry_z_score' in df.columns:
        z = df.entry_z_score.abs()
        print(f"  |z| at entry: mean={z.mean():.2f}, median={z.median():.2f}, p75={z.quantile(0.75):.2f}")
    if 'entry_coherence' in df.columns:
        print(f"  Coherence: mean={df.entry_coherence.mean():.3f}, median={df.entry_coherence.median():.3f}")
    if 'entry_hurst' in df.columns:
        print(f"  Hurst: mean={df.entry_hurst.mean():.3f}, median={df.entry_hurst.median():.3f}")
    if 'entry_lagrange_zone' in df.columns:
        print(f"  Lagrange zones: {df.entry_lagrange_zone.value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description="Compare two OOS forward-pass runs")
    parser.add_argument('--prev', default='reports/is/oracle_trade_log_prev.csv')
    parser.add_argument('--curr', default='reports/is/oracle_trade_log.csv')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("Loading...")
    prev = load(args.prev)
    curr = load(args.curr)

    section("SUMMARY COMPARISON")
    print(f"  {'Metric':<25s} {'PREV':>15s} {'CURRENT':>15s} {'Delta':>15s}")
    print(f"  {'─' * 70}")
    n_p, n_c = len(prev), len(curr)
    wr_p, wr_c = (prev.pnl > 0).mean(), (curr.pnl > 0).mean()
    pnl_p, pnl_c = prev.pnl.sum(), curr.pnl.sum()
    bars_p = prev._bars.mean() if not prev._bars.isna().all() else 0
    bars_c = curr._bars.mean() if not curr._bars.isna().all() else 0
    print(f"  {'Trades':<25s} {n_p:>15,d} {n_c:>15,d} {n_c - n_p:>+15,d}")
    print(f"  {'Win Rate':<25s} {wr_p:>14.1%} {wr_c:>14.1%} {wr_c - wr_p:>+14.1%}")
    print(f"  {'Total PnL':<25s} {'$' + f'{pnl_p:,.0f}':>14s} {'$' + f'{pnl_c:,.0f}':>14s} {'$' + f'{pnl_c - pnl_p:+,.0f}':>14s}")
    print(f"  {'Avg Hold (bars)':<25s} {bars_p:>15.1f} {bars_c:>15.1f} {bars_c - bars_p:>+15.1f}")

    for label, df in [("PREV", prev), ("CURRENT", curr)]:
        exit_breakdown(label, df)
        direction_analysis(label, df)
        trail_stop_analysis(label, df)
        depth_breakdown(label, df)
        mfe_capture(label, df)
        entry_quality(label, df)


if __name__ == '__main__':
    main()
