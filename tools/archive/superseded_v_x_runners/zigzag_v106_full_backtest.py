"""
zigzag_v106_full_backtest.py -- Full-window Python sim of ZigzagRunner v1.0.6-RC
with genetic-optimization defaults. Tests if the +$74/day Playback result
holds across the full 12-month ATLAS dataset.

Logic mirrors v1.0.6-RC source:
  - Zigzag pivots at R-point retracement
  - Direction policy: counter-trend default (HIGH→Short, LOW→Long)
  - v1.0.4 always-exit-before-entry
  - EOD force-close + entry cutoff
  - Rule 1: hard SL at MaxUnrealizedLossPoints
  - Rule 2: MFE-cut at MfeCutBarsAfterEntry if MFE ≤ MfeCutThresholdUsd
  - Rule 4: trail at TrailActivatePoints / TrailGivebackPct
  - Commission: $1.90/trade RT
  - Slippage: 0.25 pts/fill (NT8 sim engine equivalent)

CAVEAT: prior parity tests showed Python sim produces ~2× trade count vs NT8.
Numbers here are directional indication, not authoritative.

Usage:
    python tools/zigzag_v106_full_backtest.py
    python tools/zigzag_v106_full_backtest.py --atlas DATA/ATLAS
    python tools/zigzag_v106_full_backtest.py --atlas DATA/ATLAS_NT8
"""
from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
import pandas as pd
import numpy as np

DOLLAR_PER_POINT = 2.0  # MNQ
COMMISSION_RT = 1.90    # $/round-trip per contract


def load_1m_bars_yearly(atlas_root: str, tf: str = "1m") -> pd.DataFrame:
    """Load all 1m parquet files from ATLAS_root/{tf}/ into a single sorted DF."""
    parts = []
    folder = Path(atlas_root) / tf
    if not folder.exists():
        return pd.DataFrame()
    for f in sorted(folder.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df.empty:
            continue
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    out["dt_utc"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out


class V106Sim:
    """Stateful 1m-bar processor mirroring ZigzagRunner v1.0.6-RC.
    Note: 1m-resolution only — does not simulate intra-1m-bar SL fires.
    Real NT8 SL fires at 1s cadence; this sim approximates with 1m close.
    """
    def __init__(self,
                 r_points: float = 45.0,
                 contracts: int = 1,
                 eod_h_utc: int = 20, eod_m_utc: int = 55,
                 entry_cut_h_utc: int = 20, entry_cut_m_utc: int = 30,
                 max_loss_pts: float = 90.0,
                 mfe_cut_bars: int = 17,
                 mfe_cut_threshold_usd: float = 2.0,
                 trail_activate_pts: float = 21.0,
                 trail_giveback_pct: float = 0.05,
                 commission_rt_usd: float = 1.90,
                 slippage_pts: float = 0.25,
                 ):
        self.r = r_points
        self.contracts = contracts
        self.eod_mins = eod_h_utc * 60 + eod_m_utc
        self.cut_mins = entry_cut_h_utc * 60 + entry_cut_m_utc
        self.max_loss_pts = max_loss_pts
        self.mfe_cut_bars = mfe_cut_bars
        self.mfe_cut_threshold_usd = mfe_cut_threshold_usd
        self.trail_activate_pts = trail_activate_pts
        self.trail_giveback_pct = trail_giveback_pct
        self.commission = commission_rt_usd
        self.slippage_pts = slippage_pts

        # Zigzag state
        self.direction = 0
        self.extreme_price = float("nan")

        # Position state
        self.pos_dir = 0
        self.pos_entry_px = 0.0
        self.pos_entry_dt = None
        self.pos_entry_name = ""

        # Per-trade state (resets on entry)
        self.trade_bars_held = 0
        self.trade_mfe_usd = 0.0
        self.trail_armed = False

        # Output
        self.trades = []
        self.next_id = 1

    def _open(self, side, entry_px, entry_dt, entry_name):
        # Apply slippage: long fill = bar_open + slippage; short fill = bar_open - slippage
        slipped_px = entry_px + (side * self.slippage_pts)
        self.pos_dir = side
        self.pos_entry_px = slipped_px
        self.pos_entry_dt = entry_dt
        self.pos_entry_name = entry_name
        self.trade_bars_held = 0
        self.trade_mfe_usd = 0.0
        self.trail_armed = False

    def _close(self, exit_px, exit_dt, exit_name):
        if self.pos_dir == 0:
            return
        # Apply slippage on exit (worse for our direction)
        slipped_px = exit_px - (self.pos_dir * self.slippage_pts)
        side_s = "Long" if self.pos_dir > 0 else "Short"
        pnl_pts = self.pos_dir * (slipped_px - self.pos_entry_px)
        pnl_usd_gross = pnl_pts * DOLLAR_PER_POINT * self.contracts
        pnl_usd_net = pnl_usd_gross - self.commission

        self.trades.append({
            "trade_id":   self.next_id,
            "side":       side_s,
            "entry_dt":   self.pos_entry_dt,
            "exit_dt":    exit_dt,
            "entry_px":   self.pos_entry_px,
            "exit_px":    slipped_px,
            "exit_name":  exit_name,
            "pnl_usd":    pnl_usd_net,
            "pnl_usd_gross": pnl_usd_gross,
            "bars":       self.trade_bars_held,
            "mfe_usd":    self.trade_mfe_usd,
        })
        self.next_id += 1
        self.pos_dir = 0
        self.trade_bars_held = 0
        self.trade_mfe_usd = 0.0
        self.trail_armed = False

    def step(self, bar_close, bar_close_dt, next_open, next_open_dt):
        mins_of_day = bar_close_dt.hour * 60 + bar_close_dt.minute

        # EOD force-close (uses bar close price + slippage)
        if mins_of_day >= self.eod_mins:
            if self.pos_dir != 0:
                ename = "EodExitLong" if self.pos_dir > 0 else "EodExitShort"
                self._close(next_open, next_open_dt, ename)
            return

        # Update per-trade state if position open
        if self.pos_dir != 0:
            self.trade_bars_held += 1
            unrealized_usd = self.pos_dir * (bar_close - self.pos_entry_px) * DOLLAR_PER_POINT * self.contracts
            unrealized_pts = self.pos_dir * (bar_close - self.pos_entry_px)
            if unrealized_usd > self.trade_mfe_usd:
                self.trade_mfe_usd = unrealized_usd

            # Rule 4: Trail
            if self.trail_activate_pts > 0:
                activate_usd = self.trail_activate_pts * DOLLAR_PER_POINT
                if not self.trail_armed and self.trade_mfe_usd >= activate_usd:
                    self.trail_armed = True
                if self.trail_armed:
                    trail_threshold = self.trade_mfe_usd * (1.0 - self.trail_giveback_pct)
                    if unrealized_usd <= trail_threshold:
                        ename = "TrailExitLong" if self.pos_dir > 0 else "TrailExitShort"
                        self._close(next_open, next_open_dt, ename)
                        # don't return — pivot logic might still fire on this bar (entry only)

            # Rule 1: Hard SL
            if self.pos_dir != 0 and self.max_loss_pts > 0 and unrealized_pts <= -self.max_loss_pts:
                ename = "HardStopLong" if self.pos_dir > 0 else "HardStopShort"
                self._close(next_open, next_open_dt, ename)

            # Rule 2: MFE-cut at bar N
            if self.pos_dir != 0 and self.mfe_cut_bars > 0 and self.trade_bars_held == self.mfe_cut_bars:
                if self.trade_mfe_usd <= self.mfe_cut_threshold_usd:
                    ename = "MfeCutLong" if self.pos_dir > 0 else "MfeCutShort"
                    self._close(next_open, next_open_dt, ename)

        # Init extreme on first bar
        if pd.isna(self.extreme_price):
            self.extreme_price = bar_close
            return

        # Zigzag state machine
        pivot_confirmed = False
        new_pivot_dir = 0

        if self.direction == 0:
            if bar_close - self.extreme_price >= self.r:
                pivot_confirmed = True; new_pivot_dir = -1
                self.direction = +1; self.extreme_price = bar_close
            elif self.extreme_price - bar_close >= self.r:
                pivot_confirmed = True; new_pivot_dir = +1
                self.direction = -1; self.extreme_price = bar_close
        elif self.direction == +1:
            if bar_close > self.extreme_price:
                self.extreme_price = bar_close
            elif self.extreme_price - bar_close >= self.r:
                pivot_confirmed = True; new_pivot_dir = +1
                self.direction = -1; self.extreme_price = bar_close
        else:  # direction == -1
            if bar_close < self.extreme_price:
                self.extreme_price = bar_close
            elif bar_close - self.extreme_price >= self.r:
                pivot_confirmed = True; new_pivot_dir = -1
                self.direction = +1; self.extreme_price = bar_close

        if not pivot_confirmed:
            return

        # Past entry cutoff: skip entry (no exit, no entry)
        if mins_of_day >= self.cut_mins:
            return

        # Counter-trend: HIGH pivot -> Short, LOW pivot -> Long
        action_side = -1 if new_pivot_dir == +1 else +1
        pivot_label = "HighPivot" if new_pivot_dir == +1 else "LowPivot"

        # ALWAYS exit existing position first (v1.0.4 fix)
        if self.pos_dir != 0:
            ename = "PivotExitLong" if self.pos_dir > 0 else "PivotExitShort"
            self._close(next_open, next_open_dt, ename)

        # New entry
        ename = ("LongAt" if action_side == +1 else "ShortAt") + pivot_label
        self._open(action_side, next_open, next_open_dt, ename)


def run_full_backtest(bars: pd.DataFrame, sim: V106Sim) -> pd.DataFrame:
    n = len(bars)
    closes = bars["close"].to_numpy()
    opens = bars["open"].to_numpy()
    dts = bars["dt_utc"].to_list()

    for i in range(n - 1):
        bar_close = float(closes[i])
        bar_close_dt = dts[i].to_pydatetime() + timedelta(seconds=60)
        next_open = float(opens[i + 1])
        next_open_dt = dts[i + 1].to_pydatetime()
        sim.step(bar_close, bar_close_dt, next_open, next_open_dt)

    return pd.DataFrame(sim.trades)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="DATA/ATLAS",
                    help="ATLAS root (default: DATA/ATLAS — 12 months Databento)")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD lower bound")
    ap.add_argument("--end",   default=None, help="YYYY-MM-DD upper bound")
    ap.add_argument("--out-csv", default="reports/findings/2026-04-28_v106_full_backtest.csv")
    ap.add_argument("--out-md",  default="reports/findings/2026-04-28_v106_full_backtest.md")
    # v1.0.6-RC genetic defaults
    ap.add_argument("--r", type=float, default=45.0)
    ap.add_argument("--max-loss", type=float, default=90.0)
    ap.add_argument("--mfe-cut-bars", type=int, default=17)
    ap.add_argument("--mfe-cut-usd", type=float, default=2.0)
    ap.add_argument("--trail-activate", type=float, default=21.0)
    ap.add_argument("--trail-giveback", type=float, default=0.05)
    ap.add_argument("--slippage", type=float, default=0.25)
    ap.add_argument("--commission", type=float, default=1.90)
    args = ap.parse_args()

    print("=" * 80)
    print("v1.0.6-RC FULL-WINDOW PYTHON BACKTEST (genetic-optimized defaults)")
    print("=" * 80)
    print(f"Atlas: {args.atlas}")
    print(f"Params: R={args.r} SL={args.max_loss}pt MFE-cut bar {args.mfe_cut_bars}/${args.mfe_cut_usd:.2f}")
    print(f"        Trail activate {args.trail_activate}pt giveback {args.trail_giveback*100:.0f}%")
    print(f"        Slippage {args.slippage}pt  Commission ${args.commission}/RT")

    bars = load_1m_bars_yearly(args.atlas)
    if bars.empty:
        print(f"FATAL: no 1m bars at {args.atlas}/1m/")
        return
    if args.start:
        bars = bars[bars["dt_utc"] >= pd.Timestamp(args.start, tz="UTC")].reset_index(drop=True)
    if args.end:
        bars = bars[bars["dt_utc"] <= pd.Timestamp(args.end, tz="UTC")].reset_index(drop=True)
    print(f"\n1m bars: {len(bars):,}  ({bars['dt_utc'].iloc[0]} -> {bars['dt_utc'].iloc[-1]})")

    sim = V106Sim(
        r_points=args.r,
        max_loss_pts=args.max_loss,
        mfe_cut_bars=args.mfe_cut_bars,
        mfe_cut_threshold_usd=args.mfe_cut_usd,
        trail_activate_pts=args.trail_activate,
        trail_giveback_pct=args.trail_giveback,
        slippage_pts=args.slippage,
        commission_rt_usd=args.commission,
    )

    print("\nRunning sim...")
    trades = run_full_backtest(bars, sim)
    print(f"  {len(trades)} trades simulated")

    if len(trades) == 0:
        print("No trades generated. Check params.")
        return

    # Aggregate
    total_pnl_net   = trades["pnl_usd"].sum()
    total_pnl_gross = trades["pnl_usd_gross"].sum()
    n = len(trades)
    n_win = (trades["pnl_usd"] > 0).sum()
    days = trades["entry_dt"].dt.date.nunique() if "entry_dt" in trades.columns else 1
    days = max(days, 1)

    # Per-day PnL
    trades["day"] = trades["entry_dt"].dt.date
    by_day = trades.groupby("day")["pnl_usd"].agg(["sum", "count"]).rename(columns={"sum":"pnl","count":"n"})

    # Buy-and-hold passive long
    bars["day"] = bars["dt_utc"].dt.date
    daily_ohlc = bars.groupby("day").agg(open=("open","first"), close=("close","last"))
    bh_total_pts = daily_ohlc["close"].iloc[-1] - daily_ohlc["open"].iloc[0]
    bh_total_usd = bh_total_pts * DOLLAR_PER_POINT
    bh_per_day = bh_total_usd / len(daily_ohlc)

    # Exit reason breakdown
    exit_counts = trades["exit_name"].value_counts().to_dict()

    # Summary
    print("\n==== AGGREGATE ====")
    print(f"  Window:           {trades['entry_dt'].min().date()} -> {trades['exit_dt'].max().date()}  ({days} trading days)")
    print(f"  Trades:           {n}")
    print(f"  Trades/day:       {n/days:.1f}")
    print(f"  WR:               {100*n_win/n:.1f}%")
    print(f"  Total Net PnL:    ${total_pnl_net:+,.2f}  (after ${COMMISSION_RT}/RT commission)")
    print(f"  Total Gross PnL:  ${total_pnl_gross:+,.2f}")
    print(f"  $/day (net):      ${total_pnl_net/days:+,.2f}")
    print(f"  Best day:         ${by_day['pnl'].max():+,.2f}")
    print(f"  Worst day:        ${by_day['pnl'].min():+,.2f}")
    print(f"  Median day:       ${by_day['pnl'].median():+,.2f}")
    pos_days = (by_day["pnl"] > 0).sum()
    print(f"  Positive days:    {pos_days}/{len(by_day)} ({100*pos_days/len(by_day):.1f}%)")

    print(f"\n==== vs PASSIVE LONG ====")
    print(f"  Buy-and-hold MNQ: ${bh_total_usd:+,.2f} = ${bh_per_day:+,.2f}/day")
    print(f"  Strategy - BH:    ${total_pnl_net - bh_total_usd:+,.2f} (${(total_pnl_net/days) - bh_per_day:+,.2f}/day)")

    print(f"\n==== EXIT REASONS ====")
    for k, v in sorted(exit_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k:<25} {v:>6} ({100*v/n:.1f}%)")

    # Per-month breakdown
    print(f"\n==== PER-MONTH ====")
    trades["ym"] = pd.to_datetime(trades["entry_dt"]).dt.to_period("M")
    by_month = trades.groupby("ym").agg(pnl=("pnl_usd","sum"), n=("pnl_usd","count")).reset_index()
    print(f"  {'Month':>10} {'Trades':>8} {'Net PnL':>12} {'$/day est':>12}")
    for _, r in by_month.iterrows():
        # Approx 21 trading days per month
        print(f"  {str(r['ym']):>10} {int(r['n']):>8} ${r['pnl']:>+10,.2f} ${r['pnl']/21:>+10,.2f}")

    # Save trade ledger
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    trades.to_csv(args.out_csv, index=False)
    print(f"\nTrade ledger: {args.out_csv}")

    # Markdown report
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(f"# v1.0.6-RC full-window Python backtest\n\n")
            f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
            f.write(f"Atlas: `{args.atlas}` ({len(bars):,} 1m bars, {days} days)\n")
            f.write(f"Params: R={args.r} SL={args.max_loss}pt MFE-cut={args.mfe_cut_bars}/${args.mfe_cut_usd} ")
            f.write(f"Trail={args.trail_activate}pt/{args.trail_giveback*100:.0f}%\n\n")
            f.write(f"**CAVEAT**: Python sim has known 2× trade-count bias vs NT8 SA. ")
            f.write(f"Numbers are directional, not authoritative.\n\n")
            f.write("## Aggregate\n\n")
            f.write(f"| Metric | Value |\n|---|---:|\n")
            f.write(f"| Trades | {n:,} |\n")
            f.write(f"| Trades/day | {n/days:.1f} |\n")
            f.write(f"| Win rate | {100*n_win/n:.1f}% |\n")
            f.write(f"| Total Net PnL | ${total_pnl_net:+,.2f} |\n")
            f.write(f"| Total Gross PnL | ${total_pnl_gross:+,.2f} |\n")
            f.write(f"| $/day (net) | ${total_pnl_net/days:+,.2f} |\n")
            f.write(f"| Best day | ${by_day['pnl'].max():+,.2f} |\n")
            f.write(f"| Worst day | ${by_day['pnl'].min():+,.2f} |\n")
            f.write(f"| Positive days | {pos_days}/{len(by_day)} ({100*pos_days/len(by_day):.1f}%) |\n")
            f.write(f"| Buy-and-hold MNQ | ${bh_total_usd:+,.2f} (${bh_per_day:+,.2f}/day) |\n")
            f.write(f"| Strategy - BH | ${total_pnl_net - bh_total_usd:+,.2f} |\n\n")
            f.write("## Exit reasons\n\n")
            f.write("| Reason | Count | % |\n|---|---:|---:|\n")
            for k, v in sorted(exit_counts.items(), key=lambda x: -x[1]):
                f.write(f"| {k} | {v} | {100*v/n:.1f}% |\n")
            f.write("\n## Per-month\n\n")
            f.write("| Month | Trades | Net PnL | $/day est |\n|---|---:|---:|---:|\n")
            for _, r in by_month.iterrows():
                f.write(f"| {r['ym']} | {int(r['n'])} | ${r['pnl']:+,.2f} | ${r['pnl']/21:+,.2f} |\n")
        print(f"Report: {args.out_md}")


if __name__ == "__main__":
    main()
