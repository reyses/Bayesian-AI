"""
zigzag_v104_parity.py -- Trade-level parity between Python sim of
ZigzagRunner v1.0.4 and an NT8 Strategy Analyzer trades CSV.

Mirrors v1.0.4 logic exactly:
  - Calculate.OnBarClose on 1m primary
  - Zigzag pivot detection at R points retracement
  - Direction policy: counter-trend default (HIGH pivot -> Short, LOW -> Long)
  - v1.0.4 fix: ALWAYS exit current position before any entry decision
  - EOD force-close at (EodHourUtc, EodMinuteUtc) UTC
  - Entry cutoff at (EntryCutoffHourUtc, EntryCutoffMinuteUtc) UTC
  - Fill model: market orders submitted at bar T's close fill at bar T+1's open
  - Commission: $1.90 per round-trip (NT8 Brokerage Free template)

Inputs:
  --nt8-csv         examples/trades v1.0.4.csv     (ground truth)
  --atlas           DATA/ATLAS_NT8/1m              (parquet bar source)
  --r               30
  --eod-hour-utc    20
  --eod-minute-utc  55
  --entry-cutoff-hour-utc    20
  --entry-cutoff-minute-utc  30
  --commission      1.90    (round-trip USD per contract)

Outputs:
  Per-trade comparison: NT8 trade  vs  Python sim trade
  Summary metrics: total trades, total PnL, fill-time deltas
  CSV: reports/findings/2026-04-27_v104_parity.csv
  Markdown: reports/findings/2026-04-27_v104_parity.md

Usage:
    python tools/zigzag_v104_parity.py
    python tools/zigzag_v104_parity.py --r 30 --commission 1.90
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, time, timezone
from pathlib import Path
import pandas as pd

DOLLAR_PER_POINT = 2.0  # MNQ


# ─── NT8 trade CSV parsing ────────────────────────────────────────────────

def parse_money(s: str) -> float:
    s = s.replace("$", "").replace(",", "").strip()
    if not s:
        return 0.0
    if s.startswith("(") and s.endswith(")"):
        return -float(s[1:-1])
    return float(s)


def load_nt8_trades(path: str) -> pd.DataFrame:
    """Load NT8 Strategy Analyzer trades CSV into a normalized DataFrame."""
    rows = []
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                entry_dt = datetime.strptime(r["Entry time"].strip(), "%m/%d/%Y %I:%M:%S %p")
                exit_dt  = datetime.strptime(r["Exit time"].strip(), "%m/%d/%Y %I:%M:%S %p")
            except (ValueError, KeyError):
                continue
            rows.append({
                "trade_id":   int(r["Trade number"]),
                "side":       r["Market pos."].strip(),
                "qty":        int(r["Qty"]),
                "entry_dt":   entry_dt,
                "exit_dt":    exit_dt,
                "entry_px":   float(r["Entry price"]),
                "exit_px":    float(r["Exit price"]),
                "entry_name": r["Entry name"].strip(),
                "exit_name":  r["Exit name"].strip(),
                "pnl_usd":    parse_money(r["Profit"]),
                "commission": parse_money(r["Commission"]),
                "mae":        parse_money(r["MAE"]),
                "mfe":        parse_money(r["MFE"]),
                "bars":       int(r["Bars"]),
            })
    return pd.DataFrame(rows)


# ─── ATLAS 1m bar loading ─────────────────────────────────────────────────

def load_1m_bars(atlas_root: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load ATLAS_NT8 1m parquet files into a single chronological DataFrame.

    start_date/end_date: 'YYYY-MM-DD' inclusive bounds.
    """
    root = Path(atlas_root)
    files = sorted(root.glob("*.parquet"))
    parts = []
    for f in files:
        date_str = f.stem.replace("_", "-")  # 2026_03_20 -> 2026-03-20
        if not (start_date <= date_str <= end_date):
            continue
        df = pd.read_parquet(f)
        if df.empty:
            continue
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    # NT8 dump convention: timestamp = bar OPEN time (Databento style, set in
    # BayesianHistoryDumper.cs by subtracting bar period from Time[0]).
    # For our purposes: each row represents the bar starting at `timestamp`
    # and closing at `timestamp + 60`. We treat OHLC at bar close as the
    # decision-making price, and OPEN as the next-bar fill price.
    out["dt"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out


# ─── Python sim of v1.0.4 logic ───────────────────────────────────────────

class V104Sim:
    """Stateful 1m-bar processor mirroring ZigzagRunner v1.0.4."""

    def __init__(self,
                 r_points: float = 30.0,
                 contracts: int = 1,
                 eod_h_utc: int = 20, eod_m_utc: int = 55,
                 entry_cut_h_utc: int = 20, entry_cut_m_utc: int = 30,
                 on_high_pivot: str = "Short",   # counter-trend default
                 on_low_pivot:  str = "Long",
                 commission_rt_usd: float = 1.90,
                 ):
        self.r = r_points
        self.contracts = contracts
        self.eod_mins   = eod_h_utc * 60 + eod_m_utc
        self.cut_mins   = entry_cut_h_utc * 60 + entry_cut_m_utc
        self.on_high = on_high_pivot
        self.on_low  = on_low_pivot
        self.commission = commission_rt_usd

        # Zigzag state
        self.direction = 0
        self.extreme_price = float("nan")

        # Position state
        self.pos_dir = 0  # +1 long, -1 short, 0 flat
        self.pos_entry_px = 0.0
        self.pos_entry_dt = None
        self.pos_entry_name = ""

        # Output
        self.trades = []   # list of dicts
        self.next_trade_id = 1

    def _close_position(self, exit_px: float, exit_dt: datetime, exit_name: str):
        if self.pos_dir == 0:
            return
        side = "Long" if self.pos_dir > 0 else "Short"
        pnl_pts = self.pos_dir * (exit_px - self.pos_entry_px)
        pnl_usd_gross = pnl_pts * DOLLAR_PER_POINT * self.contracts
        pnl_usd_net   = pnl_usd_gross - self.commission

        self.trades.append({
            "trade_id":    self.next_trade_id,
            "side":        side,
            "qty":         self.contracts,
            "entry_dt":    self.pos_entry_dt,
            "exit_dt":     exit_dt,
            "entry_px":    self.pos_entry_px,
            "exit_px":     exit_px,
            "entry_name":  self.pos_entry_name,
            "exit_name":   exit_name,
            "pnl_usd":     pnl_usd_net,
            "commission":  self.commission,
        })
        self.next_trade_id += 1
        self.pos_dir = 0
        self.pos_entry_px = 0.0
        self.pos_entry_dt = None
        self.pos_entry_name = ""

    def _open_position(self, side: int, entry_px: float, entry_dt: datetime, entry_name: str):
        self.pos_dir = side  # +1 long, -1 short
        self.pos_entry_px = entry_px
        self.pos_entry_dt = entry_dt
        self.pos_entry_name = entry_name

    def step(self, bar_close: float, bar_close_dt: datetime, next_bar_open: float, next_bar_open_dt: datetime):
        """Process one 1m bar close. `bar_close_dt` is bar END time UTC.
        `next_bar_open` and `next_bar_open_dt` are the fill price/time for
        any orders submitted at this bar's close."""

        mins_of_day = bar_close_dt.hour * 60 + bar_close_dt.minute

        # ─── EOD force-close ───────────────────────────────────────
        if mins_of_day >= self.eod_mins:
            if self.pos_dir != 0:
                exit_name = "EodExitLong" if self.pos_dir > 0 else "EodExitShort"
                self._close_position(next_bar_open, next_bar_open_dt, exit_name)
            return

        # ─── Initialize extreme on first bar ──────────────────────
        if pd.isna(self.extreme_price):
            self.extreme_price = bar_close
            return

        # ─── Zigzag state machine ──────────────────────────────────
        pivot_confirmed = False
        new_pivot_dir = 0  # +1 high pivot, -1 low pivot

        if self.direction == 0:
            if bar_close - self.extreme_price >= self.r:
                pivot_confirmed = True
                new_pivot_dir = -1
                self.direction = +1
                self.extreme_price = bar_close
            elif self.extreme_price - bar_close >= self.r:
                pivot_confirmed = True
                new_pivot_dir = +1
                self.direction = -1
                self.extreme_price = bar_close
        elif self.direction == +1:
            if bar_close > self.extreme_price:
                self.extreme_price = bar_close
            elif self.extreme_price - bar_close >= self.r:
                pivot_confirmed = True
                new_pivot_dir = +1
                self.direction = -1
                self.extreme_price = bar_close
        else:  # direction == -1
            if bar_close < self.extreme_price:
                self.extreme_price = bar_close
            elif bar_close - self.extreme_price >= self.r:
                pivot_confirmed = True
                new_pivot_dir = -1
                self.direction = +1
                self.extreme_price = bar_close

        if not pivot_confirmed:
            return

        # ─── Past entry cutoff: skip entry, no exit (v1.0 behavior) ─
        if mins_of_day >= self.cut_mins:
            return

        # ─── v1.0.4: always exit current position FIRST ────────────
        action = self.on_high if new_pivot_dir == +1 else self.on_low
        pivot_label = "HighPivot" if new_pivot_dir == +1 else "LowPivot"

        if self.pos_dir != 0:
            exit_name = "PivotExitLong" if self.pos_dir > 0 else "PivotExitShort"
            self._close_position(next_bar_open, next_bar_open_dt, exit_name)

        if action == "Skip":
            return

        side = +1 if action == "Long" else -1
        entry_name = ("LongAt" if side == +1 else "ShortAt") + pivot_label
        self._open_position(side, next_bar_open, next_bar_open_dt, entry_name)


# ─── Run sim over loaded bars ─────────────────────────────────────────────

def run_sim(bars: pd.DataFrame, sim: V104Sim) -> pd.DataFrame:
    """Stream bars through the sim. Bars MUST be chronologically sorted.
    The bar at row i represents the OPEN at `dt` and ran for 60 seconds.
    The CLOSE of that bar is at `dt + 60s` and equals row i's `close`.
    For NT8-OnBarClose semantics, the next-bar-open (fill price) is row i+1's `open`.
    """
    n = len(bars)
    closes = bars["close"].to_numpy()
    opens  = bars["open"].to_numpy()
    dts    = bars["dt"].to_list()

    for i in range(n - 1):
        bar_close   = float(closes[i])
        # Bar close DT = row's open dt + 60s
        bar_close_dt = dts[i].to_pydatetime() + pd.Timedelta(seconds=60)
        next_open    = float(opens[i + 1])
        next_open_dt = dts[i + 1].to_pydatetime()

        sim.step(bar_close, bar_close_dt, next_open, next_open_dt)

    df = pd.DataFrame(sim.trades)
    return df


# ─── Parity comparison ────────────────────────────────────────────────────

def compare_trade_lists(nt8_df: pd.DataFrame, py_df: pd.DataFrame) -> pd.DataFrame:
    """Match NT8 trades to Python trades by entry timestamp (within 60s).
    Return a per-trade comparison DataFrame."""
    py_by_entry = {}
    for _, r in py_df.iterrows():
        # Bucket by entry timestamp in seconds (epoch)
        ts = int(r["entry_dt"].timestamp())
        py_by_entry[ts] = r

    rows = []
    matched_py = set()
    for _, n in nt8_df.iterrows():
        # NT8 trade entry timestamp may be UTC-converted differently; try +/- 60s window
        nt8_ts_naive = int(n["entry_dt"].timestamp())
        # NT8 CSV stores LOCAL time (no tzinfo). Try common offsets.
        # We'll find the match within +/- 24h, smallest abs difference, that hasn't been matched.
        best_match = None
        best_diff  = None
        for py_ts, py in py_by_entry.items():
            if py_ts in matched_py:
                continue
            diff_s = py_ts - nt8_ts_naive
            # Allow up to a few hours window for tz mismatch detection
            if best_diff is None or abs(diff_s) < abs(best_diff):
                best_diff = diff_s
                best_match = (py_ts, py)
        py_ts, py = best_match if best_match else (None, None)
        if py_ts is not None:
            matched_py.add(py_ts)

        rows.append({
            "nt8_id":         n["trade_id"],
            "nt8_side":       n["side"],
            "nt8_entry_dt":   n["entry_dt"],
            "nt8_entry_px":   n["entry_px"],
            "nt8_exit_px":    n["exit_px"],
            "nt8_pnl":        n["pnl_usd"],
            "py_id":          py["trade_id"] if py is not None else None,
            "py_side":        py["side"]     if py is not None else None,
            "py_entry_dt":    py["entry_dt"] if py is not None else None,
            "py_entry_px":    py["entry_px"] if py is not None else None,
            "py_exit_px":     py["exit_px"]  if py is not None else None,
            "py_pnl":         py["pnl_usd"]  if py is not None else None,
            "entry_dt_diff_s": best_diff,
            "entry_px_diff":  (py["entry_px"] - n["entry_px"]) if py is not None else None,
            "exit_px_diff":   (py["exit_px"] - n["exit_px"])   if py is not None else None,
            "pnl_diff":       (py["pnl_usd"] - n["pnl_usd"])   if py is not None else None,
        })
    return pd.DataFrame(rows)


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt8-csv", default="examples/trades v1.0.4.csv")
    ap.add_argument("--atlas",   default="DATA/ATLAS_NT8/1m")
    ap.add_argument("--r",       type=float, default=30.0)
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--eod-hour-utc", type=int, default=20)
    ap.add_argument("--eod-minute-utc", type=int, default=55)
    ap.add_argument("--entry-cutoff-hour-utc", type=int, default=20)
    ap.add_argument("--entry-cutoff-minute-utc", type=int, default=30)
    ap.add_argument("--commission", type=float, default=1.90)
    ap.add_argument("--out-csv", default="reports/findings/2026-04-27_v104_parity.csv")
    args = ap.parse_args()

    print("=" * 80)
    print("v1.0.4 PARITY: Python sim vs NT8 Strategy Analyzer")
    print("=" * 80)

    # Load NT8 ground truth
    nt8 = load_nt8_trades(args.nt8_csv)
    if nt8.empty:
        print("FATAL: no trades loaded from NT8 CSV")
        return
    nt8_start = nt8["entry_dt"].min().strftime("%Y-%m-%d")
    nt8_end   = nt8["exit_dt"].max().strftime("%Y-%m-%d")
    nt8_pnl_total = nt8["pnl_usd"].sum()
    print(f"NT8 ground truth: {len(nt8)} trades  {nt8_start} -> {nt8_end}  total \${nt8_pnl_total:+,.2f}")

    # Load 1m bars (cover NT8 window plus a buffer)
    bars = load_1m_bars(args.atlas, nt8_start, nt8_end)
    if bars.empty:
        print(f"FATAL: no 1m bars loaded from {args.atlas} in window {nt8_start}..{nt8_end}")
        return
    print(f"1m bars loaded:  {len(bars):>6}  ({bars['dt'].iloc[0]} -> {bars['dt'].iloc[-1]})")

    # Run sim
    sim = V104Sim(
        r_points=args.r,
        contracts=args.contracts,
        eod_h_utc=args.eod_hour_utc, eod_m_utc=args.eod_minute_utc,
        entry_cut_h_utc=args.entry_cutoff_hour_utc, entry_cut_m_utc=args.entry_cutoff_minute_utc,
        commission_rt_usd=args.commission,
    )
    py = run_sim(bars, sim)
    py_pnl_total = py["pnl_usd"].sum() if len(py) else 0.0
    print(f"Python sim:      {len(py):>6} trades  total \${py_pnl_total:+,.2f}")

    # Parity comparison
    print()
    print("=" * 80)
    print("AGGREGATE PARITY")
    print("=" * 80)
    print(f"{'Metric':<28} {'NT8':>15} {'Python':>15} {'delta':>15}")
    print("-" * 80)
    print(f"{'Trades':<28} {len(nt8):>15} {len(py):>15} {len(py)-len(nt8):>+15}")
    print(f"{'Total PnL ($)':<28} {nt8_pnl_total:>+15,.2f} {py_pnl_total:>+15,.2f} {py_pnl_total-nt8_pnl_total:>+15,.2f}")
    nt8_wins = (nt8["pnl_usd"] > 0).sum()
    py_wins  = (py["pnl_usd"] > 0).sum() if len(py) else 0
    print(f"{'Wins':<28} {nt8_wins:>15} {py_wins:>15} {py_wins-nt8_wins:>+15}")
    nt8_long = (nt8["side"] == "Long").sum()
    py_long  = (py["side"] == "Long").sum() if len(py) else 0
    print(f"{'Long trades':<28} {nt8_long:>15} {py_long:>15} {py_long-nt8_long:>+15}")
    nt8_short = (nt8["side"] == "Short").sum()
    py_short  = (py["side"] == "Short").sum() if len(py) else 0
    print(f"{'Short trades':<28} {nt8_short:>15} {py_short:>15} {py_short-nt8_short:>+15}")

    # Dump RAW Python trades to disk for direct inspection
    py_raw_path = args.out_csv.replace(".csv", "_python_raw.csv")
    py.to_csv(py_raw_path, index=False)
    print(f"\nRaw Python trades:    {py_raw_path}")

    # Per-day histogram
    print()
    print("RAW per-day trade counts:")
    print(f"  {'date':<12} {'NT8':>6} {'Python':>8} {'diff':>6}")
    nt8_by_day = {}
    py_by_day = {}
    for _, r in nt8.iterrows():
        d = r["entry_dt"].date()
        nt8_by_day[d] = nt8_by_day.get(d, 0) + 1
    for _, r in py.iterrows():
        d = r["entry_dt"].date()
        py_by_day[d] = py_by_day.get(d, 0) + 1
    all_days = sorted(set(nt8_by_day.keys()) | set(py_by_day.keys()))
    for d in all_days:
        a = nt8_by_day.get(d, 0); b = py_by_day.get(d, 0)
        print(f"  {str(d):<12} {a:>6} {b:>8} {b-a:>+6}")

    # Per-trade comparison
    if len(py) > 0:
        cmp = compare_trade_lists(nt8, py)
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        cmp.to_csv(args.out_csv, index=False)
        print(f"\nPer-trade comparison: {args.out_csv}")
        # Show fill timing offsets (suggests TZ mismatch)
        diffs = cmp["entry_dt_diff_s"].dropna()
        if len(diffs):
            from collections import Counter
            print(f"\nEntry timestamp diff (Python - NT8) distribution (seconds):")
            common = Counter(diffs.astype(int)).most_common(5)
            for d, c in common:
                print(f"  {d:>+8}s : {c} trades")


if __name__ == "__main__":
    main()
