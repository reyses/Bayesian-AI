"""
v104_trade_eda.py -- Phase 1 EDA on the v1.0.4 NT8 trades CSV.

Money-focused exploratory analysis on the 517-trade SA backtest ledger.
No features required (ATLAS_NT8 join is Phase 2). This phase only uses
the trade CSV + 1m bars to test cut/exit/filter hypotheses.

Headline analysis (user-flagged): "trades that start negative stay negative."
We reconstruct per-trade bar-by-bar PnL trajectories from ATLAS_NT8 1m bars
and compute P(win | PnL_state at bar N) for N = 1..30.

If P(win | negative at bar N) << overall WR, the cut rule is justified.

Other analyses produced:
  - PnL distribution + MFE/MAE/ETD physics
  - Hold-time cliff (bucket by bars held)
  - Hour-of-day + day-of-week stratification
  - Direction-stratified (long vs short physics)
  - Sequence patterns (P(win | prev loss/win), streaks)
  - Drop-shorts / drop-longs PnL simulation
  - Hard-stop sweep (theoretical PnL if cut at -X points)
  - Trail-stop simulation (theoretical PnL at various give-back levels)
  - Max-drawdown summary

Output: reports/findings/2026-04-27_v104_eda_phase1.md

Usage:
    python tools/v104_trade_eda.py
    python tools/v104_trade_eda.py --csv "examples/trades v1.0.4.csv"
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

DOLLAR_PER_POINT = 2.0  # MNQ
TICK_SIZE = 0.25


def parse_money(s: str) -> float:
    s = s.replace("$", "").replace(",", "").strip()
    if not s:
        return 0.0
    if s.startswith("(") and s.endswith(")"):
        return -float(s[1:-1])
    return float(s)


def parse_dt(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%m/%d/%Y %I:%M:%S %p")


def load_trades(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                entry_dt = parse_dt(r["Entry time"])
                exit_dt = parse_dt(r["Exit time"])
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
                "pnl":        parse_money(r["Profit"]),
                "commission": parse_money(r["Commission"]),
                "mfe":        parse_money(r["MFE"]),
                "mae":        parse_money(r["MAE"]),
                "etd":        parse_money(r["ETD"]),
                "bars":       int(r["Bars"]),
            })
    df = pd.DataFrame(rows)
    df["dir"] = df["side"].map({"Long": +1, "Short": -1})
    df["is_win"] = df["pnl"] > 0
    df["pnl_pts"] = df["pnl"] / DOLLAR_PER_POINT
    return df


def load_1m_bars_indexed_utc(atlas_root: str) -> pd.DataFrame:
    """Load all ATLAS_NT8 1m parquets, return df indexed by UTC datetime."""
    parts = []
    for f in sorted(Path(atlas_root).glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    out["dt_utc"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out


# ─── Trade trajectory reconstruction ──────────────────────────────────────

def reconstruct_trajectory(trade: pd.Series, bars: pd.DataFrame, max_bars: int = 60,
                            user_tz_offset_h: int = 7) -> dict:
    """For one trade, compute close-price PnL at each 1m bar from entry to exit.
    Returns dict with arrays of bar PnL ($) up to max_bars (or trade exit, whichever first).

    NT8 CSV times are in user LOCAL (PDT, UTC-7). Convert to UTC to match
    ATLAS_NT8 bar timestamps.
    """
    entry_utc = trade["entry_dt"] + timedelta(hours=user_tz_offset_h)
    exit_utc  = trade["exit_dt"]  + timedelta(hours=user_tz_offset_h)

    # Find the first bar at-or-after entry timestamp
    bar_dts = bars["dt_utc"].dt.tz_localize(None) if bars["dt_utc"].dt.tz else bars["dt_utc"]
    bar_times_utc = bars["dt_utc"]
    # Note: ATLAS_NT8 bar timestamps are bar OPEN (per BayesianHistoryDumper convention)
    # but with :59 second offset. We treat them as start-of-minute for indexing.
    entry_mask = bar_times_utc >= pd.Timestamp(entry_utc, tz="UTC")
    if not entry_mask.any():
        return {"bar_pnls": [], "n_bars": 0}

    start_idx = entry_mask.idxmax()
    end_mask = bar_times_utc >= pd.Timestamp(exit_utc, tz="UTC")
    end_idx = end_mask.idxmax() if end_mask.any() else len(bars) - 1

    n = min(end_idx - start_idx + 1, max_bars)
    if n <= 0:
        return {"bar_pnls": [], "n_bars": 0}

    closes = bars["close"].iloc[start_idx:start_idx + n].to_numpy()
    pnl_pts = trade["dir"] * (closes - trade["entry_px"])
    pnl_usd = pnl_pts * DOLLAR_PER_POINT * trade["qty"]
    return {"bar_pnls": pnl_usd.tolist(), "n_bars": int(n)}


# ─── Analyses ─────────────────────────────────────────────────────────────

def section_summary(df: pd.DataFrame) -> str:
    n = len(df)
    n_long  = int((df["side"] == "Long").sum())
    n_short = int((df["side"] == "Short").sum())
    n_win   = int(df["is_win"].sum())
    pnl_total = df["pnl"].sum()
    comm_total = df["commission"].sum()
    days = df["entry_dt"].dt.date.nunique()
    long_pnl  = df.loc[df["side"] == "Long",  "pnl"].sum()
    short_pnl = df.loc[df["side"] == "Short", "pnl"].sum()
    long_wr   = (df.loc[df["side"] == "Long",  "is_win"]).mean()
    short_wr  = (df.loc[df["side"] == "Short", "is_win"]).mean()
    out = []
    out.append("## Baseline summary\n")
    out.append(f"- Trades: **{n}** ({n_long}L / {n_short}S)")
    out.append(f"- Window: {df['entry_dt'].min().date()} → {df['exit_dt'].max().date()} ({days} days)")
    out.append(f"- Net PnL: **${pnl_total:+,.2f}** (after ${comm_total:,.2f} commission)")
    out.append(f"- Gross PnL: ${pnl_total + comm_total:+,.2f}")
    out.append(f"- $/day: ${pnl_total/days:+,.2f}")
    out.append(f"- Win rate: {100*n_win/n:.1f}% ({n_win}/{n})")
    out.append(f"- Long  PnL: **${long_pnl:+,.2f}** (${long_pnl/n_long:+.2f}/tr, {100*long_wr:.1f}% WR)")
    out.append(f"- Short PnL: **${short_pnl:+,.2f}** (${short_pnl/n_short:+.2f}/tr, {100*short_wr:.1f}% WR)")
    out.append(f"- Best/Worst: ${df['pnl'].max():+,.2f} / ${df['pnl'].min():+,.2f}")
    out.append(f"- Mean MFE: ${df['mfe'].mean():.2f}  Mean MAE: ${df['mae'].mean():.2f}  Mean ETD: ${df['etd'].mean():.2f}")
    capture = df["pnl"].sum() / df["mfe"].sum() if df["mfe"].sum() > 0 else 0
    out.append(f"- Capture rate: {100*capture:.1f}% of total MFE captured")
    out.append("")
    return "\n".join(out)


def section_holdtime_cliff(df: pd.DataFrame) -> str:
    out = ["## Hold-time cliff (bucket by `Bars` held)\n"]
    out.append("| Bucket (bars) | N | WR | $/trade | Total PnL |")
    out.append("|---|---:|---:|---:|---:|")
    bins = [(0,5), (5,10), (10,20), (20,30), (30,60), (60,120), (120,300), (300,99999)]
    for lo, hi in bins:
        sub = df[(df["bars"] >= lo) & (df["bars"] < hi)]
        if len(sub) == 0: continue
        wr = sub["is_win"].mean()
        per = sub["pnl"].mean()
        tot = sub["pnl"].sum()
        out.append(f"| {lo}–{hi} | {len(sub)} | {100*wr:.1f}% | ${per:+,.2f} | ${tot:+,.2f} |")
    out.append("")
    return "\n".join(out)


def section_hour_of_day(df: pd.DataFrame) -> str:
    """NT8 CSV time is PDT. Group by entry hour (PDT)."""
    out = ["## Hour-of-day stratification (entry hour, PDT)\n"]
    out.append("| Hour | N | WR | $/trade | Total PnL |")
    out.append("|---:|---:|---:|---:|---:|")
    df = df.copy()
    df["hour"] = df["entry_dt"].dt.hour
    for h in sorted(df["hour"].unique()):
        sub = df[df["hour"] == h]
        wr = sub["is_win"].mean()
        out.append(f"| {h:02d} | {len(sub)} | {100*wr:.1f}% | ${sub['pnl'].mean():+,.2f} | ${sub['pnl'].sum():+,.2f} |")
    out.append("")
    return "\n".join(out)


def section_dow(df: pd.DataFrame) -> str:
    out = ["## Day-of-week stratification\n"]
    out.append("| DoW | N | WR | $/trade | Total PnL |")
    out.append("|---|---:|---:|---:|---:|")
    df = df.copy()
    df["dow"] = df["entry_dt"].dt.day_name()
    order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    for d in order:
        sub = df[df["dow"] == d]
        if len(sub) == 0: continue
        wr = sub["is_win"].mean()
        out.append(f"| {d} | {len(sub)} | {100*wr:.1f}% | ${sub['pnl'].mean():+,.2f} | ${sub['pnl'].sum():+,.2f} |")
    out.append("")
    return "\n".join(out)


def section_direction_physics(df: pd.DataFrame) -> str:
    out = ["## Direction-stratified physics\n"]
    out.append("| Side | N | WR | $/tr | MFE mean | MAE mean | ETD mean | bars med |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for side in ["Long", "Short"]:
        sub = df[df["side"] == side]
        if len(sub) == 0: continue
        out.append(f"| {side} | {len(sub)} | {100*sub['is_win'].mean():.1f}% | "
                   f"${sub['pnl'].mean():+,.2f} | ${sub['mfe'].mean():.2f} | "
                   f"${sub['mae'].mean():.2f} | ${sub['etd'].mean():.2f} | {sub['bars'].median():.0f} |")
    out.append("")
    return "\n".join(out)


def section_drop_side(df: pd.DataFrame) -> str:
    long_pnl = df.loc[df["side"] == "Long", "pnl"].sum()
    short_pnl = df.loc[df["side"] == "Short", "pnl"].sum()
    total = df["pnl"].sum()
    out = ["## Drop-side simulation\n"]
    out.append(f"- Current (both sides): **${total:+,.2f}**")
    out.append(f"- Drop SHORTS, longs only: **${long_pnl:+,.2f}** (delta {long_pnl-total:+,.2f})")
    out.append(f"- Drop LONGS, shorts only: **${short_pnl:+,.2f}** (delta {short_pnl-total:+,.2f})")
    out.append("")
    out.append("Note: this window had a strong uptrend. Direction-asymmetric edge here is regime-specific.")
    out.append("")
    return "\n".join(out)


def section_hard_stop_sweep(df: pd.DataFrame) -> str:
    """Theoretical PnL if we cut every trade at -X points unrealized.
    Assumes the trade would have hit -X (using MAE) and the SL cap fires
    at -X * $2 + small slippage. Trades whose MAE never reached X are unaffected."""
    out = ["## Hard-stop sweep (theoretical)\n"]
    out.append("Idea: if MAE >= X dollars, force exit at -X (replacing actual PnL).\n")
    out.append("| SL pts | SL $ | Trades cut | New PnL | Δ vs current |")
    out.append("|---:|---:|---:|---:|---:|")
    base = df["pnl"].sum()
    for pts in [25, 50, 75, 100, 150, 200]:
        cap_usd = pts * DOLLAR_PER_POINT
        # MAE in df is positive $ value of max adverse excursion
        cut_mask = df["mae"] >= cap_usd
        # When SL fires at -cap_usd, trade exits at -cap_usd (worst case loss = cap)
        new_pnl = df["pnl"].where(~cut_mask, -cap_usd).sum()
        out.append(f"| {pts} | ${cap_usd:.0f} | {int(cut_mask.sum())} | ${new_pnl:+,.2f} | ${new_pnl-base:+,.2f} |")
    out.append("")
    out.append("Caveat: this is OPTIMISTIC — it assumes any trade with MAE>=X")
    out.append("would have cap-exited cleanly at -X. Real fills can slip past the cap.")
    out.append("Also: cut trades that LATER recovered and won are LOST in this simulation.")
    out.append("")
    return "\n".join(out)


def section_trail_stop_sim(df: pd.DataFrame) -> str:
    """Theoretical PnL if we exit at MFE - giveback whenever MFE > activate.
    Approximation: assume trade reached MFE at some point, would have exited
    at MFE - giveback, and that becomes PnL."""
    out = ["## Trail-stop simulation (theoretical)\n"]
    out.append("Idea: if MFE > activate, exit at MFE - giveback. Otherwise keep current PnL.\n")
    out.append("| Activate $ | Giveback $ | Trades trailed | New PnL | Δ vs current |")
    out.append("|---:|---:|---:|---:|---:|")
    base = df["pnl"].sum()
    for activate in [10, 20, 40]:
        for giveback in [5, 10, 20, 40]:
            if giveback >= activate:
                continue
            mask = df["mfe"] >= activate
            sim_pnl = df["pnl"].copy()
            sim_pnl[mask] = df.loc[mask, "mfe"] - giveback
            new = sim_pnl.sum()
            out.append(f"| {activate} | {giveback} | {int(mask.sum())} | ${new:+,.2f} | ${new-base:+,.2f} |")
    out.append("")
    out.append("Caveat: assumes trail fires AT MFE peak — best-case. Real fills lag.")
    out.append("Also assumes the trade actually reached MFE before turning back. Always true by definition (MFE = max favorable).")
    out.append("")
    return "\n".join(out)


def section_sequence(df: pd.DataFrame) -> str:
    df = df.sort_values("entry_dt").reset_index(drop=True)
    df["prev_win"] = df["is_win"].shift(1)
    df["prev_loss"] = ~df["is_win"].shift(1).fillna(False).astype(bool)

    out = ["## Sequence patterns\n"]
    overall = df["is_win"].mean()
    out.append(f"- Overall WR: {100*overall:.1f}%")

    after_win = df.loc[df["prev_win"] == True, "is_win"].mean()
    after_loss = df.loc[df["prev_win"] == False, "is_win"].mean()
    if not pd.isna(after_win):
        out.append(f"- WR after win:  {100*after_win:.1f}%  ({int((df['prev_win']==True).sum())} trades)")
    if not pd.isna(after_loss):
        out.append(f"- WR after loss: {100*after_loss:.1f}%  ({int((df['prev_win']==False).sum())} trades)")

    # Streak analysis
    df["streak"] = (df["is_win"] != df["is_win"].shift()).cumsum()
    streaks = df.groupby("streak").agg(streak_len=("is_win", "size"), is_win=("is_win", "first"))
    win_streaks = streaks[streaks["is_win"]]["streak_len"]
    loss_streaks = streaks[~streaks["is_win"]]["streak_len"]
    out.append(f"- Longest win streak: {int(win_streaks.max()) if len(win_streaks) else 0}")
    out.append(f"- Longest loss streak: {int(loss_streaks.max()) if len(loss_streaks) else 0}")
    out.append("")
    return "\n".join(out)


def section_trajectory(df: pd.DataFrame, bars: pd.DataFrame, max_bars: int = 30) -> str:
    """Headline analysis: P(win | PnL state at bar N)."""
    out = ["## TRAJECTORY ANALYSIS — does negative-early stay negative?\n"]
    out.append("For each trade, reconstruct close-price PnL at bar N from entry.")
    out.append("Then group trades by 'PnL state at bar N' and report final WR per group.\n")

    n_with_traj = 0
    bar_states = defaultdict(list)  # bar_idx -> list of (pnl_at_bar, final_pnl)

    for _, t in df.iterrows():
        traj = reconstruct_trajectory(t, bars, max_bars=max_bars)
        if traj["n_bars"] < 2:
            continue
        n_with_traj += 1
        for bi, p in enumerate(traj["bar_pnls"]):
            bar_states[bi].append((float(p), float(t["pnl"])))

    out.append(f"Trajectories reconstructed: {n_with_traj}/{len(df)}\n")

    out.append("| Bar | N | %neg at bar | %pos at bar | WR(neg) | WR(pos) | E[$] if neg | E[$] if pos |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bi in sorted(bar_states.keys()):
        if bi >= max_bars: break
        pairs = bar_states[bi]
        if len(pairs) < 20: continue  # need enough N
        neg = [(b,f) for b,f in pairs if b < 0]
        pos = [(b,f) for b,f in pairs if b > 0]
        zer = [(b,f) for b,f in pairs if b == 0]
        # group zeros with neg conservatively
        neg_or_zer = neg + zer
        wr_neg = (sum(1 for _,f in neg_or_zer if f > 0) / len(neg_or_zer)) if len(neg_or_zer) else 0
        wr_pos = (sum(1 for _,f in pos if f > 0) / len(pos)) if len(pos) else 0
        e_neg = (sum(f for _,f in neg_or_zer) / len(neg_or_zer)) if len(neg_or_zer) else 0
        e_pos = (sum(f for _,f in pos) / len(pos)) if len(pos) else 0
        pct_neg = 100 * len(neg_or_zer) / len(pairs)
        pct_pos = 100 * len(pos) / len(pairs)
        out.append(f"| {bi} | {len(pairs)} | {pct_neg:.1f}% | {pct_pos:.1f}% | "
                   f"{100*wr_neg:.1f}% | {100*wr_pos:.1f}% | "
                   f"${e_neg:+,.2f} | ${e_pos:+,.2f} |")

    out.append("")
    out.append("**Interpretation**: if `WR(neg)` drops well below overall WR (36.8%) at low bar")
    out.append("indices, then 'trades that go negative early are unlikely to recover' is empirically")
    out.append("supported. Cut rule: at bar N, if PnL < 0, exit. Test the value of N where")
    out.append("`WR(neg)` is minimized AND sample size is sufficient.")
    out.append("")
    out.append("**Cut rule simulation**: for each candidate bar N, compute simulated PnL if we")
    out.append("force-exit (at bar N's close price) any trade whose bar-N PnL < 0:")
    out.append("")
    out.append("| Cut bar N | Trades cut | Cut sum | Untouched sum | Total simulated | Δ vs base |")
    out.append("|---:|---:|---:|---:|---:|---:|")
    base = df["pnl"].sum()
    for N in [1, 2, 3, 5, 7, 10, 15, 20]:
        if N not in bar_states: continue
        # For each trade, compute bar-N PnL
        traj_cache = {}
        for tid, t in df.iterrows():
            traj = reconstruct_trajectory(t, bars, max_bars=max_bars)
            traj_cache[tid] = traj
        cut_sum = 0.0
        keep_sum = 0.0
        n_cut = 0
        for tid, t in df.iterrows():
            traj = traj_cache[tid]
            if traj["n_bars"] <= N:
                # trade closed before bar N anyway, keep actual PnL
                keep_sum += t["pnl"]
                continue
            barN_pnl = traj["bar_pnls"][N]
            if barN_pnl < 0:
                # cut at bar N's close price
                cut_sum += barN_pnl
                n_cut += 1
            else:
                keep_sum += t["pnl"]
        sim = cut_sum + keep_sum
        out.append(f"| {N} | {n_cut} | ${cut_sum:+,.2f} | ${keep_sum:+,.2f} | ${sim:+,.2f} | ${sim-base:+,.2f} |")
    out.append("")
    out.append("Caveat: cut PnL ignores commission for the cut leg. Real cut would add ~$1.90 commission.")
    out.append("If 'cut bar N' shows Δ > $0, the rule has merit; lower bar N = faster cut = stricter.")
    return "\n".join(out)


def section_per_day_pnl(df: pd.DataFrame) -> str:
    out = ["## Per-day PnL distribution\n"]
    daily = df.groupby(df["entry_dt"].dt.date)["pnl"].agg(["sum", "count"]).rename(columns={"sum": "pnl", "count": "n"})
    daily = daily.sort_values("pnl")
    n_days = len(daily)
    n_pos = int((daily["pnl"] > 0).sum())
    n_neg = int((daily["pnl"] < 0).sum())
    out.append(f"- Total days: {n_days}")
    out.append(f"- Positive days: {n_pos} ({100*n_pos/n_days:.1f}%)")
    out.append(f"- Negative days: {n_neg} ({100*n_neg/n_days:.1f}%)")
    out.append(f"- Best day: ${daily['pnl'].max():+,.2f}")
    out.append(f"- Worst day: ${daily['pnl'].min():+,.2f}")
    out.append(f"- Median day: ${daily['pnl'].median():+,.2f}")
    out.append(f"- Std day: ${daily['pnl'].std():,.2f}")
    out.append("")
    out.append("Worst 5 days:")
    for d, r in daily.head(5).iterrows():
        out.append(f"- {d}: ${r['pnl']:+,.2f} ({int(r['n'])} trades)")
    out.append("")
    out.append("Best 5 days:")
    for d, r in daily.tail(5).iterrows():
        out.append(f"- {d}: ${r['pnl']:+,.2f} ({int(r['n'])} trades)")
    out.append("")
    return "\n".join(out)


def section_max_drawdown(df: pd.DataFrame) -> str:
    out = ["## Max drawdown analysis\n"]
    df = df.sort_values("entry_dt").reset_index(drop=True)
    df["cum"] = df["pnl"].cumsum()
    df["peak"] = df["cum"].cummax()
    df["dd"] = df["cum"] - df["peak"]
    max_dd = df["dd"].min()
    max_dd_idx = df["dd"].idxmin()
    peak_at = df["peak"].iloc[max_dd_idx]
    out.append(f"- Max drawdown: **${max_dd:,.2f}** (at trade #{int(df['trade_id'].iloc[max_dd_idx])})")
    out.append(f"- Peak before drawdown: ${peak_at:+,.2f}")
    out.append(f"- Cumulative PnL at drawdown bottom: ${df['cum'].iloc[max_dd_idx]:+,.2f}")
    out.append("")
    return "\n".join(out)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="examples/trades v1.0.4.csv")
    ap.add_argument("--atlas-1m", default="DATA/ATLAS_NT8/1m")
    ap.add_argument("--out", default="reports/findings/2026-04-27_v104_eda_phase1.md")
    ap.add_argument("--user-tz-offset", type=int, default=7,
                    help="NT8 CSV uses local time. Offset to UTC in hours (PDT=7).")
    args = ap.parse_args()

    print(f"Loading trades from {args.csv}")
    df = load_trades(args.csv)
    print(f"  Loaded {len(df)} trades.")

    print(f"Loading 1m bars from {args.atlas_1m}")
    bars = load_1m_bars_indexed_utc(args.atlas_1m)
    print(f"  Loaded {len(bars)} 1m bars.")

    sections = []
    sections.append(f"# v1.0.4 Phase 1 EDA — {datetime.now():%Y-%m-%d %H:%M}\n")
    sections.append(f"Source: `{args.csv}`\n")
    sections.append(f"Bars: `{args.atlas_1m}` ({len(bars)} 1m bars)\n")
    sections.append(section_summary(df))
    sections.append(section_max_drawdown(df))
    sections.append(section_trajectory(df, bars))
    sections.append(section_holdtime_cliff(df))
    sections.append(section_direction_physics(df))
    sections.append(section_drop_side(df))
    sections.append(section_hard_stop_sweep(df))
    sections.append(section_trail_stop_sim(df))
    sections.append(section_hour_of_day(df))
    sections.append(section_dow(df))
    sections.append(section_sequence(df))
    sections.append(section_per_day_pnl(df))

    text = "\n".join(sections)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nReport written: {args.out}")
    # Also dump to stdout
    print()
    print(text)


if __name__ == "__main__":
    main()
