"""
v104_trade_trajectory.py -- Reconstruct bar-by-bar PnL trajectory for every
trade in a NT8 trade-export CSV using ATLAS_NT8 1m bars.

For each trade we compute, at each bar from entry to exit:
  - close_pnl: PnL using bar close
  - high_pnl: PnL using bar high (potential MFE intra-bar)
  - low_pnl: PnL using bar low (potential MAE intra-bar)
  - running_mfe: max(high_pnl seen so far)
  - running_mae: min(low_pnl seen so far)

Plus per-trade summary fields:
  - bar_of_max_mae: bar index where running_mae was minimized
  - bar_of_max_mfe: bar index where running_mfe was maximized
  - mae_realized_pct: realized MAE / final MAE (from CSV) — sanity check
  - trajectory_class: HARD_LOSER / SOFT_LOSER / WINNER_DIPPED / WINNER_CLEAN

Output:
  - Enriched per-trade CSV: reports/findings/<csv_stem>_trajectories.csv
  - Markdown summary: reports/findings/<csv_stem>_trajectory_analysis.md

Usage:
    python tools/v104_trade_trajectory.py --csv "examples/trades v1.0.4 playback.csv"
    python tools/v104_trade_trajectory.py --csv "examples/trades v1.0.4.csv"
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

DOLLAR_PER_POINT = 2.0  # MNQ
MAX_BARS = 60   # cap on per-trade trajectory length (1m bars)


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
            row = {
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
                "mfe_csv":    parse_money(r["MFE"]),
                "mae_csv":    parse_money(r["MAE"]),
                "etd_csv":    parse_money(r.get("ETD", "0")),
                "bars":       int(r["Bars"]),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df["dir"] = df["side"].map({"Long": +1, "Short": -1})
    df["is_win"] = df["pnl"] > 0
    return df


def load_1m(atlas_root: str) -> pd.DataFrame:
    parts = []
    for f in sorted(Path(atlas_root).glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty: continue
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    out["dt_utc"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out


def reconstruct(trade: pd.Series, bars: pd.DataFrame, max_bars: int = MAX_BARS,
                user_tz_offset_h: int = 7) -> dict:
    """Walk bars from entry to exit, return per-bar PnL trajectory using
    high/low/close prices. NT8 CSV times are PDT (UTC-7) by default."""
    entry_utc = trade["entry_dt"] + timedelta(hours=user_tz_offset_h)
    exit_utc  = trade["exit_dt"]  + timedelta(hours=user_tz_offset_h)

    # Find first bar at-or-after entry timestamp (UTC-aware comparison)
    entry_ts = pd.Timestamp(entry_utc, tz="UTC")
    exit_ts  = pd.Timestamp(exit_utc,  tz="UTC")
    mask_entry = bars["dt_utc"] >= entry_ts
    if not mask_entry.any():
        return None
    start_idx = mask_entry.idxmax()
    mask_exit = bars["dt_utc"] >= exit_ts
    end_idx = mask_exit.idxmax() if mask_exit.any() else len(bars) - 1

    n_avail = min(end_idx - start_idx + 1, max_bars)
    if n_avail < 1:
        return None

    seg = bars.iloc[start_idx:start_idx + n_avail]
    closes = seg["close"].to_numpy()
    highs  = seg["high"].to_numpy()
    lows   = seg["low"].to_numpy()
    times  = seg["dt_utc"].to_list()

    d = trade["dir"]
    e = trade["entry_px"]
    q = trade["qty"]
    f = DOLLAR_PER_POINT * q

    close_pnl = (d * (closes - e)) * f
    high_pnl  = (d * (highs  - e)) * f if d > 0 else (d * (lows  - e)) * f  # for long, max is at high; for short, max is at low
    low_pnl   = (d * (lows   - e)) * f if d > 0 else (d * (highs - e)) * f  # for long, min is at low; for short, min is at high

    # Running MFE and MAE (cumulative max/min)
    running_mfe = np.maximum.accumulate(high_pnl)
    running_mae = np.minimum.accumulate(low_pnl)

    return {
        "n_bars": int(n_avail),
        "times":  times,
        "close_pnl":   close_pnl.tolist(),
        "high_pnl":    high_pnl.tolist(),
        "low_pnl":     low_pnl.tolist(),
        "running_mfe": running_mfe.tolist(),
        "running_mae": running_mae.tolist(),
    }


def classify_trajectory(traj: dict, is_win: bool) -> str:
    """Categorize each trade by its bar-1 state and final outcome."""
    if traj is None or traj["n_bars"] < 2:
        return "NO_DATA"
    bar1_pnl = traj["close_pnl"][1] if traj["n_bars"] >= 2 else traj["close_pnl"][0]
    if is_win:
        return "WINNER_CLEAN" if bar1_pnl >= 0 else "WINNER_DIPPED"
    else:
        return "SOFT_LOSER" if bar1_pnl >= 0 else "HARD_LOSER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--atlas-1m", default="DATA/ATLAS_NT8/1m")
    ap.add_argument("--out-csv-dir", default="reports/findings")
    ap.add_argument("--out-md-dir", default="reports/findings")
    ap.add_argument("--user-tz-offset", type=int, default=7,
                    help="NT8 CSV uses local time. Offset to UTC in hours (PDT=7).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    stem = csv_path.stem.replace(" ", "_")  # for output filenames

    print(f"Loading trades from {csv_path}")
    trades = load_trades(str(csv_path))
    print(f"  {len(trades)} trades loaded.")

    print(f"Loading 1m bars from {args.atlas_1m}")
    bars = load_1m(args.atlas_1m)
    print(f"  {len(bars)} bars loaded ({bars['dt_utc'].min()} -> {bars['dt_utc'].max()})")

    # Reconstruct
    print("\nReconstructing trajectories...")
    enriched_rows = []
    bar_close_pnls = {}    # bar_idx -> list of close pnls (one per trade)
    bar_running_mae = {}   # bar_idx -> list of running MAE
    bar_running_mfe = {}   # bar_idx -> list of running MFE
    classes = []
    bar_of_mae_list = []
    bar_of_mfe_list = []
    n_with_traj = 0

    for _, t in trades.iterrows():
        traj = reconstruct(t, bars, max_bars=MAX_BARS, user_tz_offset_h=args.user_tz_offset)
        if traj is None:
            classes.append("NO_DATA")
            continue
        n_with_traj += 1
        cls = classify_trajectory(traj, bool(t["is_win"]))
        classes.append(cls)

        # Find bar index of running MAE minimum and MFE maximum
        rmae = np.array(traj["running_mae"])
        rmfe = np.array(traj["running_mfe"])
        # Bar where running_mae first reached its minimum (index of new low formation)
        bar_of_mae = int(np.argmin(rmae))
        bar_of_mfe = int(np.argmax(rmfe))
        bar_of_mae_list.append(bar_of_mae)
        bar_of_mfe_list.append(bar_of_mfe)

        # Build enriched row
        row = dict(t)
        row["traj_class"] = cls
        row["traj_n_bars"] = traj["n_bars"]
        row["recon_mae"] = float(rmae.min())
        row["recon_mfe"] = float(rmfe.max())
        row["bar_of_mae"] = bar_of_mae
        row["bar_of_mfe"] = bar_of_mfe
        # Per-bar close pnl up to MAX_BARS (trim if shorter)
        for i in range(MAX_BARS):
            row[f"close_pnl_b{i:02d}"] = traj["close_pnl"][i] if i < traj["n_bars"] else np.nan
            row[f"running_mae_b{i:02d}"] = traj["running_mae"][i] if i < traj["n_bars"] else np.nan
            row[f"running_mfe_b{i:02d}"] = traj["running_mfe"][i] if i < traj["n_bars"] else np.nan
            # Aggregates for histogram
            if i < traj["n_bars"]:
                bar_close_pnls.setdefault(i, []).append(traj["close_pnl"][i])
                bar_running_mae.setdefault(i, []).append(traj["running_mae"][i])
                bar_running_mfe.setdefault(i, []).append(traj["running_mfe"][i])
        enriched_rows.append(row)

    enriched = pd.DataFrame(enriched_rows)
    print(f"  {n_with_traj}/{len(trades)} trajectories reconstructed.")

    # Output enriched CSV
    out_csv = Path(args.out_csv_dir) / f"{stem}_trajectories.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_csv, index=False)
    print(f"\nEnriched per-trade CSV: {out_csv}")

    # Build markdown summary
    md = []
    md.append(f"# Trade trajectory analysis — {csv_path.name}\n")
    md.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n")
    md.append(f"Bars: `{args.atlas_1m}`\n")
    md.append(f"Trades: {len(trades)}, trajectories reconstructed: {n_with_traj}\n")

    # Reconstruct vs CSV MAE/MFE sanity check
    md.append("## Reconstruction sanity check\n")
    md.append("Comparing reconstructed (intra-bar high/low) MAE/MFE vs CSV-reported.\n")
    md.append(f"- Mean CSV MAE: ${trades['mae_csv'].mean():.2f}  vs Mean reconstructed: ${enriched['recon_mae'].abs().mean():.2f}")
    md.append(f"- Mean CSV MFE: ${trades['mfe_csv'].mean():.2f}  vs Mean reconstructed: ${enriched['recon_mfe'].mean():.2f}")
    md.append("Differences expected (reconstruction caps at "
              f"{MAX_BARS} bars; CSV MAE/MFE is full-trade).\n")

    # Trajectory class breakdown
    md.append("## Trajectory class breakdown\n")
    cls_counts = pd.Series(classes).value_counts()
    md.append(f"| Class | N | Description |")
    md.append(f"|---|---:|---|")
    descriptions = {
        "WINNER_CLEAN":  "Won, started positive",
        "WINNER_DIPPED": "Won, but went negative first",
        "SOFT_LOSER":    "Lost, briefly went positive",
        "HARD_LOSER":    "Lost, never went positive",
        "NO_DATA":       "Trajectory could not be reconstructed (bars missing)",
    }
    for cls, cnt in cls_counts.items():
        md.append(f"| {cls} | {cnt} | {descriptions.get(cls, '')}|")
    md.append("")

    # Per-class WR & PnL
    md.append("### Class summary stats\n")
    md.append("| Class | N | Mean PnL | Mean MAE (CSV) | Mean MFE (CSV) | Median bars |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for cls in ["WINNER_CLEAN", "WINNER_DIPPED", "SOFT_LOSER", "HARD_LOSER"]:
        sub = enriched[enriched["traj_class"] == cls]
        if len(sub) == 0: continue
        md.append(f"| {cls} | {len(sub)} | ${sub['pnl'].mean():+,.2f} | ${sub['mae_csv'].mean():.2f} | ${sub['mfe_csv'].mean():.2f} | {sub['bars'].median():.0f} |")
    md.append("")

    # Bar-of-MAE distribution (the core question)
    md.append("## When does MAE occur? (running-MAE bar index)\n")
    md.append("Bar index at which the trade reached its deepest negative point so far.\n")
    md.append("| Cohort | N | Mean bar-of-MAE | Median | p25 | p75 | p90 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for cohort, mask in [
        ("ALL", enriched["traj_class"] != "NO_DATA"),
        ("Winners (any)", enriched["is_win"] == True),
        ("Losers (any)", enriched["is_win"] == False),
        ("HARD_LOSER", enriched["traj_class"] == "HARD_LOSER"),
        ("SOFT_LOSER", enriched["traj_class"] == "SOFT_LOSER"),
        ("WINNER_DIPPED", enriched["traj_class"] == "WINNER_DIPPED"),
    ]:
        sub = enriched[mask]
        if len(sub) == 0: continue
        b = sub["bar_of_mae"]
        md.append(f"| {cohort} | {len(sub)} | {b.mean():.1f} | {b.median():.0f} | {b.quantile(0.25):.0f} | {b.quantile(0.75):.0f} | {b.quantile(0.90):.0f} |")
    md.append("")

    # Bar-by-bar percentile heatmap of running MAE
    md.append("## Bar-by-bar running MAE percentiles (across all trades)\n")
    md.append("How deep has MAE reached by bar N?\n")
    md.append("| Bar | N | p10 | p25 | p50 (median) | p75 | p90 | mean |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bi in [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 59]:
        if bi not in bar_running_mae or len(bar_running_mae[bi]) < 20:
            continue
        arr = np.array(bar_running_mae[bi])
        md.append(f"| {bi} | {len(arr)} | ${np.percentile(arr,10):.2f} | ${np.percentile(arr,25):.2f} | "
                  f"${np.percentile(arr,50):.2f} | ${np.percentile(arr,75):.2f} | "
                  f"${np.percentile(arr,90):.2f} | ${arr.mean():.2f} |")
    md.append("")

    # Same heatmap restricted to losers
    md.append("## Bar-by-bar running MAE percentiles — LOSERS ONLY\n")
    md.append("This tells us how losers' drawdowns evolve. If MAE reaches max early,")
    md.append("a time-based cut works. If MAE deepens monotonically, a path-asymmetry rule helps.\n")
    loser_indices = enriched.index[enriched["is_win"] == False].tolist()
    bar_loser_mae = {}
    for idx in loser_indices:
        for bi in range(MAX_BARS):
            v = enriched.iloc[idx].get(f"running_mae_b{bi:02d}")
            if pd.notna(v):
                bar_loser_mae.setdefault(bi, []).append(v)
    md.append("| Bar | N | p10 | p25 | p50 | p75 | p90 | mean |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bi in [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 59]:
        if bi not in bar_loser_mae or len(bar_loser_mae[bi]) < 20:
            continue
        arr = np.array(bar_loser_mae[bi])
        md.append(f"| {bi} | {len(arr)} | ${np.percentile(arr,10):.2f} | ${np.percentile(arr,25):.2f} | "
                  f"${np.percentile(arr,50):.2f} | ${np.percentile(arr,75):.2f} | "
                  f"${np.percentile(arr,90):.2f} | ${arr.mean():.2f} |")
    md.append("")

    # Same for winners
    md.append("## Bar-by-bar running MAE percentiles — WINNERS ONLY\n")
    md.append("This tells us how deep winners dip. Critical for SL placement: the SL must")
    md.append("be wider than the typical winner's worst dip, else it kills winners.\n")
    win_indices = enriched.index[enriched["is_win"] == True].tolist()
    bar_winner_mae = {}
    for idx in win_indices:
        for bi in range(MAX_BARS):
            v = enriched.iloc[idx].get(f"running_mae_b{bi:02d}")
            if pd.notna(v):
                bar_winner_mae.setdefault(bi, []).append(v)
    md.append("| Bar | N | p10 | p25 | p50 | p75 | p90 | mean |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bi in [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 59]:
        if bi not in bar_winner_mae or len(bar_winner_mae[bi]) < 20:
            continue
        arr = np.array(bar_winner_mae[bi])
        md.append(f"| {bi} | {len(arr)} | ${np.percentile(arr,10):.2f} | ${np.percentile(arr,25):.2f} | "
                  f"${np.percentile(arr,50):.2f} | ${np.percentile(arr,75):.2f} | "
                  f"${np.percentile(arr,90):.2f} | ${arr.mean():.2f} |")
    md.append("")

    # MFE-conditional cut sweep — the actual money question
    md.append("## MFE-conditional cut rule sweep\n")
    md.append("Rule: at bar N, if MFE so far is <= X, exit at bar N close. Otherwise let trade run.\n")
    md.append("The intent is to cut trades that haven't shown ANY upside by bar N (likely losers)")
    md.append("while preserving trades that DID show upside (potential winners).\n")
    md.append("| Bar N | MFE threshold | trades cut | winners cut | losers cut | sim PnL | Δ vs base |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    base_pnl = enriched["pnl"].sum()
    candidates = []
    for N in [3, 5, 7, 10, 15, 20]:
        for X in [0, 5, 10, 20, 30]:
            n_cut = 0; n_win_cut = 0; n_loss_cut = 0
            sim_pnl = 0.0
            for _, t in enriched.iterrows():
                bars_n = t.get("traj_n_bars", 0)
                if bars_n <= N:
                    sim_pnl += t["pnl"]; continue
                running_mfe_at_N = t.get(f"running_mfe_b{N:02d}", np.nan)
                close_pnl_at_N = t.get(f"close_pnl_b{N:02d}", np.nan)
                if pd.isna(running_mfe_at_N) or pd.isna(close_pnl_at_N):
                    sim_pnl += t["pnl"]; continue
                if running_mfe_at_N <= X:
                    # Cut at bar N's close
                    sim_pnl += close_pnl_at_N
                    n_cut += 1
                    if t["is_win"]: n_win_cut += 1
                    else: n_loss_cut += 1
                else:
                    sim_pnl += t["pnl"]
            delta = sim_pnl - base_pnl
            candidates.append({"N": N, "X": X, "cut": n_cut, "win_cut": n_win_cut, "loss_cut": n_loss_cut, "sim": sim_pnl, "delta": delta})
            md.append(f"| {N} | ${X} | {n_cut} | {n_win_cut} | {n_loss_cut} | ${sim_pnl:+,.2f} | ${delta:+,.2f} |")
    md.append("")
    best = max(candidates, key=lambda c: c["delta"])
    md.append(f"**Best combination**: bar N={best['N']}, MFE<=${best['X']}: cuts {best['cut']} trades "
              f"({best['win_cut']} winners + {best['loss_cut']} losers), sim PnL ${best['sim']:+,.2f}, "
              f"Δ ${best['delta']:+,.2f}")
    md.append("")

    out_md = Path(args.out_md_dir) / f"{stem}_trajectory_analysis.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Markdown report:        {out_md}")
    print(f"\nSummary:")
    print(f"  Trades reconstructed:   {n_with_traj}/{len(trades)}")
    print(f"  Best MFE-conditional cut: bar {best['N']}, MFE<=${best['X']}, Δ ${best['delta']:+,.2f} on base ${base_pnl:+,.2f}")


if __name__ == "__main__":
    main()
