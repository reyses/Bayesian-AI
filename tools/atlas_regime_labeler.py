"""
atlas_regime_labeler.py -- Labels each session day in DATA/ATLAS as
UP / DOWN / CHOP / QUIET / TRANSITIONAL based on daily OHLC metrics, and
produces per-regime test datasets so strategies can be tested against
specific market scenarios in isolation.

WHY THIS EXISTS: aggregate backtests over all 14 months MIX regimes,
which hides regime-specific behavior. Counter-trend strategies bleed in
trending days, win in chop. To diagnose and fix that, we need to test
the strategy ON EACH REGIME ALONE and see where it lives or dies.

LABELING SCHEME (daily, in user's local time = America/Los_Angeles):

  1. daily_range        = high - low                 (raw price range)
  2. net_move           = close - open               (signed daily move)
  3. directional_strength = |net_move| / daily_range (0=chop, 1=pure trend)
  4. efficiency_ratio   = |net_move| / sum(|bar_diff|) (Kaufman's ER)
  5. range_expansion    = daily_range / 20d_avg_range (>1 = expanding)

CLASSIFICATION (in priority order):

  QUIET         range_expansion < 0.5                     (small range day)
  UP            directional_strength >= dir_threshold AND range_expansion >= exp_threshold AND net_move > 0
  DOWN          directional_strength >= dir_threshold AND range_expansion >= exp_threshold AND net_move < 0
  CHOP          directional_strength < chop_threshold     (lots of churn, no net)
  TRANSITIONAL  everything else                           (ambiguous)

Default thresholds: dir_threshold=0.6, chop_threshold=0.4, exp_threshold=1.0.
All tunable via CLI.

OUTPUTS:

  DATA/ATLAS/regime_labels.csv              -- one row per session day with all metrics
  DATA/ATLAS/scenarios/UP_bars.parquet      -- all 1m bars from UP-labeled days
  DATA/ATLAS/scenarios/DOWN_bars.parquet
  DATA/ATLAS/scenarios/CHOP_bars.parquet
  DATA/ATLAS/scenarios/QUIET_bars.parquet
  DATA/ATLAS/scenarios/TRANSITIONAL_bars.parquet
  reports/findings/regime_eda/<date>_regime_summary.md
  reports/findings/regime_eda/<date>_regime_timeline.png
  reports/findings/regime_eda/<date>_regime_distribution.png

USAGE FROM OTHER TOOLS:

    from tools.atlas_regime_labeler import load_scenario
    bars_up = load_scenario("UP")
    bars_chop = load_scenario("CHOP")
    bars_all = load_scenario("ALL")  # everything
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tools.zigzag_genetic import load_1m_bars

REGIME_COLORS = {
    "UP":           "#22c55e",  # green
    "DOWN":         "#ef4444",  # red
    "CHOP":         "#f59e0b",  # amber
    "QUIET":        "#94a3b8",  # slate
    "TRANSITIONAL": "#a78bfa",  # violet
}
REGIME_ORDER = ["UP", "DOWN", "CHOP", "QUIET", "TRANSITIONAL"]


# =============================================================================
# Daily metrics + classification
# =============================================================================

def compute_daily_metrics(bars: pd.DataFrame, tz: str = "America/Los_Angeles",
                            atr_window: int = 20) -> pd.DataFrame:
    """Aggregate 1m bars into daily OHLC + derived metrics.

    Session day = calendar day in `tz` (default Pacific, matching v1.0.4 EOD).
    Returns DataFrame indexed by date with columns:
      open, close, high, low, n_bars, range, net_move,
      directional_strength, efficiency_ratio, atr_20d, range_expansion
    """
    if bars.empty:
        raise ValueError("no bars provided")
    work = bars.copy()
    work["date"] = work["dt_utc"].dt.tz_convert(tz).dt.date

    # OHLC daily
    daily = (
        work.groupby("date")
        .agg(
            open=("open", "first"),
            close=("close", "last"),
            high=("high", "max"),
            low=("low", "min"),
            n_bars=("close", "count"),
            first_dt=("dt_utc", "first"),
            last_dt=("dt_utc", "last"),
        )
        .reset_index()
    )

    # Efficiency ratio: |net move| / sum(|bar diff|)
    # Aggregate the per-bar |close diff| sum per day
    work["close_prev"] = work["close"].shift(1)
    # Make diff zero across day boundaries (so we don't compute across midnight)
    work["date_prev"] = work["date"].shift(1)
    work["bar_diff"] = (work["close"] - work["close_prev"]).abs()
    work.loc[work["date"] != work["date_prev"], "bar_diff"] = 0.0
    er = work.groupby("date")["bar_diff"].sum().rename("close_diff_sum")
    daily = daily.merge(er.reset_index(), on="date", how="left")

    daily["range"] = daily["high"] - daily["low"]
    daily["net_move"] = daily["close"] - daily["open"]
    daily["directional_strength"] = (
        daily["net_move"].abs() / daily["range"].replace(0, np.nan)
    )
    daily["efficiency_ratio"] = (
        daily["net_move"].abs()
        / daily["close_diff_sum"].replace(0, np.nan)
    )

    # Rolling N-day ATR (use range as ATR proxy — close enough for daily classification)
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["atr_20d"] = daily["range"].rolling(atr_window, min_periods=5).mean()
    daily["range_expansion"] = daily["range"] / daily["atr_20d"]

    return daily


def classify_regime(row, dir_threshold: float = 0.6,
                      chop_threshold: float = 0.4,
                      exp_threshold: float = 1.0,
                      quiet_threshold: float = 0.5) -> str:
    """Classify one daily row into a regime label."""
    ds = row["directional_strength"]
    re_ = row["range_expansion"]
    nm = row["net_move"]
    if pd.isna(ds) or pd.isna(re_):
        return "UNKNOWN"
    # QUIET: tiny range relative to recent vol
    if re_ < quiet_threshold:
        return "QUIET"
    # Strong directional with expansion
    if ds >= dir_threshold and re_ >= exp_threshold:
        return "UP" if nm > 0 else "DOWN"
    # Pure chop: low directional strength regardless of expansion
    if ds < chop_threshold:
        return "CHOP"
    # Ambiguous (medium directional, normal range)
    return "TRANSITIONAL"


def label_regimes(bars: pd.DataFrame,
                    dir_threshold: float = 0.6,
                    chop_threshold: float = 0.4,
                    exp_threshold: float = 1.0,
                    quiet_threshold: float = 0.5,
                    atr_window: int = 20,
                    tz: str = "America/Los_Angeles") -> pd.DataFrame:
    """Compute daily metrics + regime labels."""
    daily = compute_daily_metrics(bars, tz=tz, atr_window=atr_window)
    daily["regime"] = daily.apply(
        classify_regime,
        axis=1,
        dir_threshold=dir_threshold,
        chop_threshold=chop_threshold,
        exp_threshold=exp_threshold,
        quiet_threshold=quiet_threshold,
    )
    return daily


# =============================================================================
# Scenario extraction
# =============================================================================

def write_scenario_parquets(bars: pd.DataFrame, daily: pd.DataFrame,
                              out_dir: str, tz: str = "America/Los_Angeles"):
    """For each regime, write a parquet of all 1m bars from days with that label."""
    bars = bars.copy()
    bars["date"] = bars["dt_utc"].dt.tz_convert(tz).dt.date
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for regime in REGIME_ORDER + ["UNKNOWN"]:
        days = set(daily[daily["regime"] == regime]["date"].tolist())
        if not days:
            continue
        subset = bars[bars["date"].isin(days)].drop(columns=["date"]).reset_index(drop=True)
        out_path = os.path.join(out_dir, f"{regime}_bars.parquet")
        subset.to_parquet(out_path)
        rows.append({"regime": regime, "n_days": len(days), "n_bars": len(subset),
                     "path": out_path})
        print(f"  {regime:>14s}: {len(days):>4d} days, {len(subset):>9,} 1m bars -> {out_path}")
    return pd.DataFrame(rows)


# =============================================================================
# Visualization
# =============================================================================

def plot_distribution(daily: pd.DataFrame, out_png: str):
    """Bar chart of regime distribution."""
    counts = daily["regime"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    regimes = [r for r in REGIME_ORDER if r in counts.index]
    if "UNKNOWN" in counts.index:
        regimes.append("UNKNOWN")
    values = [counts[r] for r in regimes]
    colors = [REGIME_COLORS.get(r, "#888") for r in regimes]
    bars = ax.bar(regimes, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, str(v),
                ha="center", va="bottom", fontsize=10, color="white")
    ax.set_ylabel("Days", color="#ccc")
    ax.set_title(f"Regime distribution  (N={len(daily)} session days)",
                  color="white")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#0a0a0a")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="#aaa")
    plt.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor="#0a0a0a")
    plt.close(fig)


def plot_timeline(daily: pd.DataFrame, out_png: str):
    """Timeline showing regime per day, color-coded."""
    daily = daily.copy()
    daily["date_ts"] = pd.to_datetime(daily["date"])
    daily["color"] = daily["regime"].map(REGIME_COLORS).fillna("#666")

    fig, (ax_r, ax_p) = plt.subplots(2, 1, figsize=(16, 7),
                                        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.05},
                                        sharex=True)
    # Top: regime strip
    for regime in REGIME_ORDER:
        sub = daily[daily["regime"] == regime]
        if sub.empty:
            continue
        ax_r.scatter(sub["date_ts"], [0]*len(sub), color=REGIME_COLORS[regime],
                      s=60, label=regime, marker="s")
    ax_r.set_yticks([])
    ax_r.legend(loc="upper right", fontsize=8, ncol=5,
                 bbox_to_anchor=(1, 1.6), facecolor="#1a1a1a", labelcolor="white")
    ax_r.set_facecolor("#1a1a1a")
    ax_r.set_title(f"Daily regime labels  (N={len(daily)} days)", color="white")

    # Bottom: net move per day, colored by regime
    ax_p.bar(daily["date_ts"], daily["net_move"], color=daily["color"], width=0.9)
    ax_p.axhline(0, color="white", alpha=0.3, lw=0.5)
    ax_p.set_ylabel("Net move (close - open, pts)", color="#ccc")
    ax_p.set_xlabel("Session day", color="#ccc")
    ax_p.set_facecolor("#1a1a1a")
    ax_p.xaxis.set_major_locator(mdates.MonthLocator())
    ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_p.get_xticklabels(), rotation=45, ha="right")

    fig.patch.set_facecolor("#0a0a0a")
    for ax in (ax_r, ax_p):
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.tick_params(colors="#aaa")
    plt.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor="#0a0a0a", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Markdown summary report
# =============================================================================

def write_summary(daily: pd.DataFrame, out_md: str, args):
    counts = daily["regime"].value_counts().to_dict()
    total = len(daily)
    by_regime = (
        daily.groupby("regime")
        .agg(
            n_days=("date", "count"),
            avg_net_move=("net_move", "mean"),
            median_range=("range", "median"),
            avg_dir_strength=("directional_strength", "mean"),
            avg_eff_ratio=("efficiency_ratio", "mean"),
            avg_range_exp=("range_expansion", "mean"),
        )
        .reset_index()
    )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Daily regime labels — DATA/ATLAS\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Settings\n\n")
        f.write(f"- Source: `{args.atlas}`\n")
        f.write(f"- Timezone: `{args.tz}` (session day = calendar day in this tz)\n")
        f.write(f"- ATR window: {args.atr_window} days\n")
        f.write(f"- Thresholds: directional={args.dir_threshold}, chop={args.chop_threshold}, "
                  f"expansion={args.exp_threshold}, quiet={args.quiet_threshold}\n\n")

        f.write(f"## Distribution\n\n")
        f.write(f"Total session days: {total}\n\n")
        f.write("| Regime | Days | % |\n|---|---:|---:|\n")
        for r in REGIME_ORDER + ["UNKNOWN"]:
            n = counts.get(r, 0)
            if n == 0:
                continue
            pct = 100.0 * n / total
            f.write(f"| {r} | {n} | {pct:.1f}% |\n")
        f.write("\n")

        f.write(f"## Per-regime metrics\n\n")
        f.write("```\n")
        f.write(by_regime.to_string(index=False))
        f.write("\n```\n\n")

        # Sample dates per regime (first 5)
        f.write(f"## Sample days per regime\n\n")
        for r in REGIME_ORDER:
            samples = daily[daily["regime"] == r].head(5)
            if samples.empty:
                continue
            f.write(f"### {r}\n\n")
            f.write("| Date | Open | Close | Net | Range | DirStr | Eff | RangeExp |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for _, row in samples.iterrows():
                f.write(f"| {row['date']} | {row['open']:.2f} | {row['close']:.2f} | "
                          f"{row['net_move']:+.2f} | {row['range']:.2f} | "
                          f"{row['directional_strength']:.2f} | "
                          f"{row['efficiency_ratio']:.2f} | "
                          f"{row['range_expansion']:.2f} |\n")
            f.write("\n")

        f.write(f"## How to use\n\n")
        f.write("```python\n")
        f.write("from tools.atlas_regime_labeler import load_scenario\n")
        f.write("bars_up = load_scenario('UP')      # 1m bars from UP-labeled days only\n")
        f.write("bars_chop = load_scenario('CHOP')  # 1m bars from CHOP-labeled days\n")
        f.write("bars_all = load_scenario('ALL')    # everything\n")
        f.write("```\n\n")
        f.write("Per-regime parquets are at `DATA/ATLAS/scenarios/<REGIME>_bars.parquet`.\n")


# =============================================================================
# Public loader (for use from other tools)
# =============================================================================

def load_scenario(scenario: str, atlas_root: str = "DATA/ATLAS") -> pd.DataFrame:
    """Load 1m bars filtered to days matching the given regime label.

    scenario: one of 'UP', 'DOWN', 'CHOP', 'QUIET', 'TRANSITIONAL', 'ALL', or
              comma-separated list like 'UP,DOWN' for a union.
    Returns DataFrame with same schema as load_1m_bars (timestamp, dt_utc, OHLCV, etc.).
    """
    scenario = scenario.upper().strip()
    if scenario == "ALL":
        return load_1m_bars(atlas_root)

    scenario_dir = Path(atlas_root) / "scenarios"
    if not scenario_dir.exists():
        raise FileNotFoundError(
            f"{scenario_dir} not found. Run: python tools/atlas_regime_labeler.py first."
        )

    parts = []
    for r in scenario.split(","):
        r = r.strip()
        path = scenario_dir / f"{r}_bars.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Re-run labeler.")
        parts.append(pd.read_parquet(path))
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if "dt_utc" not in out.columns:
        out["dt_utc"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    if "mins_of_day_utc" not in out.columns:
        out["mins_of_day_utc"] = out["dt_utc"].dt.hour * 60 + out["dt_utc"].dt.minute
    return out


def load_regime_labels(atlas_root: str = "DATA/ATLAS") -> pd.DataFrame:
    """Load the daily regime labels CSV."""
    path = Path(atlas_root) / "regime_labels.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python tools/atlas_regime_labeler.py first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="DATA/ATLAS",
                    help="ATLAS root directory")
    ap.add_argument("--tz", default="America/Los_Angeles",
                    help="Timezone for session day boundary")
    ap.add_argument("--atr-window", type=int, default=20,
                    help="Rolling window for ATR baseline (in days)")
    ap.add_argument("--dir-threshold", type=float, default=0.6,
                    help="Min directional_strength for UP/DOWN")
    ap.add_argument("--chop-threshold", type=float, default=0.4,
                    help="Max directional_strength for CHOP")
    ap.add_argument("--exp-threshold", type=float, default=1.0,
                    help="Min range_expansion for UP/DOWN (0.5 = half of 20d ATR)")
    ap.add_argument("--quiet-threshold", type=float, default=0.5,
                    help="Below this range_expansion = QUIET")
    ap.add_argument("--no-charts", action="store_true",
                    help="Skip PNG generation")
    ap.add_argument("--no-parquets", action="store_true",
                    help="Skip per-scenario parquet generation")
    ap.add_argument("--out-report-dir", default="reports/findings/regime_eda")
    args = ap.parse_args()

    print("=" * 80)
    print("ATLAS REGIME LABELER")
    print("=" * 80)
    print(f"Source: {args.atlas}")
    print(f"TZ:     {args.tz}")
    print(f"Thresholds: dir={args.dir_threshold}, chop={args.chop_threshold}, "
           f"exp={args.exp_threshold}, quiet={args.quiet_threshold}")
    print()

    print("Loading 1m bars...")
    t0 = time.time()
    bars = load_1m_bars(args.atlas)
    print(f"  {len(bars):,} bars  ({bars['dt_utc'].iloc[0]} -> {bars['dt_utc'].iloc[-1]})  "
           f"in {time.time()-t0:.1f}s")
    print()

    print("Computing daily metrics + regime labels...")
    t0 = time.time()
    daily = label_regimes(
        bars,
        dir_threshold=args.dir_threshold,
        chop_threshold=args.chop_threshold,
        exp_threshold=args.exp_threshold,
        quiet_threshold=args.quiet_threshold,
        atr_window=args.atr_window,
        tz=args.tz,
    )
    print(f"  {len(daily)} session days labeled in {time.time()-t0:.1f}s")
    counts = daily["regime"].value_counts()
    print()
    print("Distribution:")
    total = len(daily)
    for r in REGIME_ORDER + ["UNKNOWN"]:
        n = counts.get(r, 0)
        if n > 0:
            print(f"  {r:>14s}: {n:>4d} days  ({100.0*n/total:.1f}%)")
    print()

    # Save labels CSV
    csv_path = Path(args.atlas) / "regime_labels.csv"
    daily.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")

    # Per-scenario parquets
    if not args.no_parquets:
        print("\nWriting per-regime scenario parquets...")
        scen_dir = Path(args.atlas) / "scenarios"
        write_scenario_parquets(bars, daily, str(scen_dir), tz=args.tz)
        print()

    # Visualizations
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(args.out_report_dir, exist_ok=True)
    if not args.no_charts:
        print("Generating charts...")
        dist_png = f"{args.out_report_dir}/{today}_regime_distribution.png"
        plot_distribution(daily, dist_png)
        print(f"  Wrote: {dist_png}")
        timeline_png = f"{args.out_report_dir}/{today}_regime_timeline.png"
        plot_timeline(daily, timeline_png)
        print(f"  Wrote: {timeline_png}")

    # Markdown report
    md_path = f"{args.out_report_dir}/{today}_regime_summary.md"
    write_summary(daily, md_path, args)
    print(f"\nWrote: {md_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("Other tools can load specific scenarios via:")
    print("    from tools.atlas_regime_labeler import load_scenario")
    print("    bars_up = load_scenario('UP')")
    print()


if __name__ == "__main__":
    main()
