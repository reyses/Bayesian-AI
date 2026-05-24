"""
nt8_strategyanalyzer_parser.py -- Parse NT8 Strategy Analyzer XML log files
into a tidy DataFrame for analysis. Pulls each run's parameters + summary
performance metrics. Defaults to v1.5-RC runs.

Schema per run:
  guid, date, strategy, from, to, instrument
  + parameter columns (RPoints, BleedThresholdZ, ...)
  + summary metrics (TotalNetProfit, TotalNumTrades, PercentProfitable, ...)

Usage:
    python tools/nt8_strategyanalyzer_parser.py
    python tools/nt8_strategyanalyzer_parser.py --pattern "ZigzagRunner_v15"
    python tools/nt8_strategyanalyzer_parser.py --since 2026-04-27
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


DEFAULT_LOG_DIR = Path(r"C:\Users\reyse\OneDrive\Documents\NinjaTrader 8\strategyanalyzerlogs")


def parse_summary(text: str) -> dict:
    """Parse the SummaryPerformancesSerialize string.
    Format: 'metric_name;all_value;long_value;short_value|...'
    We take the 'all_value' (first after metric name)."""
    out = {}
    if not text:
        return out
    for entry in text.split("|"):
        parts = entry.split(";")
        if len(parts) < 2:
            continue
        metric = parts[0]
        try:
            out[metric] = float(parts[1])
        except ValueError:
            out[metric] = parts[1]
    return out


def parse_xml(path: Path) -> dict | None:
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        return None
    root = tree.getroot()
    entry = root.find("StrategyAnalyzerGridEntry")
    if entry is None:
        return None

    out = {"file": path.name}

    # Top-level fields
    for f in ("Date", "From", "To", "Instrument", "Guid", "Action"):
        node = entry.find(f)
        if node is not None and node.text:
            out[f.lower()] = node.text

    # Strategy name from NinjaScriptFile path or from the file pattern
    nsfile = entry.find("NinjaScriptFile")
    if nsfile is not None and nsfile.text:
        m = re.search(r"@@@(.+?)_\d{4}_\d{2}_\d{2}_\d+\.cs$", nsfile.text)
        if m:
            out["strategy"] = m.group(1)
    if "strategy" not in out:
        m = re.match(r"@@@(.+?)_\d{4}_\d{2}_\d{2}_\d+\.xml$", path.name)
        if m:
            out["strategy"] = m.group(1)

    # OptimizationParameters (NOTE: these store the LAST sweep value, not necessarily
    # the run-time value. The authoritative source is the inner StrategyTemplate.)
    # We pull both, then prefer StrategyTemplate where they differ.
    opt = entry.find("OptimizationParameters")
    if opt is not None:
        for param in opt.findall("Parameter"):
            name = param.findtext("Name")
            val = param.findtext("ValueSerializable")
            if name and val is not None:
                try:
                    if "." in val or "e" in val.lower():
                        out[f"p_{name}"] = float(val)
                    else:
                        out[f"p_{name}"] = int(val)
                except ValueError:
                    out[f"p_{name}"] = val

    # Inner StrategyTemplate — HTML-encoded XML containing the ACTUAL run-time
    # property values. This is the authoritative source for the run.
    tmpl_node = entry.find("StrategyTemplate")
    if tmpl_node is not None and tmpl_node.text:
        tmpl_decoded = tmpl_node.text  # already decoded by ElementTree
        prop_re = re.compile(r"<(\w+)>([^<>]+)</\1>")
        for m in prop_re.finditer(tmpl_decoded):
            name, val = m.group(1), m.group(2)
            # Only override params we already have from OptimizationParameters
            # (avoids polluting with hundreds of internal NT8 props)
            key = f"p_{name}"
            if key in out:
                try:
                    if "." in val or "e" in val.lower():
                        out[key] = float(val)
                    else:
                        out[key] = int(val)
                except ValueError:
                    out[key] = val

    # SummaryPerformances — first one is in points/dollars (Currency mode)
    perfs = entry.findall(".//SummaryPerformancesSerialize")
    if perfs and perfs[0].text:
        summary = parse_summary(perfs[0].text)
        for k, v in summary.items():
            out[f"m_{k}"] = v

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=str(DEFAULT_LOG_DIR))
    ap.add_argument("--pattern", default="ZigzagRunner_v15",
                    help="Substring to match in filename (default: ZigzagRunner_v15)")
    ap.add_argument("--since", default=None,
                    help="Only include runs after this date (YYYY-MM-DD)")
    ap.add_argument("--out-csv", default="reports/findings/2026-04-27_v15_strategyanalyzer_runs.csv")
    ap.add_argument("--out-md", default="reports/findings/2026-04-27_v15_strategyanalyzer_runs.md")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"Log dir not found: {logdir}")
        sys.exit(1)

    files = sorted(logdir.glob(f"@@@*{args.pattern}*.xml"))
    print(f"Found {len(files)} XML files matching '{args.pattern}'")
    if args.since:
        cutoff = datetime.strptime(args.since, "%Y-%m-%d")
        files = [f for f in files if datetime.fromtimestamp(f.stat().st_mtime) >= cutoff]
        print(f"After --since {args.since}: {len(files)} files")

    rows = []
    for f in files:
        rec = parse_xml(f)
        if rec:
            rows.append(rec)

    if not rows:
        print("No parseable rows.")
        return

    df = pd.DataFrame(rows)
    # Sort by Date
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # Output CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nCSV: {args.out_csv}")

    # Print key columns
    key_cols = ["file", "from", "to"]
    p_cols = sorted([c for c in df.columns if c.startswith("p_")])
    metric_cols = ["m_TotalNetProfit", "m_TotalNumTrades", "m_PercentProfitable",
                   "m_ProfitFactor", "m_AverageTrade", "m_MaxDrawdown",
                   "m_AverageNumTradesPerDay", "m_SharpeRatio"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    show = key_cols + p_cols + metric_cols

    print(f"\nKey columns: {show}\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}" if isinstance(x, float) else str(x))
    print(df[show].to_string(index=False))

    # Markdown summary
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(f"# v1.5-RC Strategy Analyzer Runs — {datetime.now():%Y-%m-%d %H:%M}\n\n")
            f.write(f"Source: `{logdir}` matching `{args.pattern}`\n\n")
            f.write(f"Total runs: {len(df)}\n\n")
            f.write("## Runs (sorted by Date)\n\n")
            # Build markdown table
            cols = show
            f.write("| " + " | ".join(c.replace("p_", "").replace("m_", "") for c in cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
            for _, r in df.iterrows():
                vals = []
                for c in cols:
                    v = r.get(c, "")
                    if isinstance(v, float):
                        vals.append(f"{v:.2f}")
                    else:
                        vals.append(str(v) if v else "")
                f.write("| " + " | ".join(vals) + " |\n")
        print(f"\nReport: {args.out_md}")


if __name__ == "__main__":
    main()
