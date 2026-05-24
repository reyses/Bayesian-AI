"""
atlas_nt8_rebuild.py -- Rebuild DATA/ATLAS_NT8 parquet files from the
NT8-exported aggregated dumps. Produces a parallel ATLAS_NT8 dataset that
mirrors the canonical DATA/ATLAS schema, so all existing tools work by
swapping --atlas DATA/ATLAS -> DATA/ATLAS_NT8.

INPUT (NT8 chart-export CSVs):
    DATA/ATLAS_NT8/{1s,1m,1h,1D}/MNQ_06-26/YYYY_MM_DD.csv
    schema: timestamp,open,high,low,close,volume  (UTF-8 BOM)

OUTPUT (parquet, flat, ATLAS-style):
    DATA/ATLAS_NT8/{1s,5s,15s,30s,1m,5m,15m,30m,1h,1D}/YYYY_MM_DD.parquet
    schema: timestamp,open,high,low,close,volume  (same as canonical ATLAS)

AGGREGATION:
    Missing TFs (5s/15s/30s/5m/15m/30m) are computed from 1s using
    `floor(timestamp / N) * N` bin alignment. open=first, high=max,
    low=min, close=last, volume=sum, timestamp=bin_end.

VALIDATION:
    Re-aggregates 1m from 1s and compares to NT8-provided 1m. Reports
    bar-count parity, OHLC mean-abs-error, and worst-bar deviation.

Usage:
    python tools/atlas_nt8_rebuild.py
    python tools/atlas_nt8_rebuild.py --root DATA/ATLAS_NT8 --contract MNQ_06-26
    python tools/atlas_nt8_rebuild.py --no-validate    # skip parity check
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm


TF_SECONDS = {
    "1s":  1,
    "5s":  5,
    "15s": 15,
    "30s": 30,
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
    "4h":  14400,   # required by core_v2/features.py TF_ORDER
    "1D":  86400,
}

# TFs that come straight from NT8 dumps (we trust these as ground truth)
NATIVE_TFS = ["1s", "1m", "1h", "1D"]
# TFs we synthesise from 1s
SYNTH_TFS  = ["5s", "15s", "30s", "5m", "15m", "30m", "4h"]


def read_nt8_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read an NT8 export CSV (with UTF-8 BOM). Returns None on failure."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        print(f"  WARN read fail {path.name}: {e}")
        return None
    expected = ["timestamp", "open", "high", "low", "close", "volume"]
    if list(df.columns) != expected:
        print(f"  WARN bad schema {path.name}: {list(df.columns)}")
        return None
    if df.empty:
        return None
    df["timestamp"] = df["timestamp"].astype("int64")
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float64")
    df["volume"] = df["volume"].astype("int64")
    return df.reset_index(drop=True)


def aggregate_1s_to(df_1s: pd.DataFrame, n_seconds: int) -> pd.DataFrame:
    """Aggregate 1s OHLCV to N-second bars using bin floor alignment.
    Output timestamp = bin_end (matches NT8 native convention)."""
    if df_1s.empty:
        return df_1s
    ts = df_1s["timestamp"].to_numpy()
    bin_start = (ts // n_seconds) * n_seconds
    df = df_1s.copy()
    df["bin"] = bin_start
    grp = df.groupby("bin", sort=True)
    out = pd.DataFrame({
        "timestamp": grp["bin"].first().to_numpy() + n_seconds - 1,
        "open":      grp["open"].first().to_numpy(),
        "high":      grp["high"].max().to_numpy(),
        "low":       grp["low"].min().to_numpy(),
        "close":     grp["close"].last().to_numpy(),
        "volume":    grp["volume"].sum().to_numpy(),
    })
    return out.reset_index(drop=True)


def list_input_dates(root: Path, contract: str, tf: str) -> list[str]:
    folder = root / tf / contract
    if not folder.exists():
        return []
    return sorted([p.stem for p in folder.glob("*.csv")])


def csv_path(root: Path, contract: str, tf: str, date: str) -> Path:
    return root / tf / contract / f"{date}.csv"


def parquet_path(root: Path, tf: str, date: str) -> Path:
    return root / tf / f"{date}.parquet"


def write_parquet(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def parity_compare(synth_1m: pd.DataFrame, native_1m: pd.DataFrame) -> dict:
    """Compare re-aggregated 1m vs NT8-native 1m. Return parity stats."""
    if synth_1m.empty or native_1m.empty:
        return {"status": "EMPTY", "n_synth": len(synth_1m), "n_native": len(native_1m)}

    # Align by timestamp
    merged = synth_1m.merge(native_1m, on="timestamp", suffixes=("_synth", "_nat"), how="outer", indicator=True)
    n_match = (merged["_merge"] == "both").sum()
    n_synth_only = (merged["_merge"] == "left_only").sum()
    n_nat_only = (merged["_merge"] == "right_only").sum()

    matched = merged[merged["_merge"] == "both"]
    if len(matched) == 0:
        return {"status": "NO_OVERLAP", "n_synth": len(synth_1m), "n_native": len(native_1m)}

    err_open  = (matched["open_synth"]  - matched["open_nat"]).abs().mean()
    err_high  = (matched["high_synth"]  - matched["high_nat"]).abs().mean()
    err_low   = (matched["low_synth"]   - matched["low_nat"]).abs().mean()
    err_close = (matched["close_synth"] - matched["close_nat"]).abs().mean()
    err_vol   = (matched["volume_synth"] - matched["volume_nat"]).abs().mean()
    worst_close = (matched["close_synth"] - matched["close_nat"]).abs().max()

    return {
        "status": "OK" if (n_synth_only == 0 and n_nat_only == 0 and worst_close < 0.5) else "DRIFT",
        "n_synth": len(synth_1m),
        "n_native": len(native_1m),
        "n_match": int(n_match),
        "n_synth_only": int(n_synth_only),
        "n_nat_only": int(n_nat_only),
        "mae_open": float(err_open),
        "mae_high": float(err_high),
        "mae_low": float(err_low),
        "mae_close": float(err_close),
        "mae_volume": float(err_vol),
        "worst_close_dev": float(worst_close),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="DATA/ATLAS_NT8",
                    help="Root of NT8 atlas (default: DATA/ATLAS_NT8)")
    ap.add_argument("--contract", default="MNQ_06-26",
                    help="Contract subfolder name in input CSV layout")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip 1m re-aggregation parity check")
    ap.add_argument("--out-md", default="reports/findings/2026-04-27_atlas_nt8_rebuild.md")
    args = ap.parse_args()

    root = Path(args.root)
    print(f"=" * 80)
    print(f"ATLAS_NT8 REBUILD")
    print(f"=" * 80)
    print(f"Root: {root}")
    print(f"Contract: {args.contract}")

    # Determine all dates that have 1s data (this is the universe we can synth from)
    dates_1s = list_input_dates(root, args.contract, "1s")
    if not dates_1s:
        print(f"FATAL: no 1s CSVs found under {root}/1s/{args.contract}/")
        return
    print(f"\n1s coverage: {len(dates_1s)} days  ({dates_1s[0]} -> {dates_1s[-1]})")
    coverage = {tf: list_input_dates(root, args.contract, tf) for tf in NATIVE_TFS}
    for tf in NATIVE_TFS:
        print(f"  {tf:>3} native dumps: {len(coverage[tf])} days")

    # ── Step 1: Convert native CSVs (1s/1m/1h/1D) directly to parquet ────────
    print(f"\n[1/3] Converting native NT8 CSVs to parquet...")
    converted = {tf: 0 for tf in NATIVE_TFS}
    for tf in NATIVE_TFS:
        for d in tqdm(coverage[tf], desc=f"  {tf}", leave=False):
            df = read_nt8_csv(csv_path(root, args.contract, tf, d))
            if df is None or df.empty:
                continue
            write_parquet(df, parquet_path(root, tf, d))
            converted[tf] += 1
        print(f"  {tf:>3} -> {converted[tf]} parquet files")

    # ── Step 2: Aggregate missing TFs from 1s ────────────────────────────────
    print(f"\n[2/3] Aggregating missing TFs from 1s...")
    synth_counts = {tf: 0 for tf in SYNTH_TFS}
    parity_per_date = []  # for 1m re-agg validation
    for d in tqdm(dates_1s, desc="  days"):
        df_1s = read_nt8_csv(csv_path(root, args.contract, "1s", d))
        if df_1s is None or df_1s.empty:
            continue
        for tf in SYNTH_TFS:
            df_tf = aggregate_1s_to(df_1s, TF_SECONDS[tf])
            if not df_tf.empty:
                write_parquet(df_tf, parquet_path(root, tf, d))
                synth_counts[tf] += 1

        # Validation: also re-aggregate 1m and compare
        if not args.no_validate and d in coverage["1m"]:
            df_1m_synth = aggregate_1s_to(df_1s, 60)
            df_1m_native = read_nt8_csv(csv_path(root, args.contract, "1m", d))
            if df_1m_native is not None:
                stats = parity_compare(df_1m_synth, df_1m_native)
                stats["date"] = d
                parity_per_date.append(stats)

    for tf in SYNTH_TFS:
        print(f"  {tf:>3} -> {synth_counts[tf]} parquet files")

    # ── Step 3: Validation report ────────────────────────────────────────────
    print(f"\n[3/3] Validation: 1m re-aggregation parity vs NT8 native")
    if parity_per_date:
        ok_count = sum(1 for s in parity_per_date if s["status"] == "OK")
        drift_count = sum(1 for s in parity_per_date if s["status"] == "DRIFT")
        worst = max((s for s in parity_per_date if "worst_close_dev" in s),
                    key=lambda s: s["worst_close_dev"], default=None)
        print(f"  Days validated: {len(parity_per_date)}")
        print(f"    OK:    {ok_count}")
        print(f"    DRIFT: {drift_count}")
        if parity_per_date:
            mae_close_avg = np.mean([s.get("mae_close", 0.0) for s in parity_per_date])
            mae_vol_avg = np.mean([s.get("mae_volume", 0.0) for s in parity_per_date])
            print(f"  Avg close MAE:  {mae_close_avg:.4f} pts")
            print(f"  Avg volume MAE: {mae_vol_avg:.2f} contracts")
        if worst:
            print(f"  Worst day: {worst['date']} -> close MAE {worst['mae_close']:.4f}, "
                  f"worst-bar dev {worst['worst_close_dev']:.4f}")

    # ── Final coverage summary ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"FINAL PARQUET COVERAGE")
    print(f"{'='*80}")
    for tf in ["1s", "5s", "15s", "30s", "1m", "5m", "15m", "30m", "1h", "4h", "1D"]:
        files = sorted((root / tf).glob("*.parquet")) if (root / tf).exists() else []
        if files:
            dates = [f.stem for f in files]
            print(f"  {tf:>3}: {len(files)} parquet files  ({dates[0]} -> {dates[-1]})")
        else:
            print(f"  {tf:>3}: 0 parquet files")

    # ── Markdown report ──────────────────────────────────────────────────────
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("# ATLAS_NT8 rebuild report\n\n")
            f.write(f"Generated: {datetime.now().isoformat(timespec='minutes')}\n\n")
            f.write(f"Root: `{root}` -- Contract: `{args.contract}`\n\n")
            f.write("## Coverage\n\n")
            f.write("| TF | Native CSVs | Parquet files |\n|---|---:|---:|\n")
            for tf in ["1s", "5s", "15s", "30s", "1m", "5m", "15m", "30m", "1h", "4h", "1D"]:
                native_n = len(coverage.get(tf, []))
                pq_files = sorted((root / tf).glob("*.parquet")) if (root / tf).exists() else []
                f.write(f"| {tf} | {native_n if tf in NATIVE_TFS else '(synth)'} | {len(pq_files)} |\n")

            f.write("\n## 1m re-aggregation parity\n\n")
            if parity_per_date:
                f.write(f"- Days validated: {len(parity_per_date)}\n")
                f.write(f"- OK:    {sum(1 for s in parity_per_date if s['status']=='OK')}\n")
                f.write(f"- DRIFT: {sum(1 for s in parity_per_date if s['status']=='DRIFT')}\n\n")
                f.write("| date | n_synth | n_native | match | mae_close | mae_volume | worst_close_dev | status |\n")
                f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
                for s in parity_per_date:
                    f.write(f"| {s['date']} | {s.get('n_synth',0)} | {s.get('n_native',0)} | "
                            f"{s.get('n_match',0)} | {s.get('mae_close',0):.4f} | "
                            f"{s.get('mae_volume',0):.2f} | {s.get('worst_close_dev',0):.4f} | "
                            f"{s['status']} |\n")
            else:
                f.write("(validation skipped)\n")
        print(f"\nReport: {args.out_md}")


if __name__ == "__main__":
    main()
