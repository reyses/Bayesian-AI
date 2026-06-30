"""EVOLVING ENVIRONMENT: how the oscillation-period distribution transitions over time.

Not two static year-buckets — a SLIDING WINDOW across the full 2024->2025 timeline. For each
window we pool the return-to-zero (one-oscillation) samples and look at the whole distribution:
a heatmap (time x period-bin density) with the median traced on top, plus the never-recover rate.

The literal mode is pinned at the shortest bin (the period dist is monotonic-decreasing), so the
transition signal is the MEDIAN and the TAIL weight — the heatmap shows both directly.

Reuses opportunity_cost.process_day (same seeded random entries) for consistency.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import opportunity_cost as oc  # noqa: E402

ROOT = oc.ROOT
ONE_M = oc.ONE_M
REPDIR = os.path.join(ROOT, "research", "recovery_dynamics", "reports")
ASSETS = os.path.join(REPDIR, "assets")

WIN_DAYS = 21        # sliding window length (~1 trading month)
STEP_DAYS = 5        # window step (~weekly)
PERIOD_CAP = 180     # heatmap y-axis cap (min); longer pooled into the overflow row
BIN = 5              # period bin width (min)


def spark(vals):
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    return "".join(blocks[min(int((v - lo) / rng * (len(blocks) - 1)), len(blocks) - 1)] for v in vals)


def main():
    rng = np.random.default_rng(oc.SEED)
    files = sorted(glob.glob(os.path.join(ONE_M, "2024_*.parquet")) +
                   glob.glob(os.path.join(ONE_M, "2025_*.parquet")))
    days = []   # (label 'YYYY-MM', periods list, never, n)
    for f in tqdm(files, desc="days", unit="day"):
        try:
            close = pd.read_parquet(f)["close"].to_numpy(np.float64)
        except Exception:
            continue
        r = oc.process_day(close, rng)
        base = os.path.basename(f)
        days.append((f"{base[:4]}-{base[5:7]}", r["period"], r["never"], len(r["period"]) + r["never"]))

    # sliding windows over the day index
    nb = PERIOD_CAP // BIN + 1                      # +1 overflow row
    cols, labels, medians, never_rates, ns = [], [], [], [], []
    for s in range(0, len(days) - WIN_DAYS + 1, STEP_DAYS):
        chunk = days[s:s + WIN_DAYS]
        per = [p for d in chunk for p in d[1]]
        nev = sum(d[2] for d in chunk)
        tot = sum(d[3] for d in chunk)
        if len(per) < 50:
            continue
        h = np.zeros(nb)
        for p in per:
            h[min(p // BIN, nb - 1)] += 1
        h /= h.sum()                                # column-normalized density
        cols.append(h)
        labels.append(chunk[len(chunk) // 2][0])    # center month label
        medians.append(float(np.median(per)))
        never_rates.append(nev / max(tot, 1))
        ns.append(len(per))

    M = np.array(cols).T                            # rows=period bins, cols=time

    # ---- PNG (heatmap + median overlay; never-rate panel) ----
    png = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(ASSETS, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), height_ratios=[3, 1], sharex=True)
        im = ax1.imshow(M, aspect="auto", origin="lower", cmap="magma",
                        extent=[0, len(cols), 0, PERIOD_CAP + BIN])
        ax1.plot(np.arange(len(medians)) + 0.5, medians, color="cyan", lw=1.8, label="median period")
        ax1.set_ylabel("oscillation period (min)")
        ax1.set_title("Oscillation-period distribution evolving over 2024→2025 "
                      f"({WIN_DAYS}d sliding window, {STEP_DAYS}d step)")
        ax1.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax1, label="density")
        ax2.plot(np.arange(len(never_rates)) + 0.5, np.array(never_rates) * 100, color="red", lw=1.5)
        ax2.set_ylabel("never-recover %"); ax2.set_xlabel("time")
        # month ticks
        ticks = [i for i in range(len(labels)) if i == 0 or labels[i] != labels[i - 1]]
        ax2.set_xticks([t + 0.5 for t in ticks])
        ax2.set_xticklabels([labels[t] for t in ticks], rotation=90, fontsize=7)
        ax1.grid(alpha=0.15); ax2.grid(alpha=0.2)
        plt.tight_layout()
        png = os.path.join(ASSETS, "period_evolution.png")
        plt.savefig(png, dpi=110); plt.close()
    except Exception as e:
        print(f"(matplotlib unavailable: {e})")

    # ---- markdown report ----
    L = []
    def w(s):
        print(s.encode("ascii", "replace").decode()); L.append(s)   # console is cp1252-safe; file is utf-8
    w("# Oscillation-period evolution (2024→2025, sliding window)")
    w(f"{WIN_DAYS}-day window, {STEP_DAYS}-day step, {len(cols)} windows, {sum(ns)} pooled oscillations.\n")
    if png:
        w(f"![period evolution](assets/period_evolution.png)\n")
    w("## Median oscillation period over time (min)")
    w("```")
    w("sparkline: " + spark(medians))
    w(f"range: {min(medians):.0f} → {max(medians):.0f} min  (start {medians[0]:.0f}, end {medians[-1]:.0f})")
    w("```")
    w("## Never-recover rate over time (%)")
    w("```")
    w("sparkline: " + spark(never_rates))
    w(f"range: {min(never_rates)*100:.0f}% → {max(never_rates)*100:.0f}%")
    w("```")
    # transition callouts: biggest window-over-window median jumps
    deltas = [(abs(medians[i] - medians[i - 1]), labels[i - 1], labels[i], medians[i - 1], medians[i])
              for i in range(1, len(medians))]
    deltas.sort(reverse=True)
    w("\n## Biggest period transitions (window-over-window median shift)")
    for dmag, a, b, va, vb in deltas[:6]:
        w(f"- {a} → {b}: {va:.0f} → {vb:.0f} min  (Δ{vb-va:+.0f})")
    w("\n## Read")
    w("If the median wanders and jumps, the recovery clock is non-stationary -> a fixed cut-time is")
    w("wrong; the cut-threshold must track the CURRENT window's period. The heatmap's bright band is")
    w("the live oscillation timescale; the red panel flags regimes where wrong trades stop coming back.")
    os.makedirs(REPDIR, exist_ok=True)
    open(os.path.join(REPDIR, "period_evolution.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {os.path.join(REPDIR, 'period_evolution.md')}")


if __name__ == "__main__":
    main()
