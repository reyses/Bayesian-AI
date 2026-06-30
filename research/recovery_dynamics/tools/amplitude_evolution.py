"""The REGIME over time: amplitude is the volume knob (period is the fixed clock). Track the local
amplitude scale across 2024->2025 to see calm<->wide transitions — the live setting that would set
the 'normal oscillation' envelope.

Per-day scalars (from the same anchor first-return measurement):
  - vol_scale = median(amplitude / sqrt(period))  -> the diffusion/volatility coefficient (regime).
  - ref_amp   = median amplitude of 8-15 min oscillations -> 'typical size of a ~10-min swing today'.
  - trend_share = no-return fraction.
Sliding-window smoothed; PNG + sparkline. START of the framework, descriptive baseline (not causal yet).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import anchor_period as ap  # noqa: E402  (reuse day_periods / ONE_M / ROOT / MAXLOOK)

ROOT, ONE_M = ap.ROOT, ap.ONE_M
REPORT = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "amplitude_evolution.md")
ASSETS = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "assets")
CACHE = os.path.join(ROOT, "artifacts", "amp_evolution_cache.npz")
WIN = 21   # sliding window (days)


def per_day(fresh):
    if os.path.exists(CACHE) and not fresh:
        z = np.load(CACHE, allow_pickle=True)
        return list(z["lab"]), z["vol"], z["ref"], z["trend"]
    lab, vol, ref, trend = [], [], [], []
    files = sorted(glob.glob(os.path.join(ONE_M, "2024_*.parquet")) +
                   glob.glob(os.path.join(ONE_M, "2025_*.parquet")))
    for f in tqdm(files, desc="days", unit="day"):
        try:
            close = pd.read_parquet(f)["close"].to_numpy(np.float64)
        except Exception:
            continue
        per, amps, nr, tot = ap.day_periods(close)
        if len(per) < 50:
            continue
        per = np.asarray(per); amps = np.asarray(amps)
        band = (per >= 8) & (per < 15)
        b = os.path.basename(f)
        lab.append(f"{b[:4]}-{b[5:7]}-{b[8:10]}")
        vol.append(float(np.median(amps / np.sqrt(per))))
        ref.append(float(np.median(amps[band])) if band.sum() else np.nan)
        trend.append(nr / tot)
    vol, ref, trend = np.array(vol), np.array(ref), np.array(trend)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, lab=np.array(lab), vol=vol, ref=ref, trend=trend)
    return lab, vol, ref, trend


def roll(x, w):
    return np.array([np.nanmedian(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


def spark(v):
    bl = "▁▂▃▄▅▆▇█"
    v = np.asarray(v); lo, hi = np.nanmin(v), np.nanmax(v); rng = (hi - lo) or 1
    return "".join(bl[min(int((x - lo) / rng * 7), 7)] for x in v)


def main():
    fresh = "--fresh" in sys.argv
    lab, vol, ref, trend = per_day(fresh)
    rvol, rref, rtr = roll(vol, WIN), roll(ref, WIN), roll(trend, WIN)

    png = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(ASSETS, exist_ok=True)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6.5), height_ratios=[2, 1], sharex=True)
        x = np.arange(len(lab))
        a1.plot(x, rref, color="darkorange", lw=1.8, label=f"typical ~10-min swing (pt, {WIN}d med)")
        a1.plot(x, roll(vol, WIN) * np.nanmedian(rref) / np.nanmedian(rvol), color="steelblue",
                lw=1.0, alpha=0.7, label="vol scale amp/√period (rescaled)")
        a1.set_ylabel("amplitude (pt)"); a1.legend(fontsize=8, loc="upper left")
        a1.set_title("Amplitude regime over 2024→2025 — the 'volume knob' on the fixed period clock")
        boundary = next((i for i, L in enumerate(lab) if L.startswith("2025")), None)
        if boundary:
            for ax in (a1, a2):
                ax.axvline(boundary, color="gray", ls="--", lw=1)
            a1.text(boundary, a1.get_ylim()[1] * 0.95, " 2025", color="gray", fontsize=8)
        a2.plot(x, rtr * 100, color="crimson", lw=1.5)
        a2.set_ylabel("trend share %"); a2.set_xlabel("time")
        ticks = [i for i in range(len(lab)) if i == 0 or lab[i][:7] != lab[i - 1][:7]]
        a2.set_xticks(ticks); a2.set_xticklabels([lab[t][:7] for t in ticks], rotation=90, fontsize=7)
        a1.grid(alpha=0.15); a2.grid(alpha=0.2); plt.tight_layout()
        png = os.path.join(ASSETS, "amplitude_evolution.png")
        plt.savefig(png, dpi=110); plt.close()
    except Exception as e:
        print(f"(matplotlib unavailable: {e})")

    L = []
    def w(s):
        print(s.encode("ascii", "replace").decode()); L.append(s)
    w("# Amplitude regime over time — the volume knob (2024→2025)")
    w(f"{len(lab)} days, {WIN}-day sliding median. Period is the fixed clock; this is how WIDE the "
      f"oscillations are (the regime that sets the 'normal' envelope).\n")
    if png:
        w("![amplitude evolution](assets/amplitude_evolution.png)\n")
    w("## Typical ~10-min swing amplitude (pt) over time")
    w("```")
    w("spark: " + spark(rref[~np.isnan(rref)]))
    w(f"range {np.nanmin(rref):.0f} → {np.nanmax(rref):.0f} pt  | "
      f"2024 median {np.nanmedian(rref[:next((i for i,l in enumerate(lab) if l.startswith('2025')), len(lab))]):.0f}pt"
      f" vs 2025 median {np.nanmedian(rref[next((i for i,l in enumerate(lab) if l.startswith('2025')), 0):]):.0f}pt")
    w("```")
    w("## Trend share (no-return) over time")
    w("```")
    w("spark: " + spark(rtr))
    w(f"range {np.nanmin(rtr)*100:.0f}% → {np.nanmax(rtr)*100:.0f}%")
    w("```")
    w("\n## Read (START of a framework, not the framework)")
    w("Amplitude is non-stationary where period is constant: the 'normal swing size' breathes by regime.")
    w("A LIVE causal estimate of this scale would set the expectation envelope — what oscillation is")
    w("normal/acceptable right now, and what excursion is abnormal (a regime widening or a runaway). That")
    w("causal estimator + validation that breaching the envelope predicts the runaway = the unbuilt rest.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
