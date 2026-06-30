"""THE UNIFYING VIEW: there are no good/bad trades — just OSCILLATORS and the TREND TAIL.

Moises' point: every trade oscillates around its entry, so it returns to zero eventually. The only
exceptions are the small taper that catches a trend and runs away (never returns). A 'big win' and a
'death' are the SAME event — a runaway — on opposite sides. The entire edge is in that taper; the
oscillating bulk nets to zero.

Measure (random entries, random direction, horizon = rest of day):
  - a trade becomes 'real' once |PnL| >= MIN_MOVE.
  - OSCILLATOR: after that, PnL returns to zero before EOD (the wash).
  - RUNAWAY: never returns to zero -> terminal PnL sign = win-runaway (rode trend) or loss-runaway (death).
Output: the taper fraction, the win/loss symmetry of runaways, and the signed terminal-PnL distribution
(central spike at zero + the two runaway tails).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import opportunity_cost as oc  # noqa: E402

ROOT, ONE_M = oc.ROOT, oc.ONE_M
REPORT = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "oscillator_vs_trend.md")
MIN_MOVE = 5.0   # pts before a trade counts as 'real' (either direction)


def classify_day(close, rng):
    n = len(close)
    hi = int(n * oc.ENTRY_FRAC)
    out = {"noise": 0, "osc": 0, "win_run": 0, "loss_run": 0, "term": [], "run_mag": []}
    if hi < 5:
        return out
    for e in rng.integers(0, hi, size=oc.N_PER_DAY):
        e = int(e)
        d = 1 if rng.random() < 0.5 else -1
        pnl = d * (close[e + 1:] - close[e])
        if pnl.size < 2:
            continue
        real = np.where(np.abs(pnl) >= MIN_MOVE)[0]
        if len(real) == 0:
            out["noise"] += 1
            continue
        r0 = real[0]
        # after becoming real, does it touch zero again? (sign change relative to its real-side)
        side = np.sign(pnl[r0])
        after = pnl[r0:]
        crossed = np.where(side * after <= 0)[0]      # returned to (or through) zero
        if len(crossed):
            out["osc"] += 1
            out["term"].append(0.0)                    # oscillator -> nets to ~zero
        else:
            term = float(pnl[-1])                      # terminal PnL at EOD (the runaway reach)
            out["term"].append(term)
            out["run_mag"].append(abs(term))
            out["win_run" if term > 0 else "loss_run"] += 1
    return out


def signed_hist(vals, width, lo, hi):
    edges = np.arange(lo, hi + width, width)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for v in vals:
        b = int((np.clip(v, lo, hi - 1e-9) - lo) // width)
        counts[min(b, len(counts) - 1)] += 1
    peak = counts.max() or 1
    out = []
    for i, c in enumerate(counts):
        lbl = f"{edges[i]:>+5.0f}..{edges[i+1]:>+5.0f}"
        out.append(f"{lbl} |{'#'*int(round(46*c/peak))} {c}")
    return out


def main():
    rng = np.random.default_rng(oc.SEED)
    files = sorted(glob.glob(os.path.join(ONE_M, "2024_*.parquet")) +
                   glob.glob(os.path.join(ONE_M, "2025_*.parquet")))
    A = {"noise": 0, "osc": 0, "win_run": 0, "loss_run": 0, "term": [], "run_mag": []}
    for f in tqdm(files, desc="days", unit="day"):
        try:
            close = pd.read_parquet(f)["close"].to_numpy(np.float64)
        except Exception:
            continue
        r = classify_day(close, rng)
        for k in ("noise", "osc", "win_run", "loss_run"):
            A[k] += r[k]
        A["term"] += r["term"]; A["run_mag"] += r["run_mag"]

    real = A["osc"] + A["win_run"] + A["loss_run"]
    runaway = A["win_run"] + A["loss_run"]
    L = []
    def w(s):
        print(s.encode("ascii", "replace").decode()); L.append(s)
    w("# Oscillators vs the trend tail — there are no good/bad trades, just trades (2024+2025)")
    w(f"random entries, horizon = rest of day, 'real' once |PnL|>= {MIN_MOVE}pt. n_real={real}\n")
    w(f"- **OSCILLATORS (return to zero, the wash): {A['osc']/real:.1%}**")
    w(f"- **RUNAWAYS (trend tail, never return): {runaway/real:.1%}** "
      f"= the taper, split win {A['win_run']/max(runaway,1):.0%} / loss {A['loss_run']/max(runaway,1):.0%}")
    w(f"  (random entry => ~symmetric: the trend doesn't care which way you bet)")
    w(f"- (plus {A['noise']} entries that never moved {MIN_MOVE}pt = sub-threshold noise)\n")
    if A["run_mag"]:
        rm = A["run_mag"]
        w(f"- runaway reach (terminal |PnL|): median {np.median(rm):.0f}pt (${np.median(rm)*2:.0f}), "
          f"mean {np.mean(rm):.0f}pt, 90th {np.percentile(rm,90):.0f}pt — the 'to infinity' magnitude\n")
    w("## Signed terminal PnL distribution (oscillators pinned at 0; runaways = the two tails)")
    w("```")
    w("\n".join(signed_hist(A["term"], 10, -100, 100)))
    w("```")
    w("\n## Read")
    w("The bulk returns to zero (no edge — a wash). The small symmetric taper is the trend tail, and a")
    w("'win' vs 'death' is the SAME runaway on opposite sides. So the only edge is: (1) get on the right")
    w("side of a runaway, and (2) RIDE it (don't cut at zero) while CUTTING the adverse runaway fast.")
    w("The oscillating majority is indifferent — which is why entry-direction prediction found no edge.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
