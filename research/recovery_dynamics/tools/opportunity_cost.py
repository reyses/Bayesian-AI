"""THE EXERCISE: what does holding a WRONG trade back to breakeven actually cost?

Per Moises: pick a long/short, let it go underwater, hold until PnL returns to ZERO, and
measure how many OTHER trades we could have taken in that dead-hold window — the opportunity
cost of not realizing we were wrong.

Method (causal-honest; the grader sees the path, the 'trade' does not predict it):
  - sample candidate entries across a real day; evaluate BOTH long and short.
  - a side is a WRONG trade if it draws down >= MIN_ADVERSE_PTS before it ever profits.
  - hold it until PnL first returns to >= 0 (back to breakeven) -> that's the dead-hold window.
    If it never returns within the day -> 'never recovered' bucket (the genuinely dead trade).
  - within [entry, back-to-zero], count tradeable swings (>= SWING_PTS legs) = trades foregone.

Output: mode-first histograms (dead-hold minutes, trades foregone, drawdown depth) + a worked
example + the never-recovered tail. 1-min bars => bars == minutes.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPORT = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "opportunity_cost.md")

# --- tunable thresholds (named, not magic; sensitivity is a TODO) ---
MIN_ADVERSE_PTS = 5.0    # a trade is "wrong/underwater" once it draws down >= this from entry
PROFIT_FIRST_PTS = 5.0   # if it profits >= this BEFORE drawing down, it's a "right" trade -> skip
SWING_PTS = 8.0          # a foregone swing must travel >= this to count as a tradeable opportunity
ENTRY_STEP = 3           # sample a candidate entry every N bars (both long & short each)
POINT_VALUE = 2.0        # MNQ $ per point (tick 0.25 * $0.50)


def swing_legs(closes, lo, hi, thresh):
    """Count directional legs >= thresh within [lo,hi] — a simple zigzag swing count.
    Each confirmed thresh-sized reversal = one tradeable swing."""
    seg = closes[lo:hi + 1]
    if len(seg) < 2:
        return 0
    legs, direction, last_pivot, extreme = 0, 0, seg[0], seg[0]
    for p in seg[1:]:
        if direction == 0:                      # establishing the first leg
            if p - last_pivot >= thresh:
                direction = 1; extreme = p; legs += 1
            elif last_pivot - p >= thresh:
                direction = -1; extreme = p; legs += 1
        elif direction == 1:                    # in an up-leg
            if p > extreme:
                extreme = p
            elif extreme - p >= thresh:          # reversed down -> new down leg
                legs += 1; direction = -1; last_pivot = extreme; extreme = p
        else:                                   # in a down-leg
            if p < extreme:
                extreme = p
            elif p - extreme >= thresh:          # reversed up -> new up leg
                legs += 1; direction = 1; last_pivot = extreme; extreme = p
    return legs


def hist(vals, width, label, unit):
    if not vals:
        return [f"  (no {label})"]
    vmax = max(vals)
    nb = int(vmax // width) + 1
    counts = [0] * nb
    for v in vals:
        counts[min(int(v // width), nb - 1)] += 1
    peak = max(counts) or 1
    mode_bin = counts.index(peak)
    out = [f"{label} (mode bin = {mode_bin*width:.0f}-{(mode_bin+1)*width:.0f} {unit}, n={len(vals)})"]
    for i, c in enumerate(counts):
        bar = "#" * int(round(40 * c / peak))
        out.append(f"{i*width:>4.0f}-{(i+1)*width:<4.0f}{unit} |{bar} {c}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2024_02_20")
    a = ap.parse_args()
    close = pd.read_parquet(os.path.join(ROOT, "DATA", "ATLAS", "1m",
                                          f"{a.day}.parquet"))["close"].to_numpy(np.float64)
    n = len(close)

    dead_min, foregone, depth, never = [], [], [], 0
    examples = []   # (foregone, bars_to_zero, depth, entry, dir)
    for e in range(0, n - 5, ENTRY_STEP):
        p0 = close[e]
        for d in (+1, -1):
            pnl = d * (close[e + 1:] - p0)          # path of PnL (points) from next bar
            # classify: did it profit first (right) or draw down first (wrong)?
            adv = np.where(pnl <= -MIN_ADVERSE_PTS)[0]
            pro = np.where(pnl >= PROFIT_FIRST_PTS)[0]
            if len(adv) == 0:
                continue                            # never went underwater -> not a wrong trade
            t_adv = adv[0]
            if len(pro) and pro[0] < t_adv:
                continue                            # profited before underwater -> a 'right' trade
            # WRONG trade: hold until PnL returns to >= 0
            back = np.where(pnl[t_adv:] >= 0)[0]
            mae = float(-pnl[:].min()) if pnl.size else 0.0
            if len(back) == 0:
                never += 1
                continue
            zero_bar = t_adv + back[0]              # bars after entry+1 to breakeven
            bars_to_zero = int(zero_bar + 1)        # +1 (path started at e+1)
            trough = float(-pnl[:zero_bar + 1].min())
            fg = swing_legs(close, e, e + bars_to_zero, SWING_PTS)
            dead_min.append(bars_to_zero); foregone.append(fg); depth.append(trough)
            examples.append((fg, bars_to_zero, trough, e, d))

    L = []
    def w(s):
        print(s); L.append(s)
    w(f"# Opportunity cost of holding a WRONG trade to breakeven | {a.day}")
    w(f"thresholds: underwater>={MIN_ADVERSE_PTS}pt, swing>={SWING_PTS}pt, entry every {ENTRY_STEP} bars\n")
    w(f"wrong trades that DID return to zero: {len(dead_min)} | never recovered (held to EOD): {never}\n")
    if dead_min:
        w(f"MODE dead-hold time: ~{max(set(dead_min), key=dead_min.count)} min | "
          f"MODE trades foregone: ~{max(set(foregone), key=foregone.count)}")
        w(f"median dead-hold {int(np.median(dead_min))} min | median foregone {int(np.median(foregone))} | "
          f"median depth {np.median(depth):.1f}pt (${np.median(depth)*POINT_VALUE:.0f})\n")
        w("```")
        w("\n".join(hist(foregone, 2, "TRADES FOREGONE while waiting to break even", "")))
        w("")
        w("\n".join(hist(dead_min, 15, "DEAD-HOLD TIME (min underwater -> back to zero)", "m")))
        w("")
        w("\n".join(hist(depth, 5, "DRAWDOWN DEPTH (max underwater)", "pt")))
        w("```")
        # worked example: the most painful (most trades foregone)
        ex = max(examples, key=lambda x: x[0])
        w(f"\n## Worked example (worst opportunity cost)")
        w(f"- {'LONG' if ex[4]>0 else 'SHORT'} entry at bar {ex[3]} (price {close[ex[3]]:.2f})")
        w(f"- went {ex[2]:.1f}pt (${ex[2]*POINT_VALUE:.0f}) underwater, took {ex[1]} min to crawl back to $0")
        w(f"- in that window: {ex[0]} tradeable swings (>= {SWING_PTS}pt) went by — uncaptured")
    w("\n## Read")
    w("Even the trades that 'came back' cost real money: the dead-hold window is time + foregone")
    w("trades, not a free round-trip. The never-recovered bucket is the pure loss on top.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
