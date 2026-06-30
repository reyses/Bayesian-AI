"""APEX RUNG — gemma4 trades a real day BAR-BY-BAR with ZERO lookahead.

This is the hardest probe and the whole point of the exercise. We already KNOW
LLM-as-decider is in the graveyard (latency, non-determinism, no CI discipline). This run
does NOT try to make money — it empirically documents HOW a local LLM behaves as a causal
decider: decision distribution, format-failure rate, latency per bar, and the realized PnL.

ZERO-LOOKAHEAD FIREWALL (load-bearing, the cardinal rule of this repo):
  - At decision bar t the prompt is built from bars[t-W : t+1] ONLY (through close[t]).
    It NEVER sees bar t+1.. — `_build_prompt` slices `df.iloc[lo:t+1]` and asserts it.
  - The decision made on close[t] is FILLED at open[t+1] (realistic next-bar execution).
    Filling at close[t] would be a 1-bar lookahead; we do not do that.

Run (heavy ~ one LLM call per bar): user runs it; outputs go to reports/ only.
  python research/llm_capability/pipeline/causal_trader.py --day 2024_02_20 --max_bars 120
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from ollama_client import chat, extract_json, TEXT_MODEL  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPORT_DIR = os.path.join(ROOT, "research", "llm_capability", "reports")

# MNQ contract economics (named constants — no magic numbers).
TICK_SIZE = 0.25          # MNQ tick size (points)
TICK_VALUE = 0.50         # $ per tick per contract
POINT_VALUE = TICK_VALUE / TICK_SIZE   # = $2.00 per index point per contract

SYSTEM = (
    "You are a futures day-trader for MNQ (Micro Nasdaq). Each step you see the most recent "
    "1-minute closes and your current position, and you choose ONE action for the NEXT bar.\n"
    "Actions: LONG (open/stay long), SHORT (open/stay short), CLOSE (go flat), HOLD (no change).\n"
    "You CANNOT see the future. Trade to grow PnL; avoid churning in chop.\n"
    "Reply ONLY JSON: {\"action\":\"LONG|SHORT|CLOSE|HOLD\",\"reason\":\"<10 words>\"}."
)


def _build_prompt(df, t, W, pos, entry, unreal):
    """Causal prompt for decision at bar t. ASSERTS no row > t is ever referenced."""
    lo = max(0, t - W)
    window = df.iloc[lo:t + 1]                      # <-- through close[t] ONLY
    assert window.index.max() == t, "LOOKAHEAD: window must end at the decision bar t"
    closes = window["close"].tolist()
    recent = " ".join(f"{c:.2f}" for c in closes)
    posdesc = {0: "FLAT", 1: f"LONG from {entry:.2f} (unreal ${unreal:+.0f})",
               -1: f"SHORT from {entry:.2f} (unreal ${unreal:+.0f})"}[pos]
    return (f"Last {len(closes)} 1-min closes (oldest..newest): {recent}\n"
            f"Current close: {closes[-1]:.2f}\nPosition: {posdesc}\nChoose the action for the next bar.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2024_02_20")
    ap.add_argument("--warmup", type=int, default=20, help="bars of history before first decision")
    ap.add_argument("--max_bars", type=int, default=120, help="decision bars (call budget)")
    a = ap.parse_args()

    df = pd.read_parquet(os.path.join(ROOT, "DATA", "ATLAS", "1m", f"{a.day}.parquet")).reset_index(drop=True)
    W = a.warmup
    end = min(len(df) - 1, W + a.max_bars)           # need open[t+1] -> stop before last bar

    pos, entry, realized = 0, 0.0, 0.0               # pos in {-1,0,1}
    trades, decisions = [], []
    acts = {"LONG": 0, "SHORT": 0, "CLOSE": 0, "HOLD": 0, "_BAD": 0}
    lat = []

    def close_at(price):
        nonlocal pos, entry, realized
        pnl = (price - entry) * pos * POINT_VALUE
        realized += pnl
        trades.append(dict(side=("LONG" if pos == 1 else "SHORT"), entry=entry, exit=price, pnl=pnl))
        pos, entry = 0, 0.0

    for t in range(W, end):
        cur = df.at[t, "close"]
        unreal = (cur - entry) * pos * POINT_VALUE if pos else 0.0
        prompt = _build_prompt(df, t, W, pos, entry, unreal)
        reply, dt = chat(prompt, system=SYSTEM, model=TEXT_MODEL, num_predict=120)
        lat.append(dt)
        js = extract_json(reply)
        action = str(js.get("action", "")).upper().strip() if js else ""
        if action not in ("LONG", "SHORT", "CLOSE", "HOLD"):
            acts["_BAD"] += 1
            action = "HOLD"                            # safe default on bad output (counted)
        else:
            acts[action] += 1
        fill = df.at[t + 1, "open"]                    # FILL NEXT BAR (causal execution)

        if action == "CLOSE" and pos != 0:
            close_at(fill)
        elif action == "LONG":
            if pos == -1:
                close_at(fill)
            if pos == 0:
                pos, entry = 1, fill
        elif action == "SHORT":
            if pos == 1:
                close_at(fill)
            if pos == 0:
                pos, entry = -1, fill

        decisions.append(dict(t=int(t), close=float(cur), action=action,
                              reason=(js.get("reason", "") if js else reply[:40]),
                              pos=pos, realized=round(realized, 2)))

    # close any open position at the final available open (causal)
    if pos != 0:
        close_at(df.at[end, "open"])

    # PF-based Trade WR (project canonical): (sum win / |sum loss|) - 1
    wins = sum(tr["pnl"] for tr in trades if tr["pnl"] > 0)
    losses = abs(sum(tr["pnl"] for tr in trades if tr["pnl"] < 0))
    pf_wr = (wins / losses - 1) if losses > 0 else float("inf") if wins > 0 else 0.0

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, f"causal_decisions_{a.day}.jsonl"), "w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")

    L = []
    def w(s):
        print(s); L.append(s)
    w(f"# APEX — gemma4 causal trade run (ZERO lookahead) | {a.day}")
    w(f"model={TEXT_MODEL} | warmup={W} | decision bars={end-W} | fill=next-bar-open\n")
    w("## Result")
    w(f"- Realized PnL: **${realized:+.2f}** (1 contract, MNQ ${POINT_VALUE:.2f}/pt)")
    w(f"- Trades: {len(trades)}  | PF-based Trade WR: {pf_wr:+.3f}  (0=breakeven, +1=PF 2)")
    w(f"- gross wins ${wins:.0f} / gross losses ${losses:.0f}")
    w("\n## Behavior (the real findings)")
    w(f"- action counts: {acts}")
    w(f"- BAD/unparseable outputs (forced HOLD): {acts['_BAD']}/{end-W} = {acts['_BAD']/(end-W):.0%}")
    w(f"- latency/bar: mean {sum(lat)/len(lat):.2f}s  -> a 390-bar RTH day ~ {sum(lat)/len(lat)*390/60:.0f} min of inference")
    w(f"- wall time: {sum(lat)/60:.1f} min for {end-W} bars")
    w("\n## Limitations observed")
    w("- (latency vs a 5s/1m live cadence; non-determinism even at temp=0; format failures; churn) "
      "— see action counts + BAD rate above; full decision log in causal_decisions_*.jsonl")
    open(os.path.join(REPORT_DIR, f"apex_causal_trading_{a.day}.md"), "w").write("\n".join(L) + "\n")
    print(f"\nwrote apex_causal_trading_{a.day}.md + causal_decisions_{a.day}.jsonl")


if __name__ == "__main__":
    main()
