"""RUNG 3 — VISION: read a trade chart PNG and classify it.

Ground truth is free: research/edge_case_triage/reports/trade_NNNNN_<LABEL>.png where
<LABEL> in {CHOP, CLEAN_RIDE, GAVE_BACK, SMALL_WIN, ...}. We show gemma4:e2b (multimodal)
the image and ask for the archetype. Scores accuracy + a coarse "did it even read the chart"
signal (does the reply mention price direction). This is the hardest perception rung: small
local multimodal models are weak at fine chart reading — we MEASURE how weak.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ollama_client import chat, extract_json, VISION_MODEL  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
IMG_DIR = os.path.join(ROOT, "research", "edge_case_triage", "reports")
REPORT = os.path.join(ROOT, "research", "llm_capability", "reports", "rung3_vision_chart.md")

LABELS = ["CLEAN_RIDE", "GAVE_BACK", "CHOP", "SMALL_WIN", "STOP_LOSS", "BIG_WIN", "BIG_LOSS"]

SYSTEM = (
    "You are shown a candlestick chart of one futures trade with entry/exit markers. "
    "Classify its archetype into exactly one of: " + ", ".join(LABELS) + ". "
    "CLEAN_RIDE=caught a big move, little giveback; GAVE_BACK=had a big move then surrendered it; "
    "CHOP=tiny noisy range, no real move; SMALL_WIN=modest gain; STOP_LOSS=adverse, stopped out. "
    "Reply ONLY JSON: {\"archetype\":\"<one>\",\"direction\":\"up|down\",\"reason\":\"<10 words>\"}."
)


def truth_of(path):
    base = os.path.basename(path).replace(".png", "")
    parts = base.split("_", 2)  # trade_NNNNN_LABEL...
    return parts[2].upper() if len(parts) >= 3 else "?"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "trade_*.png")))[: a.n]

    lines, lat = [], []
    correct = fmt_fail = read_ok = 0
    def w(s):
        print(s); lines.append(s)
    w(f"# RUNG 3 — vision chart classify | model={VISION_MODEL} | n={len(imgs)}\n")
    w(f"{'file':>34} | {'truth':>11} | {'pred':>11} | ok")
    w("-" * 78)
    for p in imgs:
        truth = truth_of(p)
        reply, dt = chat("Classify this trade chart.", system=SYSTEM, model=VISION_MODEL,
                         images=[p], num_predict=200)
        lat.append(dt)
        js = extract_json(reply)
        if reply and ("up" in reply.lower() or "down" in reply.lower()):
            read_ok += 1
        if not js or "archetype" not in js:
            fmt_fail += 1
            w(f"{os.path.basename(p):>34} | {truth:>11} | {'<no-json>':>11} | !")
            continue
        pred = str(js["archetype"]).upper().strip()
        ok = pred == truth
        correct += ok
        w(f"{os.path.basename(p):>34} | {truth:>11} | {pred:>11} | {'Y' if ok else 'n'}")
    n = max(len(imgs), 1)
    w("")
    w(f"ARCHETYPE accuracy: {correct}/{len(imgs)} = {correct/n:.0%}")
    w(f"'read the chart' (mentions direction): {read_ok}/{len(imgs)} = {read_ok/n:.0%}")
    w(f"JSON-format failures: {fmt_fail}/{len(imgs)} = {fmt_fail/n:.0%}")
    w(f"latency/call: mean {sum(lat)/len(lat):.1f}s" if lat else "no calls")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
