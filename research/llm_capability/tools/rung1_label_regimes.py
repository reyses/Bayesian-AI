"""RUNG 1 (easiest) — REGIME LABELING against the user's manual labels.

Real ground truth: DATA/ATLAS/regime_labels_2d.csv (348 day records, 2D taxonomy
direction{FLAT,UP,DOWN} x variation{SMOOTH,CHOPPY}, with IS/OOS/VAL split).

Task: hand gemma4 a day's summary stats; it classifies the 2D regime; we score against
the label on the HELD-OUT OOS days only. Tests numeric reasoning + instruction-following
on real domain data. The taxonomy rules are GIVEN in the prompt — so this measures whether
the model can APPLY a stated rule to numbers, not whether it guesses our scheme.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ollama_client import chat, extract_json, TEXT_MODEL  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CSV = os.path.join(ROOT, "DATA", "ATLAS", "regime_labels_2d.csv")
REPORT = os.path.join(ROOT, "research", "llm_capability", "reports", "rung1_label_regimes.md")

FEATS = ["net_move", "range", "directional_strength", "efficiency_ratio", "range_expansion"]

SYSTEM = (
    "You classify a trading day's price-action regime on TWO axes from its summary statistics.\n"
    "DIRECTION axis: UP (sustained net advance, high directional_strength & efficiency_ratio, positive net_move), "
    "DOWN (sustained decline, negative net_move, high directional_strength), "
    "FLAT (net_move small relative to range, low efficiency_ratio).\n"
    "VARIATION axis: SMOOTH (efficient, range_expansion near or below 1, orderly), "
    "CHOPPY (range_expansion elevated above 1, inefficient, lots of back-and-forth).\n"
    "Reply ONLY JSON: {\"direction\":\"UP|DOWN|FLAT\",\"variation\":\"SMOOTH|CHOPPY\"}."
)


def fmt(row):
    return " ".join(f"{k}={row[k]:.4g}" for k in FEATS if pd.notna(row[k]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="max held-out days to score")
    ap.add_argument("--split", default="OOS")
    a = ap.parse_args()

    df = pd.read_csv(CSV)
    df = df[df["split"] == a.split].dropna(subset=FEATS)
    if len(df) > a.n:
        df = df.sample(a.n, random_state=0)
    df = df.reset_index(drop=True)

    lines, lat = [], []
    dir_ok = var_ok = joint_ok = fmt_fail = 0
    def w(s):
        print(s); lines.append(s)
    w(f"# RUNG 1 — regime labeling vs manual labels | model={TEXT_MODEL} | split={a.split} n={len(df)}\n")
    w(f"{'date':>10} | truth dir/var | pred dir/var | dir var")
    w("-" * 60)
    for _, row in df.iterrows():
        reply, dt = chat(fmt(row), system=SYSTEM)
        lat.append(dt)
        js = extract_json(reply)
        if not js or "direction" not in js or "variation" not in js:
            fmt_fail += 1
            w(f"{row['date']:>10} | {row['direction_axis']:>4}/{row['variation_axis']:<6} | <no-json>")
            continue
        pd_ = str(js["direction"]).upper().strip()
        pv = str(js["variation"]).upper().strip()
        d_ok = pd_ == row["direction_axis"]
        v_ok = pv == row["variation_axis"]
        dir_ok += d_ok; var_ok += v_ok; joint_ok += (d_ok and v_ok)
        w(f"{row['date']:>10} | {row['direction_axis']:>4}/{row['variation_axis']:<6} | "
          f"{pd_:>4}/{pv:<6} | {'Y' if d_ok else 'n'}   {'Y' if v_ok else 'n'}")
    n = len(df)
    w("")
    w(f"DIRECTION accuracy: {dir_ok}/{n} = {dir_ok/n:.0%}   (3-class chance ~33%)")
    w(f"VARIATION accuracy: {var_ok}/{n} = {var_ok/n:.0%}   (2-class chance ~50%)")
    w(f"JOINT (both right): {joint_ok}/{n} = {joint_ok/n:.0%}")
    w(f"JSON-format failures: {fmt_fail}/{n} = {fmt_fail/n:.0%}")
    w(f"latency/call: mean {sum(lat)/len(lat):.1f}s")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
