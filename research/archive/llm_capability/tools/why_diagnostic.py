"""WHY did the apex decider HOLD every bar? Capture the real chain-of-thought.

The apex run used think:false (fast, valid JSON) which DISABLES gemma4's reasoning and only
keeps a terse `reason`. Here we re-run a HANDFUL of bars with think:TRUE and dump
`message.thinking` — and we deliberately pick bars where price is CLEARLY MOVING (largest
trailing moves), to test whether the model still rationalizes inaction during an obvious trend.
"""
import json
import os
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from causal_trader import SYSTEM, _build_prompt, ACTION_SCHEMA  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPORT = os.path.join(ROOT, "research", "llm_capability", "reports", "apex_why_diagnostic.md")
URL = "http://127.0.0.1:11434/api/chat"


def call_with_thinking(prompt, system):
    body = {"model": "gemma4:latest", "stream": False, "think": True,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "format": ACTION_SCHEMA,
            "options": {"temperature": 0, "num_predict": 600, "seed": 0}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    m = json.loads(urllib.request.urlopen(req, timeout=180).read()).get("message", {})
    return m.get("thinking", ""), m.get("content", "")


def main():
    day = "2024_02_20"
    W = 20
    df = pd.read_parquet(os.path.join(ROOT, "DATA", "ATLAS", "1m", f"{day}.parquet")).reset_index(drop=True)
    end = min(len(df) - 1, W + 90)
    # trailing 10-bar move at each decision bar; pick the 3 biggest up and 3 biggest down
    cand = [(t, df.at[t, "close"] - df.at[t - 10, "close"]) for t in range(W + 10, end)]
    cand.sort(key=lambda x: x[1])
    picks = cand[:3] + cand[-3:]   # strongest down-moves + strongest up-moves

    L = []
    def w(s):
        print(s); L.append(s)
    w(f"# APEX 'why' diagnostic — think:TRUE on trending bars | {day}\n")
    w("Picked the bars with the LARGEST trailing 10-min moves (clear trends). Position = FLAT "
      "(the apex never entered). If it still HOLDs here, the inaction bias is real, not chop-specific.\n")
    for t, mv in picks:
        prompt = _build_prompt(df, t, W, 0, 0.0, 0.0)
        thinking, content = call_with_thinking(prompt, SYSTEM)
        act = (json.loads(content).get("action") if content else "?")
        w(f"## bar t={t}  trailing 10-min move = {mv:+.2f} pts  -> ACTION: {act}")
        w(f"_thinking_:\n> " + thinking.strip().replace("\n", "\n> ")[:1200] + "\n")
    open(REPORT, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
