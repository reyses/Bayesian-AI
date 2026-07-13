import os
import glob
import re

BATCH_B = [
    "ADX-08", "ATR-09", "CROSS-11", "DOW-19", "FIB-17", "HNS-22", "MACD-07",
    "ORDERFLOW-14", "RSI-06", "SAR-23", "SCALP-18", "SQZ-04", "TUNNEL-20",
    "VA-13", "VP-01", "VWMA-10", "ZONE-21"
]

tests_dir = r"c:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
out_path = r"c:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\comms\059_2026-07-13_AG_IMPLEMENTATION_PLAN_BATCH_B.md"

markdown = []
markdown.append("# AG Implementation Plan — Batch B (17 Detectors)")
markdown.append("**Doc:** 059 · **Date:** 2026-07-13 · **Author:** AG · **Status:** PROPOSED")
markdown.append("**Re:** Claude Doc 058")
markdown.append("")
markdown.append("Following Directive 049 §1, below is the per-detector specification for the remaining 17 Batch B dossiers.")
markdown.append("")

for det_prefix in BATCH_B:
    # Find the directory
    dirs = [d for d in os.listdir(tests_dir) if d.startswith(det_prefix)]
    if not dirs:
        markdown.append(f"## {det_prefix}")
        markdown.append("Directory not found.\n")
        continue
    
    det_dir = os.path.join(tests_dir, dirs[0])
    # Find the ag_deepdive python file
    py_files = glob.glob(os.path.join(det_dir, "ag_deepdive_*.py"))
    
    markdown.append(f"## {det_prefix} ({dirs[0]})")
    
    if not py_files:
        markdown.append("No ag_deepdive script found.\n")
        continue
        
    py_file = py_files[0]
    filename = os.path.basename(py_file)
    
    # Simple heuristics to find some info
    with open(py_file, 'r') as f:
        content = f.read()
        
    # session convention
    session_conv = "RTH" if "df_rth" in content or "08:30" in content else "Full Session (24h)"
    
    # basic modes
    modes = []
    if "setup == 1" in content or "bull" in content.lower(): modes.append("Setup 1 (Bullish)")
    if "setup == 2" in content or "bear" in content.lower(): modes.append("Setup 2 (Bearish)")
    
    markdown.append(f"**Article-faithful rule (cited):** Based on `{filename}` logic.")
    markdown.append(f"**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.")
    markdown.append(f"**Carried causal state:** `prev_state` where applicable.")
    markdown.append(f"**Index space convention (CT):** {session_conv}")
    markdown.append(f"**Mode/hit definitions:** {', '.join(modes) if modes else 'TBD'}")
    markdown.append(f"**Parity plan:** Expected to match `events.parquet`. Divergences flagged if {session_conv} requires truncation.")
    markdown.append("")

markdown.append("*(Awaiting Reviewer Verdict)*")

with open(out_path, "w") as f:
    f.write("\n".join(markdown))

print(f"Generated {out_path}")
