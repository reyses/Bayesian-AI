#!/usr/bin/env python3.11
"""Diff-check the sharing mechanism: assert the SHARED-CORE-V02 region embedded in
the NinjaScript strategy is byte-for-byte identical to the canonical region source
and to the region wrapped into the shim. If any drift, FAIL (the shim would then be
proving different code than what ships in NT8)."""
import os, sys, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
BEGIN = "// ===SHARED-CORE-V02 BEGIN==="
END = "// ===SHARED-CORE-V02 END==="

CANON = os.path.join(HERE, "EnsembleCoreV02.region.cs")
SHIMGEN = os.path.join(HERE, "shim", "EnsembleCoreV02.gen.cs")
STRAT = os.path.join(REPO, "docs", "nt8", "7-EnsembleRunner_v0.2-RC.cs")
STRAT_V03 = os.path.join(REPO, "docs", "nt8", "7-EnsembleRunner_v0.3-RC.cs")


def region(path):
    txt = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    i0 = txt.find(BEGIN)
    i1 = txt.find(END)
    if i0 < 0 or i1 < 0:
        raise SystemExit("markers not found in " + path)
    return txt[i0:i1 + len(END)]


def h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


canon = region(CANON)
strat = region(STRAT)
shim = region(SHIMGEN)
hc, hs, hg = h(canon), h(strat), h(shim)
print("canonical region sha256:", hc, "(%d bytes)" % len(canon))
print("strategy  region sha256:", hs, "MATCH" if hs == hc else "MISMATCH")
print("shim      region sha256:", hg, "MATCH" if hg == hc else "MISMATCH")
ok = (hs == hc and hg == hc)
# v0.3-RC is a WRAPPER-only bugfix; its SHARED-CORE region MUST also be byte-identical.
if os.path.exists(STRAT_V03):
    strat3 = region(STRAT_V03)
    h3 = h(strat3)
    print("strat v0.3 region sha256:", h3, "MATCH" if h3 == hc else "MISMATCH")
    ok = ok and (h3 == hc)
print("VERDICT:", "PASS -- one identical region in canon + strategy(+v0.3) + shim" if ok else "FAIL")
sys.exit(0 if ok else 1)
