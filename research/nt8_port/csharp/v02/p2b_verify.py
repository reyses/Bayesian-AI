#!/usr/bin/env python3.11
"""P2b parity verifier: score the v0.2 shared-core shim output (out_v02/) against
the GOLDEN vectors + compact reference, the SAME way research/nt8_port proved the
harness at 100%. Also byte-diffs out_v02/ vs out_baseline/ (the golden-matching
harness output). Writes reports/p2b_v02_parity.md fragment values to stdout.
"""
import os, sys, json, glob, hashlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.normpath(os.path.join(HERE, ".."))          # csharp/
PORT = os.path.normpath(os.path.join(CS, ".."))          # nt8_port/
GOLD = os.path.join(PORT, "golden")
REP = os.path.join(PORT, "reports")
V02 = os.path.join(CS, "out_v02")
BASE = os.path.join(CS, "out_baseline")

meta = json.load(open(os.path.join(REP, "_parity_meta.json")))
topk = meta["topk"]
thr = meta["compact_threshold"]

days = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(GOLD, "*.parquet")))

per_stream = {d: [0, 0] for d in topk}
fire_mm = fire_tot = 0
entry_mm = entry_tot = 0
Pmax = 0.0
P_onlyone = 0
zz_leg_mm = zz_conf_mm = zz_tot = 0
zz_age_max = zz_px_max = 0.0
gov_mm = gov_tot = 0

for day in days:
    C = pd.DataFrame(json.load(open(os.path.join(V02, f"{day}.json")))["bars"]).set_index("bar_ts")
    G = pd.read_parquet(os.path.join(GOLD, f"{day}.parquet")).set_index("bar_ts")
    R = pd.read_parquet(os.path.join(REP, f"_ref_{day}.parquet")).set_index("bar_ts")
    bt = G.index.values
    C = C.reindex(bt); R = R.reindex(bt)
    for d in topk:
        g = G[f"f_{d}"].values
        c = np.nan_to_num(C[f"f_{d}"].values, nan=-999).astype(int)
        mm = int((g != c).sum())
        per_stream[d][0] += mm; per_stream[d][1] += len(g)
        fire_mm += mm; fire_tot += len(g)
    cP = C["P_compact"].values.astype(float)
    rP = R["P_compact"].values.astype(float)
    both = np.isfinite(cP) & np.isfinite(rP)
    if both.any():
        Pmax = max(Pmax, float(np.nanmax(np.abs(cP[both] - rP[both]))))
    P_onlyone += int((np.isfinite(cP) != np.isfinite(rP)).sum())
    ce = np.nan_to_num(C["entry"].values, nan=0).astype(int)
    re = R["entry"].values.astype(int)
    entry_mm += int((ce != re).sum()); entry_tot += len(ce)
    # governing direction (only where an entry fires on both)
    if "gov_dir" in C.columns and "gov_dir" in R.columns:
        cg = np.nan_to_num(C["gov_dir"].values, nan=0).astype(int)
        rg = np.nan_to_num(R["gov_dir"].values, nan=0).astype(int)
        m = (re == 1)
        gov_mm += int((cg[m] != rg[m]).sum()); gov_tot += int(m.sum())
    # zigzag R-trigger
    if all(c in C.columns for c in ("zz_leg", "zz_confirm")) and "zz_leg" in G.columns:
        gl = G["zz_leg"].values.astype(int)
        cl = np.nan_to_num(C["zz_leg"].values, nan=-999).astype(int)
        gc = G["zz_confirm"].values.astype(int)
        cc = np.nan_to_num(C["zz_confirm"].values, nan=-999).astype(int)
        zz_leg_mm += int((gl != cl).sum()); zz_conf_mm += int((gc != cc).sum())
        zz_tot += len(gl)
        ga = G["zz_pivot_age_min"].values.astype(float)
        ca = np.nan_to_num(C["zz_pivot_age_min"].values, nan=-1e9).astype(float)
        gp = G["zz_pivot_price"].values.astype(float)
        cpx = np.nan_to_num(C["zz_pivot_price"].values, nan=-1e9).astype(float)
        zz_age_max = max(zz_age_max, float(np.nanmax(np.abs(ga - ca))))
        zz_px_max = max(zz_px_max, float(np.nanmax(np.abs(gp - cpx))))

# byte-diff v02 vs baseline
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()
byte_ok = True
for day in days:
    if md5(os.path.join(V02, f"{day}.json")) != md5(os.path.join(BASE, f"{day}.json")):
        byte_ok = False
        print("BYTE DIFF:", day)

fire_ok = 100 * (1 - fire_mm / max(1, fire_tot))
entry_ok = 100 * (1 - entry_mm / max(1, entry_tot))
n_entries = 0
for day in days:
    R = pd.read_parquet(os.path.join(REP, f"_ref_{day}.parquet"))
    n_entries += int((R["entry"].values == 1).sum())

print(f"days={len(days)}")
print(f"fire-state cells: {fire_tot-fire_mm}/{fire_tot} = {fire_ok:.3f}%  (mismatch={fire_mm})")
print(f"entry bars      : {entry_tot-entry_mm}/{entry_tot} = {entry_ok:.3f}%  (mismatch={entry_mm})")
print(f"entries fired   : {n_entries}")
print(f"gov_dir @entry  : {gov_tot-gov_mm}/{gov_tot} match (mismatch={gov_mm})")
print(f"P max|d|        : {Pmax:.3e}  (P-defined-disagree={P_onlyone})")
print(f"zz_leg          : {zz_tot-zz_leg_mm}/{zz_tot} = {100*(1-zz_leg_mm/max(1,zz_tot)):.3f}%")
print(f"zz_confirm      : {zz_tot-zz_conf_mm}/{zz_tot} = {100*(1-zz_conf_mm/max(1,zz_tot)):.3f}%")
print(f"zz_pivot_age max|d|  : {zz_age_max:.3e}")
print(f"zz_pivot_price max|d|: {zz_px_max:.3e}")
print(f"byte-identical vs out_baseline: {byte_ok}")
worst = max(per_stream.items(), key=lambda kv: kv[1][0])
print(f"worst stream: {worst[0]} mm={worst[1][0]}")
print("threshold:", thr)
allpass = (fire_mm == 0 and entry_mm == 0 and gov_mm == 0 and zz_leg_mm == 0
           and zz_conf_mm == 0 and Pmax <= 1e-6 and byte_ok)
print("VERDICT:", "PASS 100.000%" if allpass else "FAIL")
sys.exit(0 if allpass else 1)
