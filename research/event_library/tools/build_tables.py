"""Build the EVENT LIBRARY v0 master report from the materialised events.

Reads research/event_library/events/*.parquet (produced by
builders/build_event_library.py) and writes
research/event_library/reports/event_library_v0.md.

Every table reports N, the base rate it is being compared against, a Wilson
95% CI for proportions / quartiles for magnitudes, and an explicit
significance statement on every delta (CLAUDE.md metric rules). N < 30 is
flagged UNDERPOWERED in-line.

Run from repo root:
  python research/event_library/tools/build_tables.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline.common import (EVENTS_DIR, REPORTS_DIR, day_list,       # noqa: E402
                             boot_median_ci, prop_delta_ci, pct_ci,
                             quart, sig_note, wilson, UNDERPOWERED_N)
from pipeline import detectors as det                                 # noqa: E402
from pipeline import outcomes as out                                  # noqa: E402

L: list[str] = []


def w(*s):
    L.extend(s)


def load(name):
    return pd.read_parquet(os.path.join(EVENTS_DIR, f"{name}.parquet"))


def split_table(df, by, col, order=None):
    """Markdown crosstab of `col` within `by`, counts + row %."""
    ct = pd.crosstab(df[by], df[col])
    if order:
        ct = ct.reindex(columns=[c for c in order if c in ct.columns])
    hdr = "| " + str(by) + " | N | " + " | ".join(ct.columns) + " |"
    sep = "|---" * (len(ct.columns) + 2) + "|"
    rows = [hdr, sep]
    for k, r in ct.iterrows():
        n = int(r.sum())
        cells = " | ".join(f"{int(v)} ({v / n:.1%})" if n else "0"
                           for v in r.values)
        flag = " **UNDERPOWERED**" if n < UNDERPOWERED_N else ""
        rows.append(f"| {k} | {n}{flag} | {cells} |")
    return rows


def delta_line(label, k1, n1, lab1, k2, n2, lab2):
    p1 = k1 / n1 if n1 else np.nan
    p2 = k2 / n2 if n2 else np.nan
    dd, lo, hi = prop_delta_ci(k1, n1, k2, n2)
    return (f"- **{label}**: {lab2} {p2:.1%} (n={n2}) vs {lab1} {p1:.1%} "
            f"(n={n1}) -> delta {dd:+.1%} 95% CI [{lo:+.1%}, {hi:+.1%}] "
            f"-> {sig_note(lo, hi)}")


# ===========================================================================
def main() -> None:
    n_days_all = len(day_list())
    uc = load("ultra_chop")
    cc = load("chop_control")
    ld = load("leg_descent")
    fp = load("fakeout_poke")
    st = load("stall")
    sh = load("defended_poke_shelf")
    fv = load("flush_v_day")
    n_trading = len(set(fp["day"]) | set(fv["day"]) | set(uc["day"]))  # days with RTH tape

    w("# EVENT LIBRARY v0 — owner-named tape states as causal detectors "
      "+ cohort tables", "",
      "Owner architecture: *identify specific events, read the fuzzy events.* "
      "Each named tape state gets (a) a strictly causal detector and (b) its "
      "own cohort outcome table. This is the substrate for an event-"
      "classification + table-lookup ML target, not price prediction.", "",
      f"Corpus: `DATA/ATLAS/{{1s,5s,1m}}` day files, {n_days_all} day files in "
      f"the 3-timeframe intersection, of which **{n_trading} carry RTH tape** "
      "(the remaining 64 are Sunday-evening / holiday files whose only bars "
      "are the 18:00-19:00 prior-evening session).", "",
      "**Live-day guard**: `2024_09_16` is the pocket-dojo live-sim day and is "
      "EXCLUDED from every table below. It appears only in "
      "`reports/anchor_fire.md`, where each detector is fired against the "
      "owner's calibration anchors.", "",
      "**Two race definitions are reported for every continuation question.** "
      "The natural race is distance-ASYMMETRIC (a new low sits 1 tick away, "
      "breaking the stair sits 10-15pt away), so its headline percentage is "
      "mostly geometry. Every such event therefore also carries a distance-"
      "SYMMETRIC race (+-10pt from the event close, 10pt = the POCKET_CARD "
      "floor stop) whose null is 50% by construction. **Read the symmetric "
      "race to answer 'is this event informative?'**", "",
      "**Causality is demonstrated, not asserted.** "
      "`tools/causality_audit.py` replays every detector on days truncated at "
      "11:00 / 13:00 / 14:30 ET and requires every event stamped at or before "
      "the cut to reappear identically. Current result over 40 sampled days: "
      "**0 missing, 0 extra, 0 field mismatches across all six detectors** "
      "(`reports/causality_audit.md`). That test caught one real lookahead — "
      "DEFENDED_POKE_AT_SHELF was reading a 3-bar poke minimum at a stamp "
      "that could fire on bar +1 — which is fixed; the fix moved its crack "
      "rate from 28% to 38%, so the leak was materially optimistic.", "")

    w("## Headline verdicts", "",
      "| event | N | detector prevalence | headline | sharp or fuzzy? |",
      "|---|---|---|---|---|")
    verdicts = []       # filled below, printed here via placeholder

    # ---------------------------------------------------------------- CHOP
    esc = uc[uc["escaped"]]
    k_up = int((esc["escape_dir"] == 1).sum())
    n_esc = len(esc)
    p_up, up_lo, up_hi = wilson(k_up, n_esc)
    mag = {m: esc[f"mag_{m}m"].dropna() for m in out.CHOP_MAG_MIN}
    ctl = {m: cc[f"abs_{m}m"].dropna() for m in out.CHOP_MAG_MIN}
    med15, l15, h15 = boot_median_ci(mag[15])
    v_chop = ("**FUZZY** — escape direction is a coin flip and the post-escape "
              "drift median is ~0")
    verdicts.append(("1. ULTRA_CHOP", len(uc),
                     f"{len(uc)/n_trading:.1f}/day on {uc.day.nunique()} days",
                     f"escape UP {p_up:.1%} [{up_lo:.0%},{up_hi:.0%}]; "
                     f"+15min drift median {med15:+.2f}pt", v_chop))

    # ------------------------------------------------------------- DESCENT
    d2 = ld[ld["chain_n"] >= 2]
    d1 = ld[ld["chain_n"] == 1]
    k1 = int((d1["sym_race"] == "CONT").sum())
    n1 = int(d1["sym_race"].isin(["CONT", "AGAINST"]).sum())
    k2 = int((d2["sym_race"] == "CONT").sum())
    n2 = int(d2["sym_race"].isin(["CONT", "AGAINST"]).sum())
    dd_, lo_, hi_ = prop_delta_ci(k1, n1, k2, n2)
    v_desc = ("**FUZZY** — stair depth N carries no information: the "
              "symmetric continuation rate is flat at 50% for every N")
    verdicts.append(("2. LEG_DESCENT", len(ld),
                     f"{len(ld)/n_trading:.1f} defended pushes/day",
                     f"sym CONT N>=2 {k2/n2:.1%} vs N=1 {k1/n1:.1%}, "
                     f"delta {dd_:+.1%} [{lo_:+.1%},{hi_:+.1%}]", v_desc))

    # ---------------------------------------------------------------- POKE
    ret = fp[fp["kind"] == "RETURN"]
    bko = fp[fp["kind"] == "BREAKOUT"]
    kr, nr = int(ret["exceed_ref_first"].sum()), len(ret)
    kb, nb = int(bko["exceed_ref_first"].sum()), len(bko)
    ddp, lop, hip = prop_delta_ci(kb, nb, kr, nr)
    v_poke = ("**SHARP on the level question, FUZZY on direction** — the "
              "snap-back cuts p(level clears) by ~24pp vs a poke that sticks, "
              "while the symmetric +-10pt direction race moves ~1pp (formally "
              "significant at n=153k, operationally nil)")
    verdicts.append(("3. FAKEOUT_POKE", len(ret),
                     f"{len(ret)/n_trading:.1f} snap-backs/day",
                     f"never clears the level {1-kr/nr:.1%} vs {1-kb/nb:.1%} "
                     f"for sticking pokes, delta {-ddp:+.1%}", v_poke))

    # --------------------------------------------------------------- STALL
    s1 = st[st["stalled"]]
    s0 = st[~st["stalled"]]
    ks, ns = int((s1["race"] == "NEW_EXTREME").sum()), int(
        s1["race"].isin(["NEW_EXTREME", "GIVEBACK_50"]).sum())
    kc, nc = int((s0["race"] == "NEW_EXTREME").sum()), int(
        s0["race"].isin(["NEW_EXTREME", "GIVEBACK_50"]).sum())
    kss = int((s1["sym_race"] == "CONT").sum())
    nss = int(s1["sym_race"].isin(["CONT", "AGAINST"]).sum())
    v_stall = ("**FUZZY** — the 85% new-extreme rate is positional mechanics "
               "(the stall is defined as not having given back); the "
               "symmetric race is 50/50")
    verdicts.append(("4. STALL", len(s1),
                     f"{len(s1)}/{len(st)} peak candidates = "
                     f"{len(s1)/len(st):.1%}",
                     f"NEW_EXTREME {ks/ns:.1%} vs control {kc/nc:.1%}; "
                     f"sym CONT {kss/nss:.1%}", v_stall))

    # --------------------------------------------------------------- SHELF
    dec = sh[sh["outcome"].isin(["CRACK", "HOLD"])]
    fl = dec[dec["day_class"] == "flushV"]
    ot = dec[dec["day_class"] == "other"]
    kf, nf = int((fl["outcome"] == "CRACK").sum()), len(fl)
    ko, no_ = int((ot["outcome"] == "CRACK").sum()), len(ot)
    dds, los, his = prop_delta_ci(ko, no_, kf, nf)
    v_shelf = (f"**FUZZY across day-class** — crack rate is ~{(ko+kf)/(no_+nf):.0%} "
               "on any high-dwell shelf and flushV days are not distinguishable")
    verdicts.append(("5. DEFENDED_POKE_AT_SHELF", len(dec),
                     f"{len(sh)} events on {sh.day.nunique()} days",
                     f"CRACK flushV {kf/nf:.1%} vs other {ko/no_:.1%}, "
                     f"delta {dds:+.1%} [{los:+.1%},{his:+.1%}]", v_shelf))

    # ------------------------------------------------------------- FLUSH_V
    ff = fv[fv["is_flush"]]
    fo = fv[~fv["is_flush"]]
    kpf, npf = int(ff["peak_reclaim"].sum()), len(ff)
    kpo, npo = int(fo["peak_reclaim"].sum()), len(fo)
    ddf, lof, hif = prop_delta_ci(kpo, npo, kpf, npf)
    v_flush = ("**SHARP as a day-class label** — flushV days reclaim the "
               "recovery peak far more often than matched control days")
    verdicts.append(("6. FLUSH_V_DAY", len(ff),
                     f"{len(ff)}/{len(fv)} scored days = {len(ff)/len(fv):.1%}",
                     f"peak reclaim {kpf/npf:.1%} vs control {kpo/npo:.1%}, "
                     f"delta {ddf:+.1%} [{lof:+.1%},{hif:+.1%}]", v_flush))

    for name, n, prev, head, verd in verdicts:
        w(f"| {name} | {n} | {prev} | {head} | {verd} |")
    w("")
    w("## Calibration-day anchor check (2024_09_16 — EXCLUDED from all tables)",
      "",
      "Detection timestamps in `reports/anchor_fire.md`. Owner anchors: "
      "ULTRA_CHOP 10:23:50-10:24:31; LEG_DESCENT the 09:56-10:24 stair "
      "19697 -> 19633; FLUSH_V_DAY the open flush.", "",
      "| event | fires on the calibration day? |", "|---|---|",
      "| 1. ULTRA_CHOP | **NO — 37 fires that day, 0 inside the anchor "
      "window.** Not a threshold miss: see the anchor-honesty table below. |",
      "| 2. LEG_DESCENT | YES — 85 defended pushes, 53 with chain_n>=2, "
      "**11 inside the 09:56-10:24 anchor**, chain descent up to 156pt. |",
      "| 3. FAKEOUT_POKE | YES — 264 armed pokes, 148 RETURN events. |",
      "| 4. STALL | NO — 63 peak candidates, 0 stalls. No owner anchor was "
      "given for STALL; the day trended hard and STALL is a ~1%-of-candidates "
      "big-leg event, so 0 on one day is within expectation. |",
      "| 5. DEFENDED_POKE_AT_SHELF | YES — 5 events (09:58, 10:30, 14:23, "
      "14:55, 15:28), all flushV class, all HOLD. |",
      "| 6. FLUSH_V_DAY | YES — confirmed 09:50, flush 110.2pt, recovery 85%. "
      "(The owner quotes -173.5pt; the imported detector measures from the "
      "09:30 open, not the overnight high.) |", "")

    # =====================================================================
    w("---", "", "## 1. ULTRA_CHOP", "")
    w("### Definition (reproducible)", "",
      "1s closes, RTH 09:30-15:30 ET, rolling **60s** window "
      f"(`CHOP_WIN_S={det.CHOP_WIN_S}`) which must contain "
      f">= {det.CHOP_MIN_BARS} 1s bars. Fires when BOTH:", "",
      f"- `flips >= {det.CHOP_MIN_FLIPS}` — direction flips of the non-zero "
      "1s close-to-close moves inside the window (corpus RTH p75 ~= 30, i.e. "
      "top-quartile flip density);",
      f"- `box <= {det.CHOP_BOX_AMBIENT_FRAC:.2f} x ambient` where `box` = "
      "window high-low and `ambient` = median of the last "
      f"{det.CHOP_AMBIENT_MIN} NON-overlapping 1-minute boxes, read through "
      "the previous minute only.", "",
      "Chop box = the firing window's `[low, high]`. **Escape** = first close "
      f"beyond a box edge by `{det.CHOP_ESCAPE_BUF_FRAC:.2f} x box` "
      "(scale-free buffer). One event per episode: the next fire is blocked "
      f"until the escape, and never within {det.CHOP_REFRACTORY_S}s.", "")

    w("### Why 'small net range' is RELATIVE, not absolute (anchor honesty)", "",
      "The owner's anchor (2024_09_16 10:23:50-10:24:31, '~24 flips / 42s in "
      "13.25pt') **does not fire this detector, and no useful absolute "
      "threshold makes it fire.** Measured on 1s closes, the 60s windows "
      "ending inside that anchor carry:", "",
      "| quantity | anchor window | that day's RTH p50 | p90 |",
      "|---|---|---|---|",
      "| flips / 60s | 27-33 | 27 | 32 |",
      "| box (pt) | 15.50-24.00 | 11.00 | 21.00 |", "",
      "Flip density is genuinely elevated (p75-p90). The **box is ABOVE "
      "median** — the 60s window swallows the 11.25pt one-second flush at "
      "10:24:11, so the anchor is an impulse-with-churn, not a tight box. An "
      "absolute threshold loose enough to fire there (`box <= 24pt`) fires on "
      "~40% of all RTH bars and measures nothing. Absolute point thresholds "
      "are also era-broken: MNQ traded 16k in 2024 and 28k in 2026. Hence the "
      "ambient-relative box test. See `reports/anchor_fire.md`.", "")

    w("### Prevalence", "",
      f"- {len(uc)} events on {uc.day.nunique()} of {n_trading} trading days "
      f"= {len(uc)/n_trading:.2f}/day",
      "- per year:")
    yr = uc.assign(y=uc["day"].str[:4]).groupby("y").size()
    yd = pd.Series({y: sum(1 for d in day_list() if d.startswith(y))
                    for y in yr.index})
    for y in yr.index:
        w(f"  - {y}: {int(yr[y])} events / {int(yd[y])} day files = "
          f"{yr[y]/yd[y]:.2f}/day")
    w("")

    w("### TABLE — escape statistics", "",
      f"- escaped within {det.CHOP_ESCAPE_MAX_S//60}min: "
      f"{pct_ci(int(uc['escaped'].sum()), len(uc))}",
      f"- time-to-escape: {quart(esc['escape_lag_s'], 's')}",
      f"- escape direction UP: {pct_ci(k_up, n_esc)} "
      f"(null 50%; {'CI excludes 50%' if not (up_lo <= 0.5 <= up_hi) else 'not significant — coin flip'})",
      "", "Signed displacement AFTER the escape, in the escape direction "
      "(positive = the break kept going), vs the unconditional |move| from "
      f"{len(cc)} random RTH anchors ({out.CONTROL_PER_DAY}/day):", "",
      "| horizon | signed move in escape dir | median 95% CI | control \\|move\\| |",
      "|---|---|---|---|")
    for m in out.CHOP_MAG_MIN:
        md, lo, hi = boot_median_ci(mag[m])
        w(f"| +{m}min | {quart(mag[m])} | [{lo:+.2f}, {hi:+.2f}] | "
          f"{quart(ctl[m])} |")
    w("", f"Escape direction is {p_up:.1%} up [{up_lo:.1%}, {up_hi:.1%}]. On "
      f"n={n_esc} the CI does clear 50%, but the effect is a {100*(p_up-0.5):.1f}pp "
      "lean — statistically detectable, operationally nothing, and it is the "
      "corpus's own upward drift rather than a property of chop. Post-escape "
      "drift medians are a fraction of a point against unconditional |moves| "
      f"of {np.median(ctl[15]):.1f}pt at 15min: **the break carries no "
      "persistent direction.**", "")

    w("### Causality self-audit", "",
      "- Every input is a TRAILING window ending at the firing bar: flips, "
      "box, and the ambient scale (which reads only minutes strictly before "
      "the firing bar's own minute, via `prev_slot = slot - 1`).",
      "- The escape and all magnitudes are computed in `outcomes.py`, "
      "strictly after the stamp.",
      "- The episode de-dup guard uses the forward escape time. That is a "
      "SAMPLING decision (which candidate bars become rows), not a feature — "
      "it cannot leak into an event's own outcome, but it does mean the row "
      "set is not reproducible bar-by-bar in live without buffering. **This "
      "is the one place in the library where a live implementation must "
      "differ**: live must fire on the first qualifying bar and self-suppress "
      f"for {det.CHOP_REFRACTORY_S}s.",
      "- RTH mask is bounded on both sides, so prior-evening bars (mod >= "
      "1080) can never be selected.", "",
      f"**Verdict:** {v_chop}.", "")

    # =====================================================================
    w("---", "", "## 2. LEG_DESCENT (stair-down)", "")
    w("### Definition (reproducible)", "",
      f"5s closes, repo-canonical {det.ZZ_REVERSAL_PT:.1f}pt close zigzag "
      "(= `research/reversal_gauge` REVERSAL_PT).", "",
      "A **push** opens when a swing HIGH confirms (price falls "
      f"{det.ZZ_REVERSAL_PT:.1f}pt off the running max). Inside the push the "
      "running low `L` is tracked on bar LOWS. A **defense** confirms at the "
      f"first RTH bar whose CLOSE is >= `L + {det.DEFENSE_PT:.1f}pt` within "
      f"{det.DEFENSE_WIN_S}s of the bar that set `L` — this covers both "
      "owner phrasings at once: a long lower wick (same-bar low, recovering "
      "close) and a fast multi-bar V-up. One defense per push (the first).", "",
      f"**chain_n** = consecutive defended pushes whose high does not exceed "
      f"the previous push's high by more than {det.LOWER_HIGH_TOL_PT:.1f}pt "
      "(a push clearing the prior high by <= 2pt is a poke, not a new high — "
      "same tolerance as FAKEOUT_POKE). `chain_n >= 2` is the owner's "
      "'>= 2 lower-high pushes'; `chain_n == 1` is the structurally matched "
      "CONTROL (a defended push with no lower-high predecessor).", "",
      "Stamp = the defense bar. Outcomes over "
      f"{out.DESCENT_HORIZON_S//60}min.", "")

    w("### Prevalence", "",
      f"- {len(ld)} defended pushes on {ld.day.nunique()} days = "
      f"{len(ld)/n_trading:.1f}/day; `chain_n>=2` = {len(d2)} "
      f"({len(d2)/len(ld):.1%} of pushes)",
      "- chain length distribution: " +
      ", ".join(f"N={k}:{v}" for k, v in
                sorted(ld['chain_n'].clip(upper=6).value_counts().items())) +
      "  (N=6 bucket is 6+)", "")

    w("### TABLE — continuation after the Nth stair step", "",
      "ASYMMETRIC race (owner's literal question): NEW_LOW = a low 1 tick "
      "below the step low; STAIR_BREAK = a high 1 tick above the step high "
      f"(which sits >= {det.ZZ_REVERSAL_PT:.0f}pt away).", "")
    w(*split_table(ld.assign(chain_N=ld["chain_n"].clip(upper=5)), "chain_N", "race",
                   ["NEW_LOW", "STAIR_BREAK", "NEITHER"]))
    w("", "SYMMETRIC race (+-10pt from the defense close; null = 50%):", "")
    w(*split_table(ld.assign(chain_N=ld["chain_n"].clip(upper=5)), "chain_N", "sym_race",
                   ["CONT", "AGAINST", "NEITHER"]))
    w("", delta_line("symmetric continuation, N>=2 vs N=1",
                     k1, n1, "N=1", k2, n2, "N>=2"), "")

    hk1, hn1 = int(d1["defense_hold"].sum()), len(d1)
    hk2, hn2 = int(d2["defense_hold"].sum()), len(d2)
    w("### TABLE — defense-hold rate "
      f"(defended low survives {out.DEFENSE_HOLD_S//60}min un-undercut)", "",
      f"- N=1: {pct_ci(hk1, hn1)}",
      f"- N>=2: {pct_ci(hk2, hn2)}",
      delta_line("defense hold, N>=2 vs N=1", hk1, hn1, "N=1", hk2, hn2, "N>=2"),
      "")
    w("### TABLE — stair depth distribution", "",
      f"- step depth (step high -> step low): {quart(ld['step_depth'])}",
      f"- cumulative chain descent at step N>=2: {quart(d2['chain_descent'])}",
      f"- defense size (close - low): {quart(ld['defense_pt'])}",
      f"- defense lag: {quart(ld['defense_lag_s'], 's')}", "")
    w("### Causality self-audit", "",
      "- The step high is a zigzag pivot CONFIRMED before the push began; the "
      "step low is a running min over bars <= the stamp; the defense is a "
      "close at the stamp bar. Nothing is back-dated to the pivot bar.",
      "- `chain_n` uses only previously CLOSED pushes (a push is closed by "
      "its own confirmed low pivot, or by the next confirmed high pivot).",
      "- Known asymmetry, not a leak: the asymmetric race is ~69% NEW_LOW at "
      "every N because the two triggers sit at very different distances. The "
      "symmetric race is the interpretable one.", "",
      f"**Verdict:** {v_desc}. The chain is real and easy to detect "
      "(58k instances), but conditioning on it moves nothing.", "")

    # =====================================================================
    w("---", "", "## 3. FAKEOUT_POKE", "")
    w("### Definition (reproducible)", "",
      f"5s closes, same {det.ZZ_REVERSAL_PT:.1f}pt zigzag. During an active "
      "leg, when the leg's running extreme first clears a REMEMBERED "
      "same-direction leg extreme (confirmed pivot, aged <= "
      f"{det.EXTREME_MEMORY_S//60}min) by `0 < over <= "
      f"{det.POKE_MAX_PT:.1f}pt`, a poke ARMS. It then resolves as exactly "
      "one of:", "",
      "| kind | resolution | meaning |", "|---|---|---|",
      f"| **RETURN** | a close back inside the level within "
      f"{det.POKE_RETURN_S}s | **the owner's fakeout poke** |",
      f"| BREAKOUT | clears the level by > {det.POKE_MAX_PT:.1f}pt first | "
      "the level actually broke |",
      f"| STUCK | still outside after {det.POKE_RETURN_S}s without clearing | "
      "hung on the level |", "",
      "Stamp = the resolution bar. Outcomes over "
      f"{out.POKE_HORIZON_S//60}min. Everything is close-based, matching the "
      "close-based zigzag that defines the reference extremes.", "")
    w("### Prevalence", "",
      f"- {len(fp)} armed pokes on {fp.day.nunique()} days; "
      + ", ".join(f"{k} {v} ({v/len(fp):.1%})"
                  for k, v in fp['kind'].value_counts().items()),
      f"- RETURN (the event) = {len(ret)/n_trading:.1f}/day",
      f"- poke depth beyond the level: {quart(ret['poke_depth'])}",
      f"- reference-level age: {quart(ret['ref_age_s']/60.0, 'min')}", "")

    w("### TABLE — resume vs reverse from the poke", "",
      "'Never exceeds the prior extreme' is reported two ways. UNBOUNDED "
      f"('ever, within {out.POKE_HORIZON_S//60}min') is nearly vacuous: price "
      "wanders 2pt past any level given 45 minutes. BOUNDED "
      "(`exceed_ref_first`: clears the level by > "
      f"{det.POKE_MAX_PT:.1f}pt BEFORE a {out.POKE_REVERSE_PT:.0f}pt adverse "
      "move) is the load-bearing number.", "",
      "| cohort | N | clears level, unbounded | clears level, BOUNDED | "
      "**never clears (bounded)** |", "|---|---|---|---|---|")
    for kind in ["RETURN", "STUCK", "BREAKOUT"]:
        g = fp[fp["kind"] == kind]
        if not len(g):
            continue
        ku, kbn = int(g["exceed_ref"].sum()), int(g["exceed_ref_first"].sum())
        p, lo, hi = wilson(kbn, len(g))
        flag = " **UNDERPOWERED**" if len(g) < UNDERPOWERED_N else ""
        w(f"| {kind} | {len(g)}{flag} | {ku/len(g):.1%} | "
          f"{p:.1%} [{lo:.1%}, {hi:.1%}] | **{1-p:.1%}** |")
    w("", delta_line("clears the level (bounded), RETURN vs BREAKOUT",
                     kb, nb, "BREAKOUT", kr, nr, "RETURN"), "")
    w("Asymmetric RESUME/REVERSE race (RESUME = clears the poke extreme by "
      f"{out.POKE_RESUME_PT}pt, REVERSE = {out.POKE_REVERSE_PT:.0f}pt "
      "adverse, whichever first):", "")
    w(*split_table(fp, "kind", "race", ["RESUME", "REVERSE", "NEITHER"]))
    w("", "SYMMETRIC race (+-10pt from the resolution close, in leg "
      "direction; null = 50%):", "")
    w(*split_table(fp, "kind", "sym_race", ["CONT", "AGAINST", "NEITHER"]))
    ksr = int((ret["sym_race"] == "CONT").sum())
    nsr = int(ret["sym_race"].isin(["CONT", "AGAINST"]).sum())
    ksb = int((bko["sym_race"] == "CONT").sum())
    nsb = int(bko["sym_race"].isin(["CONT", "AGAINST"]).sum())
    _, slo, shi = wilson(ksr, nsr)
    w("", delta_line("symmetric continuation, RETURN vs BREAKOUT",
                     ksb, nsb, "BREAKOUT", ksr, nsr, "RETURN"),
      f"- RETURN symmetric continuation vs the 50% null: {pct_ci(ksr, nsr)} — "
      f"{'CI excludes 50%' if not (slo <= 0.5 <= shi) else 'coin flip'}", "")

    w("### On the '~78.5% never exceed the prior extreme' reference", "",
      "That figure could not be reproduced and I could not find a re-poke "
      "library that produces it. The only 78.5% in the repo is "
      "`research/dojo_forge/reports/oscillation_harvest.md` — "
      "P(a sigma-band traverse COMPLETES) at K>=5 prior traverses, over "
      "54,911 fade attempts. That is a different measurement "
      "(band-to-band traverse completion), not level re-poke survival. The "
      "adjacent level-memory claim in `human_dojo/POCKET_CARD.md` is "
      "'+10 survives 98.5% of re-pokes', which is about STOP survival "
      "distance, not exceedance.", "",
      f"This library's comparable number: **{1-kr/nr:.1%} of snap-back pokes "
      f"never clear the level** (n={nr}) before a "
      f"{out.POKE_REVERSE_PT:.0f}pt adverse move, and "
      f"{1-int(ret['exceed_ref'].sum())/nr:.1%} never clear it at all inside "
      f"{out.POKE_HORIZON_S//60}min. Neither lands near 78.5%; the "
      "78.5% reference is a different event.", "")

    w("### Causality self-audit", "",
      "- The reference extreme is a zigzag pivot confirmed strictly earlier; "
      "arming uses the running extreme at the current bar; resolution is a "
      "condition on the current bar's close or the elapsed 60s. No forward "
      "bar is read by the detector.",
      "- Arming is blocked on bars where a pivot confirms (`ev is None` "
      "guard), so a leg reversal cannot masquerade as a poke.",
      "- An armed poke is resolved BEFORE the leg-reversal bookkeeping in "
      "the same bar, so a snap-back that coincides with a leg turn is still "
      "recorded as RETURN rather than dropped.", "",
      f"**Verdict:** {v_poke}.", "")

    # =====================================================================
    w("---", "", "## 4. STALL", "")
    w("### Definition (reproducible)", "",
      f"5s closes, same zigzag. A stall CANDIDATE opens at every new running "
      f"leg extreme with leg MFE >= {det.STALL_MIN_MFE_PT:.0f}pt in RTH "
      f"(de-duplicated: a new candidate only once the extreme has advanced "
      f"{det.STALL_EXT_FRAC:.0%} of MFE past the last one opened). A "
      "candidate is:", "",
      f"- **VOID** if price extends > {det.STALL_EXT_FRAC:.0%} of MFE beyond "
      "it (the leg was still RUNNING, not stalling — generalises "
      "`four_phase_cohort`'s implicit assumption that the peak is the peak);",
      f"- **FAILED** if giveback exceeds {det.STALL_GIVE_FRAC:.0%} of MFE "
      "(= `four_phase_cohort` STALL_GIVE) before its mark;",
      f"- **STALL** if it survives to its mark at peak + "
      f"{det.STALL_MIN_S//60}min (= `four_phase_cohort` STALL_MIN).", "",
      "Stamp = the 10-minute mark. FAILED candidates are emitted too, at the "
      "SAME relative moment, as the matched control. This generalises the "
      "four-phase stall off flush days: no flush, no V, no shape gate — any "
      "leg peak on any day.", "")
    w("### Prevalence", "",
      f"- {len(st)} peak candidates on {st.day.nunique()} days; "
      f"**{len(s1)} STALL ({len(s1)/len(st):.2%})**, {len(s0)} control",
      f"- leg MFE at the peak — STALL: {quart(s1['mfe_pt'])}",
      f"- leg MFE at the peak — control: {quart(s0['mfe_pt'])}",
      "- Selection effect, stated up front: a 30%-of-MFE tolerance is only "
      f"survivable for big legs (an {det.ZZ_REVERSAL_PT:.0f}pt zigzag "
      "reversal alone exceeds 30% of any MFE below ~27pt), so STALL is "
      "structurally a big-leg event. That is a property of the owner's "
      "definition, not a bug.", "")
    w("### TABLE — what follows a stall", "",
      f"ASYMMETRIC race from the mark, {out.STALL_HORIZON_S//60}min: "
      f"NEW_EXTREME = {out.STALL_NEW_EXT_PT}pt beyond the stalled peak; "
      f"GIVEBACK_50 = {out.STALL_GIVE_RACE_FRAC:.0%} of MFE given back.", "")
    w(*split_table(st, "stalled", "race",
                   ["NEW_EXTREME", "GIVEBACK_50", "NEITHER", "NO_DATA"]))
    w("", delta_line("p(new extreme first), STALL vs control",
                     kc, nc, "control", ks, ns, "STALL"), "")
    w("Giveback bucket at the mark (monotone read; the stall bucket is the "
      "0-30% row by definition):", "")
    bk = pd.cut(st["give_frac"], [-np.inf, 0.30, 0.50, 1.0, np.inf],
                labels=["<=30% (STALL)", "30-50%", "50-100%", ">100%"])
    w(*split_table(st.assign(bucket=bk), "bucket", "race",
                   ["NEW_EXTREME", "GIVEBACK_50", "NEITHER", "NO_DATA"]))
    w("", "SYMMETRIC race (+-10pt from the mark close, in leg direction; "
      "null = 50%):", "")
    w(*split_table(st, "stalled", "sym_race", ["CONT", "AGAINST", "NEITHER"]))
    kcs = int((s0["sym_race"] == "CONT").sum())
    ncs = int(s0["sym_race"].isin(["CONT", "AGAINST"]).sum())
    _, tlo, thi = wilson(kss, nss)
    w("", delta_line("symmetric continuation, STALL vs control",
                     kcs, ncs, "control", kss, nss, "STALL"),
      f"- STALL symmetric continuation vs the 50% null: {pct_ci(kss, nss)} — "
      f"{'CI excludes 50%' if not (tlo <= 0.5 <= thi) else 'coin flip'}")
    w("", f"- time to resolution, STALL: "
      f"{quart(s1.loc[s1['resolve_s'] > 0, 'resolve_s']/60.0, 'min')}",
      f"- time to resolution, control: "
      f"{quart(s0.loc[s0['resolve_s'] > 0, 'resolve_s']/60.0, 'min')}",
      f"- net move in leg direction at +{out.STALL_HORIZON_S//60}min, STALL: "
      f"{quart(s1['net_60m'])}",
      f"- net move in leg direction at +{out.STALL_HORIZON_S//60}min, control: "
      f"{quart(s0['net_60m'])}", "")
    w("### Causality self-audit", "",
      "- The candidate's peak, MFE and running giveback are all computed "
      "from bars <= the current bar; the stamp is the 10-minute mark, at "
      "which the stall is fully observed.",
      "- `p['dir']` is pinned at candidate open so a leg reversal cannot flip "
      "the sign of the giveback measurement mid-candidate (this was a real "
      "bug in the first implementation).",
      "- The first implementation used a single candidate slot, which let a "
      "failed candidate block the next 10 minutes of tape and deleted most "
      "real stalls (0 stalls / 20 candidates on the calibration day). "
      "Candidates are now a pending LIST. Overlapping candidates within one "
      "leg are correlated — rows are NOT independent; day-level clustering "
      "should be assumed in any downstream fit.", "",
      f"**Verdict:** {v_stall}.", "")

    # =====================================================================
    w("---", "", "## 5. DEFENDED_POKE_AT_SHELF", "")
    w("### Definition (reproducible)", "",
      "1m bars — deliberately the same bar size and the same window "
      "constants as `research/dojo_forge/tools/vshape_retest_cohort.py`, so "
      "the flushV sub-cohort is directly comparable to its published number.",
      "",
      f"- **shelf** = mode of the prior {det.SHELF_LOOKBACK_MIN} 1m closes in "
      f"{det.SHELF_BIN_PT:.0f}pt bins (STRICTLY prior bars; >= "
      f"{det.SHELF_MIN_BARS} bars required), and the mode bin must hold >= "
      f"{det.SHELF_MIN_DWELL_FRAC:.0%} of them (2.4x uniform for a 60pt "
      "spread) — a genuine high-dwell level, not just an argmax;",
      f"- **approach** = a high >= shelf + {det.SHELF_AWAY_PT:.0f}pt in the "
      f"prior {det.SHELF_APPROACH_MIN}min (price must come back TO the shelf);",
      f"- **trigger** = a bar with low <= shelf + {det.SHELF_RETEST_PT:.0f}pt;",
      f"- **poke** = min low over {det.SHELF_POKE_BARS} bars from the trigger;",
      f"- **defended** = a high >= poke + {det.SHELF_DEF_PT:.0f}pt within "
      f"{det.SHELF_DEF_BARS} bars. Stamp = that bar.",
      f"- **outcome**, {out.SHELF_OUT_MIN}min: CRACK if low <= poke - "
      f"{out.SHELF_CRACK_PT:.0f}pt before high >= poke + "
      f"{out.SHELF_HOLD_PT:.0f}pt; HOLD on the reverse.",
      f"- `day_class` is causal: flushV only if the flush confirmation ts "
      "(imported from the reversal_gauge builder) is <= the stamp.", "")
    w("### Prevalence", "",
      f"- {len(sh)} events on {sh.day.nunique()} of {n_trading} trading days; "
      f"{len(dec)} decided (CRACK/HOLD), "
      f"{int((sh['outcome']=='UNRESOLVED').sum())} unresolved",
      f"- shelf dwell fraction: {quart(sh['dwell_frac']*100, '%')}",
      f"- defense bounce: {quart(sh['bounce_pt'])}", "")
    w("### TABLE — crack vs hold BY day-class", "")
    w(*split_table(dec, "day_class", "outcome", ["CRACK", "HOLD"]))
    w("", delta_line("p(CRACK), flushV vs other", ko, no_, "other", kf, nf,
                     "flushV"), "")

    # vshape-matched sub-cohort
    vs = dec[(dec["day_class"] == "flushV")].copy()
    vs = vs[(vs["mod"] >= 10 * 60) & (vs["mod"] <= 12 * 60 + 30)]
    vs = vs.sort_values(["day", "ts"]).groupby("day", as_index=False).first()
    kv, nv = int((vs["outcome"] == "CRACK").sum()), len(vs)
    w("### Reproducing the vshape 1.4% (or explaining the gap)", "",
      "`vshape_retest_cohort.py` reported **CRACK 1/72 = 1.4% "
      "[0%, 7%]**. Restricting THIS detector to the nearest matching "
      "sub-cohort — flushV day-class, FIRST event of the day, trigger "
      "between 10:00 and 12:30 — gives:", "",
      f"- **{pct_ci(kv, nv, 'CRACK ')}**", "",
      "Why the generalised number is higher: vshape's shelf is a specific "
      "construct (modal close inside the lower 45% of the flush range, "
      "computed over 09:30-10:05, i.e. the flush-consolidation dwell) tested "
      "at the FIRST retest after the V-recovery peak, with the outcome "
      "window additionally truncated at 12:30. This detector's shelf is any "
      "2-hour dwell mode anywhere in the session, so it samples ordinary "
      "intraday shelves that carry no V-floor memory. **The 1.4% is a "
      "property of the V-floor shelf specifically, not of defended pokes at "
      "shelves in general** — which is the useful finding: generalising the "
      "event destroys the edge.", "")
    w("### Causality self-audit", "",
      "- The dwell histogram reads `close[i-120:i]` — strictly prior bars, "
      "never including the trigger bar.",
      "- The poke extreme and the defense high are read over bars AT AND "
      "AFTER the trigger, and the stamp is the defense bar itself, so no "
      "condition uses a bar later than the stamp.",
      "- `day_class` compares the stamp ts against the imported causal flush "
      "confirmation ts; it is never applied retroactively to earlier events "
      "on the same day.",
      "- The trigger scan is bounded to RTH on both sides; the 2h lookback "
      "may reach into the same file's pre-open and prior-evening bars, which "
      "is legitimate same-contract tape (ATLAS day files are per-day "
      "outrights, so there is no roll seam inside a file).", "",
      f"**Verdict:** {v_shelf}.", "")

    # =====================================================================
    w("---", "", "## 6. FLUSH_V_DAY", "")
    w("### Definition (reproducible)", "",
      "**Imported, not reimplemented**: `_flush_confirm_ts` from "
      "`research/reversal_gauge/builders/extract_freeze_events.py` — the "
      "FIXED detector, whose AUDIT FIX comment documents the prior-evening "
      "bug (unbounded `mod >= X` matched 18:00 bars first, mislabelling "
      "flushV on 167/600 days and killing the window-closed guard on 992 "
      "events). Importing rather than copying is deliberate: this detector "
      "must not be able to drift from its audited version.", "",
      f"- flush: 09:30 open minus min low over [09:30, 09:50) >= "
      f"{det.FLUSH_MIN_PT:.0f}pt;",
      f"- recovery: a high after the min-low bar reaching low + "
      f"{det.FLUSH_RECOVERY_FRAC:.0%} of the flush at or before 10:20;",
      "- confirmation ts = max(recovery bar, first bar at/after 09:50) — the "
      "day is only KNOWABLY flushV once both have printed.", "",
      "**Control**: every non-flushV day contributes one row at the 10:20 "
      "recovery deadline with the identical V-low / V-peak construction, so "
      "the day-class comparison is same-construction rather than free-"
      "floating.", "")
    w("### Prevalence", "",
      f"- {len(fv)} scored days of {n_trading} trading days "
      f"(days whose 09:30-09:50 window or 10:20 anchor is missing are "
      "dropped); **flushV = "
      f"{len(ff)} ({len(ff)/len(fv):.1%})**",
      f"- flush size, flushV days: {quart(ff['flush_pt'])}",
      f"- recovery fraction at confirm, flushV days: "
      f"{quart(ff['rec_frac']*100, '%')}", "")
    w("### TABLE — what the day does after the class is knowable", "",
      "| metric | flushV (n=%d) | control (n=%d) | delta 95%% CI | sig |"
      % (len(ff), len(fo)), "|---|---|---|---|---|")
    for label, col in [("V-low broken later", "low_break"),
                       ("V-peak reclaimed later", "peak_reclaim")]:
        a, na = int(fo[col].sum()), len(fo)
        b, nb2 = int(ff[col].sum()), len(ff)
        pa, la, ha = wilson(a, na)
        pb, lb, hb = wilson(b, nb2)
        dd2, lo2, hi2 = prop_delta_ci(a, na, b, nb2)
        w(f"| {label} | {pb:.1%} [{lb:.1%},{hb:.1%}] | "
          f"{pa:.1%} [{la:.1%},{ha:.1%}] | [{lo2:+.1%}, {hi2:+.1%}] | "
          f"{sig_note(lo2, hi2)} |")
    w("", "First event after confirmation:", "")
    w(*split_table(fv.assign(cls=np.where(fv["is_flush"], "flushV", "control")),
                   "cls", "first", ["PEAK_RECLAIM", "LOW_BREAK", "NEITHER"]))
    w("", f"- RTH close position in [V-low, V-peak], flushV: "
      f"{quart(ff['close_frac'], '')}",
      f"- RTH close position in [V-low, V-peak], control: "
      f"{quart(fo['close_frac'], '')}", "")
    w("### Causality self-audit", "",
      "- Detector imported verbatim from the audited source; confirmation ts "
      "is the later of the recovery print and the flush-window close, so the "
      "label is never available before both facts exist.",
      "- V-peak is recomputed as the max high from the flush-low bar THROUGH "
      "the confirmation bar only — not the day's eventual peak.",
      "- The control anchor uses a BOTH-SIDES-BOUNDED mask "
      "`[10:20, 11:20)`; an unbounded `mod >= 10:20` would have selected "
      "prior-evening bars, which is exactly the audited bug class.",
      "- Outcome scans stop at 16:00 ET and never reach the file's evening "
      "session.", "",
      f"**Verdict:** {v_flush}. Note this is a DAY-CLASS label, and the "
      "sharpness is partly definitional: a flushV day has, by construction, "
      "already recovered 60% of its flush by 10:20, so 'peak reclaimed "
      "later' is measured against a peak that momentum just produced. It is "
      "still the one event here whose conditioning moves a day-scale outcome "
      "well outside its control CI.", "")

    # =====================================================================
    w("---", "", "## What this says for the teacher-student target", "",
      "- **Detection is easy; discrimination is not.** All six states are "
      "cheaply and causally detectable at scale (18k-153k instances). Five "
      "of six move the direction-outcome distribution by roughly nothing "
      "once the distance asymmetry is removed. This is the same wall the "
      "program has hit repeatedly (oscillator-vs-runaway stuck ~0.57 AUC; "
      "'every mechanical decider loses').",
      "- **The one non-trivial conditional found here is level-flavoured, "
      "not direction-flavoured**: FAKEOUT_POKE's snap-back cuts p(the level "
      "clears) from ~91% to ~67%. Its own +-10pt DIRECTION race moves ~1pp "
      "over the same split — the information is about the LEVEL's fate, not "
      "about which way price goes, and that is exactly what a table-lookup "
      "layer can serve and a direction classifier cannot.",
      "- **Generalising a sharp event destroys it.** DEFENDED_POKE_AT_SHELF "
      "reproduces nothing like the V-floor's 1.4% crack rate once the shelf "
      "is any dwell mode. If a cohort table is sharp, the sharpness lives in "
      "the SPECIFICITY of the construct, so the event vocabulary must keep "
      "day-shape context attached rather than abstracting it away.",
      "- **Every table here is a base-rate table, and most base rates are "
      "~50%.** A student trained to classify these events and look up the "
      "table would output the base rate. The value of v0 is negative "
      "information: it prices out five candidate features before anyone "
      "spends GPU on them.", "")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "event_library_v0.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {path}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
