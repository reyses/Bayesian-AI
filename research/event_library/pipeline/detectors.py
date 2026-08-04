"""EVENT LIBRARY detectors — one causal detector per owner-named tape state.

Each detector returns a list of dicts. Every dict carries `ts` = the epoch
second of the bar at which ALL defining conditions became OBSERVABLE. Nothing
in a detector reads a bar with index > that bar. Forward-looking outcome fields
are added later, in `outcomes.py` — the causality boundary is a module
boundary, on purpose.

Threshold provenance is documented on every constant. Calibration anchors come
from 2024_09_16 (the live-sim day) which is EXCLUDED from all tables.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from .common import (Day, RTH_START_MIN, RTH_END_MIN, ZZ_REVERSAL_PT,
                     EXTREME_MEMORY_S, ZigZag, TICK_PT, REPO_ROOT)

# Single source of truth for the flushV day class: the FIXED detector in the
# reversal_gauge builder (its AUDIT FIX comment documents the prior-evening bug
# that mislabelled 167/600 days). Imported, not copied, so it cannot drift.
sys.path.insert(0, os.path.join(REPO_ROOT, "research", "reversal_gauge", "builders"))
from extract_freeze_events import (                      # noqa: E402
    _flush_confirm_ts, FLUSH_MIN_PT, FLUSH_RECOVERY_FRAC,
    FLUSH_WIN_END_MIN, RECOVERY_DEADLINE_MIN)


# ===========================================================================
# 1. ULTRA_CHOP
# ===========================================================================
# Owner's anchor (2024_09_16 10:23:50-10:24:31): ~24 direction flips in 42s
# inside a 13.25pt box. Measured on 1s closes the anchor's 60s windows carry
# 27-33 flips (day RTH p50 = 27, p90 = 32 -> genuinely elevated) but a
# 15.5-24.0pt box (day RTH p50 = 11.0, p90 = 21.0 -> ABOVE median, because the
# 60s window swallows the 11.25pt one-second flush at 10:24:11).
#
# Consequence, recorded here because it drove the design: an ABSOLUTE
# "small box" threshold that fires at the anchor (box <= 24pt) also fires on
# ~40% of all RTH bars and is useless. The anchor is NOT a tight-box event at
# 60s resolution. So "small net range" is implemented RELATIVE to the market's
# own recent scale (below), which is also the only era-robust choice: MNQ ran
# 16k in 2024 and 28k in 2026 and absolute point thresholds drift with it.
CHOP_WIN_S = 60                  # owner's stated window
CHOP_MIN_BARS = 40               # 1s bars required inside the window; day p1 of
                                 # window bar-count is 48, so 40 only rejects
                                 # genuinely gappy tape (holidays, halts)
CHOP_MIN_FLIPS = 30              # corpus RTH p75 of flips/60s (~30) - "high
                                 # direction-flip rate" means top quartile
CHOP_BOX_AMBIENT_FRAC = 0.60     # window box <= 60% of the ambient 1-minute box
CHOP_AMBIENT_MIN = 60            # ambient = median of the last 60 non-overlapping
                                 # 1-minute boxes (causal, trailing hour)
CHOP_ESCAPE_MAX_S = 30 * 60      # escape scan horizon; beyond this -> censored
CHOP_ESCAPE_BUF_FRAC = 0.50      # escape = close beyond the box edge by half a
                                 # box height. Without a buffer the "escape" is
                                 # a 1-tick drift out of a trailing 60s range
                                 # and resolves in a median 8s — measuring
                                 # nothing. Scale-free (fraction of the box).
CHOP_REFRACTORY_S = 60           # one event per chop episode: never two fires
                                 # inside one window length of each other


def _flip_flags(close: np.ndarray) -> np.ndarray:
    """1.0 on bars whose non-zero close-to-close move reverses the sign of the
    previous non-zero move. Vectorised."""
    d = np.diff(close, prepend=close[0])
    s = np.sign(d)
    nz = np.flatnonzero(s != 0)
    out = np.zeros(close.size)
    if nz.size > 1:
        sn = s[nz]
        out[nz[1:]] = (sn[1:] != sn[:-1]).astype(float)
    return out


def detect_ultra_chop(d: Day) -> list[dict]:
    """1s bars. Fires at the first bar of a chop episode: a 60s window with
    top-quartile flip density whose high-low box is <= 60% of the ambient
    1-minute box (median of the trailing 60 one-minute boxes).

    Causality: every quantity is a trailing window ending at the firing bar.
    The chop BOX that defines the escape is the firing window's box, known at
    the firing bar. Episodes are de-duplicated forward: after a fire, no new
    event until price has escaped that box (or CHOP_ESCAPE_MAX_S elapses).
    """
    if d.n < CHOP_MIN_BARS:
        return []
    idx = pd.DatetimeIndex(pd.to_datetime(d.ts, unit="s", utc=True))
    flip = _flip_flags(d.close)
    frame = pd.DataFrame({"h": d.high, "l": d.low, "c": d.close, "f": flip},
                         index=idx)
    r = frame.rolling(f"{CHOP_WIN_S}s")
    box_hi = r["h"].max().to_numpy()
    box_lo = r["l"].min().to_numpy()
    flips = r["f"].sum().to_numpy()
    cnt = r["c"].count().to_numpy()
    box = box_hi - box_lo

    # ambient scale: non-overlapping 1-minute boxes -> rolling median of the
    # last CHOP_AMBIENT_MIN of them, mapped back to bars CAUSALLY (a bar sees
    # only minutes strictly before its own).
    minute = d.ts // 60
    mfirst = np.flatnonzero(np.diff(minute, prepend=minute[0] - 1) != 0)
    mhi = np.maximum.reduceat(d.high, mfirst)
    mlo = np.minimum.reduceat(d.low, mfirst)
    mbox = pd.Series(mhi - mlo)
    amb_per_min = mbox.rolling(CHOP_AMBIENT_MIN).median().to_numpy()
    # bar i sees the ambient computed through the PREVIOUS minute
    slot = np.searchsorted(minute[mfirst], minute, side="left")
    prev_slot = slot - 1
    ambient = np.where(prev_slot >= 0, amb_per_min[np.maximum(prev_slot, 0)], np.nan)

    ok = (d.rth_mask() & (cnt >= CHOP_MIN_BARS) & (flips >= CHOP_MIN_FLIPS)
          & np.isfinite(ambient) & (ambient > 0)
          & (box <= CHOP_BOX_AMBIENT_FRAC * ambient))
    cand = np.flatnonzero(ok)
    if cand.size == 0:
        return []

    rows: list[dict] = []
    guard_ts = -1
    for i in cand:
        i = int(i)
        if d.ts[i] <= guard_ts:
            continue
        lo, hi = float(box_lo[i]), float(box_hi[i])
        buf = CHOP_ESCAPE_BUF_FRAC * float(box[i])
        # forward escape scan (this is the de-dup driver, not an event feature)
        j_end = int(np.searchsorted(d.ts, d.ts[i] + CHOP_ESCAPE_MAX_S, side="right"))
        seg = d.close[i + 1:j_end]
        out_mask = (seg > hi + buf) | (seg < lo - buf)
        k = int(np.argmax(out_mask)) if out_mask.any() else -1
        esc_ts = (int(d.ts[i + 1 + k]) if k >= 0
                  else int(d.ts[i]) + CHOP_ESCAPE_MAX_S)
        guard_ts = max(esc_ts, int(d.ts[i]) + CHOP_REFRACTORY_S)
        rows.append(dict(
            day=d.day, ts=int(d.ts[i]), i=i, box_lo=lo, box_hi=hi,
            esc_buf=float(buf),
            box_pt=float(box[i]), flips=float(flips[i]),
            ambient_pt=float(ambient[i]),
            box_ambient_ratio=float(box[i] / ambient[i]),
            mid_px=float(d.close[i]), mod=float(d.mod[i])))
    return rows


# ===========================================================================
# 2/3/4. LEG_DESCENT, FAKEOUT_POKE, STALL — one shared 5s pass
# ===========================================================================
# All three ride the repo-canonical 8.0pt close zigzag, so "leg" means exactly
# what it means in research/reversal_gauge.

# --- LEG_DESCENT -----------------------------------------------------------
DEFENSE_PT = 2.0                 # owner: "fast V-up >= 2pt within seconds"
DEFENSE_WIN_S = 30               # "within seconds" -> 30s (6 x 5s bars)
LOWER_HIGH_TOL_PT = 2.0          # a push whose high exceeds the previous push's
                                 # high by <= 2pt is a POKE, not a new high
                                 # (same 2pt tolerance as FAKEOUT_POKE)

# --- FAKEOUT_POKE ----------------------------------------------------------
POKE_MAX_PT = 2.0                # owner: "pokes beyond a recent extreme by <=2pt"
POKE_RETURN_S = 60               # "...then returns inside within 60s"

# --- STALL -----------------------------------------------------------------
STALL_MIN_MFE_PT = 8.0           # leg must be a real leg (= ZZ_REVERSAL_PT)
STALL_MIN_S = 10 * 60            # owner / four_phase_cohort: ">= 10min"
STALL_GIVE_FRAC = 0.30           # "...holding within 30% giveback" (four_phase)
STALL_EXT_FRAC = 0.25            # a candidate peak that extends by more than
                                 # 25% of leg MFE is the leg still RUNNING, not
                                 # stalling -> the stall clock restarts there


def scan_5s(d: Day) -> dict[str, list[dict]]:
    """One causal bar loop producing LEG_DESCENT, FAKEOUT_POKE and STALL
    events (plus their structurally matched control cohorts)."""
    zz = ZigZag(ZZ_REVERSAL_PT)
    ts, close, high, low, mod = d.ts, d.close, d.high, d.low, d.mod
    rth = (mod >= RTH_START_MIN) & (mod < RTH_END_MIN)

    descent: list[dict] = []
    poke: list[dict] = []
    stall: list[dict] = []

    # --- LEG_DESCENT state
    push_on = False
    push_high = push_high_i = 0
    run_lo = 0.0
    run_lo_i = 0
    push_defended = False
    prev_defended = False
    prev_high = np.inf
    prev_chain = 0
    chain_head_high = np.nan
    cur_chain = 0

    # --- FAKEOUT_POKE state
    extremes: list[tuple[int, float, int]] = []   # (dir, px, ts) confirmed
    poke_armed = False
    poke_dir = 0
    poke_ref = 0.0
    poke_ref_ts = 0
    poke_arm_i = 0
    poke_ext = 0.0

    # --- STALL state: a LIST of pending peak candidates, not one slot. With a
    # single slot a failed candidate blocks the next 10 minutes of tape, which
    # silently deleted most real stalls (0 stalls / 20 candidates on the
    # calibration day). Candidates are pruned three ways: VOID if the leg
    # extends past them (the leg was still running, not stalling), FAILED if
    # giveback exceeds the tolerance, else emitted at their 10-minute mark.
    pending: list[dict] = []
    last_cand_px = None
    last_cand_dir = 0

    for i in range(d.n):
        c = close[i]
        ev = zz.step(i, c)

        # ---------------- FAKEOUT_POKE ---------------------------------
        # Resolve an ARMED poke first, using the direction it was armed in —
        # a leg that reverses on this bar must not silently void a poke that
        # already snapped back inside.
        if poke_armed:
            dd = poke_dir
            if dd * (c - poke_ext) > 0:
                poke_ext = float(c)
            over = dd * (poke_ext - poke_ref)
            kind = None
            if over > POKE_MAX_PT:
                kind = "BREAKOUT"            # exceeded the level for real
            elif dd * (c - poke_ref) < 0:
                kind = "RETURN"              # <- THE EVENT: poked and snapped back
            elif ts[i] - ts[poke_arm_i] > POKE_RETURN_S:
                kind = "STUCK"               # still outside after 60s
            if kind is not None:
                poke.append(dict(day=d.day, ts=int(ts[i]), i=i, dir=int(dd),
                                 ref_px=poke_ref, poke_ext=poke_ext,
                                 poke_depth=float(dd * (poke_ext - poke_ref)),
                                 ref_age_s=int(ts[i] - poke_ref_ts),
                                 arm_ts=int(ts[poke_arm_i]),
                                 kind=kind, mod=float(mod[i])))
                poke_armed = False

        if ev is not None:
            # every confirmed pivot is an extreme of the leg that just ended
            extremes.append((-ev["d"], float(ev["pivot_px"]), int(ts[ev["pivot_i"]])))

        if (not poke_armed) and ev is None and zz.d != 0 \
                and zz.peak_i == i and rth[i]:
            # arm on a fresh running extreme that has just edged past a
            # remembered same-direction extreme by <= POKE_MAX_PT
            dd = zz.d
            t = int(ts[i])
            for ld, lpx, lts in reversed(extremes):
                if t - lts > EXTREME_MEMORY_S:
                    break
                if ld == dd and 0.0 < dd * (zz.peak_px - lpx) <= POKE_MAX_PT:
                    poke_armed = True
                    poke_dir = dd
                    poke_ref = lpx
                    poke_ref_ts = lts
                    poke_arm_i = i
                    poke_ext = float(zz.peak_px)
                    break

        # ---------------- LEG_DESCENT ----------------------------------
        if ev is not None and ev["kind"] == "pivot" and ev["d"] == -1:
            # a swing HIGH just confirmed -> a new down-push begins
            if push_on:
                prev_high = push_high
                prev_defended = push_defended
                prev_chain = cur_chain if push_defended else 0
            push_on = True
            push_high = float(ev["pivot_px"])
            push_high_i = int(ev["pivot_i"])
            run_lo = float(low[i])
            run_lo_i = i
            push_defended = False
        elif ev is not None and ev["kind"] == "pivot" and ev["d"] == 1 and push_on:
            # swing LOW confirmed -> the push is over
            prev_high = push_high
            prev_defended = push_defended
            prev_chain = cur_chain if push_defended else 0
            push_on = False
        elif push_on:
            if low[i] < run_lo:
                run_lo = float(low[i])
                run_lo_i = i
            if (not push_defended) and rth[i] \
                    and (ts[i] - ts[run_lo_i]) <= DEFENSE_WIN_S \
                    and (c - run_lo) >= DEFENSE_PT:
                push_defended = True
                if prev_defended and push_high <= prev_high + LOWER_HIGH_TOL_PT:
                    cur_chain = prev_chain + 1
                else:
                    cur_chain = 1
                    chain_head_high = push_high
                descent.append(dict(
                    day=d.day, ts=int(ts[i]), i=i, chain_n=int(cur_chain),
                    step_high=push_high, step_low=run_lo,
                    step_depth=float(push_high - run_lo),
                    chain_head_high=float(chain_head_high),
                    chain_descent=float(chain_head_high - run_lo),
                    defense_pt=float(c - run_lo),
                    defense_lag_s=int(ts[i] - ts[run_lo_i]),
                    mod=float(mod[i])))

        # ---------------- STALL ----------------------------------------
        # p['dir'] is pinned at candidate open: a leg reversal must not flip
        # the sign of the giveback measurement mid-candidate.
        if pending:
            keep = []
            for p in pending:
                dd = p["dir"]
                if dd * (c - p["px"]) > STALL_EXT_FRAC * p["mfe"]:
                    continue                      # VOID: leg still running
                give = dd * (p["px"] - c)
                if give > p["maxgive"]:
                    p["maxgive"] = float(give)
                if p["maxgive"] > STALL_GIVE_FRAC * p["mfe"]:
                    p["failed"] = True
                if ts[i] - ts[p["i"]] >= STALL_MIN_S:
                    stall.append(dict(
                        day=d.day, ts=int(ts[i]), i=i, dir=int(dd),
                        peak_i=p["i"], peak_ts=int(ts[p["i"]]),
                        peak_px=p["px"], mfe_pt=p["mfe"],
                        give_frac=float(p["maxgive"] / p["mfe"]),
                        stalled=bool(not p["failed"]),
                        mod=float(mod[i])))
                else:
                    keep.append(p)
            pending = keep

        if zz.d != 0 and zz.peak_i == i and zz.mfe >= STALL_MIN_MFE_PT and rth[i]:
            dd = int(zz.d)
            # de-dup: a fresh candidate only once the extreme has advanced by
            # STALL_EXT_FRAC of leg MFE past the last one opened
            if (last_cand_px is None or last_cand_dir != dd
                    or dd * (c - last_cand_px) > STALL_EXT_FRAC * zz.mfe):
                pending.append(dict(i=i, dir=dd, px=float(c),
                                    mfe=float(zz.mfe), maxgive=0.0,
                                    failed=False))
                last_cand_px, last_cand_dir = float(c), dd

    return dict(leg_descent=descent, fakeout_poke=poke, stall=stall)


# ===========================================================================
# 5. DEFENDED_POKE_AT_SHELF  +  6. FLUSH_V_DAY — one shared 1m pass
# ===========================================================================
# Windows deliberately mirror research/dojo_forge/tools/vshape_retest_cohort.py
# (1m bars, 3-bar poke, 5-bar defense, 5pt retest tolerance) so the flushV
# sub-cohort is directly comparable to its published 1/72 = 1.4% [0%, 7%].
SHELF_LOOKBACK_MIN = 120         # dwell window = prior 2h of 1m closes
SHELF_MIN_BARS = 100             # of 120 possible; rejects thin pre-open windows
SHELF_BIN_PT = 2.0               # dwell histogram bin (vshape used 2pt bins)
SHELF_MIN_DWELL_FRAC = 0.08      # mode bin must hold >= 8% of the 2h closes.
                                 # A 2h window spread over a 60pt range in 2pt
                                 # bins is 30 bins -> uniform = 3.3%; 8% is 2.4x
                                 # uniform, i.e. a genuine dwell shelf.
SHELF_AWAY_PT = 10.0             # price must have been this far ABOVE the shelf
SHELF_APPROACH_MIN = 30          # ...within the last 30 minutes (an approach)
SHELF_RETEST_PT = 5.0            # trigger: low <= shelf + 5   (vshape RETEST_PT)
SHELF_POKE_BARS = 3              # poke extreme = min low over 3 bars (vshape)
SHELF_DEF_BARS = 5               # defense must print within 5 bars (vshape)
SHELF_DEF_PT = 5.0               # ">= 5pt bounce"                (vshape DEF_PT)
SHELF_REFRACTORY_MIN = 30        # bars before the same shelf may fire again


def scan_1m(d: Day) -> dict[str, list[dict]]:
    ts, opn, high, low, close, mod = d.ts, d.open, d.high, d.low, d.close, d.mod
    n = d.n

    # ---------------- FLUSH_V_DAY (imported detector) -------------------
    # One row per day either way: is_flush=True at the causal confirm bar,
    # is_flush=False at the matched 10:20 clock bar (the flushV deadline), so
    # the day-class table has a same-construction control instead of a
    # free-floating "all days" number.
    flush_ts = _flush_confirm_ts(ts, opn, high, low, mod)
    flush_rows: list[dict] = []
    win = np.flatnonzero((mod >= RTH_START_MIN) & (mod < FLUSH_WIN_END_MIN))
    if win.size:
        open_px = float(opn[win[0]])
        rel = int(np.argmin(low[win]))
        v_low = float(low[win[rel]])
        t_low_i = int(win[rel])
        if flush_ts is not None:
            ci = min(int(np.searchsorted(ts, flush_ts, side="left")), n - 1)
        else:
            # matched control anchor: the flushV recovery deadline, BOUNDED on
            # both sides (an open-ended `mod >= X` would land in the prior
            # evening — the audit bug this package is built to avoid)
            ctrl = np.flatnonzero((mod >= RECOVERY_DEADLINE_MIN)
                                  & (mod < RECOVERY_DEADLINE_MIN + 60))
            ci = int(ctrl[0]) if ctrl.size else -1
        if ci >= 0 and ci > t_low_i:
            v_peak = float(high[t_low_i:ci + 1].max())
            flush_pt = float(open_px - v_low)
            flush_rows.append(dict(
                day=d.day, ts=int(ts[ci]), i=ci,
                is_flush=bool(flush_ts is not None),
                open_px=open_px, v_low=v_low, v_peak=v_peak,
                flush_pt=flush_pt,
                rec_frac=float((v_peak - v_low) / flush_pt) if flush_pt > 0
                else float("nan"),
                mod=float(mod[ci])))

    # ---------------- DEFENDED_POKE_AT_SHELF ----------------------------
    shelf_rows: list[dict] = []
    cool_until = -1
    i = SHELF_LOOKBACK_MIN
    while i < n:
        if not (RTH_START_MIN <= mod[i] < RTH_END_MIN) or i <= cool_until:
            i += 1
            continue
        seg = close[i - SHELF_LOOKBACK_MIN:i]          # STRICTLY prior bars
        if seg.size < SHELF_MIN_BARS:
            i += 1
            continue
        lo_e = np.floor(seg.min() / SHELF_BIN_PT) * SHELF_BIN_PT
        hi_e = np.ceil(seg.max() / SHELF_BIN_PT) * SHELF_BIN_PT + SHELF_BIN_PT
        edges = np.arange(lo_e, hi_e + SHELF_BIN_PT, SHELF_BIN_PT)
        if edges.size < 3:
            i += 1
            continue
        hist, _ = np.histogram(seg, bins=edges)
        k = int(np.argmax(hist))
        if hist[k] < SHELF_MIN_DWELL_FRAC * seg.size:
            i += 1
            continue
        shelf = float(edges[k] + SHELF_BIN_PT / 2.0)
        ap0 = max(0, i - SHELF_APPROACH_MIN)
        if float(high[ap0:i].max()) < shelf + SHELF_AWAY_PT:
            i += 1
            continue
        if low[i] > shelf + SHELF_RETEST_PT:
            i += 1
            continue
        # Walk the defense window bar by bar. CAUSALITY FIX (caught by
        # tools/causality_audit.py truncation replay): vshape takes the poke
        # extreme over 3 bars, but the defense can confirm on bar +1 — reading
        # the 3-bar min at that stamp would use up to 2 FUTURE bars. The poke
        # extreme is therefore the running min through the defense bar only.
        i_def = None
        poke_px = float(low[i])
        for j in range(i + 1, min(n, i + 1 + SHELF_DEF_BARS)):
            if j < i + SHELF_POKE_BARS:
                poke_px = min(poke_px, float(low[j]))
            if high[j] >= poke_px + SHELF_DEF_PT:
                i_def = j
                break
        if i_def is None:
            cool_until = i + SHELF_REFRACTORY_MIN
            i += 1
            continue
        day_class = ("flushV" if (flush_ts is not None and ts[i_def] >= flush_ts)
                     else "other")
        shelf_rows.append(dict(
            day=d.day, ts=int(ts[i_def]), i=i_def, trigger_i=i,
            trigger_ts=int(ts[i]), shelf_px=shelf, poke_px=poke_px,
            dwell_frac=float(hist[k] / seg.size),
            bounce_pt=float(high[i_def] - poke_px),
            day_class=day_class, mod=float(mod[i_def])))
        cool_until = i_def + SHELF_REFRACTORY_MIN
        i = i_def + 1

    return dict(defended_poke_shelf=shelf_rows, flush_v_day=flush_rows)


__all__ = ["detect_ultra_chop", "scan_5s", "scan_1m",
           "_flush_confirm_ts", "FLUSH_MIN_PT", "FLUSH_RECOVERY_FRAC",
           "RECOVERY_DEADLINE_MIN", "TICK_PT"]
