"""Forward-looking OUTCOME measurement for the event library.

Everything in this module reads bars STRICTLY AFTER an event's stamp `ts`.
Nothing here may be used inside `detectors.py` — the split is the causality
boundary. Forward scans are hard-clipped at 16:00 ET (`OUTCOME_END_MIN`) so no
outcome can read the evening session that opens the NEXT trading day's file.
"""
from __future__ import annotations

import numpy as np

from .common import (Day, TICK_PT, RTH_START_MIN, RTH_END_MIN,
                     OUTCOME_END_MIN, forward_slice)
from .detectors import (CHOP_ESCAPE_MAX_S, POKE_MAX_PT, STALL_GIVE_FRAC,
                        SHELF_DEF_PT)

# --- horizons ---------------------------------------------------------------
CHOP_MAG_MIN = (5, 15, 30)          # owner-specified read-out offsets, minutes
DESCENT_HORIZON_S = 30 * 60         # stair continuation race window
DEFENSE_HOLD_S = 5 * 60             # "does the defended low hold" window
POKE_HORIZON_S = 45 * 60            # = reversal_gauge LABEL_WINDOW_S
POKE_RESUME_PT = 0.5                # = reversal_gauge RESUME_EXCEED_PT
POKE_REVERSE_PT = 10.0              # adverse excursion that calls the poke a
                                    # failure. 10pt = the POCKET_CARD floor
                                    # stop; below it we are inside the noise.
STALL_HORIZON_S = 60 * 60           # what follows a stall
STALL_NEW_EXT_PT = 0.5              # = POKE_RESUME_PT (a real new extreme)
STALL_GIVE_RACE_FRAC = 0.50         # "50% giveback race" (owner)
SHELF_OUT_MIN = 90                  # = vshape_retest_cohort OUT_MIN (1m bars)
SHELF_CRACK_PT = 5.0                # = vshape CRACK_PT
SHELF_HOLD_PT = 15.0                # = vshape HOLD_PT
CONTROL_PER_DAY = 12                # random RTH anchors per day for the
                                    # unconditional magnitude baseline

# Every "does it continue?" race in this library is distance-ASYMMETRIC by
# construction (a new low is 1 tick away, breaking the stair is 10-15pt away),
# so its headline percentage is mostly mechanics, not information. Each such
# event therefore also carries a distance-SYMMETRIC race: +-SYM_PT from the
# event close, whose null is 50% by construction. 10pt = the POCKET_CARD floor
# stop, i.e. the smallest move the repo treats as outside the noise.
SYM_PT = 10.0


def _sym_race(d: Day, i0: int, cont_dir: int, horizon_s: int) -> tuple[str, int]:
    """First of +-SYM_PT from the event close, expressed as CONT / AGAINST."""
    j0, j1 = forward_slice(d, i0, horizon_s)
    seg = d.close[j0:j1]
    if seg.size == 0:
        return "NO_DATA", -1
    c0 = d.close[i0]
    up = np.flatnonzero(cont_dir * (seg - c0) >= SYM_PT)
    dn = np.flatnonzero(cont_dir * (c0 - seg) >= SYM_PT)
    i_up = j0 + int(up[0]) if up.size else None
    i_dn = j0 + int(dn[0]) if dn.size else None
    if i_up is not None and (i_dn is None or i_up < i_dn):
        return "CONT", int(d.ts[i_up] - d.ts[i0])
    if i_dn is not None:
        return "AGAINST", int(d.ts[i_dn] - d.ts[i0])
    return "NEITHER", -1


def _at_offset(d: Day, i0: int, secs: int) -> float:
    """Close `secs` after bar i0, or NaN if that lands past 16:00 / EOD."""
    t = int(d.ts[i0]) + int(secs)
    j = int(np.searchsorted(d.ts, t, side="right")) - 1
    if j <= i0 or j >= d.n:
        return np.nan
    if d.mod[j] >= OUTCOME_END_MIN:
        return np.nan
    return float(d.close[j])


# ===========================================================================
# 1. ULTRA_CHOP — escape statistics
# ===========================================================================
def outcomes_ultra_chop(d: Day, rows: list[dict]) -> list[dict]:
    for r in rows:
        i0 = r["i"]
        lo, hi = r["box_lo"], r["box_hi"]
        buf = r["esc_buf"]
        j0, j1 = forward_slice(d, i0, CHOP_ESCAPE_MAX_S)
        seg = d.close[j0:j1]
        out = (seg > hi + buf) | (seg < lo - buf)
        if not out.any():
            r.update(escaped=False, escape_lag_s=np.nan, escape_dir=0,
                     escape_px=np.nan,
                     **{f"mag_{m}m": np.nan for m in CHOP_MAG_MIN})
            continue
        k = j0 + int(np.argmax(out))
        sgn = 1 if d.close[k] > hi + buf else -1
        r.update(escaped=True, escape_lag_s=int(d.ts[k] - d.ts[i0]),
                 escape_dir=sgn, escape_px=float(d.close[k]))
        for m in CHOP_MAG_MIN:
            px = _at_offset(d, k, m * 60)
            r[f"mag_{m}m"] = (np.nan if not np.isfinite(px)
                              else float(sgn * (px - d.close[k])))
    return rows


def random_controls(d: Day, seed: int) -> list[dict]:
    """Unconditional |move| baseline at matched offsets from random RTH bars.
    Needed because 'the market moved 8pt in 15 minutes' is only interesting
    against what an arbitrary 15 minutes does."""
    idx = np.flatnonzero(d.rth_mask(RTH_START_MIN, RTH_END_MIN))
    if idx.size < CONTROL_PER_DAY:
        return []
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=CONTROL_PER_DAY, replace=False)
    out = []
    for i0 in np.sort(pick):
        i0 = int(i0)
        row = dict(day=d.day, ts=int(d.ts[i0]), mod=float(d.mod[i0]))
        for m in CHOP_MAG_MIN:
            px = _at_offset(d, i0, m * 60)
            row[f"abs_{m}m"] = (np.nan if not np.isfinite(px)
                                else abs(float(px - d.close[i0])))
        out.append(row)
    return out


# ===========================================================================
# 2. LEG_DESCENT — continuation after the Nth stair step
# ===========================================================================
def outcomes_leg_descent(d: Day, rows: list[dict]) -> list[dict]:
    for r in rows:
        i0 = r["i"]
        j0, j1 = forward_slice(d, i0, DESCENT_HORIZON_S)
        lows, highs = d.low[j0:j1], d.high[j0:j1]
        nl = np.flatnonzero(lows <= r["step_low"] - TICK_PT)
        sb = np.flatnonzero(highs >= r["step_high"] + TICK_PT)
        i_nl = j0 + int(nl[0]) if nl.size else None
        i_sb = j0 + int(sb[0]) if sb.size else None
        if i_nl is not None and (i_sb is None or i_nl < i_sb):
            race, t_res = "NEW_LOW", int(d.ts[i_nl] - d.ts[i0])
        elif i_sb is not None:
            race, t_res = "STAIR_BREAK", int(d.ts[i_sb] - d.ts[i0])
        else:
            race, t_res = "NEITHER", -1
        hold = (i_nl is None) or (d.ts[i_nl] - d.ts[i0] > DEFENSE_HOLD_S)
        px30 = _at_offset(d, i0, DESCENT_HORIZON_S)
        sr, srt = _sym_race(d, i0, -1, DESCENT_HORIZON_S)   # continuation = DOWN
        r.update(race=race, resolve_s=t_res, defense_hold=bool(hold),
                 sym_race=sr, sym_resolve_s=srt,
                 net_30m=(np.nan if not np.isfinite(px30)
                          else float(px30 - d.close[i0])))
    return rows


# ===========================================================================
# 3. FAKEOUT_POKE — resume vs reverse
# ===========================================================================
def outcomes_fakeout_poke(d: Day, rows: list[dict]) -> list[dict]:
    for r in rows:
        i0, dd = r["i"], r["dir"]
        j0, j1 = forward_slice(d, i0, POKE_HORIZON_S)
        seg = d.close[j0:j1]
        if seg.size == 0:
            r.update(race="NO_DATA", resolve_s=-1, exceed_ref=False,
                     exceed_poke=False, mfe_beyond_ref=np.nan)
            continue
        # 'never exceeds the prior extreme' == never clears it by more than the
        # poke tolerance (inside the tolerance is the poke itself, not a break)
        ex_ref = np.flatnonzero(dd * (seg - r["ref_px"]) > POKE_MAX_PT)
        ex_poke = np.flatnonzero(dd * (seg - r["poke_ext"]) >= POKE_RESUME_PT)
        rev = np.flatnonzero(dd * (d.close[i0] - seg) >= POKE_REVERSE_PT)
        # unbounded "ever exceeds within 45min" is ~always true (price wanders
        # 2pt past any level eventually), so the load-bearing version is
        # BOUNDED: clears the level before the adverse move calls it dead.
        i_exr = j0 + int(ex_ref[0]) if ex_ref.size else None
        i_res = j0 + int(ex_poke[0]) if ex_poke.size else None
        i_rev = j0 + int(rev[0]) if rev.size else None
        if i_res is not None and (i_rev is None or i_res < i_rev):
            race, t_res = "RESUME", int(d.ts[i_res] - d.ts[i0])
        elif i_rev is not None:
            race, t_res = "REVERSE", int(d.ts[i_rev] - d.ts[i0])
        else:
            race, t_res = "NEITHER", -1
        ext = float(seg.max() if dd > 0 else seg.min())
        sr, srt = _sym_race(d, i0, dd, POKE_HORIZON_S)
        r["sym_race"], r["sym_resolve_s"] = sr, srt
        r.update(race=race, resolve_s=t_res,
                 exceed_ref=bool(ex_ref.size), exceed_poke=bool(ex_poke.size),
                 exceed_ref_first=bool(i_exr is not None
                                       and (i_rev is None or i_exr < i_rev)),
                 mfe_beyond_ref=float(dd * (ext - r["ref_px"])))
    return rows


# ===========================================================================
# 4. STALL — new extreme vs 50% giveback
# ===========================================================================
def outcomes_stall(d: Day, rows: list[dict]) -> list[dict]:
    for r in rows:
        i0, dd = r["i"], r["dir"]
        j0, j1 = forward_slice(d, i0, STALL_HORIZON_S)
        seg = d.close[j0:j1]
        if seg.size == 0:
            r.update(race="NO_DATA", resolve_s=-1, net_60m=np.nan)
            continue
        ne = np.flatnonzero(dd * (seg - r["peak_px"]) >= STALL_NEW_EXT_PT)
        gv = np.flatnonzero(dd * (r["peak_px"] - seg)
                            >= STALL_GIVE_RACE_FRAC * r["mfe_pt"])
        i_ne = j0 + int(ne[0]) if ne.size else None
        i_gv = j0 + int(gv[0]) if gv.size else None
        if i_ne is not None and (i_gv is None or i_ne < i_gv):
            race, t_res = "NEW_EXTREME", int(d.ts[i_ne] - d.ts[i0])
        elif i_gv is not None:
            race, t_res = "GIVEBACK_50", int(d.ts[i_gv] - d.ts[i0])
        else:
            race, t_res = "NEITHER", -1
        px = _at_offset(d, i0, STALL_HORIZON_S)
        sr, srt = _sym_race(d, i0, dd, STALL_HORIZON_S)
        r.update(race=race, resolve_s=t_res, sym_race=sr, sym_resolve_s=srt,
                 net_60m=(np.nan if not np.isfinite(px)
                          else float(dd * (px - d.close[i0]))))
    return rows


# ===========================================================================
# 5. DEFENDED_POKE_AT_SHELF — crack vs hold (vshape_retest_cohort semantics)
# ===========================================================================
def outcomes_shelf(d: Day, rows: list[dict]) -> list[dict]:
    for r in rows:
        i0 = r["i"]
        j1 = min(d.n, i0 + SHELF_OUT_MIN + 1)
        after = np.flatnonzero(d.mod[i0 + 1:j1] >= OUTCOME_END_MIN)
        if after.size:
            j1 = i0 + 1 + int(after[0])
        lows, highs = d.low[i0 + 1:j1], d.high[i0 + 1:j1]
        ck = np.flatnonzero(lows <= r["poke_px"] - SHELF_CRACK_PT)
        hd = np.flatnonzero(highs >= r["poke_px"] + SHELF_HOLD_PT)
        i_ck = int(ck[0]) if ck.size else None
        i_hd = int(hd[0]) if hd.size else None
        if i_ck is not None and (i_hd is None or i_ck < i_hd):
            out, t_res = "CRACK", int(i_ck + 1)
        elif i_hd is not None:
            out, t_res = "HOLD", int(i_hd + 1)
        else:
            out, t_res = "UNRESOLVED", -1
        r.update(outcome=out, resolve_min=t_res)
    return rows


# ===========================================================================
# 6. FLUSH_V_DAY — what the V does after the classification is knowable
# ===========================================================================
def outcomes_flush_v(d: Day, rows: list[dict]) -> list[dict]:
    for r in rows:
        i0 = r["i"]
        j1 = d.n
        after = np.flatnonzero(d.mod[i0 + 1:] >= OUTCOME_END_MIN)
        if after.size:
            j1 = i0 + 1 + int(after[0])
        lows, highs = d.low[i0 + 1:j1], d.high[i0 + 1:j1]
        if lows.size == 0:
            r.update(low_break=False, peak_reclaim=False, first="NO_DATA",
                     close_frac=np.nan)
            continue
        lb = np.flatnonzero(lows < r["v_low"])
        pr = np.flatnonzero(highs > r["v_peak"])
        i_lb = int(lb[0]) if lb.size else None
        i_pr = int(pr[0]) if pr.size else None
        if i_lb is not None and (i_pr is None or i_lb < i_pr):
            first = "LOW_BREAK"
        elif i_pr is not None:
            first = "PEAK_RECLAIM"
        else:
            first = "NEITHER"
        rng = r["v_peak"] - r["v_low"]
        r.update(low_break=i_lb is not None, peak_reclaim=i_pr is not None,
                 first=first,
                 close_frac=(np.nan if rng <= 0 else
                             float((d.close[j1 - 1] - r["v_low"]) / rng)))
    return rows
