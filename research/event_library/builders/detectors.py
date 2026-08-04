"""EVENT LIBRARY v0 — the six owner-vocabulary detectors.

Every detector is STRICTLY CAUSAL: an event row is stamped at the first bar
where ALL of its defining conditions are observable (confirmation-time
stamping, same contract as the reversal_gauge zigzag). Forward-looking
quantities appear ONLY in outcome fields — the thing each table measures.

Per-detector causality self-audits live in the docstrings below and are
reproduced in the master report.

Timeframes: ULTRA_CHOP on 1s closes; LEG_DESCENT / FAKEOUT_POKE / STALL on
5s bars; DEFENDED_POKE_AT_SHELF / FLUSH_V_DAY on 1m bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.event_library.builders.common import (
    RTH_START_MIN, RTH_EVENT_END_MIN, SESSION_END_MIN,
)

# ---------------------------------------------------------------------------
# Constants (origin comments mandatory — no magic numbers)
# ---------------------------------------------------------------------------

# Shared close-based zigzag — same threshold as reversal_gauge's extractor so
# "leg" means the same thing across the research program.
REVERSAL_PT = 8.0

# --- ULTRA_CHOP (1s closes) ---
CHOP_WIN_S = 60            # task spec: rolling 60s window
CHOP_FLIP_MIN = 24         # anchor 2024_09_16 10:23:50-10:24:31: ~24 flips/42s
CHOP_RANGE_MAX_PT = 15.0   # anchor box 13.25pt, rounded up (margin ~1 bin)
CHOP_HORIZONS_MIN = (5, 15, 30)  # task spec: escape magnitude at +5/15/30min

# --- LEG_DESCENT (5s bars) ---
LD_PUSH_MIN_PT = 5.0       # push trigger: drop off the bounce high; anchor
                           # steps 09:56-10:24 are 8-28pt (probe 2026-08-03)
LD_DEFENSE_PT = 2.0        # task spec: fast V-up >= 2pt
LD_DEF_BARS = 6            # "within seconds" => <= 30s on 5s bars
LD_CONT_WIN_S = 60 * 60    # continuation race window after a step confirm
LD_HOLD_WIN_S = 10 * 60    # defense-hold horizon

# --- FAKEOUT_POKE (5s bars) ---
FP_POKE_MAX_PT = 2.0       # task spec: poke beyond the extreme by <= 2pt
FP_RETURN_WIN_S = 60       # task spec: back inside within 60s
FP_PRIOR_EXT_WIN_S = 90 * 60  # recent-extreme lookback; reversal_gauge REPOKE_WINDOW_S
FP_REV_PT = REVERSAL_PT    # REVERSE = one full zigzag reversal inside the poked extreme
FP_RACE_WIN_S = 60 * 60    # resume-vs-reverse race window

# --- STALL (5s bars, shared zigzag) ---
ST_MFE_MIN_PT = 15.0       # leg must be meaningful; ~2x the zigzag reversal
ST_HOLD_S = 10 * 60        # task spec: >= 10min holding
ST_GIVE_FRAC = 0.30        # task spec: within 30% giveback (four_phase STALL_GIVE)
ST_RESUME_EXCEED_PT = 0.5  # new-extreme confirm margin; reversal_gauge RESUME_EXCEED_PT
ST_FAIL_GIVE_FRAC = 0.50   # task spec: the 50%-giveback race target
ST_OUT_WIN_S = 120 * 60    # outcome race window

# --- DEFENDED_POKE_AT_SHELF (1m bars; mirrors vshape_retest_cohort for
#     comparability with its known flushV number) ---
DP_SHELF_LOOKBACK_S = 120 * 60  # task spec: dwell-mode level from the prior 2h
DP_SHELF_BIN_PT = 2.0      # vshape shelf histogram bin
DP_SHELF_DWELL_MIN = 15    # min closes in the modal 2pt bin (12.5% of a 2h
                           # window; ~2.5-3x uniform across a 40pt range)
DP_RETEST_PT = 5.0         # vshape RETEST_PT
DP_AWAY_BARS = 15          # bars clear of the shelf zone before a "retest"
                           # (makes it a return, not continuous dwell)
DP_POKE_BARS = 3           # vshape: poke extreme from first 3 retest bars
DP_DEF_BARS = 5            # vshape: defense within 5 bars
DP_DEF_PT = 5.0            # task spec + vshape DEF_PT: >= 5pt bounce
DP_CRACK_PT = 5.0          # vshape CRACK_PT
DP_HOLD_PT = 15.0          # vshape HOLD_PT
DP_OUT_MIN = 90            # vshape OUT_MIN
DP_FIRST_MIN = 10 * 60.0   # first trigger scan 10:00 ET (>=2h of history)
DP_LAST_MIN = 14 * 60 + 30.0  # last trigger 14:30 ET (room for the 90min race)

# --- FLUSH_V_DAY (1m bars; constants = reversal_gauge _flush_confirm_ts) ---
FLUSH_MIN_PT = 60.0
FLUSH_RECOVERY_FRAC = 0.60
FLUSH_WIN_END_MIN = 9 * 60 + 50.0     # flush window [09:30, 09:50)
RECOVERY_DEADLINE_MIN = 10 * 60 + 20.0  # recovery at/before 10:20:00
FV_FATE_ANCHOR_MIN = 10 * 60 + 20.0   # fate measured from 10:20 — both the
                                      # flush window and the recovery deadline
                                      # have closed, so dump_low/v_peak are
                                      # fully observable at the anchor


# ---------------------------------------------------------------------------
# 1. ULTRA_CHOP
# ---------------------------------------------------------------------------
def detect_ultra_chop(b: dict) -> list[dict]:
    """ULTRA_CHOP on 1s closes.

    Definition: rolling 60s window (bars with ts in (t-60, t]); direction
    flips = sign changes between consecutive NONZERO close deltas (zero
    deltas carry the last sign); fire when flips >= CHOP_FLIP_MIN AND
    close-range of the window <= CHOP_RANGE_MAX_PT, confirm bar in RTH
    [09:30, 15:30). The chop box = [win_min, win_max] closes at confirm.
    One event per episode: no re-fire until price CLOSES outside the box
    (the escape) or the session ends.

    CAUSALITY AUDIT: window is trailing-only ((t-60, t]); box frozen at the
    confirm bar from trailing closes; escape/magnitudes are outcome fields.
    1s bars have gaps (p99 ~12s): windows are TIME-based via searchsorted,
    never bar-count, so a sparse window cannot smuggle in old bars. The flip
    counter needs >= 25 traded seconds in the window to reach 24 flips, so
    dead-tape windows cannot fire. No lookahead found.
    """
    ts, close, mod = b["ts"], b["close"], b["mod"]
    n = b["n"]
    if n < CHOP_WIN_S:
        return []
    delta = np.diff(close)
    sgn = np.sign(delta)
    nz = sgn != 0
    idx = np.where(nz, np.arange(n - 1), -1)
    last_nz = np.maximum.accumulate(idx)          # last nonzero-delta pos <= j
    prev_last = np.concatenate(([-1], last_nz[:-1]))
    prev_sign = np.where(prev_last >= 0, sgn[np.clip(prev_last, 0, None)], 0.0)
    flip = nz & (prev_sign != 0) & (sgn != prev_sign)   # flip lands on bar j+1
    flip_at_bar = np.concatenate(([False], flip))
    cf = np.concatenate(([0], np.cumsum(flip_at_bar)))
    w_start = np.searchsorted(ts, ts - CHOP_WIN_S, side="right")
    flips_w = cf[np.arange(n) + 1] - cf[w_start]

    s = pd.Series(close, index=pd.to_datetime(ts, unit="s", utc=True))
    roll = s.rolling(f"{CHOP_WIN_S}s")
    win_max = roll.max().to_numpy()
    win_min = roll.min().to_numpy()
    rng_w = win_max - win_min

    fire = ((flips_w >= CHOP_FLIP_MIN) & (rng_w <= CHOP_RANGE_MAX_PT)
            & (mod >= RTH_START_MIN) & (mod < RTH_EVENT_END_MIN))
    fire_idx = np.flatnonzero(fire)
    if not fire_idx.size:
        return []

    sess = np.flatnonzero(mod < SESSION_END_MIN)
    sess_end = int(sess[-1]) if sess.size else n - 1

    rows: list[dict] = []
    ptr = 0
    while ptr < fire_idx.size:
        i = int(fire_idx[ptr])
        box_lo, box_hi = float(win_min[i]), float(win_max[i])
        seg = slice(i + 1, sess_end + 1)
        out = np.flatnonzero((close[seg] > box_hi) | (close[seg] < box_lo))
        row = dict(day=b["day"], ts=int(ts[i]), hms=str(b["hms"][i]),
                   flips=int(flips_w[i]), box_lo=box_lo, box_hi=box_hi,
                   box_pt=round(box_hi - box_lo, 2))
        if out.size:
            j = i + 1 + int(out[0])
            esc_px = float(close[j])
            esc_dir = 1 if esc_px > box_hi else -1
            row.update(escaped=True, t_escape_s=int(ts[j] - ts[i]),
                       esc_dir=esc_dir, esc_hms=str(b["hms"][j]))
            for m in CHOP_HORIZONS_MIN:
                tgt = ts[j] + m * 60
                k = int(np.searchsorted(ts, tgt, side="right")) - 1
                if k <= j or k > sess_end or mod[k] >= SESSION_END_MIN:
                    row[f"mag{m}"] = np.nan
                else:
                    row[f"mag{m}"] = round(esc_dir * (close[k] - esc_px), 2)
            next_start = j + 1
        else:
            row.update(escaped=False, t_escape_s=-1, esc_dir=0, esc_hms="")
            for m in CHOP_HORIZONS_MIN:
                row[f"mag{m}"] = np.nan
            next_start = sess_end + 1
        rows.append(row)
        while ptr < fire_idx.size and fire_idx[ptr] < next_start:
            ptr += 1
    return rows


# ---------------------------------------------------------------------------
# 2. LEG_DESCENT (stair-down)
# ---------------------------------------------------------------------------
def detect_leg_descent(b: dict) -> list[dict]:
    """LEG_DESCENT on 5s bars, RTH only.

    Definition: a chain of DEFENDED lower-high pushes. A push begins when
    price prints LD_PUSH_MIN_PT below the running bounce high (the rejected
    peak). The push's defense = first close >= (min low of the push so far)
    + LD_DEFENSE_PT within LD_DEF_BARS bars of that low (covers both the
    long-lower-wick bar and the fast V-up). A defended push extends the
    chain iff its origin high < previous step's origin high AND its low <
    previous step's low; otherwise it starts a fresh chain. A slow
    (undefended) recovery or a close above the last step's origin high
    (reclaim) terminates the chain. An event row is stamped at the defense
    confirm of every step with step_n >= 2.

    CAUSALITY AUDIT: ceiling (origin high), floor (push low) and the defense
    close are all printed at/before the confirm bar. The defense clock
    restarts whenever the push makes a new low, so a defense is only ever
    measured against a low that is already LD_DEF_BARS bars old at most.
    Outcome fields (next-low / reclaim race, defense-hold) are forward
    scans, i.e. the measured thing. No lookahead found.
    """
    ts, high, low, close, mod = b["ts"], b["high"], b["low"], b["close"], b["mod"]
    rth = np.flatnonzero((mod >= RTH_START_MIN) & (mod < SESSION_END_MIN))
    if rth.size < 3:
        return []
    i0, i1 = int(rth[0]), int(rth[-1])

    BOUNCE, PUSH = 0, 1
    state = BOUNCE
    bounce_high = high[i0]
    steps: list[dict] = []          # current chain
    episode_seq = 0
    rows: list[dict] = []

    def flush_chain():
        nonlocal steps, episode_seq
        steps = []
        episode_seq += 1

    push_origin = push_low = 0.0
    push_low_i = 0

    for i in range(i0, i1 + 1):
        if state == BOUNCE:
            if high[i] > bounce_high:
                bounce_high = high[i]
            if steps and close[i] > steps[-1]["origin"]:
                flush_chain()                       # reclaim: chain over
            if bounce_high - low[i] >= LD_PUSH_MIN_PT:
                state = PUSH
                push_origin = bounce_high
                push_low, push_low_i = low[i], i
            continue
        # state == PUSH
        if low[i] < push_low:
            push_low, push_low_i = low[i], i
        if close[i] - push_low >= LD_DEFENSE_PT:
            if i - push_low_i <= LD_DEF_BARS:
                # impulse defense confirmed at bar i
                valid = (not steps
                         or (push_origin < steps[-1]["origin"]
                             and push_low < steps[-1]["low"]))
                if not valid:
                    flush_chain()
                steps.append(dict(origin=push_origin, low=push_low))
                if len(steps) >= 2 and mod[i] < RTH_EVENT_END_MIN:
                    rows.append(dict(
                        day=b["day"], ts=int(ts[i]), hms=str(b["hms"][i]),
                        i=i, episode=f'{b["day"]}#{episode_seq}',
                        step_n=len(steps), ceiling=float(push_origin),
                        floor=float(push_low),
                        depth_pt=round(steps[0]["origin"] - push_low, 2)))
            else:
                flush_chain()                       # slow recovery: chain broken
            state = BOUNCE
            bounce_high = high[i]
    # outcome post-pass
    for r in rows:
        i = r.pop("i")
        floor, ceiling = r["floor"], r["ceiling"]
        end_t = ts[i] + LD_CONT_WIN_S
        j = i + 1
        nl_t = rc_t = None
        while j <= i1 and ts[j] <= end_t:
            if nl_t is None and low[j] < floor:
                nl_t = int(ts[j] - ts[i])
            if rc_t is None and close[j] > ceiling:
                rc_t = int(ts[j] - ts[i])
            if nl_t is not None and rc_t is not None:
                break
            j += 1
        if nl_t is not None and (rc_t is None or nl_t <= rc_t):
            r["outcome"] = "NEXT_LOW"
        elif rc_t is not None:
            r["outcome"] = "RECLAIM"
        else:
            r["outcome"] = "NEITHER"
        r["t_next_low_s"] = nl_t if nl_t is not None else -1
        r["t_reclaim_s"] = rc_t if rc_t is not None else -1
        r["defense_held_10m"] = bool(nl_t is None or nl_t > LD_HOLD_WIN_S)
    return rows


# ---------------------------------------------------------------------------
# 3+4. FAKEOUT_POKE and STALL — one shared-zigzag pass
# ---------------------------------------------------------------------------
def detect_zigzag_events(b: dict) -> tuple[list[dict], list[dict]]:
    """FAKEOUT_POKE + STALL on 5s bars over the shared 8pt close zigzag
    (identical state machine to reversal_gauge extract_freeze_events).

    FAKEOUT_POKE definition: during an active leg, the leg's intrabar reach
    crosses the most recent SAME-direction prior zigzag extreme (confirmed
    pivot, printed within FP_PRIOR_EXT_WIN_S), the excursion beyond it never
    exceeds FP_POKE_MAX_PT, and a CLOSE returns inside within
    FP_RETURN_WIN_S of the first cross. Confirm = the return-inside bar.
    Excursions that exceed 2pt (genuine break), stay outside > 60s, or whose
    start was not observed (leg-birth jump) consume the target with no event.

    STALL definition: leg MFE >= ST_MFE_MIN_PT and, for ST_HOLD_S after the
    leg extreme printed, the worst close-giveback stays <= ST_GIVE_FRAC of
    MFE. Confirm = first bar at/after extreme_ts + ST_HOLD_S with the band
    intact. One stall per leg (first). NOTE: the shared zigzag kills a leg
    at 8pt giveback, so for MFE > 26.7pt the binding hold constraint is
    8pt, not 30% — recorded per-row as the actual band.

    CAUSALITY AUDIT (both): prior extremes enter the candidate list only at
    their CONFIRM bar (reversal confirmed), mirroring the reversal_gauge
    contract; poke state uses only current-bar reach/close; the fakeout
    confirm requires the return-inside close to have printed; the stall
    band uses the running max giveback since the extreme (trailing only).
    A same-bar poke+return (wick fakeout) is stamped on that single bar —
    both facts are observable at its close. Outcome fields are forward
    scans. No lookahead found.
    """
    ts, high, low, close, mod = b["ts"], b["high"], b["low"], b["close"], b["mod"]
    n = b["n"]
    if n < 2:
        return [], []

    sess = np.flatnonzero(mod < SESSION_END_MIN)
    sess_end = int(sess[-1]) if sess.size else n - 1

    fakeouts: list[dict] = []
    stalls: list[dict] = []
    prior_extremes: list[tuple[int, float, int]] = []  # (dir, px, extreme_ts)

    d = 0
    run_min_px, run_min_i = close[0], 0
    run_max_px, run_max_i = close[0], 0
    anchor_px, anchor_i = 0.0, 0
    peak_px, peak_i = 0.0, 0
    max_give = 0.0
    stall_fired = False
    consumed: set[tuple] = set()
    exc_target = None          # (px, ext_ts)
    exc_t0, exc_max = 0, 0.0

    def reset_leg_state():
        nonlocal max_give, stall_fired, consumed, exc_target
        max_give, stall_fired = 0.0, False
        consumed, exc_target = set(), None

    for i in range(n):
        c = close[i]
        if d == 0:
            if c < run_min_px:
                run_min_px, run_min_i = c, i
            if c > run_max_px:
                run_max_px, run_max_i = c, i
            up_conf = c - run_min_px >= REVERSAL_PT
            dn_conf = run_max_px - c >= REVERSAL_PT
            if up_conf and dn_conf:
                up_conf = run_min_i > run_max_i
                dn_conf = not up_conf
            if up_conf or dn_conf:
                d = 1 if up_conf else -1
                anchor_px, anchor_i = ((run_min_px, run_min_i) if up_conf
                                       else (run_max_px, run_max_i))
                prior_extremes.append((-d, float(anchor_px), int(ts[anchor_i])))
                peak_px, peak_i = c, i
                reset_leg_state()
            continue

        reach = high[i] if d == 1 else low[i]
        prev_peak = peak_px
        if d * (c - peak_px) > 0:
            peak_px, peak_i = c, i
            max_give = 0.0
        else:
            max_give = max(max_give, d * (peak_px - c))
        mfe = d * (peak_px - anchor_px)

        # ---- FAKEOUT_POKE ----
        if exc_target is None:
            for (ld, lpx, lts) in reversed(prior_extremes):
                if ts[i] - lts > FP_PRIOR_EXT_WIN_S:
                    break
                if ld != d or (ld, lpx, lts) in consumed:
                    continue
                if d * (reach - lpx) > 0:            # crossing now
                    if d * (prev_peak - lpx) > 0:
                        # already beyond before this bar: start unobserved
                        consumed.add((ld, lpx, lts))
                        continue
                    exc_target = (lpx, lts)
                    exc_t0, exc_max = int(ts[i]), 0.0
                break                                # nearest recent target only
        if exc_target is not None:
            t_px, t_ts = exc_target
            exc_max = max(exc_max, d * (reach - t_px))
            done = False
            if exc_max > FP_POKE_MAX_PT:
                done = True                          # genuine break, not a poke
            elif d * (c - t_px) < 0:                 # closed back inside
                if ts[i] - exc_t0 <= FP_RETURN_WIN_S:
                    if RTH_START_MIN <= mod[i] < RTH_EVENT_END_MIN:
                        fakeouts.append(dict(
                            day=b["day"], ts=int(ts[i]), hms=str(b["hms"][i]),
                            i=i, dir=int(d), target_px=float(t_px),
                            target_age_s=int(ts[i] - t_ts),
                            poke_ext=float(t_px + d * exc_max),
                            poke_depth_pt=round(exc_max, 2),
                            dur_s=int(ts[i] - exc_t0)))
                done = True
            elif ts[i] - exc_t0 > FP_RETURN_WIN_S:
                done = True                          # camped outside too long
            if done:
                consumed.add((d, t_px, t_ts))
                exc_target = None

        # ---- STALL ----
        if (not stall_fired and mfe >= ST_MFE_MIN_PT
                and ts[i] - ts[peak_i] >= ST_HOLD_S
                and max_give <= ST_GIVE_FRAC * mfe
                and RTH_START_MIN <= mod[i] < RTH_EVENT_END_MIN):
            stall_fired = True
            stalls.append(dict(
                day=b["day"], ts=int(ts[i]), hms=str(b["hms"][i]), i=i,
                dir=int(d), anchor_px=float(anchor_px), ext_px=float(peak_px),
                mfe_pt=round(mfe, 2), hold_band_pt=round(
                    min(ST_GIVE_FRAC * mfe, REVERSAL_PT), 2),
                give_frac_now=round(d * (peak_px - c) / mfe, 3)))

        if d * (peak_px - c) >= REVERSAL_PT:
            prior_extremes.append((d, float(peak_px), int(ts[peak_i])))
            anchor_px, anchor_i = peak_px, peak_i
            d = -d
            peak_px, peak_i = c, i
            reset_leg_state()

    # ---- outcome post-passes ----
    for r in fakeouts:
        i, dd = r.pop("i"), r["dir"]
        reach_arr = high if dd == 1 else low
        seg = slice(i + 1, sess_end + 1)
        ex = np.flatnonzero(dd * (reach_arr[seg] - r["poke_ext"]) > 0)
        rv = np.flatnonzero(dd * (close[seg] - r["target_px"]) <= -FP_REV_PT)
        ex_t = int(ts[i + 1 + ex[0]] - ts[i]) if ex.size else -1
        rv_t = int(ts[i + 1 + rv[0]] - ts[i]) if rv.size else -1
        r["exceed_sess"] = bool(ex_t >= 0)
        r["t_exceed_s"] = ex_t
        r["t_reverse_s"] = rv_t
        if 0 <= ex_t <= FP_RACE_WIN_S and (rv_t < 0 or ex_t <= rv_t):
            r["race60"] = "RESUME"
        elif 0 <= rv_t <= FP_RACE_WIN_S:
            r["race60"] = "REVERSE"
        else:
            r["race60"] = "UNRESOLVED"

    for r in stalls:
        i, dd = r.pop("i"), r["dir"]
        ext, mfe = r["ext_px"], r["mfe_pt"]
        seg = slice(i + 1, sess_end + 1)
        ne = np.flatnonzero(dd * (close[seg] - ext) >= ST_RESUME_EXCEED_PT)
        gb = np.flatnonzero(dd * (ext - close[seg]) >= ST_FAIL_GIVE_FRAC * mfe)
        ne_t = int(ts[i + 1 + ne[0]] - ts[i]) if ne.size else -1
        gb_t = int(ts[i + 1 + gb[0]] - ts[i]) if gb.size else -1
        if 0 <= ne_t <= ST_OUT_WIN_S and (gb_t < 0 or ne_t <= gb_t):
            r["outcome"] = "NEW_EXTREME"
        elif 0 <= gb_t <= ST_OUT_WIN_S:
            r["outcome"] = "GIVEBACK50"
        else:
            r["outcome"] = "UNRESOLVED"
        r["t_new_ext_s"] = ne_t
        r["t_give50_s"] = gb_t
    return fakeouts, stalls


# ---------------------------------------------------------------------------
# 5. DEFENDED_POKE_AT_SHELF
# ---------------------------------------------------------------------------
def flush_confirm_ts(b: dict) -> int | None:
    """Epoch ts at which the flushV day-class becomes knowable, else None.
    Port of reversal_gauge _flush_confirm_ts INCLUDING its audit fix: day
    files start 18:00 the previous evening, so every time-of-day selector is
    bounded to the morning (an unbounded `mod >= X` first-match would land
    on prior-evening bars)."""
    ts, opn, high, low, mod = b["ts"], b["open"], b["high"], b["low"], b["mod"]
    win = np.flatnonzero((mod >= RTH_START_MIN) & (mod < FLUSH_WIN_END_MIN))
    if win.size == 0:
        return None
    open_px = opn[win[0]]
    rel_min = int(np.argmin(low[win]))
    flush_low = low[win[rel_min]]
    t_low_idx = win[rel_min]
    flush_pt = open_px - flush_low
    if flush_pt < FLUSH_MIN_PT:
        return None
    target = flush_low + FLUSH_RECOVERY_FRAC * flush_pt
    idx = np.arange(ts.size)
    rec = np.flatnonzero((idx > t_low_idx) & (mod <= RECOVERY_DEADLINE_MIN)
                         & (high >= target))
    if rec.size == 0:
        return None
    closed = np.flatnonzero((mod >= FLUSH_WIN_END_MIN)
                            & (mod < RECOVERY_DEADLINE_MIN + 60))
    if closed.size == 0:
        return None
    return int(max(ts[rec[0]], ts[closed[0]]))


def detect_defended_poke(b: dict, flush_ts: int | None) -> list[dict]:
    """DEFENDED_POKE_AT_SHELF on 1m bars — vshape_retest_cohort generalized
    beyond flush-V days to ANY high-dwell shelf on ANY day.

    Definition: at bar t (10:00-14:30 ET), shelf = center of the modal
    DP_SHELF_BIN_PT close-bin over the prior 2h (bars strictly before t),
    valid only if the modal bin holds >= DP_SHELF_DWELL_MIN closes. Retest
    trigger (support side): low[t] enters shelf+DP_RETEST_PT after ALL of
    the prior DP_AWAY_BARS bars stayed clear (approach from above);
    resistance side mirrored. Defense: first bar j in (t, t+DP_DEF_BARS]
    whose bounce off the running poke extreme (min low over COMPLETED bars
    t..j-1) is >= DP_DEF_PT. Confirm = bar j. Outcome (the vshape race):
    within DP_OUT_MIN bars of j, CRACK if the level side gives way
    (poke_ext -/+ DP_CRACK_PT) before the defended side runs DP_HOLD_PT.
    day_class = 'flushV' iff the day's flush confirm ts <= confirm ts
    (causal tagging), else 'other'.

    CAUSALITY AUDIT: shelf histogram uses bars strictly before the trigger
    bar; the away condition uses completed prior bars; the defense compares
    bar j's high against lows of COMPLETED bars only (stricter than vshape,
    whose poke extreme could deepen through bar t+2 after the defense —
    that intrabar-order ambiguity is removed here); flushV tagging requires
    the flush confirm ts to have passed. The outcome race starts AT the
    defense bar (vshape convention) — bar j's own low can in principle both
    defend and crack within one minute; kept for comparability, noted.
    """
    ts, high, low, close, mod = b["ts"], b["high"], b["low"], b["close"], b["mod"]
    n = b["n"]
    scan = np.flatnonzero((mod >= DP_FIRST_MIN) & (mod <= DP_LAST_MIN))
    if not scan.size:
        return []
    rows: list[dict] = []
    next_allowed = 0
    for t in scan:
        t = int(t)
        if t < next_allowed or t < DP_AWAY_BARS:
            continue
        lb = int(np.searchsorted(ts, ts[t] - DP_SHELF_LOOKBACK_S, side="right"))
        seg = close[lb:t]
        if seg.size < DP_SHELF_DWELL_MIN:
            continue
        edges = np.arange(np.floor(seg.min()), seg.max() + DP_SHELF_BIN_PT,
                          DP_SHELF_BIN_PT)
        if edges.size < 2:
            continue
        hist, _ = np.histogram(seg, bins=edges)
        if hist.max() < DP_SHELF_DWELL_MIN:
            continue
        shelf = float(edges[int(np.argmax(hist))] + DP_SHELF_BIN_PT / 2)
        away = slice(t - DP_AWAY_BARS, t)
        sup = (low[t] <= shelf + DP_RETEST_PT
               and bool(np.all(low[away] > shelf + DP_RETEST_PT)))
        res = (high[t] >= shelf - DP_RETEST_PT
               and bool(np.all(high[away] < shelf - DP_RETEST_PT)))
        if sup == res:                       # neither, or contradictory wide bar
            continue
        side = 1 if sup else -1              # 1 = support test (from above)
        # defense scan: poke extreme over completed bars t..j-1
        def_j = None
        poke_ext = low[t] if sup else high[t]
        for j in range(t + 1, min(t + DP_DEF_BARS, n - 1) + 1):
            if side == 1 and high[j] >= poke_ext + DP_DEF_PT:
                def_j = j
                break
            if side == -1 and low[j] <= poke_ext - DP_DEF_PT:
                def_j = j
                break
            if j - t < DP_POKE_BARS:         # poke window: extreme may deepen
                poke_ext = min(poke_ext, low[j]) if sup else max(poke_ext, high[j])
        if def_j is None:
            next_allowed = t + DP_POKE_BARS + DP_DEF_BARS
            continue
        # outcome race from the defense bar
        ow = slice(def_j, min(def_j + DP_OUT_MIN, n - 1) + 1)
        if side == 1:
            crack = np.flatnonzero(low[ow] <= poke_ext - DP_CRACK_PT)
            hold = np.flatnonzero(high[ow] >= poke_ext + DP_HOLD_PT)
        else:
            crack = np.flatnonzero(high[ow] >= poke_ext + DP_CRACK_PT)
            hold = np.flatnonzero(low[ow] <= poke_ext - DP_HOLD_PT)
        ci = int(crack[0]) if crack.size else None
        hi_ = int(hold[0]) if hold.size else None
        if ci is not None and (hi_ is None or ci < hi_):
            outcome = "CRACK"
        elif hi_ is not None:
            outcome = "HOLD"
        else:
            outcome = "UNRESOLVED"
        day_class = ("flushV" if flush_ts is not None and ts[def_j] >= flush_ts
                     else "other")
        rows.append(dict(
            day=b["day"], ts=int(ts[def_j]), hms_trigger=str(b["hms"][t]),
            hms=str(b["hms"][def_j]), side="SUP" if sup else "RES",
            shelf=shelf, dwell=int(hist.max()), poke_ext=float(poke_ext),
            day_class=day_class, outcome=outcome,
            t_resolve_min=(ci if outcome == "CRACK" else hi_
                           if outcome == "HOLD" else -1)))
        next_allowed = def_j + DP_DEF_BARS
    return rows


# ---------------------------------------------------------------------------
# 6. FLUSH_V_DAY
# ---------------------------------------------------------------------------
def detect_flush_v_day(b: dict, flush_ts: int | None) -> dict | None:
    """FLUSH_V_DAY on 1m bars — day-class detector + V-fate outcomes.

    Definition (= reversal_gauge, audit-fixed): 09:30 open minus min low
    over [09:30, 09:50) >= FLUSH_MIN_PT AND a bar high recovers
    FLUSH_RECOVERY_FRAC of the flush at/before 10:20 ET. Confirm ts =
    flush_confirm_ts (both the flush window closed and the recovery bar
    printed). Emits a row for EVERY day (is_flushv flag) so the fate table
    has an unconditional base rate. Fate quantities are anchored at 10:20
    (dump_low, v_peak = max high from the flush low through 10:20 — fully
    observable there); outcomes scan bars after 10:20 to 16:00.

    CAUSALITY AUDIT: all selectors bounded to the morning (prior-evening
    bars cannot match — the exact bug class fixed in reversal_gauge);
    confirm ts >= max(recovery bar, flush-window close); fate anchor 10:20
    is at/after every input's print time. Outcome fields forward-only. On
    non-flush days the same 10:20-anchored quantities are emitted for the
    base-rate row. No lookahead found.
    """
    ts, opn, high, low, close, mod = (b["ts"], b["open"], b["high"], b["low"],
                                      b["close"], b["mod"])
    win = np.flatnonzero((mod >= RTH_START_MIN) & (mod < FLUSH_WIN_END_MIN))
    if win.size == 0:
        return None
    open_px = float(opn[win[0]])
    rel = int(np.argmin(low[win]))
    dump_low = float(low[win[rel]])
    t_low_idx = int(win[rel])
    flush_pt = open_px - dump_low
    vwin = np.flatnonzero((np.arange(ts.size) > t_low_idx)
                          & (mod <= FV_FATE_ANCHOR_MIN))
    v_peak = float(high[vwin].max()) if vwin.size else dump_low
    post = np.flatnonzero((mod > FV_FATE_ANCHOR_MIN) & (mod <= SESSION_END_MIN))
    if not post.size:
        return None
    lb = np.flatnonzero(low[post] < dump_low)
    pr = np.flatnonzero(high[post] > v_peak)
    li = int(lb[0]) if lb.size else None
    pi = int(pr[0]) if pr.size else None
    first = ("LOW_BREAK" if li is not None and (pi is None or li < pi)
             else "PEAK_RECLAIM" if pi is not None else "NEITHER")
    close_px = float(close[post[-1]])
    denom = v_peak - dump_low
    close_frac = (close_px - dump_low) / denom if denom > 0 else np.nan
    eti = pd.to_datetime(flush_ts, unit="s", utc=True).tz_convert(
        "America/New_York").strftime("%H:%M:%S") if flush_ts else ""
    return dict(day=b["day"], is_flushv=bool(flush_ts is not None),
                confirm_hms=eti, flush_pt=round(flush_pt, 2),
                dump_low=dump_low, v_peak=v_peak,
                low_break=bool(li is not None), peak_reclaim=bool(pi is not None),
                first=first, close_frac=round(float(close_frac), 3)
                if np.isfinite(close_frac) else np.nan)
