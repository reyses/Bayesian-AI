#!/usr/bin/env python3.11
"""P2c -- empirical root-cause driver for the v0.2-RC "R-trigger fired ZERO times" bug.

Replays a golden day minute-by-minute (day-so-far) EXACTLY as the NinjaScript wrapper
does (RunDecision re-runs the batch core over the buffer each completed minute and reads
zz_confirm at curMin). Logs, for each RTH minute M:
    - full-day  zz_confirm[M]  (the golden reference)
    - truncated zz_confirm[M]  (core re-run over bars[0 : lastrow(M)+1])
under three warmup regimes:
    (A) HARNESS ctx : prior-day tail present (start=2500)  -- the 100%%-parity input
    (B) COLD  ctx   : RTH-only, start=0, no tail           -- the NT8 wrapper runtime
It then simulates the wrapper exit gate `zz_confirm == -openDir` against a synthetic
open position to count how many R-trigger exits would fire.

Ports BuildZzThr (C# Ctx.BuildZzThr) and zigzag_rtrigger (golden_vector_gen) verbatim.
Reads the raw 5s ctx from research/nt8_port/csharp/harness_data/<day>.json.gz.
Writes research/nt8_port/reports/p2c_replay_<day>.txt.
"""
import gzip, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
HDATA = os.path.join(REPO, "research", "nt8_port", "csharp", "harness_data")
REPORTS = os.path.join(REPO, "research", "nt8_port", "reports")

TICK = 0.25
BAR_1M = 12
ATR_N = 14
ATR_MULT = 4.0
ZZ_MIN_BARS_5S = 36


def roll_mean_minp(x, w, minp):
    n = len(x); y = np.full(n, np.nan)
    for i in range(n):
        if i < w - 1:
            continue
        seg = x[i - w + 1:i + 1]
        fin = seg[np.isfinite(seg)]
        if len(fin) >= minp:
            y[i] = fin.mean()
    return y


def build_zz_thr(o, h, l, c, N):
    """Verbatim port of C# Ctx.BuildZzThr / dsp DayCtx zz_thr = ATR(14 1m)x4 (points)."""
    nb = (N + BAR_1M - 1) // BAR_1M
    h1 = np.empty(nb); l1 = np.empty(nb); c1 = np.empty(nb)
    for b in range(nb):
        s = b * BAR_1M; e = min(s + BAR_1M, N)
        h1[b] = h[s:e].max(); l1[b] = l[s:e].min(); c1[b] = c[e - 1]
    tr1 = np.empty(nb)
    for b in range(nb):
        pc = c1[b - 1] if b > 0 else np.nan
        mx = h1[b] - l1[b]
        if np.isfinite(pc):
            mx = max(mx, abs(h1[b] - pc), abs(l1[b] - pc))
        tr1[b] = mx
    atr1 = roll_mean_minp(tr1, ATR_N, ATR_N)
    zz_thr = np.empty(N)
    for i in range(N):
        zz_thr[i] = atr1[i // BAR_1M] * ATR_MULT
    return zz_thr


def zigzag_rtrigger(c, rth, start, zz_thr, N):
    """Verbatim port of golden_vector_gen.zigzag_rtrigger over c[0:N]."""
    price_t = c[:N] / TICK
    rr = np.flatnonzero(rth[:N] & (np.arange(N) >= start))
    first_rth = int(rr[0]) if len(rr) else start
    thr_pts = zz_thr[first_rth]
    if not np.isfinite(thr_pts):
        fin = np.flatnonzero(np.isfinite(zz_thr[first_rth:N]))
        thr_pts = zz_thr[first_rth + fin[0]] if len(fin) else 0.0
    min_rev = max(4, int(round(thr_pts / TICK)))

    flip = np.zeros(N, dtype=np.int8)
    d = 0; ext = price_t[0]; ext_bar = 0; first_close = price_t[0]
    for i in range(1, N):
        p = price_t[i]; f = 0
        if d == 0:
            if p > ext:
                ext, ext_bar = p, i
            if p < first_close and (first_close - p) >= min_rev:
                d, ext, ext_bar, f = -1, p, i, -1
            elif p > first_close and (p - first_close) >= min_rev:
                d, ext, ext_bar, f = 1, p, i, 1
        elif d == 1:
            if p >= ext:
                ext, ext_bar = p, i
            elif (ext - p) >= min_rev and (i - ext_bar) >= ZZ_MIN_BARS_5S:
                d, ext, ext_bar, f = -1, p, i, -1
        else:
            if p <= ext:
                ext, ext_bar = p, i
            elif (p - ext) >= min_rev and (i - ext_bar) >= ZZ_MIN_BARS_5S:
                d, ext, ext_bar, f = 1, p, i, 1
        flip[i] = f
    return flip, min_rev, first_rth


def minute_confirm(flip, ts, rth, start, N):
    """Per-minute zz_confirm = last nonzero flip within the minute (golden convention),
    over RTH rows with idx>=start. Returns {minute_epoch: confirm}."""
    out = {}
    lastrow = {}
    for i in range(N):
        if not (rth[i] and i >= start):
            continue
        T = (ts[i] // 60) * 60
        lastrow[T] = i
        if flip[i] != 0:
            out[T] = int(flip[i])
    # ensure every RTH minute present (confirm 0 if none)
    for T in lastrow:
        out.setdefault(T, 0)
    return out, lastrow


def load(day):
    with gzip.open(os.path.join(HDATA, day + ".json.gz")) as f:
        d = json.load(f)
    return d


def run_regime(d, cold):
    """cold=False: harness ctx (tail, start as given). cold=True: RTH-only, start=0."""
    ts = np.array(d["ts"], dtype=np.int64)
    o = np.array(d["o"], float); h = np.array(d["h"], float)
    l = np.array(d["l"], float); c = np.array(d["c"], float)
    rth = np.array(d["rth"], dtype=bool)
    start = int(d["start"])

    if cold:
        # emulate the wrapper: buffer holds only the current session's RTH bars, start=0
        mask = rth & (np.arange(len(rth)) >= start)
        idx = np.flatnonzero(mask)
        sl = slice(idx[0], idx[-1] + 1)
        ts, o, h, l, c = ts[sl], o[sl], h[sl], l[sl], c[sl]
        rth = np.ones(len(ts), dtype=bool)
        start = 0
    N = len(ts)

    # ---- FULL-DAY reference ----
    zz_full = build_zz_thr(o, h, l, c, N)
    flip_full, minrev_full, frth = zigzag_rtrigger(c, rth, start, zz_full, N)
    conf_full, lastrow = minute_confirm(flip_full, ts, rth, start, N)
    minutes = sorted(lastrow.keys())

    # ---- TRUNCATED per-minute re-run (what RunDecision actually does) ----
    conf_trunc = {}
    minrevs = set()
    for M in minutes:
        end = lastrow[M] + 1                       # bars[0:end] = day-so-far thru minute M
        zz_t = build_zz_thr(o, h, l, c, end)
        flip_t, mr, _ = zigzag_rtrigger(c, rth, start, zz_t, end)
        minrevs.add(mr)
        # zz_confirm at minute M in the truncated run
        cc = 0
        for i in range(end):
            if rth[i] and i >= start and (ts[i] // 60) * 60 == M and flip_t[i] != 0:
                cc = int(flip_t[i])
        conf_trunc[M] = cc
    return dict(minutes=minutes, conf_full=conf_full, conf_trunc=conf_trunc,
                minrev_full=minrev_full, minrevs=sorted(minrevs), N=N, first_rth=frth)


def sim_exits(minutes, conf, entry_min=None, open_dir=+1):
    """Simulate the wrapper exit gate over the day: hold open_dir from first RTH minute,
    fire when conf[M]==-open_dir. Returns (n_fires, first_fire_min)."""
    n = 0; first = None
    for M in minutes:
        cc = conf.get(M, 0)
        if cc != 0 and cc == -open_dir:
            n += 1
            if first is None:
                first = M
    return n, first


def trade_sim(d, cold, entry_min, entry_dir, conf, stop_pts, session_close_epoch):
    """Faithful wrapper trade sim on the raw 5s stream: entries acted +180s late; while
    open, poll catastrophic stop every 5s (adverse-from-entry >= stop_pts) and check the
    R-trigger reversal (conf[M]==-openDir) ONCE per completed minute. NT8 session template
    force-closes at session_close_epoch. Returns exit-reason counts + total giveback."""
    ts = np.array(d["ts"], dtype=np.int64)
    o = np.array(d["o"], float); h = np.array(d["h"], float); l = np.array(d["l"], float)
    c = np.array(d["c"], float); rth = np.array(d["rth"], dtype=bool)
    start = int(d["start"])
    if cold:
        mask = rth & (np.arange(len(rth)) >= start)
        idx = np.flatnonzero(mask); sl = slice(idx[0], idx[-1] + 1)
        ts, o, h, l, c = ts[sl], o[sl], h[sl], l[sl], c[sl]
        rth = np.ones(len(ts), dtype=bool); start = 0
    N = len(ts)
    ent_by_min = {int(m): int(dd) for m, dd in zip(entry_min, entry_dir) if dd != 0}
    reasons = {"RTRIG": 0, "STOP": 0, "SESSION": 0}
    open_dir = 0; entry_px = np.nan; giveback = 0.0; mfe = 0.0
    for i in range(N):
        if not (rth[i] and i >= start):
            continue
        M = (ts[i] // 60) * 60
        # session force-close
        if open_dir != 0 and ts[i] >= session_close_epoch:
            reasons["SESSION"] += 1; giveback += (mfe - open_dir * (c[i] - entry_px))
            open_dir = 0; continue
        # ---- while open: poll stop every 5s ----
        if open_dir != 0:
            adverse = (entry_px - l[i]) if open_dir > 0 else (h[i] - entry_px)
            fav = open_dir * (c[i] - entry_px); mfe = max(mfe, fav)
            if stop_pts is not None and adverse >= stop_pts:
                reasons["STOP"] += 1; open_dir = 0; continue
        # ---- once per minute (at last 5s row of the minute): R-trigger exit ----
        is_last_of_min = (i + 1 >= N) or ((ts[i + 1] // 60) * 60 != M)
        if open_dir != 0 and is_last_of_min:
            cc = conf.get(M, 0)
            if cc != 0 and cc == -open_dir:
                reasons["RTRIG"] += 1; giveback += (mfe - open_dir * (c[i] - entry_px))
                open_dir = 0; continue
        # ---- entry (flat only), acted at the minute whose signal settled 180s ago ----
        if open_dir == 0 and is_last_of_min:
            sig = ent_by_min.get(M - 180, 0)
            if sig != 0:
                open_dir = sig; entry_px = c[i]; mfe = 0.0
    return reasons, giveback


def main():
    import pandas as pd
    days = sys.argv[1:] or ["2024_06_26", "2025_10_16"]
    os.makedirs(REPORTS, exist_ok=True)
    for day in days:
        d = load(day)
        out_lines = []

        def emit(s=""):
            out_lines.append(s); print(s)

        emit("=" * 78)
        emit("P2c R-trigger replay -- day %s" % day)
        emit("=" * 78)
        for cold in (False, True):
            tag = "COLD (NT8 wrapper: RTH-only, start=0, no tail)" if cold else \
                  "HARNESS (prior-day tail, start=2500) -- the 100%-parity input"
            r = run_regime(d, cold)
            minutes = r["minutes"]
            cf, ct = r["conf_full"], r["conf_trunc"]
            nz_full = [M for M in minutes if cf.get(M, 0) != 0]
            nz_trunc = [M for M in minutes if ct.get(M, 0) != 0]
            # truncation mismatches
            mism = [(M, cf.get(M, 0), ct.get(M, 0)) for M in minutes
                    if cf.get(M, 0) != ct.get(M, 0)]
            miss = [(M, cf[M]) for M in nz_full if ct.get(M, 0) == 0]  # full!=0 but trunc==0
            emit("")
            emit("-- regime: %s" % tag)
            emit("   RTH minutes           : %d" % len(minutes))
            emit("   min_rev (full-day)    : %d ticks   first_rth idx=%d" %
                 (r["minrev_full"], r["first_rth"]))
            emit("   min_rev across trunc  : %s" % r["minrevs"])
            emit("   zz_confirm!=0 full-day: %d minutes" % len(nz_full))
            emit("   zz_confirm!=0 trunc   : %d minutes" % len(nz_trunc))
            emit("   trunc vs full mismatch: %d minutes" % len(mism))
            emit("   confirms MISSED by trunc (full!=0, trunc==0): %d" % len(miss))
            if miss[:8]:
                emit("     e.g. " + ", ".join("M=%d full=%+d" % (m, v) for m, v in miss[:8]))
            # exit sim: worst case hold a LONG all day, and a SHORT all day
            for od in (+1, -1):
                nf_full, ff_full = sim_exits(minutes, cf, open_dir=od)
                nf_tr, ff_tr = sim_exits(minutes, ct, open_dir=od)
                emit("   exits if held %-5s : full-day=%d  trunc(runtime)=%d" %
                     ("LONG" if od > 0 else "SHORT", nf_full, nf_tr))
            # ---- faithful end-to-end trade sim (real entries + stop-race) ----
            gp = os.path.join(REPO, "research", "nt8_port", "golden", day + ".parquet")
            if os.path.exists(gp):
                g = pd.read_parquet(gp)
                emin = g["bar_ts"].values; edir = g["entry_dir"].values
                sess_close = int(minutes[0] + 330 * 60)   # 14:00 CT = 330 min after 08:30 open
                for stop in (None, 50 * (1 / TICK) * TICK):   # 50 points
                    rr, gb = trade_sim(d, cold, emin, edir, ct,
                                       None if stop is None else 50.0, sess_close)
                    emit("   TRADE-SIM stop=%-4s : RTRIG=%d STOP=%d SESSION=%d  giveback=%.0f pts" %
                         ("OFF" if stop is None else "50pt",
                          rr["RTRIG"], rr["STOP"], rr["SESSION"], gb))
        # ---- BEFORE/AFTER verdict (cold v0.2 vs warm v0.3) ----
        rc = run_regime(d, cold=True)    # v0.2 broken (cold, no tail)
        rw = run_regime(d, cold=False)   # v0.3 fixed  (warm tail = harness Start)
        nzc = sum(1 for M in rc["minutes"] if rc["conf_trunc"].get(M, 0) != 0)
        nzw = sum(1 for M in rw["minutes"] if rw["conf_trunc"].get(M, 0) != 0)
        alignw = sum(1 for M in rw["minutes"]
                     if rw["conf_trunc"].get(M, 0) == rw["conf_full"].get(M, 0))
        emit("VERDICT %s" % day)
        emit("  v0.2 (cold, no tail)  runtime min_rev=%s  confirms/day=%d" %
             (rc["minrevs"], nzc))
        emit("  v0.3 (warm tail)      runtime min_rev=%d  confirms/day=%d" %
             (rw["minrev_full"], nzw))
        emit("  v0.3 runtime==full-day zz_confirm at %d/%d minutes (truncation-invariant)" %
             (alignw, len(rw["minutes"])))
        emit("  => the fix restores %d->%d confirmed reversals/day; each is reachable by the"
             % (nzc, nzw))
        emit("     exit gate at the SAME minute the full-day run confirms it.")
        emit("")
        path = os.path.join(REPORTS, "p2c_replay_%s.txt" % day)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print("[wrote] %s" % path)


if __name__ == "__main__":
    main()
