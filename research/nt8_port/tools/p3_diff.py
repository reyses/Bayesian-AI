"""
P3 FIRST-PASS DIFF: python wrapper-policy simulation over the window goldens vs the
NT8 v0.2-RC backtest trade export.

Simulates the EnsembleRunner v0.2 wrapper policy on the golden per-1m-bar decisions:
  * one position at a time
  * entry = first `entry==1` bar while flat, ACTED at signal_minute + 180s settle
    (p2b_v02_parity.md sec.5); entry fill = open of the action bar
  * ride-only, NO R-trigger exit (fired 0/44 in v0.2 -> disabled here)
  * 50-pt catastrophic stop, polled on 1m bar high/low
  * session flatten at the last golden RTH bar of the day (harness RTH ends 15:15 CT)

Timezone: NT8 export times are US/Pacific; CT = PT + 2h (empirically pinned by exact
entry/exit price matches -- see report). All comparisons are in CT epoch seconds.

Compares the simulated trade list to the NT8 export (window days with data only) by:
entry minute (+/-2 min), direction, entry price (+/-2 pts). Reports match rate,
harness-entries-not-taken, NT8-entries-unexplained, R-trigger would-be exits.

Usage: python3.11 research/nt8_port/tools/p3_diff.py
"""
import os
import csv
import glob
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))
GOLD = os.path.join(PROJ, 'golden_backtest')
NT8_1M = os.path.join(ROOT, 'DATA', 'ATLAS_NT8', '1m')
TRADES = os.path.join(PROJ, 'reports', 'backtest_v02_trades_2026-06-22_07-17.csv')
OUTREP = os.path.join(PROJ, 'reports', 'p3_first_diff.md')

CT = pytz.timezone('America/Chicago')
PT_TO_CT_HOURS = 2                 # NT8 display = US/Pacific; CT = PT + 2h (empirical)
SETTLE_S = 180                     # 180s consensus settle before an entry acts
CAT_STOP_PTS = 50.0
TICK = 0.25
MATCH_MIN = 2                      # entry-minute tolerance (minutes)
MATCH_PTS = 2.0                    # entry-price tolerance (points)


def load_bars(day):
    df = pd.read_parquet(os.path.join(NT8_1M, f'{day}.parquet'))
    df = df.sort_values('timestamp').reset_index(drop=True)
    # 1m ts are minute-CLOSE (+59). key by minute-OPEN to match golden bar_ts.
    df['min_open'] = ((df['timestamp'] - 59) // 60) * 60
    return df.set_index('min_open')


def bar_px(bars, min_open):
    """Return (open,high,low,close) for the 1m bar at minute-open, or None."""
    if min_open in bars.index:
        r = bars.loc[min_open]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        return float(r['open']), float(r['high']), float(r['low']), float(r['close'])
    return None


def _session_open_ct(day, hour, minute):
    dt = CT.localize(datetime.strptime(day, '%Y_%m_%d') + timedelta(hours=hour, minutes=minute))
    return int(dt.timestamp())


def simulate_day(day, min_entry_ct=None):
    """Run the wrapper policy over one day's golden. Returns (sim_trades, rtrig_events).

    min_entry_ct: if set, entries before this CT epoch are ignored (session-open align)."""
    g = pd.read_parquet(os.path.join(GOLD, f'{day}.parquet')).sort_values('bar_ts').reset_index(drop=True)
    bars = load_bars(day)
    bar_ts = g['bar_ts'].values.astype(np.int64)
    entry = g['entry'].values.astype(int)
    entry_dir = g['entry_dir'].values.astype(int)
    zz_confirm = g['zz_confirm'].values.astype(int)
    zz_leg = g['zz_leg'].values.astype(int)
    last_ts = int(bar_ts[-1])

    trades = []
    rtrig = []           # would-be R-trigger exits (zz_confirm opposing the open pos)
    i = 0
    n = len(g)
    while i < n:
        if min_entry_ct is not None and int(bar_ts[i]) < min_entry_ct:
            i += 1
            continue
        if entry[i] == 1 and entry_dir[i] != 0:
            sig_ts = int(bar_ts[i])
            d = int(entry_dir[i])
            act_ts = sig_ts + SETTLE_S
            # find action bar (minute-open == act_ts)
            k = int(np.searchsorted(bar_ts, act_ts))
            if k >= n:
                break                      # settle runs past session end -> no fill
            fill_ts = int(bar_ts[k])
            px = bar_px(bars, fill_ts)
            if px is None:
                i = k + 1
                continue
            entry_px = px[0]               # fill = open of the action bar
            # ---- ride until stop or session end ----
            exit_ts = last_ts
            exit_px = None
            exit_reason = 'SessionEnd(RTH15:15CT)'
            would_rtrig = None
            j = k
            while j < n:
                tpx = bar_px(bars, int(bar_ts[j]))
                if tpx is not None:
                    o, h, l, c = tpx
                    # catastrophic 50pt stop (poll)
                    if d > 0 and l <= entry_px - CAT_STOP_PTS:
                        exit_ts, exit_px, exit_reason = int(bar_ts[j]), entry_px - CAT_STOP_PTS, 'CatStop50'
                        break
                    if d < 0 and h >= entry_px + CAT_STOP_PTS:
                        exit_ts, exit_px, exit_reason = int(bar_ts[j]), entry_px + CAT_STOP_PTS, 'CatStop50'
                        break
                # would-be R-trigger: a zz_confirm opposing the position leg
                if would_rtrig is None and zz_confirm[j] != 0 and zz_confirm[j] == -d and j > k:
                    would_rtrig = int(bar_ts[j])
                j += 1
            if exit_px is None:
                epx = bar_px(bars, exit_ts)
                exit_px = epx[3] if epx else entry_px
            trades.append(dict(day=day, entry_ts=fill_ts, sig_ts=sig_ts, dir=d,
                               entry_px=round(entry_px, 2), exit_ts=exit_ts,
                               exit_px=round(exit_px, 2), exit_reason=exit_reason,
                               would_rtrig_ts=would_rtrig))
            if would_rtrig is not None:
                rtrig.append(would_rtrig)
            # resume scanning AFTER the exit bar (one position at a time)
            i = int(np.searchsorted(bar_ts, exit_ts)) + 1
        else:
            i += 1
    return trades, rtrig


def load_nt8_trades():
    rows = list(csv.DictReader(open(TRADES)))
    out = []
    for r in rows:
        d = datetime.strptime(r['Entry time'], '%m/%d/%Y %I:%M:%S %p')
        xd = datetime.strptime(r['Exit time'], '%m/%d/%Y %I:%M:%S %p')
        day = d.strftime('%Y_%m_%d')
        # PT -> CT epoch
        ct_entry = CT.localize(d + timedelta(hours=PT_TO_CT_HOURS))
        ct_exit = CT.localize(xd + timedelta(hours=PT_TO_CT_HOURS))
        out.append(dict(
            n=int(r['Trade number']), day=day,
            dir=1 if r['Market pos.'].strip().lower() == 'long' else -1,
            entry_px=float(r['Entry price']), exit_px=float(r['Exit price']),
            entry_ct=int(ct_entry.timestamp()), exit_ct=int(ct_exit.timestamp()),
            entry_pt=r['Entry time'], exit_name=r['Exit name'].strip()))
    return out


def ctstr(ep):
    return datetime.fromtimestamp(ep, CT).strftime('%H:%M')


def match_sim_nt8(sim_all, nt8_win, days):
    matched, nt8_unexpl, harness_not_taken = [], [], []
    used_sim = set()
    for t in nt8_win:
        cand = [(i, s) for i, s in enumerate(sim_all)
                if s['day'] == t['day'] and s['dir'] == t['dir'] and i not in used_sim
                and abs(s['entry_ts'] - t['entry_ct']) <= MATCH_MIN * 60]
        cand.sort(key=lambda x: abs(x[1]['entry_ts'] - t['entry_ct']))
        hit = None
        for i, s in cand:
            if abs(s['entry_px'] - t['entry_px']) <= MATCH_PTS:
                hit = (i, s); break
        if hit is None and cand:
            hit = cand[0]
        if hit is not None:
            i, s = hit
            used_sim.add(i)
            matched.append((t, s, abs(s['entry_px'] - t['entry_px'])))
        else:
            nt8_unexpl.append(t)
    for i, s in enumerate(sim_all):
        if i not in used_sim and s['day'] in days:
            harness_not_taken.append(s)
    return matched, nt8_unexpl, harness_not_taken


def main():
    days = sorted(os.path.basename(p)[:10] for p in glob.glob(os.path.join(GOLD, '*.parquet')))
    nt8 = load_nt8_trades()
    nt8_days = sorted(set(t['day'] for t in nt8))
    missing = [d for d in nt8_days if d not in days]
    nt8_win = [t for t in nt8 if t['day'] in days]

    # variant A: native harness RTH (08:30 CT open); variant B: session-aligned (10:30 CT = 08:30 PT)
    sim_all, rtrig_all = [], {}
    simB = []
    for d in days:
        st, rt = simulate_day(d)
        sim_all.extend(st)
        rtrig_all[d] = rt
        stB, _ = simulate_day(d, min_entry_ct=_session_open_ct(d, 10, 30))
        simB.extend(stB)

    matched, nt8_unexpl, harness_not_taken = match_sim_nt8(sim_all, nt8_win, days)
    matchedB, unexplB, notTakenB = match_sim_nt8(simB, nt8_win, days)
    n_dirtime = len(matched)
    n_pxok = sum(1 for _, _, dp in matched if dp <= MATCH_PTS)
    nB_dirtime = len(matchedB)
    nB_pxok = sum(1 for _, _, dp in matchedB if dp <= MATCH_PTS)

    # first-entry-per-day agreement (variant B, aligned)
    def first_by_day(trades):
        fb = {}
        for s in sorted(trades, key=lambda x: x['entry_ts']):
            fb.setdefault(s['day'], s)
        return fb
    nt8_first = {}
    for t in sorted(nt8_win, key=lambda x: x['entry_ct']):
        nt8_first.setdefault(t['day'], t)
    simB_first = first_by_day(simB)
    fe_rows, fe_dir_ok, fe_px_ok = [], 0, 0
    for d in days:
        nt = nt8_first.get(d); sb = simB_first.get(d)
        if nt and sb:
            dok = nt['dir'] == sb['dir']
            pok = abs(nt['entry_px'] - sb['entry_px']) <= MATCH_PTS
            fe_dir_ok += int(dok); fe_px_ok += int(pok)
            fe_rows.append((d, nt, sb, dok, pok))
        else:
            fe_rows.append((d, nt, sb, False, False))

    # ---- write report ----
    L = []
    L.append('# P3 first-pass diff — python wrapper sim vs NT8 v0.2-RC backtest')
    L.append('')
    L.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} · executor: Opus data/diff drone · commits: none')
    L.append('')
    L.append('## Step 1 — ATLAS_NT8 conversion (data pipeline)')
    L.append('- Raw source: `D:/Bayesian-AI-data/DATA/RAW_NT8/{MNQ_06-26,MNQ_09-26}/{1s,1m}/*.csv` '
             '(BayesianHistoryDumper per-TF CSV). Converter: `tools/sourcing/convert_nt8_csv_to_parquet.py` '
             '(the importer matching THIS raw layout; the DATA/pipeline/README nt8_* tools are for .txt/tick formats).')
    L.append('- **Before**: ATLAS_NT8 all TFs maxed at **2026-06-12**. **After**: 1s/1m extended to **2026-07-08** '
             '(147 day-files; 6.17M 1s bars). Derived TFs (5s/15s/5m/15m/1h/4h/1D) built by '
             '`DATA/pipeline/build_timeframes.py` (incremental) — its OHLC-vs-control validation PASSED '
             '(0 mismatches: 1s-vs-1m, 5s-vs-1m, 15s-vs-1m, 1m-vs-1h, 5m-vs-1h, 15m-vs-1h).')
    L.append(f'- **Raw stops at 2026-07-08** — the window\'s last 7 trading days (07-09..07-17) have NO raw NT8 '
             'data and cannot be built. 07-08 itself is truncated (dump cut mid-session).')
    L.append('- SFE features: `python core_v2/build_dataset.py --atlas DATA/ATLAS_NT8 --start 2026-06-13 --end 2026-07-08` '
             '(authorized standard build, incremental). Completed on the RTX 3060, no OOM, 18 day-files written. '
             '**It wrote `L3_1m_z_se_30`** (see z_se caveat).')
    L.append('')
    L.append('## Step 2 — goldens window summary')
    try:
        ws = pd.read_csv(os.path.join(GOLD, '_window_summary.csv'))
        L.append(f'- {len(ws)} golden days -> `research/nt8_port/golden_backtest/` (frozen `golden/` untouched). '
                 f'entry-eligible 1m bars/day: min {int(ws.entries.min())}, median {int(ws.entries.median())}, '
                 f'max {int(ws.entries.max())} (these are P>=thr bars; the wrapper collapses them to ~1-6 trades/day). '
                 f'zz_confirms/day median {int(ws.zz_confirms.median())}.')
    except Exception:
        pass
    L.append('')
    L.append('## Timezone finding')
    L.append(f'- NT8 export display = **US/Pacific**; **CT = PT + {PT_TO_CT_HOURS}h**. Pinned empirically by '
             'exact price matches, not assumed: e.g. 7/1 short 30304.00 @ "8:40 AM" fills at 10:38–10:40 CT '
             '(bar 30304.25); session-close exits @ "2:00 PM" land at 16:00 CT (the CME session close), exact.')
    L.append('- Consequence: NT8 first entries cluster ~10:40 CT — **~2h into the harness RTH (08:30 CT)**. '
             'The NT8 data-series session is shifted +2h vs the python RTH window; this is the dominant structural gap (below).')
    L.append('')
    L.append('## Coverage')
    L.append(f'- Golden window days generated (data available): **{len(days)}** ({days[0]}..{days[-1]}).')
    L.append(f'- NT8 export days: {len(nt8_days)} ({nt8_days[0]}..{nt8_days[-1]}).')
    L.append(f'- **Missing (no raw NT8 data past 2026-07-08): {missing}** — trades on these days cannot be diffed.')
    L.append(f'- NT8 trades in comparable window: **{len(nt8_win)}** (of {len(nt8)} total).')
    # partial-day flags from golden bar counts (full RTH ~406 1m bars)
    partial = []
    for d in days:
        nb = len(pd.read_parquet(os.path.join(GOLD, f'{d}.parquet')))
        if nb < 380:
            last = ctstr(int(pd.read_parquet(os.path.join(GOLD, f'{d}.parquet'))['bar_ts'].max()))
            partial.append(f'{d} ({nb} RTH bars, ends ~{last} CT)')
    L.append(f'- **Partial/truncated days (raw dump cut mid-session): {partial}** — NT8 entries after the last '
             'golden bar on these days fall outside available data (e.g. 07-08 NT8 traded 10:40+ CT but data ends ~09:28 CT).')
    L.append('')
    L.append('## Match result (dir + entry-minute ±2 + entry-price ±2pt)')
    L.append(f'- NT8 window trades: {len(nt8_win)}; sim trades (variant A / B): {len(sim_all)} / {len(simB)}')
    L.append('')
    L.append('| variant | sim session open | matched dir+minute | entry-px ±2pt | NT8 unexplained | harness-not-taken |')
    L.append('|---|---|---|---|---|---|')
    L.append(f'| A native | 08:30 CT (harness RTH) | {n_dirtime}/{len(nt8_win)} ({100*n_dirtime/max(len(nt8_win),1):.0f}%) '
             f'| {n_pxok}/{len(nt8_win)} | {len(nt8_unexpl)} | {len(harness_not_taken)} |')
    L.append(f'| B aligned | 10:30 CT (= 08:30 PT) | {nB_dirtime}/{len(nt8_win)} ({100*nB_dirtime/max(len(nt8_win),1):.0f}%) '
             f'| {nB_pxok}/{len(nt8_win)} | {len(unexplB)} | {len(notTakenB)} |')
    L.append('')
    L.append('**Reading**: variant A (harness RTH 08:30 CT) enters ~2h before NT8 every day -> near-zero match; '
             'the gap is a session-window shift, not a decision-logic error. Variant B (sim session opened at '
             '10:30 CT to mirror NT8) tests whether the decision core agrees once the window is aligned.')
    L.append('')
    L.append('*Note*: the aggregate match is low mostly because after the first entry the wrapper trade '
             '*sequence* (immediate re-entry after each cat-stop) diverges from NT8, and NT8 has 1–6 trades/day. '
             'The cleanest decision-core comparison is the **first entry per day** below.')
    L.append('')
    L.append('## First entry per day — NT8 vs session-aligned sim (variant B)')
    L.append(f'- direction agrees: **{fe_dir_ok}/{len(days)}**;  entry-price ±2pt: **{fe_px_ok}/{len(days)}**')
    L.append('| day | NT8 first (CT dir px) | simB first (CT dir px) | dir? | px? |')
    L.append('|---|---|---|---|---|')
    for d, nt, sb, dok, pok in fe_rows:
        ns = f"{ctstr(nt['entry_ct'])} {'L' if nt['dir']>0 else 'S'} {nt['entry_px']:.2f}" if nt else '—'
        ss = f"{ctstr(sb['entry_ts'])} {'L' if sb['dir']>0 else 'S'} {sb['entry_px']:.2f}" if sb else '—'
        L.append(f"| {d} | {ns} | {ss} | {'Y' if dok else 'n'} | {'Y' if pok else 'n'} |")
    L.append('')
    L.append('## Per-day: NT8 entries (CT) vs sim entries (CT)')
    L.append('| day | NT8 entries (CT, dir, px) | sim entries (CT, dir, px, exit) |')
    L.append('|---|---|---|')
    for d in days:
        nn = [f"{ctstr(t['entry_ct'])} {'L' if t['dir']>0 else 'S'} {t['entry_px']:.0f}" for t in nt8_win if t['day'] == d]
        ss = [f"{ctstr(s['entry_ts'])} {'L' if s['dir']>0 else 'S'} {s['entry_px']:.0f}/{s['exit_reason'][:6]}" for s in sim_all if s['day'] == d]
        L.append(f"| {d} | {'; '.join(nn) if nn else '—'} | {'; '.join(ss) if ss else '—'} |")
    L.append('')
    L.append('## Matched pairs (NT8 ↔ sim)')
    L.append('| NT8# | day | dir | NT8 CT | sim CT | NT8 px | sim px | dpx |')
    L.append('|---|---|---|---|---|---|---|---|')
    for t, s, dp in matched:
        L.append(f"| {t['n']} | {t['day']} | {'L' if t['dir']>0 else 'S'} | {ctstr(t['entry_ct'])} | "
                 f"{ctstr(s['entry_ts'])} | {t['entry_px']:.2f} | {s['entry_px']:.2f} | {dp:.2f} |")
    L.append('')
    L.append('## NT8 entries unexplained by the sim')
    L.append('| NT8# | day | dir | NT8 CT | NT8 PT | px | exit |')
    L.append('|---|---|---|---|---|---|---|')
    for t in nt8_unexpl:
        past = ' (post-RTH 15:15CT)' if int(datetime.fromtimestamp(t['entry_ct'], CT).strftime('%H%M')) > 1515 else ''
        L.append(f"| {t['n']} | {t['day']} | {'L' if t['dir']>0 else 'S'} | {ctstr(t['entry_ct'])}{past} | {t['entry_pt']} | {t['entry_px']:.2f} | {t['exit_name']} |")
    L.append('')
    L.append(f'## Harness sim entries NOT taken by NT8 (first {min(40,len(harness_not_taken))} of {len(harness_not_taken)})')
    L.append('| day | dir | sim CT | px | exit |')
    L.append('|---|---|---|---|---|')
    for s in harness_not_taken[:40]:
        L.append(f"| {s['day']} | {'L' if s['dir']>0 else 'S'} | {ctstr(s['entry_ts'])} | {s['entry_px']:.2f} | {s['exit_reason']} |")
    L.append('')
    L.append('## R-trigger would-be exits (evidence for the v0.3 fix)')
    L.append('- v0.2 fired the R-trigger 0/44. Below: sim positions where a `zz_confirm` OPPOSING the '
             'open leg occurred during the trade (i.e. the R-trigger reversal that *should* have exited).')
    n_with = sum(1 for s in sim_all if s['would_rtrig_ts'] is not None)
    L.append(f'- sim trades total: **{len(sim_all)}**; with an opposing zz_confirm during the ride: **{n_with}** '
             f'({100*n_with/max(len(sim_all),1):.0f}%).')
    L.append('| day | dir | entry CT | would-be R-trig CT | actual sim exit |')
    L.append('|---|---|---|---|---|')
    for s in sim_all:
        if s['would_rtrig_ts'] is not None:
            L.append(f"| {s['day']} | {'L' if s['dir']>0 else 'S'} | {ctstr(s['entry_ts'])} | "
                     f"{ctstr(s['would_rtrig_ts'])} | {s['exit_reason']} |")
    L.append('')
    L.append('## Caveats')
    L.append('- **z_se N-skew**: the standard SFE build wrote `L3_1m_z_se_30` (code `N_BASE[1m]=30`), but the '
             'frozen golden reference and the C# port consume `z_se_15`. The 6 NMP/NMP9 top-K streams here fire '
             'off N=30 state — NMP-governed entries are not bit-faithful to what NT8 ran. Non-NMP entries are unaffected.')
    L.append('- **Session window**: golden decides on RTH 08:30–15:15 CT; NT8 trades a +2h-shifted session to '
             '16:00 CT. Sim session-end exits at 15:15 CT vs NT8 16:00 CT; NT8 entries after 15:15 CT are outside '
             'the harness window by construction.')
    L.append('- Entry fill = open of the +180s action bar; NT8 market fill may differ by a bar (≈ the residual dpx).')

    with open(OUTREP, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))

    # machine-readable sidecar
    with open(os.path.join(PROJ, 'reports', 'p3_sim_trades.json'), 'w') as f:
        json.dump(dict(sim_trades=sim_all, nt8_window=len(nt8_win),
                       matched=len(matched), px_ok=n_pxok,
                       unexplained=[t['n'] for t in nt8_unexpl],
                       missing_days=missing), f, indent=1, default=str)

    print(f"days={len(days)} sim_trades={len(sim_all)} nt8_win={len(nt8_win)} "
          f"matched_dirtime={n_dirtime} px_ok={n_pxok} unexpl={len(nt8_unexpl)} "
          f"harness_not_taken={len(harness_not_taken)}")
    print(f"wrote {OUTREP}")


if __name__ == '__main__':
    main()
