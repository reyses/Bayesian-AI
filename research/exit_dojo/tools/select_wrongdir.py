"""
Wrong-Direction Dojo -- 50/50 population selector, ECONOMIC-TRUTH cut
(research/exit_dojo/tools/select_wrongdir.py)

Task 099 (reviewer re-cut 2026-07-17). Moises' purpose: "identify conditions to spot BAD
TRADES and cut in time BEFORE it makes damage." So truth is ECONOMIC, not the label
taxonomy. Each engagement is judged by its TERMINAL drift (favorable-signed points from
entry at the end of its window -- entry=0.00, positive=win, negative=loss):

  WRONG = terminal <= -BAND   (a real loss / "damage")
  GOOD  = terminal >= +BAND   (a real win)
  DROP  = |terminal| < BAND    (ambiguous near-zero scratch -- dead-band, excluded)

BAND starts at 4 pts (16 ticks) and widens along BAND_LADDER only if 4 pts cannot fill a
clean 100 WRONG / 100 GOOD on DISTINCT 2025-26 days. The terminal-drift histogram + the
settled BAND are written into selection_table.md so the split is transparent.

good_kind (the hard case): for GOOD trades, `dipped` = went <= -DIP_PTS (4 pts) adverse at
ANY point before recovering (the "don't cut a temporary dip" trap) vs `clean` (never dipped
that far). The scorer reports false-bail split by dipped/clean.

POPULATION: replica of phold_exit_model.engagements() -- econ_drift_rows.parquet, split
=='test', P >= p90(train P) frozen on train, 60s/day/dir de-dup, 2025-26. Pilot / exit-run
days NOT excluded (overlap allowed -- different question, doc 099). One DISTINCT day per ep.

Writes (under research/exit_dojo/reports/wrongdir/):
  selection.json        manifest the telescope_packet_builder --selection consumes
                        (each episode carries the HIDDEN truth: truth_label + good_kind;
                         meta carries the settled BAND so the scorer classifies identically)
  selection_table.md    terminal histogram + BAND + 50/50 balance + dipped/clean split

Run:
  python3.11 research/exit_dojo/tools/select_wrongdir.py [--seed N] [--n-per-class 100]
      [--band 4]
Then build packets with the EXISTING builder:
  python3.11 research/exit_dojo/builders/telescope_packet_builder.py \
      --selection research/exit_dojo/reports/wrongdir/selection.json \
      --outdir research/exit_dojo/reports/wrongdir
"""
import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDERS = os.path.abspath(os.path.join(HERE, '..', 'builders'))
sys.path.insert(0, BUILDERS)
import episode_builder as eb                 # verified drift + window helpers
import telescope_packet_builder as tb        # verified window + eid helpers

DOJO_ROOT = os.path.abspath(os.path.join(HERE, '..'))
WRONGDIR_DIR = os.path.join(DOJO_ROOT, 'reports', 'wrongdir')
SELECTION_JSON = os.path.join(WRONGDIR_DIR, 'selection.json')
SELECTION_TABLE = os.path.join(WRONGDIR_DIR, 'selection_table.md')

# ---- constants (house rule: no bare magic numbers) -----------------------------------
P_PCTL = tb.P_PCTL                # 90: entry-P percentile defining an engagement
DEDUP_S = tb.DEDUP_S              # 60: co-fires within this / day / dir = one engagement
MIN_WINDOW_MIN = tb.MIN_WINDOW_MIN
SELECTION_SEED = 20260717
N_PER_CLASS = 100                 # target N_wrong = N_good = 100 (200 total)
BAND_LADDER = [4.0, 5.0, 6.0, 8.0, 10.0]   # pts; smallest that fills 100/100 wins
DIP_PTS = 4.0                     # a GOOD trade "dipped" if it went <= -DIP_PTS before winning


def engagements() -> pd.DataFrame:
    """phold_exit_model.engagements() replica: test split, P>=p90(train), 60s/day/dir
    de-dup, 2025-26. Pilot / exit-run days NOT excluded (overlap allowed, doc 099)."""
    econ = pd.read_parquet(eb.ECON_DRIFT_PATH,
                           columns=['ts', 'day', 'det', 'is_long', 'P', 'split'])
    thr = float(np.percentile(econ.loc[econ.split == 'train', 'P'].values, P_PCTL))
    sub = econ[(econ.split == 'test') & (econ.P >= thr) &
               (econ.day.str[:4].isin(['2025', '2026']))].copy()
    sub = sub.sort_values(['day', 'is_long', 'ts', 'det']).reset_index(drop=True)
    last: Dict[Tuple[str, bool], int] = {}
    keep = []
    for r in sub.itertuples():
        k = (r.day, bool(r.is_long))
        if k in last and r.ts - last[k] <= DEDUP_S:
            continue
        last[k] = r.ts
        keep.append(r.Index)
    dd = sub.loc[keep].reset_index(drop=True)
    dd.attrs['p90_thr'] = thr
    return dd


def scan(eng: pd.DataFrame):
    """Per day: every engagement's terminal drift + worst adverse dip + params. Returns
    (day_engs: day -> [eng dict], all_terminals: np.ndarray for the histogram)."""
    days = sorted(eng['day'].unique())
    groups = {d: g for d, g in eng.groupby('day', sort=False)}
    day_engs: Dict[str, List[dict]] = defaultdict(list)
    all_terminals: List[float] = []
    for day in tqdm(days, desc='scan days'):
        dd = eb.load_day_data(day)
        if dd is None:
            continue
        for r in groups[day].itertuples(index=False):
            ets, isl = int(r.ts), bool(r.is_long)
            lem = eb.label_flip_minute(
                dd.oracle_ivals, ets, isl,
                int(min(tb.WINDOW_CAP, (dd.session_end - ets) // 60)))
            wm = tb._window_minutes(dd.session_end, ets, lem)
            if wm < MIN_WINDOW_MIN:
                continue
            dp, entry_price = eb.signed_drift_path(dd.ts5, dd.c5, ets, isl, wm)
            terminal = float(dp[wm])
            mindrift = float(min(dp))
            all_terminals.append(terminal)
            om = lem if lem is not None else wm
            day_engs[day].append(dict(
                day=day, ts=ets, is_long=isl, P=float(r.P), det=r.det,
                window_minutes=wm, label_end_minute=lem, oracle_minute=om,
                oracle_capture=float(dp[om]), per_minute_forward_drift=dp,
                entry_price=entry_price, chop_tol=None,
                terminal=terminal, mindrift=mindrift))
    return dict(day_engs), np.array(all_terminals, float)


def _tag(e: dict, truth_label: str, good_kind: Optional[str]) -> dict:
    r = dict(e)
    r.pop('terminal', None)
    r.pop('mindrift', None)
    r['type'] = truth_label            # builder writes truth['type'] = this (wrong/good)
    r['truth_label'] = truth_label
    r['good_kind'] = good_kind
    r['terminal_drift'] = e['terminal']
    r['min_drift'] = e['mindrift']
    return r


def per_day_candidates(day_engs, band, dip):
    """day -> dict(wrong, good_dipped, good_clean) first-qualifying engagement (by ts)."""
    out = {}
    for day, engs in day_engs.items():
        es = sorted(engs, key=lambda e: e['ts'])
        w = gd = gc = None
        for e in es:
            if w is None and e['terminal'] <= -band:
                w = e
            if e['terminal'] >= band:
                if e['mindrift'] <= -dip and gd is None:
                    gd = e
                elif e['mindrift'] > -dip and gc is None:
                    gc = e
        out[day] = dict(wrong=w, good_dipped=gd, good_clean=gc)
    return out


def allocate(day_engs, days, band, dip, seed, n):
    """One DISTINCT day per episode; WRONG first, then GOOD (dipped then clean, balanced,
    with cross top-up). Enforce N_wrong == N_good == min(achieved)."""
    cand = per_day_candidates(day_engs, band, dip)
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(days))
    used: set = set()

    def grab(want, key):
        got = []
        for day in perm:
            if len(got) >= want:
                break
            if day in used:
                continue
            c = cand[day][key]
            if c is not None:
                got.append(c)
                used.add(day)
        return got

    wrong = [_tag(e, 'wrong', None) for e in grab(n, 'wrong')]
    half = n // 2
    gd = [_tag(e, 'good', 'dipped') for e in grab(half, 'good_dipped')]
    gc = [_tag(e, 'good', 'clean') for e in grab(n - len(gd), 'good_clean')]
    if len(gd) + len(gc) < n:                       # clean short -> top up with dipped
        gd += [_tag(e, 'good', 'dipped') for e in grab(n - len(gd) - len(gc), 'good_dipped')]
    good = gd + gc

    m = min(len(wrong), len(good))
    wrong = wrong[:m]
    if len(good) > m:                               # trim over-represented kind first
        dips = [g for g in good if g['good_kind'] == 'dipped']
        cls = [g for g in good if g['good_kind'] == 'clean']
        t_d = min(len(dips), (m + 1) // 2)
        t_c = m - t_d
        if t_c > len(cls):
            t_c = len(cls); t_d = m - t_c
        good = dips[:t_d] + cls[:t_c]

    selected = wrong + good
    selected.sort(key=lambda s: (0 if s['truth_label'] == 'wrong' else 1, s['day']))
    return selected, wrong, good


def _manifest_row(s: dict) -> dict:
    r = dict(s)
    r['eid'] = tb._eid(s)
    return r


def _ascii_hist(terminals, band):
    """Text histogram of terminal drift (5-pt bins), clipped to +-100, mode-first note."""
    bw = 5.0
    clip = 100.0
    x = np.clip(terminals, -clip, clip)
    edges = np.arange(-clip, clip + bw, bw)
    h, e = np.histogram(x, bins=edges)
    kmode = int(np.argmax(h))
    mode_ctr = (e[kmode] + e[kmode + 1]) / 2
    peak = max(h.max(), 1)
    lines = []
    for i in range(len(h)):
        if h[i] == 0:
            continue
        bar = '#' * max(1, int(round(40 * h[i] / peak)))
        lines.append(f"  [{e[i]:+6.0f},{e[i+1]:+6.0f}) {h[i]:5d} {bar}")
    n = len(terminals)
    n_wrong = int((terminals <= -band).sum())
    n_good = int((terminals >= band).sum())
    n_dead = n - n_wrong - n_good
    return lines, mode_ctr, n, n_wrong, n_good, n_dead


def write_table(selected, wrong, good, meta, terminals):
    n_days = len({s['day'] for s in selected})
    dips = sum(1 for g in good if g['good_kind'] == 'dipped')
    cln = sum(1 for g in good if g['good_kind'] == 'clean')
    band = meta['band']
    hist, mode_ctr, n, nw, ng, nd = _ascii_hist(terminals, band)
    L = []
    A = L.append
    A('# Wrong-Direction Dojo -- ECONOMIC-TRUTH selection (cut BAD trades before damage)')
    A('')
    A(f"Seed={meta['seed']}; phold engagement population (entry-P p{P_PCTL} frozen on train "
      f"= {meta['p90_thr']:.5f}, 60s/day/dir de-dup, test split, 2025-26). Pilot / exit-run "
      f"days NOT excluded. Truth is ECONOMIC (terminal drift), not the label taxonomy.")
    A('')
    A('## Truth definition')
    A(f"- **WRONG** = terminal drift <= -{band:.0f} pts (a real loss / damage)")
    A(f"- **GOOD**  = terminal drift >= +{band:.0f} pts (a real win)")
    A(f"- DROP (dead-band) = |terminal| < {band:.0f} pts (ambiguous scratch, excluded)")
    A(f"- good_kind: **dipped** = went <= -{DIP_PTS:.0f} pts adverse before winning "
      f"(the hard 'temporary dip' case); **clean** = never dipped that far")
    A(f"- **BAND settled = {band:.0f} pts** (started at {BAND_LADDER[0]:.0f}; "
      f"{'held' if band == BAND_LADDER[0] else 'widened to fill 100/100 on distinct days'})")
    A('')
    A('## Terminal-drift histogram (all engagements, 5-pt bins, clipped +-100)')
    A(f"N engagements = {n}; mode bin center {mode_ctr:+.1f} pts. At BAND={band:.0f}: "
      f"WRONG(<=-{band:.0f})={nw}, DROP(|.|<{band:.0f})={nd}, GOOD(>=+{band:.0f})={ng}.")
    A('```')
    L.extend(hist)
    A('```')
    A('')
    A('## 50/50 balance')
    A(f"- **WRONG**: {len(wrong)}  | **GOOD**: {len(good)}  (dipped {dips}, clean {cln})")
    A(f"- Total: {len(selected)}  |  N_wrong==N_good: "
      f"{'OK' if len(wrong) == len(good) else 'BROKEN'}")
    A(f"- Distinct real days: {n_days} of {len(selected)} "
      f"({'all distinct' if n_days == len(selected) else 'some reuse'})")
    A(f"- Pool capacity at BAND={band:.0f}: WRONG-days={meta['cap_wrong']}, "
      f"GOOD-days={meta['cap_good']} (dipped-days={meta['cap_dipped']}, "
      f"clean-days={meta['cap_clean']})")
    if len(wrong) < N_PER_CLASS or len(good) < N_PER_CLASS:
        A('')
        A('## Declared shortfall')
        A(f"target {N_PER_CLASS}/{N_PER_CLASS} not reached even at BAND={band:.0f}; "
          f"50/50 invariant held by capping at the scarcer class.")
    A('')
    A('| ep | eid | truth | good_kind | real day | det | entry ts | dir | P | '
      'window(min) | terminal(pts) | min drift(pts) |')
    A('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for i, s in enumerate(selected, 1):
        A(f"| {i:03d} | {tb._eid(s)} | {s['truth_label']} | {s['good_kind'] or '-'} | "
          f"{s['day']} | {s['det']} | {s['ts']} | {'LONG' if s['is_long'] else 'SHORT'} | "
          f"{s['P']:.3f} | {s['window_minutes']} | {s['terminal_drift']:+.1f} | "
          f"{s['min_drift']:+.1f} |")
    A('')
    A('_HIDDEN TRUTH -- never served to agents. selection.json meta carries BAND so the '
      'scorer classifies identically._')
    os.makedirs(WRONGDIR_DIR, exist_ok=True)
    with open(SELECTION_TABLE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=SELECTION_SEED)
    ap.add_argument('--n-per-class', type=int, default=N_PER_CLASS)
    ap.add_argument('--band', type=float, default=None,
                    help='force a specific BAND (pts); default = smallest ladder rung that fills')
    args = ap.parse_args()

    os.makedirs(WRONGDIR_DIR, exist_ok=True)
    eng = engagements()
    thr = eng.attrs['p90_thr']
    print(f'[select] engagements: {len(eng)} fires over {eng["day"].nunique()} test 2025-26 days; '
          f'p{P_PCTL} thr={thr:.5f}')
    day_engs, terminals = scan(eng)
    days = sorted(day_engs.keys())
    n = args.n_per_class

    ladder = [args.band] if args.band else BAND_LADDER
    chosen = None
    for band in ladder:
        cand = per_day_candidates(day_engs, band, DIP_PTS)
        cap_wrong = sum(1 for d in days if cand[d]['wrong'] is not None)
        cap_dipped = sum(1 for d in days if cand[d]['good_dipped'] is not None)
        cap_clean = sum(1 for d in days if cand[d]['good_clean'] is not None)
        cap_good = sum(1 for d in days if cand[d]['good_dipped'] is not None
                       or cand[d]['good_clean'] is not None)
        selected, wrong, good = allocate(day_engs, days, band, DIP_PTS, args.seed, n)
        print(f'[band {band:.0f}] cap wrong-days={cap_wrong} good-days={cap_good} '
              f'(dip {cap_dipped}/clean {cap_clean}) -> chose WRONG={len(wrong)} GOOD={len(good)}')
        chosen = dict(band=band, selected=selected, wrong=wrong, good=good,
                      cap_wrong=cap_wrong, cap_good=cap_good,
                      cap_dipped=cap_dipped, cap_clean=cap_clean)
        if len(wrong) >= n and len(good) >= n:
            break

    band = chosen['band']
    selected, wrong, good = chosen['selected'], chosen['wrong'], chosen['good']
    print(f'[select] BAND={band:.0f}; WRONG={len(wrong)} GOOD={len(good)} '
          f"(dip {sum(1 for g in good if g['good_kind']=='dipped')}/"
          f"clean {sum(1 for g in good if g['good_kind']=='clean')}); "
          f"distinct days={len({s['day'] for s in selected})}")

    meta = dict(seed=args.seed, p90_thr=thr, n_days_scanned=len(days), band=band,
                dip_pts=DIP_PTS, n_per_class=n, n_wrong=len(wrong), n_good=len(good),
                n_selected=len(selected), cap_wrong=chosen['cap_wrong'],
                cap_good=chosen['cap_good'], cap_dipped=chosen['cap_dipped'],
                cap_clean=chosen['cap_clean'])
    write_table(selected, wrong, good, meta, terminals)
    with open(SELECTION_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(meta=meta, episodes=[_manifest_row(s) for s in selected]), f, indent=2)
    print(f'[select] wrote {SELECTION_TABLE}')
    print(f'[select] wrote {SELECTION_JSON}')


if __name__ == '__main__':
    main()
