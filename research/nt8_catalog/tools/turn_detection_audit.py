"""
TURN-DETECTION AUDIT — anchor at the KNOWN label flips; measure what signals sit
on them and how accurately each stream detects them (Moises 2026-07-16).

Ground truth: interior label boundaries (label k>=1 start == end of label k-1;
labels chain). Test years (2025+26) only, RTH.

Per stream (from reports/signal_rows_<det>.parquet):
  - RECALL@W: fraction of turns with >=1 fire within +-W (day-block CI at W=2m)
  - DIR-RECALL@W: ...with >=1 fire whose direction == the NEW label's direction
  - PRECISION@W: fraction of the stream's fires within +-W of any turn
  - CHANCE@W: fraction of RTH seconds within +-W of a turn (the null for precision)
  - LEAD/LAG: median & mode of (nearest matched fire ts - turn ts) in minutes
    (negative = early warning)
Output: reports/turn_detection_audit.md, sorted by DIR-RECALL@2m.
"""
import os, glob, json, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dossier_signal_pipeline import LBL, REP, day_block_ci

W_LIST = [60, 120]          # +-1m, +-2m windows (seconds)
RTH_SECONDS = 6.75 * 3600


def load_turns():
    """Per test day: list of (turn_ts, new_dir_is_long). Interior boundaries only."""
    turns = {}
    for f in glob.glob(os.path.join(LBL, 'ai_picks_*_multi.json')):
        iso = os.path.basename(f)[9:19]
        if iso[:4] == '2024':
            continue
        tr = [t for t in json.load(open(f)).get('trades', []) if t.get('exit_ts')]
        tr.sort(key=lambda t: t['entry_ts'])
        day = iso.replace('-', '_')
        turns[day] = [(t['entry_ts'], t.get('direction') == 'LONG') for t in tr[1:]]
    return turns


def audit_stream(det, turns):
    p = os.path.join(REP, f'signal_rows_{det}.parquet')
    F = pd.read_parquet(p, columns=['ts', 'is_long', 'day'])
    F = F[F['day'].str[:4] != '2024']
    if len(F) < 100:
        return None
    res = {}
    for W in W_LIST:
        hit, dhit, hit_days, leads = [], [], [], []
        n_fires = 0
        fire_near = 0
        for day, tl in turns.items():
            g = F[F['day'] == day]
            if len(tl) == 0:
                continue
            ts = np.sort(g['ts'].values)
            n_fires += len(ts)
            tarr = np.array([t for t, _ in tl])
            if len(ts):
                # precision side: fires within W of any turn
                idx = np.searchsorted(tarr, ts)
                near = np.zeros(len(ts), dtype=bool)
                for k_off in (-1, 0):
                    kk = np.clip(idx + k_off, 0, len(tarr) - 1)
                    near |= np.abs(ts - tarr[kk]) <= W
                fire_near += int(near.sum())
            for t0, new_long in tl:
                m = g[(g['ts'] >= t0 - W) & (g['ts'] <= t0 + W)]
                hit.append(int(len(m) > 0))
                dhit.append(int((m['is_long'] == new_long).any()))
                hit_days.append(day)
                if len(m):
                    j = (m['ts'] - t0).abs().idxmin()
                    leads.append((m.loc[j, 'ts'] - t0) / 60.0)
        hit = np.array(hit); dhit = np.array(dhit)
        r = dict(n_turns=len(hit), recall=float(hit.mean()), dir_recall=float(dhit.mean()),
                 precision=(fire_near / n_fires if n_fires else np.nan), n_fires=n_fires)
        if W == 120:
            lo, hi = day_block_ci(dhit.astype(float), np.array(hit_days))
            r['dir_ci'] = (lo, hi)
            if leads:
                la = np.array(leads)
                hb = np.histogram(la, bins=np.arange(-2.25, 2.5, 0.5))
                r['lead_median'] = float(np.median(la))
                r['lead_mode'] = float(hb[1][np.argmax(hb[0])] + 0.25)
        res[W] = r
    return res


def main():
    turns = load_turns()
    n_turns_total = sum(len(v) for v in turns.values())
    # chance precision: fraction of RTH time within +-W of a turn
    chance = {}
    for W in W_LIST:
        cov = []
        for day, tl in turns.items():
            if not tl:
                continue
            iv = sorted((t - W, t + W) for t, _ in tl)
            merged = []
            for a, b in iv:
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
            cov.append(min(1.0, sum(b - a for a, b in merged) / RTH_SECONDS))
        chance[W] = float(np.mean(cov))
    dets = sorted(os.path.basename(f)[12:-8] for f in glob.glob(os.path.join(REP, 'signal_rows_*.parquet')))
    rows = []
    for det in dets:
        r = audit_stream(det, turns)
        if r is None:
            continue
        rows.append((det, r))
        r2 = r[120]
        print(f"{det:14} dir-recall@2m {r2['dir_recall']:.2f} recall {r2['recall']:.2f} "
              f"prec {r2['precision']:.2f} (chance {chance[120]:.2f}) "
              f"lead med {r2.get('lead_median', float('nan')):+.1f}m N_fires {r2['n_fires']}")
    rows.sort(key=lambda x: -x[1][120]['dir_recall'])
    lines = [f'# Turn-detection audit — {n_turns_total} interior label turns (test 2025+26)',
             f'(chance precision: ±1m {chance[60]:.2f}, ±2m {chance[120]:.2f} — fraction of RTH within a window of some turn)\n',
             '| stream | dir-recall@2m [CI] | recall@2m | recall@1m | precision@2m | lead med/mode (min) | fires |',
             '|---|---|---|---|---|---|---|']
    for det, r in rows:
        r2, r1 = r[120], r[60]
        ci = r2.get('dir_ci', (np.nan, np.nan))
        lines.append(f"| {det} | {r2['dir_recall']:.2f} [{ci[0]:.2f},{ci[1]:.2f}] | {r2['recall']:.2f} "
                     f"| {r1['recall']:.2f} | {r2['precision']:.2f} | "
                     f"{r2.get('lead_median', float('nan')):+.1f} / {r2.get('lead_mode', float('nan')):+.1f} | {r2['n_fires']} |")
    lines.append(f'\nPrecision null: a fire placed at random lands within ±2m of a turn '
                 f'{chance[120]:.0%} of the time — any precision must beat that, not 0.')
    out = os.path.join(REP, 'turn_detection_audit.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('wrote', out)


if __name__ == '__main__':
    main()
