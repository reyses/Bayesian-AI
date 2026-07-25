#!/usr/bin/env python3
"""GROUND-TRUTH label comparison in F-space (owner 2026-07-25: "I meant our
ground truth labels"). Oracle exit = the episode's true peak (max favorable
px). Measures:
  1. What a perfect exit was WORTH vs never-bail (peak - final px) — the
     exit-head's ceiling on these episodes, vs friction.
  2. How far each labeler sits from the oracle: gen-0 exits (minutes early,
     points captured), never-bail (giveback at expiry).
  3. The F-space signature AT the oracle peaks — is the top visible in the
     features at all? Compare against gen-0's (giveback-panic) signature.
CPU-only. Writes reports/fspace_groundtruth.md.
"""
import glob
import json
import math
import os
import re
import statistics as st

DOJO = os.path.join(os.path.dirname(__file__), '..')
PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
OUT = os.path.join(DOJO, 'reports', 'fspace_groundtruth.md')

F_KEYS = ['price_velocity_30', 'price_accel_30', 'reversion_prob_30',
          'band_pos_30', 'hurst_30', 'z_se_30', 'vr_exact', 'z_21']
KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
PX = re.compile(r'px ([+-]?\d+(?:\.\d+)?)pts')
GB = re.compile(r'giveback (\d+)%')


def parse_frame(text):
    f = {}
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('[1m]'):
            kv = dict(KV.findall(s))
            for k in F_KEYS:
                if k in kv:
                    f[k] = float(kv[k])
        elif s.startswith('local:'):
            m = PX.search(s)
            if m:
                f['px'] = float(m.group(1))
            m = GB.search(s)
            if m:
                f['giveback_pct'] = float(m.group(1))
    return f


def load_runs(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    out[r['episode_id']] = r
                except json.JSONDecodeError:
                    pass
    return out


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sp = math.sqrt((st.pvariance(a) + st.pvariance(b)) / 2) or 1e-9
    return (st.mean(a) - st.mean(b)) / sp


def main():
    gen0 = load_runs(os.path.join(DOJO, 'gate_state', 'acceptance_results_tiered.jsonl'))
    on_table, nb_giveback = [], []
    g0_early_min, g0_left_pts = [], []
    peak_feats, all_feats = {}, {}
    n_eps = 0
    for pkt_path in sorted(glob.glob(os.path.join(PACKETS, '*.json'))):
        eid = os.path.basename(pkt_path).replace('.json', '')
        pkt = json.load(open(pkt_path))
        fvs = [parse_frame(fr['text']) for fr in pkt['frames']]
        pxs = [f.get('px') for f in fvs]
        if not any(p is not None for p in pxs):
            continue
        n_eps += 1
        valid = [(i, p) for i, p in enumerate(pxs) if p is not None]
        peak_i, peak_px = max(valid, key=lambda t: t[1])
        final_px = valid[-1][1]
        on_table.append(peak_px - final_px)
        nb_giveback.append(peak_px - final_px)
        for i, f in enumerate(fvs):
            for k in F_KEYS:
                if k in f:
                    all_feats.setdefault(k, []).append(f[k])
                    if i == peak_i:
                        peak_feats.setdefault(k, []).append(f[k])
        g0 = gen0.get(eid)
        if g0 and g0.get('exit_frame') is not None:
            ei = g0['exit_frame']
            if ei < len(pxs) and pxs[ei] is not None:
                g0_early_min.append(peak_i - ei)
                g0_left_pts.append(peak_px - pxs[ei])

    rows = []
    for k in sorted(all_feats, key=lambda k: -abs(cohens_d(peak_feats.get(k, []), all_feats[k]))):
        d = cohens_d(peak_feats.get(k, []), all_feats[k])
        rows.append(f"| {k} | {st.mean(peak_feats.get(k, [0])):+.3f} "
                    f"| {st.mean(all_feats[k]):+.3f} | {d:+.2f} |")

    lines = [
        '# Ground-truth (oracle) labels vs ours — F-space',
        f'{n_eps} episodes. Oracle exit = true peak of favorable px.',
        '',
        '## 1. What a perfect exit was worth (the exit-head ceiling here)',
        f'- peak−final (points left by never-bail): mean {st.mean(on_table):+.1f}, '
        f'median {st.median(on_table):+.1f}, p90 {st.quantiles(on_table, n=10)[-1]:+.1f} pts/ep',
        f'- friction floor: ~0.9 pts RT — the ceiling is {st.mean(on_table)/0.9:.0f}x friction on average.',
        '',
        '## 2. Distance of our labels from the oracle',
        f'- NEVER-BAIL: leaves the full peak−final on the table (above) but pays zero churn.',
        f'- GEN-0 (N={len(g0_early_min)} exited eps): median {st.median(g0_early_min) if g0_early_min else 0:.0f} min '
        f'EARLY vs oracle; median {st.median(g0_left_pts) if g0_left_pts else 0:+.1f} pts left vs peak '
        f'(exits before the move develops — worse than never-bail both ways).',
        '',
        '## 3. Is the top VISIBLE in F-space? (features at oracle peaks vs all frames)',
        '| feature | mean@peak | mean@all | d |', '|---|---|---|---|',
        *rows,
        '',
        'Reading: large |d| = the true top has an observable signature these '
        'features capture (an exit head CAN learn it). All |d| small = tops '
        'are F-space-invisible here and never-bail wins by information '
        'default — the curriculum cannot teach exits.',
    ]
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
