#!/usr/bin/env python3
"""F-space label divergence (owner 2026-07-25: "at F-space level how different
are our labels?"). Parses each packet frame's 1m feature line + local line
into a compact F-vector, then measures:
  1. The F-space SIGNATURE of gen-0's exits: per-feature effect size
     (Cohen's d, exit frames vs all frames).
  2. Gen-1's response ON THE SAME FRAMES (paired by episode+frame): median
     p_exit on gen-0-exit frames vs elsewhere. Elevated => same boundary,
     merely damped. Flat => the boundary genuinely moved.
CPU-only; reads existing jsonl + packets. Writes
reports/fspace_label_divergence.md.
"""
import glob
import json
import math
import os
import re
import statistics as st

DOJO = os.path.join(os.path.dirname(__file__), '..')
PACKETS = os.path.join(DOJO, 'reports', 'gen0', 'packets')
OUT = os.path.join(DOJO, 'reports', 'fspace_label_divergence.md')

F_KEYS = ['price_velocity_30', 'price_accel_30', 'reversion_prob_30',
          'band_pos_30', 'hurst_30', 'z_se_30', 'vr_exact', 'z_21']
KV = re.compile(r'(\w+)=([+-]?\d+(?:\.\d+)?)')
GB = re.compile(r'giveback (\d+)%')
LEG = re.compile(r'leg age (\d+)m amp ([\d.]+)pts')


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
            m = GB.search(s)
            if m:
                f['giveback_pct'] = float(m.group(1))
            m = LEG.search(s)
            if m:
                f['leg_age_m'] = float(m.group(1))
                f['leg_amp'] = float(m.group(2))
    return f


def load_runs(path):
    out = {}
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
    gen1 = {}
    for name in ('acceptance_results_gen1_oneshot_partial.jsonl',
                 'acceptance_results_gen1.jsonl'):
        p = os.path.join(DOJO, 'gate_state', name)
        if os.path.exists(p):
            gen1.update(load_runs(p))

    feats_exit, feats_all = {}, {}
    g1_on_exit, g1_elsewhere = [], []
    n_exit_frames = 0
    for eid, rec in gen0.items():
        pkt_path = os.path.join(PACKETS, f'{eid}.json')
        if not os.path.exists(pkt_path):
            continue
        pkt = json.load(open(pkt_path))
        g1rec = gen1.get(eid)
        g1_by_idx = ({f['frame_idx']: f for f in g1rec['frames']}
                     if g1rec else {})
        for fr in rec['frames']:
            i = fr['frame_idx']
            if i >= len(pkt['frames']) or fr.get('decision') is None:
                continue
            fv = parse_frame(pkt['frames'][i]['text'])
            is_exit = fr['decision'] == 'EXIT'
            n_exit_frames += is_exit
            for k, v in fv.items():
                feats_all.setdefault(k, []).append(v)
                if is_exit:
                    feats_exit.setdefault(k, []).append(v)
            g1f = g1_by_idx.get(i)
            if g1f and g1f.get('p_exit') is not None:
                (g1_on_exit if is_exit else g1_elsewhere).append(g1f['p_exit'])

    rows = []
    for k in sorted(feats_all, key=lambda k: -abs(cohens_d(feats_exit.get(k, []), feats_all[k]))):
        d = cohens_d(feats_exit.get(k, []), feats_all[k])
        rows.append(f"| {k} | {st.mean(feats_exit.get(k, [0])):+.3f} "
                    f"| {st.mean(feats_all[k]):+.3f} | {d:+.2f} |")

    def q(xs, p):
        return st.quantiles(xs, n=100)[p - 1] if len(xs) >= 10 else float('nan')

    lines = [
        '# F-space label divergence — gen-0 exits vs gen-1 response',
        f'gen-0: {len(gen0)} eps, {n_exit_frames} EXIT frames. '
        f'gen-1 overlap: {len(g1_on_exit)} paired exit-frames, '
        f'{len(g1_elsewhere)} paired non-exit frames.',
        '',
        '## 1. F-space signature of gen-0 exits (Cohen\'s d, exit vs all)',
        '| feature | mean@exit | mean@all | d |', '|---|---|---|---|',
        *rows,
        '',
        '## 2. Gen-1 response on the SAME frames (paired)',
        f'- median p_exit on gen-0-EXIT frames: '
        f'{st.median(g1_on_exit) if g1_on_exit else float("nan"):.4f} '
        f'(p90 {q(g1_on_exit, 90):.4f})',
        f'- median p_exit elsewhere:            '
        f'{st.median(g1_elsewhere) if g1_elsewhere else float("nan"):.4f} '
        f'(p90 {q(g1_elsewhere, 90):.4f})',
        '',
        'Reading: if gen-1 p_exit is materially ELEVATED on the frames where '
        'gen-0 pulled the trigger, the two labelers share a decision boundary '
        'and the genome only damped the gain. If flat, the boundary MOVED — '
        'education/rules changed WHERE the teacher looks, not just how hard '
        'it fires.',
    ]
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
