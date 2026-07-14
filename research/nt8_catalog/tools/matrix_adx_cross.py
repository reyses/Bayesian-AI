"""
Compact parity MATRIX for the doc-071 ruling (Claude-executed).

Answers two questions with numbers, not opinions:
  1. ADX-08-SMA (legacy smoothing) vs ADX-08-WILDER (canonical RMA) — Moises' "keep both"
     ruling. Does the smoothing choice actually change the events?
  2. CROSS-11 restored to FIRST-CROSS-ONLY (doc 070/071) — does it now match legacy?

Reuses verify_batch_b's daily-context builder verbatim; samples N days for speed and
prints an aggregate matrix (per-day verbose output is what made the full run unusable).
A day MATCHES if native's first trigger equals legacy's first trigger in BOTH timestamp
and mode (doc 066 rule: count+ts+mode, no vibes).
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))
sys.path.insert(0, ROOT)

from core_v2.FPS.forward_pass_system import ForwardPassSystem          # noqa: E402

# verify_batch_b.py hijacks sys.stdout AT MODULE LEVEL (its lines 3-4:
# `out_file = open("verifier_output.txt","w"); sys.stdout = out_file`), so merely
# IMPORTING it silently redirects this whole process's output into that file — and
# truncates it each time. Save/restore around the import so our results reach stdout.
_real_stdout = sys.stdout
import verify_batch_b as V                                             # noqa: E402
sys.stdout = _real_stdout

from batch_b_detectors import ADX08_SMA_Detector, ADX08_Wilder_Detector, CROSS11Detector  # noqa: E402

RTH0, RTH1 = pd.Timestamp('08:30').time(), pd.Timestamp('15:15').time()
N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def rth_ts(day):
    df = pd.read_parquet(os.path.join(ROOT, 'DATA', 'ATLAS', '5s', f'{day}.parquet'),
                         columns=['timestamp'])
    dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    m = (dt.dt.time >= RTH0) & (dt.dt.time <= RTH1)
    return df['timestamp'].values[m.values].astype(np.int64)


def legacy_first(dossier, day, ts_map):
    p = os.path.join(HERE, '..', 'tests', dossier, 'events.parquet')
    df = pd.read_parquet(p)
    d = df[df['day'] == day]
    if len(d) == 0:
        return None
    r = d.iloc[0]
    i = int(r['event_idx'])
    ts = int(ts_map[i]) if i < len(ts_map) else 0
    return {'ts': ts, 'mode': str(r['mode'])}


def main():
    daily, valid = V.build_daily_context()
    days = valid[15:][:N_DAYS]          # skip warmup days needing 14-day context
    print(f'Sampling {len(days)} days: {days[0]} .. {days[-1]}\n')

    stat = {k: {'match': 0, 'div': 0, 'native_only': 0, 'legacy_only': 0, 'both_none': 0}
            for k in ['ADX-08-SMA', 'ADX-08-WILDER', 'CROSS-11']}
    adx_disagree = 0
    examples = []

    for day in days:
        idx = valid.index(day)
        try:
            ts_map = rth_ts(day)
            prior = valid[idx - 1]
            pre = pd.read_parquet(os.path.join(ROOT, 'DATA', 'ATLAS', '5s', f'{prior}.parquet'),
                                  columns=['close'])['close'].values.tolist()
            tod = pd.read_parquet(os.path.join(ROOT, 'DATA', 'ATLAS', '5s', f'{day}.parquet'),
                                  columns=['timestamp', 'close'])
            dtt = pd.to_datetime(tod['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
            pre.extend(tod['close'].values[(dtt.dt.time < RTH0).values].tolist())

            dets = {'ADX-08-SMA': ADX08_SMA_Detector(),
                    'ADX-08-WILDER': ADX08_Wilder_Detector(),
                    'CROSS-11': CROSS11Detector(prefill_closes=pre)}
            first = {k: None for k in dets}

            fps = ForwardPassSystem(day=day, atlas_root=os.path.join(ROOT, 'DATA', 'ATLAS'),
                                    features_root=os.path.join(ROOT, 'DATA', 'ATLAS', 'FEATURES_5s_v2'),
                                    labels_csv=os.path.join(ROOT, 'DATA', 'ATLAS', 'regime_labels_2d.csv'),
                                    tfs=['5s'], layers=['L1'], build_v2_dict=False, use_5s_price=True)
            for st in fps:
                for k, d in dets.items():
                    s, m = d.on_bar(st)
                    if s != 0 and first[k] is None:
                        first[k] = {'ts': int(st.ohlcv_5s['timestamp']), 'mode': m}
        except Exception as e:
            print(f'  [skip {day}: {type(e).__name__}]')
            continue

        # ADX variants: do they disagree with EACH OTHER? (the ruling's real question)
        a, b = first['ADX-08-SMA'], first['ADX-08-WILDER']
        if (a is None) != (b is None) or (a and b and (a['ts'] != b['ts'] or a['mode'] != b['mode'])):
            adx_disagree += 1
            if len(examples) < 3:
                examples.append((day, a, b))

        for k, doss in [('ADX-08-SMA', 'ADX-08_Trend_Gate'),
                        ('ADX-08-WILDER', 'ADX-08_Trend_Gate'),
                        ('CROSS-11', 'CROSS-11_Golden_Cross')]:
            leg = legacy_first(doss, day, ts_map)
            nat = first[k]
            if nat is None and leg is None:
                stat[k]['both_none'] += 1
            elif nat is None:
                stat[k]['legacy_only'] += 1
            elif leg is None:
                stat[k]['native_only'] += 1
            elif nat['ts'] == leg['ts'] and nat['mode'] == leg['mode']:
                stat[k]['match'] += 1
            else:
                stat[k]['div'] += 1

    print(f"{'detector':16} {'MATCH':>6} {'DIVERGE':>8} {'nat-only':>9} {'leg-only':>9} {'both-0':>7}")
    for k, s in stat.items():
        print(f"{k:16} {s['match']:>6} {s['div']:>8} {s['native_only']:>9} "
              f"{s['legacy_only']:>9} {s['both_none']:>7}")

    print(f"\nADX SMA-vs-WILDER disagree on {adx_disagree}/{len(days)} days "
          f"({100*adx_disagree/max(1,len(days)):.0f}%)")
    for day, a, b in examples:
        print(f"  {day}: SMA={a} | WILDER={b}")


if __name__ == '__main__':
    main()
