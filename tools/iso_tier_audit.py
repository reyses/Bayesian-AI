"""
ISO tier-regime audit — measure each nn_v2 tier classification on current
iso trades using post-lookahead-fix features.

The question: do the 8 regime buckets from training/nightmare_blended.py
(CASCADE, KILL_SHOT, FREIGHT_TRAIN, FADE_AGAINST, RIDE_AGAINST, MTF_EXHAUSTION,
MTF_BREAKOUT, FADE_CALM) still have edge now that lookahead bias is out?

Approach — NO refitting, pure measurement:
  1. Load iso_is.pkl (277-day NMP trades, entry_91d features included).
  2. Classify each trade by running the nn_v2 decision cascade on its
     entry features.
  3. Per regime bucket, measure:
       - N trades
       - Vanilla WR, mean PnL, total PnL
       - Share that "flipped" — regime's direction ≠ iso's vanilla NMP dir
       - If corrected_is.pkl available, oracle best-action distribution
         and counter-capture delta.
  4. Verdict per regime: KEEP, FLIP, SKIP, NOISE (low volume).

Writes reports/findings/iso_tier_audit_<timestamp>.md.

Usage:
    python tools/iso_tier_audit.py
"""
import os
import sys
import pickle
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 91D iso feature indices (core 12 × 6 TFs + helper 3 × 6 TFs + 1 global) ──
# TF order: 15s(0), 1m(1), 5m(2), 15m(3), 1h(4), 1D(5)
N_CORE = 12
HELPER_START = 72
N_HELPER = 3


def core_idx(tf_idx: int, core: int) -> int:
    return tf_idx * N_CORE + core


def help_idx(tf_idx: int, helper: int) -> int:
    return HELPER_START + tf_idx * N_HELPER + helper


# Map nn_v2 constants → 91D indices
_Z = 0
_DMI = 1
_VR = 2
_VEL = 3
_ACCEL = 4
_VOL_REL = 5
_BAR_RANGE = 6
_HURST = 7
_REVERSION = 8
_P_CENTER = 9
_WICK = 2          # helper slot

TF_15S, TF_1M, TF_5M, TF_15M, TF_1H, TF_1D = 0, 1, 2, 3, 4, 5

# Tier thresholds (verbatim from training/nightmare_blended.py)
ROCHE = 2.0
WICK_5M_MIN = 0.83
WICK_15M_MIN = 0.77
VELOCITY_THRESHOLD = 50.0
FREIGHT_TRAIN_THRESHOLD = 100.0
FREIGHT_TRAIN_VR_MAX = 0.50       # freight requires stable regime (short-term compression)
H1_Z_MIN = 1.0                    # |1h_z| above this = "1h strong"
H1_AGAINST_Z_MIN = 1.5            # |1h_z| opposing fade
H1_VEL_AGAINST_MIN = 3.0          # |1h_vel| opposing fade
MTF_Z_MIN = 2.0
MTF_VR_MIN = 1.0
MTF_VOL_MIN = 1.0
MTF_5M_VEL_MIN = 10.0
MTF_1M_VEL_ALIVE = 5.0
MTF_BREAKOUT_Z_MIN = 1.3
MTF_BREAKOUT_DMI_AGAINST = -5.0


def classify_regime(feat: np.ndarray, z: float) -> tuple:
    """Port of training/nightmare_blended.py::_classify_full_tier.

    Returns (regime_direction, regime_name).
    regime_direction is the direction the regime's rule recommends —
    which may differ from vanilla fade. A "flip" regime returns the
    opposite of the NMP-fade direction.
    """
    # NMP default direction: fade the z
    fade_dir = 'short' if z > 0 else 'long'

    # Feature reads (91D indices)
    wick_5m = feat[help_idx(TF_5M, _WICK)]
    wick_15m = feat[help_idx(TF_15M, _WICK)]
    h1_z = feat[core_idx(TF_1H, _Z)]
    h1_vel = feat[core_idx(TF_1H, _VEL)]
    m1_vel = feat[core_idx(TF_1M, _VEL)]
    m1_accel = feat[core_idx(TF_1M, _ACCEL)]
    m1_vr = feat[core_idx(TF_1M, _VR)]
    m1_dmi = feat[core_idx(TF_1M, _DMI)]
    m1_hurst = feat[core_idx(TF_1M, _HURST)]
    m1_vol_rel = feat[core_idx(TF_1M, _VOL_REL)]
    v5_vel = feat[core_idx(TF_5M, _VEL)]
    v5_accel = feat[core_idx(TF_5M, _ACCEL)]
    z_5m = feat[core_idx(TF_5M, _Z)]
    z_15m = feat[core_idx(TF_15M, _Z)]
    abs_vel = abs(m1_vel)
    abs_z = abs(z)

    has_wick = (wick_5m > WICK_5M_MIN) and (wick_15m > WICK_15M_MIN)
    h1_aligned = ((fade_dir == 'long' and h1_z < -H1_Z_MIN) or
                  (fade_dir == 'short' and h1_z > H1_Z_MIN))
    h1_against_fade = ((fade_dir == 'long' and h1_z > H1_AGAINST_Z_MIN) or
                       (fade_dir == 'short' and h1_z < -H1_AGAINST_Z_MIN))
    h1_vel_against = ((fade_dir == 'long' and h1_vel < -H1_VEL_AGAINST_MIN) or
                      (fade_dir == 'short' and h1_vel > H1_VEL_AGAINST_MIN))

    # 1. FREIGHT_TRAIN — extreme velocity + accelerating + vr low
    if (abs_vel >= FREIGHT_TRAIN_THRESHOLD
            and m1_vel * m1_accel > 0
            and m1_vr < FREIGHT_TRAIN_VR_MAX):
        return ('long' if m1_vel > 0 else 'short', 'FREIGHT_TRAIN')

    # 2. KILL_SHOT — wick, no 1h
    if has_wick and not h1_aligned:
        return (fade_dir, 'KILL_SHOT')

    # 3. CASCADE — wick + 1h aligned
    if has_wick and h1_aligned:
        return (fade_dir, 'CASCADE')

    # 4. RIDE_AGAINST — 1h vel opposes, but 1h z not extreme
    if h1_vel_against and not h1_against_fade:
        return ('long' if h1_vel > 0 else 'short', 'RIDE_AGAINST')

    # 5. FADE_AGAINST — 1h z extreme against + 5m weak
    if h1_against_fade and abs(v5_vel) < 10.0:
        return (fade_dir, 'FADE_AGAINST')

    # 6. MTF_EXHAUSTION — 5m decel + 1m alive + z/vr/vol all high
    if (v5_accel < 0 and abs(v5_vel) > MTF_5M_VEL_MIN
            and abs(m1_vel) > MTF_1M_VEL_ALIVE
            and abs_z > MTF_Z_MIN
            and m1_vr > MTF_VR_MIN
            and m1_vol_rel > MTF_VOL_MIN):
        return ('long' if v5_vel > 0 else 'short', 'MTF_EXHAUSTION')

    # 7. MTF_BREAKOUT — 5m + 15m both strong z, dmi not against
    if abs(z_5m) > MTF_BREAKOUT_Z_MIN and abs(z_15m) > MTF_BREAKOUT_Z_MIN:
        breakout_dir = 'long' if z > 0 else 'short'
        dmi_aligned = ((breakout_dir == 'long' and m1_dmi > MTF_BREAKOUT_DMI_AGAINST)
                       or (breakout_dir == 'short' and m1_dmi < -MTF_BREAKOUT_DMI_AGAINST))
        if dmi_aligned:
            return (breakout_dir, 'MTF_BREAKOUT')

    # 8. FADE_CALM — default, but skip if higher TFs strongly opposing
    higher_tf_opposing = False
    if fade_dir == 'long' and v5_vel < -3 and h1_vel < -3:
        higher_tf_opposing = True
    if fade_dir == 'short' and v5_vel > 3 and h1_vel > 3:
        higher_tf_opposing = True
    if not higher_tf_opposing:
        return (fade_dir, 'FADE_CALM')

    return (None, 'NO_TRADE')


def _passes_1d_filter(t, feat, z):
    """Replicate the current engine's 1D-alignment gate on NMP_RIDE.

    The engine drops NMP_RIDE trades whose direction aligns with yesterday's
    1D dir_vol sign. Vanilla fade direction is kept as-is. For audit we
    re-derive entry_tier from the features: vr < 1.0 → FADE (always kept);
    vr >= 1.0 → RIDE (keep only if opposed to 1D dir_vol).
    """
    vr = float(feat[core_idx(TF_1M, _VR)])
    if vr < 1.0:
        return True  # FADE — always kept
    # RIDE — only kept if opposed to 1D dir_vol
    dir_vol_1d = float(feat[help_idx(TF_1D, 1)])  # helper slot 1 = dir_vol
    ride_dir = 'long' if z > 0 else 'short'
    dir_sign = 1 if ride_dir == 'long' else -1
    dv_sign = 1 if dir_vol_1d > 0 else (-1 if dir_vol_1d < 0 else 0)
    aligned = (dir_sign * dv_sign) == 1
    return not aligned


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--no-filter', action='store_true',
                   help='skip the 1D-alignment filter (see raw pre-filter trades)')
    args = p.parse_args()

    trade_path = 'training_iso/output/trades/iso_is.pkl'
    corrected_path = 'training_iso/output/trades/corrected_is.pkl'

    with open(trade_path, 'rb') as f:
        trades = pickle.load(f)
    print(f'Loaded {len(trades)} iso trades (raw)')

    corrected_map = {}
    if os.path.exists(corrected_path):
        with open(corrected_path, 'rb') as f:
            corrected = pickle.load(f)
        for c in corrected:
            key = (c.get('day'), c.get('trade_id'))
            corrected_map[key] = c
        print(f'Loaded {len(corrected_map)} corrected trades')
    else:
        print('(no corrected_is.pkl — oracle columns skipped)')

    # Per-regime accumulator
    regimes = defaultdict(lambda: {
        'n': 0, 'pnl': [], 'wins': 0, 'flips': 0,
        'oracle_actions': Counter(), 'counter_pnl': [],
        'original_tier': Counter(),
    })

    n_skipped = 0
    n_filtered = 0
    for t in trades:
        entry_91d = t.get('entry_79d')  # field name is stale; it's 91D now
        if entry_91d is None or len(entry_91d) < 91:
            n_skipped += 1
            continue
        feat = np.asarray(entry_91d, dtype=np.float32)
        z = float(feat[core_idx(TF_1M, _Z)])

        # Apply 1D-alignment filter (match current engine) unless --no-filter
        if not args.no_filter and not _passes_1d_filter(t, feat, z):
            n_filtered += 1
            continue

        regime_dir, regime_name = classify_regime(feat, z)

        bucket = regimes[regime_name]
        bucket['n'] += 1
        bucket['pnl'].append(t['pnl'])
        if t['pnl'] > 0:
            bucket['wins'] += 1
        # "Flipped" = regime rule direction differs from vanilla fade
        vanilla_dir = 'short' if z > 0 else 'long'
        if regime_dir is not None and regime_dir != vanilla_dir:
            bucket['flips'] += 1
        bucket['original_tier'][t.get('entry_tier', '?')] += 1

        # Oracle lookup
        key = (t.get('day'), t.get('trade_id'))
        c = corrected_map.get(key)
        if c is not None:
            bucket['oracle_actions'][c.get('best_action', '?')] += 1
            # Counter-direction PnL = pnl if regime FLIPPED us; else -pnl
            # Approximation: |pnl| is the magnitude; counter takes opposite sign
            bucket['counter_pnl'].append(-t['pnl'])  # sign flip of original

    # Report
    rows = []
    total_trades = sum(r['n'] for r in regimes.values())
    total_pnl = sum(sum(r['pnl']) for r in regimes.values())

    ordered = sorted(regimes.items(), key=lambda kv: -sum(kv[1]['pnl']))
    print()
    print(f'{"Regime":<17} {"N":>6} {"WR%":>5} {"MeanPnL":>8} {"TotPnL":>10} {"Flip%":>6} {"CounterMeanPnL":>14} {"OracleBestAct":>28}')
    print('-' * 115)
    for name, r in ordered:
        if r['n'] == 0:
            continue
        wr = r['wins'] / r['n'] * 100
        mean_pnl = np.mean(r['pnl'])
        tot = sum(r['pnl'])
        flip_pct = r['flips'] / r['n'] * 100
        counter_mean = np.mean(r['counter_pnl']) if r['counter_pnl'] else np.nan
        top_actions = ', '.join(f'{a}:{c}' for a, c in r['oracle_actions'].most_common(3))
        print(f'{name:<17} {r["n"]:>6,} {wr:>4.1f}% ${mean_pnl:>+7.2f} ${tot:>+9,.0f} {flip_pct:>5.0f}% ${counter_mean:>+13.2f}  {top_actions[:28]}')
        rows.append({
            'regime': name, 'n': r['n'], 'wr': wr, 'mean_pnl': mean_pnl, 'total_pnl': tot,
            'flip_share': flip_pct, 'counter_mean_pnl': counter_mean,
            'oracle_actions': dict(r['oracle_actions']),
            'original_tier': dict(r['original_tier']),
        })

    print('-' * 115)
    print(f'{"TOTAL":<17} {total_trades:>6,} {"":>5} {"":>8} ${total_pnl:>+9,.0f}')
    print()
    print(f'Skipped {n_skipped} trades (missing/short features)')
    print(f'1D-filter dropped {n_filtered} RIDE-aligned trades (use --no-filter to see them)')

    # Write markdown report
    out_dir = 'reports/findings'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md_path = os.path.join(out_dir, f'iso_tier_audit_{ts}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# ISO tier-regime audit — {ts}\n\n')
        f.write(f'Classified {total_trades:,} iso trades into nn_v2 regimes.\n')
        f.write(f'Baseline total: ${total_pnl:+,.0f} over {total_trades:,} trades.\n\n')
        f.write('## Per-regime measurement\n\n')
        f.write('| Regime | N | WR% | Mean PnL | Total PnL | Flip% | CounterMean | OracleTop3 |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---|\n')
        for r in rows:
            oracle_str = ', '.join(f'{k}:{v}' for k, v in
                                   sorted(r['oracle_actions'].items(), key=lambda kv: -kv[1])[:3])
            f.write(f'| {r["regime"]} | {r["n"]:,} | {r["wr"]:.1f}% | '
                    f'${r["mean_pnl"]:+.2f} | ${r["total_pnl"]:+,.0f} | '
                    f'{r["flip_share"]:.0f}% | ${r["counter_mean_pnl"]:+.2f} | {oracle_str} |\n')
        f.write('\n## Verdict rubric\n\n')
        f.write('- **KEEP**: WR > 55%, mean PnL > baseline, N > 100\n')
        f.write('- **SKIP**: WR < 48% OR mean PnL < -$2 (just don\'t trade this regime)\n')
        f.write('- **FLIP**: counter-mean-pnl > vanilla-mean-pnl by >$5 AND sign reliable (N > 100)\n')
        f.write('- **NOISE**: N < 100 — not enough data\n')
    print(f'\nWrote report: {md_path}')


if __name__ == '__main__':
    main()
