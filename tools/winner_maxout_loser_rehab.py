"""
Winner max-out + loser rehab analysis.

Two-phase asymmetry strategy:
  Phase 1 (winner max-out): winners currently give back $ between peak
    and exit. A trail stop at X% giveback captures more of the peak.
  Phase 2 (loser rehab): losers that TOUCHED positive territory are
    "round-trippers" — catch them at a modest $+ threshold before they
    revert. Converts losers into moderate winners.

This tool is analysis only — no engine changes. For each tier per dataset:
  1. Winner giveback stats: median/mean peak, exit, giveback $, peak capture %
  2. Round-tripper stats: what % of LOSERS touched +$3, +$5, +$10, +$20?
  3. Rule simulation: apply (trail%, round_trip_threshold) combos post-hoc.
     Report $ delta per tier for IS and OOS separately.

Simplifying assumptions for post-hoc simulation:
  - Winner with peak P, current exit E: under trail_pct G, new_exit = max(P*(1-G), E).
    (The trail would fire at P*(1-G) on the way down; if current exit was better,
    keep it — our actual exit already beat the trail.)
  - Round-tripper loser (pnl<0, peak>=T): new_exit = T.
    (Assumes take-profit at T when peak touches T.)
  - Non-round-tripper loser (pnl<0, peak<T): keep original.

Usage:
    python tools/winner_maxout_loser_rehab.py
    python tools/winner_maxout_loser_rehab.py --source blended

Output: reports/findings/winner_maxout_loser_rehab_<source>.md
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TRADES_DIR = 'training_iso/output/trades'
OUT_PATH_TEMPLATE = 'reports/findings/winner_maxout_loser_rehab_{source}.md'

SOURCES = {
    'iso':     {'IS': 'iso_is.pkl',     'OOS': 'iso_oos.pkl'},
    'blended': {'IS': 'blended_is.pkl', 'OOS': 'blended_oos.pkl'},
}

TRAIL_PCTS       = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
ROUND_TRIP_THRS  = [3.0, 5.0, 7.5, 10.0, 15.0, 20.0]

# Min tier N for sweep reporting (below this, rule sim is noise)
MIN_TIER_N = 50


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def winner_rehab_stats(trades):
    """Return per-tier diagnostic dict."""
    by_tier = defaultdict(lambda: {
        'n': 0, 'winners': [], 'losers': [], 'total_pnl': 0.0,
    })
    for t in trades:
        tier = t.get('entry_tier', '?')
        pnl = float(t.get('pnl', 0.0))
        peak = float(t.get('peak', 0.0))
        rec = by_tier[tier]
        rec['n'] += 1
        rec['total_pnl'] += pnl
        if pnl > 0:
            rec['winners'].append({'pnl': pnl, 'peak': peak})
        elif pnl < 0:
            rec['losers'].append({'pnl': pnl, 'peak': peak})

    out = {}
    for tier, r in by_tier.items():
        W = r['winners']
        L = r['losers']
        w_n, l_n = len(W), len(L)
        if w_n == 0 and l_n == 0:
            continue
        # Winner stats
        w_peaks = np.array([w['peak'] for w in W]) if W else np.array([0.0])
        w_pnls  = np.array([w['pnl']  for w in W]) if W else np.array([0.0])
        w_giveback = w_peaks - w_pnls
        w_capture  = np.where(w_peaks > 0, w_pnls / w_peaks, 0.0)
        # Loser / round-tripper stats
        l_peaks = np.array([l['peak'] for l in L]) if L else np.array([0.0])
        l_pnls  = np.array([l['pnl']  for l in L]) if L else np.array([0.0])
        round_trip_rates = {thr: float((l_peaks >= thr).mean()) if l_n else 0.0
                            for thr in ROUND_TRIP_THRS}
        out[tier] = {
            'n': r['n'],
            'total_pnl': r['total_pnl'],
            'n_winners': w_n,
            'n_losers': l_n,
            'winner_mean_peak': float(w_peaks.mean()) if W else 0.0,
            'winner_mean_pnl':  float(w_pnls.mean())  if W else 0.0,
            'winner_mean_giveback': float(w_giveback.mean()) if W else 0.0,
            'winner_median_giveback': float(np.median(w_giveback)) if W else 0.0,
            'winner_mean_capture_pct':   float(w_capture.mean() * 100) if W else 0.0,
            'winner_median_capture_pct': float(np.median(w_capture) * 100) if W else 0.0,
            'loser_mean_peak': float(l_peaks.mean()) if L else 0.0,
            'loser_mean_pnl':  float(l_pnls.mean())  if L else 0.0,
            'round_trip_rates': round_trip_rates,
        }
    return out


def simulate_rules(trades, trail_pct=None, round_trip_thr=None,
                   trail_min_peak=5.0):
    """Apply (trail, round-trip) post-hoc. Return per-tier delta dict.

    trail_pct: e.g. 0.20 = trail at 20% giveback (new_exit = peak * 0.80).
               None = no trail.
    round_trip_thr: e.g. 5.0 = exit at +$5 if loser touched +$5.
                    None = no round-trip rule.
    trail_min_peak: only apply trail if peak >= this (don't trail noise trades).
    """
    by_tier = defaultdict(lambda: {'orig': 0.0, 'new': 0.0,
                                    'n_trail_fired': 0,
                                    'n_roundtrip_fired': 0,
                                    'winner_lift': 0.0,
                                    'rehab_lift': 0.0})
    for t in trades:
        tier = t.get('entry_tier', '?')
        pnl = float(t.get('pnl', 0.0))
        peak = float(t.get('peak', 0.0))
        r = by_tier[tier]
        r['orig'] += pnl
        new_pnl = pnl
        if pnl > 0:
            # Winner — apply trail if configured
            if trail_pct is not None and peak >= trail_min_peak:
                trail_exit = peak * (1 - trail_pct)
                if trail_exit > pnl:
                    new_pnl = trail_exit
                    r['n_trail_fired'] += 1
                    r['winner_lift'] += (new_pnl - pnl)
        elif pnl < 0:
            # Loser — apply round-trip rehab if configured
            if round_trip_thr is not None and peak >= round_trip_thr:
                new_pnl = round_trip_thr
                r['n_roundtrip_fired'] += 1
                r['rehab_lift'] += (new_pnl - pnl)
        r['new'] += new_pnl
    # Post-process
    out = {}
    for tier, r in by_tier.items():
        out[tier] = {
            'orig_pnl': r['orig'],
            'new_pnl': r['new'],
            'delta': r['new'] - r['orig'],
            'n_trail_fired': r['n_trail_fired'],
            'n_roundtrip_fired': r['n_roundtrip_fired'],
            'winner_lift': r['winner_lift'],
            'rehab_lift': r['rehab_lift'],
        }
    return out


def render_diag_table(label, diag, out):
    out.append(f'## {label} — winner giveback + loser round-trip diagnostics')
    out.append('')
    out.append('| Tier | N | $ | Winners | Loser | W avg peak | W avg exit | '
               'W avg giveback | W capture% median | L avg peak | '
               'RT ≥$3 | ≥$5 | ≥$10 | ≥$20 |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    tiers = sorted(diag.keys(), key=lambda t: -diag[t]['total_pnl'])
    for tier in tiers:
        d = diag[tier]
        rt = d['round_trip_rates']
        out.append(
            f'| {tier} | {d["n"]:,} | ${d["total_pnl"]:+,.0f} | '
            f'{d["n_winners"]:,} | {d["n_losers"]:,} | '
            f'${d["winner_mean_peak"]:+.2f} | '
            f'${d["winner_mean_pnl"]:+.2f} | '
            f'${d["winner_mean_giveback"]:+.2f} | '
            f'{d["winner_median_capture_pct"]:.0f}% | '
            f'${d["loser_mean_peak"]:+.2f} | '
            f'{rt[3.0]*100:.0f}% | {rt[5.0]*100:.0f}% | '
            f'{rt[10.0]*100:.0f}% | {rt[20.0]*100:.0f}% |')
    out.append('')


def render_sweep_per_tier(label, trades, diag, out):
    """Sweep (trail, round_trip) per tier for a single dataset."""
    out.append(f'## {label} — per-tier rule sweep (Δ vs baseline)')
    out.append('')
    out.append('For each tier, best trail % and best round-trip threshold '
               '(optimized independently then combined). Only tiers with '
               f'N ≥ {MIN_TIER_N} trades shown.')
    out.append('')
    tiers = [t for t, d in diag.items() if d['n'] >= MIN_TIER_N]
    tiers.sort(key=lambda t: -diag[t]['total_pnl'])

    out.append('| Tier | N | Baseline $ | Best trail % | +Trail Δ$ | '
               'Best RT $ | +RT Δ$ | Combined Δ$ | New total $ |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for tier in tiers:
        # Trail-only sweep
        best_trail_pct = None
        best_trail_delta = 0
        for tp in TRAIL_PCTS:
            sim = simulate_rules(trades, trail_pct=tp, round_trip_thr=None)
            if tier in sim and sim[tier]['delta'] > best_trail_delta:
                best_trail_delta = sim[tier]['delta']
                best_trail_pct = tp
        # Round-trip-only sweep
        best_rt_thr = None
        best_rt_delta = 0
        for rt in ROUND_TRIP_THRS:
            sim = simulate_rules(trades, trail_pct=None, round_trip_thr=rt)
            if tier in sim and sim[tier]['delta'] > best_rt_delta:
                best_rt_delta = sim[tier]['delta']
                best_rt_thr = rt
        # Combined sweep (full grid)
        best_combo_delta = 0
        best_combo = (None, None)
        for tp in TRAIL_PCTS + [None]:
            for rt in ROUND_TRIP_THRS + [None]:
                if tp is None and rt is None:
                    continue
                sim = simulate_rules(trades, trail_pct=tp, round_trip_thr=rt)
                if tier in sim and sim[tier]['delta'] > best_combo_delta:
                    best_combo_delta = sim[tier]['delta']
                    best_combo = (tp, rt)
        baseline = diag[tier]['total_pnl']
        new_total = baseline + best_combo_delta
        trail_s = f'{best_trail_pct*100:.0f}%' if best_trail_pct else '—'
        rt_s = f'${best_rt_thr}' if best_rt_thr else '—'
        out.append(
            f'| {tier} | {diag[tier]["n"]:,} | ${baseline:+,.0f} | '
            f'{trail_s} | ${best_trail_delta:+,.0f} | '
            f'{rt_s} | ${best_rt_delta:+,.0f} | '
            f'${best_combo_delta:+,.0f} | ${new_total:+,.0f} |')
    out.append('')


def render_walk_forward(trades_is, trades_oos, diag_is, diag_oos, out):
    """For each (trail, rt) combo optimal on IS, evaluate OOS impact."""
    out.append('## Walk-forward check — IS-optimal rules applied to OOS')
    out.append('')
    out.append('Per tier: the trail% and round-trip $ chosen from IS sweep, '
               'applied as-is to OOS. Positive OOS Δ = rule generalizes. '
               'Negative = rule is IS-overfit.')
    out.append('')
    tiers = sorted(set(t for t, d in diag_is.items() if d['n'] >= MIN_TIER_N)
                   & set(diag_oos.keys()),
                   key=lambda t: -diag_is[t]['total_pnl'])
    out.append('| Tier | IS best trail | IS best RT | IS Δ | OOS Δ | Status |')
    out.append('|---|---:|---:|---:|---:|---|')
    for tier in tiers:
        # Find IS best combo (same sweep logic)
        best_combo_delta = 0
        best_combo = (None, None)
        for tp in TRAIL_PCTS + [None]:
            for rt in ROUND_TRIP_THRS + [None]:
                if tp is None and rt is None:
                    continue
                sim = simulate_rules(trades_is, trail_pct=tp, round_trip_thr=rt)
                if tier in sim and sim[tier]['delta'] > best_combo_delta:
                    best_combo_delta = sim[tier]['delta']
                    best_combo = (tp, rt)
        # Apply to OOS
        oos_sim = simulate_rules(trades_oos, trail_pct=best_combo[0],
                                  round_trip_thr=best_combo[1])
        oos_delta = oos_sim.get(tier, {}).get('delta', 0)
        trail_s = f'{best_combo[0]*100:.0f}%' if best_combo[0] else '—'
        rt_s = f'${best_combo[1]}' if best_combo[1] else '—'
        if best_combo_delta <= 0:
            status = 'no IS lift'
        elif oos_delta > 0:
            status = '✓ lifts both'
        elif oos_delta >= -best_combo_delta * 0.2:
            status = '~ OOS tiny drag'
        else:
            status = '❌ IS-overfit'
        out.append(f'| {tier} | {trail_s} | {rt_s} | '
                   f'${best_combo_delta:+,.0f} | ${oos_delta:+,.0f} | {status} |')
    out.append('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=list(SOURCES.keys()), default='iso')
    args = ap.parse_args()

    files = SOURCES[args.source]
    out_path = OUT_PATH_TEMPLATE.format(source=args.source)

    is_path  = os.path.join(TRADES_DIR, files['IS'])
    oos_path = os.path.join(TRADES_DIR, files['OOS'])
    print(f'Loading {is_path}...')
    is_trades = load(is_path)
    print(f'Loading {oos_path}...')
    oos_trades = load(oos_path)

    diag_is  = winner_rehab_stats(is_trades)
    diag_oos = winner_rehab_stats(oos_trades)

    out = [f'# Winner max-out + loser rehab — {args.source}', '']
    out.append('**Strategy**: (1) trail stop to capture more of winner peaks; '
               '(2) take-profit at $T for losers that touched +$T peak '
               '("round-trippers"). Analysis only — no engine changes.')
    out.append('')

    render_diag_table('IS', diag_is, out)
    render_diag_table('OOS', diag_oos, out)
    render_sweep_per_tier('IS', is_trades, diag_is, out)
    render_sweep_per_tier('OOS', oos_trades, diag_oos, out)
    render_walk_forward(is_trades, oos_trades, diag_is, diag_oos, out)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print()
    print(f'Wrote: {out_path}')

    # Console summary: walk-forward table only
    print()
    print('=== WALK-FORWARD SUMMARY ===')
    print(f'{"Tier":<18} {"IS$":>8} {"IS-tr%":>7} {"IS-RT":>6} '
          f'{"ISd":>7} {"OOSd":>7} Status')
    tiers = sorted(
        set(t for t, d in diag_is.items() if d['n'] >= MIN_TIER_N)
        & set(diag_oos.keys()),
        key=lambda t: -diag_is[t]['total_pnl'])
    for tier in tiers:
        best_combo_delta = 0
        best_combo = (None, None)
        for tp in TRAIL_PCTS + [None]:
            for rt in ROUND_TRIP_THRS + [None]:
                if tp is None and rt is None:
                    continue
                sim = simulate_rules(is_trades, trail_pct=tp,
                                     round_trip_thr=rt)
                if tier in sim and sim[tier]['delta'] > best_combo_delta:
                    best_combo_delta = sim[tier]['delta']
                    best_combo = (tp, rt)
        oos_sim = simulate_rules(oos_trades, trail_pct=best_combo[0],
                                  round_trip_thr=best_combo[1])
        oos_delta = oos_sim.get(tier, {}).get('delta', 0)
        trail_s = f'{best_combo[0]*100:.0f}%' if best_combo[0] else '-'
        rt_s = f'${best_combo[1]:.0f}' if best_combo[1] else '-'
        status = '+' if best_combo_delta > 0 and oos_delta > 0 else (
            '?' if best_combo_delta > 0 else '.')
        print(f'{tier:<18} ${diag_is[tier]["total_pnl"]:>+7,.0f} '
              f'{trail_s:>7} {rt_s:>6} ${best_combo_delta:>+6,.0f} '
              f'${oos_delta:>+6,.0f} {status}')


if __name__ == '__main__':
    main()
