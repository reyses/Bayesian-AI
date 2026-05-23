"""
Loser Physics — study what makes NMP losers lose, then find where flipping
direction would have rescued them.

This is the playbook that built the original ExNMP tiers (RIDE_AGAINST,
MTF_EXHAUSTION, etc.): instead of hunting for a winner pattern, find the
consistent loser pattern, then INVERT the physics in that regime.

Pipeline:
  1. Load training_iso V2 trades + regret output
  2. Split by mode (NMP_FADE, NMP_RIDE) and outcome (win/loss)
  3. Per mode: univariate Cohen-d between losers and winners across 91
     features — the top-|d| features define the "loser regime"
  4. For each top feature, threshold by loser-mean side and check:
     - loser density in the regime (what % of losers concentrate there)
     - flippable density (of those losers, how many have
       counter_extended_valid OR counter_in_trade_valid = True)
     - expected flip PnL (sum of post_horizon_peak_counter +
       in_trade_peak_counter for flippable losers in the regime)
  5. Rank regimes by (flippable-loser count × per-flip $). Top regimes
     are the new tier candidates.

Output:
  reports/findings/loser_physics_summary.md
  reports/findings/loser_physics_{mode}.md (per-mode detail)

Usage:
  python tools/loser_physics.py                    # both modes
  python tools/loser_physics.py NMP_FADE           # one mode only
  python tools/loser_physics.py --topk 12          # more features
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.features import FEATURE_NAMES

TRADES_PATH = 'training_iso/output/trades/iso_is.pkl'
REGRET_PATH = 'training_iso/output/tree/regret_analysis.csv'
OUT_DIR = 'reports/findings'

DEFAULT_TOPK = 10
MIN_REGIME_N = 100   # don't bother with regimes smaller than this


def load_data():
    with open(TRADES_PATH, 'rb') as f:
        trades = pickle.load(f)
    regret = pd.read_csv(REGRET_PATH)

    rows = []
    for i, t in enumerate(trades):
        ef = t.get('entry_79d') or t.get('entry_features')
        if ef is None:
            continue
        if isinstance(ef, list):
            ef = np.array(ef)
        if len(ef) < 91:
            continue
        row = {f: float(ef[j]) for j, f in enumerate(FEATURE_NAMES[:91])}
        row['_pnl'] = float(t['pnl'])
        row['_tier'] = t.get('entry_tier', '?')
        row['_win'] = 1 if t['pnl'] > 0 else 0
        row['_held'] = int(t.get('held', 0))
        row['_day'] = t.get('day', '')
        rows.append(row)
    df = pd.DataFrame(rows)

    # Join regret by row order (both derive from the same trades list)
    if len(df) == len(regret):
        for col in ('counter_extended_valid', 'counter_in_trade_valid',
                    'same_extended_valid', 'same_shorter_valid',
                    'post_horizon_peak_counter', 'in_trade_peak_counter',
                    'best_action', 'regret'):
            if col in regret.columns:
                df[f'_r_{col}'] = regret[col].values
    return df


def cohen_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def analyze_mode(df_mode: pd.DataFrame, mode: str, topk: int) -> dict:
    """Per-mode analysis: find loser-concentrating features and flippability."""
    losers = df_mode[df_mode['_win'] == 0]
    winners = df_mode[df_mode['_win'] == 1]
    n_losers, n_winners = len(losers), len(winners)

    # Univariate Cohen-d: positive d = feature value higher in LOSERS
    # (we pass losers first so the sign points at the loser regime).
    ranked = []
    for feat in FEATURE_NAMES[:91]:
        d = cohen_d(losers[feat].values, winners[feat].values)
        ranked.append((feat, d))
    ranked.sort(key=lambda x: abs(x[1]), reverse=True)

    top_features = ranked[:topk]

    # For each top feature, build the "loser regime" threshold and score it
    regimes = []
    for feat, d in top_features:
        # Loser regime = side of the distribution where losers concentrate.
        # Use the median of all trades as the split point.
        split = df_mode[feat].median()
        if d > 0:
            # Losers > median; regime = feature > split
            regime_mask = df_mode[feat] > split
            regime_desc = f'{feat} > {split:.4f}'
        else:
            regime_mask = df_mode[feat] < split
            regime_desc = f'{feat} < {split:.4f}'

        regime = df_mode[regime_mask]
        if len(regime) < MIN_REGIME_N:
            continue

        r_losers = regime[regime['_win'] == 0]
        r_winners = regime[regime['_win'] == 1]
        r_wr = len(r_winners) / max(len(regime), 1) * 100
        r_pnl = regime['_pnl'].sum()
        r_avg = regime['_pnl'].mean()

        # Flippability of the losers in this regime
        if '_r_counter_extended_valid' in regime.columns:
            flippable_mask = (r_losers['_r_counter_extended_valid']
                              | r_losers['_r_counter_in_trade_valid'])
            n_flippable = int(flippable_mask.sum())
            flippable_losers = r_losers[flippable_mask]
            # Expected flip capture: for each flippable loser, the best
            # counter-direction value from the regret gates (whichever was
            # valid). Conservative: min of the two peaks.
            flip_gains = []
            for _, row in flippable_losers.iterrows():
                gains = []
                if row['_r_counter_extended_valid']:
                    gains.append(row['_r_post_horizon_peak_counter'])
                if row['_r_counter_in_trade_valid']:
                    gains.append(row['_r_in_trade_peak_counter'])
                flip_gains.append(max(gains) if gains else 0)
            flip_total = float(np.sum(flip_gains))
            flip_avg = float(np.mean(flip_gains)) if flip_gains else 0.0
            # Current pnl of these same trades (what we'd give up by flipping)
            current_pnl_on_flips = float(flippable_losers['_pnl'].sum())
            # Net delta if we flipped direction on these
            net_delta = flip_total - current_pnl_on_flips
        else:
            n_flippable = 0
            flip_total = 0.0
            flip_avg = 0.0
            current_pnl_on_flips = 0.0
            net_delta = 0.0

        regimes.append({
            'feature': feat,
            'cohen_d': d,
            'split': split,
            'regime_desc': regime_desc,
            'n_regime': len(regime),
            'n_losers_in_regime': len(r_losers),
            'regime_wr': r_wr,
            'regime_total_pnl': r_pnl,
            'regime_avg_pnl': r_avg,
            'n_flippable_losers': n_flippable,
            'flip_pct_of_losers': (n_flippable / max(len(r_losers), 1)) * 100,
            'flip_total_gain': flip_total,
            'flip_avg_per_trade': flip_avg,
            'current_pnl_flipped_subset': current_pnl_on_flips,
            'net_delta_if_flipped': net_delta,
        })

    regimes.sort(key=lambda r: r['net_delta_if_flipped'], reverse=True)

    return {
        'mode': mode,
        'n_trades': len(df_mode),
        'n_losers': n_losers,
        'n_winners': n_winners,
        'wr': n_winners / max(len(df_mode), 1) * 100,
        'top_features': top_features,
        'regimes': regimes,
    }


def write_mode_report(result: dict, out_path: str):
    lines = []
    mode = result['mode']
    lines.append(f'# Loser Physics — {mode}')
    lines.append('')
    lines.append(f'**Trades:** {result["n_trades"]:,}  '
                 f'**Losers:** {result["n_losers"]:,}  '
                 f'**Winners:** {result["n_winners"]:,}  '
                 f'**WR:** {result["wr"]:.1f}%')
    lines.append('')

    lines.append('## Top features separating LOSERS from winners (|Cohen d|)')
    lines.append('')
    lines.append('Positive d = feature value is *higher* in losers. '
                 'Negative = *lower* in losers.')
    lines.append('')
    lines.append('| Feature | Cohen d | loser regime |')
    lines.append('|---|---:|---|')
    for feat, d in result['top_features']:
        side = 'HIGH' if d > 0 else 'LOW'
        lines.append(f'| {feat} | {d:+.3f} | {side} |')
    lines.append('')

    lines.append('## Loser regimes ranked by net-delta-if-flipped')
    lines.append('')
    lines.append('For each loser regime: split trades on the regime feature, '
                 'count flippable losers (those with valid counter-direction '
                 'peaks in the regret output), and compute the PnL delta we\'d '
                 'get by flipping direction on those losers specifically.')
    lines.append('')
    lines.append('| Regime | N | loser N | WR | flippable losers | flip $/trade | net $ if flipped |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for r in result['regimes']:
        lines.append(f'| {r["regime_desc"]} | {r["n_regime"]:,} | '
                     f'{r["n_losers_in_regime"]:,} | {r["regime_wr"]:.1f}% | '
                     f'{r["n_flippable_losers"]:,} '
                     f'({r["flip_pct_of_losers"]:.0f}%) | '
                     f'${r["flip_avg_per_trade"]:+.2f} | '
                     f'${r["net_delta_if_flipped"]:+,.0f} |')
    lines.append('')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def write_summary(all_results: list, out_path: str):
    lines = []
    lines.append('# Loser Physics — Summary')
    lines.append('')
    lines.append('Each row is a candidate "loser regime" — a feature-threshold '
                 'condition where losses concentrate in the current NMP engine. '
                 '`net $ if flipped` estimates what the tier would add if we '
                 'flipped direction specifically on the flippable losers in '
                 'this regime.')
    lines.append('')
    lines.append('| Mode | Regime | N | flippable losers | flip $/trade | net $ if flipped |')
    lines.append('|---|---|---:|---:|---:|---:|')
    combined = []
    for res in all_results:
        for r in res['regimes']:
            combined.append({
                'mode': res['mode'],
                **r,
            })
    combined.sort(key=lambda r: r['net_delta_if_flipped'], reverse=True)
    for r in combined[:40]:
        lines.append(f'| {r["mode"]} | {r["regime_desc"]} | {r["n_regime"]:,} | '
                     f'{r["n_flippable_losers"]:,} '
                     f'({r["flip_pct_of_losers"]:.0f}%) | '
                     f'${r["flip_avg_per_trade"]:+.2f} | '
                     f'${r["net_delta_if_flipped"]:+,.0f} |')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modes', nargs='*', help='modes to analyze (default: all)')
    parser.add_argument('--topk', type=int, default=DEFAULT_TOPK,
                        help='top features per mode to analyze')
    args = parser.parse_args()

    print('Loading trades + regret...')
    df = load_data()
    print(f'  {len(df):,} trades loaded')
    print(f'  Tiers: {dict(df["_tier"].value_counts())}')
    print()

    modes = args.modes or ['NMP_FADE', 'NMP_RIDE']
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'{"Mode":<12} {"N":>7} {"WR":>6} {"top regime":<50} {"net $ flip":>10}')
    print('-' * 95)

    all_results = []
    for mode in modes:
        sub = df[df['_tier'] == mode]
        if len(sub) < 100:
            print(f'{mode:<12} SKIP (only {len(sub)} trades)')
            continue
        res = analyze_mode(sub, mode, args.topk)
        all_results.append(res)

        out_path = os.path.join(OUT_DIR, f'loser_physics_{mode}.md')
        write_mode_report(res, out_path)

        if res['regimes']:
            top = res['regimes'][0]
            print(f'{mode:<12} {len(sub):>7,} {res["wr"]:>5.1f}%  '
                  f'{top["regime_desc"][:50]:<50}  '
                  f'${top["net_delta_if_flipped"]:>+9,.0f}')
        else:
            print(f'{mode:<12} {len(sub):>7,} {res["wr"]:>5.1f}%  (no regimes met MIN_REGIME_N)')

    summary_path = os.path.join(OUT_DIR, 'loser_physics_summary.md')
    write_summary(all_results, summary_path)
    print()
    print(f'Summary: {summary_path}')
    for res in all_results:
        mode_name = res['mode']
        print(f'  {os.path.join(OUT_DIR, f"loser_physics_{mode_name}.md")}')


if __name__ == '__main__':
    main()
