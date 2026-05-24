"""
Cycle 03 — Signal capture only: does the RM pivot correctly identify
price direction from that point?

No exit logic. Pure entry-quality test.

Method:
  1. RM zigzag at R=$4 (Cycle 1 sweet spot)
  2. At each confirmed pivot, call direction: LOW → LONG, HIGH → SHORT
  3. Look forward at horizons H ∈ {5, 10, 15, 30, 60, 120, 240} minutes
  4. Move = (price[entry+H] − entry_price) × direction_sign
  5. Aggregate: hit-rate (% move > 0), mean/median signed move, mean |move|

Output:
  research/rm_pivot/findings/YYYY-MM-DD_signal_capture.md
  research/rm_pivot/findings/YYYY-MM-DD_signal_capture.png

Reproduction:
  python tools/measure_rm_pivot_entry_direction.py
"""
import os
import sys
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Paths ──────────────────────────────────────────────────────────────
ATLAS_1M_DIR = 'DATA/ATLAS/1m'
FINDINGS_DIR = 'research/rm_pivot/findings'

# ── Constants ──────────────────────────────────────────────────────────
REG_WINDOW = 60
R_DOLLARS = 4.0   # overridable via --r
DOLLAR_PER_POINT = 2.0
R_POINTS = R_DOLLARS / DOLLAR_PER_POINT
PIVOT_SOURCE = 'rm'  # 'rm' or 'price' — overridable via --source
# Positive = look forward (Q1: did price go our way?)
# Negative = look backward (Q2: was pre-trend against us, confirming a reversal?)
HORIZONS_MIN = [-60, -30, -15, -10, -5, 5, 10, 15, 30, 60, 120, 240]
EOD_UTC_SEC = 20 * 3600 + 55 * 60


# ══════════════════════════════════════════════════════════════════════
# Math
# ══════════════════════════════════════════════════════════════════════

def rolling_rm(closes, window=REG_WINDOW):
    n = len(closes)
    rm = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)
    xm = x.mean()
    dx = x - xm
    denom = float((dx * dx).sum())
    if denom < 1e-12:
        return rm
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        ym = y.mean()
        slope = float((dx * (y - ym)).sum() / denom)
        intercept = ym - slope * xm
        rm[i] = intercept + slope * (window - 1)
    return rm


def zigzag_confirmations(series, r_points):
    out = []
    leg_dir = None
    extreme_val = None
    for i in range(len(series)):
        v = series[i]
        if np.isnan(v):
            continue
        if extreme_val is None:
            extreme_val = v
            continue
        if leg_dir is None:
            if v - extreme_val >= r_points:
                leg_dir, extreme_val = 'up', v
            elif extreme_val - v >= r_points:
                leg_dir, extreme_val = 'down', v
            else:
                extreme_val = v
        elif leg_dir == 'up':
            if v > extreme_val:
                extreme_val = v
            elif extreme_val - v >= r_points:
                out.append((i, 'HIGH'))
                leg_dir, extreme_val = 'down', v
        else:
            if v < extreme_val:
                extreme_val = v
            elif v - extreme_val >= r_points:
                out.append((i, 'LOW'))
                leg_dir, extreme_val = 'up', v
    return out


def seconds_past_midnight(ts):
    return int(ts) % 86400


# ══════════════════════════════════════════════════════════════════════
# Per-day collection
# ══════════════════════════════════════════════════════════════════════

def collect_entries(day_path):
    """For each RM pivot, collect signed moves at multiple horizons (past
    and future) PLUS the oracle-best exit ($ captured if we could pick the
    most-favorable bar inside each future window)."""
    day = os.path.basename(day_path).replace('.parquet', '')
    df = (pd.read_parquet(day_path)
          .sort_values('timestamp').reset_index(drop=True))
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64) if 'high' in df.columns else closes
    lows = df['low'].values.astype(np.float64) if 'low' in df.columns else closes
    ts_1m = df['timestamp'].values.astype(np.int64)

    # Source for zigzag: regression mean (smoothed) or raw price
    if PIVOT_SOURCE == 'price':
        source_series = closes
    else:
        source_series = rolling_rm(closes, REG_WINDOW)
    confs = zigzag_confirmations(source_series, R_POINTS)

    rows = []
    for conf_idx, ptype in confs:
        entry_ts = int(ts_1m[conf_idx])
        if seconds_past_midnight(entry_ts) >= EOD_UTC_SEC:
            continue

        direction = 'LONG' if ptype == 'LOW' else 'SHORT'
        dir_sign = +1 if direction == 'LONG' else -1
        entry_price = float(closes[conf_idx])

        for H in HORIZONS_MIN:
            target_ts = entry_ts + H * 60
            idx = np.searchsorted(ts_1m, target_ts, side='right') - 1
            if H > 0 and (idx <= conf_idx or idx >= len(closes)):
                continue
            if H < 0 and (idx < 0 or idx >= conf_idx):
                continue
            if seconds_past_midnight(int(ts_1m[idx])) >= EOD_UTC_SEC:
                continue

            price_at_h = float(closes[idx])
            signed_move = (price_at_h - entry_price) * dir_sign * DOLLAR_PER_POINT

            # Oracle-best exit: for future H, the max favorable move inside [entry, entry+H]
            oracle_move = float('nan')
            if H > 0:
                end_idx = idx + 1
                if direction == 'LONG':
                    best_price = float(highs[conf_idx + 1: end_idx].max()) if end_idx > conf_idx + 1 else entry_price
                    oracle_move = (best_price - entry_price) * DOLLAR_PER_POINT
                else:
                    best_price = float(lows[conf_idx + 1: end_idx].min()) if end_idx > conf_idx + 1 else entry_price
                    oracle_move = (entry_price - best_price) * DOLLAR_PER_POINT

            rows.append({
                'day': day,
                'entry_ts': entry_ts,
                'pivot_type': ptype,
                'direction': direction,
                'entry_price': entry_price,
                'horizon_min': H,
                'price_at_h': price_at_h,
                'signed_move': signed_move,
                'oracle_move': oracle_move,
            })
    return rows


# ══════════════════════════════════════════════════════════════════════
# Aggregation + report
# ══════════════════════════════════════════════════════════════════════

def summarize(df_rows, label):
    """Per-horizon summary."""
    out = []
    for H in HORIZONS_MIN:
        sub = df_rows[df_rows.horizon_min == H]
        if len(sub) == 0:
            continue
        moves = sub.signed_move.values
        hit_rate = float((moves > 0).mean()) * 100
        mean_move = float(moves.mean())
        median_move = float(np.median(moves))
        mean_abs = float(np.mean(np.abs(moves)))
        out.append({
            'label': label,
            'horizon': H,
            'n': len(sub),
            'hit_rate': hit_rate,
            'mean_move': mean_move,
            'median_move': median_move,
            'mean_abs': mean_abs,
        })
    return out


def summarize_split(df_rows, label):
    """Per-horizon × direction split (LONG vs SHORT)."""
    out = []
    for H in HORIZONS_MIN:
        for d in ['LONG', 'SHORT']:
            sub = df_rows[(df_rows.horizon_min == H) & (df_rows.direction == d)]
            if len(sub) == 0:
                continue
            moves = sub.signed_move.values
            out.append({
                'label': label,
                'horizon': H,
                'direction': d,
                'n': len(sub),
                'hit_rate': float((moves > 0).mean()) * 100,
                'mean_move': float(moves.mean()),
            })
    return out


def render_report(is_rows, oos_rows):
    lines = []
    lines.append('# RM Pivot — Signal Portfolio Dashboard (Cycle 03)')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f'Ref: `research/rm_pivot/cycle_03.md`')
    lines.append('')
    lines.append('**Big question**: can we trade the RM-pivot signal reliably enough?')
    lines.append('Each sub-question below is a necessary component. All need to pass.')
    lines.append('')
    lines.append('## Method')
    lines.append('')
    lines.append(f'- RM zigzag at R=${R_DOLLARS:.0f} (Cycle 1 sweet spot)')
    lines.append('- At each confirmed pivot: LOW → LONG, HIGH → SHORT')
    lines.append(f'- Horizons (min): {HORIZONS_MIN} (negative = backward, positive = forward)')
    lines.append('- signed_move = (price[entry+H] − entry_price) × dir_sign × $2/pt')
    lines.append('- oracle_move = max favorable move within (entry, entry+H] × $2/pt (forward H only)')
    lines.append('- EOD cutoff 20:55 UTC: pivots and horizons past it are dropped')
    lines.append('')

    def _q1_direction(rows, label):
        """Q1: does price move in called direction at future horizons? (hit-rate)"""
        lines.append(f'### {label} — Q1: Direction hit-rate at forward horizons')
        lines.append('')
        lines.append('| H (min) | N | Hit-rate | Mean signed | Median | Gate |')
        lines.append('|---:|---:|---:|---:|---:|---|')
        for s in summarize(rows, label):
            if s['horizon'] <= 0:
                continue
            gate = '✓' if s['hit_rate'] >= 55 else '✗'
            lines.append(
                f'| +{s["horizon"]} | {s["n"]} | **{s["hit_rate"]:.1f}%** '
                f'| ${s["mean_move"]:+.2f} | ${s["median_move"]:+.2f} | {gate} |'
            )
        lines.append('')

    def _q2_turning(rows, label):
        """Q2: was there actually a reversal? signed_move at BACKWARD
        horizons should be NEGATIVE (pre-trend was AGAINST our direction)."""
        lines.append(f'### {label} — Q2: Turning-point correctness at backward horizons')
        lines.append('')
        lines.append('(Signed move at past = (past_price − entry_price) × dir. Should be NEGATIVE at a real reversal — we came AGAINST the called direction.)')
        lines.append('')
        lines.append('| H (min) | N | % pre-trend against us | Mean signed (past) | Gate |')
        lines.append('|---:|---:|---:|---:|---|')
        for s in summarize(rows, label):
            if s['horizon'] >= 0:
                continue
            # Hit-rate at negative horizons = % of moves > 0 = price went in our direction vs past price
            # "pre-trend against us" = % where signed_move < 0 = 100 - hit_rate
            pct_against = 100 - s['hit_rate']
            gate = '✓' if pct_against >= 55 else '✗'
            lines.append(
                f'| {s["horizon"]} | {s["n"]} | **{pct_against:.1f}%** '
                f'| ${s["mean_move"]:+.2f} | {gate} |'
            )
        lines.append('')

    def _q3_magnitude(rows, label):
        """Q3: how much $ is capturable per trade at each horizon? (mean signed move)"""
        lines.append(f'### {label} — Q3: Mean $ captured at exit timing = horizon')
        lines.append('')
        lines.append('(Just mean signed move at forward horizons — what you get if you exit exactly at H min.)')
        lines.append('')
        lines.append('| H (min) | Mean $ | Median $ | Mean \\|move\\| | Gate ($5) |')
        lines.append('|---:|---:|---:|---:|---|')
        for s in summarize(rows, label):
            if s['horizon'] <= 0:
                continue
            gate = '✓' if s['mean_move'] >= 5 else '✗'
            lines.append(
                f'| +{s["horizon"]} | ${s["mean_move"]:+.2f} '
                f'| ${s["median_move"]:+.2f} | ${s["mean_abs"]:.2f} | {gate} |'
            )
        lines.append('')

    def _q4_oracle(rows, label):
        """Q4: oracle ceiling — what if we could exit at best price in window?"""
        lines.append(f'### {label} — Q4: Oracle-best exit ceiling (perfect timing)')
        lines.append('')
        lines.append('(For each pivot, max favorable price within (entry, entry+H]. Upper bound on what any exit rule can achieve.)')
        lines.append('')
        lines.append('| H (min) | N | Mean oracle $ | Median oracle $ | P25 | P75 |')
        lines.append('|---:|---:|---:|---:|---:|---:|')
        for H in HORIZONS_MIN:
            if H <= 0:
                continue
            sub = rows[rows.horizon_min == H]
            sub = sub[sub.oracle_move.notna()]
            if len(sub) == 0:
                continue
            arr = sub.oracle_move.values
            lines.append(
                f'| +{H} | {len(arr)} | ${arr.mean():+.2f} '
                f'| ${np.median(arr):+.2f} '
                f'| ${np.percentile(arr, 25):+.2f} '
                f'| ${np.percentile(arr, 75):+.2f} |'
            )
        lines.append('')

    def _q5_daily(rows, label):
        """Q5: day-level aggregation — fraction of days where signal stacks positive."""
        lines.append(f'### {label} — Q5: Daily aggregation (signed move pooled by day)')
        lines.append('')
        lines.append('For each day, sum signed_move across that day\'s pivots, per horizon.')
        lines.append('')
        lines.append('| H (min) | n_days | DayWR (%days>0) | Daily mean | Daily median | Daily p25 | Daily p75 |')
        lines.append('|---:|---:|---:|---:|---:|---:|---:|')
        for H in HORIZONS_MIN:
            if H <= 0:
                continue
            sub = rows[rows.horizon_min == H]
            if len(sub) == 0:
                continue
            daily = sub.groupby('day')['signed_move'].sum().values
            day_wr = float((daily > 0).mean()) * 100
            lines.append(
                f'| +{H} | {len(daily)} | **{day_wr:.0f}%** '
                f'| ${daily.mean():+.2f} '
                f'| ${np.median(daily):+.2f} '
                f'| ${np.percentile(daily, 25):+.2f} '
                f'| ${np.percentile(daily, 75):+.2f} |'
            )
        lines.append('')

    def _section(title, rows):
        lines.append(f'## {title}')
        lines.append('')
        _q1_direction(rows, title)
        _q2_turning(rows, title)
        _q3_magnitude(rows, title)
        _q4_oracle(rows, title)
        _q5_daily(rows, title)

    _section('IS (2025)', is_rows)
    _section('OOS (2026)', oos_rows)

    # ── PORTFOLIO SUMMARY ──
    def _portfolio(rows, label):
        s = summarize(rows, label)
        fwd = [x for x in s if x['horizon'] > 0]
        back = [x for x in s if x['horizon'] < 0]

        q1_best_hr = max((x['hit_rate'] for x in fwd), default=0)
        q1_ge55 = sum(1 for x in fwd if x['hit_rate'] >= 55)
        q1_pass = q1_ge55 >= 2

        q2_best_against = max(((100 - x['hit_rate']) for x in back), default=0)
        q2_ge55 = sum(1 for x in back if (100 - x['hit_rate']) >= 55)
        q2_pass = q2_ge55 >= 2

        q3_best_mean = max((x['mean_move'] for x in fwd), default=0)
        q3_ge5 = sum(1 for x in fwd if x['mean_move'] >= 5)
        q3_pass = q3_ge5 >= 1

        # Q4 oracle ceiling — max mean oracle across horizons
        q4_best = 0.0
        for H in HORIZONS_MIN:
            if H <= 0:
                continue
            sub = rows[rows.horizon_min == H]
            sub = sub[sub.oracle_move.notna()]
            if len(sub) > 0:
                q4_best = max(q4_best, float(sub.oracle_move.mean()))
        q4_pass = q4_best >= 20  # meaningful oracle ceiling

        # Q5 daily WR — best day-wr across horizons
        q5_best_daywr = 0.0
        for H in HORIZONS_MIN:
            if H <= 0:
                continue
            sub = rows[rows.horizon_min == H]
            if len(sub) == 0:
                continue
            daily = sub.groupby('day')['signed_move'].sum().values
            day_wr = float((daily > 0).mean()) * 100
            q5_best_daywr = max(q5_best_daywr, day_wr)
        q5_pass = q5_best_daywr >= 60

        return {
            'label': label,
            'q1': (q1_best_hr, q1_pass),
            'q2': (q2_best_against, q2_pass),
            'q3': (q3_best_mean, q3_pass),
            'q4': (q4_best, q4_pass),
            'q5': (q5_best_daywr, q5_pass),
            'total_pass': sum([q1_pass, q2_pass, q3_pass, q4_pass, q5_pass]),
        }

    is_p = _portfolio(is_rows, 'IS')
    oos_p = _portfolio(oos_rows, 'OOS')

    lines.append('## Signal Portfolio Dashboard — Can we trade reliably enough?')
    lines.append('')
    lines.append('| Sub-question | Metric (best across horizons) | IS | OOS | Gate | IS✓ | OOS✓ |')
    lines.append('|---|---|---:|---:|---|:---:|:---:|')
    lines.append(f'| Q1 Direction right? | hit-rate % | {is_p["q1"][0]:.1f}% | {oos_p["q1"][0]:.1f}% | ≥55% at 2+ H | {"✓" if is_p["q1"][1] else "✗"} | {"✓" if oos_p["q1"][1] else "✗"} |')
    lines.append(f'| Q2 Real turning point? | pre-trend-against-us % | {is_p["q2"][0]:.1f}% | {oos_p["q2"][0]:.1f}% | ≥55% at 2+ H | {"✓" if is_p["q2"][1] else "✗"} | {"✓" if oos_p["q2"][1] else "✗"} |')
    lines.append(f'| Q3 Enough $ per trade? | mean signed $ | ${is_p["q3"][0]:+.2f} | ${oos_p["q3"][0]:+.2f} | ≥$5 at 1+ H | {"✓" if is_p["q3"][1] else "✗"} | {"✓" if oos_p["q3"][1] else "✗"} |')
    lines.append(f'| Q4 Oracle-exit ceiling | mean max-favorable $ | ${is_p["q4"][0]:+.2f} | ${oos_p["q4"][0]:+.2f} | ≥$20 | {"✓" if is_p["q4"][1] else "✗"} | {"✓" if oos_p["q4"][1] else "✗"} |')
    lines.append(f'| Q5 Signal stacks by day? | best DayWR | {is_p["q5"][0]:.0f}% | {oos_p["q5"][0]:.0f}% | ≥60% | {"✓" if is_p["q5"][1] else "✗"} | {"✓" if oos_p["q5"][1] else "✗"} |')
    lines.append('')
    lines.append(f'**IS portfolio**: {is_p["total_pass"]}/5 gates pass')
    lines.append(f'**OOS portfolio**: {oos_p["total_pass"]}/5 gates pass')
    lines.append('')
    if is_p['total_pass'] >= 4 and oos_p['total_pass'] >= 4:
        lines.append('**Verdict: TRADEABLE signal** — fix the exit in Cycle 4.')
    elif is_p['total_pass'] <= 1 or oos_p['total_pass'] <= 1:
        lines.append('**Verdict: SIGNAL DEAD** — abandon RM-pivot, pivot research direction.')
    else:
        lines.append('**Verdict: MARGINAL** — signal exists but edge is thin. Investigate which sub-questions fail and whether they can be filtered.')
    lines.append('')

    lines.append('## Reproduction')
    lines.append('')
    lines.append('```')
    lines.append('python tools/measure_rm_pivot_entry_direction.py')
    lines.append('```')
    lines.append('')
    return '\n'.join(lines), (is_p, oos_p)


def plot_signal(is_rows, oos_rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: hit-rate vs horizon (IS & OOS)
    ax = axes[0]
    is_sum = summarize(is_rows, 'IS')
    oos_sum = summarize(oos_rows, 'OOS')
    if is_sum:
        ax.plot([s['horizon'] for s in is_sum], [s['hit_rate'] for s in is_sum],
                'o-', color='tab:blue', label='IS')
    if oos_sum:
        ax.plot([s['horizon'] for s in oos_sum], [s['hit_rate'] for s in oos_sum],
                's-', color='tab:orange', label='OOS')
    ax.axhline(50, color='black', linewidth=0.6, label='random')
    ax.axhline(55, color='tab:green', linestyle='--', linewidth=0.8, label='55% gate')
    ax.set_xlabel('Horizon (min)')
    ax.set_ylabel('Hit-rate (% moves in called direction)')
    ax.set_title('Direction quality vs horizon')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(40, 80)

    # Right: mean signed move vs horizon
    ax = axes[1]
    if is_sum:
        ax.plot([s['horizon'] for s in is_sum], [s['mean_move'] for s in is_sum],
                'o-', color='tab:blue', label='IS')
    if oos_sum:
        ax.plot([s['horizon'] for s in oos_sum], [s['mean_move'] for s in oos_sum],
                's-', color='tab:orange', label='OOS')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axhline(5, color='tab:green', linestyle='--', linewidth=0.8, label='$5 gate')
    ax.set_xlabel('Horizon (min)')
    ax.set_ylabel('Mean signed move ($)')
    ax.set_title('Magnitude in called direction')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    global R_DOLLARS, R_POINTS, PIVOT_SOURCE
    ap = argparse.ArgumentParser()
    ap.add_argument('--r', type=float, default=R_DOLLARS,
                    help='Zigzag retracement $ (default 4)')
    ap.add_argument('--source', choices=['rm', 'price'], default='rm',
                    help='Zigzag source series (default rm)')
    ap.add_argument('--tag', default='',
                    help='Tag suffix on output filename')
    args = ap.parse_args()
    R_DOLLARS = args.r
    R_POINTS = args.r / DOLLAR_PER_POINT
    PIVOT_SOURCE = args.source

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    print(f'IS: {len(is_paths)} days  |  OOS: {len(oos_paths)} days')
    print(f'Source: {PIVOT_SOURCE}  R=${R_DOLLARS}')

    def _run(paths, label):
        all_rows = []
        for p in tqdm(paths, desc=label, unit='day'):
            rows = collect_entries(p)
            all_rows.extend(rows)
        return pd.DataFrame(all_rows)

    is_df = _run(is_paths, 'IS')
    oos_df = _run(oos_paths, 'OOS')
    print(f'IS entries: {len(is_df)//len(HORIZONS_MIN) if len(is_df) else 0} pivots × {len(HORIZONS_MIN)} horizons')
    print(f'OOS entries: {len(oos_df)//len(HORIZONS_MIN) if len(oos_df) else 0} pivots × {len(HORIZONS_MIN)} horizons')

    os.makedirs(FINDINGS_DIR, exist_ok=True)
    date_tag = datetime.now().strftime('%Y-%m-%d')
    tag_suffix = f'_{args.tag}' if args.tag else f'_{args.source}R{int(args.r)}'
    md_path = os.path.join(FINDINGS_DIR, f'{date_tag}_signal_capture{tag_suffix}.md')
    png_path = os.path.join(FINDINGS_DIR, f'{date_tag}_signal_capture{tag_suffix}.png')

    report, gate_info = render_report(is_df, oos_df)
    report += f'\n## Chart\n\n![signal capture]({os.path.basename(png_path)})\n'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    plot_signal(is_df, oos_df, png_path)

    print()
    is_p, oos_p = gate_info
    print(f'IS portfolio: {is_p["total_pass"]}/5 gates — Q1 {is_p["q1"][0]:.1f}% HR, '
          f'Q2 {is_p["q2"][0]:.1f}% pre-against, Q3 ${is_p["q3"][0]:+.2f} mean, '
          f'Q4 ${is_p["q4"][0]:+.2f} oracle, Q5 {is_p["q5"][0]:.0f}% DayWR')
    print(f'OOS portfolio: {oos_p["total_pass"]}/5 gates — Q1 {oos_p["q1"][0]:.1f}% HR, '
          f'Q2 {oos_p["q2"][0]:.1f}% pre-against, Q3 ${oos_p["q3"][0]:+.2f} mean, '
          f'Q4 ${oos_p["q4"][0]:+.2f} oracle, Q5 {oos_p["q5"][0]:.0f}% DayWR')
    print(f'Wrote: {md_path}')
    print(f'Wrote: {png_path}')


if __name__ == '__main__':
    main()
