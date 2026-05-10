"""V2-native flip-signal discovery.

Recreates the legacy 9-tier discovery methodology in V2:

    1. Run base entry (REVERSION = V2 NMP) on IS, no enhancements
    2. For each trade, compare:
         FADE_peak  = peak realized in original direction (regret.peak_pnl)
         FLIP_peak  = peak realized if we'd taken the OPPOSITE direction
                       = -mae_pnl (the most adverse moment for FADE was the
                       best moment for FLIP)
       Classify each trade:
         FADE_BETTER  : FADE_peak > FLIP_peak + MARGIN AND FADE_peak > MARGIN
         FLIP_BETTER  : FLIP_peak > FADE_peak + MARGIN AND FLIP_peak > MARGIN
         CHOP_SKIP    : neither direction had a clear peak

    3. For each of 185 V2 columns at entry, compute the effect size
       (Cohen's d) between the FADE_BETTER cohort and the FLIP_BETTER cohort.
       Columns with high |d| are CANDIDATE FLIP SIGNALS — V2 features whose
       value at entry tells us this trade should be flipped.

    4. Cross-validate the discovered signal with a held-out IS split to
       weed out IS-only noise.

Output:
    Markdown report ranking V2 columns by their flip-discrimination power,
    plus per-cohort regime / direction stats so we can read the signal.

Usage:
    python -m training_v2.tier_discovery
    python -m training_v2.tier_discovery --margin 20 --train-frac 0.7
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core_v2.features import FEATURE_NAMES
from training_iso_v2.regret import RegretLabel
from training_iso_v2.state import REGIME_VOCAB
from training_iso_v2.ledger import ClosedTrade


DEFAULT_MARGIN = 15.0   # $ — minimum peak required to declare a "better" direction
DEFAULT_TRAIN_FRAC = 0.7  # split for IS train/validate
TOP_K = 30


# ─── Load + join trades + regret ──────────────────────────────────────────

def load_joined(trades_path: str, regret_path: str) -> pd.DataFrame:
    """Join closed trades with their regret labels into one DataFrame.

    Both lists are produced in the same order, so we can zip them by index.
    Filter to entries where entry_v2 is populated (185D).
    """
    with open(trades_path, 'rb') as f:
        trades: List[ClosedTrade] = pickle.load(f)
    with open(regret_path, 'rb') as f:
        labels: List[RegretLabel] = pickle.load(f)

    if len(trades) != len(labels):
        # Some trades may have been filtered by the regret labeller (no 5s data
        # for that day, etc.). Re-join by (entry_day, entry_ts) unique key.
        label_by_key = {(l.entry_day, l.entry_ts): l for l in labels}
        rows = []
        for t in trades:
            key = (t.entry_day, t.entry_ts)
            l = label_by_key.get(key)
            if l is None:
                continue
            rows.append((t, l))
    else:
        rows = list(zip(trades, labels))

    out = []
    for t, l in rows:
        if t.entry_v2 is None or len(t.entry_v2) != len(FEATURE_NAMES):
            continue
        out.append({
            'day': t.entry_day,
            'ts': t.entry_ts,
            'direction': t.direction,
            'regime_idx': t.entry_regime_idx,
            'actual_pnl': t.pnl,
            'peak_pnl': l.peak_pnl,
            'mae_pnl': l.mae_pnl,
            'flip_peak': -l.mae_pnl,           # peak if we'd flipped direction
            'fade_peak': l.peak_pnl,
            'time_to_peak_s': l.time_to_peak_s,
            'capture_ratio': l.capture_ratio,
            'entry_v2': np.asarray(t.entry_v2, dtype=np.float32),
        })
    return pd.DataFrame(out)


# ─── Classify trades into FADE_BETTER / FLIP_BETTER / CHOP_SKIP ───────────

def classify(df: pd.DataFrame, margin: float = DEFAULT_MARGIN,
                  mode: str = 'flip') -> pd.Series:
    """Per-trade label.

    mode = 'flip'  : FADE_BETTER / FLIP_BETTER / CHOP_SKIP
                      "which direction had the higher peak"
    mode = 'winner': WINNER / LOSER / NEUTRAL
                      "did this trade make money as taken"
    mode = 'peak'  : HIGH_PEAK / LOW_PEAK / NEUTRAL
                      "did this trade reach a meaningful peak in the original direction"
    """
    cls = pd.Series(['NEUTRAL'] * len(df), index=df.index)
    if mode == 'flip':
        fade = df['fade_peak']
        flip = df['flip_peak']
        cls[:] = 'CHOP_SKIP'
        cls[(fade > margin) & (fade > flip + margin)] = 'FADE_BETTER'
        cls[(flip > margin) & (flip > fade + margin)] = 'FLIP_BETTER'
    elif mode == 'winner':
        # WINNER if realized > +margin, LOSER if realized < -margin
        actual = df['actual_pnl']
        cls[actual > margin] = 'WINNER'
        cls[actual < -margin] = 'LOSER'
    elif mode == 'peak':
        # HIGH_PEAK if fade_peak > margin (original direction had an opportunity)
        # LOW_PEAK if fade_peak <= 0 (the move was always against us in 60-min horizon)
        fade = df['fade_peak']
        cls[fade > margin] = 'HIGH_PEAK'
        cls[fade <= 0] = 'LOW_PEAK'
    else:
        raise ValueError(f"mode must be 'flip' | 'winner' | 'peak'; got {mode!r}")
    return cls


# ─── Cohen's d per V2 column ──────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference between two samples; NaN-aware."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((len(a)-1)*sa**2 + (len(b)-1)*sb**2) / max(len(a)+len(b)-2, 1))
    if pooled < 1e-9 or not np.isfinite(pooled):
        return 0.0
    return float((mb - ma) / pooled)


def feature_separation(df: pd.DataFrame, classes: pd.Series,
                              cls_a: str = 'FADE_BETTER',
                              cls_b: str = 'FLIP_BETTER',
                              ) -> pd.DataFrame:
    """For each V2 column, compute Cohen's d between two classes."""
    a_mask = (classes == cls_a).values
    b_mask = (classes == cls_b).values
    feat_matrix = np.stack(df['entry_v2'].values)  # (N, 185)

    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        col_a = feat_matrix[a_mask, j]
        col_b = feat_matrix[b_mask, j]
        # Drop NaN warmup samples so they don't poison means/stds
        col_a = col_a[~np.isnan(col_a)]
        col_b = col_b[~np.isnan(col_b)]
        if len(col_a) < 30 or len(col_b) < 30:
            continue
        d = cohens_d(col_a, col_b)
        # Decision rule: if value > median of cls_b, take action B; else action A
        # Estimate the breakpoint as the threshold that maximizes separation
        all_vals = np.concatenate([col_a, col_b])
        all_lbls = np.concatenate([np.zeros(len(col_a)), np.ones(len(col_b))])
        sorted_idx = np.argsort(all_vals)
        sorted_vals = all_vals[sorted_idx]
        sorted_lbls = all_lbls[sorted_idx]
        # Best threshold = max separation of cls_b above
        best_split_acc = max(
            sorted_lbls.mean(),                 # always-B accuracy if cls_b dominant
            1 - sorted_lbls.mean(),             # always-A
        )
        best_thr = None
        for k in range(10, len(sorted_vals) - 10, max(1, len(sorted_vals) // 100)):
            thr = sorted_vals[k]
            below_b = (sorted_lbls[:k] == 1).mean()
            above_b = (sorted_lbls[k:] == 1).mean()
            # Best decision: rule "value > thr -> B" or "value <= thr -> B"
            acc_above = max(above_b, 1 - above_b) * (len(sorted_vals) - k) / len(sorted_vals) \
                                + max(below_b, 1 - below_b) * k / len(sorted_vals)
            if acc_above > best_split_acc:
                best_split_acc = acc_above
                best_thr = thr

        rows.append({
            'feature': name,
            'cohens_d': d,
            'abs_d': abs(d),
            'mean_a': float(np.nanmean(col_a)) if len(col_a) else 0.0,
            'mean_b': float(np.nanmean(col_b)) if len(col_b) else 0.0,
            'std_a': float(np.nanstd(col_a)) if len(col_a) else 0.0,
            'std_b': float(np.nanstd(col_b)) if len(col_b) else 0.0,
            'n_a': int(len(col_a)),
            'n_b': int(len(col_b)),
            'best_split_acc': float(best_split_acc),
            'best_thr': (None if best_thr is None else float(best_thr)),
        })
    out = pd.DataFrame(rows).sort_values('abs_d', ascending=False).reset_index(drop=True)
    return out


# ─── Walk-forward survival check ──────────────────────────────────────────

def survival_check(df: pd.DataFrame, classes: pd.Series,
                          train_frac: float = DEFAULT_TRAIN_FRAC,
                          top_k: int = TOP_K,
                          cls_a: str = 'FADE_BETTER',
                          cls_b: str = 'FLIP_BETTER') -> pd.DataFrame:
    """For each top-K column, fit on first `train_frac` of data, validate on rest."""
    df_sorted = df.sort_values('ts').copy()
    classes_sorted = classes.loc[df_sorted.index].reset_index(drop=True)
    df_sorted = df_sorted.reset_index(drop=True)
    n = len(df_sorted)
    cut = int(n * train_frac)
    df_train = df_sorted.iloc[:cut]
    df_val = df_sorted.iloc[cut:]
    cls_train = classes_sorted.iloc[:cut]
    cls_val = classes_sorted.iloc[cut:]

    sep_train = feature_separation(df_train, cls_train, cls_a=cls_a, cls_b=cls_b).head(top_k)
    sep_val = feature_separation(df_val, cls_val, cls_a=cls_a, cls_b=cls_b)
    val_lookup = sep_val.set_index('feature')['cohens_d'].to_dict()

    rows = []
    for _, r in sep_train.iterrows():
        d_train = r['cohens_d']
        d_val = val_lookup.get(r['feature'], 0.0)
        same_sign = (np.sign(d_train) == np.sign(d_val)) if d_train != 0 and d_val != 0 else False
        survives = bool(same_sign and abs(d_val) >= 0.1)
        rows.append({
            'feature': r['feature'],
            'd_train': d_train,
            'd_val': d_val,
            'mean_a_train': r['mean_a'],
            'mean_b_train': r['mean_b'],
            'n_a': r['n_a'],
            'n_b': r['n_b'],
            'survives': survives,
        })
    return pd.DataFrame(rows)


# ─── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='V2 flip-signal discovery')
    p.add_argument('--trades', default='training_iso_v2/output/nmp_only.pkl')
    p.add_argument('--regret', default='training_iso_v2/output/regret_nmp.pkl')
    p.add_argument('--out', default='reports/findings/v2_tier_discovery.md')
    p.add_argument('--margin', type=float, default=DEFAULT_MARGIN,
                       help='$ margin for FADE_BETTER vs FLIP_BETTER classification')
    p.add_argument('--train-frac', type=float, default=DEFAULT_TRAIN_FRAC)
    p.add_argument('--top-k', type=int, default=TOP_K)
    p.add_argument('--mode', type=str, default='flip',
                       choices=['flip', 'winner', 'peak'],
                       help='flip: which direction wins. winner: does as-taken win. '
                              'peak: does the trade reach a meaningful peak.')
    return p.parse_args()


def main():
    args = parse_args()
    print(f'Loading trades + regret...')
    df = load_joined(args.trades, args.regret)
    print(f'  Joined: {len(df)} trades with entry_v2 populated')

    classes = classify(df, margin=args.margin, mode=args.mode)
    counts = classes.value_counts()

    # Pick which two classes we're comparing
    if args.mode == 'flip':
        cls_a, cls_b = 'FADE_BETTER', 'FLIP_BETTER'
        all_cls = ['FADE_BETTER', 'FLIP_BETTER', 'CHOP_SKIP']
    elif args.mode == 'winner':
        cls_a, cls_b = 'WINNER', 'LOSER'
        all_cls = ['WINNER', 'LOSER', 'NEUTRAL']
    else:  # peak
        cls_a, cls_b = 'HIGH_PEAK', 'LOW_PEAK'
        all_cls = ['HIGH_PEAK', 'LOW_PEAK', 'NEUTRAL']

    print(f'\nClass distribution at margin=${args.margin} (mode={args.mode}):')
    for c in all_cls:
        n = int(counts.get(c, 0))
        print(f'  {c:<14} {n:>6} ({n/len(df):.1%})')

    if int(counts.get(cls_a, 0)) < 50 or int(counts.get(cls_b, 0)) < 50:
        print('\nToo few in either class. Lower --margin or get more trades.')
        return

    sep = feature_separation(df, classes, cls_a=cls_a, cls_b=cls_b)
    print(f'\nTop {args.top_k} V2 columns by {cls_a}-vs-{cls_b} effect size '
              f'(positive d = higher value -> {cls_b}):')
    print()
    print(f'  {"feature":<32} {"d":>6} {f"mean_{cls_a}":>15} {f"mean_{cls_b}":>15} '
              f'{"split_acc":>9}')
    head = sep.head(args.top_k)
    for _, r in head.iterrows():
        print(f'  {r["feature"]:<32} {r["cohens_d"]:>+6.3f} '
                  f'{r["mean_a"]:>+14.3f} {r["mean_b"]:>+14.3f}   '
                  f'{r["best_split_acc"]:>7.1%}')

    print(f'\nWalk-forward survival on top {args.top_k} (train {args.train_frac:.0%} / val {1-args.train_frac:.0%}):')
    surv = survival_check(df, classes, train_frac=args.train_frac,
                                top_k=args.top_k, cls_a=cls_a, cls_b=cls_b)
    n_survive = int(surv['survives'].sum())
    print(f'  Survived: {n_survive}/{args.top_k}')
    print()
    print(f'  {"feature":<32} {"d_train":>8} {"d_val":>8} {"OK":>4}')
    for _, r in surv.iterrows():
        ok = 'YES' if r['survives'] else 'no'
        print(f'  {r["feature"]:<32} {r["d_train"]:>+7.3f} {r["d_val"]:>+7.3f}   {ok}')

    # Per-class cohort info
    print(f'\nPer-class actual outcomes:')
    for c in all_cls:
        sub = df[classes == c]
        if len(sub) == 0:
            continue
        print(f'  {c:<14}: n={len(sub):>5}  '
                  f'actual_pnl ${sub["actual_pnl"].sum():>+10.0f} '
                  f'(${sub["actual_pnl"].mean():>+6.2f}/trade)  '
                  f'fade_peak ${sub["fade_peak"].mean():>+6.1f}  '
                  f'flip_peak ${sub["flip_peak"].mean():>+6.1f}')

    # Save the markdown
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(f'# V2 Flip-Signal Discovery\n\n')
        f.write(f'NMP-only IS trades: {len(df)} (margin = ${args.margin})\n\n')
        f.write(f'## Class distribution\n\n')
        for c in ['FADE_BETTER', 'FLIP_BETTER', 'CHOP_SKIP']:
            n = int(counts.get(c, 0))
            f.write(f'- {c}: {n} ({n/len(df):.1%})\n')
        f.write(f'\n## Top {args.top_k} flip-discriminating V2 columns\n\n')
        f.write(f'Cohen\'s d sign convention: positive = higher value -> FLIP_BETTER cohort.\n\n')
        f.write(head.to_markdown(index=False, floatfmt='.3f'))
        f.write(f'\n\n## Walk-forward survival (train {args.train_frac:.0%}, val {1-args.train_frac:.0%})\n\n')
        f.write(surv.to_markdown(index=False, floatfmt='.3f'))
        f.write(f'\n')
    print(f'\nReport saved -> {args.out}')


if __name__ == '__main__':
    main()
