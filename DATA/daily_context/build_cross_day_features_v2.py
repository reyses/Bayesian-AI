"""Dev augmenter: read canonical cross_day_features_with_target.parquet,
join LLM news-intensity column(s), write to dev/.

This is the Phase A (and optionally Phase B) feature-engineering step. It
intentionally does NOT rebuild the canonical features -- it consumes the
production parquet as-is and augments with news_intensity columns derived
from the LLM scoring output (DATA/CROSS_DAY/dev/news_scores_v1.parquet).

Phase A (default):
  Adds one column: news_intensity_today
  (max intensity of releases with release_ts_et < 09:30 ET on each date)

Phase B (--include-prior):
  Adds a second column: news_intensity_prior
  (max intensity of releases on prior trading day, release_ts_et >= 09:30 ET)

In both phases the three useless binary flags (is_fomc, is_cpi, is_nfp)
are NOT dropped from the parquet -- the trainer FEATURE_COLS list does
the exclusion. Keeping them in the parquet preserves diagnostic info.

Inputs (read-only, canonical untouched):
  DATA/CROSS_DAY/cross_day_features_with_target.parquet
  DATA/CROSS_DAY/dev/news_scores_v1.parquet

Output:
  DATA/CROSS_DAY/dev/cross_day_features_with_target_v2.parquet

Run:
  python tools/sourcing/build_cross_day_features_v2.py            # Phase A
  python tools/sourcing/build_cross_day_features_v2.py --include-prior   # Phase B
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.sourcing.llm_news.join import (
    compute_news_intensity_today,
    compute_news_intensity_prior,
)

CANONICAL_FEATS = Path('DATA/CROSS_DAY/cross_day_features_with_target.parquet')
NEWS_SCORES = Path('DATA/CROSS_DAY/dev/news_scores_v1.parquet')
OUT_PATH = Path('DATA/CROSS_DAY/dev/cross_day_features_with_target_v2.parquet')


def main(include_prior: bool = False) -> dict:
    if not CANONICAL_FEATS.exists():
        raise FileNotFoundError(
            f'Canonical features parquet missing: {CANONICAL_FEATS}\n'
            f'Run tools/sourcing/drs_a_step4_aggregate_day_pnl.py first.'
        )
    if not NEWS_SCORES.exists():
        raise FileNotFoundError(
            f'News scores parquet missing: {NEWS_SCORES}\n'
            f'Run `python -m tools.sourcing.llm_news.cli score` first.'
        )

    feats = pd.read_parquet(CANONICAL_FEATS)
    news = pd.read_parquet(NEWS_SCORES)
    print(f'Canonical features: {len(feats)} rows, {len(feats.columns)} cols')
    print(f'  source breakdown: {feats["source"].value_counts().to_dict()}')
    print(f'News scores: {len(news)} rows')
    if len(news) > 0:
        print(f'  intensity mean={news["intensity"].mean():.2f}  '
              f'std={news["intensity"].std():.2f}  '
              f'min={news["intensity"].min()}  max={news["intensity"].max()}')
        print(f'  per event type: {news.groupby("event_type").size().to_dict()}')

    dates = feats['date']
    trading_dates_sorted = sorted(feats['date'].unique())

    feats['news_intensity_today'] = compute_news_intensity_today(news, dates).values

    if include_prior:
        feats['news_intensity_prior'] = compute_news_intensity_prior(
            news, dates, trading_dates_sorted).values
    else:
        feats['news_intensity_prior'] = np.nan

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT_PATH, index=False)
    print()
    print(f'Wrote: {OUT_PATH}  ({len(feats)} rows, {len(feats.columns)} cols)')

    print()
    print('--- news_intensity_today distribution ---')
    n_today_pos = int((feats['news_intensity_today'] > 0).sum())
    print(f'  nonzero days: {n_today_pos}/{len(feats)} '
          f'({100*n_today_pos/len(feats):.1f}%)')
    if n_today_pos > 0:
        sub = feats[feats['news_intensity_today'] > 0]
        print(f'  mean (nonzero): {sub["news_intensity_today"].mean():.2f}')
        print(f'  std  (nonzero): {sub["news_intensity_today"].std():.2f}')

    if include_prior:
        print()
        print('--- news_intensity_prior distribution ---')
        n_prior_pos = int((feats['news_intensity_prior'] > 0).sum())
        print(f'  nonzero days: {n_prior_pos}/{len(feats)} '
              f'({100*n_prior_pos/len(feats):.1f}%)')
        if n_prior_pos > 0:
            sub = feats[feats['news_intensity_prior'] > 0]
            print(f'  mean (nonzero): {sub["news_intensity_prior"].mean():.2f}')
            print(f'  std  (nonzero): {sub["news_intensity_prior"].std():.2f}')

    return {
        'n_rows': len(feats),
        'n_news_today_nonzero': n_today_pos,
        'out_path': str(OUT_PATH),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Augment canonical cross-day features with LLM news intensity')
    p.add_argument('--include-prior', action='store_true',
                   help='Also compute news_intensity_prior (Phase B). Default: Phase A only.')
    args = p.parse_args()
    main(include_prior=args.include_prior)
