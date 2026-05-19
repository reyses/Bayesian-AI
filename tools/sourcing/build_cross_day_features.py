"""Build cross-day features for the Day-Regime Sizer (DRS).

For every trading day in DATA/ATLAS/1m/ and DATA/ATLAS_NT8/1m/, compute
session-start features that the DRS can train on. ALL features are
lookahead-honest: known at or before 09:30 ET on the target day.

Internal features (computed from MNQ 1m bars):
  - overnight_gap_pct   = (today_RTH_open - yest_RTH_close) / yest_RTH_close
  - overnight_range_pct = (max_high - min_low between 16:00 ET yest and 09:30 ET today) / yest_close
  - prior_day_range_pct = (yest_high - yest_low) / yest_open    in RTH 09:30-16:00 ET
  - prior_day_c2c_pct   = (yest_close - day_before_close) / day_before_close
  - dow                 = day-of-week (0=Mon, 4=Fri)
  - is_opex             = 1 if 3rd Friday of month

External features (joined from Stooq + calendar CSVs):
  - vix_close_prior, vix_chg_prior
  - dxy_close_prior, dxy_chg_prior
  - is_fomc, is_cpi, is_nfp
  - days_since_fomc, days_to_next_fomc

Target columns (filled later by the DRS trainer, included here as NaN):
  - target_day_pnl     (to be joined from forward-pass output)

Run: python tools/sourcing/build_cross_day_features.py
Output: DATA/CROSS_DAY/cross_day_features.parquet
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ATLAS_1M = Path('DATA/ATLAS/1m')
ATLAS_NT8_1M = Path('DATA/ATLAS_NT8/1m')
RAW_DIR = Path('DATA/CROSS_DAY/raw')
OUT_PATH = Path('DATA/CROSS_DAY/cross_day_features.parquet')

# ET trading-day windows (winter standard; DST handled by zoneinfo)
RTH_OPEN_HOUR = 9
RTH_OPEN_MIN  = 30
RTH_CLOSE_HOUR = 16   # 16:00 ET cash close (a few mins of futures spillover ignored)

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo('America/New_York')
except ImportError:
    import pytz
    ET = pytz.timezone('America/New_York')


def load_1m(day_label: str, root: Path) -> pd.DataFrame | None:
    p = root / f'{day_label}.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    # Convert int64 unix ts to UTC-aware datetime then to ET
    df['ts_utc'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df['ts_et']  = df['ts_utc'].dt.tz_convert(ET)
    df['hour_et'] = df['ts_et'].dt.hour
    df['minute_et'] = df['ts_et'].dt.minute
    df['date_et'] = df['ts_et'].dt.date
    return df


def rth_slice(df: pd.DataFrame, target_date) -> pd.DataFrame:
    """Return bars in 09:30 ET <= ts < 16:00 ET on target_date."""
    same_day = df[df['date_et'] == target_date]
    mask = ((same_day['hour_et'] > RTH_OPEN_HOUR) |
            ((same_day['hour_et'] == RTH_OPEN_HOUR) & (same_day['minute_et'] >= RTH_OPEN_MIN))) & \
           (same_day['hour_et'] < RTH_CLOSE_HOUR)
    return same_day[mask].reset_index(drop=True)


def overnight_slice(prev_df: pd.DataFrame, today_df: pd.DataFrame,
                    prev_date, today_date) -> pd.DataFrame:
    """Return bars in [16:00 ET prev_date, 09:30 ET today_date).

    Across weekends (Fri close -> Mon open), today_df contains Sun-evening
    futures bars labelled date_et=Sun -- include those too.
    """
    prev_post = prev_df[(prev_df['date_et'] == prev_date) &
                        (prev_df['hour_et'] >= RTH_CLOSE_HOUR)]
    today_pre = today_df[(today_df['date_et'] == today_date) &
                         ((today_df['hour_et'] < RTH_OPEN_HOUR) |
                          ((today_df['hour_et'] == RTH_OPEN_HOUR) &
                           (today_df['minute_et'] < RTH_OPEN_MIN)))]
    # Intervening-date bars in today_df (weekends across Fri->Mon)
    weekend = today_df[(today_df['date_et'] > prev_date) &
                       (today_df['date_et'] < today_date)]
    return pd.concat([prev_post, weekend, today_pre], ignore_index=True)


def is_third_friday(d: datetime.date) -> bool:
    if d.weekday() != 4:
        return False
    return 15 <= d.day <= 21


def parse_label(label: str) -> datetime.date:
    return datetime.strptime(label, '%Y_%m_%d').date()


def label_from_date(d: datetime.date) -> str:
    return d.strftime('%Y_%m_%d')


def load_market_csv(path: Path, value_col_candidates=('close', 'Close')) -> pd.DataFrame:
    df = pd.read_parquet(path)
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = cols_lower.get('date') or 'date'
    val_col  = next((cols_lower[c] for c in ('close', 'adj close') if c in cols_lower), None)
    if val_col is None:
        raise RuntimeError(f'No close column in {path}: cols={list(df.columns)}')
    out = df[[date_col, val_col]].copy()
    out.columns = ['date', 'close']
    out['date'] = pd.to_datetime(out['date']).dt.date
    return out.sort_values('date').reset_index(drop=True)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---- Collect all trading days (ATLAS + NT8 dedup, drop Sat/Sun) ----
    atlas_days = sorted(p.stem for p in ATLAS_1M.glob('*.parquet'))
    nt8_days   = sorted(p.stem for p in ATLAS_NT8_1M.glob('*.parquet'))
    raw_days = sorted(set(atlas_days + nt8_days))
    # Sunday futures-only sessions have no cash-market RTH; exclude.
    # (Saturdays not in data, but filter defensively.)
    all_days = [d for d in raw_days if parse_label(d).weekday() < 5]
    dropped = len(raw_days) - len(all_days)
    print(f'Trading days: {len(all_days)} (ATLAS={len(atlas_days)}, NT8={len(nt8_days)}, dropped weekend={dropped})')

    # day_label -> source root preference (ATLAS first; fall back to NT8)
    day_to_root = {d: (ATLAS_1M if d in atlas_days else ATLAS_NT8_1M) for d in all_days}

    # ---- Load external feeds ----
    vix = load_market_csv(RAW_DIR / 'vix_daily.parquet')
    dxy = load_market_csv(RAW_DIR / 'dxy_daily.parquet')
    print(f'VIX: {len(vix)} rows, {vix["date"].iloc[0]} -> {vix["date"].iloc[-1]}')
    print(f'DXY: {len(dxy)} rows, {dxy["date"].iloc[0]} -> {dxy["date"].iloc[-1]}')

    vix_map = dict(zip(vix['date'], vix['close']))
    dxy_map = dict(zip(dxy['date'], dxy['close']))

    fomc_dates = pd.read_csv(RAW_DIR / 'fomc_dates.csv')
    cpi_dates  = pd.read_csv(RAW_DIR / 'cpi_dates.csv')
    nfp_dates  = pd.read_csv(RAW_DIR / 'nfp_dates.csv')
    fomc_set = set(pd.to_datetime(fomc_dates['date']).dt.date)
    cpi_set  = set(pd.to_datetime(cpi_dates['date']).dt.date)
    nfp_set  = set(pd.to_datetime(nfp_dates['date']).dt.date)
    fomc_sorted = sorted(fomc_set)
    print(f'Calendar: FOMC={len(fomc_set)}, CPI={len(cpi_set)}, NFP={len(nfp_set)}')

    # ---- Per-day feature loop ----
    rows = []
    cache_1m: dict[str, pd.DataFrame] = {}

    def get_1m(label):
        if label not in cache_1m:
            df = load_1m(label, day_to_root.get(label, ATLAS_1M))
            cache_1m[label] = df
            # Bounded cache: keep last 3 days only
            if len(cache_1m) > 4:
                cache_1m.pop(next(iter(cache_1m)))
        return cache_1m[label]

    skipped = 0
    for i, today_lbl in enumerate(tqdm(all_days, desc='days')):
        if i < 2:
            skipped += 1
            continue  # need yesterday + day_before for c2c

        yest_lbl  = all_days[i - 1]
        prior_lbl = all_days[i - 2]

        today_df = get_1m(today_lbl)
        yest_df  = get_1m(yest_lbl)
        prior_df = get_1m(prior_lbl)
        if today_df is None or yest_df is None or prior_df is None:
            skipped += 1
            continue

        today_date = parse_label(today_lbl)
        yest_date  = parse_label(yest_lbl)
        prior_date = parse_label(prior_lbl)

        # Internal: today's RTH open (the asof-09:30 close, lookahead-safe at 09:30+ET)
        # NOTE: we use first 1m bar that STARTS at or after 09:30 ET.
        today_rth = rth_slice(today_df, today_date)
        if len(today_rth) == 0:
            skipped += 1
            continue
        today_rth_open  = float(today_rth['open'].iloc[0])    # 9:30 1m bar OPEN
        today_rth_open_ts = today_rth['timestamp'].iloc[0]

        # Yesterday's RTH range and close
        yest_rth = rth_slice(yest_df, yest_date)
        if len(yest_rth) == 0:
            skipped += 1
            continue
        yest_rth_open  = float(yest_rth['open'].iloc[0])
        yest_rth_high  = float(yest_rth['high'].max())
        yest_rth_low   = float(yest_rth['low'].min())
        yest_rth_close = float(yest_rth['close'].iloc[-1])

        # Day-before close (RTH)
        prior_rth = rth_slice(prior_df, prior_date)
        if len(prior_rth) == 0:
            skipped += 1
            continue
        prior_rth_close = float(prior_rth['close'].iloc[-1])

        # Overnight window 16:00 ET yest -> 09:30 ET today
        # yest's post-16:00 bars live in yest_df (date_et == yest_date) AND in
        # today_df if today_df starts before 16:00. Both yest_df and today_df
        # are 24h windows; merge dates correctly.
        overnight = overnight_slice(yest_df, today_df, yest_date, today_date)
        if len(overnight) == 0:
            on_high = today_rth_open
            on_low  = today_rth_open
        else:
            on_high = float(overnight['high'].max())
            on_low  = float(overnight['low'].min())

        overnight_gap_pct   = (today_rth_open - yest_rth_close) / yest_rth_close
        overnight_range_pct = (on_high - on_low) / yest_rth_close
        prior_day_range_pct = (yest_rth_high - yest_rth_low) / yest_rth_open
        prior_day_c2c_pct   = (yest_rth_close - prior_rth_close) / prior_rth_close

        # External lookups: use YESTERDAY's VIX/DXY close (known at today's open)
        vix_prior = vix_map.get(yest_date, np.nan)
        vix_d_minus2 = vix_map.get(prior_date, np.nan)
        vix_chg = (vix_prior - vix_d_minus2) if (not np.isnan(vix_prior) and not np.isnan(vix_d_minus2)) else np.nan

        dxy_prior = dxy_map.get(yest_date, np.nan)
        dxy_d_minus2 = dxy_map.get(prior_date, np.nan)
        dxy_chg = (dxy_prior - dxy_d_minus2) if (not np.isnan(dxy_prior) and not np.isnan(dxy_d_minus2)) else np.nan

        # Calendar
        is_fomc = int(today_date in fomc_set)
        is_cpi  = int(today_date in cpi_set)
        is_nfp  = int(today_date in nfp_set)
        is_opex = int(is_third_friday(today_date))

        # Days since/until FOMC
        past_fomc = [d for d in fomc_sorted if d < today_date]
        next_fomc = [d for d in fomc_sorted if d > today_date]
        days_since_fomc = (today_date - past_fomc[-1]).days if past_fomc else -1
        days_to_next_fomc = (next_fomc[0] - today_date).days if next_fomc else -1

        rows.append({
            'date_label': today_lbl,
            'date': today_date,
            'dow': today_date.weekday(),
            'source': 'NT8' if today_lbl in nt8_days else 'ATLAS',
            # Internal MNQ features
            'today_rth_open': today_rth_open,
            'yest_rth_close': yest_rth_close,
            'overnight_gap_pct': overnight_gap_pct,
            'overnight_range_pct': overnight_range_pct,
            'prior_day_range_pct': prior_day_range_pct,
            'prior_day_c2c_pct': prior_day_c2c_pct,
            # External
            'vix_close_prior': vix_prior,
            'vix_chg_prior':   vix_chg,
            'dxy_close_prior': dxy_prior,
            'dxy_chg_prior':   dxy_chg,
            # Calendar flags
            'is_fomc': is_fomc,
            'is_cpi':  is_cpi,
            'is_nfp':  is_nfp,
            'is_opex': is_opex,
            'days_since_fomc':   days_since_fomc,
            'days_to_next_fomc': days_to_next_fomc,
            # Reserved for trainer (filled by join with forward-pass output)
            'target_day_pnl': np.nan,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT_PATH, index=False)
    print(f'\nWrote: {OUT_PATH}')
    print(f'  days included: {len(out_df)} (skipped {skipped})')
    print(f'  cols: {list(out_df.columns)}')
    print('\n--- nulls per column ---')
    nulls = out_df.isna().sum()
    print(nulls[nulls > 0].to_string() if nulls.sum() > 0 else '  (none)')
    print('\n--- first 3 rows ---')
    print(out_df.head(3).to_string())
    print('\n--- summary stats (numeric cols) ---')
    print(out_df.select_dtypes(include=[np.number]).describe().to_string())


if __name__ == '__main__':
    main()
