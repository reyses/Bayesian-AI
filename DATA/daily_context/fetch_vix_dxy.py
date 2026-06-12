"""Download VIX and DXY daily OHLCV via yfinance.

Tickers:
  ^VIX        -> CBOE Volatility Index (cash)
  DX-Y.NYB    -> US Dollar Index (cash, ICE)

Fallback DXY ticker if primary fails: DX=F (DXY futures continuous).

Run: python tools/sourcing/fetch_vix_dxy.py
Output:
  DATA/CROSS_DAY/raw/vix_daily.parquet
  DATA/CROSS_DAY/raw/dxy_daily.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT_DIR = Path('DATA/CROSS_DAY/raw')
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = '2024-12-01'   # 1 month before earliest ATLAS day
END   = '2026-06-01'   # 1 month past latest NT8 day

CFG = [
    ('vix', ['^VIX']),
    ('dxy', ['DX-Y.NYB', 'DX=F']),
]


def fetch_one(tickers: list[str]) -> pd.DataFrame:
    last_err = None
    for tk in tickers:
        try:
            df = yf.download(tk, start=START, end=END, progress=False,
                             auto_adjust=False, threads=False)
            if df is None or len(df) == 0:
                last_err = f'{tk}: empty'
                continue
            df = df.reset_index()
            # yfinance returns MultiIndex when multiple tickers; flatten if so
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [str(c).lower() for c in df.columns]
            if 'close' not in df.columns:
                last_err = f'{tk}: no close col; cols={list(df.columns)}'
                continue
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            last_err = f'{tk}: {e}'
            continue
    raise RuntimeError(f'All tickers failed. Last: {last_err}')


def main():
    for name, tickers in CFG:
        print(f'Fetching {name} (tickers={tickers})...')
        try:
            df = fetch_one(tickers)
        except Exception as e:
            print(f'  FAIL {name}: {e}', file=sys.stderr)
            continue
        out = OUT_DIR / f'{name}_daily.parquet'
        df.to_parquet(out, index=False)
        first = df['date'].iloc[0].strftime('%Y-%m-%d')
        last  = df['date'].iloc[-1].strftime('%Y-%m-%d')
        print(f'  {len(df):4d} rows, {first} -> {last}  -> {out}')
        print(f'  cols: {list(df.columns)}')


if __name__ == '__main__':
    main()
