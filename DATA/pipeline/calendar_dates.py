"""Hand-curated economic calendar dates for cross-day feature build.

VERIFY THESE: the dates below are my best-effort recall of publicly-announced
FOMC / CPI / NFP release dates. They are NOT scraped from official sources.
If exact accuracy matters, replace each CSV with values from:
  - FOMC:   federalreserve.gov/monetarypolicy/fomccalendars.htm
  - CPI:    bls.gov/schedule/news_release/cpi.htm
  - NFP:    bls.gov/schedule/news_release/empsit.htm

Run once: python tools/sourcing/calendar_dates.py
Output:   DATA/CROSS_DAY/raw/{fomc,cpi,nfp}_dates.csv
"""
from __future__ import annotations
import csv
from pathlib import Path

OUT_DIR = Path('DATA/CROSS_DAY/raw')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FOMC announcement days (day 2 of the 2-day meeting; press conference day)
FOMC_DATES = [
    # 2025
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-10',
    # 2026 (announced schedule)
    '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17',
    '2026-07-29', '2026-09-16', '2026-11-04', '2026-12-16',
]

# CPI release dates (BLS, typically 2nd Tuesday or Wednesday of month)
CPI_DATES = [
    # 2025
    '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10',
    '2025-05-13', '2025-06-11', '2025-07-15', '2025-08-12',
    '2025-09-11', '2025-10-15', '2025-11-13', '2025-12-10',
    # 2026
    '2026-01-14', '2026-02-11', '2026-03-11', '2026-04-14',
    '2026-05-13', '2026-06-10', '2026-07-15', '2026-08-12',
    '2026-09-10', '2026-10-14', '2026-11-12', '2026-12-10',
]

# NFP release dates (1st Friday of month; shifted if 1st Friday is a holiday)
NFP_DATES = [
    # 2025
    '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04',
    '2025-05-02', '2025-06-06', '2025-07-03', '2025-08-01',
    '2025-09-05', '2025-10-03', '2025-11-07', '2025-12-05',
    # 2026
    '2026-01-09', '2026-02-06', '2026-03-06', '2026-04-03',
    '2026-05-01', '2026-06-05', '2026-07-02', '2026-08-07',
    '2026-09-04', '2026-10-02', '2026-11-06', '2026-12-04',
]


def _write(name: str, dates: list[str]) -> Path:
    p = OUT_DIR / f'{name}_dates.csv'
    with p.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'event'])
        for d in dates:
            w.writerow([d, name.upper()])
    print(f'  {name}: {len(dates):3d} dates -> {p}')
    return p


if __name__ == '__main__':
    print('Writing calendar CSVs to', OUT_DIR)
    _write('fomc', FOMC_DATES)
    _write('cpi',  CPI_DATES)
    _write('nfp',  NFP_DATES)
    print('\nVERIFY: see header comment for official sources.')
