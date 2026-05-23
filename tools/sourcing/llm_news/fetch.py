"""Press release scrapers for Fed (FOMC) and BLS (CPI / NFP).

For each (date, event_type) in DATA/CROSS_DAY/raw/{fomc,cpi,nfp}_dates.csv,
fetch the official release HTML, strip to plain text, archive to
DATA/CROSS_DAY/raw/press_releases/{date}_{event_type}.txt, and append
a row to a JSON sidecar with the release timestamp in ET.

Pattern modeled after tools/sourcing/fetch_vix_dxy.py:
  - Idempotent: skip if .txt already exists.
  - Failures logged to _fetch_errors.log, do not abort the run.
  - Output is additive only -- nothing in production reads this directory.

URL patterns (best-effort; archives are not formally guaranteed):
  FOMC:  https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm
  CPI:   https://www.bls.gov/news.release/archives/cpi_{MMDDYYYY}.htm
  NFP:   https://www.bls.gov/news.release/archives/empsit_{MMDDYYYY}.htm

If the primary URL 404s, the failure is logged and processing continues.
The user can manually save the release text to the expected path and
re-run the scoring step.

Release times (ET, hard-coded; matches official schedules):
  FOMC: 14:00 ET (statement; press conference at 14:30)
  CPI:  08:30 ET
  NFP:  08:30 ET

Run:
  python -m tools.sourcing.llm_news.cli fetch
"""
from __future__ import annotations
import json
from datetime import datetime, time
from pathlib import Path
from typing import Iterable

import pandas as pd

RAW_DIR = Path('DATA/CROSS_DAY/raw')
OUT_DIR = RAW_DIR / 'press_releases'
ERR_LOG = OUT_DIR / '_fetch_errors.log'
META_OUT = OUT_DIR / '_metadata.json'

USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

RELEASE_TIMES_ET: dict[str, time] = {
    'fomc': time(14, 0),
    'cpi':  time(8, 30),
    'nfp':  time(8, 30),
}


def _fomc_url(d: datetime.date) -> str:
    return f'https://www.federalreserve.gov/newsevents/pressreleases/monetary{d.strftime("%Y%m%d")}a.htm'


def _cpi_url(d: datetime.date) -> str:
    return f'https://www.bls.gov/news.release/archives/cpi_{d.strftime("%m%d%Y")}.htm'


def _nfp_url(d: datetime.date) -> str:
    return f'https://www.bls.gov/news.release/archives/empsit_{d.strftime("%m%d%Y")}.htm'


URL_BUILDERS = {'fomc': _fomc_url, 'cpi': _cpi_url, 'nfp': _nfp_url}


def _html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
        tag.decompose()
    text = soup.get_text(separator='\n')
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return '\n'.join(lines)


def _append_error(msg: str) -> None:
    ts = datetime.utcnow().isoformat()
    ERR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ERR_LOG.open('a', encoding='utf-8') as f:
        f.write(f'{ts}  {msg}\n')


def _read_dates_csv(event: str) -> list[datetime.date]:
    p = RAW_DIR / f'{event}_dates.csv'
    if not p.exists():
        raise FileNotFoundError(f'Missing calendar CSV: {p}. Run tools/sourcing/calendar_dates.py first.')
    df = pd.read_csv(p)
    return [pd.Timestamp(d).date() for d in df['date']]


def fetch_one(event: str, d: datetime.date) -> tuple[bool, str]:
    """Fetch one release; return (ok, message)."""
    import requests
    url = URL_BUILDERS[event](d)
    try:
        r = requests.get(url, timeout=30, headers={'User-Agent': USER_AGENT})
    except requests.RequestException as e:
        return False, f'{event} {d} {url} :: REQUEST {e}'
    if r.status_code != 200:
        return False, f'{event} {d} {url} :: HTTP {r.status_code}'
    text = _html_to_text(r.text)
    if len(text) < 200:
        return False, f'{event} {d} {url} :: TEXT_TOO_SHORT ({len(text)} chars)'
    out_path = OUT_DIR / f'{d.isoformat()}_{event}.txt'
    out_path.write_text(text, encoding='utf-8')
    return True, str(out_path)


def main(events: Iterable[str] = ('fomc', 'cpi', 'nfp')) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {'ok': 0, 'skip': 0, 'fail': 0, 'per_event': {}}
    metadata: list[dict] = []
    for event in events:
        ev_ok = ev_skip = ev_fail = 0
        try:
            dates = _read_dates_csv(event)
        except FileNotFoundError as e:
            _append_error(str(e))
            summary['per_event'][event] = {'ok': 0, 'skip': 0, 'fail': 1}
            summary['fail'] += 1
            continue
        print(f'== {event.upper()}: {len(dates)} dates ==')
        for d in dates:
            out_path = OUT_DIR / f'{d.isoformat()}_{event}.txt'
            if out_path.exists():
                ev_skip += 1
                metadata.append({
                    'date': d.isoformat(),
                    'event_type': event,
                    'release_ts_et': pd.Timestamp.combine(d, RELEASE_TIMES_ET[event])
                                       .tz_localize('America/New_York')
                                       .isoformat(),
                    'path': str(out_path),
                    'status': 'cached',
                })
                continue
            ok, msg = fetch_one(event, d)
            if ok:
                ev_ok += 1
                print(f'  OK   {d.isoformat()}  -> {msg}')
                metadata.append({
                    'date': d.isoformat(),
                    'event_type': event,
                    'release_ts_et': pd.Timestamp.combine(d, RELEASE_TIMES_ET[event])
                                       .tz_localize('America/New_York')
                                       .isoformat(),
                    'path': msg,
                    'status': 'fetched',
                })
            else:
                ev_fail += 1
                print(f'  FAIL {msg}')
                _append_error(msg)
        summary['per_event'][event] = {'ok': ev_ok, 'skip': ev_skip, 'fail': ev_fail}
        summary['ok'] += ev_ok
        summary['skip'] += ev_skip
        summary['fail'] += ev_fail

    META_OUT.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print()
    print(f'Total: ok={summary["ok"]} skip={summary["skip"]} fail={summary["fail"]}')
    print(f'Metadata: {META_OUT}  ({len(metadata)} entries)')
    if summary['fail']:
        print(f'See {ERR_LOG} for failure details.')
    return summary


if __name__ == '__main__':
    main()
