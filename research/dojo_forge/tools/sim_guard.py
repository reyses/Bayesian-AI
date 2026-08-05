"""Truncate any ad-hoc query at the sim clock (2026-08-05 contamination).

The dojo tools all respect the fogged clock. My own verification queries did
not: checking "the days around the test day" printed the SEALED day's full
RTH range and close, and contaminated me on a live blind test.

Use instead of pd.read_parquet when touching the live sim day:

    from sim_guard import load_upto
    df = load_upto('2025_08_07', '1m')      # truncated at the sim clock
"""
import json
import os

import pandas as pd

REPO = '/media/moi/WindowsCode/Bayesian-AI'
STATE = os.path.join(REPO, 'research', 'dojo_forge', 'gate_state',
                     'pocket_dojo_state.json')


def sim_cutoff():
    """(day, last visible epoch second) or (None, None)."""
    try:
        s = json.load(open(STATE))
        day = s['day']
        if s.get('halt_ts5'):
            return day, int(s['halt_ts5'])
        d = pd.read_parquet(os.path.join(REPO, 'DATA', 'ATLAS', '1m',
                                         day + '.parquet'))
        return day, int(d['timestamp'].iloc[s['cur']]) + 59
    except Exception:
        return None, None


def load_upto(day, tf='1m'):
    """Read a day, HARD-TRUNCATED at the sim clock if it is the live day."""
    df = pd.read_parquet(os.path.join(REPO, 'DATA', 'ATLAS', tf,
                                      day + '.parquet'))
    live, cut = sim_cutoff()
    if live and day == live and cut:
        df = df[df['timestamp'] <= cut]
    return df


def assert_not_live(day):
    """Raise if `day` is the sealed live day — for queries that must never
    touch it at all."""
    live, _ = sim_cutoff()
    if live and day == live:
        raise RuntimeError(f'{day} is the LIVE sealed sim day — use '
                           f'load_upto() or exclude it')
