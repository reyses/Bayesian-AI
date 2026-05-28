"""Light wrapper -- runs every trade-outcome question sequentially.

Loads the per-leg excursion dataset once (IS + OOS), runs all 15 question
functions from questions.py in order, and writes one consolidated markdown
report with a verdict index up top.

Usage:
    python tools/trade_outcome_suite/run_all.py            # cache-first
    python tools/trade_outcome_suite/run_all.py --rebuild  # rebuild per-leg data
"""
from __future__ import annotations
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import excursions as ex  # noqa: E402
from questions import QUESTIONS  # noqa: E402

OUT = ex.OUT_DIR / f'{date.today().isoformat()}_trade_outcome_full_report.md'


SRC_BLURB = {
    'causal_flat': ('CAUSAL streaming zigzag, no model filters. Honest '
                    'lookahead-free leg population — includes the whipsaws a '
                    'live engine actually takes.'),
    'hardened':    ('OFFLINE zigzag (whole-day pivots, lookahead). Zero '
                    'whipsaw by construction. Findings describe an optimistic '
                    'leg population, NOT the live forward pass engine.'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=list(ex.SOURCES),
                    default=ex.DEFAULT_SOURCE,
                    help='leg-list source (default: causal_flat = honest forward pass; '
                         "'hardened' = legacy lookahead population)")
    ap.add_argument('--rebuild', action='store_true',
                    help='force rebuild of the per-leg excursion parquet cache')
    args = ap.parse_args()

    src = args.source
    print(f'Loading per-leg excursion data (source={src})...')
    IS = ex.load('IS', source=src, rebuild=args.rebuild)
    OOS = ex.load('OOS', source=src, rebuild=args.rebuild)
    print(f'  IS  {len(IS):,} legs / {IS["day"].nunique()} days')
    print(f'  OOS {len(OOS):,} legs / {OOS["day"].nunique()} days\n')

    results = []
    for fn in QUESTIONS:
        title, verdict, body = fn(IS, OOS)
        results.append((title, verdict, body))
        print(f'  [done] {title}')

    head = [
        '# Trade Outcome Suite - Full Report',
        '',
        f'Generated {date.today().isoformat()} by '
        f'`tools/trade_outcome_suite/run_all.py` on leg-list source '
        f'**`{src}`**. {SRC_BLURB.get(src, "")}',
        '',
        f'Population: IS {len(IS):,} legs / {IS["day"].nunique()} days, '
        f'OOS {len(OOS):,} legs / {OOS["day"].nunique()} days, reported '
        'separately (never pooled).',
        '',
        'Pure descriptive diagnostics -- no model fit. All dollars are MNQ '
        f'($2/point); `pnl_usd` is net of ${ex.FRICTION_USD:.0f}/leg friction. '
        f'Conditional cells carry n + 95% bootstrap CI ({ex.N_BOOT} resamples); '
        f'cells with n < {ex.MIN_CELL_N} are flagged ` !`.',
    ]
    cmp_hint = (' For comparison vs the lookahead-tainted population, '
                'run with `--source hardened`.' if src == 'causal_flat' else
                ' For the honest forward pass numbers, run with `--source causal_flat`.')
    head += ['', '## Standing caveats', '',
             '- `close ~= MFE - R`: a zigzag leg exits ~1R below its peak by '
             'construction. Every winner gives back ~1R; every loser is '
             'recovered ~1R off its low. The R-trigger exit is the system\'s '
             'adaptive stop.',
             '- These tables describe the EXISTING R-trigger exit. Every '
             'fixed-dollar overlay tested here (cut winners, lock breakeven, '
             'bail losers) loses to it on EV.',
             '- IS/OOS reported separately. Trust a cell only where both agree.',
             '- A table gives a POPULATION frequency, not a per-trade '
             'probability -- it cannot tell one trade from another at the same '
             'state.',
             f'- This report runs on the `{src}` leg-list source.' + cmp_hint,
             '']

    out_path = (ex.OUT_DIR /
                f'{date.today().isoformat()}_{src}_trade_outcome_full_report.md')
    parts = ['\n'.join(head)] + [b for _, _, b in results]
    ex.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n\n---\n\n'.join(parts), encoding='utf-8')
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
