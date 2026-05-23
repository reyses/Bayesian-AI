"""Single CLI entry point for the LLM news-intensity module.

Individual steps:
  python -m tools.sourcing.llm_news.cli fetch
  python -m tools.sourcing.llm_news.cli score [--skip-synthetic]
  python -m tools.sourcing.llm_news.cli test-synthetic

Full sequential pipeline (the light wrapper):
  python -m tools.sourcing.llm_news.cli run                     # Phase A: fetch -> score -> build -> train
  python -m tools.sourcing.llm_news.cli run --phase B           # Phase B: build -> train (assumes A done)
  python -m tools.sourcing.llm_news.cli run --skip-fetch        # iterate scoring without re-scraping
  python -m tools.sourcing.llm_news.cli run --skip-fetch --skip-score   # re-build + re-train only

Build (augmenter) and train (dev DRS) can also be invoked directly:
  python tools/sourcing/build_cross_day_features_v2.py [--include-prior]
  python tools/sourcing/drs_canonical_gbm_v2.py [--include-prior]
"""
from __future__ import annotations
import argparse
import sys
import time
import traceback


def _banner(text: str) -> None:
    line = '=' * 78
    print()
    print(line)
    print(text)
    print(line)


def _step(idx: int, total: int, name: str) -> float:
    print()
    print('-' * 78)
    print(f'[{idx}/{total}] {name}')
    print('-' * 78)
    return time.time()


def _done(t0: float) -> None:
    print(f'  ... done in {time.time() - t0:.1f}s')


def cmd_run(args) -> int:
    """Run the full Phase A (or B) pipeline sequentially."""
    phase = args.phase.upper()
    if phase not in ('A', 'B'):
        print(f'Unknown phase: {args.phase!r}. Use A or B.', file=sys.stderr)
        return 2

    _banner(f'LLM news-intensity pipeline -- Phase {phase}')
    overall_t0 = time.time()

    if phase == 'A':
        total = 4
        if not args.skip_fetch:
            t0 = _step(1, total, 'FETCH press releases')
            from tools.sourcing.llm_news import fetch
            fetch.main()
            _done(t0)
        else:
            _step(1, total, 'FETCH (skipped via --skip-fetch)')

        if not args.skip_score:
            t0 = _step(2, total, 'SCORE releases (loads LLM; includes synthetic anti-cheating test)')
            from tools.sourcing.llm_news import score
            score.main(skip_synthetic=False)
            _done(t0)
        else:
            _step(2, total, 'SCORE (skipped via --skip-score)')

        t0 = _step(3, total, 'BUILD dev feature parquet (Phase A: news_intensity_today only)')
        from tools.sourcing import build_cross_day_features_v2 as bld
        bld.main(include_prior=False)
        _done(t0)

        t0 = _step(4, total, 'TRAIN dev DRS GBM (Phase A) + gate verdict')
        from tools.sourcing import drs_canonical_gbm_v2 as trn
        result = trn.main(include_prior=False)
        _done(t0)

    else:  # phase == 'B'
        if not args.skip_fetch and not args.skip_score:
            print('NOTE: Phase B assumes Phase A already ran (news_scores_v1.parquet exists).')
            print('      Skipping fetch + score by default. Use --force-rescore to override.')
        total = 2

        if args.force_rescore:
            t0 = _step(1, 4, 'FETCH press releases (--force-rescore)')
            from tools.sourcing.llm_news import fetch
            fetch.main()
            _done(t0)
            t0 = _step(2, 4, 'SCORE releases (--force-rescore)')
            from tools.sourcing.llm_news import score
            score.main(skip_synthetic=False)
            _done(t0)
            total = 4
            offset = 2
        else:
            offset = 0

        t0 = _step(offset + 1, total + offset,
                    'BUILD dev feature parquet (Phase B: --include-prior)')
        from tools.sourcing import build_cross_day_features_v2 as bld
        bld.main(include_prior=True)
        _done(t0)

        t0 = _step(offset + 2, total + offset,
                    'TRAIN dev DRS GBM (Phase B) + gate verdict')
        from tools.sourcing import drs_canonical_gbm_v2 as trn
        result = trn.main(include_prior=True)
        _done(t0)

    _banner(f'PIPELINE COMPLETE -- Phase {phase}   (total {time.time() - overall_t0:.1f}s)')
    print('Read the gate verdict at:')
    print('  research/llm_news_intensity/findings/<today>_phase_'
          f'{phase.lower()}_results.md')
    if isinstance(result, dict) and 'gate_pass' in result:
        print(f'\nGate pass: {result["gate_pass"]}')
        if not result['gate_pass']:
            return 3  # gate failed, non-zero exit so cron / wrapper scripts notice
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='tools.sourcing.llm_news.cli',
                                description='LLM news-intensity pipeline for DRS')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('fetch', help='Scrape Fed + BLS press releases')

    p_score = sub.add_parser('score', help='Score releases via local Llama-3.1-8B GGUF')
    p_score.add_argument('--skip-synthetic', action='store_true',
                          help='Skip the hawkish-vs-dovish anti-cheating synthetic test')

    sub.add_parser('test-synthetic', help='Run only the anti-cheating synthetic test')

    p_run = sub.add_parser('run', help='Run the full pipeline sequentially')
    p_run.add_argument('--phase', default='A', choices=('A', 'a', 'B', 'b'),
                        help='A = news_intensity_today only; B = also news_intensity_prior. Default A.')
    p_run.add_argument('--skip-fetch', action='store_true',
                        help='Skip the fetch step (Phase A only; iterate on scoring without re-scraping)')
    p_run.add_argument('--skip-score', action='store_true',
                        help='Skip the score step (Phase A only; re-build + re-train with existing scores)')
    p_run.add_argument('--force-rescore', action='store_true',
                        help='Phase B only: re-run fetch + score before build + train')

    args = p.parse_args(argv)

    try:
        if args.cmd == 'fetch':
            from tools.sourcing.llm_news import fetch
            fetch.main()
            return 0

        if args.cmd == 'score':
            from tools.sourcing.llm_news import score
            score.main(skip_synthetic=args.skip_synthetic)
            return 0

        if args.cmd == 'test-synthetic':
            from tools.sourcing.llm_news import score
            ok = score.test_synthetic()
            return 0 if ok else 1

        if args.cmd == 'run':
            return cmd_run(args)
    except Exception:
        print('\nPipeline aborted with exception:', file=sys.stderr)
        traceback.print_exc()
        return 1

    print(f'Unknown command: {args.cmd}', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
