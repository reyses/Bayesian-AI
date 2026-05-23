"""LLM news-intensity scoring module for DRS cross-day features.

Self-contained module. Nothing else in the repo imports from here during
the Phase A/B research cycles -- production paths are byte-identical until
promotion (see research/llm_news_intensity/project.md).

CLI entry:
    python -m tools.sourcing.llm_news.cli {fetch | score | build | train}
"""
__version__ = '0.1.0'
