# tools/trade_outcome_suite

Consolidated diagnostic suite over every hardened zigzag-leg trade. Answers,
as standing probability tables, every question worked through in the
2026-05-22 session. Pure descriptive statistics — no model fit, no leakage,
no production code touched.

## Run it

```
python tools/trade_outcome_suite/run_all.py            # cache-first (fast)
python tools/trade_outcome_suite/run_all.py --rebuild  # rebuild per-leg data
```

Output: `reports/findings/trade_outcome_table/YYYY-MM-DD_trade_outcome_full_report.md`
(one consolidated report, verdict index up top).

## Layout

| File | Role |
|---|---|
| `excursions.py` | Shared data layer — builds/loads the per-leg excursion dataset (full entry→exit MAE/MFE from the 5s path), cached to `per_leg_excursions_{IS,OOS}.parquet`. Stats + formatting helpers. |
| `questions.py` | One function per question; each returns `(title, verdict, markdown)`. `QUESTIONS` lists them in order. |
| `run_all.py` | Light wrapper — loads data once, runs every question, writes the consolidated report. |

## The 15 questions

1. Distributions of entry-to-close / MAE / MFE
2. Joint MFE × MAE → P(close>0)
3. Continuation — given up +$x, where does it close
4. Conditional — at +$x, P(it continues another step) *(the "$100→$150" question)*
5. Cut-and-bank a winner vs hold
6. Giveback — how much of the peak survives to the close
7. Given an MFE of +$300, where does it close (cumulative)
8. Equity-loss map — P(close<0) by MFE reached
9. Recovery — given down −$d, does it work out
10. Full MAE → close recovery sweep
11. Probability a drawdown gets worse
12. Iterative drawdown chain (n → n+1)
13. Cut a loser vs hold
14. When the MAE happens + how long recoverers take
15. The bimodal split — winners vs losers

Each question is standalone-importable from `questions.py`. To add a new one:
write a `qNN_*(IS, OOS) -> (title, verdict, body)` function and append it to
`QUESTIONS`.
