# research/recovery_dynamics

**Question (Moises):** what does holding a *wrong* trade back to breakeven actually cost? Not
"does it recover" (trend-conditioned illusion) but **how long, how much, and how many other
trades you forwent** while waiting — the opportunity cost of not realizing you were wrong in time.

## Layout
- `tools/opportunity_cost.py` — THE exercise. Samples long/short entries on a real day, keeps the
  ones that go underwater (wrong trades), holds each to breakeven, and measures dead-hold time,
  drawdown depth, and tradeable swings foregone in that window. Mode-first histograms + worked example.
- `reports/opportunity_cost.md` — output.

## Run
```bash
python research/recovery_dynamics/tools/opportunity_cost.py --day 2024_02_20
```

## Status / findings
**Multi-year (536 days, 2024+2025, ~21k random seeded wrong trades)** — `recovery_2024_2025.md`:
- **Return-to-zero = one full oscillation** of price around the entry level (Moises). The period
  distribution is heavily right-skewed: **MODE ~5 min** (most wrong trades bounce fast) with a long
  tail to hours. ~2 trades foregone (mode & median) per wrong trade, depth ~12–14pt ($25–28).
- **The clock is NOT a fixed constant** — it shifts by regime/year: median oscillation **2024 = 27 min
  vs 2025 = 15 min** (2025 reverts faster but kills more: never-recovered **11% vs 7%**). The MODE is
  stable (~5 min); the median/tail are not. So a fixed "cut at N min" rule is unsafe — the cut clock
  must be **adaptive to the current period**, which is itself a readable environment variable.
- **The genuinely-wrong tail = 7–11% never recover same-day** — the population where cutting actually
  matters. The other ~90% bounce, mostly fast; blanket cutting churns those.

(First single-day pass — `opportunity_cost.md` reports — agreed: mode 2 foregone / 6 min, 41 never.)

## Caveats
- "Foregone" = swings that *existed*, not guaranteed wins (capturing them needs correct direction).
- One day, one threshold set (`MIN_ADVERSE_PTS`, `SWING_PTS`). Sensitivity + multi-day = TODO.
- Next: condition recovery on the *causal environment read* (regime / λ / z / vol) to predict WHICH
  wrong trades enter the tail vs bounce — the deducible pattern.
