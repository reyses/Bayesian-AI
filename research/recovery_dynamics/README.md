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
- First pass (2024_02_20, 1-min): mode **2** trades foregone / **6 min** dead-hold; median 2 / 21 min;
  median depth 12.5pt ($25). 418 recovered, **41 never recovered**. Worst case: 139pt ($278) underwater,
  **298 min** to crawl back, **48 swings** missed.
- **The cost is bimodal:** most wrong trades bounce back cheap (median ~21 min, ~2 swings) — so blanket
  "cut the loser" would just churn the cheap bounces. The damage is the **tail + the never-recovers**.
- This gives a **recovery clock**: median recovery ~21 min → a trade still underwater well past that is
  entering the expensive tail. That is "realize you're wrong *in time*" made quantitative.

## Caveats
- "Foregone" = swings that *existed*, not guaranteed wins (capturing them needs correct direction).
- One day, one threshold set (`MIN_ADVERSE_PTS`, `SWING_PTS`). Sensitivity + multi-day = TODO.
- Next: condition recovery on the *causal environment read* (regime / λ / z / vol) to predict WHICH
  wrong trades enter the tail vs bounce — the deducible pattern.
