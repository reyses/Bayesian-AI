# HANDOVER → Gemini — Orange-line smoother + big-move capture (2026-06-16)

Claude built the candidate and the EDA; you continue the run. Read this fully before coding.

## HARD RULES (prior Gemini work violated these — do NOT repeat)
1. **MNQ = $2/point.** NOT $20. Every earlier script (`nmp_triple_timeline*.py`, `nmp_geometric_inflection.py`) hardcoded `*20` / `USD_PER_PT=20.0` = full-size NQ = a **10× error** on PnL, cost, AND the −$100 stop. Use **2.0** everywhere.
2. **Causal only, no hindsight.** Kalman = forward **FILTER**, never the RTS smoother. Any exit/entry uses only past+present bars. No fitting on post-entry data.
3. **Metrics:** PF-based Trade WR = `(Σwin/|Σloss|)−1` (NOT count-based %). Report **$/day with 95% day-block CI** (4000 resamples). Split **IS=2024 / OOS=2025-26** and report each.
4. **No magic numbers** — every constant a named, commented config field.
5. **Write results to files** (`reports/findings/*.md`), update `docs/daily/2026-06-16.md`, and **default to GIFs, not static plots** — the user finds animations more illustrative. Use **`research/plot_option2_trade_gif.py` as the template**: unfolding price frame-by-frame with the curves (Kalman position/velocity/accel, blue, orange) animating, entry/exit markers, PnL in the title. Every candidate/strategy result should ship a GIF of a representative trade or day.

## WHERE WE ARE (don't re-derive)
- **Strategy thesis:** direction = the smoothed price-curve slope (multi-timescale); enter the big swings, **exit on a COARSE signal, not on every curve turn.**
- **"PF 2.0" was a mirage** — it came from `nmp_triple_timeline_forward.py`'s *broken* 3-second velocity gate (1.9 trades/day). The **correct-math** 3-sign sim (`nmp_triple_timeline_v2.py`, real poly-slopes, $2/pt) churns **427 trades/day → −$1008/day**.
- **The opportunity (MFE diagnostic):** avg captured **−$0.36** vs avg **MFE +$8.99** (winners keep only **58%** of peak). The big moves ARE there; the fast orange-flip exit clips them. **Capturing the move is the whole game.**
- **Orange EDA (`orange_line_eda.py`):** the 7.5m cubic turns **~378×/day**; median swing **3.3 pts / 3 min**, **big swings (top 25%) ~14 pts**. Curvature flips a median **~20 s** before the slope turn (causal, but short).
- **Why the cubic is rough:** read at its **endpoint** = highest-variance point of the fit. That's why we're trying the Kalman.

## THE CANDIDATE (`research/orange_kalman_candidate.py`)
Kalman **constant-acceleration** filter → outputs position (orange-line analog), velocity, acceleration in ONE adaptive causal model. It unifies the orange curve + the pink kinematics.
- **CURRENT TUNING IS WRONG:** `Q_JERK=1e-4, R_MEAS=1.0` → **5534 velocity turns vs cubic 400** (too responsive). 
- **TASK 1 — tune q/r:** lower `Q_JERK` (try `1e-6 → 1e-8`) and/or raise `R_MEAS` (try `2 → 9`) until Kalman **velocity turns < cubic's 400** while it **still tracks the big run without lagging through the peak** (clipping the top). Verify with a **GIF** (template: `plot_option2_trade_gif.py`) showing Kalman vs cubic animating over a day, so the lag/clip difference is visible. The q/r ratio is the smooth-vs-lag dial.

## THE RUN TO CONTINUE (in order)
1. **Tune the Kalman** (Task 1 above) → pick q,r; document the turn-count and a plot.
2. **Exit-comparison EDA** on the big orange swings — compare CAUSAL exits head-to-head: (a) orange/velocity turn, (b) acceleration-flip (leading), (c) **blue 24m-quad turn** (coarse), (d) **trailing give-back** (exit when price retraces X pts/% from peak). Report **captured / full-swing %** and pts left on the table per exit. This answers "right exit, no hindsight."
3. **Build the strategy:** Kalman velocity (smoothed) for direction + entry on big-swing setups; blue (24m) as confirm/cleaner; **exit = the winner from step 2**. MNQ $2/pt, explicit costs (~$2/trade RT), MFE/MAE logged.
4. **Full-dataset run** (604 days, `DATA/ATLAS/1s/`), IS/OOS split, PF-WR + day-block CI. Write `reports/findings/` + journal + a trade GIF.

## FILE INVENTORY
- `research/orange_kalman_candidate.py` — the Kalman CA candidate (TUNE IT).
- `research/orange_line_eda.py` — orange value/slope/curvature EDA + swing stats.
- `research/nmp_triple_timeline_v2.py` — correct-math 3-sign sim, $2/pt, IS/OOS, MFE diag (the −$1008/day baseline).
- `research/plot_option2_trade_gif.py` — the user's intent GIF (24m quad + 7.5m cubic + 15m kinematic projection); style template for visuals.
- Reports: `reports/findings/orange_line_eda*.md/png`, `nmp_triple_timeline_v2.md`, `orange_kalman_vs_cubic_*.png`.

## DATA / SCHEMA NOTES
- `DATA/ATLAS/1s/` is clean (roll-fixed, session-day boundary, built 06-15). 604 days, 2024-01 → 2026-03. Short half-days have <1440 1s bars — skip (the 24m window can't fill).
- `core_v2/features.py`: L5 bake is **restored** (N_FEATURES=393, 37/TF) per user — leave as-is.
- Poly-slope math (correct): `get_poly_slope_weights` (Vandermonde pinv) in `nmp_triple_timeline.py` — slope of an OLS fit at the endpoint. This is the right approach; the Kalman is the smoother upgrade.

## OPEN QUESTION for the user (ask before the full strategy run)
"Big swing" must be selected **causally** (amplitude is hindsight). Decide with the user how to flag a developing big move in real time (e.g., velocity magnitude threshold, acceleration confirmation, or blue-macro alignment) — this is the unsolved entry-selection piece.
