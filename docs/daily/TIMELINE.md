# Project Timeline — story arcs
> The "go back to" navigator. Each arc = a date span + a short narrative + the PIVOTAL days (date — why it matters — key number). For one line per session see `INDEX.md`; for full detail see the day file `docs/daily/YYYY-MM-DD.md`. Condensed 2026-06-12.

## Arc 0 — Foundations: waveform & physics era (2026-02-28 → 2026-03-21)
Shape-primitive waveform analysis and a physics-metaphor engine. Direction was learnable on segments but the whole edge later proved to rest on a peak-detection lookahead bug. NOT quantum physics — the metaphors were purged.
- **2026-02-28** — first journal: 20-shape seed-primitive library; GBC direction 54.1%→70.6% on 193D fractal features (1h/4h hurst dominate).
- **2026-03-22** — first-principles C&E matrix; `variance_ratio` replaces Hurst+ADX; "Nightmare Protocol = this framework in physics metaphors."
- **2026-03-21** — LOOKAHEAD #0: a 1-bar peak-detection bug since day one inflated all prior numbers. First of many reckonings.

## Arc 1 — Blended-CNN era + lookahead reckoning #1 (2026-03-22 → 2026-04-25)
The supervised stack: 79D features → NMP tiers → CNN flip/hold/risk + killshot + blended. Phantom-spike "edge" on NT8 data died on clean Databento. The honest baseline collapsed once lookahead was removed.
- **2026-04-03** — clean Databento data turns +edge into −$2,427: phantom spikes were fake. 79D vector designed.
- **2026-04-07** — CNN flip 70.6% → blended $367 IS / $397 OOS; entry-only CNN beats path CNN.
- **2026-04-09** — $740/day OOS headline (74 days) — later proven LOOKAHEAD.
- **2026-04-17** — LOOKAHEAD #1: `build_dataset` searchsorted picked future 5s data. Fixed → honest baseline −$164/day; every tier at the noise floor (~50% WR).
- **2026-04-18** — tier-fixing via the three-question method: $76 → $149.86/day (2×).

## Arc 2 — NT8 v1.0 live + zigzag/regime discovery (2026-04-27 → 2026-05-04)
Live deployment on NinjaTrader; the zigzag strategy; chop-vs-trend regime structure; the V1→V2 schema migration begins. Versioning policy (`-RC` suffix) adopted.
- **2026-04-25** — Regime+Envelope filter +$79.88/day OOS (3.3× IS gain = not overfit); honest bar-by-bar sim v1.0 −$223 → v1.5-RC −$80/day.
- **2026-04-27** — chop-edge: 95-day NT8 ledger splits cleanly at 2026-02-26 (+$89 → −$95/day); hour-of-day = a $15k swing.
- **2026-05-04** — V1→V2 audit: V2 lookahead-clean (28/28), but z_se/p_at_center drift means the −$164 baseline WILL shift on migration.

## Arc 3 — V2 feature & regime EDA (2026-05-01 → 2026-05-09)
185D layered V2 features; descriptive EDA stacks; the regime-conditional framework; the 3-body envelope; the segmentation substrate. Quantile-cell overfitting quantified.
- **2026-05-01** — MA-alignment WINNER: 7-of-8 TF vwap alignment → 70.5% direction on 20% of bars (+17.6% lift), deterministic.
- **2026-05-03** — feature×feature (Track D) survives OOS 95-100% (vs 25.8% for feature×price) → use as a regime DETECTOR; quantile-cell overfit rule born.
- **2026-05-09** — macro-event problem: reversion breaks in 2-3hr impulses (honest OOS $172 → −$40); 5-level segmentation substrate (203k notes).

## Arc 4 — The regret-oracle arc (2026-05-14 → 2026-05-17)
The daisy-chain oracle defines the opportunity ceiling; direction is the lever; the during-trade B-stack paradigm emerges; B9/B10 get built. The information ceiling on V2 entry features is ~83% direction.
- **2026-05-14** — daisy-chain oracle: 7,925 IS trades, $4,492/day sequential ceiling; 50% of joint entry×exit cells are 100% direction-skewed.
- **2026-05-16** — binary direction classifier AUC 0.864 (the L4 selector); lead-in PCA REJECTED at every lookback; info ceiling ~83% confirmed.
- **2026-05-17** — L5 during-trade paradigm: B9 remaining-amplitude (continuous sizing) OOS +$67/day CI[+32,+106]; binary cut KILLED. R-trigger structurally optimal.

## Arc 5 — L5 stack build + SIM parity + engine fixes (2026-05-18 → 2026-05-21)
The L5Decider thin-wrapper, forward-pass validation, first live SIM, and the engine bugs that the parity discipline caught. DRS investigated and shelved.
- **2026-05-18** — DRS = DO NOT DEPLOY on canonical Path A (OOS Pearson crosses zero; anti-predictive on negative days). C12/C13 fail → R-trigger optimal.
- **2026-05-19** — engine validated decision-identical to forward_pass_full_stack (20/20 exact prices); full L5 stack wired behind `--engine-mode l5`. Phase-1 delta +$42/day NOT SIG.
- **2026-05-20** — first SIM −$379 was an ENGINE BUG: B9 fired off `bars_held` (MINUTES) vs K=5 (5s-bar units), 12× too late. Fixed → validated day ≈−$59.
- **2026-05-21** — mid-leg late-join REJECTED (structural); zigzag cold-start divergence BOUNDED (~27min, 0/216 never-resync).

## Arc 6 — THE LOOKAHEAD RECKONING #2 (2026-05-21 → 2026-05-22)
The central scar. The +$454/day baseline this whole project was quoted against is a HINDSIGHT artifact — the offline zigzag connects true highs/lows after the fact. The causal system loses money.
- **2026-05-21** — FLAT ATR sweep is oracle-contaminated: OOS $/day rises monotonically as ATR shrinks (X=1 = $3,480/day, PF 5.25 — impossible causally).
- **2026-05-22** — replay (+$417) vs causal stream (−$185) through the SAME harness: the ~$600/day swing IS the lookahead. Causal reality = −$100 to −$185/day. **DO NOT deploy real money.**

## Arc 7 — Diagnostic-suites era (2026-05-21 → 2026-05-22)
With the baseline exposed, the work turned diagnostic: probability tables over the legs, not new overlays. The conclusion was negative and durable.
- **2026-05-21** — conditional probability table (4 entries): chop begets chop; leg-death hazard is HUMP-shaped, not monotone.
- **2026-05-22** — trade-outcome suite (15 tables): R-trigger is an adaptive stop — every fixed-dollar cut/lock/bail overlay LOSES; the only lever is the entry filter.
- **2026-05-21** — OOS bad-days: NO day/session-level fix; the B-stack already shaves bad days +$175/day CI[+98,+269].

## Arc 8 — RL pivot + PW-CRL (2026-05-28 → 2026-06-08)
The supervised stack is deleted and replaced by a reinforcement-learning engine. Mid-training, not deployed. Infrastructure and a regime-segmentation research corpus follow.
- **2026-05-28** — RL pivot committed (4b658e2a, 164 files): CNN/blended/nightmare DELETED → PW-CRL RL engine. Hardcoded Telegram token caught (ROTATE).
- **2026-06-08** — regime-segmentation audit; FISTA CV step bug found+fixed (Jaccard 0→0.981) + parity guard; CI effective-N pseudoreplication rule.
- **2026-06-10** — VM telemetry/integrity tooling; Telegram completion alerts.

## Arc 9 — λ-completion program (2026-06-11 →, ACTIVE)
The thesis that ties it all together: the equity-burning drawdown is the integral of the MISSING λ term. NMP's `λ` was hardcoded 0.0 and its real `vr` gate was dropped from V2. Complete the equation.
- **2026-06-11** — λ-completion program founded; `ROADMAP_LAMBDA_COMPLETION.md` + `NMP_V2_FEATURE_MAP.md` written.
- **2026-06-11** — causal λ̂/vr derivation layer built+verified (`research/nmp_state/`); Z* recalibrated to 1.8481/0.4752 on z_15; cross-TF vr proxy dead.
- **Next** — Stage-0 NMP-V2 forward pass (Jules build → review → IS/OOS); KT1 oracle / KT2 nowcast kill-tests on the segment corpus.

## Where decisions live
- **Metric definitions, hard rules, NT8 deploy gate** → `CLAUDE.md` (project root) — commands, overrides everything.
- **Active program (stages / gates / graveyard / log)** → `ROADMAP_LAMBDA_COMPLETION.md` (repo root).
- **Durable knowledge (what's true now)** → `docs/memory/MEMORY.md` (the condensed knowledge base).
- **The long project arc (era detail)** → `docs/memory/PROJECT_HISTORY.md`.
- **What happened on a given day** → `docs/daily/INDEX.md` → the day file.
- **File layout / known issues / entry points** → `AGENTS.ini`. **RL architecture** → `rl_whitepaper.md`.
