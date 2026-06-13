# Bayesian-AI Knowledge Base (condensed 2026-06-12)
> Loads on every session start. This is the INDEX + load-bearing facts. Detail lives in `archive/` (pre-condense originals, `*_pre-condense_2026-06-12.md`), `PROJECT_HISTORY.md` (long arc), `AGENT_FEEDBACK_RULES.md`, `USER_PERSONA_AND_PROTOCOL.md`, and `docs/daily/`. Never let this grow back — collapse dated narrative to one status line.

## 0. START HERE — current program
- **ACTIVE (2026-06-11→): λ-completion program.** Master living roadmap = `ROADMAP_LAMBDA_COMPLETION.md` (repo root) — read at session start; stages/gates/prohibitions/log live there. Companion: `docs/Active/NMP_V2_FEATURE_MAP.md`. Daily context: `docs/daily/INDEX.md`.
- **Thesis**: the NMP master equation is `|Z|>Z* ∧ λ<0 → fade; λ>0 → ride`. λ was HARDCODED 0.0 (never computed); the de-facto gate was `variance_ratio = std(10)/std(60) < 1`, which got DROPPED from the V2 schema. The equity-burning drawdown = the integral of the missing λ term. Complete the equation → recover the edge.
- **Honest causal reality**: the causal streaming system LOSES ~$150–185/day OOS as configured. The famous +$454/day baseline is a HINDSIGHT artifact (offline zigzag). DO NOT deploy real money (see §3).
- **Deployment**: RL engine (PW-CRL) NOT deployed; mid-curriculum. Live/SIM flows through `live/engine_v2.py` (zigzag + L5 decider). NOT quantum physics — physics metaphors are historical and purged.

## 1. HARD RULES (process — never skip)
- **JOURNALS** (start AND end of EVERY session): update `docs/daily/YYYY-MM-DD.md` + one-line in `docs/daily/INDEX.md` (newest first) + `docs/reference/RESEARCH_JOURNAL.txt` + finding reports in `reports/findings/`. Session start: read INDEX.md (not full journals). Lost journals = lost days.
- **MEMORY**: update this file when architecture/preferences/patterns change. Append historical entries with dates; never delete them (CLAUDE.md rule).
- **TOOLS INVENTORY**: before building a tool, check `research/TOOLS_INDEX.md` (~106 tools) + `tools/_<cat>/README.md`. Extend, don't rebuild. New tool → add to both indexes.
- **DMAIC** per project (`research/<topic>/project.md`: Define/Measure/Analyze/Improve/Control). **PDCA** per iteration (`cycle_NN.md`; Plan BEFORE code; Check actual-vs-predicted; artifacts in `findings/`).
- **One question at a time** — never batch questions; the user processes sequentially.
- **Blender first, then drill** — run the unrestricted broad-strokes pass first, drill into surprises; do not pre-narrow scope.
- **V2 features only** — 185D layered per-family parquets are the ONLY sanctioned schema. No V1 91D, no hybrid, no adapters. V1 is debt to migrate or delete.
- **OOS-only after training on IS** — a model trained on IS has seen it; only OOS $/day is honest. IS=ATLAS (Databento), OOS=ATLAS_NT8 (what live trades).
- **CLI scripts are false-orphans** — standalone `python x.py` scripts have ZERO Python imports but ARE active. Grep the bare filename across docs/, config/, subprocess before any deletion sweep. `_v2`/`_v1` suffix is a tell.
- **No magic numbers** — every constant is a named `TradingConfig` field or a commented module constant (except π, e, …).
- **User runs training** — NEVER run training/heavy builds via Bash. Say "run with --fresh when ready" and stop. Bash fills context fast.
- **Research before code** — standalone analysis validates an idea before touching production. SUDO = execute it, but present risk (breaks/blast-radius/rollback) + ≥1 alternative FIRST, then proceed.

## 2. METRIC DEFINITIONS (verbatim — one canonical definition)
- **Trade WR (PF-based, NOT count-based)**: `Trade WR = (∑ profit_of_winners / |∑ loss_of_losers|) − 1`. 0 = break-even; +1 = winners 2× loss-size (PF 2); −0.5 = PF 0.5. NEVER use count-based %-winners (hides asymmetric winner/loser sizing).
- **Day WR (count-based)**: `Day WR = winning_days / total_active_days`.
- **$/trade & $/day**: report MODE (histogram; $2 bins for $/trade, $25 for $/day) AND MEAN with 95% bootstrap CI (4,000 resamples, percentile). Always the CI, not just the point estimate.
- **CI on deltas (A vs B)**: bootstrap each population independently (4,000 resamples), take CI of `mean(B) − mean(A)`. If 95% CI includes 0 → NOT significant; say so explicitly.
- **Operational rule**: NEVER report a $/day claim without (1) the 95% CI on the delta, (2) an explicit significance statement, (3) the N (days, trades).
- **Anti-doom-cascade**: don't treat Python-sim point estimates as ground truth (Python-vs-NT8 gap ~$680/day); report deployment risk under MULTIPLE gap assumptions (0/30/60/100%); assume realistic intervention (halt-after-N, drawdown caps, EOD review); don't compound pessimism into a 99.6% doom number. Tool: `tools/risk/blowout_with_intervention.py`.
- **$/day lift framing**: frame against the HONEST FLOOR (post-caveat number), not the headline; translate to $/year; weight tail-risk reduction separately; sizing layers (per-leg × per-hour × per-day) MULTIPLY, not add.
- **CI pseudoreplication / effective-N**: N agents sharing one net make byte-identical trades; the unit of independence = unique trade → entry → day. Day-block bootstrap. (research_A IS-gate, 2026-06-08)

## 3. STATISTICAL TRAPS (one line each + why)
- **THE LOOKAHEAD FAMILY (central scar)**: the +$454/day offline-zigzag baseline is HINDSIGHT — offline zigzag connects true highs/lows after the fact (zero whipsaw); the causal streaming detector eats the whipsaws → causal = −$100 to −$185/day. Replay(+$417) vs stream(−$185) through the SAME harness = the ~$600/day swing IS the lookahead.
- **FLAT / oracle passes are hindsight partitions** — FLAT ATR sweep $/day rises MONOTONICALLY as ATR shrinks (X=1 → $3,480/day OOS, 100% win-days, PF 5.25 — impossible causally); valid only at FIXED X. Any "FLAT" or "hardened-leg" number trades the offline zigzag → not tradeable.
- **The old $740/day baseline was pure lookahead** (higher-TF aggregation with future data; `searchsorted` shifted by period). Honest post-fix IS baseline was −$164/day. ALL pre-2026-04-17 numbers ($740/$620/$613) are contaminated.
- **Quantile-cell selection overfits massively** — 75% of top-|lift| triplet cells flip sign or collapse IS→OOS (survival 25.8%). Trust large-n structural cells over high-lift small-n; OOS-validate before quoting.
- **Per-cell continuous filters overfit** — 70/30 walk-forward INSIDE IS is NOT enough; quantile thresholds break on date-disjoint OOS (FilteredRegimeAwareReversion OOS −$19.85/day).
- **Outlier-day trap** — total-PnL optimizer hid a $49k one-day lottery as "+$713/day"; default objective = MEDIAN day, not total. (VEL_BODY_CHORD killed for this.)
- **SEGMENT FIREWALL** — segment quantities (membership/tier/betas/boundaries/life) leak the future via boundary/fit/existence channels → LABEL-SIDE ONLY. Practicality test = oracle-vs-nowcast recovered fraction, boundary-conditional, lead-time>0, vs an age-only baseline, gated at action level.
- **Hurst is causal but LAGGING** (N_BASE×8 window) with a warmup doc-bug (emits 0.5/partial estimates, no NaN mask). Not a fast signal.

## 4. GRAVEYARD (never retry — measured costs)
- **R-trigger is STRUCTURALLY OPTIMAL for binary exit.** 7+ overlay families ALL lose; only CONTINUOUS SIZING on signed amplitude (B9) wins. The only rewarded lever is the ENTRY filter.
  1. Per-trade fixed stop ≈ −$31/day; "cut at −$100" rejected (76% of −$100 legs recover; B9 AUC 0.465 there).
  2. Intraday session-P&L stop: −$79/day CI[−154,−22] sig LOSS (81% of stopped OOS days recover).
  3. Cut-and-bank a winner: LOSES — hold−cut EV positive at every level (heavy right tail pays for the giveback toll, a fixed ~1R).
  4. Cut/bail a loser: LOSES at every drawdown level — R-trigger recovers ~1R off the low.
  5. Pyramid attenuation (C15): AUC 0.883 yet ALL recall budgets −$29 to −$155/day (88.4% of B9's pyramids pay off).
  6. Vol-adaptive exit thresholds: OOS −$112/day (fat-tailed peaks overshoot mean-based formulas).
  7. Binary cut/preempt/cap (C11/C12/C13) + composite trail-tightening (−$0.29/leg): all fail walk-forward.
- **Lead-in PCA / B1-B6 augmentation** — HURTS the direction classifier at every lookback; B1-B6 preds stacked = −$76/day @ K=5 sig negative (B9 already extracts pivot structure from raw V2).
- **Mid-leg / missed-signal late-join** — structural non-starter (−$1/day, 4 late-joins/51d); hardened legs are a sequential partition (no "busy, missed a parallel leg" population). "Lost signals" = cold-start (~20min warmup) or B7 skips.
- **Day-level bad-day fixes** — hour-of-day skip (not sig), session stop (loss), no day clustering. The B-stack already shaves bad days +$175/day; residual = irreducible chop cost.
- **LLM-as-decider REJECTED** (latency / non-determinism / CI-discipline / live already heavily babysat). LLM-as-FEATURE only (post-hoc; never in the live decision path).
- **FADE_AT_BAND entry rule** died (IS −$17.27/day; the OOS+ was a 6-FLAT_SMOOTH-day fluke). **VEL_BODY_CHORD** permanently killed (lottery artifact).
- **DRS (daily risk sizer)** = DO-NOT-DEPLOY on the clean 51-day sample (anti-predictive on negative days; rank sizing collapsed to +$3/day CI[−27,+38]).
- **5s level is inherently noise** — substrate not predictor; anchor predictions at 15s/1m/5m.
- **Direction-classifier-alone is NOT a live strategy** — AUC 0.864 but every TP/SL grid loses OOS except one non-sig config; entry timing is the unsolved bottleneck. Info ceiling ~83% on V2 entry features.

## 5. VALIDATED IMPROVEMENTS (with CIs — read the §3 lookahead caveat first)
> All B-stack deltas are measured ON TOP of the OFFLINE HARDENED-LEG baseline, which is itself lookahead. On causal NT8 OOS the full L5 stack (B7+B9+B10) is a WASH (−$175 → −$185, −$10/day). These are real RELATIVE deltas; the absolute baseline is hindsight and the causal system loses money. The λ term (§0) is the missing piece.
- **B9** during-trade remaining-amplitude regressor: +$66/day @ K=5 CI[+41,+94] on the 51-day fresh dump (orig +$67 CI[+32,+106] on 31 days). Continuous sizing on signed amplitude. THE validated trade-level path. K=10 (+$31), K=30 (+$15) newly sig.
- **B10** vol-regime sizer: +$69/day CI[+7,+144] sealed single-shot, IS-selected thresholds. Composes multiplicatively with B9. OOS AUC 0.949. Action INVERTED from intuition (zigzag WANTS vol): boost 1.3× on P(high)≥0.5, cap 0.7× on P(low)≥0.7. Models `b10_vol_regime_{high,low}.pkl`.
- **B-stack bad-day shave**: +$175/day CI[+98,+269] sig — the real bad-day mitigation (the non-sig +$42/day headline hid it).
- **NMP threshold recalibration** (verified 2026-06-11): Z_ENTRY=**1.8481**, Z_EXIT=**0.4752** on `L3_1m_z_se_15` (quantile-matched to V1 `|z_21|>2.0 / <0.5`; thresholds DON'T transfer due to 21→15 window drift). vr cross-TF proxy DEAD (Spearman 0.38–0.73). Trigger parity 7.11% vs 7.51%.
- **λ̂/vr derivation layer verified** (`research/nmp_state/`): λ̂ = trailing OLS slope of `log(|z_se|+EPS)`, EPS=0.1 (outlier damping); slope/SE vs np.polyfit 3.9e-16; vr rolling vs brute-force 1.5e-12. λ̂ t-stat ±2.0–2.4 (1m), ±1.6–1.7 (5m). NOT yet gated — Stage-0 forward pass pending.
- **R-trigger reversal exit = adaptive stop** recovering ~1R off the low — the structural reason fixed-dollar overlays lose (§4).

## 6. ARCHITECTURE FACTS (current)
- **V2 schema**: 8 TFs × 25 features + L0 = 201 cols (185D entry vector). N_BASE {5s:9, 15s:12, 1m:15, 5m:9, 15m:12, 1h:12, 4h:18, 1D:5}; N_HURST_MULT=8; OU_BOUNDARY=3.0. `assemble_v2_grid` is name-keyed (anti-scramble). `_last_closed_idx` is load-bearing causality. CUDA-only, no CPU fallback. Tick 0.25 / $0.50 (MNQ).
- **Live = `live/engine_v2.py`** (zigzag default; L5 via `--engine-mode l5`). Thin-wrapper rule: a new live capability = a new `Engine.evaluate(state)→DecisionBatch`, NOT sidecars. L5Decider = zigzag + B7 skip + B9 cut + B10 sizer. Thresholds B7_SKIP 1.90/2.10, B9_CUT +5/+15. Do NOT edit `live/live_engine.py` (legacy blended, broken-by-import).
- **RL engine (PW-CRL)** in `training/rl_engine/` — CNN+LSTM DQN, V-trace, hindsight-regret shadow queue, curriculum EXIT_NMP→ENTRY_NMP→YOLO, 8-agent DOE. Mid-training, NOT deployed. `research_A` variant trialed in parallel. ONNX export = the deployable artifact.
- **Supervised CNN/blended/nightmare stack DELETED 2026-05-28** (RL pivot, commit 4b658e2a). `training/nightmare_blended.py` is a compat shim → frozen snapshot. ~15 importers of deleted `nightmare`/`compute_features`/`ai`/`physics_labels` still dangle (see `AGENTS.ini [known_issues]`).
- **IS/OOS defined by data source**: IS=`DATA/ATLAS` (Databento); OOS=`DATA/ATLAS_NT8` (NT8 dump = what live trades; the OOS-2 gate). B7's skip discrimination is data-coupled to Databento noise — does NOT transfer to NT8's cleaner tape.
- **bars_held = MINUTES** (`//60` in ledger); K horizons are 5s-bar units. NEVER time a during-trade action off `pos.bars_held` (the B9-horizon bug fired actions 12× too late). Parity-check every SIM run vs the offline forward pass before interpreting P&L.
- **FISTA CV bug** fixed 2026-06-08 (`core_v2/math/fista_gpu.py::elasticnet_fista_cv` — scalar 1/L step ignored lam_l2 Lipschitz → NaN → argmin picked alpha_max). Fix = per-column step + nan_to_num. Guard: `research/Regression segments/test_fista_parity.py`. See `reference_fista_gpu_cv_step_bug.md`.
- **Zigzag**: ATR(14)×4 on 1m bars, pivots on 5s closes, min_bars=36. Bounded-lookback (~27min memory); cold-start divergence BOUNDED (0/216 never-resync, median 27min) → low-priority same-day catch-up only.
- **NT8 versioning + deploy gate**: released=no suffix (only v1.0 RELEASED); -RC=built-not-deployed; -RC.REJECTED=artifact. NEVER copy a .cs into the NT8 Strategies/ folder without explicit per-revision user approval; each revision = its own file + class. See `docs/nt8/VERSIONING.md`.

## 7. USER PROFILE & PROTOCOL
- **Moises** — discretionary MNQ futures trader (manual level / VP zone-map background); this is a REAL-MONEY system. Wants Claude as a **critical collaborator**, not a yes-man: challenge ideas, find the flaw/failure mode, commit to disagreements, propose alternatives, say no when warranted. Data beats intuition.
- **SUDO** = accept and execute the instruction, but ALWAYS present a risk assessment (what breaks, blast radius, rollback) + ≥1 alternative FIRST, then proceed.
- **Thinks topic-at-a-time** — sequential questions only; configurable defaults over preemptive engineering ("we won't know if we don't try"). Format: mode > mean; honest-floor $-framing; always show CIs.
- **Metaphor scope (2026-06-12)**: "the metaphors are for communication, not for the code" — use his physics vocabulary (Roche limit, nightmare field, three-body) in chat/design discussion; code, labels, chart titles stay statistical-only. Two layers, no contradiction.
- **Plain-language rule (2026-06-12)**: AVOID the trading-jargon words "fade" and "reversal" — user finds them unclear. Use price-vs-mean language: fade → "snap-back bet" / "bet price returns to the mean"; ride → "run bet" / "bet price keeps running"; reversal → "the turn" / "price snaps back". (Code/labels may still use precise terms with a definition; this is for chat/explanation.)
- **Hardware**: Ryzen 5 5600X, 16GB RAM, RTX 3060 12GB VRAM — VRAM is the binding constraint (hence FPS OOM hardening). Local-LLM capacity ≈ 8B Q4 with CUDA offload.
- Detail: `USER_PERSONA_AND_PROTOCOL.md` (cognitive style, collaboration protocol, headroom/nesting framework, VP zone-map trading manual, schedule, specs). Full feedback corpus: `AGENT_FEEDBACK_RULES.md`.

## 8. POINTERS
- `ROADMAP_LAMBDA_COMPLETION.md` (repo root) — ACTIVE master roadmap (stages/gates/graveyard/append-only log).
- `docs/Active/NMP_V2_FEATURE_MAP.md` — NMP→V2 term-by-term map + standing traps #1–8.
- `PROJECT_HISTORY.md` — the long arc (chronological era timeline at top; detailed legacy sources below). `archive/` — all pre-condense originals (`*_pre-condense_2026-06-12.md`).
- `AGENT_FEEDBACK_RULES.md` / `USER_PERSONA_AND_PROTOCOL.md` — full feedback/persona corpus (referenced, not loaded on start; persona file deduplicated + mojibake-repaired 2026-06-12, one clean copy of each of its 7 topics).
- `reference_fista_gpu_cv_step_bug.md` — the FISTA CV gotcha. `ce_methodology.md` — collaboration-engagement method. `ROADMAP.md` — obsolete pointer (→ active roadmap).
- `AGENTS.ini` — maintained file-layout index + [known_issues] + [entry_points]. `rl_whitepaper.md` — RL architecture. `docs/daily/INDEX.md` + `docs/daily/TIMELINE.md` — chronological navigation.
- Claude's private auto-memory at `~/.claude/projects/<hash>/memory/` syncs INTO docs/memory/ via a commit hook — **edits to MEMORY.md / PROJECT_HISTORY.md must be made in BOTH copies or the hook reverts them at commit time.**
