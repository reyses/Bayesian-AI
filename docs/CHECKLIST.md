# Implementation Checklist
> Updated: 2026-03-21
> Priority: top to bottom. Check off as completed.

## READY TO IMPLEMENT (code exists or trivial)

- [x] Rolling window PID (cumsum -> rolling mean 200) — IS $209K validated
- [x] Cat brain (8-channel rolling delta regime classifier) — `--cat` flag
- [x] Heartbeat log (60s CSV) — deployed
- [x] ATLAS warmup (per-TF pre-computed states) — deployed
- [x] Dashboard daily redraw (was per-bar, froze GUI) — deployed
- [x] 1m worker loading (was inactive) — deployed
- [x] 1s/5s daily filtering (was loading full month) — deployed
- [x] Order lifecycle strict (awaiting_fill) — deployed
- [x] Volume-relative giveback + adaptive threshold — hook connected
- [x] Proportional sensor gate (z-score from cat window) — committed
- [x] Dashboard PIN button (always-on-top toggle) — committed
- [x] Skip log timestamp fix (??:??:?? -> bar timestamp) — committed
- [ ] **Run proportional gate + compare 3 baselines**
- [ ] Fake peak -> exit override (giveback conversation with accountability)
- [x] Capture rate 6 buckets on dashboard — already wired (session_tracker + gui_bridge + dashboard)
- [ ] Peak-implied direction (skip cascade for peak trades, use peak + 1m)
- [ ] Sensor gate: if no cat brain, use lenient fallback (-50.0)

## NEEDS RESEARCH FIRST

- [ ] Resonance cascade — multi-TF peak agreement validation
  - Build `tools/resonance_cascade_research.py`
  - Hypothesis: 5/5 TF pairs agree = crash/rally (90%+ accuracy?)
  - Each TF pair: child detects peak, parent validates real/fake
  - "Trend" = decay of peaks in one direction over time
- [ ] Macro peak detection — run peak detector on 1h/4h/1D data
  - Same code as 15s peak detection, just different scale
  - Validates: does 1h peak predict next 3-8 hours direction?

## NEEDS BUILD (estimated hours)

- [ ] Counterfactual engine (goat brain) — ~8h
  - Phantom trades for every skip and every trade
  - Alternative thresholds evaluated in parallel
  - Spec: `docs/specs/COUNTERFACTUAL_ENGINE.md`
- [ ] Crow brain (k-NN seed matching) — 8-12h
  - Enrich 31K auto seeds with rolling delta features
  - FAISS index for fast nearest-neighbor
  - Replace lizard brain counter with context-aware lookup
- [ ] Monkey brain (CNN) — 12-18h
  - Three-head model: P(reversal), direction, expected MFE
  - Train on enriched seeds
  - Spec: `docs/specs/CNN_PEAK_CLASSIFIER.md`
- [ ] 1s peak entry — architecture change
  - Peak detection at 1s with full+partial 1m confirmation
  - Reduces worst-case entry lag from 29.9s to <1s
- [ ] NT8 data pipeline — auto-save all TFs to ATLAS on connect

## LIVE READINESS GATE

- [ ] Proportional gate validated (IS + OOS comparable to no-gate)
- [ ] Live parity confirmed (F_momentum matches OOS after warmup)
- [ ] NT8 Market Replay test (replayed day matches OOS)
- [ ] 1 week clean live sim (no crashes, no orphan orders)
- [ ] Feb 9 protection validated (daily DD stop OR resonance cascade)

## BRAIN EVOLUTION

```
Lizard:  counter (DEAD — replaced)
Cat:     rolling delta regime classifier (DEPLOYED — --cat flag)
Crow:    k-NN seed matching (PLANNED — 8-12h)
Monkey:  CNN three-head (PLANNED — 12-18h)
Goat:    counterfactual engine (SPECCED — 8h)
```

## BASELINES (for comparison)

| Run | Gate | IS Trades | IS PnL | IS WR | OOS Trades | OOS PnL | OOS WR |
|-----|------|----------|--------|-------|-----------|---------|--------|
| Lizard (cumsum PID) | old | 2,010 | $1,943 | 56.8% | 4,810 | $22,812 | 66.3% |
| Cat (no 1m gate) | none | 34,446 | $209,734 | 70.4% | 4,636 | $21,631 | 67.1% |
| Cat (absolute gate) | strict | 21,779 | $114,328 | 66.6% | 3,957 | $16,409 | 65.4% |
| Cat (proportional) | z-score | — | — | — | — | — | — |
