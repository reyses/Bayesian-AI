# FABLE-5 Peer Review — AG "Execution Logic Extraction" (0.8748 OOS AUC)

**Verdict: the mechanics are mostly clean; the headline conclusion is circular.**
The classifier did not extract a trading edge. It rediscovered the label
generator's own selection rule, and the ~0.87 AUC reproduces a ceiling this
repo already measured and already showed loses money when acted on.

## 1. The fatal premise error: label provenance

The packet frames the labels as "a highly profitable black-box AI." They are
not. They are OUR auto-labeler (`research/ai_auto_labeler/pipeline/
ai_labeler_v2.py`, journal 2026-06-30): cubic-leg segmentation, then —
line 158, verbatim — **"ENTRY: flat-zone best bar at the leg's START turn
(0-MAE)"**, snapped to the literal 1-second price extreme (`best_bar_1s`:
lowest low for LONG / highest high for SHORT). Exit = mirror best-bar at the
cubic's actual direction change. These are HINDSIGHT-optimal placements.

## 2. Consequence: the two headline findings are tautologies

- **"AI enters at severe z_high/z_low extension" (gap 0.268).** The labeler
  snaps entries to the local price extreme. A bar at the local extreme is,
  by definition, maximally stretched against trailing structure (z_high/low,
  band_pos, z_se). The classifier learned "is this bar a local extreme" —
  i.e., the label-generation rule, not a market edge.
- **"Near-zero MAE proves the AI enters perfectly" (23t vs 77t).** The
  labeler's own comment says 0-MAE *by construction* (argmin/argmax snap;
  the 23t residual is 1m-bar measurement vs 1s snapping). Comparing an
  argmin-selected entry's MAE to a random bar's measures the snap, not skill.
- Same for "double realized velocity": the labeler only labels legs that ran
  (uncapped forward walk to the cubic turn). Survivorship by construction.

## 3. The AUC is a known ceiling, already shown untradeable

MEMORY §4 (graveyard): direction-classifier AUC **0.864** on these feature
families, "but every TP/SL grid loses OOS… entry timing is the unsolved
bottleneck. Info ceiling ~83% on V2 entry features." AG's 0.84–0.87 with
z_high/z_se/velocity is the same ceiling re-derived on the same label family.
The step that has killed this result every prior time — converting scores to
CAUSAL entries with a CAUSAL exit and measuring $/day — was not run. The
labels' profit lives in the hindsight EXIT (mirror best-bar); flagging the
entry bar does not transfer that P&L to a live policy. This week's own
confirm-then-ride backtest (−$60..−200/day OOS) is the cautionary precedent.

## 4. What survives review (genuinely useful)

- The matched-null (same day, same hour) is a good control for diurnal/regime
  base rates; the feature-side causality (`entry_ts − 1` slicing) looks right,
  and 2024→2025 transfer is a real replication.
- The stable WHERE: labels live at stretch extremes + expanded SE + deep
  contra-momentum pullback. This confirms (again, OOS-stably) the NMP fade
  geometry `|Z|>Z*` — the system's existing thesis — and gives a clean ranked
  feature list for the RL state space. That is real, keepable output.
- The candle-shape null (wicks don't matter) independently replicates our
  2026-07-08 pre-trend microstructure null. Two methods, same answer.

## 5. Secondary methodological notes

- **Effective N overstated**: trades cluster in episodes/days; 21k/20k samples
  are not independent. Day-block bootstrap would widen every CI (see the
  research_A pseudoreplication rule, MEMORY §2).
- **Deployment base rate ignored**: ~37 labeled bars vs ~1,380 bars/day. At
  0.87 AUC the precision at any actionable threshold, and trades/day at that
  threshold, are unreported — the numbers that decide usability.
- **Null hygiene**: null bars may fall inside stretched episodes adjacent to
  entries (deflates AUC, doesn't inflate — noted for fairness).

## 6. Required next test (the one that decides)

Score→entry→**causal** exit (R-trigger or trail, NOT the mirror best-bar) →
$/day on 2025 with 95% CI, per the house operational rule. Prediction from
project history: ~flat to negative. If it clears costs with CI excluding 0,
I retract §3 and this becomes the biggest finding of the program. Until then
the honest summary is: **AG built an accurate detector of where the hindsight
labeler places entries — not evidence of a live edge.**
