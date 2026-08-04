# The program: IS / IS NOT
_Assembled 2026-08-04 from measured results only. Every row cites its number._

## IS

| It IS | Evidence |
|---|---|
| **A risk machine.** Protection converts loss DISTRIBUTION, not expectancy. | Stops 51.5% → 10.0% at 3σ, std −73%; region-armed P(loss) 0.59%/0.31% |
| **A geometry calculator.** Given price between two levels, it prices which is hit first. | Driftless-null excess −0.0009, day-clustered CI [−0.0078, +0.0063] over 18,601 chop escapes |
| **An event detector.** Named tape states are identifiable causally, at scale, early. | 273k labelled events, 0 lookahead under truncation replay; onset AUC 0.643 / 0.759 / 0.820 (causal, fit-2024/score-2025H1) |
| **A latency solution.** The 800ms round-trip / 200ms decision split is solved in principle. | Inference 1.17ms batch-1; resting orders fill at level, reactive fills modelled at next-second open |
| **A corpus generator.** Every freeze is a labelled human decision with its reasoning. | 20-trade owner session + engine decision log; the north star's actual input |
| **A falsification engine.** Its main product is killing things cheaply. | 8 independent confirmations of the ~0.57 wall; 3 self-caught bugs and 4 audit-caught in one night; Mamba killed for 45 min of GPU |

## IS NOT

| It IS NOT | Evidence |
|---|---|
| **A direction predictor.** | Symmetric ±10pt races: stall 0.4885–0.5085, leg_descent 0.4887–0.5047, chop escape excess ≈ 0. Eight independent tests, none broke it |
| **An alpha source via the tables.** | Every large cell spread is barrier distance; conditioning that looked like a 0.62-AUC edge was the distance ratio in disguise |
| **Profitable from mechanical entries.** | 112 days / 1,344 trades: −8.81 pt/day, CI [−13.13, −4.73]. Gross +0.156/trade vs 0.89 friction |
| **A sequence-model story.** | Onset Mamba KILLED 3/3 heads vs 22 hand-made features (0.585/0.552/0.671 vs 0.643/0.759/0.820) |
| **Validated end-to-end.** | My backtest and the live engine disagree on the same day (−23.06 vs +11.57). Open, unreconciled |
| **Anything on N=1.** | The blind day returned +11.57 and the same rules lose significantly across 112 days |

## The one sentence

**We can see WHERE we are and WHAT it costs us to be wrong; we cannot see WHICH WAY it goes — and every attempt to buy direction has been repaid with a null.**

## What that implies for the next move

1. Stop paying for direction. Eight tests is enough.
2. Spend on execution, which is where measured wins actually came from (the
   ladder saved 12pt on one live trade; entry-touch would have saved 10.25pt
   on another).
3. The open question worth money is **entry SELECTION by the owner** — his
   1-of-101 choice, still unmeasured, quantified only as a 2-3pp traverse-rate
   bar. That is a corpus question, not a model question.
4. Reconcile the backtest with the engine before any policy number is quoted.
