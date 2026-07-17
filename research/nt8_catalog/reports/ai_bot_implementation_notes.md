# AI-bot implementation notes from the exit corpus (feeds: interim-NT8 decision + Mamba deployment)
2026-07-17 (night). Sources: raw_articles_exits/ — WPI thesis, Reddit bot threads
(rescued), GitHub ayb README, QuantInsti algo pieces. Honest scope note: the
corpus's implementation content is THIN (platform surveys + war stories); the
strongest lessons are operator experience, not architecture.

## What the sources actually say (with pointers)
1. **Regime non-stationarity is the #1 operator complaint** (reddit 1jbgf9z:
   "nothing seems to be consistent over different market regimes"; "figuring out
   what days were going to be choppy... accepting that as built-in losses").
   Maps to: our shelf-life result (median 37-41 wks but 2025-fit windows skew
   shorter) + the ER chop gate + Moises' 12-bots-per-condition parallel
   (one operator literally runs "12 bots each for a specific market condition,
   deploy by regime" — structurally our combiner + regime state).
2. **The squeeze-as-chop-filter is known AND known-laggy** (depth-3 reply:
   "always squeezing too late and for too long"). Matches our SQZ-04 experience;
   don't re-litigate.
3. **Slippage/fills discipline** (WPI :914): "slippage... especially for markets
   that fluctuate quickly" — our close-based sims UNDER-count adverse fills;
   the NT8-dump reality check (interim strategy's first job) is the answer.
4. **Platform survey (WPI ch.2)**: NT8/TradeStation for native automation;
   Zipline/QuantConnect for python-API trading. Nothing we don't have.
5. **GitHub ayb bot**: inside-bar breakout + ATR target/trail — a typical retail
   bot shape; its exit vocabulary is already covered by EXIT-01/08.
6. **Live-runner hygiene** (reddit threads): NT8 break-even/trailing stop
   glitches in strategy code (1r0utmk) — argues for keeping NT8-side logic
   MINIMAL (bridge architecture A) rather than porting complex logic into
   NinjaScript.

## Implications
- **Interim-NT8 A/B**: every operator lesson favors **A (python sensor → thin
  NT8 executor)** — minimal NinjaScript surface (their bug reports are exactly
  about complex in-strategy stop logic), canonical sensor stays verified, and
  regime/chop state stays in python where it already exists.
- **Mamba deployment**: nothing in the corpus contradicts the rl_whitepaper
  ONNX path; the operator regime-pain reinforces monthly retunes + the regime
  state in the observation.
- **No new dossier candidates** — implementation corpus adds process wisdom,
  not signals.
