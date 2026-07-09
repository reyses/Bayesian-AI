# Top 3 Indicators to Help You Stop Overtrading
**Date:** June 29, 2026
**Source:** https://ninjatrader.com/futures/blogs/indicators-to-stop-overtrading/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Top 3 Indicators to Help You Stop Overtrading

# Top 3 Indicators to Help You Stop Overtrading

By
NinjaTrader Team

June 29, 2026

If you keep taking trades you can’t quite justify, the problem usually isn’t your strategy; it’s your discipline. Overtrading occurs when a futures trader places trades that don’t meet their plan’s criteria—driven by the need to be active rather than by a real edge. The three indicators most useful for stopping it are average true range (ATR), the relative strength index (RSI), and volume-weighted average price (VWAP).

Each one targets a different impulse: ATR sizes every trade to real volatility, RSI keeps you from chasing extended moves, and VWAP filters out low-conviction entries before you click buy or sell. NinjaTrader includes all three as built-in indicators with customizable parameters, so you can fold them into one pre-trade routine without bolting on third-party tools.

Let’s dive into how each one works, and how to combine them into a checklist that keeps you honest.

| Indicator | What it measures | The overtrading behavior it stops |
| --- | --- | --- |
| Average true range (ATR) | Recent market volatility over a set lookback (14-period default) | Impulse-sized positions and stops that ignore current conditions |
| Relative strength index (RSI) | Momentum on a scale of 0 to 100, with 70/30 overbought-oversold thresholds | Chasing extended moves and fear-of-missing-out entries |
| Volume-weighted average price (VWAP) | The volume-weighted average price institutions benchmark against | Low-conviction entries that fade real opportunity |

## What overtrading actually looks like (and why indicators are part of the fix)

Overtrading isn’t one habit; it’s a cluster of them, and naming yours is the first step toward fixing it. Indicators can help with overtrading because they turn a vague feeling of “this looks good” into a concrete, repeatable read you either pass or fail.

### Define discretionary, technical, and “shotgun” overtrading

Most overtrading falls into one of three patterns:

* **Discretionary overtrading:** Trading on gut feel because the screen is open and the market is moving, with no rule behind the click.
* **Technical overtrading:** Seeing a valid setup on every chart and taking all of them, instead of waiting for the few that fit your plan.
* **“Shotgun” overtrading:** Firing off a cluster of small trades in quick succession, hoping volume makes up for the lack of a thesis.

**What it means:** If you can’t point to the rule a trade satisfied, it probably belongs to one of these patterns; an indicator-based check gives you that rule before you act.

### Connect the behavior to FOMO and revenge trading

Two emotional triggers sit behind most overtrading. The first is [FOMO in trading](/futures/blogs/understanding-fomo-in-trading-how-to-manage-emotions-effectively/), or fear of missing out, which pushes you into a move that’s already run. The second is revenge trading, where you size up right after a loss to win the money back fast.

Both replace a plan with a feeling. Indicators don’t remove the emotion, but they can put a measurable gate between the urge and the order, which is often enough to break the loop.

Once you can name the pattern and the trigger, you can assign each one an indicator that flags it in real time.

## Indicator #1: Average true range (ATR)—size every trade to real volatility

The first fix for overtrading is sizing discipline, and that starts with [ATR](/support/helpguides/nt8/NT HelpGuide English.html?average_true_range_atr.htm). Average true range (ATR) measures recent market volatility, so position sizes and stops scale to current conditions instead of an emotional impulse; NinjaTrader’s ATR indicator uses a 14-period default with Wilder’s exponential moving average.

### How ATR quantifies recent volatility (14-period default)

ATR averages the true range of each bar over a set lookback, so a single number tells you how much the market is actually moving right now. On NinjaTrader’s ATR indicator, the 14-period default smooths the reading enough to be stable without lagging badly behind a fast market.

**What it means:** A rising ATR means wider swings and more room needed on stops; a falling ATR means the market is quiet and your size assumptions from yesterday may be too large.

### Using ATR multiples for stop placement and position sizing

Many traders set stops at a multiple of ATR (for example, 1.5 or 2 times the current reading), so the stop sits beyond normal noise rather than at a round number. From there, you can size the position so the dollar risk to that stop stays constant, which is the core of treating [volatility as an indicator](/learn/technical-analysis/use-volatility-as-indicator-in-futures-trading/) rather than an afterthought.

This can help in two ways: your risk per trade can stay even across calm and wild sessions, and you can stop reaching for the same fixed size regardless of conditions.

### The “ATR check” that stops impulse-sized entries

The check is simple. Before you enter, confirm that your size and stop are derived from the current ATR reading, not from how badly you want the trade. If the volatility-adjusted size feels too small to be exciting, that’s usually a sign the trade was about excitement, not edge.

**What it means:** Letting ATR set your size removes the single most common form of impulse trading: oversizing into a market that’s already moving fast.

Sizing to real volatility takes the first leg out from under overtrading; the next is knowing when a move is already spent.

## Indicator #2: Relative strength index (RSI)—stop chasing extended moves

Chasing is the most visible symptom of overtrading, and the [RSI](/futures/blogs/relative-strength-index-in-futures-trading/) is built to flag it. The relative strength index (RSI) oscillates between 0 and 100 and flags overbought conditions above 70 and oversold conditions below 30, which helps traders avoid chasing extended moves—one of the most common triggers of overtrading.

### 70/30 overbought-oversold framing

Overbought means price has risen far and fast enough that momentum may be stretched; oversold is the same idea to the downside. The RSI indicator puts a number on it, so “this feels extended” becomes “RSI is at 78,” which is often far easier to act on consistently.

**What it means:** A reading past 70 or below 30 isn’t an automatic trade in either direction; it’s a caution flag that the easy part of the move may already be gone.

### How RSI flags FOMO entries before you take them

FOMO entries almost always come late, after a move has already extended. Glancing at the RSI indicator before you click gives you a quick gut check: if you’re about to buy into a reading near 80, you’re paying up for a move others entered far earlier. That pause is often all it takes to skip a low-quality trade.

### Pairing RSI with the trend to filter counter-trend impulse trades

RSI works better as a filter than as a standalone signal. In a strong uptrend, an overbought reading can persist for a long time, so fading every spike is its own form of overtrading. Pairing RSI with the prevailing trend can help you act on it sensibly: use oversold readings to time entries with the trend, and treat overbought readings against the trend as a reason to wait, not to short.

With sizing and momentum under control, the last filter is conviction, which is where VWAP comes in.

## Indicator #3: Volume-weighted average price (VWAP)—a conviction filter for every entry

Some entries clear the ATR and RSI checks but still lack a real reason behind them, and that’s where [VWAP](/futures/blogs/what-is-volume-weighted-average-price-vwap/) earns its place. Volume-weighted average price (VWAP) is the benchmark institutional traders use for execution quality; for retail futures traders, VWAP and its standard-deviation bands act as a conviction filter that exposes low-quality, impulse entries.

### Why institutional benchmarks anchor real opportunity

Large players measure their fills against VWAP because it reflects where real volume traded, not just where price ticked. Anchoring your own entries to that benchmark can help keep you on the side of genuine participation rather than thin, low-volume noise that tends to reverse.

### Trading with VWAP vs. fading it (and which side overtraders are usually on)

Trading with VWAP means entering in the direction price is holding relative to the benchmark; fading it means betting on a snap back toward the average. Overtraders are usually on the fading side, repeatedly trying to call a reversal against a trend that keeps holding above or below VWAP. Knowing which side you’re on is often the difference between a planned trade and a stubborn one.

### Standard-deviation bands as a “this trade is a stretch” alarm

Adding standard-deviation bands around VWAP (for example, plus or minus one standard deviation) gives you a visual alarm. When price pushes into the outer bands, the trade is statistically stretched, and a fresh entry there often means you’re late.

**What it means:** If your entry sits well outside the VWAP bands and against the trend, it’s a strong candidate for the trade you don’t take.

Each indicator handles one slice of the problem; combining them into a single routine is what can help make the discipline stick.

## Building a three-indicator pre-trade checklist (anti-overtrading rule set)

These indicators work best when they stop being three separate readings and become a single gate every trade has to clear. NinjaTrader includes ATR, RSI, and VWAP as built-in indicators with customizable parameters, and traders can combine them into a pre-trade checklist to enforce discipline and reduce overtrading.

Three-indicator pre-trade checklist

Before any entry, answer these yes/no questions:

1. **ATR:** Is my position size scaled to current volatility?
2. **RSI:** Am I entering on momentum, not chasing an already-extended move?
3. **VWAP:** Is price on the side of VWAP that fits my thesis, not a stretch away from it?

If you can’t answer yes to all three, you skip the trade. Cap the day at a fixed maximum number of trades, and stop when you hit it.

Treat this checklist as the backbone of your [futures trading plan](/futures/futures-trading-basics/futures-trading-plan/), and lean on a broader set of [the best futures trading indicators](/futures/blogs/best-futures-trading-indicators/) only after the basics are automatic.

### Three yes/no questions, one max-trades-per-day cap

The checklist above works because every answer is binary. There’s no “kind of”; a single no means you skip the trade, however good it feels, and running the three reads in the same order each time turns them into a gate rather than a suggestion.

The daily cap handles the other half of the problem. Sizing and momentum filters can help keep individual trades honest, but only a hard limit on how many you take can stop a hot streak or a run of losses from snowballing into a full session of overtrading.

### How to journal each indicator’s read after the fact

Discipline holds when you review it, so log each indicator’s reading at entry and revisit it later. Consistent [trade journaling](/futures/futures-trading-basics/trade-journaling/) can show you which checks you skip most often, and NinjaTrader’s trade performance journal can help you connect those skipped reads to your results over time.

A checklist you actually follow can do more for your equity curve than any single indicator can on its own.

## Start trading the checklist with NinjaTrader

All three indicators—ATR, RSI, and VWAP—are built into NinjaTrader, so you can add them to your charts and make the disciplined read your default before every entry. Ready to put the checklist to work?

[Open Your Free NinjaTrader Account Today](https://account.ninjatrader.com/register?utm_campaign=org-nt-us&utm_source=website&utm_medium=blog&utm_content=acq-account-open&utm_term=top-3-indicators-to-help-you-stop-overtrading)

Start trading futures with more structure and less impulse.

## FAQs about overtrading and indicators

What is overtrading?

Overtrading is trading more than your plan calls for, usually out of a need to stay active rather than because a real setup appeared. Because it’s a behavior rather than a bad strategy, the most reliable fix is a rule set that forces each trade to earn its place before you click.

What are the signs you’re overtrading?

Common signs include taking trades you can’t explain afterward, sizing up right after a loss, chasing moves that have already run, and blowing past your planned number of trades for the day. Rising commission costs relative to your results are another tell. If most of your trades lack a documented reason, you’re likely overtrading.

Can indicators alone stop overtrading?

No single indicator can stop it, because overtrading is a behavior, not a chart pattern. What indicators can do is give you objective gates that make the behavior harder to act on in the moment. Paired with a trade cap and a journal, ATR, RSI, and VWAP can help turn discipline into a repeatable process.

Which indicator is best for spotting overtrading?

It depends on your weakness. If you oversize, ATR is the most direct fix; if you chase, the RSI indicator flags extended moves before you enter; if you take low-conviction trades, VWAP exposes them. Most traders benefit from using all three together rather than picking one.

How do you combine ATR, RSI, and VWAP in a single setup?

Layer them in the order of what they control: size, then timing, then conviction. ATR sets the position size, the RSI indicator confirms you’re not entering late, and VWAP confirms there’s real participation behind the move. On NinjaTrader, all three can sit on a single chart, so the full read takes only a few seconds before you commit.

Futures, security futures, options, foreign currency, digital asset, and event contract trading involves substantial risk and is not suitable for everyone. Prior to trading security futures, review the [Risk Disclosure Statement for Security Futures Products](https://www.nfa.futures.org/investors/investor-resources/files/security-futures-disclosure.pdf).

Any trading symbols, company names, logos, or trademarks displayed are used for illustrative purposes only and remain the property of their respective owners. Their use does not imply affiliation with, endorsement by, or sponsorship of NinjaTrader, nor do any trading symbols displayed constitute investment advice or recommendations.

Previous Post

[Previous Post
Futures Trading Hours: Making the Switch From Other Asset Classes](/Futures/Blogs/futures-trading-hours)

Next Post

[$1K vs $10K vs $100K: How Account Size Changes Your Futures Trading Strategy

Next Post](/Futures/Blogs/futures-trading-account-size)

Related Posts

* ![](/assets/dist/images/dots/blog-dots.png)

  [### 3 Key Swing Trading Strategies You May Not Have Discovered Yet](/Futures/Blogs/swing-trading-strategies)

  July 02, 2026
* ![](/assets/dist/images/dots/blog-dots.png)

  [### Swing Trading vs. Day Trading Futures: Which Style Fits You?](/Futures/Blogs/swing-trading-vs-day-trading-futures)

  June 30, 2026
* ![](/assets/dist/images/dots/blog-dots.png)

  [### $1K vs $10K vs $100K: How Account Size Changes Your Futures Trading Strategy](/Futures/Blogs/futures-trading-account-size)

  June 30, 2026

Recent Posts

* [How to Choose a Futures Broker: A Decision Framework for Active Traders

  July 07, 2026](/Futures/Blogs/how-to-choose-a-futures-broker)
* [3 Key Swing Trading Strategies You May Not Have Discovered Yet

  July 02, 2026](/Futures/Blogs/swing-trading-strategies)
* [Swing Trading vs. Day Trading Futures: Which Style Fits You?

  June 30, 2026](/Futures/Blogs/swing-trading-vs-day-trading-futures)