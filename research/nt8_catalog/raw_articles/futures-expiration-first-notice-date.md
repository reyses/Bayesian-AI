# Understanding Contract Expiration and First Notice Date in the Futures Markets
**Date:** February 19, 2026
**Source:** https://ninjatrader.com/futures/blogs/futures-expiration-first-notice-date/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Understanding Contract Expiration and First Notice Date in the Futures Markets

# Understanding Contract Expiration and First Notice Date in the Futures Markets

By
NinjaTrader Team

February 19, 2026

Every futures contract comes with its own set of rules and timelines. Two dates you’ll want to keep on your radar are the expiration date and the first notice date (FND)—key milestones that can shape how you manage open positions, plan rollovers, and stay aligned with your trading strategy.

Mark them in your calendar; these dates matter! They affect what you’re trading, how long you can hold a trade, and what happens if you don’t close your position in time (hint: It’s not good).

Let’s walk through what these terms mean, how they work in NinjaTrader, and how to stay on top of them, so you can trade smarter and avoid some unpleasant surprises.

More of a visual learner? Watch this quick video explainer.

&amp;lt;span class="fr-mk" style="display: none;"&amp;gt;&amp;nbsp;&amp;lt;/span&amp;gt;&amp;lt;span class="fr-mk" style="display: none;"&amp;gt;&amp;nbsp;&amp;lt;/span&amp;gt;

## What is a futures expiration date?

Milk isn’t the only thing that sours after its expiration date. With futures, the expiration date is basically the contract’s deadline—the final day it can be traded. After it passes, the contract either settles in cash or leads to physical delivery (more on that in a bit).

For example, E-mini S&P 500 Index futures are cash settled, which means they simply stop trading on their expiration date. No physical assets change hands; it’s just a final account adjustment.

But trading something like crude oil or gold—which are physically delivered—expiration carries much more weight, literally and figuratively. Unless you’re an oil refinery or a metals warehouse, you probably don’t want 25,000 pounds of copper delivered to your doorstep. Luckily, you don’t have to; just make sure to close your position ahead of the first notice date (FND), or NinjaTrader will automatically do it for you per our broker policies.

**Pro tip:** Close your position before expiration to avoid unnecessary costs or physical delivery.

## What does cash settled vs. physically delivered mean?

Futures contracts settle in one of two ways, and understanding the difference can help you plan your trades better.

### Cash-settled contracts

Cash-settled contracts track financial products like stock indexes or interest rates, where physical delivery isn’t part of the deal. At expiration, your profit or loss is settled in cash; no physical goods are exchanged.

Because there’s no delivery to manage, these contracts are often seen as more straightforward. Most traders in cash-settled markets still close their positions before expiration to avoid dealing with exchange settlements and fees.

### Physically delivered contracts

Physically delivered contracts involve real commodities—think corn, crude oil, or gold. Producers and buyers (e.g., farmers, refineries, manufacturers) use these contracts to hedge against future price swings. If a position is held too long, the buyer and seller are on the hook to exchange the actual product.

Retail traders, though? Most don’t want delivery. That’s why it’s essential to understand when to exit or roll your position before the FND.

Knowing whether a contract is cash settled or physically delivered can help you choose the right market, manage your expiration risk, and plan when to exit or roll your position with more confidence.

### Sample expiration dates

* **E-mini S&P 500:** Quarterly   
  Expires at 8:30 am CT on the third Friday of the contract month
* **Crude oil:** Monthly   
  Expires three business days before the 25th calendar day of the month prior to the contract month
* **Gold:** Semi-monthly   
  Expires at 12:30 pm CT on the third last business day of the contract month

Want to see the specs for your contract(s)? Check expiration dates and delivery types at [CME Group](https://www.cmegroup.com/).

## What is the first notice date (FND)?

The first notice date (FND) is the earliest day that a buyer of a physically delivered futures contract can be assigned delivery of the underlying commodity. In other words, the FND is your last warning; if you’re holding a position in a physical market and haven’t closed or rolled it before the FND, you could be on the hook for actual delivery. Every futures contract has a unique FND that you must know before trading.

Don’t worry, NinjaTrader automatically closes any position that risks going into delivery. But it’s still good practice to manage it yourself ahead of time.

### Sample FND patterns

* **E-mini S&P 500:** Not applicable; no FND (cash settled)
* **Crude oil:** Two business days after the expiration date
* **Gold:** One month before the expiration date

Key takeaway

FND = time to roll or close. Always check FNDs and delivery types at CME Group before trading any futures contract.

## What is rollover and when should you do it?

Want to stay in a trade beyond the expiration date or FND? You’ll need to roll your position. Rollover just means closing your current contract and opening a new position in the next active contract month, referred to as the front month.

This is especially common for swing and position traders, who often want to stay in a position longer without dealing with expiration dates or delivery risk.

## What is the front month… and why does it matter?

The front month is the futures contract with the closest expiration date. It usually has the highest volume and liquidity, which makes it the go-to for short-term traders and chart watchers. As contracts near expiration, volume shifts to the next contract. This becomes the new front month.

NinjaTrader automatically alerts you when it’s time to roll to the next front month (see Figure 1 below). You can also update contract months manually for your charts, quote lists, and trading tools.

![Auto rollover notification in the NinjaTrader Desktop platform.](/getmedia/7f44bcc6-2356-42cf-ba31-384d2a0f9fe1/Understanding-Contract-Expiration-and-First-Notice-Date-in-the-Futures-Markets-Figure-1.png)***Figure 1:** Auto rollover notification in the NinjaTrader Desktop platform.*

Watch rollover explained in this video

## Key takeaways

As a trader, you’ve got a lot to digest, so let’s break it down simply and succinctly:

* Every futures contract has an expiration date; don’t ignore it.
* First notice date (FND) applies to physically delivered contracts; act before it.
* Use rollover to maintain positions past expiration.
* Watch for front month shifts to stay aligned with market activity.

NinjaTrader helps automate these alerts, but it’s still smart to understand the details yourself. By staying ahead of expiration and FND, you can avoid costly mistakes and trade with more confidence.

## Stay sharp with daily market prep

We make it easy to see how the pros manage expiration, rollover, and more. Join us daily on NinjaTrader Live to see expert traders walk through market prep, chart analysis, and platform tips in real time. [Watch the livestream](/learn/livestreams/) or catch the replay on our [YouTube channel](https://www.youtube.com/@NinjaTrader).

[Sign up for your NinjaTrader account today](https://account.ninjatrader.com/register) to access free live simulated trading for 14 days and practice rolling contracts, watch expiration dates, and build confidence before you go live.

Simulated trading is based on hypothetical results and does not reflect actual trading. Emotional and psychological factors of real money risk are not replicated. Use simulated trading to learn the platform and markets—not as an indicator of live performance.

Previous Post

[Previous Post
How to Apply Timeframes in the NinjaTrader Platform](/Futures/Blogs/how-to-apply-timeframes-ninjatrader)

Next Post

[Fundamental Analysis of Ethereum Futures

Next Post](/Futures/Blogs/Fundamental-Analysis-of-Ethereum-Futures)

Related Posts

* ![](/assets/dist/images/dots/blog-dots.png)

  [### How to Choose a Futures Broker: A Decision Framework for Active Traders](/Futures/Blogs/how-to-choose-a-futures-broker)

  July 07, 2026
* ![](/assets/dist/images/dots/blog-dots.png)

  [### Futures Trading Hours: Making the Switch From Other Asset Classes](/Futures/Blogs/futures-trading-hours)

  June 24, 2026
* ![](/assets/dist/images/dots/blog-dots.png)

  [### Volume Profile Shapes: The 4 Patterns Every Futures Trader Should Know](/Futures/Blogs/Trade-Futures-Understanding-the-4-Common-Volume-Profile-Shapes)

  June 15, 2026

Recent Posts

* [How to Choose a Futures Broker: A Decision Framework for Active Traders

  July 07, 2026](/Futures/Blogs/how-to-choose-a-futures-broker)
* [3 Key Swing Trading Strategies You May Not Have Discovered Yet

  July 02, 2026](/Futures/Blogs/swing-trading-strategies)
* [Swing Trading vs. Day Trading Futures: Which Style Fits You?

  June 30, 2026](/Futures/Blogs/swing-trading-vs-day-trading-futures)