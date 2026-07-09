# Managing Trade Risk Using Probabilities
**Date:** June 19, 2024
**Source:** https://ninjatrader.com/futures/blogs/managing-trade-risk-using-probabilities/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Managing Trade Risk Using Probabilities

# Managing Trade Risk Using Probabilities

By
NinjaTrader Team

June 19, 2024

Consistent futures trading requires more than just identifying higher probability trading setups—it's equally important to effectively [manage your risk](/futures/futures-trading-basics/risk-management/) on every trade. One powerful backtesting tool for evaluating and mitigating trade risk is the concept of maximum adverse excursion (MAE). MAE allows you to see typical adverse price movements experienced during winning and losing trades.

Traders can gain valuable insights for setting stop-loss levels for a trading idea or setup by understanding the statistical drawdown potential. Rather than relying on discretionary stop-loss distances or gut feelings, quantifying the MAE provides a data-driven approach to trade risk management, helping traders better avoid being prematurely stopped out but still protecting against catastrophic losses.

Most traders consider having a stop in place for every trade to be a key factor for long-term consistent trading. Watch this video to learn how to use probability-driven analysis to determine efficient [stop-loss orders](/futures/blogs/stop-loss-orders-which-order-type-to-use/).

&amp;lt;span class="fr-mk" style="display: none;"&amp;gt;&amp;nbsp;&amp;lt;/span&amp;gt;

**Additional topics discussed in this free livestream:**

* How each potential trade consists of three fundamental elements: an entry strategy, a profit target, and a protective stop
* Where to set an initial stop-loss order
* When to switch a stop-loss order to a training stop
* How MAE can help traders set statistically sound stop levels

## Using MAE to Manage Trade Risk: How is Maximum Adverse Excursion (MAE) Calculated?

MAE is calculated by reviewing a set of winning and/or losing trades to find the maximum distance the price moved against the trader's position from the entry point before exiting the trade with a profit or loss. It can be split into a drawdown or a short trade:

* **D****rawdown**: MAE is the difference between the entry price and the lowest price reached before the price started rising.

* **S****hort trade**: MAE is the difference between the entry price and the highest price reached before the price started falling.

By studying MAE over many trades, traders can statistically determine an appropriate stop-loss level that accounts for typical adverse price fluctuations to help being stopped out prematurely on winning trades. Analyzing MAE helps traders optimize their trade entries, stop-losses, and overall risk management based on the probability profile of their trading idea.

The NinjaTrader platform’s strategy development and backtesting features produce an MAE performance value as part of the strategy evaluation report. By utilizing the strategy optimization features, traders can run many historical statistical samples in a short period of time.

### Unlock Free Exclusive Training

Explore the foundational concepts of [technical analysis](/learn/technical-analysis/) with our free multi-video trading course “[Technical Analysis Made Easy](/learn/technical-analysis-made-easy/).” Learn how to analyze and anticipate market movements using market prices, volume data, and more.

### Join Us Live Each Trading Day

Get ready for the trading day ahead as our experts prep, analyze, and trade the futures markets in real time during our [daily livestream](/learn/livestreams/). Watch live here or catch what you missed on our [YouTube channel](https://www.youtube.com/channel/UCUXNA8JJOiMrqhB8NVNgClA).

### Start Trading With NinjaTrader

Sign up for your free NinjaTrader account today to kick off your [14-day trial of live simulated futures trading](https://account.ninjatrader.com/register?utm_campaign=org-nt-us&utm_source=website&utm_medium=website&utm_content=ninjatrader_blog&_gl=1*qb94vc*_gcl_aw*R0NMLjE3MDczMzQ5MDUuQ2p3S0NBaUE4WXl1QmhCU0Vpd0E1UjMtRTROakVrVmJRUmtOeC1aOTgzTFZBRTVOQURhZWJBTTlxWEczYzJzakZFU3MzRHFNQmkwc1F4b0NkSHNRQXZEX0J3RQ..*_gcl_au*ODI2NDEwNjU5LjE3MTEzNzg2MTE.).

Previous Post

[Previous Post
Market Correlation and Intramarket Relationships in Futures](/Futures/Blogs/Market-Correlation-and-Intramarket-Relationships-in-Futures)

Next Post

[Futures Trading Outlook: Treasury Bond Futures Stall

Next Post](/Futures/Blogs/Futures-Trading-Outlook-Treasury-Bond-Futures-Stall)

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