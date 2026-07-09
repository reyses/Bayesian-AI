# Track Buying & Selling Pressure with Order Flow Cumulative Delta
**Date:** March 25, 2019
**Source:** https://ninjatrader.com/futures/blogs/track-buying-selling-pressure-with-order-flow-cumulative-delta/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Track Buying & Selling Pressure with Order Flow Cumulative Delta

# Track Buying & Selling Pressure with Order Flow Cumulative Delta

By
NinjaTrader

March 25, 2019

![Order flow data](/NT/media/Documents/order-flow-banner.png)

NinjaTrader’s Order Flow + premium tool set includes powerful chart studies to assist with order flow, volume and market depth analysis. *The Cumulative Delta* indicator helps [order flow traders](https://ninjatrader.com/Order-Flow-Trading) identify and monitor buying and selling pressure.

## What is Cumulative Delta?

Delta refers to the net difference between [buying and selling volume](https://ninjatrader.com/futures/blogs/use-volumetric-bars-to-track-buyers-sellers-see-order-flow-imbalance/) at each price level. Cumulative Delta builds upon this concept by recording a cumulative tally of these differences in buying vs selling volume.

## Passive vs Aggressive Orders

For traders using Cumulative Delta, limit orders are considered passive while market orders are considered aggressive. In other words, buying at the ask or selling at the bid are considered aggressive since these orders will fill immediately. This implies a sense of urgency in the market.

With that in mind, the Cumulative Delta calculation uses the following formula:

* *Market Buy Orders – Market Sell Orders = Delta*

When this result is a positive value, the buyers are seen as more aggressive and vice versa. Cumulative Delta keeps a running tally which displays a comprehensive historical and real-time view of order flow delta activity.

## Using the Cumulative Delta Indicator

The Cumulative Delta indicator plots as candlesticks in a panel below the price at time information. One of the main uses of Cumulative Delta is to confirm or deny market trends.

![cumulative-delta-blog.png](/getattachment/1c1135a7-5bfd-4af8-aa38-4bf36a692de8/cumulative-delta-blog.png "cumulative-delta-blog.png")  
In the example shown above, while the charted price data may suggest a bullish trend, the Cumulative Delta indicator featured in the bottom panel does not confirm this bias.

## Two Views of Cumulative Delta

There are two ways to plot the Cumulative Delta indicator, *Session* and *Bar*:

1. **Session** displays the delta accumulating over the course of a trading session with the closing price of the previous bar carried over to the open of the following bar. The example above shows a session display.
2. **Bar** displays the delta value per bar with no continuity. Therefore, it appears more as a bar graph with the value of the delta plotted as a positive (green) or negative (red) bar.

The bar display of the Cumulative Delta indicator is pictured below with each bar within the Cumulative Delta corresponding to the price bar directly above it. This display is helpful for identifying reversals or sudden changes in Order Flow activity.

![cumulative-delta-blog-2.png](/getattachment/0faf52a0-9ec9-4f25-aba3-5c94f2aa73dd/cumulative-delta-blog-2.png "cumulative-delta-blog-2.png")

## Get Started with Order Flow +

The Order Flow + suite of premium features available for NinjaTrader 8 provides tools to analyze trade activity using order flow, volumetric bars & [volume profiles](https://ninjatrader.com/futures/blogs/use-volume-profile-to-track-order-flow-on-charts/).

Current NinjaTrader users can get started with Cumulative Delta Indicator and the rest of the Order Flow + suite today: [Learn More](https://ninjatrader.com/LP/Order-Flow-Intro).

New to NinjaTrader? Our award-winning trading software is always FREE to use for advanced charting, backtesting and trade simulation. [Download Now!](https://ninjatrader.com/GetStarted)

Previous Post

[Previous Post
What are Micro E-mini Dow Futures (MYM)?](/Futures/Blogs/What-are-Micro-E-mini-Dow-Futures-MYM)

Next Post

[Backtest & Optimize Automated Strategies with the Strategy Analyzer

Next Post](/Futures/Blogs/Backtest-Optimize-Automated-Strategies-with-the-Strategy-Analyzer)

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