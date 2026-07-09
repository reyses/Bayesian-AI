# Backtest & Optimize Automated Strategies with the Strategy Analyzer
**Date:** June 13, 2019
**Source:** https://ninjatrader.com/futures/blogs/backtest-optimize-automated-strategies-with-the-strategy-analyzer/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Backtest & Optimize Automated Strategies with the Strategy Analyzer

# Backtest & Optimize Automated Strategies with the Strategy Analyzer

By
NinjaTrader

June 13, 2019

![Strategy analyzer futures trading](/NT/media/Documents/strategy-analyzer-futures.png)

NinjaTrader’s Strategy Analyzer is a powerful tool to test automated strategies using historical trading data. Based on this analysis, traders can optimize their strategy for peak performance in live market conditions.

Designed for use with strategies built using NinjaScript, NinjaTrader’s modern C# based trading framework, the Strategy Analyzer provides a robust solution for [backtesting](https://ninjatrader.com/Simulate), optimizing and analyzing the performance of [automated trading strategies](https://ninjatrader.com/Trade).

## Get Started with Strategy Analyzer

From the Control Center click *New > Strategy Analyzer*. This window is separated into two main sections:

* Settings panel (outlined in yellow)
* Performance Results panel (outlined in green)

![StrategyAnalyzer-futures-trading-blog.png](/getattachment/b0b4ac73-3c61-424f-815c-bab619d41af0/StrategyAnalyzer-futures-trading-blog.png "StrategyAnalyzer-futures-trading-blog.png")

## Settings Panel

The Settings panel is where users can select the Strategy Analyzer parameters applied including:

* **Strategy**: Specify the NinjaScript strategy to backtest or optimize.
* **Instrument**: Denote which instrument or instrument list will be used. Selecting an instrument list is a great way to quickly ascertain how multiple instruments would have historically performed and compare that data.
* **Type/Value**: Specify which interval type and value will be used in the backtest or optimization.
* **Time frame**: Specify the period of time used in the backtest. *Please note that in order to run a backtest over historical data, NinjaTrader must be connected to a data provider which supplies the appropriate historical data or this data must be saved prior to running the backtest.*

Once set to the desired preferences, click *Run* to perform the backtest. The Strategy Analyzer will display a message in the bottom right corner of the window to indicate if a backtest is still running.

## Performance Results Panel

Once completed, the backtest results can be viewed in the Performance Results panel.

![StrategyAnalyzer-futures-trading-blog2.png](/getattachment/71dde3e2-d25c-4292-82e6-46c2735250c0/StrategyAnalyzer-futures-trading-blog2.png "StrategyAnalyzer-futures-trading-blog2.png")  
Located to the left of the Settings panel, the Performance Results panel displays results based on the report selected in the *Display* selector, highlighted above.

If a backtest was run using an instrument list as demonstrated in the window above, a list of each instrument will appear at the top of the Performance Results panel. Here, you can select each instrument’s individual results as well as the combined results of the entire instrument list.

Below are a few of the report styles available within the Display selector:

* **Summary**: Displays all performance statistics and metrics (pictured above)
* **Analysis**: Displays data based on various time periods for analysis
* **Chart**: Displays a price-over-time chart with order executions plotted over the price data
* **Executions**: Lists individual entries and exits
* **Trades**: Lists individual trades
* **Orders**: Lists the orders used

The example below features the Charts display selected within the Performance Results panel.

![StrategyAnalyzer-futures-trading-blog3.png](/getattachment/14c9aef8-48e5-40ed-910b-3a76b30107e2/StrategyAnalyzer-futures-trading-blog3.png "StrategyAnalyzer-futures-trading-blog3.png")  
Interested in building and testing your trading strategies using an [open source trading platform](https://ninjatrader.com/Build)? NinjaTrader is always FREE to use for advanced charting, backtesting and trade simulation. [Get started now!](https://ninjatrader.com/GetStarted)

Previous Post

[Previous Post
Track Buying & Selling Pressure with Order Flow Cumulative Delta](/Futures/Blogs/Track-Buying-Selling-Pressure-with-Order-Flow-Cumulative-Delta)

Next Post

[What Are Micro E-mini Nasdaq 100 Futures (MNQ)?

Next Post](/Futures/Blogs/What-Are-Micro-E-mini-Nasdaq-100-Futures-MNQ)

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