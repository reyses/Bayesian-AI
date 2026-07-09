# Monitor Your Available Margin with Excess Margin Columns
**Date:** August 20, 2020
**Source:** https://ninjatrader.com/futures/blogs/monitor-your-available-margin-with-excess-margin-columns/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Monitor Your Available Margin with Excess Margin Columns

# Monitor Your Available Margin with Excess Margin Columns

By
NinjaTrader

August 20, 2020

![Blue candlestick chart on PC screen](/NT/media/images/Blog/Blog%20Post%20Headers/Excess-Margin-Columns_8-20-20.png)

Futures traders are responsible for keeping their account balances within the margin guidelines specified by their broker. Fortunately for NinjaTrader users, NinjaTrader 8 provides direct visibility to your available excess margin helping you manage your positions to meet margin requirements.

## Watch a quick-start guide to managing futures margin:

## Understanding Margin & Position Management

Futures margin is the amount of money you must maintain in your brokerage account to protect against possible loss on an open trade. It generally represents a small percentage of the contract, typically 3-12% of the notional futures contract value.

* **Intraday Margin**, or [day trading margin](/futures/blogs/futures-day-trading-margins-intraday-margin), is the minimum account balance required by your broker to maintain a position of one contract (long or short) during trading hours.
* **Initial Margin**, or [exchange margin](/futures/blogs/futures-day-trading-margins-exchange-margins), is the per-contract minimum amount required by the exchange that must be maintained in your account to carry a position for multiple days. Initial margin is significantly larger than the intraday margin requirement.

As an example, while a balance of only $50 is required to maintain a position of 1 MES contract during trading hours, a much larger amount of $1320 is required to carry that position past the close. It is crucial for futures traders to understand margin requirements in order to avoid forced liquidations and fines.

### What is Excess Margin?

*Excess* margin is the amount of money in your account above the minimum margin requirements when in a position. Managing excess margin is critical in futures trading, since insufficient excess margin means you are in violation and could be subject to liquidation and/or fines from the broker trade desk.

By and large, the simplest way to manage excess margin is to trade contract sizes that are appropriate for your predetermined risk levels and account size.

While futures traders should always remain cognizant of open positions and account balance, the Excess Margin columns in NinjaTrader’s Accounts Tab display make it much easier to stay on top of margin.

## Add Visibility to Your Available Margin in NinjaTrader

The Excess Margin columns enhancement is available through the latest version of NinjaTrader 8. Existing users can upgrade to the latest version [here](https://ninjatrader.com/PlatformDirect) and new users can get started [here](https://ninjatrader.com/GetStarted).

To view the excess margin columns through your NinjaTrader platform:

1. Click *Accounts* at the bottom of the NinjaTrader Control Center. The Control Center is the default window that appears when you first launch NinjaTrader and is always displayed when NinjaTrader is running.
2. Next, right click and select *Properties*.
3. Under Columns, check *Excess initial margin* and *Excess intraday margin*.
4. Click *OK*.

![excess-margin-zoom-in.png](/getattachment/bcda7478-9142-445c-b2b2-61a0ce3d252b/excess-margin-zoom-in.png "excess-margin-zoom-in.png")The Control Center example above displays account information for a trader holding 1 Micro E-mini S&P 500 futures (MES) contract. The position is currently at breakeven, meaning there is no profit or loss on the position. The trader’s initial account balance was $1000. Note: Parenthesis indicate a negative number.

* Excess intraday margin = $950, or initial balance of $1000 minus intraday margin of $50
* Excess initial margin = ($320), or initial balance of $1000 minus initial margin of $1320

### What Do These Numbers Mean?

* **Excess Intraday Margin** is the amount of money in an account above the intraday margin required to hold a position during trading hours. In the example above, excess initial margin is $950, meaning that there is $950 in the account above the minimum $50 intraday margin for MES futures.
* **Excess Initial Margin** is this the amount in your account above the minimum margin required to hold a position overnight. In the example above, excess initial margin is ($320), meaning there is *insufficient* excess margin to hold this position beyond the close of today’s session.

Any number in parenthesis is a negative value and thus means you have insufficient excess margin.

For NinjaTrader Brokerage clients, intraday positions must be closed 15 minutes prior to session close. This is 3:45 pm CT for the majority of popular contracts which is 15 minutes before the official session close at 4:00 pm CT. At this time, traders should ensure they do not have a negative value in the *Excess intraday margin* column.

For swing or position traders, who plan on holding a position overnight or for multiple days, sufficient excess initial margin must also be maintained.

### Assign a Risk Template to Your Account

Note that in order for these columns to populate properly, there must be a **risk template** assigned to the account(s) in question.

1. To assign a risk template to an account, from the Accounts tab left click an account to select it.
2. Right click and select *Edit Account*.
3. Under Risk, select the appropriate risk template.\*
4. Click *OK*.

\*NinjaTrader Brokerage clients should select *NinjaTrader Brokerage Default*.

For more information on creating and editing risk templates, click [here](https://ninjatrader.com/support/helpGuides/nt8/?using_the_risk_window.htm).

## Get Started with NinjaTrader

NinjaTrader is always free to use for advanced charting, backtesting and simulated trading. Download our award-winning [trading software](https://ninjatrader.com/GetStarted), get your free [trading demo](/platform)& start tracking your favorite markets!

Previous Post

[Previous Post
8 Reasons to Trade Futures vs Stocks](/Futures/Blogs/8-Reasons-to-Trade-Futures-vs-Stocks)

Next Post

[Understanding Margins for Futures Trading

Next Post](/Futures/Blogs/Understanding-Margin-In-Futures-Trading)

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