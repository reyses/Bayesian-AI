# Mastering Risk Management in Futures Trading
**Date:** September 13, 2023
**Source:** https://ninjatrader.com/futures/blogs/mastering-risk-management-in-futures-trading/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Mastering Risk Management in Futures Trading

# Mastering Risk Management in Futures Trading

By
NinjaTrader Team

September 13, 2023

![Technical Analysis Explained](/NT/media/Documents/Risk-management-blog-hero.jpg)

Ask any futures trader who’s been around for a while what their secret for success has been, and there’s a high probability they will answer, “I follow a strict risk management routine on every trade.” Protecting your available capital to trade is paramount—you can’t trade if you wipe out your account. That’s why it’s important to build and follow a consistent [risk management plan](https://ninjatrader.com/futures/learn/futures-trading-basics/risk-management/) when trading futures.

Due to the increased leverage available through futures, even small price movements can result in significant profits or losses. Being aware of the potential risk is only the start. Traders can help better manage and mitigate their exposure by following a consistent disciplined approach by learning how to manage risk in futures trading.

## Account Level Stop Loss

It’s easy for a trader to get too caught up in the action by chasing more profits or trying to recover losses. This overtrading is often caused by a lack of discipline that can lead to bad results. While building an effective [futures trader’s mindset](https://ninjatrader.com/futures/learn/futures-trading-basics/traders-mindset/) can help, experienced traders also know it’s good practice to set daily or weekly limits that lock in a profit or limit further losses. Once you’ve reached your profit goal, lock it in and take the win.

NinjaTrader offers traders account-level risk settings that will automatically liquidate positions daily or weekly once a profit target or stop loss is hit. There’s also an account trailing stop risk setting, and traders can choose to lock themselves out from entering new positions until the following session. (Figure 1)

![Risk-management-blog-1-b-2.png](/getattachment/9da8e08f-5a28-4790-8408-c33d7aa5dab0/Risk-management-blog-1-b-2.png "Risk-management-blog-1-b-2.png")

*Figure 1: NinjaTrader account risk settings*

We made it easy for you! Learn how to:

* [Set a Daily Profit Trigger in Account Risk Settings](https://support.ninjatrader.com/s/article/How-do-I-set-a-Daily-Profit-Trigger-on-my-account?language=en_US)
* [Set a Daily Loss Limit in Account Risk Settings](https://support.ninjatrader.com/s/article/How-Do-I-Set-a-Daily-Loss-Limit-on-my-Account?language=en_US)

## Trade Stop Loss

Every time you get into your car, you have your planned route, check your fuel and mirrors, put on your seat belt, and off you go. On top of all that, you have car insurance just in case something goes wrong. Stop loss orders are your trading insurance, helping to protect your account from large losses.

Before placing a new trade, a trader must first try to determine the best correct stop loss amount to protect their capital and still give the market enough room for fluctuations without the position getting stopped out too early or too often. Only then can the trader set the correct trade size to stay within their account risk tolerance for a single trade.

[Stop loss orders](https://ninjatrader.com/futures/blogs/stop-loss-orders-which-order-type-to-use/) are typically stop market orders—these are conditional orders based on a selected stop price used to protect against losses on an open position. When placing a buy stop order to close a short position, the stop price must be above the current market price. And when placing a sell stop order to close a long position, the stop price must be below the current market price.

Whether you plan to be in a position for a few minutes or a few days, it’s always best practice to have an active stop loss order in the market for every open position. News events and other market conditions can turn a position against you on a dime, with devastating effects. [Learn more about stop loss orders and which order types to use](https://ninjatrader.com/futures/blogs/stop-loss-orders-which-order-type-to-use/).

It is also important to remember that if you are holding a position overnight to set the trade duration on your stop loss order to good till cancelled (GTC).

## Advanced Trade Management Strategies

When a trader places an order to enter a new position, they can attach additional advanced orders to the entry order to both capture profit and limit losses with one order. Depending on your trading style, these exit orders can be as simple as placing a stop loss and profit target bracket or a trailing stop. These types of exit orders can also be more complex, providing the flexibility to scale out of a position at different price levels or move a stop order to break even.

NinjaTrader Desktop offers [Advanced Trade Management](https://ninjatrader.com/futures/blogs/automate-stops-targets-with-advanced-trade-management/) (ATM) strategies to help manage:

* Entry and exits with multilevel orders,
* Automatic trailing and breakeven stops,
* Multilayered bracket profit and stop levels.

![Risk-management-blog-2-b-1.png](/getattachment/2fa525ff-957e-438d-afa1-255837f71768/Risk-management-blog-2-b-1.png "Risk-management-blog-2-b-1.png")

*Figure 2: Advanced Trade Management dialog to scale out of a position*

These versatile strategies can be modified to accommodate trades of any quantity and can include as many stop loss and profit target levels as the trader requires and make it easy to take advantage of powerful order placement techniques to help better manage orders and positions and automate stops and targets. There are several preconfigured ATM strategies to use, and you can customize them for any trading scenario. [Learn more on how to Automate Stops and Targets With Advanced Trade Management](https://ninjatrader.com/futures/blogs/automate-stops-targets-with-advanced-trade-management/).

## Trade Scenario Example

Let’s say you buy three contracts long in the Micro E-mini S&P 500 futures. With an attached ATM bracket strategy, the profit target and stop loss orders are automatically placed once the entry order is filled. If the profit target is hit first, the stop loss order is automatically cancelled. If the stop loss is hit first, the profit target order is also auto cancelled. This is called a bracket OCO (one cancels other) order. The profit target and stop loss orders can be set independently to different price levels as required.

## Fixed Fractional Approach to Trade Sizing

Finally, any solid risk management in a futures trading approach considers both trade risk and account risk. Trade risk is the maximum amount you’re willing to lose on any one trade, and account risk is the consistent maximum percentage of your account you’re willing to risk on any one trade. To balance these two approaches, traders will often adjust their stop loss and trade size so that every losing trade will affect their trading account in a similar way. This concept is referred to as fixed fractional.

Fixed fractional is a mathematical formula that allows you to calculate an appropriate trade size for your account equity, risk tolerance, and maximum position stop loss on a single trade. If you determine your risk tolerance is 3%, this means you only want to lose a maximum of 3% of your account size on any one trade. With a 3% risk tolerance, it would take 23 losing trades in a row to lose half your starting account equity.

The lower your risk tolerance on each trade, the more likely you are to weather a significant account drawdown. Using fixed fractional will also help you determine which markets are more appropriate to trade for your available starting account balance. For example, if you have limited funds to start, trading micro contracts will give you more flexibility in trade sizing and stop loss amounts.

## Fixed Fractional Trade Size Example

With a starting account of $10,000 and a maximum account risk percent of 3%, the maximum trade risk is $300 per trade. If you’re trading the MES Micro S&P 500 with a point value of $5:

* Set maximum stop loss amount for 1 contract to $300 or 60 points.
* Set maximum stop loss amount for 2 contracts to $150 or 30 points.
* Set maximum stop loss amount for 3 contracts to $100 or 20 points.

In this example, each trade loses or is stopped out at $300, as the stop loss amount is reduced for increased trade size contracts. Choose the trade size that would give your position the best chance to avoid being stopped out with normal price volatility.

## Get Started on Your Path to Learn How to Trade Futures

There are a variety of ways for futures traders to manage risk, and identifying an approach that fits your personality and trading approach is a critical component of any [futures trading plan](https://ninjatrader.com/futures/learn/futures-trading-basics/futures-trading-plan/).

When starting out, trading in a simulated environment is one of the best ways to master risk management, gauge stop loss levels, and practice order placement without risking real dollars. Implementing a consistent risk management in your futures trading plan on every trade is the best way to help reduce the stress and loss of confidence that can come from one or more catastrophic losses. The goal here is to protect your trading capital so you can continue your futures trading journey.

For more tips, we’ve created a free multi-video trading course, “Develop the Trader in You,” to help guide new futures traders through the first critical stages as you get started. It also provides a roadmap for all the key learning points you’ll need to master. [Watch Part 1](https://ninjatrader.com/futures/learn/develop-the-trader-in-you/).

### Start Trading Futures With NinjaTrader

NinjaTrader supports more than 800,000 futures traders worldwide with our award-winning trading platforms, world-class support and [futures brokerage](https://ninjatrader.com/Futures) services with $50 day trading margins. The NinjaTrader desktop platform is always free to use for advanced charting, strategy backtesting, technical analysis and trade simulation.

Get to know our:

* [**Futures brokerage**](https://ninjatrader.com/futures/): Open an account size of your choice—no deposit minimum required—and get free access to our desktop, web and mobile trading platforms.
* [**Free trading simulator**](https://ninjatrader.com/trading-platform/trading-simulator/): Sharpen your futures trading skills and test your ideas risk-free in our simulated trading environment with 14 days of livestreaming futures market data.

**Better futures start now**. Open your free account to access NinjaTrader’s award-winning trading platforms, plus premium training and exclusive daily market commentary.

Previous Post

[Previous Post
Identifying Trend With Moving Averages](/Futures/Blogs/Identifying-Trend-With-Moving-Averages)

Next Post

[Trading Psychology Tips to Manage Futures Trading Emotions

Next Post](/Futures/Blogs/Trading-Psychology-Tips-to-Manage-Futures-Trading-Emotions)

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