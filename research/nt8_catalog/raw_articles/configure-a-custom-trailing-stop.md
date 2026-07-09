# Trailing Stop-Loss Configuration Guide
**Date:** October 28, 2019
**Source:** https://ninjatrader.com/futures/blogs/configure-a-custom-trailing-stop/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Configure a Custom Trailing Stop

# Trailing Stop-Loss Configuration Guide

By
NinjaTrader

October 28, 2019

![Configure a trailing stop](/NT/media/Documents/Trailng-Stop-Hero.png)

With NinjaTrader’s [automated trade management (ATM) strategies](/futures/blogs/automate-stops-targets-with-advanced-trade-management/), you can create custom trailing stop loss orders specific to your preferences.

## Why Use a Trailing Stop?

A trailing stop is a dynamic stop-loss order that “trails” behind the price as it moves in your favor. As the price moves further in your favor, the trailing stop automatically moves with it, limiting losses or locking in profit. Trailing stops can be used for both long and short trades, but they can only move in one direction. For example, once a trailing stop has moved up, it cannot move back down.

## Build a Custom Trailing Stop

Use NinjaTrader’s ATM Strategy feature to create a custom trailing stop that suits your particular preferences. For example, start with a loose stop and have it cinch tighter as each profit benchmark is hit. Additionally, adjust the frequency of the trailing stop and can scale out slowly or exit your entire position at once. The possibilities are endless!

From an order entry interface such as Chart Trader or the [SuperDOM](/futures/blogs/intro-to-the-ninjatrader-desktop-superdom/), you can build ATM strategies with trailing stops a few simple steps using the Stop strategy settings:

1. Use the ATM drop-down menu to select Custom. The Custom Strategy Parameters menu will appear.
2. Enter your initial Stop Loss and Profit values, then use the Stop strategy drop-down menu to select Custom. For this example, we have entered 10 ticks for the profit target and 10 ticks for the stop-loss.![NinjaTrader Custom Strategy Parameters menu for entering custom trailing stop-loss order parameters.](/getattachment/c06bea6d-5cd7-436e-a74f-3877db17f34b/Trailing-Stop-1.png "Trailing-Stop-1.png")3. Next click 1 Step to create a one-step trailing stop. Three additional fields will appear:

* **Profit Trigger:** The number of ticks in profit to activate the trailing stop
* **Stop-Loss:**The value in ticks behind the current price to trail the stop-loss order behind
* **Frequency:** The value in ticks that defines how frequently the stop-loss order will trail

We entered 3 for the profit trigger so that after 3 ticks in our favor, the stop-loss order starts adjusting. We entered 7 for the stop loss value so that the stop-loss is adjusted from 10 ticks behind the entry price to 7 ticks behind the current price. We set the frequency to 2 so that the stop-loss trails with every 2 ticks of profit.

![NinjaTrader Custom Strategy Parameters menu for entering custom trailing stop-loss order strategy parameters.](/getattachment/0749e1e5-d749-4ba0-87c7-9ad99bf27455/Trailing-Stop-2.png "Trailing-Stop-2.png")4. Enter the desired values into these fields and click OK.  
5. Click Save as Template and then type a name to save your ATM strategy template, which will include your Auto Trail parameters.  
6. Click Save and then click OK.

## Test Your Trailing Stop with the Simulated Data Feed

Once you’ve created an ATM Strategy template, you can test your strategy using the Simulated Data Feed. Please note: The simulated data feed provides internally generated market data and is not reflective of the actual market.

When connected to the Simulated Data Feed, a trend slider will appear that allows you to control the price either up or down. This is extremely helpful for testing ATM strategies. To connect to the Simulated Data Feed, first disconnect from any other open connections and then click Connections>Simulated Data Feed.

Use the trend slider to control the direction of the market as you test your ATM strategy on both the long and short side of the market to ensure it's performing as expected.

![NinjaTrader Stimulated Data Feed for tracking ATM strategies such as trailing stop-loss orders.](/getattachment/6e3dff88-903f-4a29-99c0-41453a7947f2/Trailing-Stop-3.png "Trailing-Stop-3.png")The award-winning NinjaTrader platform supports both advanced and [basic order types](/futures/blogs/three-basic-order-types-explained/) and is always free to use for advanced charting, strategy backtesting, and trade simulation. Get started building custom ATM strategies on our powerful, flexible [trading platform today!](/GetStarted)

Previous Post

[Previous Post
What is a Stop-Limit Order in Futures Trading?](/Futures/Blogs/What-is-a-Stop-Limit-Order-in-Futures-Trading)

Next Post

[What are Micro E-mini Russell 2000 Futures (M2K)?

Next Post](/Futures/Blogs/What-are-Micro-E-mini-Russell-2000-Futures-M2K)

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