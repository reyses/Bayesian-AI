# What is a Stop-Limit Order in Futures Trading?
**Date:** October 21, 2019
**Source:** https://ninjatrader.com/futures/blogs/what-is-a-stop-limit-order-in-futures-trading/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* What is a Stop-Limit Order in Futures Trading?

# What is a Stop-Limit Order in Futures Trading?

By
NinjaTrader

October 21, 2019

![Stop limit order futures trading](/NT/media/Documents/What-is-a-Stop-Limit-Hero.png)

A stop-limit order is a [basic order type](https://ninjatrader.com/futures/blogs/three-basic-order-types-explained/) which issues a limit order once a specified price has been reached. This price level is known as the stop price, and when it is touched or surpassed, the stop-limit order becomes a limit order.

Stop-limit orders are conditional orders which combine the features of [stop orders](https://ninjatrader.com/futures/blogs/what-is-a-stop-order-in-futures-trading/) with those of limit orders. Stop-limit orders are not used solely for exiting positions and can be beneficial for entries as well. When entering a position, traders use stop-limit orders to determine where a limit order should be triggered. Conversely, traders also use stop-limit orders for exiting trades to help minimize risk and book profits.

## Buy Stop-Limit Order vs Sell Stop-Limit Order

A buy stop-limit order must be entered above the current market price, and a sell stop-limit order must be entered below the current market price. If the stop price is not touched by the market’s current value, no subsequent limit order will be issued.

Once the stop price is touched, a limit order is issued at a predefined number of ticks away from the initial stop price. This combines the features of a stop order with those of a limit order.

Advantage of Stop-Limit Orders

The main benefit of using a stop-limit order is the precise control the trader has over where an order should be executed.

Stop-limit orders provide traders with the ability to specify a price where a limit order will be triggered. Once the market reaches this stop price, a limit order with a predefined number of ticks away from the stop price is then issued to the exchange.

**Disadvantage of Stop-Limit Orders:**

While stop-limit orders offer precise control over where orders will execute, they do not guarantee a fill.

If the security does not reach the specified stop price, no limit order will be issued to the exchange. Additionally, even if the stop price is reached and the limit order is issued, it will only fill if the market reaches the limit price with enough volume.

In this sense, opportunities to enter or exit the market could be missed with stop-limit orders.

### Examples of Stop-Limit Orders

![StopLimit1.png](/getattachment/67ebbb85-a0bf-4401-9338-374b2fc8eae6/StopLimit1.png "StopLimit1.png")  
From the [E-mini Nasdaq 100 futures (NQ)](https://ninjatrader.com/futures/futures-contracts/equity-index/e-mini-nasdaq/) chart above, with the market currently trading at 7,677.50, the buy stop-limit order at 7,680.50 would require the market to move up to 7,680.50 to trigger and would then issue a *buy limit order.*

On the other side of the market, the sell stop-limit order at 7,676.75 would require the market to move down to 7,676.75 to trigger and would then issue a *sell limit order*.

**Learn more about basic order types in this quick video overview:**  
    
The award-winning NinjaTrader platform supports both basic and advanced order types.

NinjaTrader is always free for advanced charting, strategy backtesting and trade simulation. [Get started](https://ninjatrader.com/GetStarted) with a free futures data trial and try our free [trading simulator](https://ninjatrader.com/Simulate) to explore market opportunities!

Previous Post

[Previous Post
Why Are Opening and Closing Prices Significant for Traders?](/Futures/Blogs/Why-Are-Opening-Closing-Prices-Significant-for-Traders)

Next Post

[Trailing Stop-Loss Configuration Guide

Next Post](/Futures/Blogs/Configure-a-Custom-Trailing-Stop)

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