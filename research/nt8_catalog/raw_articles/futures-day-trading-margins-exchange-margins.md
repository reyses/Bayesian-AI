# Futures Day Trading Margins: Exchange Margins
**Date:** April 12, 2017
**Source:** https://ninjatrader.com/futures/blogs/futures-day-trading-margins-exchange-margins/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Futures Day Trading Margins: Exchange Margins

# Futures Day Trading Margins: Exchange Margins

By
Trade Desk

April 12, 2017

![Stock Exchange Background](/NT/media/images/Blog/Blog%20Post%20Headers/Exchange-Margins-Post_4-12-17.png)

Different from Intraday Margins specified by the broker, which represent the minimum balance an account must maintain, Exchange Margins are mandated by the exchanges.

Below is a continuation of the first [Futures Day Trading Margins: Intraday Margin](https://ninjatrader.com/blog/futures-day-trading-margins-intraday-margin-1-2/) article from the NinjaTrader Trade Desk which details the intricacies of Exchange Margins utilizing a hypothetical account owned by futures trader, Jane Smith.

## Exchange Margins – Initial Margin

Initial Margins are set by the respective exchange and represent the amount required to hold a position into the next trading sessions.

Unlike Intraday Margins, Exchange Margins can change frequently and may fluctuate based on expected upcoming volatility. Additionally, they are only applicable when the markets are closed from 4:00 PM – 5:00 PM Central Time for most CME products; equities, [metals](https://ninjatrader.com/futures/futures-contracts/metals/) and interest rates all close at 4:00 PM Central Time. Although the markets close at 4:00 PM Central Time, traders need to be in line with Exchange Margins before 3:45 PM Central Time as the risk desk liquidates accounts during the last 15 minutes of the trading session that do not meet Exchange Margin requirements to avoid margin calls.

Currently, the Initial Margin to carry one contract on the ES from one session to the next is $5,225. With an Initial Margin price of $5,225, Jane’s $10,000 trading account would only allow her to carry one contract overnight into the next trading session. Should Jane attempt to carry 2 contracts of the ES into the next trading session, she would trigger a margin call ($5,225 x 2 = $10,450 (initial margin) > $10,000 (Jane’s balance)).

A margin call is a situation where the FCM posts funds ($450 for Jane $10,450 – $10,000 = $450) on the trader’s behalf. One the margin call is activated, Jane would have 24 hours to meet the call by wiring funds posted on her behalf.

## Exchange Margins – Maintenance Margins

Maintenance Margins are a set minimum margin (per outstanding futures contract) that a trader must maintain on positions carried longer than one day.

If Jane carries just one contract to the next day, but does not trade on that subsequent day, she will only need to post a Maintenance Margin for that day. Maintenance Margins are lower than Initial Margins and are only for positions that have been open and unchanged for more than one day.

For example, Jane carries one contract for multiple days at a price of 2300. After covering margin of $5225 on day 1, she only needs to have $4,750 for each day forward, as long as she doesn’t trade. An account that adds to a position (long 1 to long 2 contracts) would have to meet initial margin for both contracts the day contract 2 is added, essentially resetting the margin.

## Margins Recap

To summarize, Intraday Margin is the minimum account balance required to enter one contract during trading hours. Exchange Margins are broken into two categories, Initial Margin and Maintenance Margin. Initial Margin is the balance required to carry one contract to a new trading session. Maintenance Margin is the amount required to carry the same position for multiple days.

Violations occur when an account is in a position with a balance below the intraday requirements or when an account carries a position without having the funds to meet the initial margin. Both violations should be avoided at all costs as they can lead to liquidations and fines from the risk desk. If you have any questions with this topic, please contact the [NinjaTrader Trade Desk!](/homepage/contact-us)

Previous Post

[Previous Post
Intraday Margins in Futures Day Trading](/Futures/Blogs/Futures-Day-Trading-Margins-Intraday-Margin)

Next Post

[Exponential Moving Average (EMA) - A Pillar Of Technical Analysis

Next Post](/Futures/Blogs/Exponential-Moving-Average-EMA-A-Pillar-Of-Technical-Analysis)

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