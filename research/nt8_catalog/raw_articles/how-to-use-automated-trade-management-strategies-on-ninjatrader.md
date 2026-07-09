# How to Use Automated Trade Management Strategies on NinjaTrader
**Date:** January 02, 2024
**Source:** https://ninjatrader.com/futures/blogs/how-to-use-automated-trade-management-strategies-on-ninjatrader/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* How to Use Automated Trade Management Strategies on NinjaTrader

# How to Use Automated Trade Management Strategies on NinjaTrader

By
NinjaTrader Team

January 02, 2024

The NinjaTrader Desktop platform comes equipped with powerful automated trade management (ATM) capabilities that empower futures traders with semi-automated trade functionality. By using these strategies, traders can stay more focused on their trading goals.    
  
ATM strategies manage positions automatically to reduce the impact of emotions on trading decisions. Within milliseconds of entering a position either long or short, stop-loss and take-profit orders are submitted based on predefined settings. Features available through ATMs include the ability to:

* Attach advanced orders to your entry orders on the fly
* Execute conditional or contingent orders, like bracket orders and trailing stops
* Access and employ ATM strategies easily through a user-friendly interface in both the NinjaTrader SuperDOM and Chart Trader

**Get Started With ATM Strategies in Your NinjaTrader Platform**  
  
Watch this how-to video with tips to help you navigate ATM strategies as a part of your futures trading:  

## How to Use the NinjaTrader Custom ATM Strategy Builder for OCO Orders

The Custom ATM Strategy Builder empowers traders to create and execute various contingent order types, including one-cancels-other (OCO) orders.   
  
![ATM](/getmedia/e31d14ba-b5d9-47e7-a841-adb258ec7820/ATM.png "ATM.png")

*ATM Custom Parameters Dialog*

### ATM Strategies: Using a Profit and Loss Exit Bracket

One of the most popular OCO order configurations is a simple bracket order, which can automatically place both a profit target and stop-loss order for an existing position.    
  
When one of the orders gets triggered and filled, the other order is automatically canceled. This results in the trader being “flat,” with no active order. Additionally, multi-target OCO orders can be employed to manage multiple bracketed profit target and stoploss levels for scaling out of a multi-contract position.   
  
![ATM-2](/getmedia/abfa25c2-6796-4ae3-b0c3-1683a428a5f3/ATM-2.png "ATM-2.png")

*Scaling Out With a Multi-Level ATM OCO Bracket Order*

### ATM Strategies: Using a Trailing Stop

Using automated trailing stops is another common ATM strategy that allows traders the ability to move (cancel and replace) a stop order automatically based on a predetermined trailing increment. The stop order trails from the position high for a long position or trails from the position low for a short position. As the market makes new highs or lows, the trail will continue to adjust until the market retraces back to the trailing stop level and the position is exited with a profit or a loss.  
  
![ATM-3](/getmedia/a98a1286-33e4-494c-a90f-e83095e338bd/ATM-3.png "ATM-3.png")

*Setting an ATM Trailing Stop*

### ATM Strategies: Using a Breakeven Stop

Another popular ATM strategy is the auto breakeven stop, which is a risk management tool designed to prevent an already profitable trade from turning into a losing trade. The breakeven stop kicks in once a position has reached a defined minimal profit level.  
  
![ATM-4](/getattachment/a3ded0fa-644d-4489-a047-17d7851472df/ATM-4.png "ATM-4.png")

*Setting an ATM Breakeven Order*

There are also other useful ATM strategies like reverse at stop, reverse at target, target chase, and limit chase, which you can employ in your trading.

![ATM-5](/getmedia/714e36fe-b320-4143-b7d0-d9470ab22e85/ATM-5.png "ATM-5.png")

*Additional ATM Strategy Functions*

### Build and Test Your Own ATM Strategies

NinjaTrader offers ATM strategies as part of its comprehensive set of trading and analytical tools for every level of trader, which you can access by [creating a free NinjaTrader account](https://account.ninjatrader.com/register). Having the ability to place powerful conditional orders using ATM strategies like OCO brackets, trailing stops, breakeven stops, and others, helps traders limit emotional decisions and manage risk more precisely.   
  
These ATM strategies also reinforce a disciplined approach to trading, making them valuable tools for both novice and experienced traders alike. Learn about these strategies and more platform tips with our [free platform training videos](https://ninjatrader.com/learn/platform-training/).

Previous Post

[Previous Post
Trading Micro Natural Gas Futures](/Futures/Blogs/Trading-Micro-Natural-Gas-Futures)

Next Post

[2024 Market Predictions: Potential FOMC Rate Cuts to Impact the Year Ahead

Next Post](/Futures/Blogs/2024-Market-Predictions-Potential-FOMC-Rate-Cuts-to-Impact-the-Year-Ahead)

Related Posts

* ![](/assets/dist/images/dots/blog-dots.png)

  [### How to Solve 5 of the Most Common Mistakes on the NinjaTrader Mobile App](/Futures/Blogs/fix-common-ninjatrader-mobile-app-mistakes)

  April 11, 2026
* ![](/assets/dist/images/dots/blog-dots.png)

  [### 8 Reasons Your New NinjaTrader Dashboard Is a Serious Upgrade](/Futures/Blogs/ninjatrader-dashboard-features)

  March 25, 2026
* ![](/assets/dist/images/dots/blog-dots.png)

  [### Introducing the All-New NinjaTrader Dashboard](/Futures/Blogs/ninjatrader-new-dashboard)

  March 19, 2026

Recent Posts

* [How to Choose a Futures Broker: A Decision Framework for Active Traders

  July 07, 2026](/Futures/Blogs/how-to-choose-a-futures-broker)
* [3 Key Swing Trading Strategies You May Not Have Discovered Yet

  July 02, 2026](/Futures/Blogs/swing-trading-strategies)
* [Swing Trading vs. Day Trading Futures: Which Style Fits You?

  June 30, 2026](/Futures/Blogs/swing-trading-vs-day-trading-futures)