# Automating Your Trading Strategy With Alerts in NinjaTrader 8
**Date:** November 05, 2025
**Source:** https://ninjatrader.com/futures/blogs/automating-your-trading-strategy-with-alerts/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Automating Your Trading Strategy With Alerts in NinjaTrader 8

# Automating Your Trading Strategy With Alerts in NinjaTrader 8

By
NinjaTrader Team

November 05, 2025

Many traders don’t realize that NinjaTrader 8’s alert system can do more than send pop-up and email notifications—it can also automatically place live orders based on your rules. This feature bridges the gap between full automation and manual trading, allowing you to execute rules-based entries directly from your charts, no programming required.

This guide explains how to use chart alerts to automate orders, why they’re useful, and how to configure them step by step.

## Why automate with alerts?

Most traders start by using alerts to notify them when a price crosses a key level or an indicator fires a signal. But alerts can also take order actions, including submitting an order to your live NinjaTrader account or [simulation account](/trading-platform/trading-simulator/).

That means you can build simple, reliable automation directly on your chart:

* Save time and react faster as your orders execute when your conditions are met.
* Reduce human error and eliminate missed trades and emotional hesitation.
* Prototype strategies quickly before moving to NinjaScript or full strategies.
* Use your existing chart setup with drawing tools and indicators.

In short, alerts give you light automation inside NinjaTrader’s charting interface—perfect if you want faster execution but don’t want to code.

[Learn more about automating orders with chart alerts in this quick video.](https://vimeo.com/1131782541)

## How chart alerts work

A chart alert in NinjaTrader 8 has three main parts:

* **Conditions:** What must happen to trigger the alert (e.g., price crosses above a moving average, RSI falls below 30)
* **Actions:** What NinjaTrader should do when that condition occurs (e.g., play a sound, display a pop-up, send an email, submit an order)
* **Re-arm settings:** How often the alert can fire again (e.g., once per bar, every tick)

You can create alerts from any chart, Market Analyzer window, or hot list. Each alert is local to that chart tab, meaning it only runs while that tab is open and enabled.

[Learn more about using the NinjaTrader Market Analyzer.](/futures/blogs/how-to-use-the-ninjatrader-market-analyzer-your-personal-real-time-quote-board/)

## When to use alerts instead of strategies

While both alerts and strategies can automate trades, they serve different needs.

Use alerts when:

* You want to automate simple entry conditions tied to chart objects.
* You prefer a visual, no-code setup.
* You want manual control of stop/target management (via ATM templates).

Use strategies when:

* Your rules depend on complex conditions, position sizing, or dynamic targets.
* You need to monitor account balance, multiple instruments, or trade states.
* You’re ready to write or import NinjaScript code.

Think of alerts as the ideal middle ground between manual trading and full-scale automation.

[Learn more about creating chart alerts in the NinjaTrader Desktop.](https://support.ninjatrader.com/s/article/Creating-Chart-Alerts-in-the-NinjaTrader-Desktop-app?language=en_US)

## Step by step: Automating orders with alerts

Let’s walk through a complete example by creating an alert that automatically submits a limit buy order when price breaks above a trendline on your chart.

***Warning:** Start in simulation. Always test alert-based orders in your Sim101 account first to ensure everything behaves correctly.*

### Step 1: Open the Chart Alert dialog

![](/getmedia/85ccf882-c868-4400-8312-417e00faf6ed/Alert-Automating-Your-Trading-Strategy-With-Alerts-in-NinjaTrader-8.png)Figure 1: The Chart Alert dialog box in NinjaTrader

**Step 1-A**

* Right-click anywhere on your chart and choose Alerts. This opens the Alert Configuration window. Each chart tab can have its own alerts, so double-check you’re on the correct instrument and timeframe.
* Create a new alert. Click Add to create a new alert and give it a clear name—e.g., “Trendline Breakout Buy.”
* Define the condition. Under Conditions, specify what triggers the alert.
* Click Add to open the Condition Builder. On the left, choose what to monitor (e.g., Price, Indicator). On the right, choose the comparison (e.g., Crosses Above, Greater Than). Select the second item to compare against—like another indicator, trendline, or constant value.

***Example 1:*** Price → Crosses Above → Trendline 1   
*This means the alert will trigger as soon as the price breaks above your trendline.*

**Step 1-B**

Add the Action “Submit an Order.” Now comes the automation part.

* In the Actions section, click Add.
* Choose Submit an Order from the action list.
* Select the order parameters in the Properties panel on the right.

**Step 1-C**

Configure order details. In the Submit Order action, specify how the order should be sent.

* Account: Choose your account (e.g., Sim101 for testing).
* Instrument: Select the chart instrument or leave as @INSTRUMENT to use the active chart’s symbol.
* Action: Select Buy, Sell, Sell Short, or Buy to Cover.
* Order Type: Select Market, Limit, Stop Market, Stop Limit, or MIT.
* Limit/Stop Price: For limit or stop orders, specify the price or offset.
* Quantity: Set the number of contracts or shares.
* Time in Force (TIF): Select Day, GTC, etc.
* ATM Strategy (optional): Attach one of your ATM templates to automate stop and target orders.

***Example 2:*** When the condition triggers, NinjaTrader will automatically submit this order to your selected account with those settings.

* Action = Buy
* Order Type = Limit
* Limit Price = Ask − 1 Tick
* Quantity = 1
* ATM Strategy = 1:2 Bracket Scalp

---

**Pro tip:**Alerts can monitor almost any chart element including indicators, drawing objects, and even time conditions.

---

### Step 2: Add notifications

You can also add multiple actions to the same alert. Many traders include a pop-up or sound notification with the order submission to confirm that the alert fired. These are completely optional.

Example:

* ***Action 1:****Submit an Order*
* ***Action 2:** Display a Pop-up Message*
* ***Action 3:** Play Sound “Order Triggered.wav”*

This creates both execution and confirmation in one step.

### Step 3: Adjust trigger behavior and re-arm settings

This is a key feature/behavior; alerts can fire multiple times if you don’t limit them.

At the bottom of the alert window:

* **Re-arm seconds:** Set how long before it can trigger again.
* **Trigger on:** Choose whether it can trigger intrabar, once per bar, or at bar close.
* **Enabled:**Make sure it’s checked before you close the dialog.

For most setups, “Once Per Bar” and a short re-arm time (e.g., 10 seconds) prevent duplicate orders during rapid price moves.

### Step 4: Save and enable the alert

Click OK to save. Back on your chart, you’ll see a small alert icon appear in the upper left corner to confirm it’s active.

You can enable or disable alerts globally using the Alerts Log window (Control Center → New → Alerts Log).

### Step 5: Test in simulation

Always test in sim first:

* Connect to a live data feed in Sim101.
* Watch your chart as price approaches your condition.
* When triggered, confirm that:
  + The alert fires only once per event.
  + The correct order type, size, and ATM template are used.
  + The order appears in the Orders and Executions tabs.

If everything works as expected, you can switch to your live account once you’re comfortable.

***Example:*** Automating a moving average crossover

Let’s build another example: a simple moving average crossover that buys when a fast MA crosses above a slow MA.

* Open Alerts on your chart.
* Add a new alert named “MA Cross Buy.”
* Condition → Indicator (fast MA) Crosses Above Indicator (slow MA).
* Action → Submit an Order:
  + Action = Buy
  + Order Type = Market
  + Quantity = 1
  + ATM Strategy = Default 1:2
* Optional → Add Pop-up/Sound
* Re-arm → Once per bar close

When the crossover occurs, NinjaTrader sends your market order immediately and applies the ATM template for stops and targets.

## Managing and monitoring alerts

Open the Alerts Log from the Control Center (New → Alerts Log). This window shows:

* Each active alert
* The last trigger time
* Any errors (e.g., “Order rejected”)
* Buttons to edit, enable/disable, and remove alerts

If you close a chart, its alerts are paused. Reopen the chart and reenable alerts before your next session.

[Learn more about the powerful NinjaTrader Control Center.](/futures/blogs/intro-to-ninjatrader-desktop-control-center/)

## Advantages and limitations of alert-based automation

By automating your orders with alerts, you can install more discipline in your trading, help reduce stress, and free up your time for additional analysis and account management.

### Advantages

|  |  |
| --- | --- |
| **Feature** | **Benefits** |
| Speed | Executes instantly on condition without manual input |
| Simplicity | Configures directly on chart with no coding |
| Flexibility | Works with indicators, price levels, and drawing tools |
| Integration | Combines with ATM templates for complete order management |
| Scalability | Runs multiple alerts |

By combining alert-based entries with ATM exit management, you can automate tactical trade setups with confidence while maintaining visual control from your chart.

### Limitations

* Alerts are chart-specific; closing the chart disables them.
* You can’t yet backtest alerts; use strategies for that.
* More advanced logic (e.g., multiple conditions, scaling, trailing) requires a NinjaScript strategy.

In short, alerts are excellent for quick, event-driven automation, but they’re not a substitute for a full trading system.

## Putting it all together

Once you have determined your order criteria, build your alert rules in charting, try experimenting with ATM strategies, then test in the simulator before trading live.

* **Plan your trigger rules:** Identify what event defines your entry.
* **Build the condition:** Use the Condition Builder to formalize it.
* **Add the order action:** Configure your account, order type, and quantity.
* **Attach an ATM:** Let the ATM manage stops and targets.
* **Test thoroughly:** Use sim mode until the workflow is consistent.
* **Trade live:** Once you’re confident, try it on your live account.

With this workflow, you can automate a wide range of setups—drawing tools, indicator signals, time-based entries, and more—directly from your chart, no coding required.

[Learn more about using ATM strategies in NinjaTrader.](/futures/blogs/how-to-use-automated-trade-management-strategies-on-ninjatrader/)

## Make alerts a natural extension of your trading plan

NinjaTrader 8’s alert system is one of its most underutilized automation tools. By combining simple indicator condition logic with order entry actions, you can help bridge the gap between manual chart trading and fully coded strategies.

For most users, the goal isn’t 100% automation; it’s situational automation—letting the platform execute precisely when your rules are met, without emotions.

Start small, test in simulation, and refine your alerts until they act as a natural extension of your trading plan. Once dialed in, you’ll discover a smoother, faster, and more disciplined way to execute your strategy.

## Take the next steps to automation

Create your first alert on a chart (try a basic moving average crossover):

* Test in sim mode until you’re confident.
* Attach ATM templates for full bracket management.
* Expand your ideas into a strategy automation framework.
* [Explore best practices and troubleshoot with help from our pros.](https://support.ninjatrader.com/s/article/Creating-Chart-Alerts-in-the-NinjaTrader-Desktop-app?language=en_US)

References: <https://support.ninjatrader.com/s/article/Creating-Chart-Alerts-in-the-NinjaTrader-Desktop-app>

![](/NT/media/images/Lifestyles%20for%20Primary%20Pages/contract-specs-hero.png)

## Trade Futures with NinjaTrader

Haven't signed up for your free NinjaTrader account yet? [Get started today](https://account.ninjatrader.com/register?_gl=1*16ak14i*_gcl_aw*R0NMLjE3MzQ0NzI5NDIuQ2p3S0NBaUEzNFM3QmhBdEVpd0FDWnp2NFpRbU1YVWVCVjlzWEp6Y1Z3bGxYUnFmSHRjRkZqX1htUHpNQ1R4b2JDemFROWJIRW9lRFFCb0NUTmNRQXZEX0J3RQ..*_gcl_au*MTcxNTQ1MjM1OS4xNzMzMjQwMzA5) with a 14-day trial of live simulated futures trading.

[Start Trading Today](https://account.ninjatrader.com/register)

Previous Post

[Previous Post
Best Indicators for Futures Trading](/Futures/Blogs/5-Key-Indicators-for-Day-Trading-Futures)

Next Post

[When to Abandon a Futures Trading Strategy (and What to Do Next)

Next Post](/Futures/Blogs/When-to-Abandon-a-Futures-Trading-Strategy)

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