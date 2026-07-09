# How to Use a NinjaScript Indicator in NinjaTrader
**Date:** February 03, 2026
**Source:** https://ninjatrader.com/futures/blogs/how-to-use-ninjascript-indicator/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* How to Use a NinjaScript Indicator in NinjaTrader

# How to Use a NinjaScript Indicator in NinjaTrader

By
NinjaTrader Team

February 03, 2026

Got a custom NinjaScript indicator you’re ready to try out in NinjaTrader? Whether you coded it yourself or downloaded it from a trusted source, getting it up and running is easier than you might think. In this guide, we’ll walk you through how to apply your custom indicator to a chart, test it in real time or in sim, and make sure it looks exactly how you want it to.

Let’s jump in!

## What is a NinjaScript indicator?

A NinjaScript indicator is a custom tool you can build—or [import](https://developer.ninjatrader.com/docs/desktop/import)—that calculates and displays specific trading signals or data. Think of it like your own personal version of a moving average, RSI, or trend-following tool, tailored to your trading style.

You can use indicators to highlight opportunities, filter trades, or simply visualize market conditions in a way that makes sense to you.

[Learn more in our NinjaScript Developer Guide: Getting Started With NinjaScript.](https://support.ninjatrader.com/s/article/Developer-Guide-Getting-Started-with-NinjaScript?language=en_US)

## Applying your custom indicator to a chart

Once your NinjaScript indicator is installed, adding it to a chart is a quick process:

1. Open the chart you want to use.
2. Right-click anywhere on the chart and choose Indicators, or press Ctrl+I.
3. In the Available list, scroll down or search for your custom indicator by name.
4. Select it, then click Add.
5. On the right, you’ll see all the settings you can customize—colors, line styles, inputs, and more.
6. When you’re ready, hit OK and your indicator will show up on the chart.

That’s it! You’ll now see your indicator updating in real time along with price movement.

## Testing your indicator in sim or Market Replay

Want to see how your custom indicator behaves before going live? Smart move. NinjaTrader gives you a couple of great ways to test:

* **Sim+ environment:** This is your practice space. Use real-time data and place trades risk-free to see how your indicator performs in actual market conditions.
* **Market Replay:** Want to test how your indicator reacts to a specific market day? Download historical data and replay it at real speed (or faster). You can pause, rewind, and even place simulated trades as you go.

Try your indicator across multiple instruments or timeframes to see where it shines… and where it might need a few tweaks.

## Visualizing your indicator clearly

Once your indicator is on the chart, take a few moments to customize how it looks. Here are some ways to make it pop:

* **Colors and styles:** Use the Plots section in the indicator settings to change line colors, thickness, and dash styles.
* **Overlay or separate panel:** Want the indicator to display right on top of your price bars? Set the Panel to “Same as input series.” Prefer it in a dedicated space below? Choose a separate panel number instead.
* **Markers and labels:** Some indicators support arrows, dots, or text to help highlight signals—these can often be toggled on/off or styled to your preference.
* **Alerts:** Want a heads-up when something important happens? If your indicator is built with alerts, you can set up pop-ups, sounds, or email notifications.

Customizing these visuals can make a big difference, especially when markets get busy.

## Editing or troubleshooting your indicator

Made your own NinjaScript indicator and something’s not quite working? No problem. Here's how to take a closer look:

1. Head to Tools > NinjaScript Editor.
2. Expand the Indicators folder and open your script.
3. Make any changes or corrections.
4. Save (Ctrl+S) and compile (F5) your code.
5. If something’s off, compile errors should appear in the NinjaScript Editor window.

You can also use Print() statements in your script to check how values are being calculated as the chart updates, which can be helpful for debugging. Any prints should appear in the NinjaScript Output window to help you track down the issue.

[Learn more about debugging your NinjaScript code.](https://developer.ninjatrader.com/docs/desktop/debugging_your_ninjascript_code)

## Using your indicator across charts and workspaces

Once you’ve got your custom indicator dialed in, it’s easy to make it part of your everyday workflow:

* Save a [chart template](https://support.ninjatrader.com/s/article/Templates-and-Presets?language=en_US) that includes the indicator so you can load it instantly on new charts.
* Add it to multiple instruments or timeframes across your workspace.
* [Export](https://developer.ninjatrader.com/docs/desktop/export) and share your indicator using the NinjaScript Export Wizard (great for collaborating with other traders or switching machines).

You can even set up a workspace with multiple charts—each with different configurations of your indicator—for a full view of the markets.

## Ready to build?

Before you build your own indicator, check out our NinjaScript documentation, examples, and walkthroughs:

* [NinjaScript Developer Guides](https://support.ninjatrader.com/s/article/Developer-Guides?language=en_US)
* [NinjaScript Best Practices](https://developer.ninjatrader.com/docs/desktop/ninjascript_best_practices)
* [NinjaScript Developer FAQ](https://support.ninjatrader.com/s/article/NinjaScript-Developer-FAQ?language=en_US)

These tools can help give you a solid starting point as you begin shaping indicators to fit your trading style.

![](/NT/media/images/Lifestyles%20for%20Primary%20Pages/man-phone-couch-1-533px.jpg)

## Make it your own with NinjaScript

Custom indicators are all about giving you more control over how you analyze the markets. Whether you're building tools from scratch or using ones shared by the community, NinjaScript indicators can help you trade the way you want, on your terms.

Ready to dive into NinjaScript indicators? Sign up for a NinjaTrader account today to get started.

[Get Started](https://account.ninjatrader.com/register?utm_campaign=org-nt-us&utm_source=website&utm_medium=medium&utm_content=acq-account-open&utm_term=ninjascript-indicator)

*Futures, options, foreign currency, and digital asset trading involves substantial risk and is not suitable for everyone. An investor may lose all or more than the initial investment. Trading should be undertaken only with risk capital—funds that can be lost without jeopardizing one’s financial security or lifestyle—and only by those who can afford such losses. Past performance is not necessarily indicative of future results. Prior to trading digital assets, review the CFTC and NFA advisories for additional information regarding the significant risks involved.*[*View Risk Disclosure Statement.*](/risk-disclosure-clearing/)

*NinjaTrader is a group of affiliated companies operating under NinjaTrader Group, LLC (“NTG”), including NinjaTrader Clearing, LLC d/b/a NinjaTrader, Kraken Derivatives US, and Tradovate (“NTC”). NTC is registered with the Commodity Futures Trading Commission (“CFTC”) as a futures commission merchant (“FCM”) and is a National Futures Association (“NFA”) Member (NFA ID: 0309379).*[*View Disclosures*](/customer-funds/)*.*

Previous Post

[Previous Post
To Become a Consistent Trader, Learn to Manage Your Risk](/Futures/Blogs/ninjatrader-risk-tools-trading-risk-management)

Next Post

[How to Use the APZ Indicator to Navigate Volatile Markets

Next Post](/Futures/Blogs/adaptive-price-zones-indicator)

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