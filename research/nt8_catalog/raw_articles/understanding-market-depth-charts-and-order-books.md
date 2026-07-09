# Understanding Market Depth Charts and Order Books
**Date:** November 15, 2021
**Source:** https://ninjatrader.com/futures/blogs/understanding-market-depth-charts-and-order-books/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Understanding Market Depth Charts and Order Books

# Understanding Market Depth Charts and Order Books

November 15, 2021

![Abstract trading chart](/NT/media/Documents/Market-Depth-Banner.png)

The Market Depth Chart in NinjaTrader is one of the simpler interfaces for viewing order book data. While not often used in futures trading, cryptocurrency traders consider the depth chart a mainstay in determining market sentiment. The Depth Chart is available for monitoring both cryptocurrency and futures instruments. As Micro Bitcoin Futures gain popularity, futures and cryptocurrency traders can both benefit from this tool.

## What are Market Depth Charts?

In trading, Market Depth refers to a market’s ability to sustain large orders without significantly affecting price. Market depth is typically evaluated by looking at the order book of a security. Order books are a list of pending orders to buy or sell at various price levels.

Market Depth Charts display bid (buy) and ask (sell) data for a particular asset at different prices. This visualization of supply and demand turns order book data into a chart that’s both easy and fast to read.

While traditional order book interfaces, such as the T&S Window, and the SuperDOM have their own unique characteristics, the Depth Chart is a fresh take on the age-old question, “What will the markets do next?”.

Though not often used in futures trading, cryptocurrency traders consider the depth chart a mainstay in determining market sentiment.

![depthchart3-768x430.gif](/getattachment/6a48f2d6-9979-4af0-923c-31f7da2daea0/depthchart3-768x430.gif "depthchart3-768x430.gif")

*Use the + and – buttons to zoom in and out on the price scale. Hover over price levels for more details about the book volume and value.*

* Green Side = lowest buy order price
* Red Side = highest sell order price

## Anatomy of the Depth Chart

The interface has three main sections: Quotes, Graph, and Tooltip.

* **Quotes:** The Quotes section displays various market data items at the top of the window. Here you’ll find dynamic values such as bid, ask, and last prices, as well as the session open, high, low, and volume values. Optionally, the Quotes section can be disabled by right-clicking on the Depth Chart and deselecting “Show Quotes”.
* **Graph:** The Graph is the primary display of the Depth Chart. It displays the cumulative buy and sell orders through the full available book, so the vertical value at any point is the result of summing all active orders in the order book to the price value on the horizontal axis. You can use the + and – buttons at the top of the chart to set the zoom level, narrowing in on more immediate prices, or zoom out to look at the big picture.
* **Tooltip:** The Tooltip displays the price level under the cursor, along with the associated cumulative order volume value and the currency cost of that cumulative volume. This can be really handy for understanding just how much market participation is occurring in the markets you’re watching. Take a look at the cash value of just the first few levels of market depth in the E-Mini S&P 500 (ES) during market hours to see for yourself!

![depthchart2-768x430.gif](/getattachment/17c2bbd9-498b-425f-b19a-0267323a700e/depthchart2-768x430.gif "depthchart2-768x430.gif")

*Use the + and – buttons to zoom in and out on the price scale. Hover over price levels for more details about the book volume and value.*

## How to Read a Market Depth Chart

So far, everything we’ve covered could likely be achieved from using another interface. So, the burning question remains: Why use the Depth Chart? Just as with historical data charts, like the [candlestick chart](https://ninjatrader.com/futures/blogs/use-candlestick-patterns-to-uncover-bullish-and-bearish-signals/), viewing order book data in a more visual manner can help traders identify patterns and plan accordingly.

## The Wall

One of the hallmarks of the Depth Chart is the “wall”. Walls can form on the buy or sell sides of the chart, and indicate price levels in which the cumulative bid or ask value increases dramatically. You can’t see this data on a standard price chart, but taking a look at the Depth Chart, you can get a sense of how other market participants are reacting to ever-changing conditions. Walls formed throughout the trading session may later form support or resistance on the price chart.

![DepthCharts.png](/getattachment/a8ed50c0-faa2-4245-a5f5-cc638b6ffddb/DepthCharts.png "DepthCharts.png")

*Here are some examples of different markets on the Depth Chart. There are several walls formed in each chart.*

## Buy Walls

Buy walls represent large numbers of buy orders, typically placed below the current price point. A higher buy wall means more pending buy orders exist at a certain price. High buy walls can also indicate that traders believe an asset will not fall below a certain price.

## Sell Walls

Conversely, sell walls represent a large number of sell orders set above the current price. High sell walls may indicate that traders do not believe an asset will surpass a certain price, while low sell walls indicate the opposite.

## Putting it Together: Market Sentiment & Trend Exhaustion

As you likely noticed from the previous example the Depth Chart may look lopsided to either the buy or sell side. This can be an indication of more bullish market sentiment if the buy-side is larger, or more bearish sentiment if the sell-side is higher. Sometimes a price chart and a depth chart may be out of agreement. For example, the price chart might look bullish, but there is a large accumulation of book data on the sell side of the Depth Chart. This could mean the bullish trend is coming to an end.

![depthchart.png](/getattachment/4a36b5ec-ed02-42c9-8351-bd40ed267da2/depthchart.png "depthchart.png")

## Add a Depth Chart to Your Workspace

You can access NinjaTrader’s Depth Chart window from the “New” menu at the top of the control center. Select “Depth Chart” then add your favorite cryptocurrency and futures instruments to monitor!

Once a new window is open, there are multiple ways to select an instrument in the Depth Chart window. Either right-clicking on the Depth Chart window and select “Instruments”, or simply begin typing the instrument symbol directly on the keyboard.

Using the Depth Chart alongside candlestick charts and other tools in NinjaTrader can help you learn to identify patterns in the markets you trade and help inform and confirm trading decisions. Check out the Depth Chart today, and get a quick view of the order book!

## Get Started with NinjaTrader

NinjaTrader supports more than 500,000 traders worldwide with a powerful and user-friendly trading platform, discount [futures brokerage](https://ninjatrader.com/futures/) and world-class support. NinjaTrader is always free to use for advanced charting & strategy backtesting through an immersive [trading simulator](https://ninjatrader.com/platform/simulation/).

Download NinjaTrader’s award-winning [trading platform](https://ninjatrader.com/platform/) and get started with a free trading demo with real-time market data today!

Previous Post

[Previous Post
Use Candlestick Patterns to Uncover Bullish and Bearish Signals](/Futures/Blogs/Use-Candlestick-Patterns-to-Uncover-Bullish-and-Bearish-Signals)

Next Post

[Relative Strength Index (RSI): Measure Price Movement

Next Post](/Futures/Blogs/Relative-Strength-Index-RSI-Measure-Price-Movement)

Related Posts

* ![](/assets/dist/images/dots/blog-dots.png)

  [### 3 Key Swing Trading Strategies You May Not Have Discovered Yet](/Futures/Blogs/swing-trading-strategies)

  July 02, 2026
* ![](/assets/dist/images/dots/blog-dots.png)

  [### Swing Trading vs. Day Trading Futures: Which Style Fits You?](/Futures/Blogs/swing-trading-vs-day-trading-futures)

  June 30, 2026
* ![](/assets/dist/images/dots/blog-dots.png)

  [### $1K vs $10K vs $100K: How Account Size Changes Your Futures Trading Strategy](/Futures/Blogs/futures-trading-account-size)

  June 30, 2026

Recent Posts

* [How to Choose a Futures Broker: A Decision Framework for Active Traders

  July 07, 2026](/Futures/Blogs/how-to-choose-a-futures-broker)
* [3 Key Swing Trading Strategies You May Not Have Discovered Yet

  July 02, 2026](/Futures/Blogs/swing-trading-strategies)
* [Swing Trading vs. Day Trading Futures: Which Style Fits You?

  June 30, 2026](/Futures/Blogs/swing-trading-vs-day-trading-futures)