# Time and Sales Data for Futures Traders in Real Time
**Date:** June 13, 2024
**Source:** https://ninjatrader.com/futures/blogs/time-and-sales-data-for-futures-traders-in-real-time/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Time and Sales Data for Futures Traders in Real Time

# Time and Sales Data for Futures Traders in Real Time

By
NinjaTrader Team

June 13, 2024

In the early days of trading, the ticker tape machine was the heartbeat of the financial markets, spewing out a continuous stream of stock quotes and trade data on a narrow paper tape. This mechanical marvel revolutionized the dissemination of real-time market information, allowing traders and investors to stay abreast of the latest price movements.

Today, the Time and Sales (T&S) window in the NinjaTrader platform provides an instantaneous view of every trade executed across the futures exchanges. The T&S window displays key information for all trades, including the trade price, volume, and time. The NinjaTrader Time and Sales window is also highly customizable and allows traders to view and filter data based on trade size or whether a trade occurred on the bid or the ask (offer).

Watch this video for an overview of the capabilities of the NinjaTrader Desktop T&S window, which displays every trade for a market in real time. Learn how to customize the window to filter on large trades and sound alerts, and explore the new T&S aggregation feature.

&amp;lt;span class="fr-mk" style="display: none;"&amp;gt;&amp;nbsp;&amp;lt;/span&amp;gt;

**The T&S window can be accessed from the Control Center.**

* Navigate to the upper right corner and click on the **New**tab.
* A drop-down menu will appear and toward the bottom you will see T&S.
* Click on the T&S option and the T&S window will open. (Figure-1)

**![T&S Window NinjaTrader](/getmedia/58166ae2-685a-4e59-847d-7c600789907d/TS1.png)*Figure 1: Opening a new Time and Sales window from the Desktop Control Center.***

## Time and Sales Window

Let’s navigate the basic T&S window. In this default view, there are three columns: (left) the time the trade occurs, (center) the price of the trade, and (right) the number of contracts traded (lot size), where each row is a single unique trade. (Figure-2)

![Time and Sales window for crude oil futures](/getmedia/6bbce6e4-3de0-4201-bd2f-05ac3a5376a1/TS2.png)*Figure 2: The NinjaTrader Desktop Time and Sales window for crude oil futures.*

### How to Select an Instrument to Track in the T&S Window

Traders can analyze any futures markets they like. To select or change the market, simply right-click anywhere in the T&S window. A pop-up menu will appear. Navigate to the **Instruments**menu option and select/apply the instrument (market) you want to display. (Figure-3)

![Instrument tracker in the T&S window](/getmedia/ea6d7df9-d8e7-42ed-8cf8-8691b6a6a223/TS3.png)*Figure 3: Selecting an instrument (market) to display in the Time and Sales window.*

### How to Color Code Trades on the Bid vs Trades on the Ask in the T&S Window

The NinjaTrader T&S window allows traders to color-code trades based on whether they were made on the bid or ask (offer). An imbalance between trades made on the bids versus asks can give traders cues regarding market direction.

As buyers and sellers place orders, an imbalance of buy and sell orders can move the market higher or lower. For example, as more buy than sell orders hit the ask (or offer), prices may rise, and if more sell than buy orders hit the bid, prices may decline.

The spread between the highest bid and the lowest ask (offer) is called the bid/ask spread. The highest bid is the best price a trader is willing to buy a [futures contract](/futures/futures-contracts/) (when you are a seller), and the lowest ask is the best price a trader is willing to sell a futures contract (when you are a buyer). In liquid markets, the bid/ask spread is generally “tight,” meaning the bid and offer are separated by the smallest tradable price increment, which is called a minimum move or tick.

**Color-coding**

The NinjaTrader T&S window allows traders to color-code each row to provide a visual tool to assess imbalances between trades on the bid versus trades on the offer. To adjust the colors, right-click within the T&S window and select Properties. (Figure-4)

![Color-coding properties](/getmedia/502e5d28-c1af-479e-9883-cf977551049c/TS4.png)*Figure 4: Color-coding properties for trades going off at the bid or ask.*

Continuous redstreaming prices in the Time and Sales data suggest more trades are occurring on the bid, resulting in downward price pressure. Greenstreaming prices in the Time and Sales data suggest more trades are occurring on the ask (offer).

### How to Apply a Size Filter in the T&S Window

The NinjaTrader T&S window can be further customized by modifying the trade size filter to allow you to only see larger trades that meet an input threshold. The default filter size is set to zero so every trade will be displayed.

Some traders may prefer to filter the Time and Sales data to only see trades of a certain size or greater. In this example, the Size Filter is set to 3 (Figure-5). This will display trades that are 3 contracts or greater. Trades of less than 3 contracts will not be displayed. To adjust the size filter, right-click within the T&S window and select Properties.

![Minimum trade size filter](/getmedia/3b5db111-05a0-40ce-bebd-e3098a8909db/TS5b.png)*Figure 5: Setting the minimum trade size filter to 3 contracts.*

Once the filter is applied, the T&S window will look something like this. Notice the size column is not displaying trades with trade volume smaller than 3 contracts. (Figure-6)

![Minimum trade size filter](/getmedia/3fb4e494-3660-45b5-94e7-3efacf6bf618/TS6.png)*Figure 6: Displaying only the minimum trade size filter of 3 contracts.*

### How to Apply Volume Aggregation by Time in the T&S Window

Although the T&S window typically displays every trade that takes place, in some fast markets, this can create a data stream that is too quick to follow effectively. In these markets, NinjaTrader allows traders to aggregate the Time and Sales data by combining trades by time segments, in number of seconds, to better represent the trade stream. (Figure-7)

![Aggregation of trades](/getmedia/eb790d80-14bd-4d3f-95cb-39a30e8939cc/TS7.png)*Figure 7: Setting the aggregation of trades to 10 seconds.*

Notice the third column shows larger trades sizes since this is aggregated data that is collected every 10 seconds. The “H” on the left signifies a new session high. (Figure-8)

![Aggregation of trade volume](/getmedia/0efd1604-b7ef-4ece-b562-b71216d3dc5d/TS8.png)*Figure 8: Displaying the aggregation of trade volume in 10-second segments/rows.*

### How to Set Up Block Alerts in the T&S Window

Block alerts are a way to highlight trades with large contract volume sizes within the data Time and Sales data stream. By selecting a minimum block size— in this example, 5 contracts or more (Figure-9)—traders can also set an audible alert for predefined block size trades. Traders can set the block size at any level according to their preference.

![Properties for displaying trade alerts](/getmedia/5a919049-6d75-46d6-97d5-3afef2cbabb9/TS9.png)*Figure 9: Setting the properties for displaying a visual block trade alerts over 5 contracts.*

The T&S window will add a fourth column to the window populated by the capital letter “B” whenever these large trades occur. (Figure-10)

![Visual block trade alert](/getmedia/049c4371-c4f0-4053-8fe5-9e9b1da16b02/TS10.png)*Figure 10: Showing the visual block trade alert with a "B" in the right column.*

The Time and Sales window is a powerful real-time trading and data tool available in the NinjaTrader Desktop trading platform. It provides the ultimate order flow analysis, which can help traders spot supply and demand imbalances and short-term bursts in buying and selling pressure. The T&S window also demonstrates the price and trade transparency provided by the futures market exchanges, which helps keep everyone on a fair and level playing field.

### Use the Time & Sales Window to Customize Your Futures Trading

Regardless of your experience level or the markets you trade, having a [customized workspace](/futures/blogs/how-to-build-a-workspace-in-the-ninjatrader-desktop-platform/) can help you trade futures in a way that works for you. This can make it easier and more efficient to place trades, monitor your trading, track market prices, and analyze the markets—which are all vital to every trader’s success.

### Get Live Pro Tips and Analysis Every Trading Day

Join our experts as they talk through the market open and close, [technical analysis](/learn/technical-analysis/), and our award-winning platform during our [daily livestream](/learn/livestreams/). Watch live here or catch what you missed on our [YouTube channel](https://www.youtube.com/channel/UCUXNA8JJOiMrqhB8NVNgClA).

### Trade Futures With the Industry Leader

Still haven’t signed up for your free NinjaTrader account? Get in the game today with a [14-day trial of live simulated futures trading](https://account.ninjatrader.com/register?utm_campaign=org-nt-us&utm_source=website&utm_medium=website&utm_content=ninjatrader_blog&_gl=1*qb94vc*_gcl_aw*R0NMLjE3MDczMzQ5MDUuQ2p3S0NBaUE4WXl1QmhCU0Vpd0E1UjMtRTROakVrVmJRUmtOeC1aOTgzTFZBRTVOQURhZWJBTTlxWEczYzJzakZFU3MzRHFNQmkwc1F4b0NkSHNRQXZEX0J3RQ..*_gcl_au*ODI2NDEwNjU5LjE3MTEzNzg2MTE.).

Previous Post

[Previous Post
5 Essential Indicators for Swing Trading Futures](/Futures/Blogs/5-Essential-Tools-for-Swing-Trading-Futures)

Next Post

[Futures Trading Outlook: E-Mini S&P 500 Index Trending Bullish

Next Post](/Futures/Blogs/Futures-Trading-Outlook-E-Mini-S-P-500-Index-Trending-Bullish)

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