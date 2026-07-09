# Using Heikin Ashi candlestick chart patterns in futures trading
**Date:** June 26, 2024
**Source:** https://ninjatrader.com/futures/blogs/heikin-ashi-candlestick-charts-explained/

---

* [Home](/homepage/)
* [Futures](/futures/)
* [Blogs](/futures/blogs/)
* Heikin Ashi Candlestick Charts Explained

# Using Heikin Ashi candlestick chart patterns in futures trading

By
NinjaTrader Team

June 26, 2024

Japanese candlestick charts were revolutionary when introduced to modern-day traders in the 1990s. They brought dimension, color, and unique patterns that helped traders understand market dynamics better than traditional open-high-low-close (OHLC) charts. However, in the ever-evolving landscape of market analysis, a complementary candlestick charting technique, Heikin Ashi, has become popular with traders.

So, what does Heikin Ashi mean?

In Japanese, “Heikin” translates to “average” and “Ashi” translates to “foot,” but when paired, “Heikin Ashi” translates to “pace.” This term is used because of the smoothing nature of the Heikin Ashi chart, as it removes some of the choppiness and noise often reflected in standard [candlestick charts](/futures/blogs/trading-with-candlestick-patterns/).

Watch this video to learn:

* Key concepts around the Heikin Ashi style of candlestick charting
* How this unique approach can help traders eliminate price noise and better identify price trends

## What are Heikin Ashi Bars?

Heikin Ashi bars are an innovative indicator that replaces the traditional candlestick chart with a unique approach to identifying market trends. Heikin Ashi candlesticks are a variation of traditional Japanese candlestick charts, with slightly different bar building rules that better highlight current trends.

Unlike regular candlesticks—which utilize only the open, high, low, and close prices of a specific bar—Heikin Ashi bars are calculated using a formula that also incorporates the previous bar's data. This approach results in a smoother, more filtered representation of price action, reducing the impact of short-term fluctuations and highlighting the underlying trend more effectively. (Figure-1)

![Heikin Ashi candlestick](/getmedia/76ad40c8-2fcc-4817-8654-c6c2d8e12029/Heikin-Ashi-1.png)*Figure-1: The Heikin Ashi candlestick chart compared to a standard candlestick chart. The red and green arrows highlight periods of strong trend. The grey arrow shows an area of smaller bodies and the possible end of the downtrend.*

## How are Heikin Ashi Bars Calculated?

Heikin Ashi bars are calculated by averaging the open and close prices of the current period with the open price of the previous period. This calculation creates a new open price, which is then combined with the high and low prices of the current bar to form the Heikin Ashi bar. The resulting bar is colored based on the relationship between the open and close prices, with green bars indicating an uptrend and red bars signifying a downtrend.

* HA-close: average of the current open, high, low, and close
* HA-open: average of the previous HA-open and previous HA-close
* HA-high: maximum of the close, HA-open, and HA-close
* HA-low: minimum of the low, HA-open, and HA-close

## How are Heikin Ashi Bars used?

One trend trading element that can challenge traders is the presence of reversal colored candlestick bars during a strong trend, which can fool traders into exiting a trade prematurely. These opposite-colored candles seem to signify a trend reversal but can often be a false signal, with the trend continuing.

Heikin Ashi bars take more data points into account and will often recolor a bar based on the current trend. The visual change allows traders to stay focused on the trend and stay in the trend trade through these short-term periods of price fluctuation and volatility.   
   
There are five modes that the Heikin Ashi chart can identify:

1. Uptrend
2. Strong uptrend
3. Downtrend
4. Strong downtrend
5. Trend change or reversal

The same color scheme used when interpreting standard candlestick charts apply to Heikin Ashi charts as well (green = uptrend, red = downtrend). In a standard uptrend, you will see multiple green candles in a row; in a strong uptrend, there are no wicks on the bottom side of the green candle, and these candles tend to be longer-bodied green candles. In a standard downtrend, you will see multiple red candles in a row; in a strong downtrend, there are no wicks on the bottom side of the green candle, and these candles tend to be longer-bodied green candles.    
   
The end of the trend or a trend reversal often displays as a change in color on smaller-bodied candles with wicks on both sides. Traders should use Heikin Ashi in conjunction with other technical indicators to help better identify these potential reversals.

## What to look out for when using heikin ashi bars

Since the Heikin Ashi chart uses data points that are either averages (HA-open and HA-close) or extremes of compared values (HA-high and HA-low), the charts may not reflect the actual bar prices, especially the closing bar price—so it’s advisable to use another type of chart that displays the actual bar prices when trading.

### We’re Live Every Trading Day

Prep for the trading day ahead, analyze the markets in real time, and explore our award-winning platform during our [daily livestream](/learn/livestreams/). Watch live here or catch what you missed on our [YouTube channel](https://www.youtube.com/channel/UCUXNA8JJOiMrqhB8NVNgClA).

### Trade Futures with NinjaTrader

Haven't signed up for your free NinjaTrader account yet? Get started today with a [14-day trial of live simulated futures trading](https://account.ninjatrader.com/register?utm_campaign=org-nt-us&utm_source=website&utm_medium=website&utm_content=ninjatrader_blog&_gl=1*19kz1gv*_gcl_aw*R0NMLjE3MTg4MTI1NTAuQ2p3S0NBandnOHF6QmhBb0Vpd0FXYWdMckQxM01MQWlwTncxb1d1Z2p5c09VYmZ2R2hqT09jRkRQM3N0MjcyU3B2T1dnVEh5Z0ExdmxCb0NPUmtRQXZEX0J3RQ..*_gcl_au*MTU3MDU0ODc0MC4xNzE5MjM0MDM0).

Previous Post

[Previous Post
Trading Psychology: 5 Reasons Traders Struggle with Losses](/Futures/Blogs/Trading-Psychology-5-Reasons-Traders-Struggle-with-Losses)

Next Post

[Futures Trading Outlook: Silver Futures Turn Down for the Week

Next Post](/Futures/Blogs/Futures-Trading-Outlook-062824)

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