# Futures Trading Explained

**Source URL:** https://blog.quantinsti.com/futures-trading/
**Site:** blog.quantinsti.com
**Fetch date:** 2026-07-17 (UTC)
**Word count:** 3989
**Status:** OK

---

By Satyapriya Chaudhari

You must have heard of derivatives in financial markets and maybe, at some point in time, might have also traded/thought of trading in them. In this article, you will learn about the futures market, which is one of the key component of the derivative market. You will learn about the characteristics and parameters of a futures contract. You will also learn about the futures data and the trend following strategy in futures market.

In this blog we will cover the following topics:

- Forwards & Futures

- Characteristics of a Futures Contract

- Parameters of a Futures Contract

- Ticker of a Futures Contract

- Profit & Loss in Futures Trading

- Fetching Futures Data

- Futures Continuation

- Futures Trading Strategy

- Index Futures

- Advantages of Trading in Futures Market

- Limitations of Trading in Futures Market

Derivative markets play a very critical role in maintaining the marketing equilibrium. Derivatives are contracts that derive their values from other financial assets. These assets are referred to as underlying assets. The underlying assets can be anything such as stocks, currency, rates, or agricultural and non-agricultural commodities.

The futures market forms an integral part of the derivative market . Before understanding the future market, it is very important to understand the concept of the forward market.

## Forward & Futures

A forward contract is the simplest form of a derivative. In a forward contract, there are two parties (referred to as a buyer and a seller) who agree to buy and sell a fixed quantity of a specified asset, on a fixed (future) date, at a fixed price.

They enter into a contract that is created on the ISDA master agreement which specifies the obligations from both parties. The price is decided by both parties on the day of the agreement. The main aim of a forward contract is to safeguard both the parties, the buyer and the seller, against future risk.

Let’s understand forward and futures with the help of an example.

Peter has a huge farm and will be harvesting wheat crops in another 3 months. The cost incurred for the harvest is $150 per metric ton. Since the harvest is due in future, there is no guarantee for the price. It can go up, or it can also fall down due to some unforeseen event. If the price of wheat will fall below $150 per metric ton, Peter will have to bear the loss.

Similarly, a food manufacturing giant like Mondelez needs tons of wheat for its products. If it buys wheat at a cost of more than $300 per metric ton, it will have a loss. Just like Peter, Mondelez is also not sure what the price of the wheat will be in three months time.

Thus, to safeguard their interests, they can sign a contract. According to this contract, Peter will sell 100 metric tons of wheat to Mondelez, in three months time, at $200 per metric ton.

In this way, both Peter (seller) and Mondelez (buyer), will protect themselves against any future risk. This is the simplest form of a contract and is called a forward contract.

According to this contract, the seller has to sell a fixed quantity of a decided good to the buyer, at a predetermined price, at a fixed date. The biggest risk involved in a forward contract is the counterparty risk. The contract will be carried forward only by the consent of both parties.

A forward contract cannot be traded further and has to be settled between the seller and the buyer. However, a similar contract can be traded with another party. Though the structure of a forward contract is fairly simple but not fixed. The terms of these contracts will vary from case to case.

Futures contracts are similar to forwards contracts but they are standardised.

Let us now look at the characteristics of a futures contract.

## Characteristics of a Futures Contract

kg-card-begin: markdown

- Standardisation: All the parameters of the futures contract are standardised. For example, in the above case, in the forward contract, the amount of wheat can be 90 metric tons or 110 metric tons based on what is decided by the buyer and the seller. But the quantity cannot be changed in a futures contract, neither can any other parameter.

- Price of the futures contract: Since the futures market forms an integral part of the derivative market, the price of a futures contract is based on the price of the underlying asset. Instead, the price of the futures contract is based on the future price of the underlying asset. It considers the spot price, risk-free rate of return, time to maturity, etc. to determine the final price.

- Time-bound: Every futures contract is time-bound. This time is called the expiry of the futures contract. A futures contract expires on the last Thursday of the month. If last Thursday is a holiday, the contract will expire on the previous working day. A futures contract is available in three different time frames. That means you can trade into futures contracts expiring at three different times in futures. Let’s understand this with an example. Let's suppose you are dealing in the month of January. Near month - A contract that is about to expire in the current month is a near month contract. In January, a near month contract will expire January. Next month - A contract that is about to expire in the next month is a next month contract. In January, next month's contract will expire in February. Far month - A contract that is about to expire in the next to next month is a far month contract. In January, next month's contract will expire in March.

- Regularised: Unlike the forward market, the futures market is highly regulated by a regulatory authority. In the US, the futures market is regulated by the Commodity Futures Trading Commission (CFTC). The role of this authority is to oversee all the futures contracts and ensure that they run smoothly. Thus, the chances of defaulting on a futures contract are removed and each contract is cleared centrally.

- Tradable: Every futures contract is tradable. This means that you do not need to wait till the expiry of the contract to honour it. If you change your mind mid-way, you can further buy/sell the contract (transfer the contract to someone else) in the futures market. Thus you can exit a futures contract at any point in time.

- Settlement: Most of the futures contracts are settled in cash. This means that only the difference amount at the time of settlement has to be paid. This settlement is monitored by the regulating body. You are not bound to make or take delivery of the good(s) traded.

kg-card-end: markdown

These are the characteristics that make a futures contract unique. This video explains it:

kg-card-begin: html

kg-card-end: html

## Parameters of a Futures Contract

kg-card-begin: markdown

- Lot size: The lot size of a futures contract specifies the minimum quantity (or a multiple of this quantity) of the asset that you will have to trade. This is a predetermined number. For example, suppose you want to trade in the futures of Apple’s shares. The lot size for Apple’s future is 100. This means that you can buy a minimum of 125 shares. Or you can buy in multiples of 100. Such as 2 lots are 200 shares, and so on. The lot size will vary from contract to contract.

- Contract value: Let's continue with the above example. You want to buy 1 lot of Apple’s shares and the Apple’s shares are trading at $125 per share. Thus the contract value of Apple’s future will be equal to $12,500 ($125 * 100), i.e. the product of the lot size and the price.

- Margin: You do not need the entire contract value to get into this trade. Instead, you need to deposit a portion of the contract value with the broker to sign the contract. This portion is called the margin. The margin is blocked when you enter a futures contract and it is unblocked when you exit it.

- Expiry date: As you already know that a futures contract is time-bound, every futures contract comes with an expiry date. This is the date after which the contract ceases to exist. You need to close your position on or before the expiry date, or the regulatory authority does it for you. On the date of expiry of a contract, a new contract is introduced. You might want to note here that for commodity futures, you need to be out before the first notice date (usually one month prior to the expiry date).

kg-card-end: markdown

## Ticker of a Futures Contract

Like every other asset/commodity you trade in the financial market, every futures contract also has a ticker. The ticker of a futures contract is a combination of the root symbol of the underlying asset, the month of expiry and the year of expiry.

kg-card-begin: markdown

- Root Symbol - Every asset has a root symbol which is usually two or three letters designating the market. For example, GC is the root symbol for the Comex Gold market, SI is the root symbol for the Comex silver market.

- Month - Now when you buy a Gold contract, you don't simply buy a gold contract, but you buy the contract for a particular month. For example, you by January 2021 Gold contract. To represent the month in the contract, a single letter is used. This is common across different futures markets. For example, January is represented by the letter F. The table below shows the letters associated with different months.

Month Code Month Code January F July N February G August Q March H September U April J October V May K November X June M December Z

- Year - This is represented by a single or two-digit number. While using a single digit, it represents the year of the current decade.

kg-card-end: markdown

Thus the ticker of the January 2021 Gold contract will be GCF1 .

## Profit & Loss in Futures Trading

As you have already seen, in order to enter a futures contract, you only require a certain margin in your account, this means that the futures market uses high leverage. The higher the leverage, the higher is the risk, and the higher is the potential profit.

For example, let’s consider you buy one lot of Apple’s shares with a required margin of $1000. Remember, for Apple’s futures we had seen the example where:

- lot size: 100

- price: $125

- contract value: $125 * 100 = $12,500

Now there can be three scenarios.

- Price goes up: Apple’s share price goes above $125, to $140. In this case, the value of 100 shares at the expiry of the contract will be $14,000. Hence, at the closure of the contract, the difference that is paid to you is $1500 ($14,000 - $12,500). Thus by blocking $1000, you made a profit of $1,500.

- Price goes down: Apple’s share price falls below $125, to $110. In this case, the value of 100 shares at the expiry of the contract will be $11,000. Still, you are obligated to buy 100 Apple’s shares at $12,500. Thus, this will result in a loss of $1,500.

- Price remains the same: Under this scenario, you will neither make any profit, nor a loss. There will be no impact on either the buyer’s or the seller’s portfolio under this condition.

Therefore, you can say that the futures market is a Zero-Sum Game. This means that the pot in the future market never grows. Money simply transfers from the losers to the winners. The profit made by the buyer is equivalent to the loss made by the seller and vice-versa.

This video explains how profit and loss are calculated in futures trading:

kg-card-begin: html

kg-card-end: html

## Fetching Futures Data

You can access the historical data for futures contracts from several web sources and Python APIs. Refer to the below articles and explore the Python codes on how to fetch the futures data.

- How to Get Historical Market Data Through Python API

- Stock Market Data And Analysis In Python

## Futures Continuation

As you have seen earlier in this post, a futures contract ceases to exist after its expiry date. Hence the data available for any contract is for a very short span of time (owing to the limited lifespan of the futures contract) , mostly one month or so. This results in one of the trickiest issues while dealing with futures data.

The issue is the lack of a long time series to analyse the market. A very simple task like analysing the one year moving average of the assets doesn’t remain so simple anymore. You need to first manipulate the data in order to achieve the continuous time series for a longer period. This is achieved by continuations.

A continuation is a time series obtained by stitching together multiple individual series.

A future continuation is a time series obtained by stitching together multiple individual futures contracts.

A very simple solution that might occur to you can be appending the monthly data at the end of the previous month. However, this can lead to disastrous results. If you will analyse the data, you will see that the price at which an asset/commodity is trading in the current month is very different from the price at which it was trading in the previous month.

For example, look at the graph below. You can see the sudden spike in the series. This is because of the difference in the price between two consecutive months. It doesn’t mean that the actual price has seen the spike.

Fig. 1: Appending monthly data

Thus, in order to stitch individual contracts, you need to be careful to avoid such errors. There are different ways in which you can stitch the contracts. Let’s look at a couple of them.

kg-card-begin: markdown

- Additive adjustment - Under this approach, you adjust the previous contract back in time, such that the last value of the previous contract matches the first value of the current contract. This is done by simply adding or subtracting a value to the entire series of the previous contract. The value is equal to the difference between the last value of the previous contract and the first value of the current contract.

$$\text{Adjustment factor = Second contract's first price − First contract's last price}$$

The major drawback of this approach is that it can leave us with negative values after certain adjustments.

- Proportional Adjustment - Under this approach, the adjustment factor is not the difference but a percentage. Here the entire series is shifted by a percentage.

$$\text{Adjustment factor = Second contract's first price/First contract's last price}$$

Instead of adding or subtracting the factor, the entire time series is multiplied by this factor. The drawback is that the absolute price level no longer reflects the real market price after the adjustment.

kg-card-end: markdown

Once you obtain a futures continuation series by applying the above adjustment, you can perform your analysis on the time series and create your trading strategy. Let’s see a simple strategy that has been used for decades in the futures market, the trend following strategy.

## Futures Trading Strategy

The inception of trend following futures strategies dates back to the 1980s and today there is an estimate of half a trillion dollars being managed through this strategy. The trend-following strategy is based on the principle that the prices often continue to move in the same direction for a sustained period of time.

The trend-following strategy does not aim to buy at the bottom and sell at the top. But it tries to capture the change in the trend. It enters a position when there is a change in the trend. This doesn’t mean that an entry signal is generated the moment there is a change, but the model waits for a certain time for the change to qualify as a trend.

The model then stays in the position as long as the trend lasts. The same logic holds while exiting the trade. An exit signal is not generated immediately after the change, but after waiting for a certain time. If the trend is bullish, you go long and if the trend is bearish, you go short.

An important point to note here is that most of the trend following trades end up losing money. Even for a successful trend following model, the winning trades are only between 30% and 40%. However, when done right, failed trades will have small losses and there would be few big wins that would make up for them, or even go way beyond them.

In order to keep the chances of profit high, it is necessary to ensure participation in all trades, and never miss out on a single one. Since the number of winning trades is less, and a big win is even lesser, in order to not miss this huge win that will make up for the other losses, it is very important to participate in all trades.

This video explains the principles of trend following strategy.

kg-card-begin: html

kg-card-end: html

Futures markets have a diverse set of assets that you can trade-in. The underlying asset of a futures contract can be anything such as stocks, currency, rates, or agricultural and non-agricultural commodities.

Some of the most liquid assets in the futures market are crude oil, natural gas, corn, soybeans and wheat. The below table shows the assets that can be traded in the futures market.

kg-card-begin: html

Currencies Agricultural Non-agricultural Equities Rates AUD Feeder wheat Crude oil S&P 500 Eurodollar GBP Corn Gold Nasdaq 2y treasury CAD Cotton Copper Nikkei 225 5y treasury EUR Feeder cattle Heating oil Dow Jones 10y treasury US Dollar Index Arabica coffee Gas oil VIX 30y bond JPY Robusta coffee Palladium NZD #5 sugar Platinum CHF #11 sugar Gasoline Oats Silver Soybeans Bitcoin Soybean meal Lumber Rough rice Cocoa

kg-card-end: html

Let us see how this strategy performs on a non-agricultural commodity, Palladium. The strategy is backtested on the historical data of Palladium from 1977 to 2020.

The step is to define the trend filter. And to do that we have considered 40-day and 80-day exponential moving averages.

The trend is defined as below:

- The trend is deemed bullish if the 40-day exponential moving average crosses above the 80-day average.

- The trend is considered to be bearish if the 40-day exponential moving average crosses below the 80-day average.

The below image shows the price trend.

Fig. 2: Price trend of Palladium

After the trend is defined, the rules to open a position is defined as below:

- Open a new long position if the trend filter shows a bullish trend and the price breaks the 50-day high.

- Conversely, open a new short position if the trend filter is bearish, and the price breaks the 50-day low.

Fig. 3: Long and shorty entry signal

Next, let us define the rules to exit the positions. To define the exit rules, we need to calculate the pullback value. The pullback value is calculated as below:

kg-card-begin: markdown

- For a long position , we calculate the pullback as per the below formula. $$Pullback=\frac{Close~price-Rolling~highest~price}{Volatility~of~the~asset}$$ A long exit signal is generated if the pullback is less than -3.

- For a short position , we calculate the pullback as per the below formula. $$Pullback=\frac{Close~price-Rolling~lowest~price}{Volatility~of~the~asset}$$ A short exit signal is generated if the pullback is greater than 3.

kg-card-end: markdown

The image below shows the long and short positions.

Fig. 4: Long and short positions

In the graphs above, the green shaded region is the period where a long position is held. Similarly, the red shaded region is the period where a short position is held.

The strategy returns were observed as follows:

Fig. 5: Cumulative returns

## Index Futures

A futures contract can not only be used for trading the commodities as stated above, but you can also trade in stock indexes using a particular type of futures contract. This is called Index Futures.

As per an index futures contract, you are obliged to trade a specific stock index at a predetermined price at a fixed date in future. Explore the blog on Index Futures to dive deeper into concepts.

## Advantages of Trading in Futures Market

Let us now look at some advantages of trading in the futures market.

- High leverage: Trading in the futures market is highly leveraged. As you would have read earlier, you don’t need the entire contract value to get into a futures trade. All you need is the margin amount, which is only a fraction of the contract value. This means that you are exposed to a much greater value of the underlying asset when dealing in the futures market, as compared to when dealing with the actual asset. This, in return, can allow you to make faster money.

- High liquidity: There are a huge number of buyers and sellers always available in the market ensuring that you can enter into a futures contract almost any time you desire. The futures market is highly liquid.

- Low commission and execution cost: The commission in the futures market is as low as 0.5% and is charged when the position is closed.

- No insider trading: It is very difficult to trade in the futures market using any insider information. Thus, the futures market has equal opportunities for all.

- Risk management tool: Getting into a futures contract safeguards you against price fluctuation in future.

## Limitations of Trading in Futures Market

Along with the advantages, there are some disadvantages too when trading in the futures market. Let us look at some of them.

- High fluctuation: High leverage in futures trading is a double-edged sword. Along with providing great exposure, it also results in high fluctuations in the future price. If you fail to predict the correct direction of the price change, you can end up losing all the margin.

- Expiration dates: Futures contract comes with an expiry date. As the contract nears its expiration date, it becomes less sought for. This might result in the contract expiring worthless.

- Complicated for a new trader: A futures contract comes with different specifications and parameters for different assets. Thus making it difficult for a new trader to understand.

### Other References

- Analysing the Assets

- Futures Trading: Quantitative Analysis without Historical Pricing

## Conclusion

The futures market is an integral part of the derivative financial market. It is a zero-sum game that maintains the equilibrium in the market. If you can speculate the future price of an asset, you can enter a futures contract.

The futures contracts are standardised, regulated and cleared by central bodies, thus avoiding the risks in the forward market. The futures market allows you to trade in any commodity, without worrying about the taking or making the actual delivery of the good(s).

In this blog, we covered the characteristics and parameters of futures markets. We also saw the trend following strategy. Explore our Futures Trading Course to dive deeper into the concept and gain knowledge on how to trade systematically in the futures market.

kg-card-begin: html

Disclaimer: All investments and trading in the stock market involve risk. Any decisions to place trades in the financial markets, including trading in stock or options or other financial instruments is a personal decision that should only be made after thorough research, including a personal risk and financial assessment and the engagement of professional assistance to the extent you believe necessary. The trading strategies or related information mentioned in this article is for informational purposes only.

kg-card-end: html

start: CTA

Download EPAT Brochure

end: CTA

start: .share-buttons

end: .share-buttons