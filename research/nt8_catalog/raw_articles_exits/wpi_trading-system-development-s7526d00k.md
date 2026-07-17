# Trading System Development (WPI thesis)

**Source URL:** https://digital.wpi.edu/downloads/s7526d00k
**Site:** digital.wpi.edu
**Fetch date:** 2026-07-17 (UTC)
**Word count:** 20833
**Status:** OK
**pass:** 2

---

Trading System Development  
  
  
 An Interactive Qualifying Project submitted to the Faculty of  
 WORCESTER POLYTECHNIC INSTITUTE   
 in partial fulfillment of the requirements for the  
 Degree of Bachelor of Science  
  
 By:  
 Stephen Andrews  
 John Bieber  
 Ted Bieber  
 Grant Espe  
  
 Advisors:  
 Professor Hossein Hakim  
 Professor Michael Radzicki  
 Date: 5/13/2018  
  
 Report submitted to:  
 Professor Hossein Hakim  
 Worcester Polytechnic Institute  
     
  
 This report represents work of WPI undergraduate students submitted to the faculty as evidence  
 of a degree requirement. WPI routinely publishes these reports on its web site without editorial  
 or peer review. For more information about the projects program at WPI, see  
 http://www.wpi.edu/Academics/Projects.   
  

 
 Abstract  
 The goal of this project was to develop a system of automated trading systems that  
 achieves a greater return than comparable alternatives, such as the S&P 500. Over the past year,  
 the inflation rate in the US was 2.1%, meaning that the buying power of $10,000 would decrease  
 by $210. In order to counteract this, it is beneficial to identify a reliable investment vehicle to  
 generate consistent returns. This team developed a system which achieved a 19.73% return on  
 $400,000 initial capital when trading from April 7th, 2017 to April 4th, 2018, beating inflation  
 by 17.63% and the S&P 500 by 9.3%. Trading on a mix of stocks and forex, the system made  
 634 trades and earned $0.27 for every dollar it risked, which compounds to $172.63 for every  
 year that it was in the market. Its system quality, which is a metric for determining the overall  
 quality and versatility of a trading system, was 2.85. The sharpe ratio for this system was 0.79;  
 this is a measure of a system’s risk-adjusted returns. The overall system of systems achieved a  
 higher system quality than the majority of its underlying systems; therefore this team  
 recommends that a combined system of systems should be used over a single system when  
 trading on the equity markets.  
   
 2  

 
 Table of Contents  
 Abstract 2  
 List of Figures 6  
 Chapter 1: Introduction 7  
 1.1 Project Description 9  
 Chapter 2: Background 11  
 2.1 Asset Classes 11  
 2.1.1 Stocks 11  
 2.1.2 Mutual Funds 12  
 2.1.3 ETFs 12  
 2.1.4 Fixed Income 13  
 2.1.5 Currency 13  
 2.1.6 Commodities 14  
 2.2 Market Types 14  
 2.3 The Stock Market 17  
 2.3.1 Exchanges 17  
 2.3.2 Indexes 18  
 2.3.3 Market Sentiment 19  
 2.3.4 Trend and Cycle 19  
 2.3.5 Sector Rotation 20  
 2.4 Difference Between Trading and Investing 20  
 2.5 Efficient Market Hypothesis 21  
 2.6 Fundamental vs Technical Analysis 22  
 2.7 Portfolio Management 23  
 2.8 Trading Platforms and Brokerage Accounts 24  
 2.8.1 TradeStation 24  
 2.8.2 Other Alternatives 25  
 2.9 Common Analysis Tools on Trading Platforms 26  
 2.9.1 Time Frames 27  
 2.9.2 Indicators 28  
 2.9.3 Optimization 31  
 2.9.4 Walk-Forward Analysis 32  
 2.9.5 Expectancy, Expectunity, System Quality, and Sharpe Ratio 33  
 2.9.6 Stock Screeners 35  
 2.10 Order Types 37  
 3  

 
 2.10.1 Market Order 37  
 2.10.2 Limit Order 37  
 2.10.3 Stop Order 38  
 2.10.4 Other Types of Orders 38  
 2.11 Buy & Hold 39  
 2.12 Trading Systems 39  
 2.13 Manual Trading 40  
 2.14 Automated Trading 41  
 2.15 Monte Carlo analysis 42  
 Chapter 3: Methodology 43  
 3.1 Research 43  
 3.2 Systematically Beating the Market 44  
 3.3 Finding an Edge 44  
 Chapter 4: Trading System Development 46  
 4.1 Pre-Market Gaps 46  
 4.1.1 Hypothesis 46  
 4.1.2 Methodology 47  
 4.1.3 Entry/Exit Conditions 49  
 4.1.4 Results 51  
 4.2 Whole Breakouts 53  
 4.2.1 Hypothesis 53  
 4.2.2 Methodology 54  
 4.2.3 Entry/Exit Conditions 56  
 4.2.4 Results 57  
 4.3 Kaufman Efficiency Day Trader 58  
 4.3.1 Hypothesis 58  
 4.3.2 Methodology 59  
 4.3.3 Entry/Exit Conditions 59  
 4.3.4 Results 60  
 4.4 Forex Stop and Reverse 64  
 4.4.1 Hypothesis 64  
 4.4.2 Methodology 64  
 4.4.3 Entry/Exit Conditions 66  
 4.4.4 Results 67  
 Chapter 5: Results 69  
 5.1 Pre-Market Gaps 71  
 4  

 
 5.2 Whole Breakouts 72  
 5.3 Kaufman Efficiency Day Trading 73  
 5.4 Forex Stop and Reverse 74  
 5.5 System of Systems 75  
 Chapter 6: Conclusions 78  
 Chapter 7: Recommendations for Future Projects 79  
 7.1 Adhering to a Methodology 79  
 7.2 Identifying Noise 79  
 References 81  
 Appendix A 85  
 Whole Breakouts Code 85  
 Appendix B 89  
 Kaufman Efficiency Day Trader Code 89  
 Appendix C 92  
 Pre-Market Gaps Code 92  
 Appendix D 95  
 Forex Stop and Reverse Code - Keltner Floors 95  
 Forex Stop and Reverse Code - Keltner Ceilings 96  
 Appendix E 98  
 Excerpts from Stephen Andrew’s Trading Journal 98  
 Excerpts from John Bieber’s Trading Journal 99  
 Excerpts from Theodore Bieber’s Trading Journal 100  
 Excerpts from Grant Espe’s Trading Journal 100  
   
 5  

 
 List of Figures  
 Figure 1.1: Purchasing power over 15 year period adjusted for inflation  
 Figure 1.2: Value of $10,000 investment over 15 year period adjusted for inflation  
 Figure 2.1: TradeStation environment with chart analysis window of $AAPL  
 Figure 2.2: Yahoo Finance screening for mid-cap technology stocks   
 Figure 2.3: Example of Monte Carlo analysis on an unreliable system  
 Figure 4.1: Pre-market screening on Finviz  
 Figure 4.2: Entries prior to VWAP filter  
 Figure 4.3: Entries after applying VWAP filter  
 Figure 4.4: Pre-market gap strategy with 15% available capital per position  
 Figure 4.5: Pre-market gap strategy with 50% available capital per position  
 Figure 4.6: Whole Breakouts Equity Curve  
 Figure 4.7: Whole Breakouts Monte Carlo analysis  
 Figure 4.8: Detailed Equity Curve of Kaufman Efficiency System  
 Figure 4.9: SPY symbol over the past year  
 Figure 4.10: Examples of successful trades for Kaufman Efficiency System  
 Figure 4.11: Example of drawdown for Kaufman Efficiency System  
 Figure 4.12: Equity Curve - EURUSED 60 min bars  
 Figure 4.13 Example positions - EURUSD 60 min bars  
 Figure 5.1: System of systems equity curve  
 Figure 5.2: Monte Carlo analysis of the system of systems  
  
  
  
   
 6  

 
 Chapter 1: Introduction  
 There are many compelling reasons to invest money in the financial markets. According  
 to numerous studies by the American Psychological Association, money and finances have been  
 the leading cause of stress for the average American since the studies began in 2007. ​ [ ] ​  For  1
 individuals and families, it is important to have some level of savings to offset the cost of living  
 and manage unexpected bills. However, excess money sitting idle in a checking account provides  
 no long term financial benefit to an individual. In fact, the level of purchasing power drastically  
 declines over time when money is not generating interest due to inflation. An account with a  
 starting balance of $10,000 in 2002 would be devalued to a purchasing power of $7,164 after 15  
 years of inactivity as shown in Figure 1.1.  
  
 Figure 1.1 Purchasing power over 15 year period adjusted for inflation  
 1  ​ “American Psychological Association Survey Shows Money Stress Weighing on Americans’ Health Nationwide,”  
 American Psychological Association ​ , 04-Feb-2015. [Online]. Available:  
 http://www.apa.org/news/press/releases/2015/02/money-stress.aspx. [Accessed: 25-Apr-2018].   
 7  

 
 For these reasons, it should become obvious that employing excess money in some  
 investment vehicle is a necessity for financial longevity. It is not uncommon for individuals to  
 avoid investing in the financial markets due to the notion that market investments bear higher  
 risk. While this is true, investing in the financial markets with a solid understanding of market  
 dynamics and maintaining realistic earnings expectations may dramatically improve an  
 individual's net worth over an extended period of time. Figure 1.2 illustrates an example of how  
 a more traditional investment vehicle compares to an investment in the financial markets.  
 Figure 1.2 Value of $10,000 investment over 15 year period adjusted for inflation  
  
 Due to a lack of historical data for interest rates of savings accounts, a one month  
 treasury bond has been used to model the returns of a savings account. Over the span of a 15 year  
 period, a starting capital of $10,000 invested in the S&P 500 index fund would yield $22,956  
 after inflation. Conversely, a starting capital of $10,000 invested in a savings account would  
 yield $8,674 after inflation.   
 8  

 
 The example above illustrates the advantages of exploring the financial markets as a  
 potential investment vehicle. However, investing in an index fund, such as the S&P 500, is only  
 one of many ways to invest in the financial markets. Other options include developing a portfolio  
 comprised of various stocks in which money will be allocated to each stock based on some  
 criteria, currency pair trading, or creating an automated strategy similar to those discussed in this  
 report. Due to the rise of trading platforms such as TradeStation, it has become significantly  
 easier for an individual to manage his own trading. Consumers now have access to advanced  
 analytical tools that were once reserved for Wall Street traders, giving them the ability to make  
 well informed decisions in a fraction of the time.  
 Additionally, those that feel overwhelmed with the prospect of managing their own  
 money within the financial markets may seek professional advice from a money manager. In  
 exchange for a fee, a money manager has a fiduciary responsibility to make well-informed  
 investments. A professional money manager does not receive commission on transactions and is  
 compensated based on a percentage of the assets under management. ​ [ ] ​  It is in the best interest for  2
 both the money manager and client to grow the portfolio over time.  
  
 1.1 Project Description  
 The objective of this Interactive Qualifying Project was to scientifically design and  
 implement a system for trading or investing in the financial markets. The financial markets  
 include: equities, fixed income, futures contracts, options contracts, currency pairs, exchange  
 traded funds, and mutual funds. The group represents a faux hedge fund designed to work  
 2  ​ I. Staff, “Money Manager,” ​ Investopedia ​ , 02-Jan-2018. [Online]. Available:  
 https://www.investopedia.com/terms/m/moneymanager.asp. [Accessed: 26-Apr-2018].  
 9  

 
 together in order to develop a profitable set of automated trading systems. In addition to  
 developing trading systems, the group was tasked with following the economy and global news  
 through various financial outlets.   
   
 10  

 
  
 Chapter 2: Background  
 In order to be able to intelligently participate in trading and investing, it is essential to  
 first understand the different asset classes and market types. Furthermore, it is beneficial to  
 understand basic principles regarding the markets such as the breadth of the market, differences  
 between trading and investing, and the efficient market hypothesis. Pairing this information with  
 an understanding of the available tools and trading platforms available will provide a solid  
 foundation to begin developing a trading or investment strategy. This chapter will discuss the  
 background information required to adequately cover the fundamental concepts required to begin  
 participating in the financial markets.  
  
 2.1 Asset Classes  
 Asset classes represent various categories of investment vehicles, each with their own  
 level of risk and return. Different asset classes can be particularly useful for diversifying an  
 investment portfolio. The main asset classes are stocks, exchange-traded funds, mutual funds,  
 fixed income, foreign currencies, commodities, cash, and real estate. The following sections will  
 provide an introduction to each asset class, how it can be used as an investment vehicle, and the  
 benefits and drawbacks of the asset class.  
  
 2.1.1 Stocks  
 Stocks, also known as equities, represent shares of ownership in publicly held companies.  
 Ownership is determined by the number of shares an individual owns relative to the amount of  
 11  

 
 total shares in existence. The available stocks are listed on various exchanges such as the New  
 York Stock Exchange and the Nasdaq. At the most basic level, an individual may buy or sell  
 stock through a brokerage firm. Depending on the brokerage firm, a fee, called “commission”,  
 will be associated with the transaction. Typically these fees range somewhere from $8 - $10.  
 Historically, stocks have outperformed other asset classes over long periods of time.   
  
 2.1.2 Mutual Funds  
 A mutual fund is a method of investing established by pooling money from many  
 investors into a variety of assets. A mutual fund is operated by a professional money manager  
 whose job entails properly allocating the fund in an attempt to generate a profit for the investors.  
 [ ] ​  A mutual fund is required by the SEC to match the investment goals stated in a formal legal  3
 document presented to the investors represented in the fund.   
  
 2.1.3 ETFs  
 An exchange-traded fund (ETF), is a security developed to reflect a specific index or  
 collection of different securities. ETFs are comprised of underlying assets such as shares of  
 stock, bonds, oil futures, gold bars, foreign currency, or other financial instruments that are  
 divided into shares. ​ [ ] ​  Perhaps the most popular and widely known ETF is the Spider, which  4
 tracks the S&P 500 Index and can be traded under the ticker $SPY. Unlike mutual funds, ETFs  
 can be traded like a stock on an exchange.   
 3  I. Staff, “Mutual Fund,” Investopedia, 15-Mar-2018. [Online]. Available:  
 https://www.investopedia.com/terms/m/mutualfund.asp. [Accessed: 29-Apr-2018].  
 4  ​ C. F. A. A. Hayes, “Exchange-Traded Fund (ETF),” ​ Investopedia ​ , 03-Apr-2018. [Online]. Available:  
 https://www.investopedia.com/terms/e/etf.asp. [Accessed: 29-Apr-2018].  
 12  

 
  
 2.1.4 Fixed Income  
 Fixed income, or bond investments, generally will pay a set interest rate over a given  
 period of time, then return the principal investment. They have set interest rates and typically  
 assume less risk than stocks, though their values can still fluctuate due to the current interest and  
 inflation rates.   
  
 2.1.5 Currency  
 The foreign exchange market (forex), is the market responsible for the buying and selling  
 of currencies. Unlike stocks, the forex market trades 24 hours a day 5 days a week. Foreign  
 exchange markets are comprised of banks, commercial companies, central banks, investment  
 management firms, hedge funds, and retail forex brokers and investors. ​ [ ] ​  On the foreign  5
 exchange market, an individual trades currency pairs. In a simple world, where there are only  
 two currencies, the idea behind currency trading would be that either currency could increase or  
 decrease in value due to many factors. This means that the value of one currency relative to the  
 other increases or decreases, depending on which currency’s value changed in which way. In the  
 real world, there are many different currencies, each with its own value. This results in multiple  
 currency pairs, which are simply the exchange rates for trading one specific currency for another.  
 The main idea behind currency trading is to trade based on changes in a currency’s value relative  
 to another currency. For example, if a trader currently has US Dollars and predicts that the Euro  
 is going to gain strength relative to the US Dollar, then the trader can trade for Euros, wait for  
 5  ​ I. Staff, “Foreign Exchange Market,” ​ Investopedia ​ , 14-May-2008. [Online]. Available:  
 https://www.investopedia.com/terms/forex/f/foreign-exchange-markets.asp. [Accessed: 29-Apr-2018].  
 13  

 
 the value of the Euro to increase, then trade Euros for US Dollars. Because the trader is dealing  
 in relative value, this sequence of events will result in this trader being in possession of more US  
 Dollars than before the trades were executed. It is the same idea as buying shares of a company,  
 except the shares are currencies, and they can be traded for any other currency.  
  
 2.1.6 Commodities  
 Commodities are physical goods, such as gold, crude oil, natural gas, or corn. The sale  
 and purchase of commodities is usually done through futures contracts on exchanges. These  
 contracts standardize quantity and minimum quality of the commodity being traded. When the  
 contract expires, the holder actually owns the physical property. For example, if an investor  
 bought a contract for 5000 bushels of apples and let it expire, he would own 5000 bushels of  
 apples. The price of the contracts are predetermined, so the value of these contracts fluctuates  
 based on what happens to the price of the goods after the contract is sold. For instance, if an  
 investor bought a contract for a commodity at a certain price, and then the actual price of the  
 commodity increased, then the contract would be worth more.  
  
 2.2 Market Types  
 There are numerous financial instruments that are profitable to trade, however some of  
 the most popular among investors and day-traders alike are stocks, derivatives, and foreign  
 exchange markets, also known as “forex.” A trader should approach all three of these particular  
 markets differently and tailor his trading systems accordingly. Some traders may find it difficult  
 14  

 
 to develop a strong understanding of multiple markets and instead decide to specialize in a single  
 financial market and cultivate a mastery of the patterns and tendencies in their particular field.   
 Stocks constitute a share of ownership in a company. Generally, if a private company  
 needs capital in order to grow, they will sell shares of ownership on the stock market. Companies  
 have a legal responsibility to operate in accordance to the interest of the shareholders, so  
 ownership of over 50% of the stock of a company essentially constitutes ownership of the  
 company. There are a few ways that traders and investors can profit off of the purchase of stocks;  
 namely by entering into temporary long or short positions, and by collecting dividends. A  
 dividend is simply a percentage of the company's earnings, proportional to an investor's share of  
 ownership based on his shareholdings, although not every company pays dividends to its  
 shareholders. The value of a share in a company is simply determined by the supply and demand  
 of that share, however this often correlates to an expected return from dividends. The prices of  
 stocks fluctuate based on announcements from the company, like expected profits, employment  
 numbers or even upcoming projects. Shares in the company also sometimes give the owner  
 voting rights at shareholders' meetings.   
 Derivatives are securities that have a value dependent on a separate asset, like stocks,  
 bonds or commodities. The most commonly traded derivatives are called futures contracts.  
 Futures contracts are like a bet between two entities based on the future price of a commodity,  
 stock or bond; they are an agreement to buy or sell the underlying asset at a future date for a  
 specific price. Two parties often will have different predictions for the price of an asset in the  
 future, so futures contracts can allow one to profit off of his correct prediction, without buying or  
 selling the asset. Modern futures contracts are often cash settled. If a trader buys a futures  
 15  

 
 contract for 10,000 barrels of crude oil at $70 a barrel in 1 year, neither party needs to own the  
 oil, the calculated profit is paid in cash. Futures contracts are traded on exchanges just like  
 stocks. Similar to futures contracts, there are derivatives called “options” which are different in  
 that the buyer of an options contract does not have to ultimately follow through with the  
 exchange at the end of the contract. It is important to note that with options that the value of the  
 contract/exchange drops to zero if the buyers prediction was incorrect because the buyer is not  
 obligated to perform the transaction if it is not profitable.  
 Foreign exchange markets are markets that exchange currencies. Forex markets are  
 extremely liquid and include most of the world's currencies. The values of currencies are  
 determined speculatively based on a number of economic indicators including imports, exports,  
 employment, housing and the performance of various economic sectors. The Forex market is the  
 largest market in the world in terms of value exchanged. Currencies are traded in pairs like  
 EURUSD, USDGBP and USDCAD. Because of the high volume (over five trillion dollars a day)  
 and copious fluctuation due to speculation, currency pairs are very popular to day trade and  
 swing trade. The high volume also makes it one of the easiest markets to short, as traders can  
 always find buyers and sellers looking to exchange. This gives a lot of flexibility for designing  
 trading systems in forex, because traders can profit off of both bullish and bearish trends.  
 Additionally, because of the number of currency pairs available, traders can profit off of several  
 different pairs at once.   
  
 16  

 
 2.3 The Stock Market  
 To become a successful trader or investor it is important to not only have an  
 understanding of the various asset classes and market types, but to also have a solid foundation  
 of how the financial markets operate in addition to the various patterns that occur. This section  
 will discuss what a stock exchange is and how it operates, followed by what an index is and how  
 popular indexes can be leveraged to make well-informed decisions. Lastly, market sentiment,  
 trend and cycle, and sector rotation will be discussed in order to identify common patterns and  
 strategies when watching the market.   
  
 2.3.1 Exchanges  
 A stock exchange is a financial institution which facilitates the buying and selling of  
 stock. ​ [ ] ​  Stocks can be traded on one or more of several possible exchanges such as the New  6
 York Stock Exchange (NYSE). In order for a stock to be listed on an exchange, a company must  
 first conduct an initial public offering (IPO). An IPO signifies the first time shares of a company  
 can be purchased by the public. Prior to an IPO, a company is considered private and generally  
 consists of a small number of shareholders comprised of early investors (such as founders, their  
 families and friends) and professional investors (such as venture capitalists). ​ [ ]  7
 The NYSE is considered the largest equities-based exchange in the world with a total  
 market capitalization of $20 trillion. The exchange is located on Wall St. in New York, New  
 York and has existed since 1792. Many of the large corporations individuals are familiar with are  
 6  ​ D. Harper, “Getting to Know the Stock Exchanges,” ​ Investopedia ​ , 05-May-2017. [Online]. Available:  
 https://www.investopedia.com/articles/basics/04/092404.asp. [Accessed: 25-Mar-2018].  
 7  ​ C. F. A. A. Hayes, “IPO Basics: What Is An IPO?,” ​ Investopedia ​ , 24-Mar-2017. [Online]. Available:  
 https://www.investopedia.com/university/ipo/ipo.asp. [Accessed: 25-Mar-2018].  
 17  

 
 listed on the NYSE such as General Electric, Walmart, Coca Cola, and many more. ​ [ ] ​  The  8
 physical location of the exchange was initially meant to be used as the place of business for  
 trading to occur, however, due to the success of the internet, many trades are now placed  
 electronically. The NYSE generates revenue through transaction fees, listing fees, data fees, and  
 trading software. ​ [ ]  9
  
 2.3.2 Indexes  
 To gain a better understanding of how the market is performing as a whole, investors  
 typically turn toward stock indexes. An index is a small sample of the market, similar to a  
 portfolio, that is representative of the market as a whole. Ideally, an index is setup in such a way  
 where a change in the price of an index represents an exactly proportional change in the stocks  
 included in the index. In other words, if an index rises by 1%, then the value of the stocks that  
 comprise that index have also risen by a net of 1%.   
 While today there are many different stock indexes that exist, the first of its kind and still  
 one of the most popular was created in 1896 by Charles H. Dow known as the Dow Jones  
 Industrial Average (DJIA). Initially the DJIA was simply an average of the top 12 stocks in the  
 market. Since its inception, revisions have been made to the DJIA to more accurately reflect  
 current market conditions. Today, the Dow uses a form of price-based weighting in a portfolio of  
 30 stocks in order to calculate its index. ​ [ ]  10
 8  ​ “The New York Stock Exchange,” ​ NYSE: Listings Directory - Stocks ​ . [Online]. Available:  
 https://www.nyse.com/listings_directory/stock. [Accessed: 26-Mar-2018].  
 9  ​ S. Seth, “How The NYSE Makes Money,” ​ Investopedia ​ , 05-May-2015. [Online]. Available:  
 https://www.investopedia.com/articles/investing/050515/how-nyse-makes-money.asp. [Accessed: 25-Mar-2018].  
 10  ​ J. Folger, “Index Investing: What Is An Index?,” ​ Investopedia ​ , 05-Dec-2017. [Online]. Available:  
 https://www.investopedia.com/university/indexes/index1.asp. [Accessed: 25-Mar-2018].  
 18  

 
 Another popular index is the S&P 500. The S&P 500 is widely regarded as the best single  
 measure of large-cap U.S. stocks. The index incorporates the top 500 U.S. companies in leading  
 industries, and captures about 80% coverage of available market capitalization. ​ [ ] ​  Many traders  11
 and financial institutions use the S&P 500 as a benchmarking utility to assess their individual  
 performance compared to the broader market.  
  
 2.3.3 Market Sentiment   
 Market sentiment is the overall attitude of investors toward a particular security or  
 financial market. ​ [ ] ​  Typically, market sentiment refers to a relatively short period of time and is  12
 particularly important to day traders and technical analysts that rely on technical indicators.  
 However, market sentiment may also apply to the feeling or tone of the market over an extended  
 period of time.   
 Market sentiment is commonly associated with bearish or bullish terminology. When  
 investors claim a market is bearish, stock prices are going down. Conversely, when the market is  
 bullish, stock prices are going up. It is important to note that market sentiment does not always  
 accurately reflect the fundamental analysis of a stock and is generally influenced by emotion.   
  
 2.3.4 Trend and Cycle  
 Trend and cycle are principles defined in Macroeconomics attributed to the expansion or  
 recession of an economy at any given point in time. In the United States the long term trend is  
 11  ​ J. Folger, “Index Investing: The Standard & Poor's 500 Index,” ​ Investopedia ​ , 05-Dec-2017. [Online]. Available:  
 https://www.investopedia.com/university/indexes/index3.asp. [Accessed: 25-Mar-2018].  
 12  ​ I. Staff, “Market Sentiment,” ​ Investopedia ​ , 18-Jan-2017. [Online]. Available:  
 https://www.investopedia.com/terms/m/marketsentiment.asp. [Accessed: 26-Apr-2018].  
 19  

 
 generally positive, despite the occasional recession. The cycle can be defined as a sinusoidal  
 pattern in economic data resulting from periods of economic boom and busts. During a boom,  
 economic growth is above the trend line, generally resulting in unemployment falling. During a  
 recession, economic growth falls below the trend line. Depending on the individual, one might  
 wish to short during downtrends and go long during uptrends.  
  
 2.3.5 Sector Rotation  
 Sector rotation is an investment strategy that involves moving money from one industry  
 sector to another with the goal of beating the market. Traders may try to employ this strategy by  
 paying close attention to economic news and trying to predict which companies could be  
 successful in the next stages of an economic cycle.  
  
 2.4 Difference Between Trading and Investing  
 Trading and investing are two very different methods wherein the end goal is to attain a  
 profit in the financial markets. Investing can be defined as an attempt to gradually build wealth  
 over an extended period of time by developing a portfolio consisting of stocks, mutual funds,  
 bonds, and other investment instruments and subsequently buying and holding these assets for an  
 extended period of time. A typical investment strategy to generate higher returns over time is to  
 reinvest profits or dividends into additional shares of stock. How long an investor chooses to  
 hold onto his investment is up to the individual. However, investments are typically held for a  
 period of multiple years enabling the investor to take advantage of perks such as interest,  
 dividends, and stock splits. Additionally, since the market is fluid and prices will inevitably  
 20  

 
 fluctuate, long term investors are less concerned with downtrends in the market expecting prices  
 to ultimately rebound and recoup any losses they may have experienced.  
 Trading typically requires more effort on behalf of the trader. Unlike investing, trading  
 involves more frequent market activity via the buying or selling of stock, commodities, currency  
 pairs, or other instruments with the end goal of outperforming a traditional buy and hold  
 investment approach. Profits from trading are generated by entering the market at a lower price  
 and selling at a higher price within a relatively short period of time.  
  
 2.5 Efficient Market Hypothesis  
 The efficient market hypothesis, developed by Eugene Fama, is a controversial theory  
 stating that it is impossible for an individual trader to outperform the market as share prices  
 always incorporate and fully reflect all relevant information. It is important to understand the  
 fundamental difference between trading and investing when considering the significance of the  
 efficient market hypothesis. This distinction between trading and investing has stemmed from  
 this polarizing theory.  
 Evidence exists both supporting and rejecting the efficient market hypothesis. During  
 2012, Eugene Fama and Kenneth French published a study showing that the distribution of  
 abnormal trades were in accordance to what would be expected of inexperienced traders. ​ [ ]  13
 However, many financial moguls such as Warren Buffet have continually produced above  
 average returns, contradicting the efficient market hypothesis. ​ [ ]   14
 13  ​ E. F., K. R., and Fama, “Size, value, and momentum in international stock returns, by Fama, Eugene F.; French,  
 Kenneth R.,” ​ Journal of Financial Economics ​ , 01-Jan-1970. [Online]. Available:  
 https://ideas.repec.org/a/eee/jfinec/v105y2012i3p457-472.html. [Accessed: 26-Apr-2018].  
 14  ​ I. Staff, “Efficient Market Hypothesis (EMH),” ​ Investopedia ​ , 28-Sep-2016. [Online]. Available:  
 https://www.investopedia.com/terms/e/efficientmarkethypothesis.asp. [Accessed: 26-Apr-2018].  
 21  

 
  
 2.6 Fundamental vs Technical Analysis  
 Fundamental analysis is a method used when assessing the value of a security. ​ [ ] ​  Looking  15
 at qualitative and quantitative factors of a company can help investors determine whether the  
 price of the stock is correct. For example, if the price is steadily increasing, but through  
 fundamental analysis it is determined that the company is not performing well, then the price is  
 being driven up by investors rather than being driven up by the company doing well. That would  
 be a good indicator that shorting the stock is a good idea. There are a variety of things that one  
 can look at when using fundamental analysis, but common factors include: earnings, profit  
 margins, return on equity, price to earnings ratio, and price to book. ​ [ ]  16
 Technical Analysis is a trading technique employed to attempt to predict a security’s  
 future movement by analyzing statistics gathered from trading activity. ​ [ ] ​  This differs from  17
 fundamental analysis because a trader who uses technical analysis is making decisions based on  
 the movements of the stock rather than the intrinsic value of a company. A well known technical  
 analysis concept is the idea of support and resistance. Support is a price level where a downtrend  
 is likely to pause. This is because demand levels will be high enough to stop the price from being  
 driven down. Resistance is basically the opposite; as the price increases, more people will be  
 selling their shares, and there will not be enough demand to keep the price increasing.   
 15  ​ I. Staff, “Fundamental Analysis,” ​ Investopedia ​ , 20-Mar-2018. [Online]. Available:  
 https://www.investopedia.com/terms/f/fundamentalanalysis.asp. [Accessed: 26-Apr-2018].  
 16  ​ “5 Important Elements in Fundamental Analysis,” ​ EuroInvestor ​ . [Online]. Available:  
 http://www.euroinvestor.com/ei-news/2012/02/12/stock-school-5-important-elements-in-fundamental-analysis/1569
 4. [Accessed: 26-Apr-2018].  
 17  ​ I. Staff, “Technical Analysis,” ​ Investopedia ​ , 24-Apr-2018. [Online]. Available:  
 https://www.investopedia.com/terms/t/technicalanalysis.asp. [Accessed: 26-Apr-2018].  
 22  

 
 Though fundamental and technical analysis differ from each other, that does not mean that they  
 cannot be combined when creating a strategy. For example, fundamental analysis could be used  
 to pick stocks that fit a certain criteria, and technical analysis could be used to trade them.   
  
 2.7 Portfolio Management  
 Portfolio management is a methodology followed by investors to avoid high exposure in  
 a particular market or stock through proper asset allocation and diversification. ​ [ ] ​  An investor  18
 who subscribes to the efficient market hypothesis may wish to take a passive approach to  
 portfolio management by investing in an index fund. Conversely, those who believe they can  
 outperform the market actively manage their portfolios using technical analysis to select their  
 holdings and the continual rebalancing their portfolios. An investor can rebalance his portfolio  
 whenever he sees fit, however, typically investors rebalance their portfolio quarterly, bianually,  
 or yearly.  
 Since it is impossible to always predict winners in the market, it is important for an  
 investor to diversify his portfolio. A properly diversified portfolio allows an investor to reduce  
 his exposure to any particular market in the event an asset begins to perform poorly.  
 Additionally, proper diversification allows an investor to capture the returns of all sectors with  
 less volatility at any given time.  
 18  ​ I. Staff, “Portfolio Management,” ​ Investopedia ​ , 03-Dec-2015. [Online]. Available:  
 https://www.investopedia.com/terms/p/portfoliomanagement.asp. [Accessed: 26-Apr-2018].  
 23  

 
  
 2.8 Trading Platforms and Brokerage Accounts  
 Every trader needs an avenue to reason about a market, test his hypotheses, and place his  
 trades in order to enter or exit positions to make a profit. A trading platform is a piece of  
 software run locally or from a web server that allows traders to chart symbols from different  
 financial instruments and do analyses over different time periods. A brokerage firm is simply an  
 intermediary that connects buyers and sellers of a financial instrument. Often, the same trading  
 platform will also allow the user to place trades, in which case they need their own brokerage  
 firm, or at least a connection to one. For comparison, something like Yahoo Finance could be  
 considered a charting platform, but not a brokerage. An application like TradeStation which  
 allows the user to analyze charts and place trades is a combination of a brokerage and a charting  
 or trading platform.  
  
 2.8.1 TradeStation  
 TradeStation made its debut in 1991 as an all in one platform which provides users with  
 real-time and historical market data, advanced financial analysis tools, and a means to facilitate  
 trades. Upon installation, TradeStation comes with a large number of pre-defined indicators,  
 trading strategies, and analysis tools which individuals can then modify or use to create their own  
 tools. ​ [ ] ​  The platform can be leveraged by both manual traders and automated strategy  19
 developers. TradeStation supports the development, testing, optimizing, and automation of  
 trading. Users can choose to run these strategies on a simulated account before trading live.  
 19  ​ “Award-Winning Trading Tools and Technology,” ​ TradeStation ​ . [Online]. Available:  
 https://www.tradestation.com/technology/. [Accessed: 26-Apr-2018].  
 24  

 
  
 Figure 2.1 TradeStation environment with chart analysis window of $AAPL  
  
 2.8.2 Other Alternatives  
 Unlike TradeStation, the other trading platforms listed below require a separate brokerage  
 account that users must link to the platform in order to execute real trades. However, if a user  
 wishes to only work with a simulated account, a brokerage account is not required.  
  
 QuantConnect  
 QuantConnect offers a web environment to develop, backtest, and run trading algorithms.  
 It is possible to trade with a paper money account, but in order to execute real trades the platform  
 requires the integration of a brokerage account. For traders that are relatively new to developing  
 automated strategies, QuantConnect boasts an active community where users can ask questions  
 25  

 
 and receive help from other members. The platform currently allows traders to write their  
 algorithms in C#, Python, and F#.  
  A potential drawback of QuantConnect is that it requires users to develop their strategies  
 within the QuantConnect ecosystem. Should a more advanced user require more flexibility in  
 how he develops his strategy, one might consider Zipline Live described below.  
  
 Zipline Live  
 Zipline Live provides a powerful API (Application Programming Interface) for users to  
 build trading strategies on using Python. With Zipline Live, users are able to access current  
 market data to feed into their systems. Like QuantConnect, Zipline Live will require a brokerage  
 account in order to execute real trades. Zipline Live provides much more flexibility than  
 QuantConnect since users are able to develop their systems directly on their computer rather than  
 through a web interface.   
  
 2.9 Common Analysis Tools on Trading Platforms  
 Whichever trading platform a trader chooses to use, there are a handful of basic tools that  
 one should expect to see on any platform. One of the most basic tools that is often taken for  
 granted is the ability to view information in multiple different time frames. Another is the ability  
 to add indicators to charts. Additionally, for any trading platform that allows traders to develop  
 their own systems, there ought to be an optimization tool available for use. Further, it is  
 important that there are analytical tools as well, such as walk-forward analysis, which is featured  
 in TradeStation. Another basic part of trading system development is the usage of metrics;  
 26  

 
 without these, it is hard to say whether one system is better than another. Finally, screeners are a  
 good addition to any trading platform, as they give the user the ability to pick assets based on  
 many criteria. All of these tools are described in detail in the following sections, 2.8.1-2.8.6.  
  
 2.9.1 Time Frames  
 On a trading platform, traders will be able to view the charts of any particular financial  
 instrument on different time frames. For example, they can view how an asset is traded on one,  
 five, fifteen, thirty and sixty minute bars. Additionally, one can utilize longer time frames such  
 as hours, days or weeks. The best traders will analyze a number of different time frames before  
 placing a trade in order to draw conclusions from trends of different sizes. Traders use a  
 technique referred to as “multiple time frame analysis”, which consists of monitoring one  
 currency pair, stock or derivative over multiple time frames. Three periods is the most typical  
 number to read, too many charts and the data will be redundant, but too few and there could be  
 missing information.  
 Regardless of how long one intends to hold his position in the market, a trader should  
 take into account a long-term time frame, a medium-term time frame, and a short-term time  
 frame. The exact lengths of these time frames will, however, depend on how long a position is  
 held. For example if the position is maintained for months, there will not be as much value in  
 looking at minute or hour bars. The chosen long-term time frame should establish an  
 overarching, dominant trend. Traders should not enter their positions on the long-term frame,  
 however, they should remember when choosing their position that if they hold too long, the  
 assets value will follow the longest term trend. If they want to short a long-term bullish trend, do  
 27  

 
 not hold for very long. When viewing the medium-term time frame, there will be apparent  
 deviations from the larger trend. This will present opportunities to profit against the broader  
 trend. The short-term time frame is where traders should be placing their trades, because this is  
 where every small variation in price can be viewed. The long-term trend should not even be  
 discernible in the shortest time frame, but remember that the shortest charts will eventually  
 follow the longer ones.  
  
 2.9.2 Indicators  
 An indicator is a mathematical calculation based on a security’s price, volume, or open  
 interest. ​ [ ] ​  In technical analysis, traders use indicators in an attempt to predict future price  20
 movements. Each indicator attempts to uniquely identify a phenomenon, and often times traders  
 will use multiple indicators in combination in order to define a set of rules for entering or exiting  
 a position. This section details several different indicators as well as how traders employ these  
 indicators within their strategies. Having a basic understanding of these may lead to inspiration  
 for a new strategy that gives a trader the edge over the rest of the market.   
  
 Volume  
 Volume is an extremely important indicator to be aware of when developing trading  
 strategies. Markets with high amounts of volume are referred to as liquid markets. A liquid  
 market ensures a trader will be able to execute his trades in a timely fashion. On the other hand,  
 20  ​ I. Staff, “Technical Indicator,” ​ Investopedia ​ , 18-Mar-2018. [Online]. Available:  
 https://www.investopedia.com/terms/t/technicalindicator.asp. [Accessed: 29-Apr-2018].  
 28  

 
 an illiquid market, a trader may get stuck in a position due to a lack of interest amongst other  
 traders.  
 Traders use volume in technical analysis to measure the relative worth of a market move.  
 If a market is making a strong price movement, then the strength of the movement directly  
 correlates to the volume for that period. The higher the volume during a price move, the more  
 significant the move. ​ [ ] ​  Additionally, volume may be used as an indication of which direction  21
 the price will move. In a bullish market, an increase in volume over time may guarantee a bullish  
 pattern in the future. During a bearish market, low volume may suggest profit taking whereas  
 high volume on the downtrend suggests shorts piling in.  
  
 Volume Weighted Average Price  
 The Volume Weighted Average Price (VWAP) is calculated by adding up the dollars  
 traded for every transaction and then dividing by the total shares traded for the day ​ [ ] ​ . Typically  22
 investors are satisfied when they buy shares below the VWAP signifying an efficient trade was  
 made. A disadvantage of using the VWAP can take effect when data becomes stale as a result of  
 calculating the VWAP too far in the past. For this reason, traders typically limit VWAP  
 calculation to one day of trade data.  
   
  
 21  ​ I. Staff, “Volume,” ​ Investopedia ​ , 09-Nov-2017. [Online]. Available:  
 https://www.investopedia.com/terms/v/volume.asp. [Accessed: 26-Apr-2018].  
 22  ​ I. Staff, “Volume Weighted Average Price - VWAP,” ​ Investopedia ​ , 25-Jul-2015. [Online]. Available:  
 https://www.investopedia.com/terms/v/vwap.asp. [Accessed: 26-Apr-2018].  
 29  

 
 Kaufman Efficiency Ratio  
 The Kaufman Efficiency Ratio is a measure of efficiency. It is calculated by dividing the  
 price change by the absolute sum of the price movement. ​ [ ] ​  The result is a number between 0  23
 and 1, with high values indicating a more efficient trending market. A higher efficiency means  
 that during a time period, the price moves mostly in one direction.  
  
 Moving Average Convergence Divergence  
 The Moving Average Convergence Divergence (MACD) indicator is a trend following  
 indicator that shows the relationship between two moving averages of prices. It is calculated by  
 subtracting a long term exponential moving average from a short term exponential moving  
 average. ​ [ ] ​  For strategies, a trader can plot a signal line over the MACD to function as a trigger  24
 for buying and selling. Traders will also often look for other indicators to use in conjunction with  
 the MACD.   
  
 Bollinger Bands  
 Bollinger Bands are volatility bands placed above and below a moving average. The  
 bands will automatically widen and contract when volatility increases and decreases,  
 respectively. ​ [ ] ​  Bollinger Bands can be used to identify a variety of different patterns, and  25
 because they are so dynamic, traders use this indicator in a large variety of places.   
 23  ​ “Kaufman's Efficiency Ratio (ER) « ETF HQ,” ​ ETF HQ ​ . [Online]. Available:  
 http://etfhq.com/blog/2011/02/07/kaufmans-efficiency-ratio/. [Accessed: 26-Apr-2018].  
 24  ​ I. Staff, “Moving Average Convergence Divergence - MACD,” ​ Investopedia ​ , 13-Mar-2018. [Online]. Available:  
 https://www.investopedia.com/terms/m/macd.asp. [Accessed: 26-Apr-2018].  
 25  ​ “Bollinger Bands,” ​ StockCharts.com ​ , 18-Sep-2017. [Online]. Available:  
 http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_bands. [Accessed:  
 26-Apr-2018].  
 30  

 
  
 Keltner Channels  
 Similar to Bollinger Bands, Keltner Channels are volatility based lines set above and  
 below an exponential moving average. Instead of using the standard deviation like the Bollinger  
 Bands, Keltner Channels use the Average True Range (ATR) to set the position of the lines. The  
 exponential moving average dictates direction and ATR determines channel width. ​ [ ]   26
   
 2.9.3 Optimization  
 A trading system may be composed of several trading strategies. As is discussed in  
 Chapter 4, there are many different types of strategies. Each of these strategies is essentially a set  
 of rules that determine what actions to take regarding a financial instrument. One such strategy  
 could be based on the simple moving average (SMA) of a stock. SMA is the sum of the prices of  
 a financial instrument over a past number of bars, divided by the number of bars. In order to  
 calculate SMA, one has to determine the number of bars over which the SMA is calculated; this  
 is called an “input.” A trading strategy could be based on several conditions, therefore requiring  
 several inputs. One example of such is a strategy that buys when the 9-day SMA crosses above  
 the 25-day SMA. The two inputs in this case are set to 9 and 25, but they each could be any  
 number. A lot of the time, it could be unclear to the trader what the best input is for his trading  
 strategy.  
 One method of determining the best input is called “optimization.” Optimization tries  
 many values in a range for an input, keeping track of the best ones by testing how the strategy  
 26  ​ “Keltner Channels,” ​ StockCharts.com ​ , 20-Dec-2017. [Online]. Available:  
 http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:keltner_channels. [Accessed:  
 26-Apr-2018].  
 31  

 
 would perform on past data. This is typically performed using some computer software, such as  
 TradeStation’s Strategy Optimization tool. Another key aspect of optimization is how it  
 determines what the “best” input value is. One might want it to simply maximize the amount of  
 money that a strategy makes, as that is generally what a trader’s goal is. However, that might not  
 guarantee the best performance in the future. For this reason, there are many different rules that a  
 trader can use for picking the best input when optimizing. Some other rules include: maximizing  
 the percentage of winning trades, maximizing the win-to-loss ratio of trades, maximizing the  
 number of consecutive winning trades, and minimizing maximum adverse excursion. Picking  
 which rules to optimize for is largely up to each trader’s preference. There are many other rules  
 than the ones listed above, though none of them guarantee future performance. This is due to an  
 idea called “overfitting,” which is when a system or strategy is optimized to perform well on past  
 data, but does not perform well in the future. When optimizing and testing a trading system, it is  
 important to keep overfitting in mind and to ensure that the results can translate well into the  
 future.  
  
 2.9.4 Walk-Forward Analysis  
 One way to combat overfitting is to use a process called “walk-forward analysis.”  
 Walk-forward analysis is a technique that optimizes a system using data from a sample in a time  
 frame, then tests the performance of the system using data from outside of that sample. This  
 technique can also be used with systems that are currently in-use to verify that they are still  
 performing well. For example, the system could be optimized over 7 weeks, then tested on the  
 most-recent week of data. If it performs as well as expected, the system stays in use. The next  
 32  

 
 week, walk-forward analysis is performed again, but the time frame is shifted forward by one  
 week. This continues on, each week the time frame stepping forward, hence why it is called  
 “walk-forward analysis.”  
  
 2.9.5 Expectancy, Expectunity, System Quality, and Sharpe Ratio  
 Another key to evaluating a trading system is to quantify it using metrics that allow  
 comparison to other trading systems. Four such metrics for this purpose are called “expectancy,”  
 “expectunity,” “system quality,” and “sharpe ratio.”  
 The first metric, expectancy, is the simplest. Expectancy is the profit or loss per each  
 dollar risked on each trade of a trading system. This must be computed based on current trading  
 data for a trading system, whether this is from real trading or backtesting. It is better to have a  
 high, positive expectancy. A negative expectancy would mean that a trading system is expected  
 to lose money, based on its past performance. This would clearly be bad. Based on the definition,  
 in order to calculate expectancy, the amount risked for each trade needs to be known. One  
 definition of “risk” is the amount that a trader is willing to lose on any given trade. For example,  
 if the trader buys a share worth $40 and decides that he will exit the trade to preserve his capital  
 only if the share drops to $30, then the risk for this trade is $10. This definition might not work,  
 however, for some trading systems, since a system does not necessarily have an exit rule that is  
 that clear-cut. It could be based on several indicators, and the amount lost on each trade might  
 vary widely. In this case, it makes more sense to use either the average loss, or the maximum  
 loss across all of a system’s trades. Using the average loss is probably closer to the risk that a  
 system might see over most trades, but a higher expectancy calculated using the maximum loss is  
 33  

 
 much more meaningful. Whichever value is chosen, it is then used to calculate the R Multiple for  
 each trade. The R Multiple for a trade is the profit or loss for the trade divided by the Risk for the  
 trade. From here, expectancy of a system is simply equal to the average of all of its R Multiples.
  
 The next metric is expectunity, which is based on expectancy. Expectunity is annualized  
 expectancy, or the profit or loss per dollar risked per year. This should give a general idea of how  
 a trading system performs over a full year. Expectunity is calculated by multiplying the  
 expectancy of a trading system by the total number of trades it made, divided by the number of  
 years over which those trades were executed.  
  
 Third, system quality is a metric that can be used as a reference for how well a trading  
 system will be able to perform using different position sizing rules. A higher system quality  
 number indicates that a trading system could be more flexible with its rules, while a lower  
 number may mean that it needs a more strict approach. The system quality metric for a system is  
 calculated by dividing expectancy by the standard deviation of its R multiples, and finally  
 multiplying this by the square root of the total number of trades that the system made.  
  
 Finally, sharpe ratio is the most widely-used metric for calculating risk-adjusted return.  
 Sharpe ratio can be used to determine if a system is worth trading, because it is inherently  
 benchmarked against risk-free assets, such as U.S. Treasury bonds. U.S. Treasury bonds are  
 34  

 
 “risk-free” because it is extremely improbable that the United States’ government would not  
 honor these contracts. If this situation arose, it probably would not matter because there would  
 have to be a much larger issue at-hand, like a nuclear war. Risk-free assets have a sharpe ratio of  
 0, simply due to the definition of sharpe ratio. A negative sharpe ratio means that it would be  
 more profitable to invest in risk-free assets than to invest money in a particular trading system. A  
 positive sharpe ratio means that a trading system is more profitable than risk-free assets. To  
 calculate the sharpe ratio for a trading system, the return of the best risk-free investment is  
 subtracted from the average return of the trading system. That value is then divided by the  
 standard deviation of the trading system’s returns, resulting in the sharpe ratio of this system.  
 Sharpe ratio can be used to analyze the potential risk-return characteristics of a trading system;  
 and although it is no guarantee of returns, combining it with other metrics can give a trader the  
 information he needs to decide whether or not to put a system into action.  
  
  
 2.9.6 Stock Screeners  
 A stock screener allows traders and investors to query a large amount of market data and  
 filter the results based on some user-defined criteria. For example, some commonly used filters  
 for screening include price, volume, market capitalization, price-to-earnings ratio and many  
 more. Incorporating a stock screener into a trading or investment strategy will significantly  
 benefit the individual as it requires forethought to identify which components of a stock matter  
 35  

 
 most to the individual. Strict rule-based strategies are helpful in avoiding personal biases and  
 emotional trades.   
 Stock screeners have a variety of uses in trading and investing. A screener may be used  
 by day traders in order to identify potential stocks to trade for that day or by portfolio managers  
 to build an initial portfolio or find stocks to rotate into their portfolio. There are many free stock  
 screeners available online provided by services such as Yahoo Finance and the Nasdaq. While  
 most free screeners are generally the same, some offer more filter criteria than others.   
 Figure 2.2 Yahoo Finance screening for mid-cap technology stocks   
 36  

 
 2.10 Order Types  
 When placing a trade manually or automatically, traders must understand the various  
 types of orders at their disposal. Understanding the benefits and drawbacks of each type of order  
 will help traders maximize their profits.  
  
 2.10.1 Market Order  
 Market orders are trades that ensure a long or short position is entered, but do not  
 guarantee the trader a price. For example, if a market order for a long position is placed while a  
 share is priced at $5.00, but by the time the order is received by the broker the price has risen to  
 $5.50, the share will be purchased for $0.50 more than originally intended. This is called  
 “slippage,” the amount of money lost between the expected price and the price of the filled trade.  
 Slippage is important to take into account, especially for markets that fluctuate quickly.  
  
 2.10.2 Limit Order  
 In order to remove slippage from their trades, traders often use limit orders. Limit orders  
 ensure that the order is filled at a particular price or better, but do not ensure that the order will  
 be inevitably filled. Limit orders will often fill very quickly in high volume markets, but in lower  
 volume markets it may be difficult to ensure an exact price, especially if there is a large breadth  
 between bids and asks.   
  
 37  

 
 2.10.3 Stop Order  
 There are also two kinds of stop orders, which are similar to market orders, however  
 allow traders to specify a price at which the order will be executed. Stop-loss orders execute a  
 market buy or market sell when the asset reaches a particular price. However it is important to  
 note that these orders do not ensure that the order is filled at their specified price, because it is a  
 market order. One major drawback of stop-loss orders is that if there is a large gap in the  
 orderbook, there could be larger losses than expected. Suppose an investor purchases a stock  
 which closes the day at $10.00 a share, and he sets a stop loss for $9.00, expecting 10% loss if  
 the stop-loss fills. If the value crashes down to $5.00 before the market opens the next day, there  
 would be a 50% loss when the stop-loss triggers. To combat this risk of selling an asset far below  
 the specified price, many seasoned investors will use stop-limit orders. Stop-limit orders are  
 similar to stop-loss orders, but they execute a limit order instead of a market order when the asset  
 reaches the specified price. Because at limit order is used, the order will only execute at the  
 specified price or better, so the benefit of a stop-limit order is that if the market price of your  
 asset crashes far below the stop-loss price, but then quickly recovers, the stop-loss will not fill at  
 the bottom.  
   
 2.10.4 Other Types of Orders  
 A few more uncommon orders that experienced traders may use are Day Orders, Fill Or  
 Kill Orders, Immediate Or Cancel Orders and Not-Held Orders. Day Orders are limit orders that  
 expire if they are not filled the day they are placed. Fill Or Kill orders tell the brokerage firm to  
 perform a transaction completely when it is received or cancel it, if it cannot be completely  
 38  

 
 filled. Fill Or Kill orders are good for purchasing large amounts of a security, and ensuring that it  
 is placed at a desired price. Immediate Or Cancel orders instruct the brokerage to fill the order  
 immediately, as much as it can, and cancel the rest of the order. If you want to scalp as much as  
 you can of a cheap asset, Immediate Or Cancel orders are useful. Not-Held orders are when an  
 investor wants to give the floor trader the discretion to seek a better price than the current market  
 price. So if an investor believes that the floor trader may have better judgement than themselves,  
 this type of order can be used. However, the investor may suffer losses from this type of order in  
 a sudden market shift.  
  
 2.11 Buy & Hold  
 Someone who believes that he cannot systematically beat the market should use a buy  
 and hold strategy. Generally speaking, this type of strategy involves selecting a diverse portfolio  
 of stocks, ideally they will not be prone to large swings in price and will have low correlation  
 with each other. This report will not go into much detail, but Modern Portfolio Theory is  
 important to know for investors employing a buy and hold strategy. Modern Portfolio Theory is a  
 theory on how investors can construct portfolios to optimize expected return based on a given  
 level of market risk, given that risk is an inherent part of a higher reward.   
  
 2.12 Trading Systems  
 Every trading platform will gives traders the ability to place trades manually, but ever  
 since computers became accessible to the general public, more and more digital trading  
 platforms have given users the option to program their own strategies and trade automatically.  
 39  

 
 Some of the major advantages of automated trading systems are that programs do not deviate  
 from their rules, they can trade all day without getting tired or needing breaks, they can analyze  
 more data at once, and they do not have emotions like greed and fear. However, automated  
 trading systems are not without their drawbacks. For example, human traders are much better at  
 analyzing fundamentals and accounting for them, for example an earthquake in Tokyo might  
 affect the price of the yen, but a robot will not be able to take this into account. Additionally,  
 manual traders gain experience over time, and may be able to “master the charts” and draw  
 conclusions and make predictions based on their own intuition rather than technical analysis.  
 There is also the possibility that a human trader is able to realize when a market is behaving  
 erratically or unreasonably, and can decide not to place trades, or profit off of the instability. A  
 robot cannot make these intuitive judgements. Finally, if trading automatically, the trading  
 system may eventually stop performing as expected due to market changes. Markets change over  
 time, and manual traders have the ability to adapt on the fly, but automated traders may need to  
 retire old systems and write new systems entirely.  
  
 2.13 Manual Trading  
 For all traders, it is important to hold one’s self to a set of rules and follow them strictly.  
 If a trader does not follow his own rules on when to enter or exit a trade, then he does not have a  
 trading system, he is just making a bet. This is especially key for manual trading. Often times,  
 emotion can get in the way, leading a trader to justify why something did not go as expected,  
 rather than exiting the trade and readjusting his strategy. Given that it is possible to justify almost  
 40  

 
 anything, this can be a dangerous mindset to find one’s self in. This is a reason that some people  
 pay professionals to manage their own money.   
 A trader must avoid making emotional mistakes when trading his money. If he is capable  
 of avoiding such mistakes, then he is on the right path to having a positive trading system. One  
 way to go about this is to practice manually trading by paper trading. Paper trading is when a  
 trader follows along with an asset in real-time, writing down the trades he would make at the  
 time on paper. In this way, a trader can test a strategy and practice the trading mindset without  
 risking money.  
  
 2.14 Automated Trading  
 For some traders, having an automated system is much more appealing than manually  
 trading. Automated systems provide many advantages that manual trading does not. Platforms  
 that support automated trading systems will often allow traders to backtest the system as if it  
 were making real trades, but on past data. This allows a trader to accurately debug and optimize  
 a system. This is not feasible with a manual system, because it is difficult to remove a trader’s  
 knowledge of what actually happened. Additionally, an automated trader will always do exactly  
 what its rules specify it to do. This means that a trader does not have to have a strong will power  
 to successfully use an automated trader. An automated system does not have to deal with  
 emotion, and it will operate with precision. Another benefit to using an automated trading system  
 is that it can put buy or sell orders in much faster than a manual trader could.   
  
 41  

 
 2.15 Monte Carlo analysis  
 Monte Carlo analysis is a method of determining probable outcomes of a trading system  
 based on its past performance. Monte Carlo analysis uses random variables in an attempt to  
 model the unknown factors in the market. Monte Carlo analysis plots the actual sequence of  
 trades as it falls around the 95% and 5% confidence intervals of these random variables. One  
 important aspect of Monte Carlo analysis is that it assumes a perfectly efficient market, so, for  
 example, performing it on a system that is subject to a large amount of emotion would not  
 necessarily be beneficial to a trader. Monte Carlo analysis can be useful for traders who are  
 trying to determine the risk in their system. If the system falls out of the Monte Carlo analysis  
 significantly, then it is a risky system. A trader must determine what his goals are before  
 deciding to trade his system, meaning that if he does not want to lose a lot of money in a short  
 period of time, he will want to reduce the risk of the system. Monte Carlo analysis can help a  
 trader make this decision, but it is still a subjective decision. Therefore, utilizing multiple  
 analysis techniques would be more beneficial than relying on Monte Carlo analysis alone.   
  
 42  

 
  
 Figure 2.3 Example of Monte Carlo analysis on an unreliable system  
 Above is an example of Monte Carlo analysis that has been performed on a system that  
 overall made a profit, but it does not look very reliable. The Monte Carlo analysis was performed  
 using a software called Market System Analyzer. The red line in the picture above represents the  
 5% confidence interval, the grey line the 50%, and the green line the 95%. During the first third  
 of the system’s trades, it was within the 5% and 95% confidence intervals, but in the middle  
 section it fell significantly out of those intervals. From a statistical standpoint, this is very  
 unlikely, which indicates that Monte Carlo analysis may not be a good predictor for this system.  
 Another factor one could look at is the average loss compared to commission. The average loss  
 including commission is about $5 on this system. With a commission of $8, that means that the  
 average loss was made at a price above the entry price. So, putting in more equity to this system  
 would decrease the average loss and could change the system entirely. In summary, Monte Carlo  
 analysis is very useful for analyzing the risk of a system, but there are other metrics that are  
 useful to determine how to improve a system.   
 43  

 
 Chapter 3: Methodology  
 This section details the approach a trader must follow in order to scientifically develop a  
 strategy to trade on the financial markets. Although these may seem like guidelines, it is  
 important to incorporate a unique perspective when developing a system. These can form a good  
 starting point for building a system, but it is important to experiment with different ideas. The  
 following methodology describes the approach used in this IQP to develop the automated  
 systems.   
  
 3.1 Research  
 Before investing in the financial markets, it is important for a trader to understand  
 everything there is to know about the markets. Knowing the terms and how to read a chart are  
 the bare minimum. Following the news, especially economic news, is extremely important as  
 well. Knowing the reasoning behind market activity will provide a trader with an advantage over  
 those who do not. Making smart, and informed decisions can go a long way to maximizing  
 returns on trades. Additionally, knowing common strategies that people use to trade can give a  
 trader the upper hand on the markets. For example, if a trader knows that a common strategy is to  
 buy after a certain pattern shows up, the trader can watch for the pattern and make a  
 countermove that benefits themselves. The last aspect to consider when trading on the financial  
 markets is that one must treat everything as a lesson so that he may learn from his mistakes or  
 victories.  
  
 44  

 
 3.2 Systematically Beating the Market  
 The first step in creating a trading or investing strategy is determining whether or not the  
 efficient market hypothesis is a legitimate limitation on potential returns (whether or not a trader  
 can systematically beat the market). An active manager is a money manager that takes the stance  
 that he can beat the market. Thus, an active manager would make trades with his money, and  
 consistently move money between assets to try to maximize profits over a period of time. A  
 passive manager is a manager who believes that the market is efficient and he cannot beat it.  
 Thus, a passive manager would invest his money into low risk long term investments. Studies  
 show that less than 25% of active managers outperform passive managers. Yet some people still  
 believe that they are capable of outperforming the market. If it were easy, then everyone would  
 be making money. But that is not possible, because someone must be getting a worse deal: if a  
 person follows the strategy “buy low and sell high”, then someone else sold low and someone  
 else bought high!  
  
 3.3 Finding an Edge  
 The final step of creating a successful system, if a trader believes he can systematically  
 beat the market, is to find an edge. An edge is anything that can give the trader an advantage  
 over the rest of the market. It is not an easy task, however, so a successful trader must constantly  
 be looking for the next thing that will give them an edge. It is possible for a system to stop  
 working for some period of time, and then eventually it might start working again. So having a  
 variety of strategies that can be employed at different times will provide a resilient trading  
 45  

 
 system. A trader must constantly keep an eye out for improvement, if one system stops working,  
 then he must substitute it for another.   
   
 46  

 
 Chapter 4: Trading System Development  
 In this chapter, the ideas and implementation details of each system are described. Each  
 system was developed by an individual member of the group, starting with a hypothesis,  
 followed by a methodology, and, lastly, the specific implementation details of the system. A  
 brief overview of the system performance is mentioned before discussing the next system.  
 However, a more in depth analysis of the system performance is described in the following  
 chapter.   
 4.1 Pre-Market Gaps  
 While the general population may see the stock market as a get rich quick scheme,  
 outperforming the market and generating a profit can require a lot of time and effort. Long term  
 exposure to the market invariably introduces risk: policy changes, earnings reports, the economic  
 calendar, and many other factors may positively or negatively affect a long term investment. It is  
 ultimately the investor’s responsibility to keep up with current events to ensure an investment is  
 safe and performing as expected.  
 As a university student with limited time, an intraday strategy was appealing since short  
 term positions require less maintenance than positions lasting for an extended period of time. For  
 this reason, this system will never hold a position overnight despite potentially losing money  
 when selling a position prior to market close.  
  
 4.1.1 Hypothesis  
 In general, more advanced investors and traders profit from the behaviors exhibited by  
 amateurs. A perfect example of this is a technique referred to as stop hunting. Stop hunting is a  
 47  

 
 strategy that attempts to force some market participants out of their position by driving the price  
 of an asset to a level where many individuals have chosen to set their stop-loss orders. ​ [ ] ​  For this  27
 reason, if one were able to understand the patterns more advanced traders follow, it may be  
 possible to make winning trades by entering and exiting alongside experienced traders.  
 Day trading provides a perfect opportunity to test this hypothesis. New traders are often  
 attracted to financial markets through the allure of producing exceptional returns in a short time  
 frame. Of course the opposite usually occurs, with experienced traders buying at their desired  
 price, waiting a bit, and dumping their shares while taking a profit. The amateur traders enter the  
 market just before the experienced traders exit the market and subsequently watch their  
 investment decline. In a panic, they sell their shares which are purchased by the experienced  
 traders and the cycle continues.  
  
 4.1.2 Methodology  
 The most important aspect of this strategy is the pre-market screening done between the  
 hours of 9:00 AM - 9:30 AM. Pre-market data usually comes at a premium and requires a  
 monthly payment to a service which offers access to such data. The screener utilized by this  
 strategy is FinViz Elite which costs $39.50/month and provides access to pre-market data in  
 addition to many other features.   
 The screening criteria is used to identify stocks in the pre-market that have a high chance  
 of continuing to breakout after a pre-market gap. The goal of the pre-market screen is to identify  
 stocks that other experienced day traders are interested in. Since this strategy performs poorly  
 27  ​ I. Staff, “Stop Hunting,” ​ Investopedia ​ , 21-Apr-2018. [Online]. Available:  
 https://www.investopedia.com/terms/s/stophunting.asp. [Accessed: 26-Apr-2018].  
 48  

 
 when volatility is relatively stagnant, an ATR of greater than 0.5 for the past 14 days is applied.  
 Average volume and current volume also play a large role in identifying stocks that the strategy  
 would be successful trading. In order to provide a liquid market, an average volume of over 300k  
 during the recent 3 month period is applied. Additionally, a current volume of over 100k is  
 desired to ensure there is enough interest in a particular stock during the pre-market. Lastly, a  
 gap filter between 4% and 10% is applied to identify potential big movers when the market  
 opens. A limit of 10% has been used for the gap filter since it has been observed that stocks  
 gapping more than 10% have a lower chance of continuing to breakout when the market opens.  
  
 Figure 4.1 Pre-market screening on Finviz  
 Once a list of stocks has been selected, the price and volume are analyzed in addition to  
 any news reports that may indicate the price of stock will continue to rise when the market  
 opens. The strategy is applied to these symbols via a 2-minute candlestick chart. Each bar is  
 analyzed and the information regarding the last green and red bar is constantly monitored.  
 Leveraging this information, the strategy executes buy and sell orders which will be described in  
 the section below.  
  
 49  

 
 4.1.3 Entry/Exit Conditions  
 This strategy performs best on 2-minute bars ​ – ​  1-minute and 5-minute bars were also  
 tested but on average drastically reduced the profit factor of the strategy. There are several  
 conditions considered in determining whether or not to enter the market. A combination of daily  
 highs, candlestick patterns, and volume weighted average price are analyzed in the entry code.   
 The strategy first looks at the high of the previous day and compares that to the price at  
 market open. It is not uncommon for traders to look at a variety of timeframes to help determine  
 which position should be taken at the present time. Based on historical trading data, it has been  
 determined that it is unwise to expect a stock to continue trending upward after a gap if its  
 opening price for the current day is not greater than the high of the previous day. For this reason,  
 the strategy prevents entering a market where this occurs.  
 Perhaps the most important component of this strategy, aside from the pre-market  
 screening, is the constant monitoring of the previous green and red bars. The color of the bar is  
 determined by its opening and closing price. When the close is greater than the open, the bar is  
 represented with green. Conversely, when the close is less than the open, the bar is represented  
 with red. At any given time, only the most recent green and red bars are tracked. When the open  
 of the next bar breaks the high of the previous red bar the system considers entering the market.   
 The last condition that must be met for the strategy to enter the market depends on the  
 difference between the open of the current bar and the VWAP. When building the strategy, it  
 was apparent that the majority of losing trades resulted in a stock channeling between a small  
 price difference. In order to prevent the strategy from entering into conditions like this, the  
 VWAP can be leveraged to determine “false” breakouts. During a long period of channeling,  
 50  

 
 using VWAP data from the previous hour of trading will yield a value relatively close to the  
 actual price of the stock. Therefore, taking the difference between the open of a bar and the  
 current VWAP value should produce a number close to zero. Based on this behavior, it can be  
 assumed that a larger gap between the open of the current bar and the VWAP indicates a  
 potential breakout. Incorporating this logic into the entry condition drastically reduced the  
 amount of losing trades as shown below in figures 4.2 and 4.3.  
 Figure 4.2 Entries prior to VWAP filter  
 51  

 
 Figure 4.3 Entries after applying VWAP filter  
 4.1.4 Results  
 Depending on the available capital for each position returns vary. Allowing more money  
 to be allocated in a position can drastically improve the overall returns. Initially, 15% of the  
 capital was provided as purchasing power for each position. It is important to note that this does  
 equate to a $15,000 risk since a stop loss is used to prevent completely losing the entire position.  
 Based on these parameters, the system made a 5% return over the course of a two month period  
 executing 133 in total as shown in figure 4.4. The largest loss was $1,717.60 with a max  
 drawdown of $2,801.24.  
 52  

 
  
 Figure 4.4 Pre-market gap strategy with 15% available capital per position  
 Increasing the available capital per position to 50% produced a return of 21.86% over the  
 same two month period and 133 trades as shown in figure 4.5. However, the largest losing trade  
 increase to $5,689.60 and max drawdown to $9,189.24.  
 53  

 
  
 Figure 4.5 Pre-market gap strategy with 50% available capital per position  
  
 4.2 Whole Breakouts  
 This system is a breakout system, meaning that it looks for quick bursts in price  
 movement, and typically trades on a short to medium time period.  
 4.2.1 Hypothesis  
 This strategy trades around the type of numbers commonly referred to as “The Wholes.”  
 The Wholes are numbers such as 1, 5, and 10 that seem appealing to people. In a scenario where  
 a stock is decreasing in price, many amateur investors will settle on The Wholes as exit points to  
 preserve the rest of their capital, while many others will eye these prices as good entry points;  
 they may think, “$10.00 is the lowest I will hold,” or “Wow! $10.00 is a great deal!” Similarly,  
 there are many investors that will set stop-losses near The Wholes. Large investors tend to be  
 aware of this; and they can abuse this fact by selling shares, thereby dropping the price below  
 54  

 
 trader’s stop-losses, then buy the shares that those stop-losses are selling at a discounted price.  
 Finally, the stock will increase in price, as the larger investors expected the whole time, and  
 those investors will cash in on the profits from the shares that they were originally holding plus  
 the shares that they purchased from other traders’ stop-losses.  
 The end result of such behavior is the formation of predictable patterns when there is a lot  
 of price-action in stocks. A stock that is worth, for example, $10.50 and decreasing may see a lot  
 of investors buying shares when it finally drops to $10.00. This increase in buying pressure tends  
 to either stop a decrease or result in an increase in price. At this point, one of two things tends to  
 happen. The price either “bounces off” the $10.00 level, due to an increase in buy orders, or it  
 breaks below it due to traders selling lower and lower. Around this time is where this trading  
 system comes in. The goal for this system is to detect this sort of price-action around The  
 Wholes and subsequently enter when it seems clear which direction the price is going.  
  
 4.2.2 Methodology  
 In order to actually put this system to work, it is important to first identify what “Whole”  
 increment to use. Depending on the stock, it could be $1, $10, or some other similar number.  
 Generally, it is easy to estimate what will work best for a given stock. For instance, if some stock  
 is currently worth $12.75, one could guesstimate that $0.50 or $1.00 could work. To find the  
 interval that works best for a symbol, a good method to use is to use an optimization tool. After  
 optimising, it is generally a good idea to use Walk-Forward Analysis to ensure that whatever  
 value chosen could work in the future.  
 55  

 
 Next, it is key to decide at what point some price action near the Whole level should be  
 considered to be “testing” it. A strict rule, only considering it “testing” if the price crosses over  
 the level, could be used; but this may not catch moves that are just barely out of reach. Less strict  
 rules could be used, but if they are too loose then it could catch moves that are not even close.  
 One rule that could be used is if the price is within some percentage of the Whole increment  
 relative to the nearest Whole level. The exact percentage could be optimized for, but the goal is a  
 rule that is consistent and strict, while not too strict.  
 Third, after a level has been tested or broken through, the next step is entering a trade. It  
 may not be apparent which direction the price will go in, as this system is based on the idea that  
 the price could break-out up or down at any Whole level. A rule such as entering when the price  
 has crossed over the Whole level plus or minus some percentage of the Whole increment could  
 help. As usual, the exact percentage could be determined with optimization. Waiting for the price  
 to move a certain amount before entering a trade will make choosing the correct trade more  
 likely. In addition to choosing when to enter a trade, it is important to choose a position sizing  
 strategy. Going all-in on every trade can be very profitable when a system makes winning trades,  
 but it hurts a lot more when it loses. If a system makes a trade that increases its capital by ten  
 percent, then a trade that decreases its capital by ten percent, then the system lost money overall.  
 This applies, of course, for any amount of money that is put into a trade, but it hopefully  
 illustrates the point that losing hurts more than winning helps.  
 Finally, the system has to exit a trade somehow. “Buy low, sell high” is a great idea, but  
 determining when “high” is can be difficult. Also, not all trades are going to be profitable. It is  
 key to choose an exit strategy that minimizes losses and maximizes gains. It may mean that not  
 56  

 
 every trade sells at the exact peak value, but rather over all of a system’s trades it exits at an ideal  
 point. There are many different exit strategies, and which one to use is entirely up to personal  
 preference, though it usually aligns with which one happens to make the most money!  
  
 4.2.3 Entry/Exit Conditions  
 The actual implementation of these rules for this project follow. To determine the Whole  
 increment, optimization is used to determine what could work best for each symbol. For  
 example, when backtesting over 500 days on $AMD, optimization determined that $0.50  
 achieved the greatest returns in that period. In contrast, over 500 days on $SPY, optimization  
 determined $1.50 to achieve the greatest returns. Which interval to use depends on the symbol,  
 so there is no one-size fits all rule for this. Next, the system decides that a level is being tested if  
 the price moves within ten percent of the Whole increment of the level. Third, it enters a trade  
 when the price moves ten percent away from the Whole level. Finally, it uses a Parabolic SAR  
 rule, combined with simple moving average. If the two indicators align to signal an exit, then the  
 system exits the trade, regardless of profit or loss.  
 In addition to these rules, a profit target exit strategy was included in this system. This is  
 because the system occasionally would stay in a profitable trade, only for it to gap in the opposite  
 direction, resulting in a loss. The profit target is optimized for the best value to avoid losing  
 still-good profits.  
  
 57  

 
 4.2.4 Results  
 The system was tested on stocks over one year, utilizing all of its equity for each trade,  
 starting with $100,000. It earned $28,945.55 in this time period, which is 28.95%. The  
 percentage of winning trades was 32.92%, while the ratio of average winning trade to average  
 losing trade was 2.7. Looking at the equity curve, figure 4.6, the system has its ups and downs.   
  
 Figure 4.6: Whole Breakouts Equity Curve  
 As a day-trading system, it trades very frequently; and given that it loses more frequently  
 than it wins, it tends to swing back and forth fairly consistently. However, looking at the  
 Monte-Carlo Analysis, figure 4.7, the actual sequence of trades fits mostly within the two  
 confidence intervals. This generally indicates that the Monte-Carlo Analysis is a good measure  
 of a system’s ability to perform, and since the system won more than it lost overall, this is a good  
 sign.  
 58  

 
  
 Figure 4.7: Whole Breakouts Monte-Carlo Analysis from Market System Analyzer  
  
 4.3 Kaufman Efficiency Day Trader  
 For trend following systems, it is possible that once a position has been entered, the trend  
 could flip at any moment. This is not ideal. Ideally, a system should enter a position only when it  
 is confident that the market conditions will continue to fit its entry conditions. In the case of a  
 trend following system, when it buys in, one would hope that the price of the stock will climb,  
 then the system exits.   
  
 4.3.1 Hypothesis  
 Once a trend has been discovered, a trader can measure the efficiency during the trend to  
 determine the strength of the trend. A higher efficiency might indicate that a trend is stronger,  
 and is more likely to continue in its current direction. A lower efficiency would indicate less  
 confidence, and there would be a lower chance that the trend will continue.   
 59  

 
  
 4.3.2 Methodology  
 To measure efficiency, the Kaufman Efficiency Ratio is used. It is calculated by  
 calculating the price change over a period and dividing it by the absolute sum of the price  
 movements during that period.   
 In determining which stocks to pick, a screener was used. The goal was to find stocks that  
 are likely to increase in price, so that way the system is more likely to enter into a winning trade.  
 A CAN SLIM Screener was selected from aaii.com. Very basically, the CAN SLIM screener is  
 designed to find stocks that have performed well last year and are performing well this year.   
 The position sizing strategy is fixed fractional, meaning that it will risk the same  
 percentage of account equity on each trade. So if the system makes money on one trade, then it  
 will be able to risk more money on the next trade because the overall account equity has gone up.  
  
 4.3.3 Entry/Exit Conditions  
 The system will enter a trade based on two conditions. The first condition is that it has  
 detected an upward trend. The second condition is that the measured efficiency has to be above a  
 certain threshold. This system was developed into two main versions. One version will sell when  
 it has made a profit, or if a stop loss is triggered. The other version will sell only when it has  
 made a profit.   
  
 60  

 
 4.3.4 Results  
 The version that has stop losses in general did not perform very well. Often times it  
 would enter in at the right moments, but the stop losses would always trigger at inopportune  
 times. The problem was that adjusting a stop loss would just shift the problem from one spot to  
 another. With a less sensitive stop loss, it would stay in trades too long, and with an over  
 sensitive stop loss, it would exit too fast. Every system has tradeoffs to make, but in this system  
 both situations were performing negatively, so the system was tested without any stop losses.  
 The system without any stop losses performed well situationally. The only exit condition  
 was that it would sell when it has made a profit, so if the price of the stock was going up there  
 was not an issue, it would make money easily. The part of this system that is not appealing is the  
 drawdown. For example, on a stock that has doubled in price over the past year, the system made  
 about 8%, with a maximum drawdown of 1%. But on an ETF that is extremely volatile and has  
 gone down substantially in the past year, it had realized gains of about 1.7%, and had a position  
 in the market that was open for 8 months, and was down 5.6%.   
 Based on these results, the second version of this system is definitely viable, but selecting  
 the right stocks is extremely important. It is easy to do in hindsight, but one can not know ahead  
 of time if a stock is going to perform well. Pairing this strategy with a screener is almost  
 essential. Additionally, it would be a good idea to manually monitor some of the trades that this  
 system makes, because often times when it has entered a trade, it will miss other opportunities  
 between when it buys and sells.   
 As a final test, the system was run on the symbol SPY with an initial equity of $100,000.  
 It wound up making 26% in a year, with an open trade worth -$2,600, so adjusted it made about  
 61  

 
 23%. The SPY symbol was up about 18% over the past year when this test was performed. One  
 thing to note with this set of results was that when the initial equity was only $10,000 the system  
 did not perform well because it was losing a lot of money to commission. Increasing the starting  
 equity improved its performance by a huge magnitude (it went from making $267 to making  
 $26,000).   
 Figure 4.8 Detailed Equity Curve of Kaufman Efficiency Trading System  
  
 62  

 
  
 Figure 4.9 SPY Symbol over the Past Year  
 As can be seen in these figures, the equity curve of the trading system heavily resembles  
 the SPY symbol. This indicates that the performance of the system is dependant on the  
 underlying symbol choice. One difference that should be noted is towards the end, the SPY  
 symbol takes a big dip that the system manages to avoid.  
  
  
 Figure 4.10 Examples of Successful Trades  
 63  

 
 In the above figure, one can see a series of successful trades. This is close to optimal  
 performance. One aspect that the system could have done better with is not buying at the peak on  
 the left side of the image, and instead buying at the dip right after it. Another inefficiency is  
 when it buys and sells rapidly, such as on the left side of the image, trades labeled 1 through 6.  
 The issue with this is that while it is making money, it is spending money on commission more  
 than it needs to. It makes 6 trades where it could have made 1 trade (A to B) and still make the  
 same amount but spending less on commission.   
  
  
 Figure 4.11 Example of Drawdown in System  
 In the above figure, one can observe a single trade that is subject to a large amount of  
 drawdown. The profit was roughly $0.4 per share, while the maximum drawdown on that trade  
 was roughly $1.4 per share. While the trade was still open, there were potentially 6 other buying  
 opportunities that the system missed because it was still in a trade. 5 out of 6 of these would have  
 been executed and were likely to make profits. The point of this is to show that this system is  
 64  

 
 imperfect. If it buys right before a dip, then it will miss all of the trades that could have happened  
 in the dip.   
  
 4.4 Forex Stop and Reverse  
 4.4.1 Hypothesis  
 Theoretically, the ideal stop and reverse system for day trading currencies should be able  
 to anticipate every change from bullish to bearish and vice versa. It would short or long the  
 currency based on the predicted direction. One common method to design a stop and reverse  
 system is to calculate channels where resistance and support lines lie, and place trades when  
 those lines are being tested. Some common examples are Keltner Channels, Bollinger bands and  
 Parabolic SAR. However, an obvious drawback from this design is that when the price of a  
 currency fluctuates heavily and breaks out from those support and resistance lines, the system  
 will have the exact wrong order placed, and you will either suffer heavy losses or have your stop  
 orders filled. In order to account for this drawback, this system was designed to detect breakouts  
 and breakdowns from resistance in order to profit from these deviations as well. It is a stop and  
 reverse system designed to profit from everyday long and short channel trading, but also predicts  
 and properly trades breakouts and breakdowns.   
  
 4.4.2 Methodology  
 Development started by optimizing and comparing a few different common stop and  
 reverse systems, primarily Keltner Channels and Bollinger Bands, but also Parabolic SAR.  
 EURUSD was used for testing, as it is the highest volume currency pair so it should be a great  
 65  

 
 baseline. Higher volume should mean a more stable market with more predictable channels.  
 After backtesting 6 months with all three stop and reverse systems, while the Bollinger Bands  
 strategy had more winning trades, the Keltner channel generated more profit, still with a greater  
 than 50% success rate. All backtesting was done with $10 for round trip commission plus  
 slippage.   
  
 Figure 4.12 Equity Curve - EURUSD 60 min bars (8/13/2017 - 2/142018)  
 For further confirmation that keltner channels would be a more promising strategy, testing was  
 done and optimized with various stop-loss strategies, including currency trailing and percentage  
 based stops. Based on these results, the bollinger band strategy could not be improved over basic  
 keltner channels with any stop loss or profit taking strategies tested.   
 To advance the strategy with a method for detecting breakouts and breakdowns from the  
 regular channel day-trading, the system was tested with a number of indicators that predict  
 66  

 
 strong market trends, including RSI, momentum and MACD. Before every trade placed by the  
 optimized channel strategy, it would check these indicators for bullish or bearish trends, then  
 reverse the channel strategy's decision if triggered. For example, if the Keltner channels want to  
 reverse the position at the top of the channel, but the Momentum indicator shows a strong bullish  
 trend, it will override the short. The goal was to find which of these indicators would have the  
 greatest success rate at predicting these trends. Using MACD resulted in <50% profitable trades,  
 even after thorough optimization, so MACD did not look like a promising indicator. RSI  
 (Relative Strength Index) however, did predict breakouts and breakdowns >50% of the time. RSI  
 is an indicator that interprets when a security's recent price performance signals that it is  
 overbought or overvalued. There were also good results with momentum, with many profitable  
 trades in initial testing. Momentum is used to show when the rate of growth of a security's price  
 is accelerating or decelerating. Testing followed through with optimization of both a momentum  
 based breakout detection, and a RSI based one. After a few days of tweaking time frames and  
 optimizing variables, Momentum became a clear favorite, with more predictions and profitable  
 trades on several time frames, including 15 minutes, 30 minutes and 60 minutes. Out of these  
 three time frames, 60 minutes turned out to be the most profitable once accounting for slippage  
 and commission. Shorter time frames would lose too much money from commission, more  
 trades would mean less profit.   
  
 4.4.3 Entry/Exit Conditions  
 This system will place its first trade at the top or bottom of an optimized keltner channel;  
 long at the bottom or short at the top. After that, it remains in the market, and when the price  
 67  

 
 reaches the top or bottom of a channel again, it will decide whether to maintain its position based  
 on momentum. If momentum is bullish at the top, or bearish at the bottom, i.e. the price is  
 accelerating at the edge of a channel, the system will maintain the current position. If the price is  
 decelerating based on momentum at the top or bottom of the channel, it will reverse the position,  
 going short at the top and long at the bottom.  
  
  
 Figure 4.13 Example Positions - EURUSD 60 min bars  
 4.4.4 Results  
 The final optimized design produced 16.9% profit over 6 months of backtesting, with  
 53.7% profitable trades. The average winning trade made $517.26, whereas the average losing  
 trade lost $260.74. All testing was done with $10 commission. This system also turned out to be  
 profitable trading GBPUSD, but with only 48.54% profitable trades. The GBPUSD system made  
 22.6% profit, but was much more risky due to fewer profitable trades, and larger average losses.  
 Based on these results, there is indication that the concept of this system can be implemented on  
 any currency with a high volume that lends itself to thorough technical analysis.  
 68  

 
 Additionally, Monte Carlo analysis showed that this system had a 90% chance of >4%  
 return, 80% chance of >8% return, and 70% chance of >10% return over the tested 6 months.  
 This is a very strong indication of the profitability of this system even when accounting for  
 random variation. Using a Monte Carlo analysis of the max drawdown of the system over 6  
 months, the mean drawdown turned out to be 3.82% of the portfolio, with a 10% chance of  
 greater than 6% drawdown. This does indicate some level of risk that the investor needs to take  
 into account.  
   
 69  

 
 Chapter 5: Results  
 After developing the systems, each was backtested against various securities pertaining to  
 the individual systems. After collecting all of the trades made by the systems, the results were  
 combined to form a system of systems. The results are detailed below in this order: Pre-Market  
 Gaps, Whole Breakouts, Kaufman Efficiency Day Trading, Forex Stop and Reverse, and finally  
 the results for the System of Systems.  
 Each trading system’s results are presented in a table, which contains various  
 information, including but not limited to: starting equity, percentage of equity traded, return on  
 equity, expectancy, expectunity, system quality, and sharpe ratio. These values were calculated  
 using a free trial of Market System Analyzer, which is software created by Adaptrade Software.  
 The starting equity for each trader’s individual system was $100,000; when these are combined  
 into a system-of-systems, the starting equity is simply the total starting equity of each of the  
 individual systems, which equals $400,000. Each system also lists the percentage of equity  
 traded. This number describes a fixed percentage of the equity that is used to purchase assets on  
 each trade. So if a system lists percentage of equity traded as 100% and it makes $50,000 on its  
 first trade, then it will use $150,000 on its next trade. Similarly, if it uses 10% of its equity and it  
 makes $50,000 on its first trade, then it will trade $15,000 on its next trade, leaving $135,000  
 outside of the trade. Return on equity, however, is based on the starting equity. Increasing or  
 decreasing the percentage of equity that a system trades can increase or decrease the return on  
 equity, but the metric is calculated based on the final equity and the starting equity.  
 The next metric, expectancy, is one that describes how much a system is expected to earn  
 for each dollar risked per trade. Recall that risk is the amount of money that a trader is willing to  
 70  

 
 lose before exiting the trade to preserve his capital. It is common to use a system’s average loss  
 or maximum loss as the risk for each trade. So if a trading system has an expectancy of $0.50  
 and it risks $2,000 on each trade, then it is expected to make $1,000 on average for each trade.  
 Similarly, expectunity measures this value over one calendar year. It is simply the expectancy  
 multiplied by the average number of trades that a system makes over one calendar year. So if the  
 above system makes one trade each day, its expectunity would be 365*$0.50, or $182.50. This is  
 meant to give the trader an idea of how a system might perform over the long-term. If this  
 system continues to risk $2,000 on each trade, then it would be expected to earn $365,000 each  
 year!  
 Another useful metric that is included is system quality. This number is meant to rate the  
 overall quality and versatility of a trading system. It is dimensionless, so on its own it can be  
 hard to decipher. However, when comparing the system quality of one trading system to that of  
 another, it can be easier to understand. Generally, a higher system quality is better, but on its  
 own it does not necessarily mean one system is better than another; it is important to analyze all  
 aspects of a system before deciding if it is better than another. Even a system that seems better in  
 backtesting may not perform well when trading in real-time. Of course, the system quality can  
 help determine this, as can the sharpe ratio.  
 As expectunity is a measure of the reward for one dollar of risk, the sharpe ratio helps  
 determine if the risk is worth taking. The sharpe ratio of a trading system is a measure of its  
 risk-adjusted return; it is inherently benchmarked against risk-free assets, so a negative or zero  
 sharpe ratio indicates that it would be better to invest in risk-free assets, like U.S. Treasury  
 bonds. All of the trading systems featured below have sharpe ratios that are greater than zero.  
 71  

 
 This is great because it means that these systems were better investments than risk-free bonds for  
 backtesting. If results such as these persist when testing in real-time, it could mean the system is  
 worth trading.  
 Sharpe ratio, system quality, and expectancy are just a few of the metrics that are helpful  
 when analyzing a system. Just because a few of these numbers look good on paper, however,  
 does not mean that a system is guaranteed to perform well. Instead, it is essential to take every  
 aspect of a trading system, from its most basic rules to the metrics discussed above, into  
 consideration when evaluating it. Only then is it possible to make comparisons between systems.  
  
 5.1 Pre-Market Gaps  
 Starting Equity  $100,000  
 Percentage of Equity Traded  15%  
 Return on Equity  5.037%  
 Trading Duration  February 12th, 2018 - April 5th, 2018  
 Number of Trades  133  
 Maximum Open Positions  3  
 Win ​  ​ Ratio  2.524  
 Expectancy (Average Loss)  0.41  
 Expectancy (Max Loss)  0.02  
 Expectunity (Average Loss)  373.77  
 Expectunity (Max Loss)  20.58  
 System Quality  1.2295714  
 Sharpe Ratio  5.355  
 72  

 
  
 The Pre-Market Gaps strategy began trading on February 12th, 2018, and made its final  
 trade on April 5th, 2018, which is a span of 52 calendar days. In total, it made 130 trades. The  
 expectancy for this system based on the average loss was $0.41, and based on the maximum loss  
 the expectancy was $0.02. These values multiplied by the number of opportunities gives  
 expectunity. The number of opportunities calculated for this system was 912.5. So, this system’s  
 expectunity based on the average loss was $373.77, and based on the maximum loss was $20.58.  
 Lastly, this system’s quality was calculated to be 1.2295714.   
  
 5.2 Whole Breakouts  
 Starting Equity  $100,000  
 Percentage of Equity Traded  100%  
 Return on Equity  28.95%  
 Trading Duration  April 7th, 2017 - April 4th, 2018  
 Number of Trades  161  
 Maximum Open Positions  1  
 Win ​  ​ Ratio  2.7  
 Expectancy (Average Loss)  0.22  
 Expectancy (Max Loss)  0.03  
 Expectunity (Average Loss)  35.49  
 Expectunity (Max Loss)  4.20  
 System Quality  1.092  
 Sharpe Ratio  0.3177  
 73  

 
  
 The Whole Breakouts strategy began trading on April 7th, 2017, and made its final trade  
 on April 4th, 2018, which is a span of 361 calendar days. In total, it made 161 trades. The  
 expectancy for this system based on the average loss was $0.22, and based on the maximum loss  
 the expectancy was $0.03. These values multiplied by the number of opportunities gives  
 expectunity. The number of opportunities calculated for this system was 162. So, this system’s  
 expectunity based on the average loss was $35.49, and based on the maximum loss was $4.20.  
 Lastly, this system’s quality was calculated to be 1.092.   
 5.3 Kaufman Efficiency Day Trading  
 Starting Equity  $100,000  
 Percentage of Equity Traded  100%  
 Return on Equity  24.92%  
 Trading Duration  April 19th, 2017 - March 14th, 2018  
 Number of Trades  238  
 Maximum Open Positions  1  
 Win ​  ​ Ratio  1.191  
 Expectancy (Average Loss)  1.15  
 Expectancy (Max Loss)  0.74  
 Expectunity (Average Loss)  304.62  
 Expectunity (Max Loss)  195.79  
 System Quality  11.91499143  
 Sharpe Ratio  1.099  
  
 74  

 
 The Kaufman Efficiency Day Trading strategy began trading on April 19th, 2017, and  
 made its final trade on March 14th, 2018, which is a span of 329 calendar days. In total, it made  
 238 trades. The expectancy for this system based on the average loss was $1.15, and based on the  
 maximum loss the expectancy was $0.74. These values multiplied by the number of  
 opportunities gives expectunity. The number of opportunities calculated for this system was  
 263.96. So, this system’s expectunity based on the average loss was $304.62, and based on the  
 maximum loss was $195.79. Lastly, this system’s quality was calculated to be 11.91499143.   
  
 5.4 Forex Stop and Reverse  
 Starting Equity  $100,000  
 Percentage of Equity Traded  100%  
 Return on Equity  19.93%  
 Trading Duration  October 16, 2017 - April 11, 2018  
 Number of Trades  105  
 Maximum Open Positions  1  
 Win ​  ​ Ratio  2.42  
 Expectancy (Average Loss)  0.79  
 Expectancy (Max Loss)  0.22  
 Expectunity (Average Loss)  171.35  
 Expectunity (Max Loss)  48.52  
 System Quality  3.02  
 Sharpe Ratio  1.05  
  
 75  

 
 The Stop and Reverse Forex strategy began trading on October 16th, 2017, and made its  
 final trade on April 11th, 2018, which is a span of 177 calendar days. In total, it made 105 trades.  
 The expectancy for this system based on the average loss was $0.79, and based on the maximum  
 loss the expectancy was $0.22. These values multiplied by the number of opportunities gives  
 expectunity. The number of opportunities calculated for this system was 216.53. So, this  
 system’s expectunity based on the average loss was $171.35, and based on the maximum loss  
 was $48.52. Lastly, this system’s quality was calculated to be 3.02.   
  
 5.5 System of Systems  
 Starting Equity  $400,000  
 Percentage of Equity Traded  78%  
 Return on Equity  19.73%  
 Trading Duration  April 7th, 2017 - April 4th, 2018  
 Number of Trades  634  
 Maximum Open Positions  4  
 Win ​  ​ Ratio  1.05  
 Expectancy (Average Loss)  0.27  
 Expectancy (Max Loss)  0.02  
 Expectunity (Average Loss)  172.63  
 Expectunity (Max Loss)  11.23  
 System Quality  2.85  
 Sharpe Ratio  0.79  
  
 76  

 
 The System of Systems was a combination of all of our trading systems. It made a total of  
 634 trades. The expectancy was $0.27 based on average loss, and $0.02 based on maximum loss.  
 The total opportunities per year was 628.83. The expectunity calculated was $172.63 based on  
 the average loss, and $11.23 based on the maximum loss. The calculated system quality was  
 2.85.  
  
 Figure 5.1 System of Systems Equity Curve  
 The above figure depicts the equity curve of the combined system of systems. It can be  
 observed that over the past year the system of systems made roughly $78,920, or 19.73% profits  
 per year.  
 77  

 
   
 Figure 5.2 Monte Carlo analysis of the System of Systems from Market System Analyzer  
 It can be observed in the figure above that the majority of the system’s performance falls  
 within the values that the Monte Carlo analysis was confident in.  
   
 78  

 
 Chapter 6: Conclusions  
 A major takeaway from the results of this IQP is that there is a lot of variability in the  
 market and therefore expecting one system to perform well for an extended period of time puts a  
 trader at a severe disadvantage. By combining the systems, the system quality surpassed the  
 majority of the individual systems. This shows that while a trader could just use one of his good  
 systems, he would be better off using a combined system of systems.   
   
 79  

 
 Chapter 7: Recommendations for Future Projects  
 Prior to participating in this IQP, each of the group members had some exposure to  
 trading and investing as a result of buying and selling stocks through Robinhood. ​ [ ] ​  Over the  28
 course of this IQP, each of the group members has been able to identify significant growth in his  
 knowledge of trading and investing in addition to an ability to make better informed decisions.  
 The following sections describe the pitfalls experienced early on in the project and what would  
 have improved the process, having overcome these challenges.   
  
 7.1 Adhering to a Methodology  
 During the early stages of developing an automated system it can be tempting to abandon  
 previous work when losing money. Following this behavior can be detrimental to the progression  
 of the system in the long run. Prior to developing an automated system, it should be clear that  
 fully and correctly implementing a methodology will produce results similar to what is expected.  
 While the system is still being developed, the actual returns might be drastically different than  
 what is expected. For this reason, it is important to implement a methodology in stages and avoid  
 becoming too concerned with its performance until fully implemented.  
  
 7.2 Identifying Noise  
 Noise was the number one factor that affected the profitability of the strategies. In trying  
 to identify noise, several indicators were used, but none effectively prevented the systems from  
 entering into noise. In hindsight, had one of the members from the group spent more time  
 28  ​ “Robinhood,” ​ Robinhood - Invest for Free ​ . [Online]. Available: https://www.robinhood.com/. [Accessed:  
 29-Apr-2018].  
 80  

 
 attempting to identify noise, it would have drastically benefitted all of the systems and increased  
 profitability.   
   
 81  

 
 References  
 1. “American Psychological Association Survey Shows Money Stress Weighing on  
 Americans’ Health Nationwide,” ​ American Psychological Association ​ , 04-Feb-2015.  
 [Online]. Available: http://www.apa.org/news/press/releases/2015/02/money-stress.aspx.  
 [Accessed: 25-Apr-2018].   
 2. I. Staff, “Money Manager,” ​ Investopedia ​ , 02-Jan-2018. [Online]. Available:  
 https://www.investopedia.com/terms/m/moneymanager.asp. [Accessed: 26-Apr-2018].  
 3. I. Staff, “Mutual Fund,” Investopedia, 15-Mar-2018. [Online]. Available:  
 https://www.investopedia.com/terms/m/mutualfund.asp. [Accessed: 29-Apr-2018].  
 4. C. F. A. A. Hayes, “Exchange-Traded Fund (ETF),” ​ Investopedia ​ , 03-Apr-2018.  
 [Online]. Available: https://www.investopedia.com/terms/e/etf.asp. [Accessed:  
 29-Apr-2018].  
 5. I. Staff, “Foreign Exchange Market,” ​ Investopedia ​ , 14-May-2008. [Online]. Available:  
 https://www.investopedia.com/terms/forex/f/foreign-exchange-markets.asp. [Accessed:  
 29-Apr-2018].  
 6. D. Harper, “Getting to Know the Stock Exchanges,” ​ Investopedia ​ , 05-May-2017.  
 [Online]. Available: https://www.investopedia.com/articles/basics/04/092404.asp.  
 [Accessed: 25-Mar-2018].  
 7. C. F. A. A. Hayes, “IPO Basics: What Is An IPO?,” ​ Investopedia ​ , 24-Mar-2017.  
 [Online]. Available: https://www.investopedia.com/university/ipo/ipo.asp. [Accessed:  
 25-Mar-2018].  
 8. “The New York Stock Exchange,” ​ NYSE: Listings Directory - Stocks ​ . [Online].  
 Available: https://www.nyse.com/listings_directory/stock. [Accessed: 26-Mar-2018].  
 9. S. Seth, “How The NYSE Makes Money,” ​ Investopedia ​ , 05-May-2015. [Online].  
 Available:  
 https://www.investopedia.com/articles/investing/050515/how-nyse-makes-money.asp.  
 [Accessed: 25-Mar-2018].  
 82  

 
 10. J. Folger, “Index Investing: What Is An Index?,” ​ Investopedia ​ , 05-Dec-2017. [Online].  
 Available: https://www.investopedia.com/university/indexes/index1.asp. [Accessed:  
 25-Mar-2018].  
 11. J. Folger, “Index Investing: The Standard & Poor's 500 Index,” ​ Investopedia ​ ,  
 05-Dec-2017. [Online]. Available:  
 https://www.investopedia.com/university/indexes/index3.asp. [Accessed: 25-Mar-2018].  
 12. I. Staff, “Market Sentiment,” ​ Investopedia ​ , 18-Jan-2017. [Online]. Available:  
 https://www.investopedia.com/terms/m/marketsentiment.asp. [Accessed: 26-Apr-2018].  
 13. E. F., K. R., and Fama, “Size, value, and momentum in international stock returns, by  
 Fama, Eugene F.; French, Kenneth R.,” ​ Journal of Financial Economics ​ , 01-Jan-1970.  
 [Online]. Available: https://ideas.repec.org/a/eee/jfinec/v105y2012i3p457-472.html.  
 [Accessed: 26-Apr-2018].  
 14. I. Staff, “Efficient Market Hypothesis (EMH),” ​ Investopedia ​ , 28-Sep-2016. [Online].  
 Available: https://www.investopedia.com/terms/e/efficientmarkethypothesis.asp.  
 [Accessed: 26-Apr-2018].  
 15. I. Staff, “Portfolio Management,” ​ Investopedia ​ , 03-Dec-2015. [Online]. Available:  
 https://www.investopedia.com/terms/p/portfoliomanagement.asp. [Accessed:  
 26-Apr-2018].  
 16. “Award-Winning Trading Tools and Technology,” ​ TradeStation ​ . [Online]. Available:  
 https://www.tradestation.com/technology/. [Accessed: 26-Apr-2018].  
 17. I. Staff, “Technical Indicator,” ​ Investopedia ​ , 18-Mar-2018. [Online]. Available:  
 https://www.investopedia.com/terms/t/technicalindicator.asp. [Accessed: 29-Apr-2018].  
 18. I. Staff, “Volume,” ​ Investopedia ​ , 09-Nov-2017. [Online]. Available:  
 https://www.investopedia.com/terms/v/volume.asp. [Accessed: 26-Apr-2018].  
 19. I. Staff, “Volume Weighted Average Price - VWAP,” ​ Investopedia ​ , 25-Jul-2015.  
 [Online]. Available: https://www.investopedia.com/terms/v/vwap.asp. [Accessed:  
 26-Apr-2018].  
 20. “Kaufman's Efficiency Ratio (ER) « ETF HQ,” ​ ETF HQ ​ . [Online]. Available:  
 http://etfhq.com/blog/2011/02/07/kaufmans-efficiency-ratio/. [Accessed: 26-Apr-2018].  
 83  

 
 21. I. Staff, “Moving Average Convergence Divergence - MACD,” ​ Investopedia ​ ,  
 13-Mar-2018. [Online]. Available: https://www.investopedia.com/terms/m/macd.asp.  
 [Accessed: 26-Apr-2018].  
 22. “Bollinger Bands,” ​ StockCharts.com ​ , 18-Sep-2017. [Online]. Available:  
 http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_b
 ands. [Accessed: 26-Apr-2018].  
 23. “Keltner Channels,” ​ StockCharts.com ​ , 20-Dec-2017. [Online]. Available:  
 http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:keltner_cha
 nnels. [Accessed: 26-Apr-2018].  
 24. I. Staff, “Fundamental Analysis,” ​ Investopedia ​ , 20-Mar-2018. [Online]. Available:  
 https://www.investopedia.com/terms/f/fundamentalanalysis.asp. [Accessed:  
 26-Apr-2018].  
 25. “5 Important Elements in Fundamental Analysis,” ​ EuroInvestor ​ . [Online]. Available:  
 http://www.euroinvestor.com/ei-news/2012/02/12/stock-school-5-important-elements-in-
 fundamental-analysis/15694. [Accessed: 26-Apr-2018].  
 26. I. Staff, “Technical Analysis,” ​ Investopedia ​ , 24-Apr-2018. [Online]. Available:  
 https://www.investopedia.com/terms/t/technicalanalysis.asp. [Accessed: 26-Apr-2018].  
 27. I. Staff, “Stop Hunting,” ​ Investopedia ​ , 21-Apr-2018. [Online]. Available:  
 https://www.investopedia.com/terms/s/stophunting.asp. [Accessed: 26-Apr-2018].  
 28. “Robinhood,” ​ Robinhood - Invest for Free ​ . [Online]. Available:  
 https://www.robinhood.com/. [Accessed: 29-Apr-2018].  
 29. J. Folger, “Introduction To Order Types,” ​ Investopedia ​ , 05-Dec-2017. [Online].  
 Available: https://www.investopedia.com/university/intro-to-order-types/. [Accessed:  
 17-Apr-2018].  
 30. Investopedia, “Types of Securities Orders,” ​ Investopedia ​ , 05-May-2016. [Online].  
 Available:  
 https://www.investopedia.com/exam-guide/series-7/securities-transactions/types-orders.a
 sp. [Accessed: 17-Apr-2018].  
 84  

 
 31. C. M. Investopedia, “Multiple Time Frames,” ​ Investopedia ​ , 09-Oct-2017. [Online].  
 Available:  
 https://www.investopedia.com/walkthrough/forex/trading-strategies/long-term/time-fram
 es.aspx. [Accessed: 17-Apr-2018].  
 32. C. F. A. E. Picardo, “Stop-Loss Order,” ​ Investopedia ​ , 22-Dec-2017. [Online]. Available:  
 https://www.investopedia.com/terms/s/stop-lossorder.asp. [Accessed: 17-Apr-2018].  
 33. I. Staff, “Forex - FX,” ​ Investopedia ​ , 20-Nov-2003. [Online]. Available:  
 https://www.investopedia.com/terms/f/forex.asp. [Accessed: 17-Apr-2018].  
 34. I. Staff, “Fill Or Kill - FOK,” ​ Investopedia ​ , 20-Nov-2003. [Online]. Available:  
 https://www.investopedia.com/terms/f/fok.asp. [Accessed: 17-Apr-2018].  
 35. I. Staff, “Immediate Or Cancel Order - IOC,” ​ Investopedia ​ , 29-Dec-2015. [Online].  
 Available: https://www.investopedia.com/terms/i/immediateorcancel.asp. [Accessed:  
 17-Apr-2018].  
 36. I. Staff, “Stock,” ​ Investopedia ​ , 29-Mar-2018. [Online]. Available:  
 https://www.investopedia.com/terms/s/stock.asp. [Accessed: 17-Apr-2018].  
 37. I. Staff, “Derivative,” ​ Investopedia ​ , 02-Apr-2018. [Online]. Available:  
 https://www.investopedia.com/terms/d/derivative.asp. [Accessed: 17-Apr-2018].  
 38. I. Staff, “Not-Held Order,” ​ Investopedia ​ , 19-Mar-2018. [Online]. Available:  
 https://www.investopedia.com/terms/n/notheldorder.asp. [Accessed: 17-Apr-2018].  
 39. “Types of Orders,” ​ Investor.gov ​ . [Online]. Available:  
 https://www.investor.gov/introduction-investing/basics/how-market-works/types-orders.  
 [Accessed: 17-Apr-2018].  
 40. “What is Forex?,” ​ FXCM UK ​ . [Online]. Available:  
 https://www.fxcm.com/uk/forex/what-is-forex/. [Accessed: 17-Apr-2018].  
  
   
 85  

 
 Appendix A  
 Whole Breakouts Code  
 Inputs:  
 Clear( 3 ),  
 Increment( 0.5 ),  
 MALength( 15 ),  
 FF( 0.02 ),  
 StartingEquity( 100000 );  
  
 Variables:  
 S( -1 ),  
 R( -1 ),  
 it( 0 ),  
 Cleared( 0 ),  
 MP( 0 ),  
 testR( 0 ),  
 testS( 0 ),  
 lastS( -1 ),  
 lastR( -1 ),  
 ReturnValue( 0 ),  
 AfStep( 0.02 ),  
 AfLimit( 0.2 ),  
 oParCl( 0 ),  
 oParOp( 0 ),  
 oPosition( 0 ),  
 oTransition( 0 ),  
 MA( 0 ),  
 lLong( False ),  
 lShort( True ),  
 PositionSize( 0 ),  
 Equity( 0 ),  
 lastPositiveL( 0 ),  
 lastNegativeH( 0 );  
  
 // Reset looking for entries on the first bar of the day. Helps reduce unwanted trades.  
 If Date <> Date[1] Then  
 Begin  
 lLong = False;  
 lShort = False;  
 End;  
  
 // Current Market Position  
 86  

 
 MP = MarketPosition;  
 ReturnValue = ParabolicSAR( AfStep, AfLimit, oParCl, oParOp, oPosition, oTransition );  
 MA = Average(Close, MALength);  
  
 // Position Sizing  
 Equity = StartingEquity +  NetProfit;  
 PositionSize = (FF*Equity)/Close;  
  
 // Determine if the close is clearing a new support or resistance  
 While it < Clear  
 Begin  
 If Close[it] > R Then  
 Begin  
 Cleared = Cleared + 1;  
 End;  
 If Close[it] < S Then  
 Begin  
 Cleared = Cleared - 1;  
 End;  
 it = it + 1;  
 End;  
 If L[Clear] > R[Clear] or H[Clear] < S[Clear] Then  
 Begin  
 lLong = False;  
 lShort = False;  
 End;  
  
 // Determine if resistance is being tested  
 If H > (R - 0.1*Increment) Then  
 Begin  
 testR = 1;  
 lShort = True;  
 lLong = False;  
 End;  
  
 If H > R Or lShort Then  
 Begin  
 If C < R And (Not(oPosition = 1 And MA > MA[1])) And C < (R - 0.1*Increment)  
 Then  
 Begin  
 // Enter short  
 SellShort ("TS") PositionSize shares next bar at open;  
 lShort = False;  
 End;  
 87  

 
 End;  
  
 // Determine if support is being tested  
 If L < (S + 0.1*Increment) Then  
 Begin  
 testS = 1;  
 lLong = True;  
 lShort = False;  
 End;  
  
 If L < S or lLong Then  
 Begin  
 If C > S And (Not(oPosition = -1 And MA < MA[1])) And C > (S + 0.1*Increment)  
 Then  
 Begin  
 // Enter long  
 Buy ("TB") PositionSize Shares Next Bar at Open;  
 lLong = False;  
 End;  
 End;  
  
 // Update Support and Resistance levels  
 If CurrentBar < Clear Or Cleared >= Clear Or Cleared <= -1*Clear Then  
 Begin  
 S = Close - Mod(Close, Increment);  
 R = S + Increment;  
 If R > R[1] Then  
 Begin  
 lShort = False;  
 End  
 Else  
 Begin  
 lLong = False;  
 End;  
 End;  
  
 // Entries  
 // Also if we have enough info (CurrentBar > Clear)  
 If CurrentBar > Clear Then  
 Begin  
 // Enter long if we increased our support and resistance this bar  
 If Cleared >= Clear Then  
 Begin  
 If testS = 0 Then  
 88  

 
 Begin  
 lLong = True;  
 lShort = False;  
 End;  
 End;  
 // Enter short if we decreased our support and resistance this bar  
 If Cleared <= -1*Clear Then  
 Begin  
 if testR = 0  Then  
 Begin  
 lShort = True;  
 lLong = False;  
 End;  
 End;  
 End;  
  
 // Exits  
  
 If MP = 1 Then  
 Begin  
 // If parabolic is deacreasing and MA is decreasing then exit long  
 If oPosition = -1 And MA < MA[1] Then  
 Begin  
 If MarketPosition > 0 Then Sell ("ParaS") All Shares Next Bar at Open;  
 End;  
 End;  
 If MP = -1 Then  
 Begin  
 // if parabolic is increasing and MA is increasing then exit short  
 If oPosition = 1 And MA > MA[1] Then  
 Begin  
 If MarketPosition < 0 Then BuyToCover ("ParaB") Next Bar at Open;  
 End;  
 End;  
  
 // Reset current bar trackers for the next bar  
 Cleared = 0;  
 it = 0;  
 testR = 0;  
 testS = 0;  
  
 89  

 
 Appendix B  
 Kaufman Efficiency Day Trader Code  
 Input: Period(10), // Distance to look back for EFFRatio  
 PSMeth(3),  
 Param1(2.0),  
 Param2(0),  
 BackTest(True),  
 StEqty(100000),  
 CurEqty(100000),  
 TrRisk(1),  
 MxLoss(1),  
 MaxDD(1000),  
 Stocks(True),  
 UseUnits(False),  
 UnitSize(1),  
 UseMinN(True),  
 MinN(1),  
 MaxN(100000),  
 InitMarg(0),  
 MargPer(100),  
 NATR(20),  
 min_ratio(0.35),  
 Sell_coeff(1.5),  
 Buy_coeff(1.5)  
 ;  
  
 Variables: Position_sizing(0),  
 change(0),   
 noise(0),   
 diff(0),   
 ratio(0),   
 signal(0),  
 open_bar(0),  
 ATR(0),  
 Par_ret(0),  
 AfStep( 0.02 ),  
 AfLimit( 0.2 ),  
 oParCl( 0 ),  
 oParOp( 0 ),  
 oPosition( 0 ),  
 oTransition( 0 ),   
 writesys32(0);  
 90  

 
  
 // KAUFMAN EFFRATIO {  
 ratio = 0;  
 diff = AbsValue(close - close[1]);  
  
 // calculate efficiency starting after the first 'period' bars of the day  
 // detects a new day  
 if(BarDateTime.Day <> BarDateTime[1].Day) then open_bar = currentbar;  
  
  
 if currentbar-open_bar > period then begin   
 change = close - close[period];  
 signal = AbsValue(change);  
 noise = summation(diff,period);  
 ratio = 0;  
 if noise <> 0 then   
 ratio = signal/noise;  
 end;  
 // }  
  
 // determine position sizing using fixed fractional  
 if MarketPosition = 0 Then   
 Position_sizing = PSCalc32(PSMeth, Param1, Param2, BackTest, StEqty, CurEqty,  
 TrRisk,  
  MxLoss, MaxDD, Stocks, UseUnits, UnitSize, UseMinN, MinN, MaxN, InitMarg,  
 MargPer, NATR);  
  
 // Parabolic SAR - currently unused  
 Par_ret = ParabolicSAR( AfStep, AfLimit, oParCl, oParOp, oPosition, oTransition );  
  
 // Entry Condition(s)  
 ATR = AvgTrueRange(period);   
  
 // Only enter when efficiency ratio is above a certain amount  
 Condition1 = ratio > min_ratio;  
  
 // Condition 2 is that the price is going up  
 Condition2 = Close > Close[1] AND Close[1] > Close[2] AND Close[2] > Close[3];  
  
 if Condition1 AND Condition2 then  
 Buy IntPortion(Position_sizing{%/high%}) Shares Next Bar at Market; //  
 position_sizing is the equity to spend, not the number of shares  
  
 // Profit Target (sell when it has made a certain amount of money)  
 91  

 
 if MarketPosition <> 0 AND C >= EntryPrice + Buy_coeff*ATR then  
 Sell("s") next bar at market;  
  
 writesys32 = WriteTrades32(Position_sizing*EntryPrice, 0, 0, period, 1,  
 "c:\efficiency_system_"+SymbolName+".csv");   
  
   
 92  

 
 Appendix C  
 Pre-Market Gaps Code  
 // Day trading gap strategy leveraging pre-market screens  
 // Written by, Stephen Andrews.  
  
 // Version 1 - Basic strategy implementation  
 // Version 2 - Added position sizing.  
  
 // Date of the gap so we only execute trades for this day  
 Input: TradingDate(1180213);  
  
 // Position sizing inputs  
 Input: PSMeth(3);  
 Input: Param1(2.0);  
 Input: Param2(0);  
 Input: BackTest(True);  
 Input: StartingEquity(100000);  
 Input: CurrentEquity(100000);  
 Input: TradeRisk(1);  
 Input: MxLoss(1);  
 Input: MaxDrawDown(1000);  
 Input: Stocks(True);  
 Input: UseUnits(False);  
 Input: UnitSize(1);  
 Input: UseMinN(True);  
 Input: MinN(1);  
 Input: MaxN(100000);  
 Input: InitMargin(0);  
 Input: MarginPer(0);  
 Input: NATR(20);  
  
 Variable: RedBar(Open < Close);  
 Variable: GreenBar(Close > Open);  
 Variable: LastRedBar(0);  
 Variable: LastGreenBar(0);  
 Variable: AvailableCapital(0);  
 Variable: EnteredAt(0);  
 Variable: StopLossAmt(0);  
 Variable: WriteTrades(0);  
 Variable: PositionSize(0);  
  
 // Track last red bar  
 93  

 
 If Close < Open then begin  
 LastRedBar = CurrentBar;  
 end;  
  
 // Track last green bar   
 If Open < Close then begin  
 LastGreenBar = CurrentBar;  
 end;  
  
 // Enter when a green candle breaks the high of the previous red candle. Buy when it breaks  
 the high of the red candle.  
 If MarketPosition = 0 and Open > High[CurrentBar - LastRedBar] and d = TradingDate then  
 begin  
 AvailableCapital = PSCalc32(PSMeth, Param1, Param2, BackTest, StartingEquity,  
 CurrentEquity,   
 TradeRisk, MxLoss, MaxDrawDown,  
 Stocks, UseUnits, UnitSize, UseMinN,   
 MinN, MaxN, InitMargin, MarginPer,  
 NATR);  
 StopLossAmt = Close - (Low[CurrentBar - LastRedBar] - 0.01);  
 PositionSize = (0.15 * StartingEquity)/Close;  
  
 // print(GetSymbolName, " Position size (wrong): ", PositionSize, " Position Size  
 (right): ", PositionSize/Close);  
  
  
 If (Open - mjr_VWA(Close, 50)) > 0.05 then begin;  
 Buy ("Enter") PositionSize shares next bar at market;  
 end;  
 Print("BUYING --- Open: ", Open, " is breaking the high of last red bar: ",  
 High[CurrentBar - LastRedBar]);  
 Print("Shares to buy: ", PositionSize);  
  
 // Set stop loss 1 cent below low of previous red candle  
 //SetStopLoss(Open next bar - (Low[CurrentBar - LastRedBar] - 0.01));  
 //Print("Setting stop loss at: $", Open[CurrentBar + 1] - (Low[CurrentBar -  
 LastRedBar] - 0.01));  
 //Print("VWAP: ", mjr_VWA(Close, 50));  
 end;  
  
 If MarketPosition = 1 and EnteredAt = 0 then begin  
 StopLossAmt = EntryPrice(0) - (Low[CurrentBar - LastRedBar] - 0.01);  
 SetStopLoss(StopLossAmt);  
 EnteredAt = 1;  
 94  

 
 Print("Setting stop loss to: ", StopLossAmt);   
 end;  
  
  
  
 // Exit: When a red candle breaks the low of the previous green candle  
 If LastRedBar = CurrentBar then begin  
 If MarketPosition = 1 and Close < Low[CurrentBar - LastGreenBar] and d =  
 TradingDate then begin  
 Sell this bar on close;  
 Print("SELLING --- Open: ", Open, " is less than the low of last green bar: ",  
 Low[CurrentBar - LastGreenBar]);  
 EnteredAt = 0;  
 end;  
 end;  
  
 // Strategy uses stop loss on a per share basis  
 SetStopShare;  
 SetExitonClose;  
  
 WriteTrades = WriteTrades32(StopLossAmt * PositionSize, 0 , 0 , 10 , 1 ,  
 "\\Mac\Home\Desktop\Trading Systems\Trade Analysis\" + GetSymbolName + ".csv");  
  
  
   
 95  

 
 Appendix D  
 Forex Stop and Reverse Code - Keltner Floors  
 [IntrabarOrderGeneration = false]  
  
 inputs:   
     Price( Close) [DisplayName = "Price"],  
     LengthKeltner(  28) [DisplayName = "LengthKeltner", ToolTip = "Bars for calculating  
 keltner channels"],  
     NumATRs(  1.5) [DisplayName = "NumATRs", ToolTip = "Number of Average True  
 Ranges"],  
     LengthMom( 6) [DisplayName = "LengthMom", ToolTip =  
      "Enter number of bars over which to calculate momentum."];  
   
 variables:   
     Avg( 0 ),  
     Shift( 0 ),  
     LowerBand( 0 ),  
     Setup( false ),  
     CrossingLow( 0 ),  
     Mom( 0 ),  
     Accel( 0 ),  
     writesys32 (0);  
   
   
  
 Avg = AverageFC( Price, LengthKeltner );  
 Shift = NumATRs * AvgTrueRange( LengthKeltner );  
 LowerBand = Avg - Shift;  
 Mom = Momentum( Price, LengthMom );  
 Accel = Momentum( Mom, 1 ); { 1 bar acceleration }  
   
 { CB > 1 check used to avoid spurious cross confirmation at CB = 1 }  
 if CurrentBar > 1 and Price crosses under LowerBand then  
 begin  
     SetUp = true;  
     CrossingLow = Low;  
 end  
 else if Setup and ( Price > Avg or Low <= CrossingLow - 1 point ) then  
     Setup = false;  
  
 if Setup and Mom < 0 and Accel < 0 then  
 begin  
 96  

 
     Sell Short ( !( "KltMomFloorSE" ) ) next bar at CrossingLow - 1 point stop;  
 end  
 else if Setup then  
 begin  
     Buy ( !( "KltMomFloorLE" ) ) next bar at CrossingLow - 1 point stop;  
 end  
  
  
 Forex Stop and Reverse Code - Keltner Ceilings  
 [IntrabarOrderGeneration = false]  
  
 inputs:   
     Price(  Close) [DisplayName = "Price"],  
     LengthKeltner(  22) [DisplayName = "LengthKeltner", ToolTip = "Bars for calculating  
 keltner channels"],  
     NumATRs(  1.5) [DisplayName = "NumATRs", ToolTip = "Number of Average True  
 Ranges"],  
     LengthMom( 11) [DisplayName = "LengthMom", ToolTip =  
      "Enter number of bars over which to calculate momentum."];  
  
  
 variables:   
     Avg( 0 ),  
     Shift( 0 ),  
     UpperBand( 0 ),  
     Setup( false ),  
     CrossingHigh( 0 ),  
     Mom( 0 ),  
     Accel( 0 ),  
     writesys32 (0);  
   
   
  
 Avg = AverageFC( Price, LengthKeltner );  
 Shift = NumATRs * AvgTrueRange( LengthKeltner );  
 UpperBand = Avg + Shift;  
 Mom = Momentum( Price, LengthMom );  
 Accel = Momentum( Mom, 1 ); { 1 bar acceleration }  
  
 { CB > 1 check used to avoid spurious cross confirmation at CB = 1 }  
 if CurrentBar > 1 and Price crosses over UpperBand then  
 begin  
 97  

 
     SetUp = true;  
     CrossingHigh = High;  
     writesys32 = WriteTrades32(100000*EntryPrice, 0, 0, 1.5, 1,  
 "c:\efficiency_system_"+SymbolName+".csv");  
 end  
 else if Setup and ( Price < Avg or High >= CrossingHigh + 1 point ) then  
     Setup = false;  
  
 if Setup and Mom > 0 and Accel > 0 then  
 begin  
     Buy ( !( "KltMomCeilingLE" ) ) next bar at CrossingHigh + 1 point stop;  
 end  
 else if Setup then  
 begin  
     Sell Short ( !( "KltMomCeilingSE" ) ) next bar at CrossingHigh + 1 point stop;  
 End  
  
   
 98  

 
 Appendix E  
 Excerpts from Stephen Andrew’s Trading Journal  
 Thursday, November 30, 2017  
 ● Bitcoin price continues to surge with discussions of US exchanges to begin  
 offering bitcoin futures.  
 ○ Bitcoin has transitioned from a form of payment to a commodity-like  
 investment.  
 ○ Unlike commodities such as gold, bitcoin has no practical application  
 outside of its domain.  
 ○ Price surge of over 1,100% in 2017 alone.  
 ○ Market cap of 190 billion, nearly four times that of Tesla and now  
 surpassing General Electric.  
 ● Gap trading for quick profits?  
 ● Ken Calhoun’s tips for identifying and trading gaps.  
 ○ Candlestick patterns are key.  
 ○ Whole numbers support resistance. Desired entry is after these clear.  
 ○ Be weary of stocks that gap more than 10%.  
 ○ Risk assessment via point spreads based on previous breakout.  
 ○ Two day support stop loss to avoid loss.  
  
 Monday, January 15, 2018  
 ● Can’t trade cryptocurrencies with TradeStation… Gap strategy it is.  
 ● Market is unpredictable. Successful traders trade with trends. Even if general  
 consensus is wrong, the trend will prevail.   
 ● Benjamin Graham’s The Intelligent Investor.  
 ○ Value investing via price-to-earnings ratios, price-to-book ratios,  
 debt-equity, free cash flow, EPS for potential growth.  
 ○ Might be useful to incorporate these into screening criteria.  
 ● Stock screening is vital to a gap strategy.  
 ○ Identify gappers during pre-market hours.  
  
  
 99  

 
 Wednesday, February 7, 2018  
 ● Incorporate sentiment analysis into strategy to identify bearish/bullish attitude?  
 ● Need to pay in order to gain access to pre-market data for screening.  
  
 Excerpts from John Bieber’s Trading Journal  
 Thursday, September 7, 2017  
 ● Use MACD indicator for long entry only if it has enough weight relative to  
 security?  
 ● Friend told me to put all of my money in $AMD. “It’s down to $12 because of  
 market panic over North Korea. Should be up to $15 by Q3 end.” I’ll watch it.  
  
 Monday, November 13, 2017  
  
 ● Door-opening robots  
 ● EMA strategy from Quantopian?  
 ● Price difference checker for arbitrage  
 ● Identify daily highs (support/resistance)  
 ○ recent highs/lows  
 ○ trend lines  
 ○ moving averages  
 ○ fibonacci retracements  
 ○ regression channels  
 ● Bollinger bands  
 ● “2 touches make a line, 3 touches make a trend line”  
  
  
  
  
  
 100  

 
 Wednesday, Dec 6, 2017  
  
 ● “Risk aversion”  
 ● Rules are important  
 ● Look into “short squeeze”  
 ● Random entry/exit can turn a profit with the correct position sizing rule  
 ● Friend told me medical marijuana stocks could be a good idea. Makes sense in  
 the larger scheme.  
  
 Excerpts from Theodore Bieber’s Trading Journal  
 Sunday, September 10, 2017  
 ● Stocks are down because of North Korea news  
 ● Also because of hurricane  
 ● Odds are they will bounce back up about 10%, so picking a few to put money in  
 would be a good idea hypothetically  
  
 Monday, September 11, 2017  
  
 ● Short Equifax/related stocks  
  
 Friday, Dec 29, 2017  
  
 ● A woman’s identity was stolen 15 times after the Equifax breach. This is just  
 one case, there could potentially be millions. The effects could potentially  
 continue for the rest of the affected people’s lives.   
  
 Excerpts from Grant Espe’s Trading Journal  
 Monday, October 30, 2017  
 ● Started testing Ichimoku Indicators and Macd LE on EURUSD  
 101  

 
 ● How will Spain/Catalonia situation affect Euro?  
 ● Look into scientific analysis that can produce replicable results for trading  
 system  
 ● Show creativity and initiative for final project/design  
  
  
 Monday, November 6, 2017  
  
 ● Oil tends to move markets  
 ● Consumer sentiment directly influences profits from goods  
 ● Trump promotes Powell to fed  
 ○ Not an economist  
 ○ Fed concerned with unemployment and inflation  
 ● Trump visit to China, looking for alternative to TPP  
  
  
 Tuesday, Dec 12, 2017  
  
 ● Jobs report did well - 4.1% unemployment, 62% Labor participation, 230,000  
 jobs created  
 ● Fed meeting this week, anticipated to set interest rates  
 ● Janet Yellen final term coming to a close  
 ● President still has to sign tax bill into law  
 ● USD dropped this week, despite good job report because wage growth was not  
 there  
  
  
  
 102