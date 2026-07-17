# Going long or short a futures contract

**Source URL:** https://learn.robinhood.com/articles/going-long-or-short-a-futures-contract/
**Site:** learn.robinhood.com
**Fetch date:** 2026-07-17 (UTC)
**Word count:** 2795
**Status:** OK

---

Updated

January 29, 2025

# Going long or short a futures contract

Robinhood Learn

Democratize Finance For All.

DEFINITION

Going long means buying a futures contract with the expectation that the price of the contract will rise. Going short means selling a futures contract with the expectation that the price of the contract will fall. Going long and buying is a bullish strategy. Going short and selling is a bearish strategy.

## 🤔 Understanding long and short futures trades

When trading futures, you can either take a long or short position, depending on whether you expect prices to rise or fall. Both approaches involve buying and selling contracts but represent different strategies based on your market outlook. While a bullish opinion means you think the market will go up, a bearish one signals that you expect the market to fall.

Going long involves buying a futures contract with the expectation that its price will rise, hopefully, allowing you to sell it later at a higher price for a profit. If the price falls and you sell below the purchase price, you’ll incur a loss. On your trading platform, a long position often appears as a positive number. For example, buying 3 contracts would show as +3.

To close a long position, you sell the contracts you hold, either partially or entirely. Once all contracts are sold, your position becomes "flat" (0 contracts). If you sell more contracts than you own, you move from being long to short. For instance, if you were long 3 contracts and sold 6, your position would change to -3, indicating you’re now short.

Going short involves selling a futures contract with the expectation that its price will fall, allowing you to buy it back later at a lower price for a profit. If you buy them back at a higher price, the trade results in a loss. It’s similar to going long, just in reverse order. In futures trading , you can sell something you don’t own to create a short position without borrowing. A short position appears as a negative number on your platform, such as -3 if you’ve sold 3 contracts to open a position.

To close a short position, you buy back the contracts you've sold. You can reduce your short position by buying back part of it, or fully close the position by buying all contracts. Once all contracts are bought back, your position is flat. Similar to the example above, if you were short 3 contracts and bought 6, your position would change to +3, indicating you’re now long.

While long and short strategies offer potential for profit, it's crucial to remember that futures trading involves significant risks, as losses can exceed the initial investment due to leverage.

### Example

Suppose a trader expects the price of crude oil to rise, so they go long by purchasing 1 crude oil futures contract at $70 per barrel. A few weeks later, the price increases to $75 per barrel, and the trader decides to sell the contract. If each contract represents 1,000 barrels, the $5 per barrel increase results in a $5,000 profit, less any commissions or fees.

Conversely, imagine that a trader expects the Dow Jones Industrial Average to decline, so they go short by selling 1 Dow Jones futures (/YM) contract at 35,000. A few weeks later, the index drops to 34,500, and the trader decides to buy back the contract. Because each /YM contract has a multiplier of $5 per point, the 500-point decline results in a $2,500 profit (500 points x $5), minus any commissions or fees.

## How do you go long or short a futures contract?

As mentioned, going long or short is straightforward—you buy or sell a futures contract. The timing and reasoning behind that decision depends on your overall strategy and market opinion. Meanwhile the "how" is determined by the type of order you choose to execute.

The three main order types used in futures are market , limit , and stop orders :

- A buy market order is an instruction to buy a futures contract at the best available ask price.

- A sell market order is an instruction to sell a futures contract at the best available bid price.

- A buy limit order allows you to specify a price or better at which to buy a futures contract. It’s typically placed at or below the current bid price.

- A sell limit order allows you to specify a price or better at which to sell a futures contract. It’s typically placed at or above the current ask price.

- A buy stop order is triggered when the market reaches a specific price. Once triggered, a buy stop order attempts to execute at the best available ask price up to a specific level called a protection point , which is set by the exchange. A buy stop order is placed at or above the current price. Otherwise, it will be rejected.

- A sell stop order is triggered when the market reaches a specific price. Once triggered, a sell stop order attempts to execute at the best available bid price up to a specific level called a protection point . A sell stop order is placed at or below the current price. Otherwise, it will be rejected.

Order types are crucial in futures trading because they help traders control the execution price, manage risk, and automate strategies, ensuring trades are executed under favorable conditions.

Each order type has a purpose and its own limitations. For example, stop-loss orders (buy stop or sell stop) are often used as risk management tools to protect positions. However, in fast-moving or volatile markets, stop orders may experience slippage, meaning the trade could be executed at a worse price than expected, potentially increasing losses. Additionally, in extremely illiquid conditions, a stop order may not be filled at all, leaving the position open and exposed to further adverse price movements.

## Examples

Let’s work through a couple of examples of long and short futures trading scenarios. Bear in mind, commissions and fees, margin requirements, and other contract specifications can change. These examples are hypothetical, do not represent actual conditions you will experience when trading on Robinhood, and are for illustrative purposes only.

### Example 1: Grace buys Micro Gold futures (/MGC)

Setup:

Grace is a gold trader with $10,000 in her futures account. She believes that if the Federal Reserve cuts interest rates, it will further weaken the U.S. dollar, driving a rally in gold futures. The current margin requirement for standard gold futures (/GC) is $11,550, which exceeds her available buying power. Instead, she opts to trade the Micro Gold contract (/MGC), which has a margin requirement of $1,155 per contract—1/10th the size of the /GC. While Grace could technically afford to trade up to 8 /MGC contracts, she has a personal rule to never allocate more than $5,000 of her buying power on a single position. As a result, she limits herself to a maximum of 4 /MGC contracts at one time.

Establishing the position:

Let’s imagine the current active contract is the December 2024 contract (/MGCZ24) and the current market is 2,649.9 bid x 2650.0 ask. Grace places a market order to buy 4 contracts and is filled immediately at 2,650. Upon entry, $4,620 in margin is deducted from her buying power (4 contracts x $1,155 margin requirement per contract), along with $5.08 in commissions and regulatory and exchange fees. Although Grace has only posted $4,620 in margin, her actual exposure is $106,000—$96,000 more than her account balance—giving her leverage of roughly 23:1.

Recap:

- Starting account balance: $10,000

- Entry price: 2,650

- Position: +4 /MGCZ24

- Margin requirement: ($4,620)

- Commissions and fees: ($5.08)

- Remaining buying power: $5,374.92

Managing risk:

Immediately after, Grace places a sell stop order at 2,600 to manage her risk, potentially limiting her losses to 50 points. Her potential loss on 4 contracts is $2,000, ignoring commissions and fees. To calculate this, start with the tick size (0.1) and multiply it by the contract multiplier of $10, resulting in a "tick value" of $1. For /MGC, there are 10 ticks per point, making the gain or loss $10 per point per contract. Since Grace holds 4 contracts, her gain or loss is $40 per point. With a 50-point move, her total potential loss is 50 points x $40 = $2,000, or 20% of her starting account value.

Grace hasn’t set a specific price target to take profits; instead, she plans to let the price of /MGC run if the trade becomes profitable, aiming for a substantial gain. While she hasn’t locked in a specific risk-to-reward ratio, Grace is aiming for a minimum profit of 100 points on this trade, which would give her a 1:2 risk-to-reward ratio.

Trade management:

Later in the week, the Federal Reserve announces a 50 basis point interest rate cut, exceeding expectations by 25 basis points. In response, the U.S. dollar weakens, and gold surges above 2,700, continuing to climb. Grace's position is now in the green. As the price reaches 2,750, her position is up by $4,000, and she decides to lock in some profits.

She sells half of her position with a market order, which gets filled at 2,750, locking in a 100-point profit on 2 contracts for a total gain of $2,000. Additionally, $2,310—half of her initial margin requirement—is returned to her buying power, minus commissions and fees of $2.54. After this trade, Grace remains long 2 contracts.

Closing her position:

Grace wants to keep profiting if /MGC continues to rise but now seeks to protect her unrealized gains. To manage this, she sets a sell stop order at 2,700, 50 points below the current price, risking $1,000 of her current $2,000 unrealized profit on the remaining contracts.

The following week, the U.S. Congress announces spending cuts. This strengthens the U.S. dollar and halts the rally in gold. The price of /MGCZ24 reverses and sells off to 2,725. Grace decides it’s time to fully exit the trade. She uses the "cancel and exit all" feature on her trading app, which simultaneously places a market order to sell her remaining 2 contracts and cancels her working sell stop order at 2,700.

As a result, Grace is now flat, with no open positions or working orders. She pays an additional $2.54 in commissions and fees for her closing trade and the remaining $2,310 of margin held is released and returned to her available buying power. In total, she made a profit of $3,500 on the trade—$2,000 from the first 2 contracts and $1,500 from the last 2.

To recap, opening the position:

- Starting account balance: $10,000

- Trade: Bought 4 at 2,650

- Position: +4 /MGCZ24

- Margin requirement: ($4,620)

- Commissions and fees: ($5.08)

- Remaining buying power: $5,374.92

Closing ½ of the position:

- Sold 2 /MGCZ24 @ 2,750

- Gain: $2,000

- Commissions and fees: ($2.54)

- Remaining position: +2 /MGCZ24

Closing the other ½ of the position:

- Sold 2 /MGCZ24 @ 2,725

- Gain: $1,500

- Commissions and fees: ($2.54)

- Ending account balance: $13,489.84

Summary:

Grace bought and sold 4 /MGCZ24 contracts. She entered the position using a market order and scaled out of her position, selling 2 contracts at a time at different prices. She entered the trade at 2,650, sold the first 2 contracts at 2,750, and exited her position by selling the remaining 2 contracts at 2,725. Overall, she made $3,500 going long /MGC, less commissions and fees. Before the trade, Grace’s account balance was $10,000. Afterward, it increased to $13,489.84, which reflects her original balance plus the $3,500 gain, minus the commissions and fees from her trades.

### Example 2: Shawn shorts the E-mini S&P 500 (/ES)

Setup:

Shawn is a S&P 500 Index trader with $40,000 in his futures account. He believes that Q3 earnings season will disappoint and drive stock prices lower. He decides to take a short position by selling the E-mini S&P 500 Index futures contract (/ES). Because the current margin requirement for /ES is $16,000, Shawn can trade 2 contracts at a time.

Establishing the position:

Let’s imagine the current active contract is the December 2024 contract (/ESZ24) and the current market is 5783.00 bid x 5783.25 ask. Shawn thinks the S&P 500 Index has a little more room to the upside before it sells off and places a limit order to sell 2 /ESZ24 at 5,800. In this instance, Shawn is willing to miss out on the trade in order to get filled at his price (5,800), or better. /ES rallies and Shawn’s limit order is filled a few days later at 5,800.

Upon entry, $32,000 in margin is deducted from his buying power (2 contracts x $16,000 margin requirement per contract), along with $4.30 in commissions, regulatory fees, and exchange fees. Although Shawn has only posted $32,000 in margin, his actual exposure is $580,000—$540,000 more than his account balance—giving him leverage of roughly 18:1.

Recap:

- Starting account balance: $40,000

- Trade: Sold 2 at 5,800

- Position: -2 /ESZ24

- Margin requirement: ($32,000)

- Commissions and fees: ($4.30)

- Remaining buying power: $7,995.70

Managing risk:

Immediately after, Shawn places a buy stop order above the market at 5,830 to manage his risk, potentially limiting his losses to 30 points. His potential loss on 2 contracts is $3,000, ignoring commissions and fees. To calculate this, start with the tick size (0.25) and multiply it by the contract multiplier of $50, resulting in a "tick value" of $12.50. For /ES, there are 4 ticks per point, making the gain or loss $50 per point per contract. Since Shawn is short 2 contracts, his gain or loss is $100 per point. With a 30-point move, his total potential loss is 30 points x $100 = $3,000, or 7.5% of his starting account value.

Shawn hasn’t set a specific price target to take profits; instead, he plans to let the price of /ES run if the trade becomes profitable, aiming for a substantial gain. While he hasn’t locked in a specific risk-to-reward ratio, he’s aiming for a minimum profit of 100 points on this trade, which would give him a 1:3 risk-to-reward ratio.

Getting stopped out:

The following week, the big tech stocks announce earnings and beat expectations. S&P 500 futures rally and Shawn is stopped out of his position. His stop price of 5,830 is triggered, but due to increased volatility, he experiences some slippage and the order fills at 5,830.50. As a result, Shawn is now flat, with no open positions or working orders. He pays an additional $4.30 in commissions and fees for the closing trade, and the margin requirement is released and returned to his available buying power. In total, he lost $3,050 on the trade.

To recap, opening the position:

- Starting account balance: $40,000

- Trade: Sold 2 /ESZ24 at 5,800

- Position: -2 /ESZ24

- Margin requirement: ($32,000)

- Commissions and fees: ($4.30)

- Remaining buying power: $7,995.70

Closing trade:

- Buy stop price: 5,830.00

- Bought 2 /ESZ24 at 5,830.50

- Loss: $3,050

- Commissions and fees: ($4.30)

- Ending account balance: $36,941.40

Summary:

Shawn shorted 2 /ESZ24 contracts. He entered the position using a sell limit order and was stopped out, buying back the 2 contracts at a higher price for a loss. He shorted the contracts at 5,800, and bought them back at 5,830.50 for a total loss of $3,050. Before the trade, Shawn’s account balance was $40,000. Afterward, it decreased to $36,941.40, which reflects his original balance minus the loss as well as the commissions and fees for his trades.

## Takeaway

Traders can profit from both rising and falling prices by going long or short futures contracts. Going long involves buying a contract to open a position, while going short means selling a contract to open a position. Long strategies are used by traders who are bullish, expecting prices to rise, while short strategies are favored by those who are bearish, anticipating price drops. In both cases, traders can manage their positions using various order types, such as stop, limit, and market orders. Additionally, flattening and scaling positions can be useful tactics for active traders.

Ready to start investing?

Sign up for Robinhood and get stock on us.

Sign up for Robinhood

Certain limitations apply

New customers need to sign up, get approved, and link their bank account. The cash value of the stock rewards may not be withdrawn for 30 days after the reward is claimed. Stock rewards not claimed within 60 days may expire. See full terms and conditions at rbnhd.co/freestock . Securities trading is offered through Robinhood Financial LLC. Futures trading offered through Robinhood Derivatives, LLC.

3925703