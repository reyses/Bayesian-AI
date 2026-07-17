# weekly results from my fully automated NinjaTrader algorithm trading NQ futures

**Source URL:** https://www.reddit.com/r/FuturesTrading/comments/1jbgf9z/weekly_results_from_my_fully_automated/
**Site:** reddit.com
**Fetch date:** 2026-07-17 (UTC)
**Word count:** 646
**Status:** OK
**pass:** 2
**Note:** Live www.reddit.com blocked this sandbox IP with 'You've been blocked by network security' (seen on BOTH HTTP 200 and 403 responses; WebFetch also hard-refuses the reddit.com domain). Rescued via Wayback Machine snapshot 2025-08-04 (http://web.archive.org/web/20250804170556/https://www.reddit.com/r/FuturesTrading/comments/1jbgf9z/). Includes full captured comment tree (not just depth-0) since total volume is small and the highest-value exit-mechanics tip (Bollinger-Keltner squeeze for chop detection) is at depth 2.

---

## Post title

weekly results from my fully automated NinjaTrader algorithm trading NQ futures

## Post body

Been working on converting my manual NQ trading strategy into a fully automated bot with NinjaTrader since October. Over that period it’s shown signs of potential but struggled at other times. I’ve continued to tweak it to where I am relatively comfortable running it live.
Here are the results from this week.
It runs on the 1 min data series updating every tick, so I’ve only managed to get my hands on historical data from the last year ish to test. In that (small) sample it has done well. I plan to continue to run it in my live personal account and provide updates on my progress. I hope it continues to work- but it’s been a fun and rewarding side project for me

## Comment tree (12 of 13 total comments captured in this snapshot; author handles anonymized to u/redacted)

- **u/redacted** (depth 0): Good for you. Been at NQ bot on NinjaTrader for a year+ now. Many variations. Nothing seems to be consistent over different market regimes
  - **u/redacted** (depth 1): For sure.
One of the most frustrating things for me was trying to figure out what days were going to be choppy. If I could do that I could avoid a lot of my losses. But I still haven’t been able to predict it before it happens without missing out on trades that would have won. At this point I’m accepting that as part of the built in losses of the strategy.
Are you still going at it? I hope we figure it out someday man
    - **u/redacted** (depth 2): Oh yeah, I still run these bots everyday. There’s a couple ways to determine chop. Tried the Bollinger-Keltner combo? Also called the squeeze
      - **u/redacted** (depth 3): Big fan of the squeeze for swing trading. I used to have it in my strategy to check for chop but at least for my entries it was always squeezing too late and would squeeze for too long (aka lagging indicator) maybe I need to play with the setting some more.
    - **u/redacted** (depth 2): I'm an active trader.  I'd suggest charting the $VIX(tradestation symbol).  Choppy days in my opinion are when your indicators are not in agreement.  MA's, MACD, Bollinger Bands etc  whatever you use.  When I stay on the same side of the VIX I do well..when I go agaist it  ie up when VIX says down I suffer.  My 2 cents.
  - **u/redacted** (depth 1): Comment deleted by user
    - **u/redacted** (depth 2): Tell me more?
      - **u/redacted** (depth 3): Comment deleted by user
        - **u/redacted** (depth 4): I dont dis-believe that you'd make 15-20K daily. I know of a trader who averages that.But, I dont think it is as simple as you say. You probably have 1000s of hours of chart time through which you've developed awareness/intuition that cant be converted into rules for an algorithm.
Maybe it is simple, but not easy.
Can you tell me more, and keep telling me more?
Yes, I'm at a stage where I realize the lost efforts when jumping from strategy to strategy. Which is why I've been sticking to NQ bot development for 1+ year (and trading for 3+ years).
- **u/redacted** (depth 0): I’m running or trying to run around 12 bots that each correspond to a specific market condition.
Goal is to take a discretionary view of the market on larger time frames and deploy the bot that has best probability within its best performing market parameters.
  - **u/redacted** (depth 1): This feels like the way to do it. I'm still in the early stages of figuring out a reliable strategy for each condition. Would love to know more about yours
- **u/redacted** (depth 0): OP: how is your bot doing up to these days?