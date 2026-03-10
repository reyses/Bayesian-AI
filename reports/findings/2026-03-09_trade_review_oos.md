COMPREHENSIVE TRADE REVIEW — 1,719 trades
Source: oos_trade_log.csv
Generated: 2026-03-09 23:22:10

==========================================================================================
1. EXECUTIVE SUMMARY
==========================================================================================
  Trades: 1,719  |  Wins: 1,692  |  Losses: 27  |  Scratch ($0): 0  |  BE ($0.50): 1,317
  Win Rate: 98.4%  |  Win+BE Rate: 98.4%
  Total PnL: $19,120.50  |  Avg/trade: $11.12  |  Median: $0.50
  Profit Factor: 13.78

==========================================================================================
2. SCRATCH vs REAL TRADE DECOMPOSITION
==========================================================================================
  Scratch trades (|PnL| <= $0.50): 1,317 (76.6%)
  Real trades    (|PnL| >  $0.50): 402 (23.4%)

  REAL TRADE STATS:
    Win Rate: 93.3%  (375W / 27L)
    Total PnL: $18,462.00  |  Avg: $45.93  |  Median: $31.50
    Avg Winner: $53.22  |  Avg Loser: $-55.43  |  Payoff: 0.96x
  SCRATCH EXIT REASONS:
    stop_loss                  1317 (100.0%)

==========================================================================================
3. PnL DISTRIBUTION & PERCENTILES
==========================================================================================
  Bin                  Count     Pct       CumPnL  Bar
  ------------------ ------- ------- ------------  ------------------------------
  < -$100                  5    0.3% $    -740.50  #
  -$100..-$50              5    0.3% $    -466.50  #
  -$50..-$10               6    0.3% $    -241.00  #
  -$10..-$0               11    0.6% $     -48.50  #
  $0-$0.50              1317   76.6% $     658.50  ######################################
  $0.51-$10               31    1.8% $     179.50  #
  $10-$50                218   12.7% $   6,028.00  ######
  $50-$100                81    4.7% $   5,797.00  ##
  $100-$500               45    2.6% $   7,954.00  #
  > $500                   0    0.0% $       0.00  

  Percentiles:
    P1  : $     -7.55
    P5  : $      0.50
    P10 : $      0.50
    P25 : $      0.50
    P50 : $      0.50
    P75 : $      0.50
    P90 : $     37.00
    P95 : $     69.50
    P99 : $    160.89

  PROFIT CONCENTRATION:
    Top  5% (  85 trades): $ 11,333.00 = 59.3% of total
    Top 10% ( 171 trades): $ 15,712.50 = 82.2% of total
    Top 20% ( 343 trades): $ 19,769.00 = 103.4% of total

==========================================================================================
4. RISK METRICS
==========================================================================================
  Trading days: 49  |  Avg daily PnL: $390.21  |  Std: $302.15
  Sharpe (annualized):  20.50
  Sortino (annualized): 20.50
  Calmar ratio:         0.00
  Profit Factor:        13.78
  Payoff ratio:         0.22x (avg win / avg loss)
  Max cumul drawdown:   $0.00
  Winning days: 49/49 (100.0%)
  Losing days:  0/49 (0.0%)

==========================================================================================
5. DRAWDOWN ANALYSIS
==========================================================================================
  TOP 10 WORST INTRADAY DIPS:
    Date                Dip    Day PnL  Trades
    2026-01-23   $   -48.00 $   293.00      40
    2026-01-18   $   -32.00 $     1.00      11
    2026-02-23   $   -11.00 $   177.00      45
    2026-01-13   $    -5.00 $   138.00      52
    2026-02-27   $    -5.00 $   329.00      37
    2026-01-02   $     0.00 $   968.00      29
    2026-01-05   $     0.00 $   128.00      30
    2026-01-06   $     0.00 $    52.50      36
    2026-01-07   $     0.00 $   135.50      28
    2026-01-08   $     0.00 $   425.00      29

  CUMULATIVE EQUITY:
    Final: $19,120.50  |  Peak: $19,120.50  |  Max DD: $0.00

==========================================================================================
6. STREAK ANALYSIS
==========================================================================================
  Max consecutive wins:   292
  Max consecutive losses: 2
  Worst consecutive loss PnL: $-196.50

==========================================================================================
7. EXIT REASON CROSS-TAB
==========================================================================================
  Exit Reason                    N     WR%    Total PnL    Avg PnL % of PnL
  ------------------------- ------ ------- ------------ ---------- --------
  peak_giveback                252   94.8% $  10,333.50 $    41.01    54.0%
  envelope_decay               121  100.0% $   7,408.50 $    61.23    38.7%
  take_profit                    9  100.0% $   1,744.50 $   193.83     9.1%
  maintenance_flat               6  100.0% $     306.00 $    51.00     1.6%
  stop_loss                   1331   98.9% $    -672.00 $    -0.50    -3.5%

  EXIT × DIRECTION:
  Exit Reason                 LONG n   LONG avg   SHORT n  SHORT avg
  ------------------------- -------- ---------- --------- ----------
  stop_loss                      259 $     0.06      1072 $    -0.64
  peak_giveback                   32 $    34.58       220 $    41.94
  envelope_decay                  29 $    44.64        92 $    66.46
  take_profit                      1 $   121.50         8 $   202.88
  maintenance_flat                 0 $     0.00         6 $    51.00

  EXIT × ORACLE CLASS:
  Exit                    correct_dir   counter_tren   genuinely_wr          noise
  envelope_decay           82/$  5614     36/$  1557      0/$     0      3/$   238
  maintenance_flat          4/$   223      2/$    83      0/$     0      0/$     0
  peak_giveback           143/$  6325     98/$  3938      5/$  -136      6/$   206
  stop_loss               726/$    -2    574/$   287     10/$  -968     21/$    10
  take_profit               8/$  1631      1/$   114      0/$     0      0/$     0

==========================================================================================
8. TAIL RISK ANALYSIS
==========================================================================================
  WORST 15 TRADES:
           PnL    Dir                 Exit  Depth   Hold   TID       Oracle           Class
    $  -196.50  SHORT            stop_loss     12   354b    74    MEGA_LONG genuinely_wrong
    $  -153.50  SHORT            stop_loss      9    58b    64    MEGA_LONG genuinely_wrong
    $  -153.50  SHORT            stop_loss      8     1b    64    MEGA_LONG genuinely_wrong
    $  -124.50  SHORT            stop_loss      9   102b    63   MEGA_SHORT     correct_dir
    $  -112.50   LONG            stop_loss     11   222b    47   MEGA_SHORT genuinely_wrong
    $   -98.00  SHORT        peak_giveback      7    48b    64    MEGA_LONG genuinely_wrong
    $   -97.50  SHORT            stop_loss     11    88b    67   MEGA_SHORT     correct_dir
    $   -97.50  SHORT            stop_loss     10     5b    17   MEGA_SHORT     correct_dir
    $   -97.50  SHORT            stop_loss     10     7b    17    MEGA_LONG genuinely_wrong
    $   -76.00  SHORT            stop_loss      9     6b    18    MEGA_LONG genuinely_wrong
    $   -49.50  SHORT            stop_loss     11     1b    29    MEGA_LONG genuinely_wrong
    $   -47.00  SHORT            stop_loss      8     9b    54    MEGA_LONG genuinely_wrong
    $   -47.00  SHORT            stop_loss     10     4b    54    MEGA_LONG genuinely_wrong
    $   -43.50  SHORT            stop_loss     12     1b    39   MEGA_SHORT     correct_dir
    $   -34.50  SHORT            stop_loss     11     1b    79    MEGA_LONG genuinely_wrong

  VALUE AT RISK:
    VaR  1%: $     -8.00  (1 in 100 trades worse than this)
    VaR  5%: $      0.50  (1 in 20 trades worse than this)
    ES   1%: $    -85.68  (avg of worst 1%)
    ES   5%: $    -17.26  (avg of worst 5%)

==========================================================================================
9. TEMPLATE CONCENTRATION
==========================================================================================
  TOP 15 TEMPLATES BY PnL:
      TID  Trades     WR%    Total PnL       Avg  % of Total   Cumul%
       21     102  100.0% $   2,443.50 $   23.96       12.8%    12.8%
       54     105   98.1% $   1,337.50 $   12.74        7.0%    19.8%
       14      65  100.0% $   1,322.00 $   20.34        6.9%    26.7%
      115      46  100.0% $   1,249.50 $   27.16        6.5%    33.2%
        1      56   98.2% $     984.00 $   17.57        5.1%    38.4%
       53      44  100.0% $     945.00 $   21.48        4.9%    43.3%
       55      69  100.0% $     940.50 $   13.63        4.9%    48.2%
       93      28  100.0% $     752.50 $   26.88        3.9%    52.2%
        0     106   96.2% $     719.00 $    6.78        3.8%    55.9%
       18     118   98.3% $     619.50 $    5.25        3.2%    59.2%
       60     105   99.0% $     604.50 $    5.76        3.2%    62.3%
       39      54   96.3% $     587.50 $   10.88        3.1%    65.4%
       46      18  100.0% $     505.00 $   28.06        2.6%    68.0%
       41      51  100.0% $     476.50 $    9.34        2.5%    70.5%
      107      40   97.5% $     444.50 $   11.11        2.3%    72.9%

  BOTTOM 5 TEMPLATES (money losers):
    TID    79: 5 trades, $-32.50 total, $-6.50/trade
    TID    -1: 1 trades, $-4.00 total, $-4.00/trade

  TEMPLATE STATS: 76 unique templates, 74 profitable, 2 unprofitable, 0 breakeven

==========================================================================================
10. TIME ANALYSIS
==========================================================================================
  BY HOUR (ET):
     Hour      N     WR%    Avg PnL    Total PnL
        0     21  100.0% $     6.24 $     131.00
        1     31  100.0% $     8.35 $     259.00
        2     35  100.0% $     2.11 $      74.00
        3     62  100.0% $     9.02 $     559.00
        4     76   97.4% $     7.22 $     549.00
        5     39   97.4% $     6.90 $     269.00
        6     39   97.4% $     2.32 $      90.50
        7     60  100.0% $     7.14 $     428.50
        8     80   98.8% $     6.35 $     508.00
        9    197   99.0% $    17.46 $   3,440.00
       10    128  100.0% $    13.86 $   1,773.50
       11    119   99.2% $    10.95 $   1,303.50
       12    121   98.3% $    16.68 $   2,018.00
       13    111   96.4% $     4.68 $     519.00
       14     84   98.8% $     7.96 $     668.50
       15     93   98.9% $     8.81 $     819.50
       16     78   97.4% $    11.78 $     919.00
       18    144   98.6% $    20.56 $   2,961.00
       19     83   96.4% $     9.80 $     813.50
       20     32   96.9% $     5.70 $     182.50
       21     36  100.0% $     7.17 $     258.00
       22     19   89.5% $     9.68 $     184.00
       23     31   96.8% $    12.66 $     392.50

  BY SESSION:
    Overnight          149 trades   100.0% WR  $     6.87/trade  $   1,023.00 total
    Europe/PreMkt      294 trades    98.3% WR  $     6.28/trade  $   1,845.00 total
    US_Open            444 trades    99.3% WR  $    14.68/trade  $   6,517.00 total
    US_Mid             316 trades    97.8% WR  $    10.14/trade  $   3,205.50 total
    US_Close           516 trades    97.7% WR  $    12.66/trade  $   6,530.00 total

  BY DAY OF WEEK:
    Mon      293 trades    97.3% WR  $     8.89/trade  $   2,606.00 total
    Tue      325 trades    99.4% WR  $     7.73/trade  $   2,513.00 total
    Wed      356 trades    97.8% WR  $    12.79/trade  $   4,554.50 total
    Thu      327 trades    98.5% WR  $    14.49/trade  $   4,739.50 total
    Fri      392 trades    99.2% WR  $    11.73/trade  $   4,596.50 total
    Sun       26 trades    96.2% WR  $     4.27/trade  $     111.00 total

==========================================================================================
11. DIRECTION AUDIT
==========================================================================================
  LONG: 321 trades (18.7%)
    Headline WR: 99.7%  |  Real WR: 98.4%  |  Oracle correct: 57.3%
    Total: $2,539.00  |  Avg: $7.91  |  PF: 23.57
  SHORT: 1398 trades (81.3%)
    Headline WR: 98.1%  |  Real WR: 92.3%  |  Oracle correct: 55.7%
    Total: $16,581.50  |  Avg: $11.86  |  PF: 12.98

==========================================================================================
12. BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples)
==========================================================================================
  Win Rate (%)        median:    98.43  95% CI: [   97.79,    99.01]
  Avg PnL ($)         median:    11.10  95% CI: [    9.42,    12.90]
  Sharpe (ann)        median:    20.71  95% CI: [   17.44,    25.04]

==========================================================================================
13. ANCHOR EXIT EFFECTIVENESS
==========================================================================================
  Trades with anchor data: 1719
  Exited BEFORE expected MFE time: 1595 (92.8%)  avg $9.11
  Exited AFTER  expected MFE time: 124 (7.2%)  avg $37.03
  Reached expected MFE: 22/1719 (1.3%)

  ANCHOR × EXIT REASON:
    Reason                      N   Avg Anchor   Avg Held    Avg PnL
    stop_loss                1331        75.3b       6.0b $    -0.50
    peak_giveback             252        63.0b      16.5b $    41.01
    envelope_decay            121        62.3b      38.1b $    61.23
    take_profit                 9        86.6b      24.9b $   193.83
    maintenance_flat            6        90.4b       1.0b $    51.00
