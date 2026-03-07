# System Status Report
> Generated on: 2026-03-07 19:41:50 UTC

## Run Logs
No run logs found in `run_logs/`.

## Forward Pass Results

### Results from `reports/is_report.txt`
```
================================================================================
FORWARD PASS COMPLETE  (run: 2026-03-07 03:56:28)
  Period: DATA\ATLAS\15s\2025_01.parquet to 2025_12 (12 files)
Total Trades: 6933
Win Rate: 77.9%
Total PnL: $86289.50

── WORKER STATES LOADED ──
  1h=469/file(12/12 files ok)  30m=958/file(12/12 files ok)  15m=1937/file(12/12 files ok)  5m=5850/file(12/12 files ok)  3m=9753/file(12/12 files ok)
  1m=29175/file(12/12 files ok)  30s=58039/file(12/12 files ok)  15s=114139/file(12/12 files ok)  5s=307307/file(12/12 files ok)  1s=1985894/file(12/12 files ok)
================================================================================

================================================================================
ORACLE PROFIT ATTRIBUTION
================================================================================

  PER-DEPTH PnL BREAKDOWN (-> depth_weights.json for next run):
    Depth     Trades    Win%    Total PnL  Avg/trade
    -----     ------    ----    ---------  ---------
    depth 3       385     72% $  4,278.00 $    11.11
    depth 4       337     77% $  4,979.00 $    14.77
    depth 5       640     84% $ 12,352.50 $    19.30
    depth 6       617     83% $ 16,235.00 $    26.31
    depth 7       624     84% $ 12,492.50 $    20.02
    depth 8       721     84% $ 11,434.50 $    15.86
    depth 9       938     79% $ 10,192.00 $    10.87
    depth 10    1,002     77% $  9,454.50 $     9.44
    depth 11    1,170     72% $  3,976.00 $     3.40
    depth 12      499     67% $    895.50 $     1.79

  DYNAMIC EXIT QUALITY:
    Belief-flip exits:  0 trades
    Physics-decay exits:0 trades
    Trail-tightened:    0 trades
    Trail-widened:      0 trades
    Loss watchdog:      0 trades
    Standard trail:     0 trades
    Decay score at exit:  WIN avg=0.000  LOSS avg=0.000
    Envelope halflife:   60.0 bars (self-tuned from 20)
    Giveback threshold:  55% (self-tuned from 70%)
    Peak giveback exits:  4474 trades  ->  avg PnL $   5.28

  EXIT QUALITY (correct-direction trades, worst → best):
    Bucket                                   n    Total PnL   Avg PnL       Hold     Cap%
    ──────────────────────────────────── ─────  ───────────  ────────  ─────────  ───────
    Reversed (mkt flipped after entry)     742  $    -7,982  avg$    -11     10bars  cap   -6%  <- leakage
    Too late  (reached peak, gave back)  1,868  $    16,284  avg$      9      8bars  cap   +4%  <- giveback
      Gave back (50-60%)                   468  $     6,587  avg$     14      9bars  cap   +7%
      Gave back (60-70%)                   586  $     5,976  avg$     10      8bars  cap   +5%
      Gave back (70-80%)                   381  $     2,530  avg$      7      7bars  cap   +4%
      Gave back (80-90%)                   265  $       974  avg$      4     10bars  cap   +2%
      Gave back (90-100%)                  168  $       218  avg$      1      7bars  cap   +1%
    Too early (<20%, never reached)        431  $    12,962  avg$     30     25bars  cap   +9%
      Partial  (20-30% captured)           189  $     9,086  avg$     48     19bars  cap  +25%
      Partial  (30-40% captured)           109  $     7,161  avg$     66     21bars  cap  +35%
      Partial  (40-50% captured)            54  $     3,708  avg$     69     29bars  cap  +44%
      Partial  (50-60% captured)            55  $     4,032  avg$     73     29bars  cap  +54%
      Partial  (60-70% captured)            40  $     3,661  avg$     92     25bars  cap  +65%
      Partial  (70-80% captured)            44  $     4,370  avg$     99     34bars  cap  +75%
    Optimal  (>=80% captured)               63  $     8,719  avg$    138     32bars  cap +111%
    Left on table (non-reversed gap):                        $   748,381

  EXIT REASON → QUALITY CROSS-BREAKDOWN (correct-direction trades):
    Exit reason           Optimal    Partial  Too early   Too late   Reversed   Total   Avg PnL
    ──────────────────  ─────────  ─────────  ─────────  ─────────  ─────────  ──────  ────────
    envelope_decay             51        371        406        267         12   1,107  $     42
    peak_giveback               2         98          0      1,596        581   2,277  $      6
    stop_loss                   0          0          0          0         35      35  $    -84
    take_profit                10         22         16          5          2      55  $    114
    trail_stop                  0          0          9          0          3      12  $      1
    watchdog                    0          0          0          0        109     109  $    -16

  CAPTURE DETAIL (correct-direction trades):
    Bucket                  Avg oracle MFE  Avg trade MFE  Avg actual PnL  Avg hold bars
    Optimal                 $          128       336.9tks  $          138           31.6
    Too early               $          153        89.0tks  $           30           24.6
    Too late                $          140        55.2tks  $            9            8.4
    Reversed                $          208        30.6tks  $          -11            9.9
    Partial 20-30%          $          106       158.8tks  $           48           19.2
    Partial 30-40%          $          105       188.4tks  $           66           20.7
    Partial 40-50%          $           96       193.3tks  $           69           28.8
    Partial 50-60%          $           77       215.5tks  $           73           29.3
    Partial 60-70%          $           96       245.2tks  $           92           25.2
    Partial 70-80%          $           70       264.8tks  $           99           34.1

  EXIT QUALITY BY DEPTH  (hold = real time from 15s bars × 15):
    Depth            n  Optimal%  Reversed%   Avg PnL    Avg Hold       Left$
    ───────────── ────  ────────  ─────────  ────────  ──────────  ──────────
    depth  3 (15m)   169       0%       31%  $     10       0h01m  $  118,199
    depth  4 (5m )   145       1%       24%  $     12       0h02m  $   40,757
    depth  5 (1m )   363       3%       18%  $     22       0h04m  $   72,122
    depth  6 (30s)   364       1%       18%  $     31       0h03m  $   99,268
    depth  7 (15s)   306       2%       14%  $     28       0h03m  $   71,706
    depth  8 (15s)   406       1%       17%  $     21       0h03m  $   87,867
    depth  9 (5s )   483       3%       18%  $     17       0h03m  $   77,204
    depth 10 (5s )   524       3%       20%  $     15       0h03m  $   58,982
    depth 11 (1s )   574       1%       25%  $      8       0h03m  $   72,122
    depth 12 (1s )   261       0%       29%  $      7       0h03m  $   50,154

  WORKER AGREEMENT AT ENTRY  (agree = worker dir matches trade direction)
    TF      WIN agree  LOSS agree    Edge  <-- positive = worker is predictive
    1h           0.69        0.71   -0.03
    30m          0.65        0.64   +0.01
    15m          0.63        0.64   -0.01
    5m           0.62        0.61   +0.01
    3m           0.61        0.62   -0.01
    1m           0.59        0.61   -0.02
    30s          0.56        0.57   -0.01
    15s          0.59        0.54   +0.05
    5s           0.51        0.53   -0.01
    1s           0.51        0.50   +0.01

  DIRECTION FLIPS BETWEEN ENTRY AND EXIT:
    (worker flipped = changed LONG/SHORT conviction side during the trade)
    1h     WIN flip=0.03  LOSS flip=0.02  diff=-0.01
    30m    WIN flip=0.06  LOSS flip=0.03  diff=-0.02
    15m    WIN flip=0.09  LOSS flip=0.06  diff=-0.04
    5m     WIN flip=0.24  LOSS flip=0.11  diff=-0.13
    3m     WIN flip=0.32  LOSS flip=0.18  diff=-0.13
    1m     WIN flip=0.43  LOSS flip=0.34  diff=-0.09
    30s    WIN flip=0.47  LOSS flip=0.42  diff=-0.06
    15s    WIN flip=0.48  LOSS flip=0.48  diff=+0.00
    5s     WIN flip=0.43  LOSS flip=0.42  diff=-0.01
    1s     WIN flip=0.50  LOSS flip=0.51  diff=+0.00

  DECISION-TF WAVE MATURITY AT ENTRY  (0=fresh wave, 1=exhausted)
    All trades:  avg=0.381
    Winners:     avg=0.381
    Losers:      avg=0.380
    Insight: maturity gap=-0.001 (positive = entering losers at wave exhaustion)

  OF 6,933 TRADES TAKEN:
    Correct direction:   3,595  (51.9%)  ->  actual: $ 62,002.00
    Wrong direction:     3,206  (46.2%)  ->  losses: $ 22,675.00
    Traded noise:          132  (1.9%)  ->  losses: $  1,612.50

  TOTAL SIGNALS SEEN BY ORACLE: 455,809
    Real moves (MEGA/SCALP):  412,441   -- worth $105,500,490.50 if perfectly traded
    Noise (no real move):     43,368

  WHAT WE DID:
    Traded:   6,933  (1.5% of all signals)
    Skipped: 448,876  (98.5% of all signals)

  WHY SIGNALS WERE SKIPPED  (total candidates evaluated: 479,244)
    Gate 0 (headroom/pattern rule): 276,904  (57.8%)
    Gate 1 (dist > 3.0, no match):   3,112  (0.6%)
    Gate 2 (brain rejected):             0  (0.0%)
    Gate 3 (conviction < thresh):   20,785  (4.3%)
    Physics QG (depth>3 or z>=0):     135  (0.0%)
    Passed all gates -> traded:      6,933  (1.4%)

  TRADED SIGNAL DEPTH (which TF level triggered the trade):
    depth 3=15m  :   385 trades  ##
    depth 4=5m   :   337 trades  #
    depth 5=1m   :   640 trades  ###
    depth 6=15s  (leaf):   617 trades  ###
    depth 7:   624 trades  ###
    depth 8:   721 trades  ####
    depth 9:   938 trades  #####
    depth 10: 1,002 trades  #####
    depth 11: 1,170 trades  ######
    depth 12:   499 trades  ##

  PROFIT GAP ANALYSIS:
    Ideal (golden-path: gate-blocked + traded, perfect exits):  $105,500,490.50
    -----------------------------------------------------
    Lost -- missed opportunities (gate-blocked):  $104,197,948.00  (98.8% of ideal)
    Lost -- wrong direction at entry:             $   22,675.00  (0.0% of ideal)
    Lost -- noise trades:                         $    1,612.50  (0.0% of ideal)
    Lost -- reversed after correct entry:         $    7,981.50  (0.0% of ideal)
    Lost -- TP underperform (non-reversed):       $  748,381.00  (0.7% of ideal)
    -----------------------------------------------------
    Actual profit:                               $   86,289.50  (0.1% of ideal)
    [info] Score-competition pool (took better same bar): $4,601,598.50  (not missed -- golden path chose better candidate)

================================================================================
DIRECTION LEARNING (oracle corrections absorbed)
================================================================================
  Templates with direction corrections: 75
  Templates with signed MFE regression: 59

  TOP 15 DIRECTION CORRECTIONS (biggest bias shift):
       TID   Orig    New  Shift  L_ok  L_bad  S_ok  S_bad      L_PnL      S_PnL
         0   0.31   0.31 +0.00    88     64   205    209 $    2,208 $    6,594
        21   0.03   0.03 +0.00     6      9   248    150 $      538 $    7,508
        20   0.51   0.51 +0.00    17     10    16     16 $      230 $      165
        14   0.61   0.61 +0.00    92     46    60     67 $    5,896 $    2,984
        61   0.83   0.83 +0.00    75     69    14     21 $    1,962 $      360
        18   0.09   0.09 +0.00    13     19   149    138 $       21 $      718
        23   0.05   0.05 +0.00     3      2    65     37 $      123 $    3,088
        22   0.36   0.36 +0.00    35     50    64     30 $    1,877 $    1,926
        41   0.54   0.54 +0.00    39     30    33     41 $      397 $      349
        55   0.05   0.05 +0.00     6      6   160    109 $       84 $    3,838
        53   0.79   0.79 +0.00    78     52    19     14 $      487 $      624
        60   0.23   0.23 +0.00    63     35   214    183 $      764 $    5,240
        29   0.54   0.54 +0.00    32     26    28     27 $      610 $      -39
        67   0.77   0.77 +0.00    45     73    13     15 $      706 $      152
        59   0.51   0.51 +0.00    46     40    43     87 $      546 $    1,003

  DIRECTION ACCURACY (this run):
    Correct: 3593/6795 (52.9%)
    LONG  correct: 1063  wrong: 920
    SHORT correct: 2530  wrong: 2282
    NOTE: Next run will use these corrected biases as starting point

  DETECTION FUNNEL (bar-level)
    Total 15s bars processed:  1,369,925  (100%)
    Bars with detection:         102,492  (7.5%)
    Bars with NO detection:    1,267,433  (92.5%)  <- model blind
    Bars slot-blocked:            35,419  (2.6%)  <- position open, can't trade
    Bars evaluated (free slot):   67,073  (4.9%)
    Candidates on those bars:    479,244  (avg 7.1/bar)

  Per-trade oracle log saved: checkpoints\oracle_trade_log.csv
  PID oracle log saved: checkpoints\pid_oracle_log.csv (100599 signals)
  Signal log saved: checkpoints\signal_log.csv  (6933 traded  +  0 skipped)

  DECISION MATRIX SUMMARY  (skipped candidates -- oracle $ = profit left on table)
    Gate                    Count  RealMove%  Avg Oracle$  Total Missed$
    traded                   6933      98.1%         176       1200097
  FN oracle log saved: checkpoints\fn_oracle_log_0000_Q1.csv  (406,229 missed real moves)

  FN WORKER AGREEMENT (did workers agree with oracle on missed moves?)
    High agree% = workers called it right but a gate still blocked the trade
    FN total=406,229  competed=28,361  no_match=377,868
    TF       All FN  Competed  No-match  <- agree% with oracle dir
    1h         0.50      0.53      0.50
    30m        0.49      0.51      0.49
    15m        0.51      0.52      0.51
    5m         0.52      0.52      0.51
    3m         0.51      0.52      0.50
    1m         0.50      0.52      0.50
    30s        0.50      0.51      0.50
    15s        0.53      0.52      0.53
    5s         0.50      0.52      0.50
    1s         0.50      0.49      0.50

  FN GATE BREAKDOWN (which gate blocked profitable signals):
    Gate 0  no pattern (Rule 1)                     0  (  0.0%)
    Gate 0  noise zone <0.5sigma (Rule 2)           0  (  0.0%)
    Gate 0  approach zone BAND_REVERSAL (Rule 3) -- no qualified tmpl      0  (  0.0%)
    Gate 0  approach zone MOMENTUM_BREAK weak trend (Rule 3)      0  (  0.0%)
    Gate 0  extreme zone nightmare field (Rule 4)  4,997  (  1.2%)
    Gate 0  extreme zone MOMENTUM_BREAK no headroom (Rule 4) 17,419  (  4.3%)
    Gate 0  Hurst < 0.5 choppy/anti-persistent (Rule 5a) 121,351  ( 29.9%)
    Gate 0  momentum override breakout likely (Rule 5b) 19,390  (  4.8%)
    Gate 0  tunnel probability < 40% (Rule 5c) 85,977  ( 21.2%)
    Gate 0.5 depth filter                       2,099  (  0.5%)
    Gate 1  no cluster match (dist>4.5)         2,797  (  0.7%)
    Gate 2  brain rejected                          0  (  0.0%)
    Gate 3  conviction below threshold         18,997  (  4.7%)
    Passed gates, lost to better score              0  (  0.0%)
  Depth weights saved: checkpoints\depth_weights.json

```
