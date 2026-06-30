# Opportunity cost of holding a WRONG trade to breakeven | 2024_02_20
thresholds: underwater>=5.0pt, swing>=8.0pt, entry every 3 bars

wrong trades that DID return to zero: 418 | never recovered (held to EOD): 41

MODE dead-hold time: ~6 min | MODE trades foregone: ~2
median dead-hold 21 min | median foregone 2 | median depth 12.5pt ($25)

```
TRADES FOREGONE while waiting to break even (mode bin = 2-4 , n=418)
   0-2    |###################### 105
   2-4    |######################################## 191
   4-6    |######## 37
   6-8    |### 14
   8-10   |###### 27
  10-12   |## 9
  12-14   |## 10
  14-16   | 1
  16-18   |# 3
  18-20   | 1
  20-22   |# 6
  22-24   |# 4
  24-26   | 1
  26-28   | 0
  28-30   |# 3
  30-32   | 0
  32-34   | 1
  34-36   | 0
  36-38   | 2
  38-40   | 0
  40-42   | 0
  42-44   | 1
  44-46   | 0
  46-48   | 1
  48-50   | 1

DEAD-HOLD TIME (min underwater -> back to zero) (mode bin = 0-15 m, n=418)
   0-15  m |######################################## 165
  15-30  m |###################### 90
  30-45  m |############ 51
  45-60  m |##### 22
  60-75  m |###### 23
  75-90  m |## 10
  90-105 m |## 7
 105-120 m |# 6
 120-135 m |# 3
 135-150 m |## 9
 150-165 m |# 6
 165-180 m |## 7
 180-195 m | 2
 195-210 m |# 3
 210-225 m | 2
 225-240 m | 0
 240-255 m | 1
 255-270 m |# 4
 270-285 m | 2
 285-300 m | 1
 300-315 m | 1
 315-330 m | 1
 330-345 m | 0
 345-360 m | 0
 360-375 m | 0
 375-390 m | 1
 390-405 m | 1

DRAWDOWN DEPTH (max underwater) (mode bin = 5-10 pt, n=418)
   0-5   pt | 0
   5-10  pt |######################################## 151
  10-15  pt |######################### 94
  15-20  pt |########### 43
  20-25  pt |######## 30
  25-30  pt |### 13
  30-35  pt |### 10
  35-40  pt |### 13
  40-45  pt |### 11
  45-50  pt |# 5
  50-55  pt |# 5
  55-60  pt |## 8
  60-65  pt |## 9
  65-70  pt |# 3
  70-75  pt | 1
  75-80  pt | 1
  80-85  pt |# 5
  85-90  pt |# 4
  90-95  pt |# 3
  95-100 pt |# 4
 100-105 pt | 0
 105-110 pt | 0
 110-115 pt |# 2
 115-120 pt | 0
 120-125 pt | 0
 125-130 pt | 1
 130-135 pt | 0
 135-140 pt |# 2
```

## Worked example (worst opportunity cost)
- LONG entry at bar 1020 (price 17593.25)
- went 139.2pt ($278) underwater, took 298 min to crawl back to $0
- in that window: 48 tradeable swings (>= 8.0pt) went by — uncaptured

## Read
Even the trades that 'came back' cost real money: the dead-hold window is time + foregone
trades, not a free round-trip. The never-recovered bucket is the pure loss on top.
