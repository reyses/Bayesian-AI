# The kicker — the second half of the oscillation (adverse-first trades) (2024+2025)
n adverse-first trades = 21383

- came back to zero (oscillators): 19509/21383 = 91%; never came back (adverse runaway/DEATH): 1874/21383 = 9%

Of the ones that came back to zero, what the SECOND leg did:
- **KICKER** (favorable swing >= 5.0pt then returns): 47%
- **JACKPOT** (favorable swing that RAN away, never returned): 5%
- STALL (no real second leg): 49%

- second-leg favorable reach A2: median 18pt ($35), mean 48pt, 90th 115pt
- first adverse leg A1: median 15pt | A2/A1 median ratio **1.13** (1.0 = symmetric oscillation)
```
KICKER amplitude (favorable 2nd-leg reach A2) (mode 5-10pt, median 17pt, n=10001)
  0-5  pt| 0
  5-10 pt|######################################## 2914
 10-15 pt|##################### 1546
 15-20 pt|############# 911
 20-25 pt|######### 680
 25-30 pt|######## 568
 30-35 pt|##### 388
 35-40 pt|#### 307
 40-45 pt|### 234
 45-50 pt|### 204
 50-55 pt|### 198
 55-60 pt|## 159
 60-65 pt|## 130
 65-70 pt|######################## 1762
```

## Read
If KICKER+JACKPOT dominate and A2/A1 ~ 1, the oscillation is symmetric: crawling back to zero is
the MIDDLE of the cycle and the favorable half follows -> cutting at breakeven discards the paying
half; holding (or flipping at the trough) captures it. The DEATH2 fraction is the adverse runaway
that ruins this -> so the hold-for-the-kicker edge REQUIRES the oscillator-vs-runaway read.
