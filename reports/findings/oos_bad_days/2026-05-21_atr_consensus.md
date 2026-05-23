# Cross-ATR consensus vs leg outcome — OFFLINE vs CAUSAL (MEASURE)

For each production ATR x4 leg, consensus = fraction of {x2,x3,x6,x8} zigzags agreeing with the x4 leg direction at the x4 entry bar. OFFLINE uses detect_swings (hindsight — inflated). CAUSAL uses a forward-pass streaming detector (live-available — the honest signal). If CAUSAL still discriminates, consensus is a trainable feature.

## IS — 17767 legs

### OFFLINE consensus (hindsight — inflated)
 consensus       n   mean $/leg                95% CI   trade WR    win%   mean amp
      0.00     608      -19.15  [   -24.9,   -13.3]      -0.53     23%       24.8
      0.25     135      -30.04  [   -34.8,   -24.8]      -0.90      7%       15.9
      0.50    3026      -37.19  [   -38.3,   -36.0]      -0.95      4%       18.0
      0.75    2223      -21.58  [   -23.0,   -20.1]      -0.87      5%       11.5
      1.00   11775      +31.08  [   +29.7,   +32.5]      +4.49     61%       23.4

  correlation +0.337   |   high(>=.75) $+22.7/leg (n=13998)   low(<=.5) $-34.0/leg (n=3769)    spread $+56.7/leg

### CAUSAL consensus (streaming — the honest, live signal)
 consensus       n   mean $/leg                95% CI   trade WR    win%   mean amp
      0.00    7077      +22.74  [   +20.8,   +24.7]      +1.81     52%       24.2
      0.25    1829      +16.32  [   +13.1,   +19.8]      +1.23     47%       21.5
      0.50    5842       +1.21  [    -0.6,    +3.0]      +0.06     35%       18.8
      0.75    1626       -2.09  [    -4.7,    +0.8]      -0.11     34%       17.3
      1.00    1393       -3.35  [    -6.3,    -0.2]      -0.17     32%       17.3

  correlation -0.141   |   high(>=.75) $-2.7/leg (n=3019)   low(<=.5) $+13.4/leg (n=14748)    spread $-16.1/leg

## OOS — 2936 legs

### OFFLINE consensus (hindsight — inflated)
 consensus       n   mean $/leg                95% CI   trade WR    win%   mean amp
      0.00     126      -43.89  [   -53.2,   -33.5]      -0.84     16%       28.1
      0.25      23      -51.15  [   -58.9,   -43.3]      -1.00      0%       22.6
      0.50     514      -47.93  [   -50.9,   -44.9]      -0.95      4%       23.8
      0.75     400      -27.97  [   -31.5,   -24.1]      -0.87      6%       15.7
      1.00    1873      +35.07  [   +31.4,   +39.1]      +3.61     62%       28.1

  correlation +0.402   |   high(>=.75) $+24.0/leg (n=2273)   low(<=.5) $-47.3/leg (n=663)    spread $+71.2/leg

### CAUSAL consensus (streaming — the honest, live signal)
 consensus       n   mean $/leg                95% CI   trade WR    win%   mean amp
      0.00     920      +22.48  [   +16.5,   +28.9]      +1.34     52%       28.2
      0.25     288      +20.22  [   +11.0,   +29.8]      +1.12     50%       28.3
      0.50    1119       -1.51  [    -5.8,    +2.8]      -0.06     35%       24.3
      0.75     316       -3.80  [   -10.7,    +3.5]      -0.15     34%       22.4
      1.00     293       -1.56  [    -9.0,    +6.5]      -0.06     35%       23.6

  correlation -0.128   |   high(>=.75) $-2.7/leg (n=609)   low(<=.5) $+10.7/leg (n=2327)    spread $-13.4/leg
