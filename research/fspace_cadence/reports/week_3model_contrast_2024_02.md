# Week flip-timing contrast — survival@45s, REAL vs FOURIER, 3 models x 5 days (2024_02)


## B2T (tiled)
  per-day real/four(gap)@45: 20:0.31/0.04(+0.27)  21:0.32/0.04(+0.28)  22:0.34/0.01(+0.33)  23:0.34/0.13(+0.21)  26:0.36/0.17(+0.19)
  mean gap +0.256  95% day-block CI [+0.215,+0.296]  -> SIG (excl 0)
  reproduces (all days gap>0)? YES

## B2C (continuous)
  per-day real/four(gap)@45: 20:0.58/0.52(+0.07)  21:0.60/0.51(+0.09)  22:0.54/0.49(+0.05)  23:0.55/0.58(-0.03)  26:0.56/0.61(-0.04)
  mean gap +0.027  95% day-block CI [-0.018,+0.073]  -> ns (incl 0)
  reproduces (all days gap>0)? NO

## RunC (bar-close)
  per-day real/four(gap)@45: 20:0.20/0.01(+0.19)  21:0.20/0.00(+0.20)  22:0.25/0.00(+0.25)  23:0.26/0.09(+0.18)  26:0.26/0.16(+0.10)
  mean gap +0.182  95% day-block CI [+0.138,+0.222]  -> SIG (excl 0)
  reproduces (all days gap>0)? YES
