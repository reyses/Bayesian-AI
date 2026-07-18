# leg_clock — is a directional leg's length memoryless or clocked?

**Origin (2026-07-07, Moises)**: reframing of "does slope persist." Wrong
question was bar-to-bar. Right question: segment sessions into legs (macro ebbs
and flows), learn the leg-length distribution in 2024, predict 2025 legs
out-of-sample. If elapsed time predicts remaining time (mean-residual-life
falls with elapsed) the leg is CLOCKED and tradeable; if flat, it's MEMORYLESS.

## Run
```
.venv_wsl/bin/python research/leg_clock/tools/leg_length_clock.py --thr 20
```
Legs = fixed-reversal zigzag on 1m closes (thr in ticks). Train=2024, test=2025,
true out-of-sample. Reports duration + extent distributions (mode/median/tail
separately — the tail is the big drive that pays), and the mean-residual-life
curve train-vs-test (does the clock transfer).

## Read
`reports/leg_clock_thr<N>.txt`. MRL falling + train≈test → clocked & transfers.
