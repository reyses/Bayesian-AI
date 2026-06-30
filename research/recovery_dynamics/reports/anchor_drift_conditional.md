# Does price return to the anchor WHILE TRENDING? (every-bar, conditioned on trailing drift)
628147 anchors, trailing drift over 30 min. If return rate stays high in strong drift,
the fixed every-bar anchor already captures the within-trend zigzag.

drift quintile  | drift(pt/min) | return rate | median period (returned)
--------------------------------------------------------------------------
calmest 20%     | 0.000-0.142 |     93.7%  | 5 min
2nd             | 0.142-0.308 |     93.6%  | 5 min
middle          | 0.308-0.567 |     93.3%  | 5 min
4th             | 0.567-1.075 |     92.4%  | 5 min
MOST-TREND 20%  | 1.075-48.033 |     91.1%  | 5 min

overall return rate: 92.8%  (no-return = abandoned levels = trend footprint)

## Read
If the most-trending quintile still returns most of the time, Moises is right: while trending,
price keeps coming back to recent anchors (the zigzag), because we anchor every bar. The
no-return share rising only modestly with drift = the trend abandons only its trailing levels,
not the oscillation. The fixed every-bar measurement already contains the within-trend oscillation.
