# Orange-line EDA (7.5m cubic, 1s) — 10 days (2024_03_01..2024_03_14)

swings (slope zero-cross to zero-cross): n=3776
  amplitude pts: median 3.31, p75 8.45, p90 16.70
  duration min : median 3.1, p75 4.6
BIG swings (amp>=p75, n=944): median amp 14.41 pts, dur 3.8 min

CAUSAL early-exit test — curvature flip leads the slope flip (the turn):
  swings with a curvature flip inside: 2676/3776
  LEAD time (curv-flip -> slope-flip): median 20s, p25 10s, p75 37s
  -> curvature warns ~20s before the orange peak, causally (no hindsight).

Read: exit on curvature-flip (leading) vs slope-flip (lagging) is the lever to test next.