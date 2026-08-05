# CHART STANDARD — dojo renders
_Owner instruction, 2026-08-05: "Why can't you use the same format? Use dashed
for major and minor levels, add the reference lines. Write it down as an
instruction somewhere."_

**Every panel that draws price obeys this. No exceptions, no per-function
dialects.** Two panels using different vocabulary for the same idea is what
made "pdH" mean two prices at once.

## 1. Reference levels — always DASHED

| class | what | style |
|---|---|---|
| **MAJOR** | session anchors: OPEN, VWAP, PD HIGH / LOW / SETTLE, NOW | `ls='--'`, lw 1.4, alpha 0.85, **bold** label, fontsize 8 |
| **MINOR** | bounds and psychology: opening range, overnight H/L, 5d H/L, round numbers | `ls='--'`, lw 0.9, alpha 0.6, normal label, fontsize 6.8 |

Never solid for a reference line. Solid is reserved for owner-drawn lines
(`line <price>`) and density-telescope levels, so hand-called levels stay
visually distinct from computed ones.

## 2. Price grid — major + minor ticks, dashed

`_price_grid(ax)` on every price panel: major locator auto-picked so <= 12
lines are visible at the current zoom, minor at major/5. Major `'--'` 0.6
alpha 0.45; minor `':'` 0.4 alpha 0.28.

## 3. The session geometry set (`_session_geometry`)

Drawn on the main panel, computed causally from bars <= cur:
OPEN, VWAP (major) · opening-range H/L, overnight LOW, nearest 50/100 rounds
(minor).

## 4. Time filters — NUMERIC, both ends bounded

```python
mod = et.dt.hour * 60 + et.dt.minute      # minutes of day
rth = (mod >= 570) & (mod < 930)          # 09:30 <= t < 15:30
```
**Never** `strftime('%H:%M') >= '09:30'`. String comparison passes "18:00"
and silently swallows the prior evening session — it produced a false
"today's open" (the previous 20:00 open), a false VWAP, and two wrong
touch-point tables inside one hour on 2026-08-05.

## 5. Label the definition

Any high/low/settle must state whether it is RTH (09:30-16:00) or full
Globex. On 2025_08_06 the two differ by 35pt on the high and 100pt on the
low. A level whose definition is ambiguous is not a level.
