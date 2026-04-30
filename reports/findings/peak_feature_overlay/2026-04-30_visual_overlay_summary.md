# Peak feature overlay — visual time-axis charts

Generated: 2026-04-30

## What this is

Companion to the 2026-04-29 statistical aggregates in this same folder.
Where the overnight outputs (`01_effect_size_table.csv`, `02_per_feature_dists.png`,
`03_h_vs_l_separation.png`, `04_per_tf_separation.png`) measured **Cohen's d**
between feature distributions at peaks vs baseline, **these charts add the
time dimension**: stacked subplots showing how the most-separating physics
features behave around each peak event in actual time.

The visual EDA was the original ask before bed on 2026-04-29; overnight
produced statistical aggregates only. These charts close that gap.

## How to read a chart

Each PNG is one trading day (UTC). Seven panels share one time axis:

| Panel | What | Reference lines |
|---|---|---|
| 1 (top, tallest) | 5m close + peaks marked. **H peaks = red ▼, L peaks = green ▲.** Marker size scales with source TF (1m smallest → 1W largest). | none |
| 2 | `5m_z_se` + `1m_z_se` overlay | ±2 (gray dashed), 0 |
| 3 | `5m_dmi_diff` + `1m_dmi_diff` (fast DMI) | 0 |
| 4 | `15m_dmi_diff` + `1h_dmi_diff` (slow DMI / trend) | 0 |
| 5 | `5m_variance_ratio` + `1m_variance_ratio` | 1.0 (revert vs trend boundary) |
| 6 | `1m_reversion_prob` + `1m_hurst` | 0.5 |
| 7 | `15m_velocity` + `1h_velocity` | 0 |

Vertical dotted guides drop from each peak through every panel — you can
read off the feature value at every peak by following the line down.

## Output location

```
reports/findings/peak_feature_overlay/per_day/YYYY_MM_DD.png
```

One PNG per peak-bearing trading day. ~209 days expected; days with no
peaks or no features data are skipped silently.

## Tool

`tools/peak_feature_overlay_chart.py`

```bash
# Sample mode (default 6 days, fast iteration)
python tools/peak_feature_overlay_chart.py --mode sample

# Sample with custom days
python tools/peak_feature_overlay_chart.py --mode sample \
    --days "2025-06-09,2026-02-19"

# Full per-day batch
python tools/peak_feature_overlay_chart.py --mode per_day
```

## Illustrative days (regime contrast)

### 2025-06-09 — chop / mean-reverting day
[per_day/2025_06_09.png](per_day/2025_06_09.png)

Visible patterns:
- Multiple H peaks (~21880) cluster mid-session 12:30-19:00 UTC
- Multiple L peaks (~21720) in early session 02:00-09:00 UTC
- `5m_z_se` oscillates cleanly between ±2 all day — classic mean-reverting
  envelope behavior
- `variance_ratio` stays mostly < 1 — confirms mean-reverting regime
- DMI slow (15m+1h) extended below zero — bearish drift underlying the chop
- This is a "fade-friendly" day in physics terms

### 2026-02-19 — DOWN-trend day
[per_day/2026_02_19.png](per_day/2026_02_19.png)

Visible patterns:
- H peaks early (24960 at session open) then **descending L peaks** at
  24800 → 24750 through the day
- `variance_ratio` > 1 during the decline — continuation, not revert
- DMI slow deeply negative throughout — strong bearish trend
- This is a "ride-the-trend" day; fade entries here lose

### Contrast

| Feature | Chop day (06-09) | Down-trend day (02-19) | Implication |
|---|---|---|---|
| variance_ratio | < 1 most of day | > 1 during decline | VR is the regime selector |
| DMI slow direction | extended bias but oscillating | persistent negative | trend strength signal |
| Peak distribution | both H and L throughout | H early, L only on decline | trend/chop visible at peak level |
| z_se range | clean ±2 oscillation | breakouts past -2 sustained | breakdown of mean-reversion |

These are visual hypotheses — the next step is to **quantify** what
combination of features at peak distinguishes a successful fade trade
from a failed one. That's the natural follow-on for the analysis cycle.

## Relationship to overnight statistical results

The visual EDA confirms the quantitative findings — but adds context the
aggregate stats can't capture:

| Statistical finding (overnight) | Visual confirmation (today) |
|---|---|
| 5m_dmi_diff Cohen's d=1.33 H vs L | DMI panel shows clear sign-flip at peak transitions |
| 15m_z_se Cohen's d=1.04 | 15m_z_se mostly inside ±2 with brief excursions at H/L peaks |
| variance_ratio noisy in aggregate | Panel 4 reveals VR as a CLEAN regime selector when read in time order |

The aggregate stats hide regime conditioning. Variance_ratio's mean is
similar at H peaks and L peaks (~1.0), so Cohen's d is small — but the
TIME PATTERN reveals it splits trend vs chop days cleanly. That's a
finding the histograms missed.

## Caveats

- **Features 5s cadence is forward-fill from each TF's native cadence** —
  so 5m_z_se updates only every 5 minutes. The step-pattern in the panels
  is the actual feature update rate, not a rendering artifact.
- **Peaks are from the existing `DATA/regime_seeds/*.json`** (manual full
  range + auto + 4274 total across 6 TFs). Quality of peak detection is
  taken as given.
- **No outcome labels in these charts** — they show feature behavior at
  peaks but don't yet tell you which peaks would have been profitable
  fades. That's the next analysis (overlay trade outcomes from
  blended_is.csv → fresh CSVs after pipeline rerun).

## Next analysis cycles

1. **Outcome overlay**: same chart format, but mark each peak with a colored
   ring (green = profitable fade, red = losing fade) using trade outcomes
   from `sandbox/training/output/trades/blended_is.csv`. Tells us at a
   glance which peak signatures worked.
2. **Per-peak zoom mode**: `tools/peak_feature_overlay_chart.py --mode per_peak`
   (deferred — implement when user requests). ±60 5s-bars window per peak,
   tighter visual inspection.
3. **Cross-peak correlation**: do features at H peaks at one TF correlate
   with features at L peaks at a different TF? (multi-TF peak alignment EDA).
4. **Refresh after blended pipeline rerun**: regenerate when fresh
   trade-outcome data is available; overlay outcomes on these same charts.
