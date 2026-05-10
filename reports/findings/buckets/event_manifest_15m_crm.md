# Event-bucketed manifest (15m CRM characterization)

_Generated 2026-05-09T23:02:27.673509_

Selection mode:  `longest`   window: +/-2.0 hr   split filter: `IS`

## Population

- 5340 IS macro events from `reports/findings/band_touch_aggregation/macro_events_1h_hl.csv`
- 15m CRM features attached at event ENTRY bar (no lookahead)

## Why event-segmented + 15m CRM (not regime_2d)

`regime_labels_2d.csv` is a DAY-AGGREGATE label computed from the
same metrics that determine whether macro events appear at all,
so picking representative days from those labels is circular
(UP_SMOOTH days have 0 macro events because the SMOOTH label is
defined by the absence of those events). The unbiased substrate
is the EVENT itself, characterized by bar-level features measured
at the event entry.

## Per-axis bucket populations

### slope (5 bins)

| bin | label | n_events | feature_q_lo | feature_q_hi |
|----:|-------|---------:|-------------:|-------------:|
| 0 | negative_high_trend | 1056 | -0.33232 | -0.02500 |
| 1 | negative_low_trend | 1055 | -0.02480 | -0.00549 |
| 2 | no_trend | 1058 | -0.00547 | +0.00654 |
| 3 | positive_low_trend | 1058 | +0.00657 | +0.02098 |
| 4 | positive_high_trend | 1059 | +0.02101 | +0.78501 |

### curvature (3 bins)

| bin | label | n_events | feature_q_lo | feature_q_hi |
|----:|-------|---------:|-------------:|-------------:|
| 0 | negative_curvature | 1587 | -0.00036 | -0.00001 |
| 1 | no_curvature | 1588 | -0.00001 | +0.00001 |
| 2 | positive_curvature | 1589 | +0.00001 | +0.00073 |

### z_close (5 bins)

| bin | label | n_events | feature_q_lo | feature_q_hi |
|----:|-------|---------:|-------------:|-------------:|
| 0 | negative_far_z | 1067 | -12.57419 | -1.67464 |
| 1 | negative_near_z | 1069 | -1.67447 | -0.60541 |
| 2 | zero_z | 1068 | -0.60378 | +0.65979 |
| 3 | positive_near_z | 1068 | +0.66440 | +1.62608 |
| 4 | positive_far_z | 1068 | +1.62612 | +15.07592 |

### sigma_rank (5 bins)

| bin | label | n_events | feature_q_lo | feature_q_hi |
|----:|-------|---------:|-------------:|-------------:|
| 0 | low_sigma | 1066 | +0.00139 | +0.09167 |
| 1 | low_mid_sigma | 1068 | +0.09236 | +0.47778 |
| 2 | mid_sigma | 1069 | +0.47847 | +0.89097 |
| 3 | high_mid_sigma | 1069 | +0.89167 | +0.95000 |
| 4 | high_sigma | 1068 | +0.95069 | +1.00000 |

### r2adj_5m (5 bins)

| bin | label | n_events | feature_q_lo | feature_q_hi |
|----:|-------|---------:|-------------:|-------------:|
| 0 | high_variation | 1068 | -0.01724 | +0.07794 |
| 1 | high_mid_variation | 1068 | +0.07797 | +0.29337 |
| 2 | mid_variation | 1068 | +0.29352 | +0.51585 |
| 3 | low_mid_variation | 1068 | +0.51607 | +0.72582 |
| 4 | low_variation | 1068 | +0.72599 | +0.97162 |

## Notes

- LONGEST-event-in-bucket selection (default) emphasizes the most
  clinically-impactful sample per bucket.
- Charts at `chart/buckets/event_15m_crm_<axis>.png`.
- The 15m CRM is the chart-validated strategic gate from 2026-05-09.
  Slope sign + curvature + z_close + sigma_rank quantile fully
  characterize the event's 15m-scale context with no lookahead.
