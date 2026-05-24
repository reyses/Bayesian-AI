# tools/_viz — Visualizers, markers, and inspectors

> Virtual folder. All files now live inside this folder (`tools/_viz/`) after the 2026-05-17 consolidation. This is the **single consolidated index** of every chart / marker / inspector tool in the project, per user request 2026-05-17.

## How to use

- **Inspectors** — interactive UIs (matplotlib windows with sliders/keys). Best for visual debugging and threshold tuning.
- **Markers** — click-to-mark workflows. Output goes to `DATA/cusp_picks/` or similar JSON files.
- **Charts** — one-shot rendering. Output goes to `chart/` or `examples/` or `reports/findings/`.

When in doubt, the active **classifier_inspector** and **cusp_marker** are the two most-used inspectors right now.

## Inspectors (interactive UIs)

| Tool | Purpose |
|------|---------|
| [`classifier_inspector.py`](classifier_inspector.py) | Classifier inspector — visual debugger for the entry-timing + direction |
| [`pivot_inspector.py`](pivot_inspector.py) | Pivot Inspector -- visualize real vs fakeout pivots on price chart. |
| [`seed_inspector.py`](seed_inspector.py) | Seed Inspector -- step through I-MR auto-seeds on a price chart for visual QA. |
| [`swing_inspector.py`](swing_inspector.py) | Swing Inspector -- grade continuous swing groups on a price chart. |

## Markers (manual / auto pivot picks)

| Tool | Purpose |
|------|---------|
| [`auto_swing_marker.py`](auto_swing_marker.py) | Auto Swing Marker -- ZigZag-based swing detector calibrated from human seeds. |
| [`cusp_marker.py`](cusp_marker.py) | Cusp Marker — peak-style single-click marker with multi-scale anchor overlays. |
| [`dmi_peak_marker.py`](dmi_peak_marker.py) | DMI + Volume peak marker — mark exhaustion points on DMI/Volume chart. |
| [`feature_marker.py`](feature_marker.py) | Feature Marker - Mark points on a price chart and toggle V2 features as overlays. |
| [`peak_marker.py`](peak_marker.py) | Peak Marker — Manually mark peaks on a price chart. |
| [`peak_marker_analysis.py`](peak_marker_analysis.py) | Peak Marker Analysis — analyze 1s data before human-marked peaks. |
| [`trade_marker.py`](trade_marker.py) | Trade Marker — Manually mark trades on a price chart with crosshair. |

## Day charts

| Tool | Purpose |
|------|---------|
| [`day_chart_other_features.py`](day_chart_other_features.py) | 1-day chart of the OTHER V2 features (beyond regression mean/slope). |
| [`day_chart_regression_means.py`](day_chart_regression_means.py) | 1-day chart: 5s price with 5m / 15m / 1h / 4h regression mean overlays. |
| [`day_chart_z_bands.py`](day_chart_z_bands.py) | High/Low REGRESSION CENTERS with σ bands in price space. |

## Chart_* — multi-purpose chart overlays

| Tool | Purpose |
|------|---------|
| [`chart_1s_trades.py`](chart_1s_trades.py) | Chart trades from the 1s-pivot forward pass on a single day. |
| [`chart_5s_1h_bands_multi_window.py`](chart_5s_1h_bands_multi_window.py) | Chart 5s price on a single day with 1h bands at MULTIPLE window sizes. |
| [`chart_5s_with_1h_bands.py`](chart_5s_with_1h_bands.py) | Chart 5s price with 1h regression-mean bands overlay. |
| [`chart_anchors_3level.py`](chart_anchors_3level.py) | Plot the three-level anchor framework for a single day: |
| [`chart_anchors_3level_full.py`](chart_anchors_3level_full.py) | Enhanced 3-anchor chart with: |
| [`chart_augmented_filters_full.py`](chart_augmented_filters_full.py) | Augmented filters chart — FULL version with multiple CRMs + multiple HL bands. |
| [`chart_augmented_filters_on_day.py`](chart_augmented_filters_on_day.py) | Visualize all four Bayesian-filter overlays on a single day. |
| [`chart_bayes_exit_oracle.py`](chart_bayes_exit_oracle.py) | Visualize Bayes exit oracle ON real tier trades for a single day. |
| [`chart_regression_z.py`](chart_regression_z.py) | Chart regression mean + z overlay on 1m price (zoomed-out view). |
| [`chart_rm_trades.py`](chart_rm_trades.py) | Chart a single day of RM-slope trades. |
| [`chart_strategy_comparison.py`](chart_strategy_comparison.py) | Visual comparison: CURRENT system (reg-flip + residual-flip + sniper) vs |
| [`chart_tf_bands_multi_window.py`](chart_tf_bands_multi_window.py) | Chart bands at any TF on a single day, with multiple window sizes. |
| [`chart_trade_dissection.py`](chart_trade_dissection.py) | Full multi-TF dissection of a single trade. |
| [`chart_trade_play_by_play.py`](chart_trade_play_by_play.py) | Bar-by-bar play-by-play of a single trade's entry trigger + progression. |

## Inspect_* — diagnostic plots

| Tool | Purpose |
|------|---------|
| [`inspect_direction_signals.py`](inspect_direction_signals.py) | Compare candidate DIRECTION signals on a single day. |
| [`inspect_noise_after_down.py`](inspect_noise_after_down.py) | Inspect the NOISE-after-STEEP_LINEAR_DOWN cell at note level. |
| [`inspect_noise_bucket.py`](inspect_noise_bucket.py) | Look at what's INSIDE the NOISE buckets — they're not idiosyncratic, they're |
| [`inspect_regression_mean.py`](inspect_regression_mean.py) | Visualize the regression mean and SE bands on top of price. |
| [`inspect_templates.py`](inspect_templates.py) | Template Inspector — Load checkpoint templates and dump a human-readable report. |

## DMI / regression / 3-body

| Tool | Purpose |
|------|---------|
| [`dmi_imr_chart.py`](dmi_imr_chart.py) | DMI I-MR Chart — Statistical Process Control for DMI diff. |
| [`dmi_peak_overlay.py`](dmi_peak_overlay.py) | Overlay existing human-marked peaks on DMI + Volume chart. |
| [`dmi_se_overlay.py`](dmi_se_overlay.py) | DMI + SE Bands + Volume overlay on historical price data. |
| [`dmi_session_chart.py`](dmi_session_chart.py) | Generate I-chart of a live session with trades, DMI, and volume. |
| [`dmi_swing_plot.py`](dmi_swing_plot.py) | Quick DMI swing analysis for a single day. |

## Other overlays / visualizers

| Tool | Purpose |
|------|---------|
| [`auto_seeds_day_chart.py`](auto_seeds_day_chart.py) | Visualize auto seeds (ZigZag) on 1m price for one day — compare with physics peaks. |
| [`entry_outcome_overlay_chart.py`](entry_outcome_overlay_chart.py) | entry_outcome_overlay_chart.py -- Time-axis visualization of fade entries with |
| [`imr_trade_chart.py`](imr_trade_chart.py) | I-MR Trade Chart — Visualize individual trade dynamics. |
| [`peak_feature_overlay_chart.py`](peak_feature_overlay_chart.py) | peak_feature_overlay_chart.py -- Time-axis visualization of physics features |
| [`peak_verification_chart.py`](peak_verification_chart.py) | Peak Verification Chart — overlay detected peaks on 1-day 1s price. |
| [`physics_day_chart.py`](physics_day_chart.py) | Visualize pure physics 5m exhaustion peaks overlaid on 1m price for one day. |
| [`physics_exit_chart.py`](physics_exit_chart.py) | Physics Exit: Enter at big 5m peak, exit at NEXT 5m peak (any size). |
| [`segment_chart_multilevel.py`](segment_chart_multilevel.py) | Render multi-level hierarchical segmentation charts for sample days. |
| [`session_overlay.py`](session_overlay.py) | Session Overlay — Map trades onto 1h + 1m price structure with adaptive Fibs. |
| [`trade_visualizer.py`](trade_visualizer.py) | Trade Visualizer -- Plots price waveform with entry/exit markers. |
| [`v2_features_overlay_viz.py`](v2_features_overlay_viz.py) | v2_features_overlay_viz.py — Visual overlay of features against price. |
| [`visual_shape_cnn.py`](visual_shape_cnn.py) | Visual Shape CNN — classifies level touches from candlestick images. |
| [`zigzag_multitf_overlay.py`](zigzag_multitf_overlay.py) | Multi-TF zigzag pivot overlay for ONE DAY. |
