# Daisy-chain oracle + entry/exit clustering — full IS 2025

**Run date:** 2026-05-14 (sleep run)
**Data:** `reports/findings/regret_oracle/daisy_chain_IS_full_daisy.csv` (7,925 trades)
**Coverage:** 2025-01-01 → 2025-12-31, 233 Globex sessions, 5s base TF, 60-min window

## TL;DR

1. **Daisy-chain oracle works structurally.** 7,925 sequential non-overlapping trades reach **$1,046,546 total MFE** for the year (theoretical sequential ceiling under the 1-hour-budget premise). That's ~$4,492/day if every chain link were tradeable.
2. **Per-cell mode>noise gate is too strict for daisy-chain output** — entries sit at previous-trade extremes (high-vol bars), so the noise floor median ($289) drowns out most cells. Compare *across* cells instead of each to noise.
3. **Entry-state clustering** (K=8, silhouette 0.16) finds **one strong asymmetric cluster (c7, n=215, mean $306)** — extreme-down state, 82% LONG. The other clusters are large (10-20% each) and mid-amplitude.
4. **Exit-state clustering** finds **two structural-pivot exits** — c4 (n=418, mean $323, 96% SHORT) = "the dump exit"; c1 (n=1089, mean $155, 94% LONG) = "the rally exit." These are clear endpoints.
5. **Joint entry×exit surfaces the rubber-band trades** — `c0→c4` (133 trades, all SHORT, **mean $361**) is "fired mid → crashed to ext_dn"; `c2→c4` (36 trades, **mean $423**) is "fired up-biased → reversed to crash."
6. **Direction is structurally implied by (entry-cluster, exit-cluster) pair** — 50% of joint cells are 100% directional. Once you know the entry and exit state classes, you know the direction.

## Full IS daisy-chain stats

| | |
|---|---|
| Total chained trades | **7,925** (LONG 4,005 / SHORT 3,920 — balanced) |
| Sessions | 233 |
| Trades/session avg | 34 |
| **MFE distribution** | mode **$59**, median **$96**, mean **$132** |
| Total MFE | **$1,046,546** |
| Duration | mode 59.5 min (cap), median 36.9 min, mean 34.6 min |
| Velocity | mode $1.75/min, median $3.43/min |
| Full-window-available | 6,805 of 7,925 (86%, after patching `full_window` semantic) |

## EDA findings (single-axis, entry-time-clean)

The noise-floor gate fails for daisy-chain (entries are at high-vol extreme bars, inflating noise floor to $289 median vs $87 for local-extrema). The **relative pattern across cells is what matters:**

**By tod (mean MFE):**
- phase_5_rth_pm: **$204** (n=1,411) — RTH afternoon dominates
- phase_6_pre_halt: $183 (n=628)
- phase_4_rth_am: $179 (n=1,245)
- phase_3_pre_rth: $112
- phase_1_post_halt: $110
- phase_2_overnight: **$90** (weakest — overnight chop)

**By liquidity quartile (monotone):**
- Q4_high: $217 mean
- Q3: $149
- Q2: $113
- Q1_low: $94

**By rail position:**
- below_Ml: **$177 mean** — largest excursions when price is below 1h low rail
- mid_band: $137
- near_Ml: $144
- near_Mh: $142
- above_Mh: **$129** — weakest (default uptrend chop)

**By 15m slope sign:**
- down: **$166** (bearish 15m slope → bigger trades)
- flat: $147
- up: $126

**By 15m z-bin:**
- ext_dn: **$169** — extreme 15m down extends the most
- mod_dn: $153
- mid: $144
- ext_up: $134
- mod_up: $130 (slight asymmetry — downside excursions slightly larger)

## Entry-state clustering (K=8, silhouette 0.161)

| Cluster | n | % | mean_$ | mode_$ | %LONG | Signature |
|---|---|---|---|---|---|---|
| **c7** | 215 | 2.7% | **$306** | $197 | 82% | Strong downward slopes; ext_dn z_15m (97%); below_Ml; extreme fan |
| c5 | 4 | 0.05% | $215 | — | 0% | TINY — degenerate |
| c3 | 1,112 | 14.0% | $159 | $105 | 79% | High fan; ext_dn z_15m; below_Ml |
| c0 | 1,463 | 18.5% | $138 | $51 | 25% | Mid z_15m; below_Ml; flat |
| c6 | 1,500 | 18.9% | $128 | $87 | 22% | ext_up z_15m; above_Mh; wide fan |
| c1 | 935 | 11.8% | $123 | $73 | 78% | ext_dn z_15m; above_Mh; wide fan |
| c4 | 1,605 | 20.3% | $113 | $69 | 83% | Mid z_15m; above_Mh |
| c2 | 1,091 | 13.8% | $104 | $59 | 19% | mod_up z_15m; above_Mh |

**Pattern**: c7 (the "extreme-down state" with strong downward 15s slopes and price well below 1h high) produces the biggest trades, with 82% being LONG entries (fading the dump). It's the rubber-band-up trade. Small n (2.7%) limits its practical weight.

The bulk of entries cluster into 4 large groups (c0, c4, c6, c2 — combined 71%), distinguished mostly by z_15m × rail position × direction. These look like the "default" oscillation entries — moderate amplitude.

**Caveat:** silhouette 0.161 is moderate-low. K-means is finding *some* structure but per MEMORY's 2026-05-03 / 2026-05-09 warnings, multi-D K-means can false-merge unrelated patterns. Treat clusters as descriptive groupings, not crisp categories.

## Exit-state clustering (K=8, silhouette 0.178)

| Cluster | n | mean_$ | mode_$ | %LONG | Signature |
|---|---|---|---|---|---|
| **c4** | 418 | **$323** | $313 | **4%** | Downward slopes; ext_dn; below_Ml; **extreme** fan — "the dump exit" |
| **c1** | 1,089 | $155 | $53 | **94%** | z_1h_low positive; ext_up; above_Mh; wide — "the rally exit" |
| c3 | 912 | $151 | $5 | 95% | mid; below_Ml — LONG exits in mid of the band |
| c6 | 937 | $137 | $97 | 94% | wide fan; ext_up; below_Ml |
| c5 | 936 | $123 | $59 | 6% | mod_dn; above_Mh; wide |
| c0 | 1,134 | $117 | $51 | 5% | mod_dn; below_Ml |
| c2 | 1,399 | $97 | $59 | 5% | mid; above_Mh |
| c7 | 1,100 | $86 | $61 | 93% | mod_up; above_Mh; normal |

**Pattern**: exits separate cleanly by direction (no mixed-direction clusters; 50% LONG-skew baseline). c4 is the "trade hit a structural floor in below_Ml extreme territory" cluster — the largest mean MFE at $323, almost exclusively SHORT exits. c1 mirrors it on the LONG side at $155 mean.

## Joint entry × exit (the rubber-band trades)

The **direction is structurally implied** by the (entry-cluster, exit-cluster) pair — 50% of joint cells have skew=100% (pure LONG or pure SHORT).

**Highest mean-$ joint cells** (each represents a specific trade-trajectory archetype):

| Entry → Exit | n | mean_$ | %LONG | Interpretation |
|---|---|---|---|---|
| **c0 → c4** | **133** | **$361** | 0% | Fired at "mid below_Ml" → crashed to "ext_dn extreme fan." Big SHORTS. |
| **c2 → c4** | 36 | **$423** | 0% | Fired at "mod_up above_Mh" → reversed to crash. Counter-trend SHORT. |
| c4 → c4 | 22 | $351 | 0% | "Above_Mh mid" entries → crash exits. SHORT extensions. |
| c6 → c4 | 21 | $451 | 0% | "ext_up above_Mh" → crash. The classic reversal-from-extension SHORT. |
| **c3 → c1** | 13 | $327 | 100% | LONG mirror — fade ext_dn back up to ext_up rally exit. |
| c1 → c4 | 42 | $268 | 0% | ext_dn-above_Mh → crash exit. |
| c1 → c1 | 68 | $284 | 100% | Same direction (ext_dn entry → ext_up exit) — full traversal LONG. |

**Highest count joint cells** (the common-flow archetypes):

| Entry → Exit | n | mean_$ | %LONG | Interpretation |
|---|---|---|---|---|
| c6 → c2 | 891 | $115 | 34% | "ext_up above_Mh" entry → "mid above_Mh" exit. Reversion back to mean (mixed direction). |
| c0 → c0 | 798 | $124 | 1% | "mid below_Ml" entry → "mod_dn below_Ml" exit. Continuation SHORTS. |
| c4 → c1 | 548 | $158 | 100% | "mid above_Mh" → "ext_up" rally exits. The standard LONG. |
| c3 → c3 | 536 | $143 | 99% | ext_dn entry → mid exit. Fade-the-dump-back-to-mean LONG. |

**Interpretation**: the dominant trade archetypes look like:
- **Standard LONG**: c4→c1 — entered mid-above_Mh, exited ext_up. 100% LONG, $158 mean.
- **Standard SHORT**: c0→c0 (continuation) or c6→c2 (reversion). Mixed amplitudes.
- **Big rubber-band SHORT**: c0→c4 / c2→c4 / c6→c4 — entered at various above-mean states, exited at the crash bottom (ext_dn extreme fan). $361-$451 mean, 0% LONG.
- **Big rubber-band LONG**: c3→c1 — entered at extreme dump, exited at rally top. $327 mean, 100% LONG.

## Caveats and what this is NOT

1. **IS-only.** Per MEMORY hard rule, no cell finding is trustworthy until OOS-validated on 2026. These clusters need 2026-OOS to confirm they aren't just K-means false-merges of 2025 idiosyncrasies.
2. **Silhouette ~0.16-0.18 is moderate-low.** Clusters describe *some* structure but aren't crisp. K-means false-merge risk applies.
3. **Direction implied at exit cluster, not entry cluster.** A real-time selector knows the entry cluster but NOT the exit cluster yet — so it can't use the "(entry, exit)" archetype directly. It would need an exit-cluster *predictor* trained on the trajectory.
4. **Noise-floor gate is broken for daisy-chain.** Don't read the `tradeable=False` for all single-axis cells as "no edge" — read the relative cell means/modes instead.

## Outputs (all under `reports/findings/regret_oracle/`)

- `daisy_chain_IS_full_daisy.csv` — 7,925 trades with entry + exit state + d_* + exit_d_* + oscillation columns
- `cell_stats_IS_full_daisy_*.csv` — per-cell EDA (one file per axis)
- `per_cell_per_trade_stats_IS_full_daisy.csv` — combined cell stats
- `cluster_entry_summary_IS_full_daisy.csv` — entry cluster profiles
- `cluster_exit_summary_IS_full_daisy.csv` — exit cluster profiles
- `cluster_entry_x_exit_IS_full_daisy.csv` — joint (entry-cluster × exit-cluster) cells
- `daisy_clusters_IS_full_daisy.csv` — full per-trade output with both cluster IDs

## Next steps (for tomorrow)

1. **2026 OOS run of the daisy-chain** to validate cluster survival. Apply same K-means model fit on 2025 to 2026 trades; check if cluster centroids still produce similar mean-$ rankings.
2. **Entry-cluster discrimination check** — train a classifier predicting which entry-cluster a trade falls into using only entry-time features. If easily separable → entry-cluster is a usable selector feature. If not → noise.
3. **Exit-cluster *prediction*** — given entry-cluster + time-since-entry + current state vector, can we predict the exit cluster? If yes, the (entry, predicted-exit) pair becomes a usable directional gate.
4. **Visual verification** — load the c0→c4 trades (the n=133 big-SHORT cluster) into `cusp_marker.py` to confirm they really look like rubber-band crashes on the chart.
5. **Reconsider K** — try K=5, 6, 10 to see if the cluster structure stabilizes or shifts. Silhouette 0.16 hints at sub-optimal K.
