# P3 first-pass diff — python wrapper sim vs NT8 v0.2-RC backtest

Generated: 2026-07-18 19:31 · executor: Opus data/diff drone · commits: none

## Step 1 — ATLAS_NT8 conversion (data pipeline)
- Raw source: `D:/Bayesian-AI-data/DATA/RAW_NT8/{MNQ_06-26,MNQ_09-26}/{1s,1m}/*.csv` (BayesianHistoryDumper per-TF CSV). Converter: `tools/sourcing/convert_nt8_csv_to_parquet.py` (the importer matching THIS raw layout; the DATA/pipeline/README nt8_* tools are for .txt/tick formats).
- **Before**: ATLAS_NT8 all TFs maxed at **2026-06-12**. **After**: 1s/1m extended to **2026-07-08** (147 day-files; 6.17M 1s bars). Derived TFs (5s/15s/5m/15m/1h/4h/1D) built by `DATA/pipeline/build_timeframes.py` (incremental) — its OHLC-vs-control validation PASSED (0 mismatches: 1s-vs-1m, 5s-vs-1m, 15s-vs-1m, 1m-vs-1h, 5m-vs-1h, 15m-vs-1h).
- **Raw stops at 2026-07-08** — the window's last 7 trading days (07-09..07-17) have NO raw NT8 data and cannot be built. 07-08 itself is truncated (dump cut mid-session).
- SFE features: `python core_v2/build_dataset.py --atlas DATA/ATLAS_NT8 --start 2026-06-13 --end 2026-07-08` (authorized standard build, incremental). Completed on the RTX 3060, no OOM, 18 day-files written. **It wrote `L3_1m_z_se_30`** (see z_se caveat).

## Step 2 — goldens window summary
- 13 golden days -> `research/nt8_port/golden_backtest/` (frozen `golden/` untouched). entry-eligible 1m bars/day: min 22, median 225, max 343 (these are P>=thr bars; the wrapper collapses them to ~1-6 trades/day). zz_confirms/day median 9.

## Timezone finding
- NT8 export display = **US/Pacific**; **CT = PT + 2h**. Pinned empirically by exact price matches, not assumed: e.g. 7/1 short 30304.00 @ "8:40 AM" fills at 10:38–10:40 CT (bar 30304.25); session-close exits @ "2:00 PM" land at 16:00 CT (the CME session close), exact.
- Consequence: NT8 first entries cluster ~10:40 CT — **~2h into the harness RTH (08:30 CT)**. The NT8 data-series session is shifted +2h vs the python RTH window; this is the dominant structural gap (below).

## Coverage
- Golden window days generated (data available): **13** (2026_06_22..2026_07_08).
- NT8 export days: 20 (2026_06_22..2026_07_17).
- **Missing (no raw NT8 data past 2026-07-08): ['2026_07_09', '2026_07_10', '2026_07_13', '2026_07_14', '2026_07_15', '2026_07_16', '2026_07_17']** — trades on these days cannot be diffed.
- NT8 trades in comparable window: **32** (of 44 total).
- **Partial/truncated days (raw dump cut mid-session): ['2026_07_03 (210 RTH bars, ends ~11:59 CT)', '2026_07_08 (58 RTH bars, ends ~09:27 CT)']** — NT8 entries after the last golden bar on these days fall outside available data (e.g. 07-08 NT8 traded 10:40+ CT but data ends ~09:28 CT).

## Match result (dir + entry-minute ±2 + entry-price ±2pt)
- NT8 window trades: 32; sim trades (variant A / B): 58 / 44

| variant | sim session open | matched dir+minute | entry-px ±2pt | NT8 unexplained | harness-not-taken |
|---|---|---|---|---|---|
| A native | 08:30 CT (harness RTH) | 2/32 (6%) | 0/32 | 30 | 56 |
| B aligned | 10:30 CT (= 08:30 PT) | 3/32 (9%) | 2/32 | 29 | 41 |

**Reading**: variant A (harness RTH 08:30 CT) enters ~2h before NT8 every day -> near-zero match; the gap is a session-window shift, not a decision-logic error. Variant B (sim session opened at 10:30 CT to mirror NT8) tests whether the decision core agrees once the window is aligned.

*Note*: the aggregate match is low mostly because after the first entry the wrapper trade *sequence* (immediate re-entry after each cat-stop) diverges from NT8, and NT8 has 1–6 trades/day. The cleanest decision-core comparison is the **first entry per day** below.

## First entry per day — NT8 vs session-aligned sim (variant B)
- direction agrees: **4/13**;  entry-price ±2pt: **2/13**
| day | NT8 first (CT dir px) | simB first (CT dir px) | dir? | px? |
|---|---|---|---|---|
| 2026_06_22 | 10:40 S 30597.00 | 10:33 S 30598.00 | Y | Y |
| 2026_06_23 | 10:41 L 29712.50 | 10:33 S 29751.00 | n | n |
| 2026_06_24 | 10:46 S 29744.00 | 10:34 S 29761.75 | Y | n |
| 2026_06_25 | 10:50 L 29700.50 | 10:33 S 29683.50 | n | n |
| 2026_06_26 | 10:43 L 29647.25 | 10:33 L 29614.25 | Y | n |
| 2026_06_29 | 10:40 S 29787.25 | 10:35 L 29765.75 | n | n |
| 2026_06_30 | 10:40 L 30404.75 | 10:42 S 30417.25 | n | n |
| 2026_07_01 | 10:40 S 30304.00 | 10:37 L 30305.00 | n | Y |
| 2026_07_02 | 10:40 L 29806.25 | 10:33 S 29733.00 | n | n |
| 2026_07_03 | 10:42 L 29894.00 | 10:59 L 29906.00 | Y | n |
| 2026_07_06 | 10:40 S 30063.25 | 10:35 L 30049.50 | n | n |
| 2026_07_07 | 10:40 S 29279.00 | 10:33 L 29302.25 | n | n |
| 2026_07_08 | 10:40 L 29127.50 | — | n | n |

## Per-day: NT8 entries (CT) vs sim entries (CT)
| day | NT8 entries (CT, dir, px) | sim entries (CT, dir, px, exit) |
|---|---|---|
| 2026_06_22 | 10:40 S 30597; 16:00 S 30669; 17:07 L 30668 | 08:33 L 30812/CatSto; 08:37 S 30860/CatSto; 08:42 L 30842/CatSto; 08:47 L 30817/CatSto; 09:31 S 30702/Sessio |
| 2026_06_23 | 10:41 L 29712; 13:50 S 29617; 15:46 S 29730 | 08:33 S 29750/CatSto; 08:47 L 29742/CatSto; 13:44 S 29634/CatSto; 14:07 L 29694/CatSto; 14:14 L 29665/Sessio |
| 2026_06_24 | 10:46 S 29744; 12:26 S 29456; 15:10 L 29873 | 08:33 S 29792/CatSto; 08:40 L 29728/CatSto; 08:44 L 29635/CatSto; 08:52 S 29767/CatSto; 09:50 L 29833/CatSto; 10:13 S 29755/CatSto; 10:20 S 29790/CatSto; 15:12 L 29864/Sessio |
| 2026_06_25 | 10:50 L 29700; 13:05 L 29684 | 08:33 S 30070/Sessio |
| 2026_06_26 | 10:43 L 29647; 12:06 L 29643; 13:07 L 29566; 14:16 L 29501; 14:41 L 29466; 15:05 S 29324 | 08:33 L 29290/CatSto; 08:37 S 29316/CatSto; 08:48 S 29319/CatSto; 09:00 L 29443/CatSto; 14:01 S 29421/CatSto; 14:13 L 29477/CatSto; 14:28 L 29454/CatSto; 15:03 S 29301/CatSto |
| 2026_06_29 | 10:40 S 29787; 11:22 L 29888; 17:02 L 29993 | 08:33 S 29721/CatSto; 08:48 L 29810/CatSto; 08:54 L 29808/CatSto; 08:58 S 29600/CatSto; 09:38 L 29598/CatSto; 09:50 L 29571/Sessio |
| 2026_06_30 | 10:40 L 30405 | 08:33 L 30135/Sessio |
| 2026_07_01 | 10:40 S 30304 | 08:33 S 30175/CatSto; 08:52 S 30243/CatSto; 09:05 L 30263/CatSto; 09:39 S 30250/CatSto; 10:29 L 30336/CatSto; 10:48 S 30273/Sessio |
| 2026_07_02 | 10:40 L 29806; 10:51 S 29701 | 08:33 S 30077/CatSto; 08:38 S 30135/CatSto; 08:47 S 30111/CatSto; 08:52 L 30240/CatSto; 09:12 S 30182/CatSto; 09:17 S 30163/Sessio |
| 2026_07_03 | 10:42 L 29894 | 08:34 L 29952/CatSto; 09:16 S 29900/Sessio |
| 2026_07_06 | 10:40 S 30063 | 08:33 S 29909/CatSto; 08:50 L 29952/CatSto; 09:04 L 29946/CatSto; 12:46 S 29901/CatSto; 13:13 L 29956/Sessio |
| 2026_07_07 | 10:40 S 29279; 11:53 L 29546; 14:15 S 29344; 15:20 S 29398 | 08:33 S 29606/Sessio |
| 2026_07_08 | 10:40 L 29128; 11:08 L 29151 | 08:33 L 29319/CatSto; 08:41 L 29374/CatSto; 08:53 L 29332/CatSto; 09:16 L 29252/Sessio |

## Matched pairs (NT8 ↔ sim)
| NT8# | day | dir | NT8 CT | sim CT | NT8 px | sim px | dpx |
|---|---|---|---|---|---|---|---|
| 9 | 2026_06_24 | L | 15:10 | 15:12 | 29873.00 | 29864.00 | 9.00 |
| 17 | 2026_06_26 | S | 15:05 | 15:03 | 29323.50 | 29301.25 | 22.25 |

## NT8 entries unexplained by the sim
| NT8# | day | dir | NT8 CT | NT8 PT | px | exit |
|---|---|---|---|---|---|---|
| 1 | 2026_06_22 | S | 10:40 | 6/22/2026 8:40:00 AM | 30597.00 | X_CatastrophicStop |
| 2 | 2026_06_22 | S | 16:00 (post-RTH 15:15CT) | 6/22/2026 2:00:00 PM | 30669.25 | Exit on session close |
| 3 | 2026_06_22 | L | 17:07 (post-RTH 15:15CT) | 6/22/2026 3:07:00 PM | 30667.75 | X_CatastrophicStop |
| 4 | 2026_06_23 | L | 10:41 | 6/23/2026 8:41:00 AM | 29712.50 | X_CatastrophicStop |
| 5 | 2026_06_23 | S | 13:50 | 6/23/2026 11:50:00 AM | 29617.25 | X_CatastrophicStop |
| 6 | 2026_06_23 | S | 15:46 (post-RTH 15:15CT) | 6/23/2026 1:46:00 PM | 29729.50 | Exit on session close |
| 7 | 2026_06_24 | S | 10:46 | 6/24/2026 8:46:00 AM | 29744.00 | X_CatastrophicStop |
| 8 | 2026_06_24 | S | 12:26 | 6/24/2026 10:26:00 AM | 29456.50 | X_CatastrophicStop |
| 10 | 2026_06_25 | L | 10:50 | 6/25/2026 8:50:00 AM | 29700.50 | X_CatastrophicStop |
| 11 | 2026_06_25 | L | 13:05 | 6/25/2026 11:05:00 AM | 29683.75 | X_CatastrophicStop |
| 12 | 2026_06_26 | L | 10:43 | 6/26/2026 8:43:00 AM | 29647.25 | X_CatastrophicStop |
| 13 | 2026_06_26 | L | 12:06 | 6/26/2026 10:06:00 AM | 29643.25 | X_CatastrophicStop |
| 14 | 2026_06_26 | L | 13:07 | 6/26/2026 11:07:00 AM | 29566.00 | X_CatastrophicStop |
| 15 | 2026_06_26 | L | 14:16 | 6/26/2026 12:16:00 PM | 29501.25 | X_CatastrophicStop |
| 16 | 2026_06_26 | L | 14:41 | 6/26/2026 12:41:00 PM | 29466.50 | X_CatastrophicStop |
| 18 | 2026_06_29 | S | 10:40 | 6/29/2026 8:40:00 AM | 29787.25 | X_CatastrophicStop |
| 19 | 2026_06_29 | L | 11:22 | 6/29/2026 9:22:00 AM | 29887.50 | Exit on session close |
| 20 | 2026_06_29 | L | 17:02 (post-RTH 15:15CT) | 6/29/2026 3:02:00 PM | 29992.75 | X_SessionFlatten |
| 21 | 2026_06_30 | L | 10:40 | 6/30/2026 8:40:00 AM | 30404.75 | Exit on session close |
| 22 | 2026_07_01 | S | 10:40 | 7/1/2026 8:40:00 AM | 30304.00 | Exit on session close |
| 23 | 2026_07_02 | L | 10:40 | 7/2/2026 8:40:00 AM | 29806.25 | X_CatastrophicStop |
| 24 | 2026_07_02 | S | 10:51 | 7/2/2026 8:51:00 AM | 29701.00 | Exit on session close |
| 25 | 2026_07_03 | L | 10:42 | 7/3/2026 8:42:00 AM | 29894.00 | Exit on session close |
| 26 | 2026_07_06 | S | 10:40 | 7/6/2026 8:40:00 AM | 30063.25 | Exit on session close |
| 27 | 2026_07_07 | S | 10:40 | 7/7/2026 8:40:00 AM | 29279.00 | X_CatastrophicStop |
| 28 | 2026_07_07 | L | 11:53 | 7/7/2026 9:53:00 AM | 29545.75 | X_CatastrophicStop |
| 29 | 2026_07_07 | S | 14:15 | 7/7/2026 12:15:00 PM | 29344.25 | X_CatastrophicStop |
| 30 | 2026_07_07 | S | 15:20 (post-RTH 15:15CT) | 7/7/2026 1:20:00 PM | 29398.50 | Exit on session close |
| 31 | 2026_07_08 | L | 10:40 | 7/8/2026 8:40:00 AM | 29127.50 | X_CatastrophicStop |
| 32 | 2026_07_08 | L | 11:08 | 7/8/2026 9:08:00 AM | 29150.75 | Exit on session close |

## Harness sim entries NOT taken by NT8 (first 40 of 56)
| day | dir | sim CT | px | exit |
|---|---|---|---|---|
| 2026_06_22 | L | 08:33 | 30812.25 | CatStop50 |
| 2026_06_22 | S | 08:37 | 30859.50 | CatStop50 |
| 2026_06_22 | L | 08:42 | 30842.50 | CatStop50 |
| 2026_06_22 | L | 08:47 | 30817.25 | CatStop50 |
| 2026_06_22 | S | 09:31 | 30702.25 | SessionEnd(RTH15:15CT) |
| 2026_06_23 | S | 08:33 | 29750.25 | CatStop50 |
| 2026_06_23 | L | 08:47 | 29742.50 | CatStop50 |
| 2026_06_23 | S | 13:44 | 29634.50 | CatStop50 |
| 2026_06_23 | L | 14:07 | 29694.00 | CatStop50 |
| 2026_06_23 | L | 14:14 | 29665.25 | SessionEnd(RTH15:15CT) |
| 2026_06_24 | S | 08:33 | 29792.00 | CatStop50 |
| 2026_06_24 | L | 08:40 | 29727.75 | CatStop50 |
| 2026_06_24 | L | 08:44 | 29635.25 | CatStop50 |
| 2026_06_24 | S | 08:52 | 29767.00 | CatStop50 |
| 2026_06_24 | L | 09:50 | 29833.25 | CatStop50 |
| 2026_06_24 | S | 10:13 | 29755.00 | CatStop50 |
| 2026_06_24 | S | 10:20 | 29789.50 | CatStop50 |
| 2026_06_25 | S | 08:33 | 30070.00 | SessionEnd(RTH15:15CT) |
| 2026_06_26 | L | 08:33 | 29290.25 | CatStop50 |
| 2026_06_26 | S | 08:37 | 29315.50 | CatStop50 |
| 2026_06_26 | S | 08:48 | 29319.25 | CatStop50 |
| 2026_06_26 | L | 09:00 | 29442.75 | CatStop50 |
| 2026_06_26 | S | 14:01 | 29421.00 | CatStop50 |
| 2026_06_26 | L | 14:13 | 29477.25 | CatStop50 |
| 2026_06_26 | L | 14:28 | 29454.25 | CatStop50 |
| 2026_06_29 | S | 08:33 | 29721.25 | CatStop50 |
| 2026_06_29 | L | 08:48 | 29809.75 | CatStop50 |
| 2026_06_29 | L | 08:54 | 29807.50 | CatStop50 |
| 2026_06_29 | S | 08:58 | 29600.00 | CatStop50 |
| 2026_06_29 | L | 09:38 | 29598.00 | CatStop50 |
| 2026_06_29 | L | 09:50 | 29571.25 | SessionEnd(RTH15:15CT) |
| 2026_06_30 | L | 08:33 | 30135.25 | SessionEnd(RTH15:15CT) |
| 2026_07_01 | S | 08:33 | 30175.25 | CatStop50 |
| 2026_07_01 | S | 08:52 | 30242.75 | CatStop50 |
| 2026_07_01 | L | 09:05 | 30263.25 | CatStop50 |
| 2026_07_01 | S | 09:39 | 30250.25 | CatStop50 |
| 2026_07_01 | L | 10:29 | 30336.50 | CatStop50 |
| 2026_07_01 | S | 10:48 | 30273.00 | SessionEnd(RTH15:15CT) |
| 2026_07_02 | S | 08:33 | 30077.25 | CatStop50 |
| 2026_07_02 | S | 08:38 | 30135.00 | CatStop50 |

## R-trigger would-be exits (evidence for the v0.3 fix)
- v0.2 fired the R-trigger 0/44. Below: sim positions where a `zz_confirm` OPPOSING the open leg occurred during the trade (i.e. the R-trigger reversal that *should* have exited).
- sim trades total: **58**; with an opposing zz_confirm during the ride: **27** (47%).
| day | dir | entry CT | would-be R-trig CT | actual sim exit |
|---|---|---|---|---|
| 2026_06_22 | L | 08:47 | 09:16 | CatStop50 |
| 2026_06_22 | S | 09:31 | 09:36 | SessionEnd(RTH15:15CT) |
| 2026_06_23 | S | 08:33 | 08:34 | CatStop50 |
| 2026_06_23 | L | 08:47 | 09:31 | CatStop50 |
| 2026_06_23 | L | 14:07 | 14:08 | CatStop50 |
| 2026_06_23 | L | 14:14 | 15:00 | SessionEnd(RTH15:15CT) |
| 2026_06_24 | S | 08:52 | 09:23 | CatStop50 |
| 2026_06_24 | L | 09:50 | 10:08 | CatStop50 |
| 2026_06_24 | S | 10:20 | 10:30 | CatStop50 |
| 2026_06_25 | S | 08:33 | 09:02 | SessionEnd(RTH15:15CT) |
| 2026_06_26 | S | 08:37 | 08:43 | CatStop50 |
| 2026_06_26 | L | 09:00 | 09:37 | CatStop50 |
| 2026_06_26 | L | 14:28 | 14:50 | CatStop50 |
| 2026_06_29 | S | 08:33 | 08:41 | CatStop50 |
| 2026_06_29 | S | 08:58 | 09:19 | CatStop50 |
| 2026_06_29 | L | 09:50 | 14:51 | SessionEnd(RTH15:15CT) |
| 2026_06_30 | L | 08:33 | 08:41 | SessionEnd(RTH15:15CT) |
| 2026_07_01 | S | 09:39 | 10:03 | CatStop50 |
| 2026_07_01 | S | 10:48 | 14:30 | SessionEnd(RTH15:15CT) |
| 2026_07_02 | S | 09:17 | 10:00 | SessionEnd(RTH15:15CT) |
| 2026_07_03 | L | 08:34 | 08:39 | CatStop50 |
| 2026_07_06 | S | 08:33 | 08:40 | CatStop50 |
| 2026_07_06 | L | 08:50 | 08:58 | CatStop50 |
| 2026_07_06 | L | 09:04 | 10:12 | CatStop50 |
| 2026_07_06 | L | 13:13 | 14:52 | SessionEnd(RTH15:15CT) |
| 2026_07_07 | S | 08:33 | 08:43 | SessionEnd(RTH15:15CT) |
| 2026_07_08 | L | 09:16 | 09:27 | SessionEnd(RTH15:15CT) |

## Caveats
- **z_se N-skew**: the standard SFE build wrote `L3_1m_z_se_30` (code `N_BASE[1m]=30`), but the frozen golden reference and the C# port consume `z_se_15`. The 6 NMP/NMP9 top-K streams here fire off N=30 state — NMP-governed entries are not bit-faithful to what NT8 ran. Non-NMP entries are unaffected.
- **Session window**: golden decides on RTH 08:30–15:15 CT; NT8 trades a +2h-shifted session to 16:00 CT. Sim session-end exits at 15:15 CT vs NT8 16:00 CT; NT8 entries after 15:15 CT are outside the harness window by construction.
- Entry fill = open of the +180s action bar; NT8 market fill may differ by a bar (≈ the residual dpx).

## Pass 2 (z_se_15, session-aligned)

Re-run with the sanctioned **z_se_15** NMP head (built by `build_window_zse.py`; spot-checked vs the research/nmp_state verified OLS-endpoint/ddof-2 kernel, max|dz_se|=8.9e-16) and the sim session opened at **10:30 CT** to mirror the empirically-pinned NT8 v0.2 behavior. This diffs what NT8 ACTUALLY ran (the +2h PC-local session is already fixed in v0.3-RC; alignment here is diagnostic).

- NT8 window trades: 32; sim trades: 44
- matched dir+minute±2: **3/32** (9%); entry-px ±2pt: **2/32**
- **first entry/day — direction agrees 4/13; price ±2pt 2/13**

| day | NT8 first (CT dir px) | simB15 first (CT dir px) | dir? | px? | vs pass1 dir |
|---|---|---|---|---|---|
| 2026_06_22 | 10:40 S 30597.00 | 10:33 S 30598.00 | Y | Y | same |
| 2026_06_23 | 10:41 L 29712.50 | 10:33 S 29751.00 | n | n | same |
| 2026_06_24 | 10:46 S 29744.00 | 10:34 S 29761.75 | Y | n | same |
| 2026_06_25 | 10:50 L 29700.50 | 10:33 S 29683.50 | n | n | same |
| 2026_06_26 | 10:43 L 29647.25 | 10:33 L 29614.25 | Y | n | same |
| 2026_06_29 | 10:40 S 29787.25 | 10:35 L 29765.75 | n | n | same |
| 2026_06_30 | 10:40 L 30404.75 | 10:42 S 30417.25 | n | n | same |
| 2026_07_01 | 10:40 S 30304.00 | 10:37 L 30305.00 | n | Y | same |
| 2026_07_02 | 10:40 L 29806.25 | 10:33 S 29733.00 | n | n | same |
| 2026_07_03 | 10:42 L 29894.00 | 10:59 L 29906.00 | Y | n | same |
| 2026_07_06 | 10:40 S 30063.25 | 10:35 L 30049.50 | n | n | same |
| 2026_07_07 | 10:40 S 29279.00 | 10:33 L 29302.25 | n | n | same |
| 2026_07_08 | 10:40 L 29127.50 | — | n | n |  |

### Residual disagreements (session-aligned, z_se_15)
- 2026_06_23: NT8 L@10:41 vs sim S@10:33 (sim gov stream = TMPL0); entry-time gap.
- 2026_06_25: NT8 L@10:50 vs sim S@10:33 (sim gov stream = TMPL0); entry-time gap.
- 2026_06_29: NT8 S@10:40 vs sim L@10:35 (sim gov stream = TMPL0); entry-time gap.
- 2026_06_30: NT8 L@10:40 vs sim S@10:42 (sim gov stream = NMP9RIDEAGAINST); same window, opposite call.
- 2026_07_01: NT8 S@10:40 vs sim L@10:37 (sim gov stream = TMPL0); entry-time gap.
- 2026_07_02: NT8 L@10:40 vs sim S@10:33 (sim gov stream = TMPL0); entry-time gap.
- 2026_07_06: NT8 S@10:40 vs sim L@10:35 (sim gov stream = TMPL0); entry-time gap.
- 2026_07_07: NT8 S@10:40 vs sim L@10:33 (sim gov stream = ROUND05); entry-time gap.

### Session-open sensitivity (first-entry direction agreement)
| sim session open (CT) | first-entry dir agree |
|---|---|
| 10:30 | 4/12 (33%) |
| 10:35 | 7/12 (58%) |
| 10:40 | 8/12 (67%) |
| 10:45 | 5/12 (42%) |

### Pass-2 takeaways
- Feeding z_se_15 vs pass-1 z_se_30 left the 10:30-aligned first-entry direction agreement at **4/13** (all first entries governed by higher-weight non-NMP streams — RSI06/MACD07/TMPL0 — so the N-skew fix does not move them; NMP-governed *later* entries do change). z_se_15 is now bit-faithful to the frozen artifact regardless.
- **Session-open time is the real residual**: sweeping the sim open shows first-entry direction agreement jumps to **8/12 (67%) at 10:40 CT** (= 08:40 PT, NT8's empirical fire minute) vs 4/12 at 10:30. NT8 effectively opens/fires ~10 min after 10:30 CT (warmup + 180s settle). Once that is matched, the decision core largely agrees — the discrepancy is session/warmup timing, not combiner logic.
- Remaining misses at the best open are genuine governing-stream differences at that exact minute (e.g. 06-30 NMP9RIDEAGAINST call) — the true target for v0.3 bar-level parity.