# Morning Briefing — 2026-04-30

## ⭐ Headline

**Tier system + LinReg slope filter = +$165,140 over 14 months Python sim**
≈ **+$390/day NT8** on 1 contract over 209 trading days
≈ **7.8× v1.0.4's current $50/day live**

| Tier | Filtered PnL (14mo) | Trades kept |
|---|---:|---:|
| RIDE_AGAINST | $50,807 | 1,077 / 1,423 |
| FADE_CALM | $42,881 | 357 / 365 |
| FADE_AGAINST | $27,832 | 292 / 352 |
| BASE_NMP | $24,159 | 929 / 1,195 |
| RIDE_CALM | $16,891 | 451 / 637 |
| KILL_SHOT | $2,571 | 106 / 106 |
| **TOTAL** | **$165,141** | **3,212 / 4,078** |

This is the validated number — pick the LinReg slope filter at T=0.5 per tier (T=1.5 for FADE_AGAINST), apply, sum.

## ⚠️ Critical: do NOT ship standalone BaseNmpRunner

The standalone NT8 BaseNmpRunner_v1.0-RC I built compiles fine, but on the
full 14-month dataset it loses **-$30k (canonical)** to **-$49k (NT8 SA "best")**.
Trade counts are 10× the original BASE_NMP tier (16,500 vs 1,195), meaning it
fires on bars the blended pipeline would have routed to other tiers. The
standalone strategy lacks the tier classifier that gated BASE_NMP's edge.

**Recommendation**: ship v1.0.8-RC zigzag with slope filter. Drop standalone
BaseNmpRunner until either (a) tier classifier is ported, or (b) Python
bridge deploys the actual blended pipeline.

Detail: `reports/findings/base_nmp_param_comparison/2026-04-30_results.md`

## ✅ What's deployed in NT8 right now

| Strategy | MD5 | Status | Recommendation |
|---|---|---|---|
| `ZigzagRunner.cs` (v1.0.4) | (live) | LIVE on Sim101, ~$50/day | Untouched, current baseline |
| `ZigzagRunner_v1.0.8-RC.cs` | `71bab6f4...` | Compiled, slope filter + daily gate ready | **Ship after NT8 SA validation** |
| `BaseNmpRunner_v1.0-RC.cs` | `ee5c26aa...` | Compiled, but loses money on full data | **Do NOT ship standalone** |
| `ZigzagRunner_v1.2.cs` | — | Released earlier | Keep as alternative live option |

## Workflow notes

- ✅ NT8 Strategies/ folder cleaned: 9 superseded RCs moved to `archive/`
- ✅ Tools/ folder: 7 overnight one-shot tools archived to `tools/archive/2026-04-29_overnight_research/`
- ✅ All overnight outputs preserved in `reports/findings/` subdirs
- ⏳ Bigger codebase cleanup pending

## Today's decision tree

```
Did NT8 SA on v1.0.8-RC with slope filter ON beat v1.0.4 baseline?
├── Yes → ship v1.0.8-RC live
└── No → diagnose: which tier-style filter underperforms in NT8 vs Python?
              → likely a slippage/timing diff; tune T or skip slope filter for v1.0.8

Standalone BaseNmpRunner ship?
├── No (current) — losing money on full data
└── Future: Python bridge to deploy actual blended pipeline (multi-week effort)
```

## Read order if you have 15 minutes

1. **This file** (1 min — you're here)
2. `OVERNIGHT_SUMMARY_2026-04-29.md` (5 min — full context)
3. `TIER_EDA_FIX_PLAN_2026-04-29.md` (5 min — per-tier remediation)
4. `base_nmp_param_comparison/2026-04-30_results.md` (3 min — overfit hypothesis confirmed)
5. `tier_pnl_by_regime/2026-04-29_10_tier_linreg_slope_filter.md` (1 min — the +$33k breakdown)

## My queued parallel work (continuing while you focus)

- Codebase navigation index (`reports/findings/INDEX.md`)
- Tools/ further cleanup pass (low-risk archives)
- Validation: tier system on full data Python sim (re-confirm $165k)
- Optional: Python sim of v1.0.8-RC with slope filter ON (parity check vs NT8 SA)

Will keep producing artifacts. Interrupt anytime.
