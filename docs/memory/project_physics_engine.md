---
name: Physics Engine — alt live engine ($132/day baseline)
description: New live engine based on K-NN trajectory matching against enriched seeds. Separate from BarProcessor. $132/day OOS proven.
type: project
---

Physics Engine is the alternative live engine replacing BarProcessor for trading.

**Why:** BarProcessor captures 0.9% of available PnL ($1,844 OOS). Physics funnel captures $5,138 ($132/day) on same data. 72x improvement.

**Config (proven OOS, no lookahead):**
- 12-feature trajectory: fm, z, dmi_p, dmi_m, adx, vel, vol, hurst, P_center, coherence, sigma, pid
- 10-bar lookback window (1m bars)
- K=20 nearest seeds from 38K enriched IS library
- Consensus > 0.65 for direction
- Coherence < 0.6 (TF disagreement = real reversal)
- Magnitude > p25 (rolling window, seeded from IS)
- Hold from matched seed median duration (3-20 bars)

**How to apply:**
- Build as `core/physics_engine.py` (NOT modifying bar_processor.py)
- Load enriched seeds at startup from `DATA/regime_seeds/auto_seeds_all_*.json`
- Pre-compute seed trajectory matrix + normalization from IS data
- Per bar: update trajectory buffer → match → filter → enter/exit
- Wire into `live/live_engine.py` as alternative to BarProcessor

**Architecture:**
- Naming: "engine" not "processor"
- Separate from existing system (competing, not replacing)
- Uses StatisticalFieldEngine for state computation (same CUDA pipeline)
- Does NOT use: TBN, brain, cat, gates, exit cascade
- DOES use: enriched seed library, K-NN matching, coherence filter

**Enriched seeds location:** `DATA/regime_seeds/auto_seeds_all_20260322_154729.json` (426MB)
**Backup:** `checkpoints/backup_seeds_run_20260321/enriched_seeds.json`
