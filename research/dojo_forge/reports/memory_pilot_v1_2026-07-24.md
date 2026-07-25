# Memory pilot v1 (arms A/B) — final stats, 2026-07-24

| arm | eps | frames | exits | memo rate | conf mean | conf sd | conf range |
|---|---|---|---|---|---|---|---|
| A: 04_08, no memory | 4 | 152 | 0 | 99% | 0.895 | 0.116 | 0.55–0.99 |
| B: 04_09, day-carry | 4 | 96 | 0 | 96% | 0.882 | 0.136 | **0.25**–0.98 |

Bank: 243 memos, 1 data-bearing (0.4%) — the v1 prompt's motto disease at
full scale; the direct motivation for tonight's v2seed sprint loop.

Findings:
1. **Memory widens confidence discrimination, not its level.** CORRECTION of
   the early-session read ("conf rose 0.65→0.80"): over ALL frames the means
   are equal (0.895 vs 0.882); what changed is the RANGE — arm B reaches down
   to 0.25 on frames where retrieved notes conflicted with the tape, and up
   to 0.98 on clean winners. The day-carry channel made it *more willing to
   be unsure* — a calibration prerequisite.
2. **Zero exits in 248 reasoning-mode frames** across both days. Never-bail
   behavior is total in generation mode on these days; whether that's the
   right call is the census/grounding question, not answerable here.
3. **Retrieval mechanics flawless**: every arm-B frame got its top-3
   prior-day memos under the guards; ledger audit-complete (retrievals,
   admissions, rejections all recorded).
4. Memo redundancy at scale: 239/243 memos are near-duplicate genome mottos.
   Info-rate 0.4% is tonight's baseline to beat (sprint bar: ≥20%).
