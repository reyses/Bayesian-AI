# Cycle 01 — Phase A: pre-market news_intensity_today

PDCA cycle. Parent project: `project.md`.

## P — Plan

**Predicted outcome** (written BEFORE running, per MEMORY.md PDCA rule):

1. **IS WF Pearson**: expect modest lift from +0.191 baseline → +0.22 to
   +0.30. Mechanism: replacing 3 useless binary cols with 1 informative
   continuous col reduces noise and adds magnitude info for the ~80
   release days. The other 213 days are unchanged (intensity = 0).
2. **OOS sealed Pearson**: this is the gate. Need lower CI > 0. Expected:
   +0.20 to +0.35 point estimate, lower CI somewhere between -0.05 and
   +0.15. **High likelihood of falling just short** — the 23 OOS days
   include only ~5-8 release days, small N for the LLM signal to dominate.
3. **`news_intensity_today` ΔMAE**: expect +$40-80/day. If <$30/day,
   feature is too weak relative to other signals; if >$100/day, suspect
   LLM is cheating (memorized market reactions).

**Predicted failure modes**:
- LLM scores cluster flat ~5 → prompt eng / model too small
- IS lifts but OOS doesn't → LLM memorized IS-era releases (2024-2025)
- All scores correlate too strongly with realized day P&L → cheating

**Steps**:
1. Run `python -m tools.sourcing.llm_news.cli fetch` → ~80 press release `.txt` files
2. Spot-check 3 random files for HTML-strip quality
3. Run anti-cheating synthetic test (hawkish vs dovish hand-crafted statements
   must score in correct order)
4. Run `python -m tools.sourcing.llm_news.cli score` → `dev/news_scores_v1.parquet`
5. Spot-check 5 random scores manually (read text, check intensity number)
6. Verify std(intensity) across release days ≥ 1.5 (if flat, abort + iterate prompt)
7. Run `python -m tools.sourcing.llm_news.cli build` → `dev/cross_day_features_with_target_v2.parquet`
8. Verify canonical paths byte-identical (hash check)
9. Run `python -m tools.sourcing.llm_news.cli train` → `dev/drs_canonical_gbm_v2.pkl`
   + `research/llm_news_intensity/findings/{today}_phase_a_results.md`
10. Apply Phase A gate (3 criteria above)

## D — Do

Implementation references:
- Press release fetch: [tools/sourcing/llm_news/fetch.py](../../tools/sourcing/llm_news/fetch.py)
- LLM scoring: [tools/sourcing/llm_news/score.py](../../tools/sourcing/llm_news/score.py)
- Join helper: [tools/sourcing/llm_news/join.py](../../tools/sourcing/llm_news/join.py)
- CLI entry: [tools/sourcing/llm_news/cli.py](../../tools/sourcing/llm_news/cli.py)
- Augmenter: [tools/sourcing/build_cross_day_features_v2.py](../../tools/sourcing/build_cross_day_features_v2.py)
- Dev trainer: [tools/sourcing/drs_canonical_gbm_v2.py](../../tools/sourcing/drs_canonical_gbm_v2.py)

Model: `models/llama-3.1-8b-instruct-q4_k_m.gguf`
(Q4_K_M from `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`).

## C — Check

(Filled after running. Compare against P predictions, do not edit
predictions retroactively.)

| Metric                       | Predicted    | Actual | Notes |
| ---------------------------- | ------------ | ------ | ----- |
| IS WF Pearson                | +0.22 – +0.30 | ?      |       |
| IS WF Pearson lower CI       | ≥ +0.098    | ?      |       |
| OOS sealed Pearson           | +0.20 – +0.35 | ?      |       |
| OOS sealed Pearson lower CI  | > 0          | ?      |       |
| `news_intensity_today` ΔMAE  | +$40 – +$80  | ?      |       |
| Score std (release days)     | ≥ 1.5        | ?      |       |
| Anti-cheating synthetic test | Hawkish > dovish | ? |       |

## A — Act

(Filled after Check. Decision tree:)

- **All gates pass** → proceed to `cycle_02.md` (Phase B EOD column)
- **IS passes but OOS doesn't** → kill the feature. Means LLM signal
  doesn't generalize. Document hypothesis death in finding doc + project.md.
- **IS doesn't improve** → prompt eng issue or model too small. Iterate
  prompt in cycle_01b (new PDCA cycle), or upgrade to Qwen2.5-14B Q4.
- **ΔMAE < threshold** → feature is technically positive but not worth
  complexity. Kill.
- **Scores show no variance (std < 1.5)** → don't even retrain DRS.
  Iterate prompt or model size first.

## Findings

(Linked from this section as gate results are produced.)

- `findings/YYYY-MM-DD_phase_a_results.md` (created by `cli.py train`)
