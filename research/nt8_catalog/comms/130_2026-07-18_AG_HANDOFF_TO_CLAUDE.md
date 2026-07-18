# HANDOFF TO CLAUDE: NT8 Native Port B (Phase 0/1 Prep)

**Status**: Interim Handoff
**Topic**: NT8 Native Port B
**Date**: 2026-07-18

## What I Did So Far
1. Set up the `research/nt8_port` WoW directories.
2. Drafted the exact specification into `implementation_plan.md` focusing on Phase 0 (Golden Vectors) and Phase 1 (Native C# Port).
3. Created a script `research/nt8_port/tools/find_top_k.py` and successfully extracted the 22 top streams that comprise >80% of the absolute coefficient sum from the 2024-sealed `combiner_preview.py` logistic model.
4. Saved the exact model weights, streams, and normalization parameters to `research/nt8_port/reports/top_k_streams.txt`.

## Pending to Hand Off (What Claude / Next Agent Needs to Do)
1. **Complete Phase P0 (Golden Vectors):**
   - Use the extracted top 22 streams to generate the per-1m-bar golden parity vectors for 20 reference days (e.g. 10 from 2024, 10 from 2025). 
   - A script `generate_golden_vectors.py` needs to be written that evaluates the `DayCtx` zigzag/R-trigger states and checks stream fires at exactly each 1-minute RTH boundary, running the logistic model using the fixed parameters in `top_k_streams.txt`.
   - Save these to `research/nt8_port/golden/*.parquet`.

2. **Phase P1 (C# Entry Port):**
   - Translate the 22 generators to C#.
   - Implement the `P` logistic inference in C# and map thresholds according to the 2024 quantiles.
   - Run C# and test against the golden parquets.

## Notes
- The user specifically requested a cron to be left open when I produce comms, so they can catch responses. I've left my polling crons active.
- `dossier_signal_pipeline.py` contains all the Python generator logic and the `DayCtx` that evaluates base causal features. 
- You can find the extracted logistic coefficients and top K streams inside `research/nt8_port/reports/top_k_streams.txt`.
