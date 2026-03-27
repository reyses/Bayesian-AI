---
name: Checkpoint every step
description: All multi-step pipelines must save intermediate results to disk after each step for crash recovery
type: feedback
---

All multi-step pipelines must save intermediate data to disk after EVERY step.

**Why:** Long-running pipelines (signal collection, L3 training data) can take hours. A crash at step 5 of 6 loses everything if steps 1-4 aren't persisted. The user lost time to this.

**How to apply:** Every step in a pipeline must:
1. Check if cached result exists on disk → load and skip if so
2. Compute the result
3. Save to disk (`.npy` for arrays, `.pkl` for complex objects, `.pt` for models)
4. Print "Saved: {path}" confirmation

Use pattern: `if os.path.exists(cache_path): load; else: compute + save`

**Shard, don't monolith:** Large datasets (signals, training samples, features) must be saved as shards (per-day or per-month files in a subdirectory), NOT as one giant file. Benefits:
- Crash during collection only loses the current shard, not everything
- Resume picks up from the last completed shard
- Memory-efficient loading (can stream shards)
- Pattern: `cache_dir/signals/2025_01_02.pkl`, `cache_dir/signals/2025_01_03.pkl`, etc.

This applies to: signal collection, training data building, model training, feature extraction — anything that takes > 30 seconds.
