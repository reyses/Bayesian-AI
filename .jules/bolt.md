# Bolt's Journal

2024-05-22 — Initial Entry
Learning: No existing journal found. Created this file to track performance insights.
Action: Will document subsequent findings here.

2024-05-22 — Optimized extract_features
Learning: getattr() overhead is significant in tight loops, especially when the object structure is known (dataclasses). math.log1p/log2 are faster than numpy equivalents for scalar values.
Action: Implemented fast-path direct attribute access with EAFP pattern in extract_features, yielding ~2.7x speedup on feature extraction.
