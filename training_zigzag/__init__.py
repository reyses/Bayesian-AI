"""training_zigzag — causal forward-pass pipeline for the L5 / zigzag engine.

Parallel to training/ (the blended baseline-740 pipeline), but for the zigzag
system. It does NOT re-implement the engine: it imports and drives the live
zigzag engine (live/l5_decider.py + core/ledger.py) bar-by-bar over historical
data, with the streaming pivot detector — a causal forward pass, no lookahead.

See docs/JULES_TRAINING_ZIGZAG.md for the design.
Entry point: forward_zigzag.py
"""
