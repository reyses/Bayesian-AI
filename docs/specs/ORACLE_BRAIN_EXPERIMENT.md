# Oracle Brain Experiment

## Hypothesis
IS PnL collapsed after removing 1-bar peak lookahead. Two possible causes:
1. **Brain quality**: IS with lookahead trained a better brain (direction biases, template WR).
   OOS inherited that quality. Without lookahead, IS trains a weak brain → everything suffers.
2. **Entry timing**: Peak detection fundamentally needs the next bar's state to time entries.
   No brain quality can compensate for bad timing.

## Experiment Design

### Phase A: Oracle Brain (lookahead IS)
Re-introduce the 1-bar lookahead ONLY for brain training purposes.
- Peak detection uses `_states_map.get(_i + 1)` instead of `_states_map.get(_i)`
- All other code paths remain on current bar (exits, compressed signals, TBN)
- Run full IS → brain learns from perfectly-timed peak entries
- Freeze brain at OOS boundary (same as before)
- Save as `checkpoints/oracle_brain.pkl`

### Phase B: Honest IS with Oracle Brain
- Load `oracle_brain.pkl` (frozen, no learning)
- Run IS with NO lookahead (current code, `_states_map.get(_i)`)
- Brain provides direction biases, template win rates, expected PnL from Phase A
- But entries use honest current-bar state

### Phase C: Control — Honest IS with Honest Brain
- This is the current `--fresh` run (already running/completed)
- Brain trained from honest entries, frozen for OOS

## What the Results Tell Us

| Phase B result | Phase C result | Diagnosis |
|---------------|---------------|-----------|
| Profitable | Negative | Brain quality is the bottleneck. Fix: better brain training. |
| Negative | Negative | Entry timing is broken. Fix: improve peak detector. |
| Both profitable | Both profitable | No problem (unlikely given current results). |
| B worse than C | — | Oracle brain is overconfident, hurts honest entries. |

## Implementation

### Option 1: CLI flag `--oracle-brain`
Add a flag that re-introduces lookahead in peak detection only:
```python
# In the IS bar loop, when building state for process_bar:
if _oracle_brain_mode:
    _bar_state = _states_map.get(_i + 1) or _states_map.get(_i)
else:
    _bar_state = _states_map.get(_i)
```

### Option 2: Two-pass pipeline
1. First pass: `python training/trainer.py --fresh --oracle-lookahead`
   - Builds oracle brain, saves to `oracle_brain.pkl`
2. Second pass: `python training/trainer.py --frozen-brain checkpoints/oracle_brain.pkl`
   - Runs honest IS with frozen oracle brain

Option 2 is cleaner — no lookahead code in the main path.

## Risk Assessment
- **Blast radius**: Minimal — only adds a CLI flag and state lookup override
- **Rollback**: Delete the flag, behavior unchanged
- **Time**: ~30 min to implement, ~6 hours for two full runs

## Notes
- Jules audit found 13 duplications (6 HIGH) in live_engine.py — noted but deferred
- The `vol=0 fm=0.0` peak entries suggest a missing minimum threshold on the detector
- Cat brain blocked 60% of detections in 1W test — may need loosening
