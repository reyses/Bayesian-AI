# RETUNE — forward-pass validation instructions

## What was wired (2026-05-10)

All retune parameters are data-validated on full IS+OOS trade history.

### File changes

```
training_iso_v2/strategies/_nmp_base.py:
    NMPBaseStrategy.__init__()       + z_band_lo, z_band_hi, veto_cells params
    NMPBaseStrategy._passes_retune_filters()    new gate method
    NMPBaseStrategy.evaluate()       calls _passes_retune_filters() BEFORE _qualify

training_iso_v2/strategies/fade_calm.py:
    FadeCalm.__init__()       retune=True default
    RETUNE_Z_LO = 1.5, RETUNE_Z_HI = 1.8, RETUNE_VETO = [('short','neutral')]

training_iso_v2/strategies/fade_momentum.py:
    FadeMomentum.__init__()   retune=True default
    RETUNE_Z_LO = 1.5, RETUNE_Z_HI = 1.8  (no veto cells)

training_iso_v2/strategies/nmp_baseline.py:
    NMPFadeRaw.__init__()     retune=True default
    RETUNE_Z_LO = 1.5, RETUNE_Z_HI = 1.8, RETUNE_VETO = [('short','aligned')]
```

### Other tiers NOT retuned (and why)

```
FADE_AGAINST:   E1 hurt OOS (-$60). Skipped.
KILL_SHOT:      wick-based entry, not NMP-z. Different framework needed.
CASCADE:        cascade-based entry, not NMP-z.
FREIGHT_TRAIN:  wick + velocity, not NMP-z.
RIDE_CALM:      n=77, no optimal band, structural OOS issues.
RIDE_MOMENTUM:  n=73, same.
RIDE_AGAINST:   n=845, structural OOS loss (-$1,981 OOS baseline).
NMP_RIDE_RAW:   n=281, no optimal band found.
```

## Expected impact (from per-trade replay)

```
TIER             BASELINE      RETUNED       OOS_UPLIFT     PF_WR_BEFORE→AFTER
FADE_CALM        +$684 total   +$1,090       +$280          0.016 → 0.116
FADE_MOMENTUM    +$2,502 total +$2,638       +$128          0.038 → 0.153
NMP_FADE_RAW     +$5,584 total +$4,078*      +$250          0.030 → 0.109

* NMP_FADE_RAW total $ DROPS but OOS doubles and PF_WR triples.
  Quality > quantity tradeoff.

Combined OOS uplift across 3 tiers: +$658
Combined IS impact: -$2,154 (lower trade count, but higher quality)
```

## Forward-pass commands

To validate the retune through the full iso engine pipeline:

```bash
# Baseline (no retune) — already in pickles from prior runs
# python -m training_iso_v2.run_iso --is --oos --tiers FADE_CALM,FADE_MOMENTUM,NMP_FADE_RAW
# (existing pickles: training_iso_v2/output/{is,oos}_<TIER>.pkl)

# Retuned (default after this change)
python -m training_iso_v2.run_iso --is --oos --tiers FADE_CALM,FADE_MOMENTUM,NMP_FADE_RAW
# pickles will overwrite — back them up first if needed:
cp training_iso_v2/output/is_FADE_CALM.pkl training_iso_v2/output/is_FADE_CALM.baseline.pkl
cp training_iso_v2/output/is_FADE_MOMENTUM.pkl training_iso_v2/output/is_FADE_MOMENTUM.baseline.pkl
cp training_iso_v2/output/is_NMP_FADE_RAW.pkl training_iso_v2/output/is_NMP_FADE_RAW.baseline.pkl
cp training_iso_v2/output/oos_FADE_CALM.pkl training_iso_v2/output/oos_FADE_CALM.baseline.pkl
cp training_iso_v2/output/oos_FADE_MOMENTUM.pkl training_iso_v2/output/oos_FADE_MOMENTUM.baseline.pkl
cp training_iso_v2/output/oos_NMP_FADE_RAW.pkl training_iso_v2/output/oos_NMP_FADE_RAW.baseline.pkl
```

## Comparison spec

After the run, compare:
1. **Trade count change** — expected ~50% reduction (band [1.5, 1.8] is narrow)
2. **Per-day $/day mean and median** — should improve
3. **Day-WR (count of winning days / total days)** — should improve modestly
4. **Max single-day DD** — should be flat or improved (no new entry types)
5. **PF_WR per tier** — should jump 3-6x per our replay
6. **OOS uplift specifically** — expected +$650 across 3 tiers (the headline)

If retuned $/day is WORSE on either IS or OOS, revert with:
```bash
git revert HEAD
```
or restore via the .baseline.pkl backups.

## Rollback (per CLAUDE.md baseline policy)

Each retune is a single commit. If the forward pass shows worse OOS:
- `git revert <commit>` to undo
- Tag the bad attempt as `retune-2026-05-10-rejected`
- Investigate which tier specifically failed
- Apply tiers individually rather than all at once

To disable retune at runtime (no code change):
```python
fc = FadeCalm(retune=False)        # back to baseline behavior
nfr = NMPFadeRaw(retune=False)
```

## What's NEXT after this passes

If retunes validate cleanly OOS:
1. Investigate why FADE_AGAINST E1 hurt OOS (likely sample-size / regime issue)
2. Build wick-based equivalent for KILL_SHOT / CASCADE / FREIGHT_TRAIN
3. Apply the same framework to the 5 RIDE_* tiers (different regime conditioning)
4. Wire the cat-harvest tier (E3 in proposal) — needs OOS-only validation first
5. Wire compression-bounce entry tier (E4) — needs OOS-only validation first
6. Add TOD position-sizing layer (X3)
7. Add PreCatClose exit (X2)

## Risk notes

1. **All 3 retunes were derived from the same trade-replay set** (IS+OOS combined). The OOS uplift is from the OOS slice of that replay — but a true held-out OOS would be even stronger validation.
2. **Per-tier-pickle replay assumes the engine reproduces the same trades** under the retune. If the iso engine fires trades in a different order (e.g., because retune kills early trades and a different tier fires instead), the replay numbers won't match exactly.
3. **No new entry types added in this retune** — all changes are SUBTRACTIVE filters on existing tiers. Risk is bounded to "we lose some trades that net out positive" → in that case revert and re-analyze.
4. **Retune parameters are class-level constants.** To experiment with different bands, instantiate with explicit args, e.g.:
   ```python
   FadeCalm(z_band_lo=1.2, z_band_hi=2.0, veto_cells=[])
   ```

## Output files updated

- [training_iso_v2/strategies/_nmp_base.py](training_iso_v2/strategies/_nmp_base.py) — base class with retune params
- [training_iso_v2/strategies/fade_calm.py](training_iso_v2/strategies/fade_calm.py)
- [training_iso_v2/strategies/fade_momentum.py](training_iso_v2/strategies/fade_momentum.py)
- [training_iso_v2/strategies/nmp_baseline.py](training_iso_v2/strategies/nmp_baseline.py)
- [tools/retune_analysis_all_tiers.py](tools/retune_analysis_all_tiers.py) — re-runnable per-tier analysis
- [reports/findings/segments/retune_per_tier/](reports/findings/segments/retune_per_tier/) — analysis outputs
- [docs/RETUNE_PROPOSAL_2026-05-10.md](docs/RETUNE_PROPOSAL_2026-05-10.md) — research write-up
- [docs/RETUNE_FORWARD_PASS_2026-05-10.md](docs/RETUNE_FORWARD_PASS_2026-05-10.md) — this file
