# cubic_wick_sensitivity

Should the pocket-dojo cubic (`research/dojo_forge/tools/cubic_regression.py`,
close-only, 8-bar/1m) be made sensitive to wick/body structure? Owner proposal
2026-07-30 ("this will have the hidden proportion of internal structure").
Comparison research only — **no live tool was modified**.

**Verdict (see `reports/comparison_findings.md`):** don't change the cubic's input
(direction a — flags the captured tells no earlier, k=1.0 is 5 bars later at the
top and +9% flippier; the naive HL2 blend is sign-opposed to the owner's read).
**Adopt** a separate 8-bar wick-bias mean read alongside the untouched price cubic
(direction b) — it reproduced both captured tells at the right bar (bias-vs-slope
divergence at the 2025_08_24 top tick; +0.39 bullish print on the 2025_06_05
bar-880 fake dip) and is 2.3x more sign-stable than the price slope itself.

## Layout

- `tools/wick_series.py` — derived series library: rejection-school price
  `close + k*(lw-uw)`, excursion-school HL2 blend, `wick_bias=(lw-uw)/range`,
  `body_frac`. Feeds the *unchanged* cubic machinery (imported from
  `cubic_regression.py`, not forked). Self-test: run the file.
- `tools/compare_on_tell_bars.py` — the comparison: two tell-bar presets
  (`top` = 2025_08_24 bar 107 exhaustion top, `dip` = 2025_06_05 bar 880 fake
  dip; provenance-guarded row indices) + a 6-day slope-whipsaw scan.
  Pre-registered flag/stability rules in the script header.
- `reports/comparison_findings.md` — findings + recommendation (evidence tables).
- `reports/comparison_results.json` — full machine-readable numbers.
- `reports/assets/tell_*.png` — candles + cubic variants + slopes + wick-bias
  panel around each tell bar.

## Run (from repo root, CPU-light, ~10s)

```
python research/cubic_wick_sensitivity/tools/compare_on_tell_bars.py            # both presets + stability
python research/cubic_wick_sensitivity/tools/compare_on_tell_bars.py --preset top
```

Data: `DATA/ATLAS/1m/<day>.parquet` (repo root; bar index = row index, matching
`pocket_dojo._bars()`). Context for the tell bars:
`research/dojo_forge/reports/human_dojo/OWNER_PROCESS.md` and
`docs/daily/2026-07-30.md`.
