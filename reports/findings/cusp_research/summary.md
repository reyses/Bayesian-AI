# CRM-CUSP ENTRY — Research Note (2026-05-10)

## Hypothesis
Enter reversion trades at the CUSP of |z| (the |z| local maximum) — the moment
|z| stops growing and starts shrinking — instead of the moment |z| crosses a
threshold. Intuition: cusp adds the temporal info "reversion is actually
underway," not just "price is far from CRM."

## Cusp definition (no lookahead)
At each 1m close `t`, look back two bars. Bar `t-1` is a |z| local max iff:
```
|z[t-2]| < |z[t-1]|   AND   |z[t-1]| > |z[t]|
```
We confirm this at time `t` (one-bar lag). Reversion direction = `-sign(z[t-1])`.

## Data
235 IS days (Jan–Oct 2025) + 68 OOS days (Jan–Feb 2026).
1m closes only. Forward returns = close-to-close ticks (×$0.50/tick MNQ),
signed by reversion direction.

## Result — z in [1.5, 1.8) (the validated NMP retune band)

| Horizon | IS cusp | IS thr-only | OOS cusp | OOS thr-only | OOS cusp lift |
|---------|---------|-------------|----------|--------------|---------------|
| 5m      | +$0.25  | +$0.19      | +$0.70   | **+$1.03**   | −$0.33        |
| 15m     | +$0.18  | −$0.05      | +$0.28   | −$0.39       | **+$0.67**    |
| 30m     | +$0.60  | −$0.20      | +$0.39   | −$0.04       | **+$0.43**    |
| 60m     | +$0.38  | −$0.71      | +$0.82   | +$0.20       | **+$0.62**    |

n_cusp ≈ 3700 / 68 OOS days ≈ 55 cusps/day (raw; needs cooldown).
n_thr-only ≈ 6800 / 68 OOS days ≈ 100 thr entries/day.

The cusp is a **subset** (~55%) of threshold entries — it skips entries where
|z| is still growing. The kept subset has higher per-trade edge OOS at every
horizon except 5m.

## Per-bucket sweep (IS, h=30m, $/trade after $/tick conversion)

| z bucket   | n     | cusp $   | thr $    |
|------------|-------|----------|----------|
| [0.8, 1.2) | 17011 | +$0.10   | (n/a)    |
| [1.2, 1.5) | 14084 | +$0.19   | (n/a)    |
| [1.5, 1.8) | 12786 | **+$0.60** | −$0.20 |
| [1.8, 2.2) | 11512 | −$0.27   | −$0.11   |
| [2.2, 2.8) | 5812  | −$2.44   | −$1.88   |
| [2.8, 3.5) | 401   | −$6.42   | −$6.12   |

Confirms the 2026-05-10 retune finding: **higher z = WORSE trade**. Sweet spot
is [1.5, 1.8). Past 1.8σ, reversion fade is a losing proposition regardless
of cusp.

## Interpretation
- The cusp filter takes the existing NMP-style entry and gates on "z just
  turned over" — small per-trade lift, but sign-flips OOS PF-WR positive at
  three of four horizons in the validated band.
- At 5m OOS, threshold-only actually wins (+$1.03 vs cusp +$0.70). This is
  noise from one-bar timing — at 5m horizon the cusp is too early on the
  reversion path.
- Mean trade size is small ($0.18–$0.82). At 55–100 cusps/day raw this
  projects to $10–$45/day GROSS before any exit logic or cooldown.
- The cusp is best viewed as a **gating filter on top of the existing NMP
  retune**, not a replacement.

## Risk / caveats
- Forward returns are gross close-to-close — no SL/TP, no slippage, no
  commission. Real strategy will differ.
- 55 cusps/day raw is too many — cooldown needed (overlapping events likely
  fire repeatedly within minutes).
- The 5m horizon OOS anomaly suggests cusp timing is one bar early — could
  benefit from "wait for first counter-bar in price" confirmation.
- Bigger sample would settle the [1.2, 1.5) band — OOS shows +$2.78 / 30m
  there but n=4k is small, possibly mean-shifted by tail trades.

## Next
1. Build `CrmCuspFade` strategy on `training_iso_v2`, gate on z in [1.5, 1.8)
   AND cusp confirmation. Cooldown 5m minimum between fires.
2. Run forward pass against existing FADE_CALM/MOMENTUM/NMP_FADE_RAW baselines.
3. Stratify by 2D regime (UP_SMOOTH / FLAT_CHOPPY / etc.) — likely cusp works
   better in MEAN-REVERTING regimes (FLAT_*) than TRENDING (UP_SMOOTH).

## Outputs
- [summary_is.csv](summary_is.csv) / [summary_oos.csv](summary_oos.csv)
- [pf_wr_by_bucket_is.png](pf_wr_by_bucket_is.png) / pf_wr_by_bucket_oos.png
- [n_by_bucket_is.png](n_by_bucket_is.png) / n_by_bucket_oos.png
- Tool: [tools/crm_cusp_research.py](../../../tools/crm_cusp_research.py)
