# ONSET MAMBA run 1 — VERDICT: **KILL** (pre-registered rule, 3/3 heads)

Run 2026-08-04, owner-ordered autonomous execution. 3 epochs, stride 5,
1,080,417 training windows, 257 train days / 112 val days, test SEALED and
never loaded. 815s/epoch as predicted (~14 min).

| head | registered bar | mamba (val, matched) | delta | verdict |
|---|---|---|---|---|
| fakeout_poke | 0.6435 | 0.5851 | −0.0584 | KILL |
| leg_descent | 0.7580 | 0.5519 | −0.2061 | KILL |
| ultra_chop | 0.8201 | 0.6707 | −0.1494 | KILL |

Rule (spec §7, registered BEFORE the run): below bar − 0.02 on any head → do
not deploy, write the null up. **All three heads fail. The GBM wins.**

## The model did learn — it is simply worse

Loss fell every epoch (0.9592 → 0.8579 → 0.8252) and every head beat its own
initialisation (fakeout 0.500 → 0.585, ultra_chop 0.488 → 0.671). This is not
a broken run; it is a losing one. A 0.47M-param SSM reading raw 1s bars did
not, in 3.2M samples, recover what 22 hand-made features already encode.

## Honest caveats, stated rather than used as an excuse

1. **Undertrained.** 3 epochs is small and the curve was still improving. A
   longer run might close some of the gap — but leg_descent is −0.21 away,
   which is not an epochs problem.
2. **Input mismatch.** Amplitude normalisation was added after the audit (era
   was recoverable at AUC 0.9835 without it) and the GBM bar was measured on
   un-normalised features. The Mamba may be paying for a fix the baseline
   never needed. Quantifying that is a separate experiment.
3. Neither caveat is grounds for re-scoring THIS run. Re-registering a bar
   after seeing results is how a program lies to itself.

## Recommendation

**Ship the GBM as the arming model** (rebuild it causally first —
`pipeline/fit_matched.py --causal --temporal`). It is 22 features, trains in
seconds, is interpretable, and beats the sequence model by 0.06–0.21.

Do not chase the Mamba further **unless** the scaling question is asked as a
NEW pre-registered experiment with its own compute budget and its own bar.
The prior is weak: what these events resolve into is barrier geometry
(`bayes_tables/reports/tables_v0.md`), so a better onset detector buys
latency, not edge — and the GBM already supplies the latency at 1000x less
compute.
