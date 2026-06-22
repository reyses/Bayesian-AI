---
name: report-distributions-and-mode
description: "User reasons in DISTRIBUTIONS and MODE, not averages — lead every metric with the distribution + mode; the mean is secondary and usually misleading for skewed trade PnL."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a1bfcbce-37ff-4c61-931f-a5324a849c31
---

Moisés does NOT trust or relate to AVERAGES (means). Verbatim (2026-06-16): "average does not
make sense to me, I prefer distributions and mode." Applies to $/trade, $/day, MFE, giveback,
and any per-unit metric.

**Why:** trade/day PnL is heavy-tailed and asymmetric — a few big winners/losers dominate the
mean, so the mean is NOT the typical outcome. The MODE (most common bin) plus the distribution
shape is how he reasons about "what usually happens." (This also underlies his "I don't trust
the day statistics" feedback — same distrust of aggregated averages.)

**How to apply — every results table/report:**
1. LEAD with the MODE (histogram bin; ~$2 bins for $/trade, ~$25 for $/day) and the DISTRIBUTION
   shape — ideally an actual histogram plot, not a single number.
2. The MEAN is SECONDARY: show it only with its 95% CI + an explicit significance call, and
   NEVER as the headline. Bucket, don't average.
3. When comparing A vs B, compare their DISTRIBUTIONS (overlaid histograms / per-bucket), not
   just mean deltas.

**PREFERRED FORMAT (verbatim, 2026-06-21: "one of the best ways to present information to me"):**
an inline ASCII bar histogram, MODE called out on top, each bin = a unicode bar scaled to the max
bin + the % of items in that bin; collapse the thin tail into one line. Template:
```
MODE: 30–40s  (55% of all regimes)
 30– 40s | ███████████████████████████ 55.0%
 40– 50s | ██████████ 20.4%
 50– 60s | █████ 11.3%
 ...
100–140s |      ~1.5%  (thin tail, max 140s)
```
Use this shape for ANY distribution (regime lengths, $/trade, $/day, tiers, MFE, giveback…). It's
fast to read in chat, mode-first, shows the whole shape. A saved PNG histogram is a fine companion
but lead with this ASCII version in the message itself.

Reinforces (and strengthens) the canonical metric rule in MEMORY §2 ("mode AND mean + CI"):
the user wants **mode/distribution FIRST, mean de-emphasized**. See [[USER_PERSONA_AND_PROTOCOL]].
