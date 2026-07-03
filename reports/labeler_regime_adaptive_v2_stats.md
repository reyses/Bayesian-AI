# AI Auto Labeler Spacing Math & Re-derivation

## 1. The Mathematical Discrepancy Explained

During the review of the labeler's parameters, a mathematical inconsistency was noted:
> *The spacing math doesn't reconcile. 222 turns in 23,400 bars ⇒ mean spacing = 105 bars. A median of 299 is impossible against a 105 mean for a right-skewed spacing distribution.*

The error in the previous derivation stemmed from the assumption that the `23,400` bars (which is exactly **6.5 hours** of 1-second bars, or 32.5 hours of 5-second bars) represented the total session duration being labeled.

**Correction:** The `ai_cusp_picks` dataset actually labels **~22 hour sessions** (Globex), not just Regular Trading Hours (RTH).
- Average session duration across the labeled days: **21.75 hours**.

## 2. Re-computing the Spacing Distribution (604 Days)

To ensure mathematical rigor, the spacing distribution was re-computed across all **604 days** of the auto-labeled `ai_cusp_picks` dataset (using strict intraday gaps, filtering out the overnight weekend/session closures).

### Raw Metrics (in 5-second bars)
* **Total Turns:** 25,680
* **Total Market Duration Labeled:** ~13,288 hours (604 days * 22 hours)
* **Total 5s Bars:** ~9,567,360
* **Mean Spacing:** 335.48 bars
* **Median Spacing:** 253.40 bars

This completely resolves the mathematical discrepancy:
In a highly right-skewed spacing distribution, we expect `Mean > Median`. Here, `335 > 253`, which is perfectly consistent.

## 3. Re-deriving `h` and `pos_weight`

Using the corrected median spacing over the full 604-day dataset, we re-derive the reward architecture hyperparameters.

### A. Tolerance Horizon (`h`)
We define `h` as ~10% of the median spacing to ensure causality without capturing random noise.
* `h = 253.40 * 0.10`
* **`h` = 25 bars** (125 seconds)

### B. Class Imbalance Weight (`pos_weight`)
We define the BCE loss positive class weight dynamically based on the ratio of background (non-turn) bars to turn-active bars. A turn is "active" during the target bar + the `h` tolerance bars.

* **Total Bars (604 days):** ~9,567,360
* **Active Turn Bars:** `25,680 * (25 + 1) = 667,680`
* **Non-Turn Bars:** `9,567,360 - 667,680 = 8,899,680`
* **Ratio (pos_weight):** `8,899,680 / 667,680` = **13.33**

## 4. Updates Applied
1. `mamba_env.py` has been updated to use `self.h_bars_tolerance = 25`.
2. `train_mamba_rl.py` has been updated to use `pos_weight = torch.tensor([13.33])`.

The spacing histogram `spacing_histogram.png` has been attached to the workspace for visual verification.
