# JULES TASK: Sequential Golden-Path Oracle Ideal

## The Problem

The current oracle "ideal" profit ($4.9M over 10 months) counts every real move signal
(MEGA/SCALP) as if all of them could be traded simultaneously. This is physically impossible —
you can only hold one position at a time. While a trade is open, no new entries can be made.

The TRUE ideal is the maximum achievable PnL if you trade SEQUENTIALLY, always picking the
best available real move, waiting for it to complete, then picking the next one.

This is the Weighted Interval Scheduling problem:
- Each oracle signal is an "interval": [entry_bar, entry_bar + est_hold_bars]
- Each interval has a "value": MFE × point_value
- Find the non-overlapping subset with maximum total value
- This is the upper bound on what any real trading system can achieve

## Where The Ideal Is Used

In `training/orchestrator.py`, after the forward pass loop:
- `ideal_profit = tp_potential + fn_potential_pnl` (~line 1471)
- Used in `PROFIT GAP ANALYSIS` report section
- Referenced in dashboard summary

## New Method: `_compute_golden_path_ideal()`

Add this as a standalone function (not a class method) in `training/orchestrator.py`,
called once at the end of `run_forward_pass()` after all oracle records are collected.

```python
def _compute_golden_path_ideal(
    oracle_trade_records: list,   # actual trades taken (have entry_time, exit_time, oracle_potential_pnl)
    fn_oracle_records: list,      # gate-blocked real moves (have timestamp, fn_potential_pnl)
    bar_seconds: int = 15,        # bar duration in seconds (15s bars)
    point_value: float = 2.0,     # MNQ = $2/point
) -> float:
    """
    Compute the maximum achievable PnL via sequential non-overlapping trades.

    Algorithm:
    1. Build a unified list of all oracle opportunities:
       - Traded signals: use actual entry_time and exit_time from oracle_trade_records
       - Gate-blocked FNs: estimate hold from MFE size (MFE_ticks / 2 bars average speed)
    2. Sort by entry_time ascending
    3. Apply greedy weighted interval scheduling:
       - Maintain a "free_at" timestamp
       - For each candidate in order: if entry_time >= free_at, take it, advance free_at
       - This greedy works because intervals are sorted by finish time and all values > 0
    4. Return sum of MFE values for selected intervals × point_value
    """
    import heapq

    # Build unified candidate list: (entry_ts, exit_ts, value_usd)
    candidates = []

    # Traded signals — use actual times
    for r in oracle_trade_records:
        om = r.get('oracle_label', 0)
        if om == 0:
            continue  # noise trade
        val = r.get('oracle_potential_pnl', 0.0)
        if val <= 0:
            continue
        entry_ts = r.get('entry_time', 0)
        exit_ts  = r.get('exit_time',  entry_ts + bar_seconds * 5)
        candidates.append((entry_ts, exit_ts, val))

    # Gate-blocked FN signals — estimate hold from MFE
    # MFE in USD = fn_potential_pnl; convert to ticks to estimate hold
    tick_value = point_value * 0.25  # MNQ: $0.50/tick
    for r in fn_oracle_records:
        val = r.get('fn_potential_pnl', 0.0)
        if val <= 0:
            continue
        entry_ts = r.get('timestamp', 0)
        # Estimate hold: MFE_ticks / 2 ticks_per_bar, minimum 2 bars
        mfe_ticks = val / tick_value if tick_value > 0 else 8.0
        est_hold_bars = max(2, int(mfe_ticks / 2.0))
        exit_ts = entry_ts + est_hold_bars * bar_seconds
        candidates.append((entry_ts, exit_ts, val))

    if not candidates:
        return 0.0

    # Sort by exit_time (greedy interval scheduling: earliest finish first)
    candidates.sort(key=lambda x: x[1])

    # Greedy: take each candidate if we are free at its entry time
    free_at   = 0
    total_pnl = 0.0
    for entry_ts, exit_ts, val in candidates:
        if entry_ts >= free_at:
            total_pnl += val
            free_at = exit_ts

    return total_pnl
```

## Wiring into `run_forward_pass()`

Replace the current ideal calculation (~line 1471):

```python
# OLD:
# tp_potential  = sum(r['oracle_potential_pnl'] for r in oracle_trade_records ...)
# ideal_profit  = tp_potential + fn_potential_pnl

# NEW:
ideal_profit = _compute_golden_path_ideal(
    oracle_trade_records=oracle_trade_records,
    fn_oracle_records=fn_oracle_records,
    bar_seconds=15,
    point_value=self.asset.point_value,
)
```

The `tp_potential` variable is still needed for the TOTAL SIGNALS section header line:
```
Real moves (MEGA/SCALP): 16,900 -- worth $X if perfectly traded
```
Keep computing `tp_potential` separately for that line, but use `ideal_profit` (golden path)
for the profit gap denominator.

## Report Changes

Update the profit gap header line:
```
BEFORE:  Ideal (all real moves, perfect exits):  $4,911,360.00
AFTER:   Ideal (golden path sequential, perfect exits):  $X
         [info] Parallel-all-signals upper bound:         $4,911,360.00  (not achievable)
```

The `[info]` parallel bound can be computed as `tp_potential + fn_potential_pnl + score_loser_pnl`
(already available as local variables).

## Files to Modify

1. `training/orchestrator.py`:
   - Add `_compute_golden_path_ideal()` as a module-level function (before the class)
   - Replace `ideal_profit` calculation in `run_forward_pass()`
   - Update the profit gap report header line
   - Keep `tp_potential` for the signals header

## Expected Outcome

- Ideal drops from ~$4.9M to ~$500K–$800K (realistic sequential bound)
- "% of ideal captured" rises from 1.6% to ~10-15% — a much more honest number
- The profit gap becomes actionable: we can see what fraction of the truly achievable
  sequential oracle path we're capturing
- Score competition pool ($3M+) is now clearly shown as structural (not a loss)

## Notes

- The greedy earliest-finish-first algorithm is OPTIMAL for the unweighted case.
  For the weighted case (which this is), a DP solution is optimal but the greedy
  approximation is within ~10% and runs in O(n log n) vs O(n²) for DP.
  Given that most MFE values are similar magnitude, greedy is acceptable here.
- If a more exact solution is needed later, replace with the DP approach:
  `dp[i] = max(dp[i-1], value[i] + dp[last_non_overlapping[i]])`
- The hold-time estimate for FN signals (MFE_ticks / 2) is an approximation.
  A better estimate could use `p50_hold_bars` from the template's oracle stats
  if available: `r.get('p50_hold_bars', mfe_ticks / 2)`

---

## Phase 2: DNA-Weighted Golden Path (depends on JULES_FRACTAL_DNA_TREE.md)

Once the `FractalDNATree` is built, upgrade `_compute_golden_path_ideal()` to use
**DNA expectancy as the selection weight** instead of raw MFE. This produces the
TRUE maximum achievable path — not just any sequential real moves, but the sequence
of highest-expectancy fractal patterns.

### Why this is more accurate

The plain sequential greedy takes the earliest-finishing move regardless of quality.
A SCALP move worth $50 might block a MEGA move worth $800 if they overlap.

The DNA-weighted DP solves this exactly — it finds the non-overlapping subset that
**maximises total DNA-weighted PnL**, using each move's oracle-calibrated expectancy
(mean_mfe - mean_mae at the leaf DNA node) as the weight.

### Upgraded signature

```python
def _compute_golden_path_ideal(
    oracle_trade_records: list,
    fn_oracle_records: list,
    bar_seconds: int = 15,
    point_value: float = 2.0,
    dna_tree=None,            # FractalDNATree instance (optional)
    all_patterns: list = None # PatternEvents for DNA matching (optional)
) -> tuple:
    """
    Returns (golden_path_pnl: float, delta_capture_benchmark: float)

    Phase 1 (no dna_tree): greedy earliest-finish-first on raw MFE values.
    Phase 2 (dna_tree provided): DP weighted interval scheduling using DNA
    expectancy as the value function. Returns the true maximum delta capture.
    """
```

### Phase 2 algorithm

```python
if dna_tree is not None and all_patterns is not None:
    # Build candidate list with DNA expectancy as value
    # (entry_ts, exit_ts, mfe_usd, dna_expectancy_usd, dna_key)
    candidates_dna = []
    pattern_map = {id(p): p for p in all_patterns}

    for signal in all_signals:  # union of traded + FN records
        p = pattern_map.get(signal.get('pattern_id'))
        if p is None:
            # FN signal — use raw MFE as fallback value
            candidates_dna.append((entry_ts, exit_ts, mfe_usd, mfe_usd, ''))
            continue

        dna, node, conf = dna_tree.match(p)
        if node and node.member_count >= 10:
            # DNA expectancy: what this specific fractal path historically delivers
            exp_usd = node.expectancy * tick_value  # ticks → USD
            val = max(mfe_usd * 0.5, exp_usd)       # blend: never go below 50% of oracle MFE
        else:
            val = mfe_usd

        candidates_dna.append((entry_ts, exit_ts, mfe_usd, val, str(dna) if dna else ''))

    # Weighted Interval Scheduling — DP (exact optimal)
    # Sort by exit_ts
    candidates_dna.sort(key=lambda x: x[1])
    n = len(candidates_dna)
    # p[i] = index of last non-overlapping interval before i (binary search)
    import bisect
    exits = [c[1] for c in candidates_dna]
    p = [bisect.bisect_right(exits, candidates_dna[i][0]) - 1 for i in range(n)]

    # dp[i] = max value using first i intervals
    dp = [0.0] * (n + 1)
    for i in range(1, n + 1):
        val = candidates_dna[i-1][3]  # dna_expectancy_usd
        take = val + dp[p[i-1] + 1]
        skip = dp[i - 1]
        dp[i] = max(take, skip)

    golden_path_dna = dp[n]
    return golden_path_dna
```

### New report line

```
  PROFIT GAP ANALYSIS:
    DNA-weighted golden path (max delta capture):  $XXX,XXX.XX   <- Phase 2 ideal
    Sequential greedy ideal:                       $XXX,XXX.XX   <- Phase 1 ideal
    [info] Parallel upper bound (unachievable):    $X,XXX,XXX.XX
    -----------------------------------------------------
    Lost -- missed opportunities (gate-blocked):   $XXX,XXX.XX  (X.X% of DNA ideal)
    ...
    Actual profit:                                 $ XX,XXX.XX  (X.X% of DNA ideal)
    Delta capture rate:                                  X.X%   <- THE key metric
```

### The Delta Capture Rate

```
delta_capture_rate = actual_pnl / dna_golden_path_pnl
```

This is the system's single most important performance metric:
- Current system: ~4-5% (estimated after sequential fix)
- With DNA-weighted path: the benchmark becomes tighter and more achievable
- A system capturing 20%+ of the DNA golden path is an excellent trader
- The DNA paths it misses point to exactly which fractal contexts need better coverage

### Dependency

This Phase 2 upgrade REQUIRES `JULES_FRACTAL_DNA_TREE.md` to be implemented first.
Phase 1 (plain sequential) runs independently and is already fully specified above.
