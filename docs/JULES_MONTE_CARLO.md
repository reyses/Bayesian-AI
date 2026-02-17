# Jules Task: Brute-Force Monte Carlo Multi-Timeframe Optimizer

## Context

The current pipeline discovers fractal patterns (Phase 2), clusters them into ~113 templates (Phase 2.5), optimizes parameters per template (Phase 3), then runs a forward pass (Phase 4) at 15-second resolution. **Results: 31.5% win rate, -$2,010 PnL, 0 production strategies.**

The problem: 15s decisions are too noisy. Hand-coded gates filter too aggressively (65/113 templates never fire). The system over-engineers entry logic instead of letting brute-force compute find what works.

**New philosophy: Monte Carlo brute force across ALL timeframes. No hand-coded gates. Let the Bayesian brain and statistics determine what's profitable.**

We have a GPU (CUDA), 403 days of tick data across 10 timeframes, and 113+ templates. Bash until something works.

---

## Architecture Overview

Replace the current Phase 3 + Phase 4 with a unified **Monte Carlo Simulation Engine**:

```
Phase 2   → Fractal Discovery (KEEP AS-IS)
Phase 2.5 → Clustering into Templates (KEEP AS-IS)
Phase 3   → NEW: Monte Carlo Multi-TF Sweep
Phase 4   → NEW: ANOVA Factor Analysis
Phase 5   → NEW: Thompson Sampling Refinement
Phase 6   → NEW: Final Validation + Strategy Report
```

---

## Phase 3: Monte Carlo Multi-TF Sweep

### What it does
For each (template, timeframe) pair, run N=2000 random parameter samples across all available data. No gates, no filters — just simulate every trade the template could generate and record outcomes.

### File: `training/monte_carlo_engine.py` (NEW — ~400 lines)

```python
class MonteCarloEngine:
    """
    Brute-force simulator: sweep templates × timeframes × parameters.
    Each (template, timeframe, param_set) is an independent trial.
    """

    def __init__(self, checkpoint_dir, asset, pattern_library, brain, num_workers=None):
        self.checkpoint_dir = checkpoint_dir
        self.asset = asset  # MNQ: tick_size=0.25, point_value=2.0
        self.pattern_library = pattern_library  # {template_id: {centroid, params, member_count}}
        self.brain = brain
        self.num_workers = num_workers or max(1, os.cpu_count() - 2)

        # Results accumulator
        self.results_db = {}  # (template_id, timeframe, param_hash) → ResultRecord

        # Timeframes to sweep (all available in ATLAS)
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        # NOTE: Skip 1s/5s/15s (too noisy), skip 1D/1W (too few bars for statistics)

    def run_sweep(self, data_root='DATA/ATLAS', iterations_per_combo=2000):
        """
        Main loop: iterate over all (template, timeframe) pairs.
        For each pair, run `iterations_per_combo` random parameter samples.
        """
        combos = list(itertools.product(
            self.pattern_library.keys(),  # template_ids
            self.timeframes
        ))

        # Load checkpoint if exists
        completed = self._load_checkpoint()
        remaining = [(tid, tf) for tid, tf in combos if (tid, tf) not in completed]

        print(f"Monte Carlo Sweep: {len(combos)} combos, {len(remaining)} remaining")
        print(f"Iterations per combo: {iterations_per_combo}")

        for batch_start in range(0, len(remaining), self.num_workers):
            batch = remaining[batch_start:batch_start + self.num_workers]

            # Parallel: each worker gets one (template, timeframe) combo
            with multiprocessing.Pool(self.num_workers) as pool:
                jobs = [
                    (tid, tf, iterations_per_combo, data_root,
                     self.pattern_library[tid], self.asset)
                    for tid, tf in batch
                ]
                results = pool.starmap(simulate_template_tf_combo, jobs)

            # Accumulate results
            for (tid, tf), result in zip(batch, results):
                self.results_db[(tid, tf)] = result
                self.brain.batch_update(result.outcomes)  # Feed all outcomes to brain
                completed.add((tid, tf))

            # Checkpoint after each batch
            self._save_checkpoint(completed)

            pct = len(completed) / len(combos) * 100
            print(f"  Progress: {len(completed)}/{len(combos)} ({pct:.1f}%)")
```

### Worker Function (same file):

```python
def simulate_template_tf_combo(template_id, timeframe, n_iterations,
                                data_root, template_info, asset):
    """
    Standalone worker: For one (template, timeframe) pair,
    run n_iterations random parameter samples across ALL months.

    Returns: ComboResult with per-iteration stats
    """
    centroid = template_info['centroid']  # 14D feature vector

    # Load all monthly data for this timeframe
    tf_dir = os.path.join(data_root, timeframe)
    monthly_files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
    all_data = [pd.read_parquet(f) for f in monthly_files]

    # Pre-compute physics states for all months (ONCE)
    # Use QuantumFieldEngine to get states per bar
    engine = QuantumFieldEngine(asset=asset, use_gpu=False)  # CPU in worker
    all_states = []
    for month_data in all_data:
        states = engine.batch_compute_states(month_data)
        all_states.append(states)

    # Extract 14D features from each state for template matching
    all_features = []
    for states in all_states:
        features = np.array([extract_features_from_state(s) for s in states])
        all_features.append(features)

    # Pre-compute template distances (ONCE) — find all bars where this template matches
    match_indices_per_month = []
    for features in all_features:
        if len(features) == 0:
            match_indices_per_month.append([])
            continue
        distances = np.linalg.norm(features - centroid, axis=1)
        matches = np.where(distances < 3.0)[0]  # Same threshold as current
        match_indices_per_month.append(matches)

    # Monte Carlo: N random parameter sets
    param_generator = DOEParameterGenerator()
    iteration_results = []

    for i in range(n_iterations):
        params = param_generator.generate_random_set(i)

        # Simulate across all months with these params
        total_pnl = 0.0
        trades = []

        for month_idx, (month_data, matches) in enumerate(zip(all_data, match_indices_per_month)):
            if len(matches) == 0:
                continue

            month_trades = simulate_month(
                month_data, matches, params, asset,
                template_id=template_id
            )
            trades.extend(month_trades)
            total_pnl += sum(t.pnl for t in month_trades)

        wins = sum(1 for t in trades if t.pnl > 0)
        losses = len(trades) - wins
        win_rate = wins / len(trades) if trades else 0

        iteration_results.append(IterationResult(
            params=params,
            total_pnl=total_pnl,
            num_trades=len(trades),
            win_rate=win_rate,
            wins=wins,
            losses=losses,
            trades=trades  # Keep for brain feeding
        ))

    # Find best iteration
    best = max(iteration_results, key=lambda r: r.total_pnl) if iteration_results else None

    return ComboResult(
        template_id=template_id,
        timeframe=timeframe,
        iterations=iteration_results,
        best_params=best.params if best else None,
        best_pnl=best.total_pnl if best else 0,
        best_win_rate=best.win_rate if best else 0,
        total_iterations=n_iterations
    )
```

### Trade Simulation (same file):

```python
def simulate_month(data, match_indices, params, asset, template_id):
    """
    Simulate trades for matched bars in one month.
    NO GATES — every match is a potential entry.
    Only constraint: can't be in two trades at once.
    """
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    entry_side = 'long'
    high_water = 0.0

    stop_ticks = params.get('stop_loss_ticks', 15)
    tp_ticks = params.get('take_profit_ticks', 40)
    max_hold = params.get('max_hold_bars', 50)  # In bars, not seconds
    trail_tight = params.get('trail_distance_tight', 10)
    trail_wide = params.get('trail_distance_wide', 30)
    cost_points = params.get('trading_cost_points', 0.5)

    prices = data['close'].values
    # Direction from z_score at entry (mean reversion: z>0 = short, z<0 = long)
    # z_scores should be pre-computed in states

    for idx in range(len(prices)):
        if in_position:
            # Check exits
            bars_held = idx - entry_idx
            price = prices[idx]

            if entry_side == 'long':
                pnl_points = price - entry_price
                high_water = max(high_water, pnl_points)

                # Stop loss
                if pnl_points <= -stop_ticks * asset.tick_size:
                    pnl_usd = (pnl_points - cost_points) * asset.point_value
                    trades.append(TradeResult(pnl=pnl_usd, side='long',
                                             bars_held=bars_held, exit_reason='stop',
                                             template_id=template_id))
                    in_position = False
                    continue

                # Take profit
                if pnl_points >= tp_ticks * asset.tick_size:
                    pnl_usd = (pnl_points - cost_points) * asset.point_value
                    trades.append(TradeResult(pnl=pnl_usd, side='long',
                                             bars_held=bars_held, exit_reason='target',
                                             template_id=template_id))
                    in_position = False
                    continue

                # Trailing stop (adaptive)
                if high_water > 0:
                    trail = trail_tight * asset.tick_size if high_water < 2.0 else trail_wide * asset.tick_size
                    if pnl_points < high_water - trail:
                        pnl_usd = (pnl_points - cost_points) * asset.point_value
                        trades.append(TradeResult(pnl=pnl_usd, side='long',
                                                 bars_held=bars_held, exit_reason='trail',
                                                 template_id=template_id))
                        in_position = False
                        continue

                # Max hold
                if bars_held >= max_hold:
                    pnl_usd = (pnl_points - cost_points) * asset.point_value
                    trades.append(TradeResult(pnl=pnl_usd, side='long',
                                             bars_held=bars_held, exit_reason='timeout',
                                             template_id=template_id))
                    in_position = False
                    continue

            else:  # short
                pnl_points = entry_price - price
                high_water = max(high_water, pnl_points)

                if pnl_points <= -stop_ticks * asset.tick_size:
                    pnl_usd = (pnl_points - cost_points) * asset.point_value
                    trades.append(TradeResult(pnl=pnl_usd, side='short',
                                             bars_held=bars_held, exit_reason='stop',
                                             template_id=template_id))
                    in_position = False
                    continue

                if pnl_points >= tp_ticks * asset.tick_size:
                    pnl_usd = (pnl_points - cost_points) * asset.point_value
                    trades.append(TradeResult(pnl=pnl_usd, side='short',
                                             bars_held=bars_held, exit_reason='target',
                                             template_id=template_id))
                    in_position = False
                    continue

                if high_water > 0:
                    trail = trail_tight * asset.tick_size if high_water < 2.0 else trail_wide * asset.tick_size
                    if pnl_points < high_water - trail:
                        pnl_usd = (pnl_points - cost_points) * asset.point_value
                        trades.append(TradeResult(pnl=pnl_usd, side='short',
                                                 bars_held=bars_held, exit_reason='trail',
                                                 template_id=template_id))
                        in_position = False
                        continue

                if bars_held >= max_hold:
                    pnl_usd = (pnl_points - cost_points) * asset.point_value
                    trades.append(TradeResult(pnl=pnl_usd, side='short',
                                             bars_held=bars_held, exit_reason='timeout',
                                             template_id=template_id))
                    in_position = False
                    continue

        else:
            # Check for new entry — NO GATES, just template match
            if idx in match_indices:
                entry_price = prices[idx]
                entry_idx = idx
                high_water = 0.0
                # Direction: use z_score from the state (mean reversion)
                # z > 0 → overbought → short; z < 0 → oversold → long
                entry_side = 'short' if z_scores[idx] > 0 else 'long'
                in_position = True

    return trades
```

### Parameter Sampling

Add a method to `DOEParameterGenerator`:

```python
def generate_random_set(self, iteration_id: int) -> Dict[str, Any]:
    """Pure random sampling from parameter ranges. No DOE structure."""
    params = {}
    for name, (lo, hi, dtype) in self._define_parameter_ranges().items():
        if dtype == 'int':
            params[name] = np.random.randint(lo, hi + 1)
        elif dtype == 'float':
            params[name] = np.random.uniform(lo, hi)
    # Add max_hold_bars (in bars, not seconds — depends on timeframe)
    params['max_hold_bars'] = np.random.randint(10, 200)
    return params
```

### Checkpoint Structure

```
checkpoints/
├── mc_sweep_state.pkl          # {completed: set of (tid, tf), results_db: dict}
├── mc_combo_results/           # Individual combo results for memory efficiency
│   ├── T144_1m.pkl
│   ├── T144_5m.pkl
│   └── ...
├── mc_brain.pkl                # Brain state after all Monte Carlo learning
└── mc_summary.json             # Human-readable: best combos, stats
```

Save after each batch of (template, timeframe) combos completes. On resume, skip completed combos.

---

## Phase 4: ANOVA Factor Analysis

### What it does
After the Monte Carlo sweep, analyze which factors actually drive profitability. This tells us where to focus refinement compute.

### File: `training/anova_analyzer.py` (NEW — ~150 lines)

```python
class ANOVAAnalyzer:
    """
    Statistical factor analysis on Monte Carlo results.
    Identifies which dimensions matter for profitability.
    """

    def analyze(self, results_db):
        """
        Run multi-factor ANOVA on results.

        Factors tested:
        - Timeframe (categorical: 1m, 5m, 15m, 1h, 4h)
        - Template cluster (categorical: template_id)
        - Stop width bucket (3 levels: tight/medium/wide)
        - Take profit bucket (3 levels: tight/medium/wide)
        - Hold time bucket (3 levels: short/medium/long)
        - Trail style bucket (2 levels: tight/wide)

        Response variable: PnL per trade (or Sharpe ratio)
        """
        # Build DataFrame from all iteration results
        rows = []
        for (tid, tf), combo_result in results_db.items():
            for iteration in combo_result.iterations:
                if iteration.num_trades > 0:
                    rows.append({
                        'template_id': tid,
                        'timeframe': tf,
                        'stop_bucket': self._bucket(iteration.params['stop_loss_ticks'], [12, 18]),
                        'tp_bucket': self._bucket(iteration.params['take_profit_ticks'], [40, 50]),
                        'hold_bucket': self._bucket(iteration.params['max_hold_bars'], [30, 100]),
                        'pnl_per_trade': iteration.total_pnl / iteration.num_trades,
                        'win_rate': iteration.win_rate,
                        'sharpe': self._compute_sharpe(iteration.trades),
                        'total_pnl': iteration.total_pnl,
                        'num_trades': iteration.num_trades
                    })

        df = pd.DataFrame(rows)

        # One-way ANOVA per factor
        from scipy.stats import f_oneway

        results = {}
        for factor in ['timeframe', 'template_id', 'stop_bucket', 'tp_bucket', 'hold_bucket']:
            groups = [group['pnl_per_trade'].values
                      for _, group in df.groupby(factor) if len(group) >= 10]
            if len(groups) >= 2:
                f_stat, p_value = f_oneway(*groups)
                results[factor] = {'f_stat': f_stat, 'p_value': p_value}

        # Report top factors (lowest p-value = most significant)
        print("\nANOVA FACTOR SIGNIFICANCE:")
        print("-" * 50)
        for factor, stats in sorted(results.items(), key=lambda x: x[1]['p_value']):
            sig = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else "ns"
            print(f"  {factor:20s}  F={stats['f_stat']:8.2f}  p={stats['p_value']:.4f}  {sig}")

        # Best (template, timeframe) combos
        combo_stats = df.groupby(['template_id', 'timeframe']).agg({
            'total_pnl': 'mean',
            'win_rate': 'mean',
            'num_trades': 'mean',
            'sharpe': 'mean'
        }).reset_index()

        # Filter: need meaningful trade count
        viable = combo_stats[combo_stats['num_trades'] >= 5]
        top_combos = viable.nlargest(20, 'sharpe')

        print("\nTOP 20 (TEMPLATE × TIMEFRAME) COMBOS BY SHARPE:")
        print(top_combos.to_string(index=False))

        return results, top_combos
```

Save report to `checkpoints/anova_report.txt`.

---

## Phase 5: Thompson Sampling Refinement

### What it does
Take the top N combos from ANOVA and throw 10x more compute at them using Thompson sampling (Bayesian bandit). The brain's Beta posteriors guide which parameter regions to explore more.

### File: `training/thompson_refiner.py` (NEW — ~200 lines)

```python
class ThompsonRefiner:
    """
    Bayesian bandit refinement: concentrate compute on promising combos.
    Uses Thompson sampling from Beta posteriors to allocate iterations.
    """

    def __init__(self, brain, asset, top_combos, checkpoint_dir):
        self.brain = brain
        self.asset = asset
        self.top_combos = top_combos  # List of (template_id, timeframe, base_params)
        self.checkpoint_dir = checkpoint_dir
        self.iteration_budget = 20000  # Total iterations to distribute

    def refine(self, data_root='DATA/ATLAS'):
        """
        Allocate iterations to combos proportional to Thompson samples.
        Each round:
        1. Sample from each combo's Beta posterior
        2. Allocate more iterations to higher samples
        3. Run simulations with tighter parameter mutations around best known
        4. Update posteriors
        """
        rounds = 50  # Number of allocation rounds
        iters_per_round = self.iteration_budget // rounds

        for round_num in range(rounds):
            # Thompson sampling: draw from each combo's posterior
            thompson_scores = []
            for tid, tf, _ in self.top_combos:
                key = f"{tid}_{tf}"
                stats = self.brain.table.get(key, {'wins': 0, 'losses': 0, 'total': 0})
                # Beta posterior: Beta(wins + 1, losses + 1)
                alpha = stats['wins'] + 1
                beta_param = stats['losses'] + 1
                sample = np.random.beta(alpha, beta_param)
                thompson_scores.append(sample)

            # Allocate iterations proportional to scores
            total_score = sum(thompson_scores)
            allocations = [max(10, int(s / total_score * iters_per_round))
                          for s in thompson_scores]

            # Run simulations for each combo with allocated iterations
            for (tid, tf, base_params), n_iters in zip(self.top_combos, allocations):
                results = simulate_template_tf_combo(
                    tid, tf, n_iters, data_root,
                    self.pattern_library[tid], self.asset,
                    mutation_base=base_params,  # Mutate around best known params
                    mutation_scale=0.1  # Tight mutations
                )

                # Update brain with results
                for iter_result in results.iterations:
                    for trade in iter_result.trades:
                        self.brain.update(TradeOutcome(
                            template_id=f"{tid}_{tf}",
                            result='WIN' if trade.pnl > 0 else 'LOSS',
                            pnl=trade.pnl,
                            state=f"{tid}_{tf}"
                        ))

            # Report progress
            print(f"  Round {round_num+1}/{rounds} — "
                  f"Best combo: {self._get_best_combo()}")

            self._save_checkpoint(round_num)

        return self._get_final_rankings()
```

### Mutation around best params:

When we have a known-good parameter set, generate nearby variants:

```python
def mutate_params(base_params, scale=0.1):
    """Generate parameter set mutated from base. Scale=0.1 means ±10% range."""
    params = {}
    ranges = DOEParameterGenerator()._define_parameter_ranges()
    for name, value in base_params.items():
        if name in ranges:
            lo, hi, dtype = ranges[name]
            spread = (hi - lo) * scale
            if dtype == 'int':
                new_val = int(value + np.random.uniform(-spread, spread))
                params[name] = np.clip(new_val, lo, hi)
            elif dtype == 'float':
                new_val = value + np.random.uniform(-spread, spread)
                params[name] = np.clip(new_val, lo, hi)
        else:
            params[name] = value
    return params
```

---

## Phase 6: Final Validation + Strategy Report

### What it does
Walk-forward validation on the top strategies. Train on months 1-10, validate on months 11-14. Generate production playbook.

### File: Modify existing `run_strategy_selection()` in `training/orchestrator.py`

```python
def run_final_validation(self, top_strategies):
    """
    Walk-forward: train on first 70% of months, validate on last 30%.
    Only strategies that are profitable in BOTH periods survive.
    """
    monthly_files = sorted(glob.glob('DATA/ATLAS/{tf}/*.parquet'))
    split_idx = int(len(monthly_files) * 0.7)
    train_months = monthly_files[:split_idx]
    val_months = monthly_files[split_idx:]

    validated = []
    for strategy in top_strategies:
        tid, tf, params = strategy

        # In-sample performance (already computed, use from Phase 5)
        is_pnl = strategy.train_pnl

        # Out-of-sample performance
        oos_result = simulate_template_tf_combo(
            tid, tf, 1, 'DATA/ATLAS',  # Single iteration with fixed params
            self.pattern_library[tid], self.asset,
            fixed_params=params,
            month_filter=val_months
        )

        oos_pnl = oos_result.best_pnl
        oos_win_rate = oos_result.best_win_rate
        oos_trades = oos_result.iterations[0].num_trades if oos_result.iterations else 0

        # Tier classification
        if (oos_trades >= 20 and oos_win_rate > 0.45 and oos_pnl > 0 and
            self._compute_sharpe(oos_result) > 0.3):
            tier = 1  # PRODUCTION
        elif oos_trades >= 10 and oos_pnl > 0:
            tier = 2  # CANDIDATE
        elif oos_trades >= 5:
            tier = 3  # UNPROVEN
        else:
            tier = 4  # TOXIC

        validated.append({
            'template_id': tid,
            'timeframe': tf,
            'params': params,
            'is_pnl': is_pnl,
            'oos_pnl': oos_pnl,
            'oos_win_rate': oos_win_rate,
            'oos_trades': oos_trades,
            'tier': tier
        })

    # Report
    self._print_validation_report(validated)

    # Save Tier 1 strategies to production playbook
    tier1 = [v for v in validated if v['tier'] == 1]
    with open(os.path.join(self.checkpoint_dir, 'production_playbook.pkl'), 'wb') as f:
        pickle.dump(tier1, f)

    return validated
```

---

## Wiring into Orchestrator

### Modify `training/orchestrator.py` main():

```python
def main():
    orch = Orchestrator(config)

    # Phase 2 + 2.5 + old Phase 3 (discovery + clustering + initial params)
    orch.train()

    # NEW Phase 3: Monte Carlo Sweep
    mc = MonteCarloEngine(
        checkpoint_dir=orch.checkpoint_dir,
        asset=orch.asset,
        pattern_library=orch.pattern_library,
        brain=orch.brain
    )
    mc.run_sweep(iterations_per_combo=2000)

    # NEW Phase 4: ANOVA
    anova = ANOVAAnalyzer()
    factor_results, top_combos = anova.analyze(mc.results_db)

    # NEW Phase 5: Thompson Refinement
    refiner = ThompsonRefiner(
        brain=orch.brain,
        asset=orch.asset,
        top_combos=top_combos,
        checkpoint_dir=orch.checkpoint_dir
    )
    refined_strategies = refiner.refine()

    # NEW Phase 6: Walk-Forward Validation
    orch.run_final_validation(refined_strategies)
```

### CLI flags:

```python
parser.add_argument('--mc-iters', type=int, default=2000,
                    help='Monte Carlo iterations per (template, timeframe) combo')
parser.add_argument('--mc-only', action='store_true',
                    help='Skip discovery, just run Monte Carlo from existing templates')
parser.add_argument('--anova-only', action='store_true',
                    help='Skip MC sweep, just run ANOVA on existing results')
parser.add_argument('--refine-only', action='store_true',
                    help='Skip MC+ANOVA, just run Thompson refinement')
```

---

## Key Implementation Details

### 1. Template Matching at Different Timeframes

The current 14D feature vector includes fields computed from the physics engine. When running at 5m or 1h instead of 15s, the physics (z_score, momentum, sigma) will have different magnitudes.

**Solution**: Normalize features per-timeframe before distance computation. The scaler fitted during clustering (Phase 2.5) was on 15s data. Either:
- (a) Re-fit scaler per timeframe, OR
- (b) Use rank-based matching (percentile of z_score within that timeframe's distribution)

**Recommended**: Option (a) — fit a `StandardScaler` per timeframe on Day 1 data, save alongside pattern_library.

### 2. Direction Logic

Direction comes from the **template centroid's z_score sign**, NOT the current bar's z_score. The template "knows" which side it was discovered on: `centroid[0] > 0 → short, centroid[0] < 0 → long`. This prevents entering long on a short-side template (or vice versa) when the current bar's z is near zero but other features match.

### 3. max_hold_bars vs max_hold_seconds

At 15s, 60 bars = 15 minutes. At 1h, 60 bars = 60 hours. Use **bars** as the unit, but adjust the range per timeframe:

```python
HOLD_RANGES = {
    '1m': (10, 120),    # 10 min to 2 hours
    '5m': (5, 60),      # 25 min to 5 hours
    '15m': (4, 30),     # 1 hour to 7.5 hours
    '1h': (2, 12),      # 2 hours to 12 hours
    '4h': (1, 6),       # 4 hours to 24 hours
}
```

### 4. Cost Model

Per-trade round-trip cost (commission + slippage):
- `cost_points = 0.5` (default) = $1.00 per trade on MNQ
- Applied as deduction from every trade's PnL
- The DOE range is (0.25, 1.0) — Monte Carlo should sweep this too

### 5. Brain Key Format

For the Monte Carlo brain, key outcomes by `f"{template_id}_{timeframe}"`:
```python
brain_key = f"{template_id}_{timeframe}"
brain.update(TradeOutcome(state=brain_key, template_id=template_id, ...))
```

This lets the brain track per-(template, timeframe) performance separately.

### 6. Memory Management

With 113 templates × 5 timeframes × 2000 iterations = 1,130,000 simulation runs, each producing ~50 trades average = ~56 million trade records.

**Don't store all trades in memory.** Store only summary stats per iteration:
```python
@dataclass
class IterationSummary:
    params_hash: str
    total_pnl: float
    num_trades: int
    win_rate: float
    sharpe: float
    max_drawdown: float
    # Don't store individual trades — feed to brain inline, then discard
```

### 7. Parallelism

Each (template, timeframe) combo is 100% independent. Use `multiprocessing.Pool(num_workers)` where `num_workers = cpu_count - 2`.

Within each worker, the simulation is sequential (bar-by-bar). The GPU is used only for the initial `batch_compute_states()` call per month — this should be done once and cached.

**Pre-compute optimization**: Before the Monte Carlo loop, pre-compute all states for all (timeframe, month) pairs and cache them as `.npy` files:

```
checkpoints/mc_states/
├── 1m_2025_01.npy
├── 1m_2025_02.npy
├── ...
├── 4h_2026_02.npy
```

This way workers just load the pre-computed states instead of re-running the physics engine.

---

## Expected Compute Time

- Pre-compute states: ~5 min (GPU, all TFs × all months)
- Monte Carlo sweep: 113 templates × 5 TFs × 2000 iters ≈ **~2-4 hours** (parallel, depends on CPU count)
- ANOVA: < 1 minute (just statistics on stored results)
- Thompson refinement: ~1-2 hours (focused on top 20 combos with 10x iterations)
- Walk-forward validation: ~10 minutes

**Total: ~4-7 hours for complete pipeline.**

---

## Files to Create
1. `training/monte_carlo_engine.py` (~400 lines) — MC sweep + simulate_month + worker
2. `training/anova_analyzer.py` (~150 lines) — ANOVA factor analysis
3. `training/thompson_refiner.py` (~200 lines) — Bayesian bandit refinement

## Files to Modify
1. `training/orchestrator.py` — Wire MC → ANOVA → Thompson → Validation into main()
2. `training/doe_parameter_generator.py` — Add `generate_random_set()` method
3. `core/bayesian_brain.py` — Add `batch_update()` for bulk outcome recording

## Files to NOT Touch
- `core/quantum_field_engine.py` — physics engine is good
- `core/cuda_physics.py` — CUDA kernels are good
- `training/fractal_discovery_agent.py` — discovery is good
- `training/fractal_clustering.py` — clustering is good

---

## Verification

1. Run `python training/orchestrator.py --fresh` → should complete all 6 phases
2. Check `checkpoints/anova_report.txt` → should show which factors are significant
3. Check `checkpoints/production_playbook.pkl` → should contain Tier 1 strategies (if any)
4. Interrupt mid-Phase 3, restart → should resume from checkpoint
5. `--mc-only` flag → should skip discovery and go straight to Monte Carlo
