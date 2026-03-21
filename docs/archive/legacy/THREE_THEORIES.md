# The Three Theories — Theoretical Physics Foundation
> Original framework by Moises. Articulated by Gemini. Reference for Claude.
> This is the WHY behind every module in the system.

---

## Theory 1: The Nightmare Protocol (The Gravitational Field)

**Concept**: The market is a localized gravitational well. Price is a particle tethered
to a moving Center of Mass (the Linear Regression Mean, mu). The Standard Error bands
(sigma) are not just statistical deviations; they are **Roche Limits** — the exact boundary
where the structural integrity of the localized trend gets torn apart by mean-reverting
tidal forces.

**Physics (Ornstein-Uhlenbeck Process)**: Price naturally decays back to the mean. The
further it stretches (High Z-Score), the exponentially stronger the restoring force becomes.

**System Rule**: If price hits a 3-Sigma limit on a timeframe, its localized energy is
spent. It has hit the "Concrete Wall."

**Code mapping**:
- `core/statistical_field_engine.py` — computes z-score, regression sigma, forces
- `core/exits/fractal_exhaust.py` — Death Hook (ADX rollover at Roche limit)
- `core/exits/band_exit.py` — Band urgent (multi-TF support/resistance broken)
- `trading_config.nightmare_z = 3.0` — the concrete wall threshold
- `gate0_r4_nightmare` — blocks entries into the nightmare field
- `tunnel_probability` — P(mean reversion) via O-U process, STILL COMPUTED but unused in scoring

**What's connected**: Death hook, band exit, nightmare gate
**What's disconnected**: tunnel_probability is computed every bar but never used in entry scoring

---

## Theory 2: The Three-Body Problem (Fractal Chaos)

**Concept**: In astrophysics, predicting the exact trajectory of three interacting
gravitational bodies is mathematically impossible (chaos). In trading, these three bodies
are the Macro (1-Hour), Meso (5-Minute), and Micro (15-Second) timeframes.

When forces conflict (e.g., Micro trending UP, Macro trending DOWN), the price path is
chaotic and untradable. We solve it probabilistically, not deterministically.

**Physics (Bayesian Resolution)**: We cannot solve the Three-Body Problem deterministically.
We solve it probabilistically. We use clustering and the BayesianBrain to map the exact
state of all three bodies.

**System Rule**: Refuse to trade in chaotic orbits. Only execute when the Three Bodies
are mathematically aligned.

**Code mapping**:
- `core/timeframe_belief_network.py` — 11 TF workers, each a "body"
- `core/execution_engine.py` — direction cascade (8-voter system)
- `core/fractal_clustering.py` — pattern templates are states of aligned bodies
- `get_belief()` — weighted geometric mean of P(direction) across all TFs
- `worker_agreement` — tracked in reports, shows alignment per TF
- Hurst exponent — H > 0.5 = trending (aligned), H < 0.5 = mean-reverting (chaotic)

**What's connected**: Worker agreement, conviction gate, DMI alignment
**What's disconnected**: The check is binary (conviction > 0.48) instead of probabilistic.
Should output P(alignment) as a continuous score, not pass/fail.

---

## Theory 3: The Resonance Cascade (The Alpha Event)

**Concept**: A Resonance Cascade occurs when the frequencies of the Three Bodies synchronize
perfectly. Instead of the Micro particle bouncing off the Meso Roche Limit (mean reversion),
its momentum resonates with the Macro gravity. The limit shatters, causing a violent,
exponential expansion in price. This is the birth of a massive trend day.

**Physics (Lyapunov Exponent > 0)**: The system transitions from stable (mean-reverting)
to chaotic (directional expansion). The Hurst Exponent surges above 0.6, and Volatility
(Standard Error) expands directionally rather than symmetrically.

**System Rule**: When a Resonance Cascade is detected, standard Take Profits are
mathematical suicide. You do not step in front of a cascade; you ride the shockwave
until the physics break down.

**Code mapping**:
- `core/exits/tidal_wave.py` — SE expansion detection (partial implementation)
- `core/exits/survival_stop.py` — time-survival trailing (rides cascades)
- `core/market_state.py` — has coherence, entropy, wave function probabilities
- Funnel tree research — detected the pre-cascade signature:
  tight price_range + high sigma = energy coiling (85% accuracy)

**What's connected**: Tidal wave exit, survival stop
**What's disconnected**: No `is_resonance_cascade` flag. No cascade mode that disables TP.
The pre-cascade detection (funnel tree) is research-only, not integrated.

**Implementation directive**: When all three TFs show ADX > 30, Hurst > 0.55, and aligned
DI vectors → `is_resonance_cascade = True` → suppress TP → trail via survival_stop only
until Macro Hurst decays.

---

## The Wave Function (Already Computed, Not Used)

The original quantum_field_engine computed these EVERY BAR:

```python
# Boltzmann-weighted probabilities for three attractors
E0 = -(z^2) / 2.0           # energy at center (mean)
E1 = -(z - 2.0)^2 / 2.0     # energy at upper singularity
E2 = -(z + 2.0)^2 / 2.0     # energy at lower singularity

prob0 = exp(E0) / (exp(E0) + exp(E1) + exp(E2))  # P(at center)
prob1 = exp(E1) / (exp(E0) + exp(E1) + exp(E2))  # P(at upper band)
prob2 = exp(E2) / (exp(E0) + exp(E1) + exp(E2))  # P(at lower band)

entropy = -sum(p * log(p))   # Shannon entropy of distribution
coherence = 1 - entropy/log(3)  # 1.0 = collapsed, 0.0 = full superposition
```

These ARE P(success) for each trade direction:
- **Mean reversion trade**: P(success) = prob0 (price returns to center)
- **Breakout LONG**: P(success) = prob1 (price reaches upper band)
- **Breakout SHORT**: P(success) = prob2 (price reaches lower band)

The probabilistic scoring system we need is ALREADY COMPUTED. It's in `MarketState` as
`prob_center`, `prob_upper`, `prob_lower`, `entropy_normalized`. The gate cascade
ignores all of it.

---

## Reconnection Plan

### Phase 1: Use What's Already Computed
1. Add `is_resonance_cascade` property to MarketState or BeliefState
2. Use wave function probabilities (prob0/1/2) in the scoring function
3. Use tunnel_probability in entry quality assessment
4. Use coherence as a cross-TF resonance detector

### Phase 2: Replace Gates with Probability
1. P(success) = f(wave_function, tunnel_prob, coherence, conviction, funnel_tree)
2. No binary gates — continuous probability
3. Position size proportional to P(success)
4. Exit timing driven by halflife decay of P(remaining profit)

### Phase 3: Cascade Mode
1. Detect resonance cascade (cross-TF ADX/Hurst/DMI alignment)
2. Suppress TP, disable giveback
3. Trail via survival_stop + tidal_wave only
4. Exit when Macro Hurst decays below 0.5

---

## Key Insight

The metaphor purge renamed the concepts but kept the math. Every bar still computes:
- Z-score (position in potential well)
- Forces (gravity + repulsion + momentum)
- Wave function probabilities (prob0, prob1, prob2)
- Entropy/coherence (superposition measure)
- Tunnel probability (P(mean reversion))
- Hurst exponent (trending vs chaotic)
- ADX/DMI (directional energy)

The system was always probabilistic. The gates made it deterministic.
Reconnecting the quantum state to the decision logic restores the original design.
