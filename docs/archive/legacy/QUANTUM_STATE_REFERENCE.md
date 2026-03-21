# Quantum State Reference — Unified Market Physics Theories
> Source: Original design document (Mar 2026)

QUANTUM STATE REFERENCE: THE UNIFIED MARKET PHYSICS
THEORIES
TARGET AUDIENCE: Claude Code (Development AI) PURPOSE: This document is the ultimate
architectural specification mapping the theoretical physics of the Bayesian-AI trading system to
the deterministic state variables and decision execution logic. Use this reference to reconnect
the quantum state computations ( bayesian_brain.py , physics_utils.py ,
fractal_dmi.py ) to the entry scoring and exit engines ( execution_engine.py ,
core/exits/* ).
1. THEORY I: THE NIGHTMARE PROTOCOL (The Gravitational Field)
The Concept: The market acts as a localized gravitational well. Price is a particle tethered to a
moving Center of Mass. The Standard Error bands define the structural integrity of this
gravitational field.
1.1 Mathematical Formalization
The Ornstein-Uhlenbeck (O-U) Process: Defines the baseline mean-reverting behavior of
the particle.
The State Position (Z-Score):
Gravitational Force (
): Pulls the particle back to the mean. It strengthens linearly
as distance increases.
Boundary Repulsion Force (
): The electrostatic-like repulsion emitted by the
Event Horizon. It scales inversely with the cubed distance to the boundary ( ).
Roche Limit (
): The boundary where tidal forces begin to tear the particle's localized
trend apart.
dX
​ =
t
θ(μ −X
​)dt +
t
σdW
​t
Z
​ =
t
​
σ
X
​ −μ
t
F
​
gravity
F
​ =
gravity
−θ ⋅Z
​ ⋅
t σ
F
​
repulsion
r
F
​ =
repulsion
​
where
r =
r3
k
3 −∣Z
​∣
t
2σ
3/15/26, 11:58 PM
Google Gemini
https://gemini.google.com/app/60110a6353dc8373
1/5


Event Horizon (
): The absolute singularity boundary. Probability density drops to near
zero.
1.2 State Variables (Encoding the Theory)
These fields in MarketState  / ThreeBodyQuantumState  carry the protocol's state:
z_score : The spatial coordinate of the particle.
micro_velocity : The first derivative of price with respect to time (
).
F_gravity : The computed magnitude of the restoring force.
F_repulsion : The computed magnitude of the boundary pushback.
net_force : 
.
1.3 Decision Rules
ENTRY (Wave Function Collapse): Enter only when net_force  vector aligns with the
target state, and price has pulled back to a stable zone (
).
EXIT (Roche Limit Hit / Death Hook): Trigger fractal_exhaust.py  when price touches
 (Roche Limit) AND micro_velocity  flips sign (kinetic energy reaches zero against the
wall).
NO-ENTRY (Event Horizon): Hard block on any trend-continuation entries if 
.
The particle is too close to the singularity.
2. THEORY II: THE THREE-BODY PROBLEM (Fractal Chaos)
The Concept: The market consists of three interacting gravitational bodies: Macro (1h), Meso
(5m), and Micro (15s). Because a three-body system is deterministically chaotic, we must model
it probabilistically using quantum wave functions and entropy.
2.1 Mathematical Formalization
The Superposition Wave Function (
): The particle exists in a superposition of three
discrete states (Center, Upper Band, Lower Band) until observed/resolved.
Boltzmann Probabilities (
): The probability of collapsing into state , calculated via the
energy potentials of the Net Force field.
3σ
dP/dt
(F
​, F
​, F
​)
∑
gravity
repulsion
momentum
∣Z∣≤1.0
2σ
∣Z∣≥2.5
Ψ
Ψ = a
​Ψ
​ +
0
center
a
​Ψ
​ +
1
upper
a
​Ψ
​
2
lower
P
​i
i
P
​ =
i
​
​ exp(−βE
​)
∑j
j
exp(−βE
​)
i
3/15/26, 11:58 PM
Google Gemini
https://gemini.google.com/app/60110a6353dc8373
2/5


Shannon Entropy (
): Measures the chaos/uncertainty of the three-body system.
Fractal Coherence (
): The dot product (alignment) of the Micro, Meso, and Macro
directional vectors.
2.2 State Variables (Encoding the Theory)
These fields carry the quantum state:
prob_center , prob_upper , prob_lower : The collapsed Boltzmann probabilities (
).
entropy : The Shannon entropy score of the current bar.
coherence : The calculated alignment [0.0 to 1.0] of the 3 timeframes.
tunnel_probability : The Monte Carlo-derived probability of the particle tunneling
through the current resistance layer.
2.3 Decision Rules
STAY FLAT (Chaos): If entropy  is HIGH (e.g., probabilities are 
) OR
coherence  is LOW, the three bodies are in chaotic misalignment. Block all entries.
ENTER (Resolution): When entropy  drops (e.g., 
) AND coherence  is
HIGH, the 3-body system has temporarily resolved into a stable orbit. Execute trade in the
direction of the highest probability state.
EXIT (Decoherence): Trigger regime_decay.py  if a previously high coherence  state
rapidly decays into high entropy  while a trade is open.
3. THEORY III: THE RESONANCE CASCADE (The Alpha Event)
The Concept: Occasionally, the frequencies of the Three Bodies synchronize. The Micro particle
does not bounce off the Meso Roche Limit; it resonates with the Macro gravity. The standard
statistical boundaries shatter, resulting in violent, exponential directional expansion (a Black
Swan / Trend Day).
3.1 Mathematical Formalization
The Lyapunov Exponent ( ): Measures the rate of separation of infinitesimally close
trajectories. Determines if the system is stable (
) or chaotic/cascading (
).
S
S = −
​P
​ ln(P
​)
i
∑
i
i
C
P
​i
≈[0.33, 0.33, 0.33]
P
​ >
upper
0.80
λ
λ < 0
λ > 0
∣δZ(t)∣≈e ∣δZ(0)∣
λt
3/15/26, 11:58 PM
Google Gemini
https://gemini.google.com/app/60110a6353dc8373
3/5


Hurst Exponent (
): Measures the long-term memory of the series.
: Mean-reverting (Stable)
: Persistent (Resonance Cascade)
3.2 State Variables (Encoding the Theory)
lyapunov_exponent : Real-time calculation of .
macro_hurst_exponent : Rolling Hurst calculation on the Macro timeframe.
is_resonance_cascade : Boolean flag. True  if 
 and 
.
3.3 Decision Rules
HOLD (Cascade Active): If is_resonance_cascade == True , standard physics are
suspended. Do not fade the edges.
EXIT (Cascade Exhaustion): Exit cascade mode only when 
 decays below 
 or  flips
negative.
4. THE CASCADE MODE PROTOCOL
When the system detects a true market anomaly, it enters a dedicated execution protocol
designed to capture maximum asymmetric upside.
Exact Conditions ( is_resonance_cascade = True ):
1. 
macro_hurst_exponent
2. 
lyapunov_exponent
3. 
coherence  is near 
 (Macro, Meso, and Micro DMI/ADX vectors perfectly aligned).
4. Macro Volatility (
) is expanding directionally.
Suppressed Exits (During Cascade):
Standard Take Profit: Disabled. We do not cap gains during a cascade.
Band Exit / Roche Limit Defense: Disabled. The 
 and 
 bands are actively breaking;
fading them is mathematically fatal.
Cascade Termination Triggers:
Regime Decay ( regime_decay.py ): The macro_hurst_exponent  drops below 
,
signaling a return to mean-reversion physics.
Tidal Wave ( tidal_wave.py ): Volatility suddenly expands violently in the opposite
direction of the cascade.
Vector Collapse: Macro ADX hooks downward violently.
H
H < 0.5
H > 0.55
λ
H > 0.55
λ > 0
H
0.50
λ
> 0.55
> 0
1.0
σ
​
macro
2σ
3σ
0.50
3/15/26, 11:58 PM
Google Gemini
https://gemini.google.com/app/60110a6353dc8373
4/5


5. THE ORIGINAL NIGHTMARE FIELD EQUATION (Compute Pipeline)
The complete computation pipeline that transforms raw OHLCV tick data into actionable
quantum states.
5.1 The Master Equation
5.2 The Compute Pipeline Step-by-Step
1. Statistical Bounds Computation:
Ingest raw OHLCV.
Calculate Linear Regression Mean ( ) and Standard Error Bands ( ) for Macro, Meso,
Micro.
Compute 
-score for current price.
2. Force Field Computation:
Calculate 
 (
).
Calculate 
 (
 from the 
 boundary).
Calculate 
 (Derivative of the ADX/DMI vector).
Sum to find 
.
3. Wave Function Computation:
Convert 
 into Potential Energies (
) for the three spatial states (Mean, Upper
Band, Lower Band).
Apply Boltzmann weights to compute 
 ( prob_center , prob_upper , prob_lower ).
4. Tunnel Probability (Monte Carlo):
Run stochastic O-U path generation projecting forward 
 bars.
Calculate the percentage of paths that successfully pierce the localized resistance
without breaching the stop-loss boundary.
Output as tunnel_probability .
5. Execution Handoff:
The ThreeBodyQuantumState  vector is populated.
Handoff to execution_engine.py  and core/exits/*  to parse the Boolean rules
defined in Sections 1-4.
dX
​ =
t
​ +
Ornstein-Uhlenbeck
​
θ(μ(t) −X
​)dt + σdW
​
t
t
​ +
Algorithmic Control
​
F
​(e)dt
PID
​
Resonance Jump
​
J (λ)
μ
σ
Z
F
​
gravity −θ ⋅Z ⋅σ
F
​
repulsion
​
r31
3σ
F
​
momentum
F
​
net
F
​
net
E
​i
P
​i
N
3/15/26, 11:58 PM
Google Gemini
https://gemini.google.com/app/60110a6353dc8373
5/5


