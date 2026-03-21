# Quantum Design Intent — Pre-Gate Scoring + Unimplemented Theories
> Source: Original design document (Mar 2026)

QUANTUM STATE REFERENCE: THE UNIFIED MARKET PHYSICS
THEORIES
TARGET AUDIENCE: Claude Code (Development AI) PURPOSE: You already possess the raw
mathematical implementation in quantum_field_engine.py  (919 lines). This document
provides the Design Intent, State Mappings, Decision Rules, and Architectural History
necessary to wire those quantum state computations back to the execution, scoring, and gating
logic.
This reference bridges the gap between the theoretical physics and the deterministic trading
behavior of the Bayesian-AI system.
1. THEORY I: THE NIGHTMARE PROTOCOL (The Gravitational Field)
1.1 Mathematical Formalization
The Ornstein-Uhlenbeck (O-U) Process: Defines the baseline mean-reverting behavior of
the particle.
Gravitational Force (
): Pulls the particle back to the mean.
Boundary Repulsion Force (
): Electrostatic-like pushback emitted by the Event
Horizon.
Roche Limit (
): Boundary where tidal forces tear the localized trend apart.
Event Horizon (
): Absolute singularity boundary; probability density 
.
1.2 Design Intent
The market is not a random walk; it is a liquidity-driven gravitational well. Market makers
algorithmicly pull price back to VWAP/Mean ( ). The Standard Error bands ( ) are not mere
statistical metrics—they represent structural physical boundaries (Roche Limits).
dX
​ =
t
θ(μ −X
​)dt +
t
σdW
​t
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
3σ
→0
μ
σ
3/16/26, 12:09 AM
Google Gemini
https://gemini.google.com/app/60110a6353dc8373
1/5


What it models: Localized mean reversion, inventory rebalancing, and the exhaustion of
kinetic order flow against limit-order walls.
What breaks if ignored: A standard trend-following bot will buy a breakout at the exact
moment the micro-trend hits a macro 
 wall, getting crushed by the "rubber band"
snapback.
1.3 State Variables
z_score : Spatial coordinate of the particle.
micro_velocity : First derivative of price (
).
F_gravity , F_repulsion , net_force : Computed magnitudes of forces.
prob_center , prob_upper , prob_lower : The Boltzmann probabilities derived from
potential energy of the Net Force.
1.4 Decision Rules
ENTER (Wave Function Collapse): Price has pulled back to a stable gravitational zone (
), and the computed Net Force strongly biases toward a specific target
band. The particle is "cleared" to travel.
STAY FLAT (Event Horizon): Price is near the singularity (
). Trend-continuation
entries are hard-blocked. You do not initiate momentum when the particle is out of bounds.
EXIT (Roche Limit Hit / Death Hook): Price touches 
 AND micro_velocity  flips sign.
The localized kinetic energy has hit the wall and died. Exit immediately.
2. THEORY II: THE THREE-BODY PROBLEM (Fractal Chaos)
2.1 Mathematical Formalization
The Superposition Wave Function (
):
Boltzmann Probabilities (
):
Shannon Entropy (
):
3σ
dP/dt
−1.0 ≤Z ≤1.0
∣Z∣≥2.5
2σ
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
S
3/16/26, 12:09 AM
Google Gemini
https://gemini.google.com/app/60110a6353dc8373
2/5


Fractal Coherence (
): Dot product (alignment) of the Micro, Meso, and Macro directional
vectors.
2.2 Design Intent
In astrophysics, three interacting bodies create deterministic chaos. In trading, these are the
Macro (1h), Meso (5m), and Micro (15s) timeframes. A trader executing solely on a 15-second
chart is blind to the 1-hour gravitational pull that will inevitably override their position.
What it models: The absolute necessity of multi-timeframe phase alignment. We model
chaos using Shannon Entropy to find brief, probabilistic windows of stability.
What breaks if ignored: "Death by a thousand cuts." The bot takes perfectly valid micro-
setups that are immediately vaporized by conflicting macro order flow.
2.3 State Variables
entropy : The Shannon entropy score of the current bar's wave function.
coherence : The calculated alignment [0.0 to 1.0] of the three timeframes.
2.4 Decision Rules
TRADEABLE (Alignment/Resolution): * entropy  is LOW (meaning the wave function has
collapsed and one directional probability is dominant, e.g., 
).
coherence  is HIGH (Macro, Meso, Micro vectors are mathematically synchronized).
UNTRADEABLE (Chaos/Misalignment): * entropy  is HIGH (probabilities are
flat/uncertain, e.g., 
).
coherence  is LOW. The system completely halts entries.
EXIT (Decoherence): A previously high coherence  state rapidly decays into high
entropy  while a trade is open.
3. THEORY III: THE RESONANCE CASCADE (The Alpha Event)
3.1 Mathematical Formalization
The Lyapunov Exponent ( ): Measures the rate of separation of infinitesimally close
trajectories. Determines stability (
) vs. cascade (
).
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
​ >
upper
0.85
[0.33, 0.33, 0.33]
λ
λ < 0
λ > 0
∣δZ(t)∣≈e ∣δZ(0)∣
λt
3/16/26, 12:09 AM
Google Gemini
https://gemini.google.com/app/60110a6353dc8373
3/5


Hurst Exponent (
): Measures the long-term memory of the series.
3.2 Design Intent
Occasionally, the O-U mean-reverting physics break down. When the frequencies of the Three
Bodies perfectly synchronize, constructive interference shatters the Roche Limits. The market
transitions from stable to chaotic, resulting in a Black Swan or a massive trend day.
What it models: Liquidity cascades, short squeezes, and algorithmic momentum ignition.
What breaks if ignored: The bot will repeatedly try to fade/short a massive breakout
because the price hit 
, effectively stepping in front of a freight train and blowing up the
account.
3.3 State Variables
lyapunov_exponent : Real-time calculation of .
macro_hurst_exponent : Rolling Hurst calculation on the Macro timeframe.
is_resonance_cascade : Boolean flag activating the Cascade Protocol.
3.4 Cascade Mode Protocol Details
When is_resonance_cascade = True , standard O-U physics are suspended.
Exact Trigger Conditions:
1. 
macro_hurst_exponent
 (Mathematical proof of persistent memory).
2. 
lyapunov_exponent
 (Mathematical proof of chaotic exponential separation).
3. 
coherence
 (Cross-TF vector synchronization).
4. Macro Volatility (
) is visibly expanding directionally.
Suppressed Exits:
Standard Take Profit (Do not arbitrarily cap gains in a Black Swan).
Band Exits / Envelope (Do not fade the Roche limits, as they are actively breaking).
Active Exits:
survival_stop  (If Bayesian win probability decays below 50% despite the cascade).
tidal_wave  (If volatility expands violently against our position).
Termination Conditions (Reverting to Nightmare Protocol):
macro_hurst_exponent  decays back toward 
 (Random walk returning).
Macro DMI/ADX vector hooks downward violently (Order flow exhaustion).
lyapunov_exponent  flips negative (Trajectory stabilizing).
H
3σ
λ
> 0.55
> 0
> 0.90
σ
​
macro
0.50
3/16/26, 12:09 AM
Google Gemini
https://gemini.google.com/app/60110a6353dc8373
4/5


4. THE ORIGINAL SCORING LOGIC (Pre-Gate Cascade)
Before the rigid boolean screening gates ( tools/analyze_gates.py , screening_gates.json )
were introduced in the newer architecture, the system relied purely on continuous quantum
probabilistic resolution.
If you need to reconstruct the pure scoring loop, the logic flowed as follows:
1. Force to Energy: The quantum_field_engine  computed 
, 
, and
, summing them into a Net Force.
2. Energy to Probability: Net Force was converted to potential energy states, which were
then converted into Boltzmann weights ( prob_center , prob_upper , prob_lower ).
3. Tunneling Computation: The system ran a Monte Carlo O-U simulation to find the
tunnel_probability —the statistical likelihood that the particle could reach the target
band without hitting the stop-loss boundary first.
4. Execution Decision: A trade was executed purely on mathematical edge:
Highest wave function probability (
) was 
AND tunnel_probability  was 
AND entropy  was low.
Note: There were no hard boolean "gates"; it was a continuous spectrum of expected
value.
5. UNIMPLEMENTED / THEORETICAL CONCEPTS
These concepts were heavily researched and mapped mathematically but were never fully
implemented in the live execution code. If you encounter references to them in the codebase or
notes, treat them as theoretical scaffolding:
1. Continuous Phase Transitions: The current execution engine treats the shift from
Nightmare Protocol (Mean Reversion) to Resonance Cascade (Trend) as a discrete boolean
switch. The original mathematical intent was a continuous phase transition (like water to
steam), where strategy weights would smoothly interpolate as the Hurst Exponent slid from
0.49 to 0.51.
2. Cascade-Specific Trailing Mathematics: A dynamically calculated trailing stop directly
proportional to the inverse of the Lyapunov Exponent (
). As chaotic separation
accelerated, the stop would loosen to allow for volatility; as  slowed, the stop would
tighten aggressively to capture the peak.
3. True Cross-TF Wave Interference: The current system uses a simplified coherence  dot-
product to measure alignment between the 3 timeframes. The unimplemented theory
F
​
gravity F
​
repulsion
F
​
momentum
P
​i
> 0.75
> 0.60
1/λ
λ
3/16/26, 12:09 AM
Google Gemini
https://gemini.google.com/app/60110a6353dc8373
5/5


