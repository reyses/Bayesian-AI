# Market Participant Hierarchy — The Three-Body Social Model

> The three-body quantum model maps to real market participants operating at
> different timescales with different power. This is the conceptual key to
> the entire system.

## The Hierarchy

| Body | Who | Timeframe | Force | Market Role |
|------|-----|-----------|-------|-------------|
| **Gods** (Institutions) | Sovereign funds, central banks, mega funds | 1h-1D+ | Gravity (μ) — the regression mean | Set fair value. Move in mysterious ways. Their agenda IS the center of mass. |
| **Demi-Gods** (Algos) | HFT, market makers, execution algos | 1m-15m | PID Control (F_pid) — Kp/Ki/Kd | Execute the divine agenda. Enforce the bands. Dampen volatility. |
| **Avatars** (Prop firms) | Prop desks, systematic traders | 5m-1h | Momentum (F_momentum) | Align with the algos. Follow the flow. Amplify moves. |
| **Mortals** (Retail) | Us. Everyone else. | 15s-5m | Noise (σdW) — Brownian motion | At the whim of the system. The leaf in the wind. |

## Force Mapping

### Gods → Gravity (O-U Mean Reversion)
- `F_gravity = -θ · Z · σ`
- The gods' positions define μ (fair value). Price is gravitationally bound to their agenda.
- When they rebalance, μ shifts and everything else adjusts.
- You cannot see their positions directly — you infer them from the regression mean.

### Demi-Gods → PID Control
- `F_PID = Kp·e + Ki·∫e + Kd·de/dt`
- **Kp (Proportional)**: Band response. "Price is at 2σ, sell." This is the standard error response.
- **Ki (Integral)**: Accumulation. "Price has been low too long, buy." This explains the spring/squeeze.
- **Kd (Derivative)**: Jitter dampening. "Velocity too high, slow down." This explains wicks at the bands.
- At 1σ, the PID has full control → price enters a "PID trance" (tight oscillation around setpoint)
- At 4σ, the PID is saturated → erratic price behavior (control system breakdown)

### Avatars → Momentum
- `F_momentum = velocity · volume / σ`
- Prop firms follow the algo flow and amplify it.
- They add momentum in the direction the demi-gods are pushing.
- During resonance cascade, they pile on and the move explodes.

### Mortals → Noise (Brownian Motion)
- `σ(v,τ)dW_t`
- Retail creates noise but doesn't move the market.
- The system exists to navigate this noise by reading the forces above.

## The Nightmare Protocol

**Being a leaf that knows the wind patterns.**

We can't control the gods, can't out-compute the demi-gods, can't out-capitalize
the avatars. But we can read the gravitational field they create and navigate it.

The Nightmare Field Equation computes all four forces acting on price:
```
dX_t = θ(μ - X_t)dt + σ(v,τ)dW_t + F_PID(e)dt + J(λ)
         gods            noise        demi-gods    black swan
```

We measure who's in control, which direction the forces point, and trade
ONLY when the physics align.

## Resonance Cascade = Divine Alignment

When gods, demi-gods, and avatars all move in the same direction simultaneously:
- The PID controllers stop defending the bands → they JOIN the flow
- The Roche limits shatter (2σ/3σ bands break)
- Price enters escape velocity
- Disable TP (don't cap gains when the control system is off)
- Trail via survival_stop only until the macro Hurst decays

## Sigma Zones

| Zone | Controller | Behavior | Trading Rule |
|------|-----------|----------|-------------|
| 0-1σ | PID dominant | Oscillation, tight range, "trance" | Don't trade — noise zone, algo territory |
| 1-2σ | PID weakening | Trend developing, bands tested | Entry zone — reversion trades |
| 2-3σ | PID losing control | Roche limit, structural stress | Exit zone — take profit / death hook |
| 3σ+ | PID saturated | Breakdown or cascade | Either stay flat (nightmare) or ride cascade |
| 4σ+ | PID broken | Erratic, chaotic | Stay flat — control system failure |

## The 1σ PID Trance (User's Original Observation)

At 1σ, the signal gets caught in a PID trance. The HFT algos are actively
managing price within their control band. The mean-reversion force is weak
(small z), the repulsion force is zero (far from boundaries), and the PID
has full authority.

Trading at 1σ is trading AGAINST the demi-gods' control system. The signal
isn't a trade — it's the algos doing their job.

## Cross-TF Tunneling

What looks like 3σ on the 5m might be 1σ on the 1h. Price "tunnels" through
the lower timeframe's Roche limit because the higher timeframe's gravity is
pulling it THROUGH the band. This isn't a failure — it's the three-body problem.
Your micro Roche limit isn't the real boundary; the macro's is.

This is why the system checks alignment across TFs before entering. A 3σ
reversion on the micro is only valid if the macro isn't pulling in the
opposite direction.
