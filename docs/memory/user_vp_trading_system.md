---
name: VP Complete Trading System (user's manual methodology)
description: User's full manual trading framework — zone map, entry protocols, risk rules. This is THE ground truth the Bayesian-AI system must replicate. All research lines serve this.
type: user
---

## VP Trading System — Complete Manual Framework

### I. Core Philosophy
- **Reaction > Prediction**: Wait for force, don't guess news
- **Asian Trap**: First move (06:30-06:45) is often liquidity grab — don't touch
- **The Wave**: Only enter when market leaves "Equality" and enters "Flow"

### II. Zone Map (The Architecture)

| Zone | Band | Rule |
|------|------|------|
| Core | Mean (0σ) | Home base. Price always wants to return. |
| Chop | ±1.5σ | Oscillation only. Buy bottom / sell top. Target mean. |
| Trend | ±2-3σ | Wave riding. This is where money is made. Hold while ADX rises. |
| Abyss | >3σ | Blind spot. MUST zoom out (fractal visibility). |
| Wall | ±4σ | HARD STOP. 99.99% exhaustion. Immediate exit. |

### III. Execution Protocols

**Entry Gate (DMI)**:
- No braids (Red/Green tangling) = sit on hands
- Trigger: DMI separation > 5 points
- Fuel: ADX must be > 20 (rising) to confirm wave

**Fractal Split (TF-specific settings)**:
- Sniping (5s / 1m): DMI(5) for instant reaction
- Mapping (15m / 1h): DMI(14 or 15) to filter noise

**Visibility Rule**:
- NEVER trade if 3σ line is off-screen (= price too extended for this TF)
- If 1m blind → switch to 5m. If 5m blind → switch to 15m.
- Always keep the "rubber band" visible

### IV. Risk Management

- **Snap-Back**: Further from mean = faster it will snap back
- **No add outside 3σ**: Don't increase position beyond abyss
- **Profit Lock**: Up +150 points AND price hits 4σ → FLATTEN immediately
- **Pre-Open Force Check**: Calculate distance to mean before session opens
  to prevent "front-running" errors

### Mapping to Architecture

Already implemented:
- z_score per TF (zone detection)
- self_adx, self_dmi_diff (entry gate data)
- parent_z (headroom data)
- TF workers with per-worker DMI
- Mean reversion forces

Missing (6 features needed):
1. Headroom gate: |Z_macro| >= 2 → BLOCKED
2. 4σ Wall exit: hard kill when |z| >= 4
3. Zone mode: chop (oscillate to mean) vs trend (ride wave)
4. Asian Trap: session-time gate, skip first 15 min
5. Visibility rule: |z| > 3 → force upshift to higher TF
6. Pre-open force check: distance to mean before first trade
