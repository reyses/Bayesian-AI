# COUNTER-PROPOSAL: MTF Sensor Array + Two-Layer CNN

**From:** Claude.ai (architect/reviewer)  
**To:** VS Code Claude (executor)  
**Status:** Approved with modifications — replaces PROPOSAL_MTF_SENSOR_ARRAY.md  
**Depends on:** Layer 1 walk-forward results (72% direction, 12 trades/day baseline)

---

## What We're Solving

Layer 1 StatePredictor proved the model CAN predict direction (72%) and
captured $1,609/day with breakeven protection. But two problems remain:

1. **No multi-TF context** — the model enters blind to hourly structure,
   5m swing context, and 1s timing. A 1m LONG at the top of a 1h SHORT
   has nowhere to go.

2. **Trade fragmentation** — the model fragments one big move into many
   small trades. Each round-trip costs ~4 ticks in slippage. Fewer trades
   on the same gross PnL = dramatically higher net.

This proposal solves both in one architecture.

---

## Architecture: Two-Layer CNN on 29D Features

```
29D Features (13D base + 16D multi-TF)
         │
         ▼
┌─────────────────────────────┐
│  LAYER 1: Direction Engine  │
│  StatePredictor CNN         │
│  Input: (10 × 29)           │
│  Output: 21 (7D × 3h)      │
│  Job: "Go long. Momentum    │
│   building across horizons." │
│  Accuracy: 72%+ (target)    │
│  ~28K params                │
└──────────────┬──────────────┘
               │ entry signal + prediction vector
               ▼
┌─────────────────────────────┐
│  LAYER 2: Duration Engine   │
│  DurationPredictor MLP      │
│  Input: L1 prediction (21)  │
│         + regime context (4) │
│         + session context (4)│
│  Output: P(take), hold_bars │
│  Job: "Take this one. Hold  │
│   for 14 bars. Don't wobble."│
│  ~800 params                │
└──────────────┬──────────────┘
               │ filtered signal + hold commitment
               ▼
┌─────────────────────────────┐
│  GUARDRAILS (unconditional) │
│  Hard SL: 40 ticks          │
│  2s slippage on every fill  │
│  Maintenance flatten         │
│  Daily loss limit            │
│  Circuit breaker (8/10 SL)  │
└─────────────────────────────┘
```

**Layer 1** predicts what price will do. **Layer 2** decides whether to
act and for how long. **Guardrails** can't be overridden by either layer.

---

## Part 1: 29D Feature Spec

### Base Features (13D — unchanged, 1m resolution)

```
Directional (7D):
  [0]  dmi_diff        state.dmi_plus - state.dmi_minus
  [1]  dmi_gap         abs(dmi_diff)
  [2]  vol_rel         volume / SMA(volume, 30)
  [3]  dir_vol         sign(price_change) × vol_rel
  [4]  velocity        state.velocity
  [5]  z_se            (price - mean) / SE over 60 bars
  [6]  price_accel     velocity - prev_velocity

Regime (4D):
  [7]  std_price        or hurst_exponent (use whichever current baseline uses)
  [8]  variance_ratio   or oscillation_entropy
  [9]  bar_range        or regression_sigma
  [10] wick_ratio       or regression_slope

Context (2D):
  [11] vwap_distance    or session_phase
  [12] time_of_day      or bars_to_close
```

Note: The exact regime/context features should match whatever produced
the $1,609/day baseline. Do NOT change these — they're proven.

### Multi-TF Features (16D — 4 features × 4 timeframes)

| Index | Feature | TF | Source | What It Answers |
|-------|---------|-----|--------|----------------|
| [13] | dmi_diff | 1s | last completed 1s bar | Who's winning RIGHT NOW inside this 1m bar? |
| [14] | z_se | 1s | last 60 × 1s bars | Where in the micro-structure? |
| [15] | velocity | 1s | last completed 1s bar | How fast is the current tick flow? |
| [16] | vol_rel | 1s | 1s vol / SMA(1s vol, 30) | Institutional participation right now? |
| [17] | dmi_diff | 5m | last completed 5m bar | Swing direction |
| [18] | z_se | 5m | last 60 × 5m bars | Where in the 5m swing? |
| [19] | velocity | 5m | last completed 5m bar | Swing momentum |
| [20] | vol_rel | 5m | 5m vol / SMA(5m vol, 30) | Swing participation |
| [21] | dmi_diff | 15m | last completed 15m bar | Session-level direction |
| [22] | z_se | 15m | last 60 × 15m bars | Where in the session range? |
| [23] | velocity | 15m | last completed 15m bar | Session momentum |
| [24] | vol_rel | 15m | 15m vol / SMA(15m vol, 30) | Session participation |
| [25] | dmi_diff | 1h | last completed 1h bar | Structural direction |
| [26] | z_se | 1h | last 60 × 1h bars | Where in the daily range? |
| [27] | velocity | 1h | last completed 1h bar | Structural momentum |
| [28] | vol_rel | 1h | 1h vol / SMA(1h vol, 30) | Structural participation |

### Why These 4 Features Repeated Across TFs

From the grounded feature research: dmi_diff (direction, 98.8% recall),
z_se (position in range), velocity (rate of change), vol_rel (participation).
Same 4 questions at each scale. The CNN learns cross-scale relationships:

```
1s velocity spiking + 1m dmi_diff crossing + 1h dmi_diff agreeing
= high confidence multi-TF aligned entry

1m dmi_diff says LONG + 1h z_se at +3σ (overbought)
= skip — fighting the structure
```

---

## Part 2: Normalization (HARD REQUIREMENT)

Each timeframe's features are z-scored INDEPENDENTLY before concatenation.
This is non-negotiable.

```python
def normalize_per_tf(feats_29d, tf_boundaries):
    """Z-score each TF's features using that TF's own statistics.
    
    1h dmi_diff has different scale than 1m dmi_diff (14-hour smoothing
    vs 14-minute). Without per-TF normalization, the CNN learns scale
    artifacts instead of cross-TF relationships.
    
    tf_boundaries maps feature indices to TF groups:
      [0:13]  = 1m base features  → z-score from 1m stats
      [13:17] = 1s features       → z-score from 1s stats
      [17:21] = 5m features       → z-score from 5m stats
      [21:25] = 15m features      → z-score from 15m stats
      [25:29] = 1h features       → z-score from 1h stats
    
    Use ROLLING z-score (30-day window) to avoid lookahead in normalization.
    """
    normalized = np.copy(feats_29d)
    
    for start, end in tf_boundaries:
        for col in range(start, end):
            series = pd.Series(feats_29d[:, col])
            rolling_mean = series.rolling(30 * 1380, min_periods=100).mean()
            rolling_std = series.rolling(30 * 1380, min_periods=100).std()
            normalized[:, col] = (series - rolling_mean) / (rolling_std + 1e-8)
    
    return normalized

TF_BOUNDARIES = [
    (0, 13),   # 1m base
    (13, 17),  # 1s
    (17, 21),  # 5m
    (21, 25),  # 15m
    (25, 29),  # 1h
]
```

---

## Part 3: MTF Alignment Validation (HARD REQUIREMENT)

Before ANY training, run this. Zero violations or stop.

```python
def validate_mtf_alignment(df_1m, higher_tf_dfs, alignment_indices):
    """Assert no lookahead in multi-TF feature alignment.
    
    For every 1m bar at time T, the higher-TF feature must come from
    a bar whose CLOSE TIME < T (strictly before, not equal).
    
    A 1h bar closing at 10:00 can be used by 1m bars from 10:01 onward.
    It CANNOT be used by the 1m bar at 10:00 (that bar opens at 10:00,
    same time the 1h bar closes — ambiguous, treat as lookahead).
    """
    total_violations = 0
    
    for tf_name, df_higher, bar_duration_sec in [
        ('1s', higher_tf_dfs['1s'], 1),
        ('5m', higher_tf_dfs['5m'], 300),
        ('15m', higher_tf_dfs['15m'], 900),
        ('1h', higher_tf_dfs['1h'], 3600),
    ]:
        violations = 0
        indices = alignment_indices[tf_name]
        
        for i in range(len(df_1m)):
            h_idx = indices[i]
            if h_idx < 0:
                continue  # no higher-TF bar available yet
            
            # Higher TF bar close time = timestamp + bar_duration
            h_close_time = df_higher.iloc[h_idx]['timestamp'] + bar_duration_sec
            m_open_time = df_1m.iloc[i]['timestamp']
            
            if h_close_time > m_open_time:
                violations += 1
                if violations <= 3:  # print first few
                    print(f"  LOOKAHEAD {tf_name}: 1m bar {i} at "
                          f"{m_open_time} uses {tf_name} bar closing at "
                          f"{h_close_time}")
        
        total_violations += violations
        status = "PASS" if violations == 0 else f"FAIL ({violations})"
        print(f"  {tf_name} alignment: {status}")
    
    assert total_violations == 0, (
        f"MTF alignment has {total_violations} lookahead violations. "
        f"Fix alignment logic before training."
    )
```

---

## Part 4: 1s Feature Extraction (Lightweight)

Full SFE on 8.8M bars is ~20-40 minutes on CUDA. Acceptable. Pre-compute
once, cache as `.npy`. But if SFE is too heavy, the 4 features we need
can be computed without full SFE:

```python
def extract_1s_features_lightweight(df_1s):
    """Extract 4 features from 1s OHLCV without full SFE.
    
    Fallback if batch_compute_states on 8.8M bars is too slow.
    These 4 features need only: price, volume, and rolling windows.
    """
    prices = df_1s['close'].values
    volumes = df_1s['volume'].values
    n = len(prices)
    
    feats = np.zeros((n, 4))
    
    # Simplified DMI-like direction signal
    # (price change smoothed over 14 bars)
    up_moves = np.maximum(np.diff(prices, prepend=prices[0]), 0)
    dn_moves = np.maximum(-np.diff(prices, prepend=prices[0]), 0)
    
    alpha = 2.0 / (14 + 1)
    smooth_up = pd.Series(up_moves).ewm(alpha=alpha).mean().values
    smooth_dn = pd.Series(dn_moves).ewm(alpha=alpha).mean().values
    
    feats[:, 0] = smooth_up - smooth_dn  # dmi_diff proxy
    
    # z_se: z-score over 60 bars
    rolling_mean = pd.Series(prices).rolling(60, min_periods=1).mean().values
    rolling_std = pd.Series(prices).rolling(60, min_periods=1).std().values
    rolling_se = rolling_std / np.sqrt(np.minimum(np.arange(1, n+1), 60))
    feats[:, 1] = (prices - rolling_mean) / (rolling_se + 1e-8)
    
    # velocity: rate of change over 5 bars
    feats[5:, 2] = (prices[5:] - prices[:-5]) / 5.0
    
    # vol_rel: volume / 30-bar SMA
    vol_sma = pd.Series(volumes).rolling(30, min_periods=1).mean().values
    feats[:, 3] = volumes / (vol_sma + 1e-8)
    
    return feats
```

Prefer full SFE if it runs in <1 hour. Only use lightweight as fallback.

---

## Part 5: Layer 1 — Direction Engine (29D StatePredictor)

Same architecture as current StatePredictor, wider input.

```python
# core/trade_cnn.py — modification

class StatePredictor(nn.Module):
    """Predicts future 7D state at 3 horizons.
    
    v1: 13D input, ~16K params
    v2: 29D input, ~28K params  ← THIS VERSION
    
    The extra 16 features are multi-TF context. The CNN learns
    cross-TF patterns: "1m momentum + 1h agreement = strong signal"
    """
    
    def __init__(self, n_features=29, lookback=10, n_timeframes=1,
                 latent_dim=64):
        super().__init__()
        self.backbone = CNNBackbone(n_features, latent_dim)
        merge_dim = latent_dim * n_timeframes
        self.merge = nn.Linear(merge_dim, 64)
        self.head = nn.Linear(64, N_OUTPUT)  # 21
        self.relu = nn.ReLU()
    
    def forward(self, *tf_inputs):
        latents = [self.backbone(x) for x in tf_inputs]
        merged = torch.cat(latents, dim=1)
        x = self.relu(self.merge(merged))
        return self.head(x)
```

### Training

Same walk-forward carry-forward as before:
```
Day 1:  cold start → train on Day 1 (30 epochs) → model_v1
Day 2:  model_v1 → PREDICT Day 2 (score) → fine-tune (5 epochs) → model_v2
...
Day N:  model_v(N-1) → PREDICT Day N (score) → fine-tune → model_vN
```

### Expected Improvement Over 13D

The 13D model had 72% direction accuracy. With MTF context:
- Entries that fight 1h structure get filtered (1h dmi_diff disagrees)
- Hold decisions improve (5m/15m still agree → hold through 1m wobble)
- 1s features give early signal before 1m bar closes

Target: 75%+ direction accuracy. Even 73-74% with fewer bad entries is a win.

---

## Part 6: Layer 2 — Duration Engine

After Layer 1 fires an entry signal, Layer 2 decides:
1. **Should we take this trade?** (P_take > 0.5)
2. **If yes, hold for how many bars?** (predicted optimal duration)

### Why Layer 2 Exists

```
Without Layer 2 (current):
  Bar 10: enter long → Bar 13: 1m wobbles → exit (+4t)
  Bar 15: re-enter  → Bar 19: 1m wobbles → exit (+3t)
  Bar 22: re-enter  → Bar 28: move done  → exit (+5t)
  = 3 trades, 3 × 4t slippage = 12t friction, net = 0t

With Layer 2:
  Bar 10: enter long, Layer 2 says "hold 18 bars"
  Bar 13: wobble → HOLD (committed)
  Bar 19: wobble → HOLD (committed)
  Bar 28: 18 bars reached → evaluate → exit (+18t)
  = 1 trade, 1 × 4t slippage = 4t friction, net = +14t
```

Same model predictions. Same direction accuracy. 3.5× more net PnL.

### Model Definition

```python
# core/trade_selector.py

class DurationPredictor(nn.Module):
    """Predicts take/skip and hold duration for Layer 1 signals.
    
    Input: Layer 1 prediction (21) + MTF regime snapshot (4) + session (4) = 29
    Output: P(take), hold_bars
    
    ~800 params. Cannot overfit on 4000+ signals.
    Trained AFTER Layer 1, on Layer 1's saved signals + actual outcomes.
    """
    
    def __init__(self, input_dim=29):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.take_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.hold_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.ReLU(),  # hold is always positive
        )
    
    def forward(self, x):
        h = self.shared(x)
        p_take = self.take_head(h).squeeze(-1)
        hold_bars = self.hold_head(h).squeeze(-1)
        return p_take, hold_bars
```

### Layer 2 Input Features (29D)

```
From Layer 1 prediction (21):
  [0:7]   predicted 7D at t+1
  [7:14]  predicted 7D at t+5
  [14:21] predicted 7D at t+10

From MTF regime snapshot at entry (4):
  [21] 1h_dmi_diff   — are we WITH or AGAINST the hourly structure?
  [22] 15m_velocity   — session-level momentum
  [23] 5m_z_se        — where in the swing?
  [24] 1s_velocity    — how aggressive is the current tick flow?

From session context (4):
  [25] session_phase   — where in the trading day (0→1)
  [26] bars_to_close   — countdown to maintenance
  [27] signal_strength  — Layer 1 dmi_gap_t10 magnitude
  [28] bars_since_last — time since last trade (cluster avoidance)
```

### Layer 2 Training Labels

```python
def build_duration_labels(all_signals, min_net_pnl=4):
    """For each Layer 1 signal, compute optimal hold with slippage.
    
    Optimal hold = bar of MFE in the slippage-adjusted trajectory.
    
    Signals where MFE after slippage < min_net_pnl are labeled
    "don't take" (p_take=0). The model learns to skip trades
    where the predicted move doesn't cover friction.
    
    min_net_pnl=4 means: after 4 ticks round-trip slippage,
    the trade must still make at least 4 ticks to be "worth it."
    This teaches the model: short moves aren't profitable.
    """
    features = []
    hold_labels = []
    take_labels = []
    pnl_labels = []
    
    for day in all_signals:
        for signal in day['signals']:
            feat = build_l2_feature_vector(signal)  # 29D
            features.append(feat)
            
            # MFE from slippage-adjusted trajectory
            mfe_bar = signal['mfe_bar']
            mfe_pnl = signal['mfe_pnl']  # already includes slippage
            
            hold_labels.append(mfe_bar)
            pnl_labels.append(mfe_pnl)
            
            # Worth taking? Must clear friction + threshold
            take_labels.append(1 if mfe_pnl >= min_net_pnl else 0)
    
    return np.array(features), np.array(hold_labels), \
           np.array(take_labels), np.array(pnl_labels)
```

### Layer 2 Loss Function

```python
def duration_loss(p_take, pred_hold, y_take, y_hold, y_pnl):
    """Combined loss: learn BOTH selectivity and duration.
    
    Loss = BCE(take) + λ × PnL-weighted MSE(hold)
    
    PnL weighting: getting the +20t trade's duration right matters
    more than the +5t trade's duration.
    """
    bce = nn.BCELoss(reduction='none')
    
    # Take/skip loss (all signals)
    weights = 1.0 + y_pnl / (y_pnl.mean() + 1e-8)
    loss_take = (weights * bce(p_take, y_take)).mean()
    
    # Hold duration loss (only on "take" signals)
    take_mask = y_take > 0.5
    if take_mask.sum() > 0:
        loss_hold = (weights[take_mask] *
                    (pred_hold[take_mask] - y_hold[take_mask]) ** 2).mean()
    else:
        loss_hold = 0.0
    
    return loss_take + 0.1 * loss_hold
```

---

## Part 7: Slippage Model (HARD REQUIREMENT)

**Every simulation from this point forward includes slippage.**
No exceptions. No "add it later." Fantasy fills are how you lose money in live.

### The Rule

Every entry and exit has 2 seconds of system latency. The fill price is
the ACTUAL market price 2 seconds after the signal fires. Not a model.
Not a distribution. The real price from the 1s data.

```
Signal fires at T=10:00:00, price = 21450.00
System latency = 2 seconds
Fill at T=10:00:02, price = 21450.75 (whatever the market did)

Slippage = 3 ticks (real, from data)
```

The randomness IS the market. Some 2-second windows are flat (0 ticks).
Some are volatile (5 ticks). Some are favorable (-1 tick, price came back).
No parameters to tune. No distribution to get wrong.

### Implementation at 1s Resolution (Primary)

We have 1s data in ATLAS. Use it directly.

```python
LATENCY_SECONDS = 2  # system latency: signal → fill

def get_fill_price(prices_1s, timestamps_1s, signal_time, side):
    """Look up actual price 2 seconds after signal.
    
    signal_time: timestamp when signal fired
    Returns: actual fill price from 1s data
    
    This is NOT a model. It's a data lookup.
    The market IS the slippage.
    """
    fill_time = signal_time + LATENCY_SECONDS
    
    # Find the 1s bar at or just after fill_time
    fill_idx = np.searchsorted(timestamps_1s, fill_time, side='left')
    fill_idx = min(fill_idx, len(prices_1s) - 1)
    
    return prices_1s[fill_idx]


def compute_slipped_entry(prices_1s, timestamps_1s, signal_time, side):
    """Entry fill = market price 2s after entry signal."""
    return get_fill_price(prices_1s, timestamps_1s, signal_time, side)


def compute_slipped_exit(prices_1s, timestamps_1s, signal_time, side):
    """Exit fill = market price 2s after exit signal."""
    return get_fill_price(prices_1s, timestamps_1s, signal_time, side)
```

### Implementation at 1m Resolution (Fallback)

When 1s data isn't loaded (e.g., quick iteration), approximate using
the 1m bar's OHLC range as a proxy:

```python
def approximate_slippage_from_1m(bar, side, entry=True):
    """Fallback: use bar's range to estimate 2s adverse fill.
    
    Approximation: adverse fill is proportional to bar's volatility.
    A bar with 8-tick range has more 2s variance than a 2-tick range bar.
    
    Use: (high - low) × 0.15 as rough 2s adverse movement.
    This is ONLY for fast iteration. Final results must use 1s data.
    """
    bar_range_ticks = (bar['high'] - bar['low']) / TICK
    estimated_slip_ticks = max(0, int(round(bar_range_ticks * 0.15)))
    
    if entry:
        if side == 'long':
            return bar['close'] + estimated_slip_ticks * TICK
        else:
            return bar['close'] - estimated_slip_ticks * TICK
    else:  # exit
        if side == 'long':
            return bar['close'] - estimated_slip_ticks * TICK
        else:
            return bar['close'] + estimated_slip_ticks * TICK
```

### In Forward Trajectory Computation

When building Layer 2 training labels, each point in the trajectory uses
the actual 1s fill price, not the 1m bar close:

```python
def compute_trajectory_with_slippage(entry_signal, prices_1s, timestamps_1s,
                                      prices_1m, timestamps_1m):
    """Build PnL trajectory with real 2s slippage at every exit point.
    
    Entry fill: actual 1s price 2s after entry signal.
    Each bar's "exit PnL": what if we exited here? Use actual 1s price
    2s after that bar's close as the hypothetical exit fill.
    """
    side = entry_signal['side']
    signal_time = entry_signal['timestamp']
    
    # Entry fill
    entry_fill = get_fill_price(prices_1s, timestamps_1s, signal_time, side)
    
    trajectory = []
    for t in range(1, 61):
        bar_idx = entry_signal['bar_idx'] + t
        if bar_idx >= len(timestamps_1m):
            break
        
        # Hypothetical exit at this bar's close + 2s latency
        exit_signal_time = timestamps_1m[bar_idx]  # bar close time
        exit_fill = get_fill_price(prices_1s, timestamps_1s,
                                    exit_signal_time, side)
        
        if side == 'long':
            pnl_ticks = (exit_fill - entry_fill) / TICK
        else:
            pnl_ticks = (entry_fill - exit_fill) / TICK
        
        trajectory.append(pnl_ticks)
    
    # MFE from slippage-adjusted trajectory
    if trajectory:
        mfe_pnl = max(trajectory)
        mfe_bar = trajectory.index(mfe_pnl) + 1
    else:
        mfe_pnl = 0
        mfe_bar = 0
    
    return trajectory, mfe_bar, mfe_pnl
```

### Why This Is Better Than a Distribution

```
Distribution model:  you GUESS what slippage looks like
Actual 1s lookup:    you KNOW what slippage looked like

Distribution:  all 2s windows are equally likely to slip 2 ticks
Reality:       2s after a volume spike = 5 ticks
               2s during lunch chop = 0 ticks
               2s into a reversal = -2 ticks (favorable)

The model learns from ACTUAL fill quality, correlated with market conditions.
It learns: "entry during a volume spike has bad fills — don't enter there."
A random distribution can't teach this.
```

### Slippage Statistics Check

After building trajectories, print actual slippage from the 1s data:

```python
def print_slippage_stats(all_entries, all_exits):
    """Print actual fill quality from 1s data lookup."""
    entry_slips = [e['fill_price'] - e['signal_price'] for e in all_entries]
    exit_slips = [e['fill_price'] - e['signal_price'] for e in all_exits]
    
    # Adjust sign: positive = adverse
    for i, e in enumerate(all_entries):
        if e['side'] == 'short':
            entry_slips[i] = -entry_slips[i]
    for i, e in enumerate(all_exits):
        if e['side'] == 'long':
            exit_slips[i] = -exit_slips[i]
    
    all_slips = [(s / TICK) for s in entry_slips + exit_slips]
    
    print(f"Actual slippage ({len(all_slips)} fills from 1s data):")
    print(f"  Mean:      {np.mean(all_slips):+.2f} ticks")
    print(f"  Median:    {np.median(all_slips):+.1f} ticks")
    print(f"  Std:       {np.std(all_slips):.2f} ticks")
    print(f"  P25/P75:   {np.percentile(all_slips,25):+.1f} / "
          f"{np.percentile(all_slips,75):+.1f}")
    print(f"  Favorable: {np.mean(np.array(all_slips)<0):.0%}")
    print(f"  0 ticks:   {np.mean(np.array(all_slips)==0):.0%}")
    print(f"  1-2 ticks: {np.mean((np.array(all_slips)>=1)&(np.array(all_slips)<=2)):.0%}")
    print(f"  3+ ticks:  {np.mean(np.array(all_slips)>=3):.0%}")
    # This tells you the REAL cost of trading — no assumptions
```

---

## Part 8: Full Trading Simulation

### The Loop (Layer 1 + Layer 2 + Actual 2s Latency + Guardrails)

```python
def simulate_two_layer(feats_29d, prices_1m, timestamps_1m,
                        prices_1s, timestamps_1s,
                        layer1_model, layer2_model,
                        device, hard_sl=40):
    """Full simulation with both layers and actual 2s latency fills.
    
    Fill prices come from 1s data at T+2s — the real market price,
    not a model or distribution.
    
    Flow per bar:
    1. Layer 1: predict 21D state at horizons
    2. If entry conditions met: ask Layer 2
    3. Layer 2: take/skip + hold duration
    4. If take: fill at actual 1s price 2s later, COMMIT to duration
    5. During hold: ONLY hard SL can override
    6. At hold expiry: re-evaluate, fill exit at actual 1s price 2s later
    """
    TICK = 0.25
    n = len(feats_29d) - LOOKBACK - max(HORIZONS)
    
    in_trade = False
    trades = []
    all_entry_slips = []  # for slippage stats
    all_exit_slips = []
    
    for i in range(n):
        idx = i + LOOKBACK
        price = prices_1m[idx]
        signal_time = timestamps_1m[idx]
        
        # ── Layer 1: direction prediction ──
        x = torch.FloatTensor(
            feats_29d[idx-LOOKBACK:idx]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = layer1_model(x).cpu().numpy()[0]
        
        if in_trade:
            # Check hard SL using current 1m price
            if side == 'long':
                pnl_now = (price - entry_fill) / TICK
            else:
                pnl_now = (entry_fill - price) / TICK
            
            if pnl_now <= -hard_sl:
                # SL exit: fill at actual 1s price 2s after this bar
                exit_fill = get_fill_price(
                    prices_1s, timestamps_1s, signal_time, side)
                exit_slip = abs(exit_fill - price) / TICK
                all_exit_slips.append(exit_slip)
                
                if side == 'long':
                    final_pnl = (exit_fill - entry_fill) / TICK
                else:
                    final_pnl = (entry_fill - exit_fill) / TICK
                
                trades.append({
                    'pnl': final_pnl, 'side': side,
                    'bars': i - entry_bar, 'reason': 'hard_sl',
                    'entry_slip': entry_slip,
                    'exit_slip': exit_slip,
                })
                in_trade = False
                continue
            
            # Check hold duration
            bars_held = i - entry_bar
            
            if bars_held >= predicted_hold:
                # Duration reached — re-evaluate
                should_enter_l1, new_side, _ = evaluate_entry(pred)
                
                if should_enter_l1 and new_side == side:
                    # Trend continuing — ask Layer 2 for extension
                    l2_feat = build_l2_feature_vector_live(pred, feats_29d[idx])
                    l2_input = torch.FloatTensor(l2_feat).unsqueeze(0).to(device)
                    with torch.no_grad():
                        p_take, new_hold = layer2_model(l2_input)
                    
                    if p_take.item() > 0.5:
                        predicted_hold = bars_held + max(3, int(new_hold.item()))
                        continue  # extend
                
                # Exit: fill at actual 1s price 2s later
                exit_fill = get_fill_price(
                    prices_1s, timestamps_1s, signal_time, side)
                exit_slip = abs(exit_fill - price) / TICK
                all_exit_slips.append(exit_slip)
                
                if side == 'long':
                    final_pnl = (exit_fill - entry_fill) / TICK
                else:
                    final_pnl = (entry_fill - exit_fill) / TICK
                
                trades.append({
                    'pnl': final_pnl, 'side': side,
                    'bars': bars_held, 'reason': 'duration_exit',
                    'entry_slip': entry_slip,
                    'exit_slip': exit_slip,
                })
                in_trade = False
            
            # else: HOLD. Don't look. Don't wobble-exit.
            continue
        
        # ── Not in trade — evaluate entry ──
        should_enter, side_signal, confidence = evaluate_entry(pred)
        
        if not should_enter:
            continue
        
        # ── Layer 2: take/skip + duration ──
        l2_feat = build_l2_feature_vector_live(pred, feats_29d[idx])
        l2_input = torch.FloatTensor(l2_feat).unsqueeze(0).to(device)
        with torch.no_grad():
            p_take, hold_bars = layer2_model(l2_input)
        
        if p_take.item() < 0.5:
            continue  # Layer 2 says skip
        
        # Entry: fill at actual 1s price 2s after signal
        side = side_signal
        entry_fill = get_fill_price(
            prices_1s, timestamps_1s, signal_time, side)
        entry_slip = abs(entry_fill - price) / TICK
        all_entry_slips.append(entry_slip)
        entry_bar = i
        predicted_hold = max(3, int(hold_bars.item()))
        in_trade = True
    
    # Print actual slippage from real data
    print_slippage_stats(all_entry_slips, all_exit_slips)
    
    return trades
```

---

## Part 9: Build Order

### Phase A: 29D Feature Pipeline (1 session)

1. Pre-compute SFE states for 1s, 5m, 15m, 1h on IS + OOS data
2. Cache as `.npy` files (one-time, ~20-40 min for 1s on CUDA)
3. Build `extract_features_29d()` with MTF alignment
4. Run `validate_mtf_alignment()` — zero violations or stop
5. Apply per-TF z-score normalization
6. Print feature correlation matrix — flag any r > 0.9
7. **Load 1s price + timestamp arrays for slippage fills** — these are
   needed by ALL simulations going forward. Cache alongside features.

### Phase B: Layer 1 on 29D (1 session)

1. Train StatePredictor(n_features=29) with walk-forward carry-forward
2. Compare vs 13D baseline: direction accuracy, feature correlation
3. Save all entry signals + slippage-adjusted trajectories per day

**Gate:**
- [ ] Direction accuracy ≥ 72% (at least matches 13D)
- [ ] MTF features contribute (feature importance shows non-zero for TF features)
- [ ] Signals saved with full trajectories + MFE

### Phase C: Layer 2 Duration Training (1 session)

1. Build duration labels from Phase B signals
2. Print diagnostics: hold distribution, "worth taking" percentage
3. Train DurationPredictor on carry-forward signals
4. Validate: take accuracy, hold MAE, simulated trades/day

**Gate:**
- [ ] take accuracy > 60%
- [ ] hold MAE < 5 bars
- [ ] trades/day drops to 3-8 range (from current 12+)

### Phase D: Full Two-Layer Simulation (1 session)

1. Run `simulate_two_layer()` on OOS with slippage
2. Compare: Layer 1 alone vs Layer 1 + Layer 2
3. Monthly breakdown, drawdown analysis

**Gate:**
- [ ] $/day positive (any amount — after actual 2s latency fills)
- [ ] trades/day is 3-8
- [ ] profitable days > 40%
- [ ] max single-day loss < $200 (400 ticks)
- [ ] slippage stats printed: mean, median, distribution from real 1s data
- [ ] gross $/day vs net $/day gap is understood (slippage cost quantified)

---

## Part 10: Baselines to Beat

```
DMI flipper + 7D CNN (proven):       $736/day
TradeCNN 13D Layer 1 (current):      $1,609/day (271 trades, needs verification)
TradeCNN 29D Layer 1 (target):       ≥ $1,609/day with better entries
TradeCNN 29D + Layer 2 (target):     ≥ $800/day NET after 4t/trade slippage
                                     with 3-8 trades/day (not 271)
```

The $1,609/day at 271 trades is $5.93/trade gross. After actual 2s latency
slippage (from 1s data — varies per fill, correlated with market conditions),
the real cost per trade is unknown until measured. Print `slippage_stats()`
first to establish the actual mean round-trip cost before projecting net PnL.
Layer 2 must concentrate PnL into fewer, larger moves where fill noise is
proportionally small relative to the captured move.

---

## Part 11: What NOT to Build

- ❌ Separate TF workers or TBN consensus
- ❌ Cascade Transformer architecture (aspirational, not earned)
- ❌ Pattern attention / prototypes (aspirational)
- ❌ CNN autoencoder on seed data (aspirational)
- ❌ More than 4 higher TFs (diminishing returns)
- ❌ Partial/forming bars from any TF (lookahead)
- ❌ Changes to label structure (still predict 1m 7D features)
- ❌ Live integration (earned by simulation results)

---

## Files

| File | Change | Phase |
|------|--------|-------|
| `training/train_trade_cnn.py` | Add `extract_features_29d()`, MTF alignment, slippage, signal saving, duration labels | A, B, C |
| `core/trade_cnn.py` | `StatePredictor(n_features=29)` | B |
| `core/trade_selector.py` | NEW: DurationPredictor | C |
| `training/validate_trade_cnn.py` | Two-layer simulation | D |
| `DATA/ATLAS/cache/` | `.npy` cache for MTF SFE states | A |

---

## Corrections From Original Proposal

| Issue | Original | Fixed |
|-------|----------|-------|
| Dimension inconsistency | "25D" in impl, "29D" in header | 29D everywhere (13 + 16) |
| Per-TF normalization | Listed as risk | Hard requirement with code |
| MTF alignment validation | Not mentioned | Hard requirement with assertion |
| Slippage | Not mentioned | Actual 1s price lookup at T+2s (real market data, not a distribution) |
| Trade fragmentation | Not addressed | Layer 2 duration predictor |
| 1s SFE cost concern | Flagged as expensive | Acceptable (<1 hour), pre-compute once |
