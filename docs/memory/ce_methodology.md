# Cause & Effect Matrix Methodology for Exit Optimization

## The Approach (proven 2026-03-07)

### Step 1: Identify the Problem from Reports
Read the forward pass report exit quality section. Look for:
- **Too early** bucket: % of correct-direction trades exiting <20% capture
- **Too late** bucket: trades that reached good peak but gave it back
- **Left on table**: dollar amount not captured
- Which `exit_reason` dominates each bucket (e.g., envelope_decay = 70% of too-early)

### Step 2: C&E T-Test on Trade Log
Load `checkpoints/oos_trade_log.csv` (OOS) or `reports/is/oracle_trade_log.csv` (IS).
Split correct-direction trades into problem bucket vs good bucket.
Run Welch t-test on every feature to find what differentiates them.

```python
# Key: correct direction = NOT NOISE
correct = df[~df['oracle_label_name'].str.contains('NOISE')]
# capture_rate is a FRACTION (0-1), not percentage
# Bucket: too_early = 0 < cap < 0.20, optimal = cap >= 0.80
```

### Step 3: Simulate the Exit Mechanic
Don't just look at features — simulate what the exit engine was doing.
Example: compute envelope tolerance at each hold_bar count to see when/why it fires.
Check PnL-by-hold-bars to find the optimal exit timing.

### Step 4: Find the Smoking Gun
In the envelope case:
- Bar-20 exits: 319 trades, $40/trade (GOOD)
- Bar 21+ exits: 136 trades, $8.4/trade (BAD — move already over)
- Trades that held longer didn't capture more — they captured LESS
- Root cause: halflife too long → envelope checks too late

### Step 5: Implement Fix + Self-Tuning
Two independent feedback loops:
- **too_early signal** (never reached peak) → grow halflife (be patient)
- **too_late signal** (reached peak, gave back) → tighten giveback threshold

Self-tuning in `ExitEngine.record_trade_outcome()`:
- Called after every trade close
- Counts too-early and too-late independently over 30-trade windows
- Adjusts `envelope_half_life_bars` (±10%) and `giveback_pct` (±5%)
- Bounded: halflife 8-60 bars, giveback 40-90%
- Drifts back toward defaults when neither signal fires

### Step 6: Add Analytics Bucket to Report
Add new bucket to forward pass report so future runs show the split:
- "Too late (reached peak, gave back)" — with trade_mfe_ticks field
- "Too early (<20%, never reached)" — truly never got there
- Report shows self-tuned final values for both parameters

## Key Data Points (OOS baseline, 2026-03-07)
- 536 trades, 88.4% WR, $10,804
- 455 envelope_decay exits (correct dir): 268 too-early, 187 good
- Too-early held LONGER (34 bars) than good (24 bars) — counterintuitive
- Oracle MFE similar (~66-72 ticks) — same size moves, different capture
- Median unrealized at exit: 18 ticks (too-early) vs 71 ticks (good)

## Implementation Files
- `core/exit_engine.py`: envelope decay, peak giveback, self-tuning
- `training/trainer.py`: trade_mfe_ticks capture, too-late analytics bucket
- `live/live_engine.py`: self-tuning wired to _close_position

## Features in Trade Log for C&E Analysis
Key columns: entry_depth, belief_conviction, wave_maturity, dmi_diff,
F_momentum, F_reversion, mom_rev_ratio, hurst, tunnel_prob, velocity,
sigma, band_speed, oracle_mfe, hold_bars, exit_conviction,
exit_wave_maturity, trade_mfe_ticks (NEW)
