---
name: RCA Process for System Improvement
description: Root Cause Analysis workflow — always follow this when improving the trading system. No shortcuts.
type: feedback
---

## RCA Process — MANDATORY for all system improvements

When the system underperforms, follow this process step by step. Do NOT skip steps.
Do NOT jump to "let me add a feature" or "let me train a model." Data first, always.

### Step 1: Run the system and get actual results
- Use the zero-lookahead ticker (1s data, aggregated to 1m for decisions)
- Run ONE DAY first, not the whole month
- Get real numbers with real execution

### Step 2: Analyze at the day and hour level
- Never look at monthly aggregates — each day must stand on its own
- What's the MODE of daily PnL? (most common outcome, not mean)
- How many winning vs losing days?

### Step 3: Pareto the exits
- Which exit category makes money? (mean_reached, max_hold_profit, lambda_flip)
- Which exit category destroys PnL? (half_cycle_loss, catastrophic_sl)
- What percentage of trades produce what percentage of profit?

### Step 4: Deep dive the losers
- Take the worst exit category (e.g., half_cycle_loss)
- List every trade: time, direction, features at entry
- What do they have in common?

### Step 5: Compare losers to winners feature by feature
- For each of the 13 features: what's the mean for losers vs winners?
- Which features are THE SAME (can't filter on these)?
- Which features are DIFFERENT (can filter on these)?
- Find the ratio — 4.9x separation is a signal, 1.1x is noise

### Step 6: Find the separator
- The separator is usually NOT in the bar features at entry
- It's in the CONTEXT: trend, DMI direction, volume participation
- Losers = extended in a trending market (keeps going)
- Winners = extended in a calm market (snaps back)

### Step 7: Apply the fix at the RIGHT point
- DO NOT filter at entry — this kills good trades too (proven: entry features identical)
- Fix at the EXIT: conditional early cut based on context at bar 2
- Check if BOTH trend AND dmi oppose → early cut
- If only one opposes → give it more time

### Step 8: Re-measure
- Run the same day again with the fix
- Did losers decrease? Did winners stay the same?
- If winners decreased, the fix was too aggressive — revert
- If losers decreased and winners held, keep the fix

### Step 9: Repeat
- After fixing one exit category, re-Pareto
- The next biggest drag becomes the target
- Continue until the mode per day is positive

## Critical Rules

**No shortcuts.** The tick-per-tick approach works because it mirrors live exactly.
If you test on pre-built bars, you get fake numbers with subtle lookahead.
Always use the 1s ticker for honest results.

**Why:** The batch SFE forward pass showed +$777/day. The honest 1s ticker showed +$48/day.
The difference was subtle lookahead in how features were computed.
The 1s ticker is the ONLY honest test. Everything else lies.

**How to apply:** Every time we modify the system:
1. Run nightmare_ticker.py on ONE day
2. Check per-exit-category PnL
3. If something regressed, find out why before running the full OOS

**No theoretical improvements.** Don't add features, models, or complexity without first
running the RCA on the current results. The data tells you what to fix — not intuition.
